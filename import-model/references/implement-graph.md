# Implement the module

Scaffolding subclasses a donor architecture (`llama3_modulev3`, `olmo3`, …).
That skeleton isn't your model. The donor module computes the donor's
math until you edit every sublayer that the delta list flagged as different
from Hugging Face.

Don't run `pixi run max serve`, coherence checks, or logit verification until
this phase's completion criteria pass. An unmodified donor served against a
foreign checkpoint either rejects the weights at compile time or runs the
donor's math on them. Either way, verification fails until the module
implements the deltas.

---

## Anti-pattern (don't do this)

| What agents skip                                     | Why it fails                                                                                                                                                                                                                       |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Serve right after `scaffold.py`                      | Donor attention / block / MoE still runs                                                                                                                                                                                           |
| Only edit `arch.py` + `model_config.py`              | The module in `<slug>.py` still matches the donor                                                                                                                                                                                  |
| Assume "Llama-compatible" means no `<slug>.py` edits | HF inheritance hides non-Llama blocks (parallel norms, sigmoid MoE, NoPE layers, …). A delta the donor already exposes as a config field (`interleaved_rope_weights`) is set in `from_donor()`; every other delta is a module edit |
| Run `compare_layers.py` before weights load cleanly  | Chasing logits when tensors are unbound or mis-mapped                                                                                                                                                                              |

Phase 1 produces a **delta list**. Implementing the module executes that
list in MAX code, one sublayer at a time, against HF `forward()`.

---

## The ModuleV3 API surface

A port is a tree of `Module` subclasses. Each one declares its call
signature, builds its parameters and child modules in `__init__()`, and
computes in `forward()`:

```python
from max import functional as F
from max.nn import Linear, Module
from max.tensor import Tensor


class MyMLP(Module[[Tensor], Tensor]):
    def __init__(self, *, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- **Imports.** Layers come from `max.nn` (`Linear`, `Embedding`,
  `RMSNorm` in `max.nn.norm`, `ModuleList` in
  `max.nn.sequential`) and the `max.nn.common_layers`
  submodules (`MLP` in `mlp`, rotary embeddings in `rotary_embedding`,
  `ColumnParallelLinear` and `RowParallelLinear` in `linear`, and the
  attention kernels in `functional_kernels`). Ops come from
  `max.functional`. `PagedCacheValues` comes from
  `max.nn.common_layers.kv_cache`. KV cache parameter types
  (`KVCacheParams`, `MultiKVCacheParams`, `KVCacheParamInterface`) and
  `MHAMaskVariant` come from `max.nn.legacy.kv_cache` and `max.nn.legacy.attention`. They
  hold configuration, not tensors. `TensorType` and `DeviceRef` come from
  `max.graph` and describe inputs.
- **Constructors.** Layer arguments are keyword-friendly and carry no
  `dtype=` or `device=`: the pipeline builds the module under a default dtype,
  and `model.py` moves it with `.to(device)`, or builds it inside
  `default_device(mesh)` for a multi-GPU port. `Linear` creates a
  bias by default, so pass `bias=False` when the checkpoint ships none.
- **Parameters.** Parameters are `Tensor` attributes. Their names are the
  attribute paths from the root module
  (`language_model.layers.0.mlp.gate_proj.weight`), and
  `compile(weights=...)` loads the checkpoint by those names.
- **Ops.** Most shape ops are `Tensor` methods (`reshape()`, `transpose()`,
  `permute()`, `split()`, `cast()`). The rest are `F.*` (`F.gather()`,
  `F.concat()`, `F.flatten()`). `@` is matmul.
- **Lists of layers.** Use `ModuleList(layers)`, which takes one iterable, so
  parameters register under `layers.<i>.`.
- **KV cache.** A registered root module's `forward()` takes one
  `DecoderInputs` and reads the KV cache inputs from `inputs.kv`. A pipeline
  model donor's root module receives them flattened as trailing
  `*variadic_args`. Either way, unflatten them with
  `self.kv_params.unflatten_kv_inputs(...)` (or `unflatten_basic_kv_tree()`
  for sliding plus global caches) and wrap each result with
  `PagedCacheValues.from_upstream(...)`. `gemma3_modulev3/gemma3.py` shows
  the registered form. On a multi-GPU mesh, pass
  `DeviceMapping.replicated(mesh)` as the mapping, not `tokens.mapping`:
  the tokens live on one device, and a cache wrapped with their mapping
  breaks on more than one GPU.
- **Compilation.** `ModuleV3PipelineModelWithKVCache.load_model()` builds the
  module under `F.lazy()`, so parameters are recorded symbolically, then calls
  `compile(*input_types, weights=state_dict)`. A registered port writes none
  of this: the framework's pipeline model finalizes the config and builds
  the module. A pipeline model port's `model.py` only implements
  `_create_model_config()` and `_instantiate_module()`.

### Ops and kernels with no `Tensor` version

Every graph-level op and kernel is callable from a ModuleV3 `forward()`.
Wrap a function over `TensorValue` with `F.functional()` to get a function
over `Tensor`:

```python
from max import functional as F
from max.graph import TensorValue, ops


def _last_tokens(h: TensorValue, offsets: TensorValue) -> TensorValue:
    return ops.gather(h, offsets[1:] - 1, axis=0)


last_tokens = F.functional(_last_tokens)
```

On a multi-GPU mesh, the wrapped function runs on each device's shard. To
call a per-device helper directly, take the shards with
`[TensorValue(s) for s in t.local_shards]` and reassemble the results with
`Tensor.from_shard_values(values, mapping)`. For an example, see
`split_replicated_batch()` in `deepseekV3_modulev3/deepseekV3.py`. If MAX
has no `Module` for a layer, write one in your port's `layers/` directory,
built from these pieces.

---

## Work order

Implement in this order, because each layer depends on the previous wiring
being correct:

1. **`model_config.py`**: wire every `config.json` key (Phase 1 config table).
2. **`list_checkpoint_keys.py`**: Hub safetensors metadata (keys, shapes,
   dtypes).
3. **`weight_adapters.py`**: HF safetensor names → the root module's
   parameter names.
4. **Embedding + final norm + LM head** in `<slug>.py`.
5. **One decoder block**: get block 0 right before cloning the pattern.
6. **Full stack**: repeat for all layers. Conditional layers (sliding vs full,
   MoE vs dense) need explicit per-layer logic matching HF.

Keep HF `modeling_<type>.py` open side-by-side. For each MAX module you edit,
trace the HF `forward()` line that corresponds to each MAX op.

---

## Component checklist

Copy this table from the Phase 1 delta list and mark each row **done** only when
MAX matches HF for that component (not when it "compiles").

| Component          | HF reference                           | MAX file / class                | Done when                                                                                          |
|--------------------|----------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------|
| Config / KV params | `*Config`                              | `model_config.py`               | Every Hub key read by the donor config or `from_donor()`, and `construct_kv_params()` matches HF   |
| Weight map         | checkpoint keys                        | `weight_adapters.py`            | Every parameter bound, and every checkpoint tensor consumed or dropped on purpose                  |
| Embedding          | `embed_tokens`                         | `<slug>.py`                     | Shape + dtype match. Tie with LM head if config says so                                            |
| Attention          | `*Attention.forward`                   | attention module in `<slug>.py` | Q/K/V layout, RoPE, mask, GQA repeat, softcap match HF                                             |
| MLP / MoE          | `*MLP.forward` / `*SparseMoeBlock`     | MLP or MoE module               | Gate/up/down or expert routing matches HF (activation name matters)                                |
| Decoder block      | `*DecoderLayer.forward`                | block module                    | **Norm order and residual wiring** match HF (pre-norm vs parallel vs dual-norm)                    |
| Final norm         | `model.norm`                           | root text module                | Same norm type (RMS vs LayerNorm) and epsilon                                                      |
| LM head            | `ForCausalLM.forward` → `lm_head(...)` | root text module in `<slug>.py` | Tied embed, pre-head divisor (`h / (hidden_size / dim_model_base)`), MuP `logits_scaling`, softcap |

Norm order and block wiring are the most common "looks Llama-ish but isn't"
bugs. Read the HF block `forward()` before editing the donor block.

---

## `weight_adapters.py`

Goal: after adapters run, **every parameter of the root module has a tensor
of the right shape and dtype, and every checkpoint tensor lands on a
parameter**.

- Match fused vs split projections (`qkv_proj` vs separate `q_proj` / `k_proj` /
  `v_proj`).
- Match MoE expert key layout (`experts.N.gate_proj` vs grouped tensors).
- Match the root module's wrapper prefix (`model.layers.` →
  `language_model.layers.` in `olmo3`).
- Delete donor-only renames that don't apply to your checkpoint.

`compile(weights=...)` raises `KeyError` for a parameter with no tensor and
`ValueError` for a shape or dtype mismatch. A checkpoint tensor whose name
matches no parameter is left out without an error, so run the audit in
[state-dict-audit.md](state-dict-audit.md) before serving. See
[rename-weights.md](rename-weights.md) for renames.

---

## `<slug>.py`: the module

This file is the port. Subclassing the donor is fine only for methods that
are identical to HF. When the delta list flagged a difference:

- **One method differs**: subclass the donor module and override that method.
- **Block wiring differs**: rewrite the block module. Don't inherit the donor
  `forward()` if norm/residual order differs.
- **New attention pattern** (MLA, sliding window per layer index): a new
  attention `Module` built from `max.nn.common_layers` and the
  kernels in `common_layers.functional_kernels`.
- **NoPE on some layers**: give those layers a rotary module with an identity
  `freqs_cis` table
  ([divergences.md](divergences.md#17-nope--skip-rope-layers-via-identity-freqs_cis)).
- **A submodule the donor's `__init__` builds** (the inner text model):
  call `super().__init__(...)` in the port's root module, then assign the
  replacement to the same attribute. The module is built under `F.lazy()`,
  so the donor's discarded submodule allocates nothing.

Don't copy-paste the donor module and change the class name. Walk HF
`forward()` and implement what it does.

### Recurrent / shared-weight stacks: mix the injection in once

Some models iterate the same Transformer stack multiple times (HRM, some
encoder-decoders, "looped transformers"). They mix a separate
"injection state" (`z_H`, `z_L`, condition embedding, prefix state) into
each stack invocation. The injection happens **once before the first
block** of the stack, not at every block. HF's `Stack.forward` is the
canonical pattern:

```python
# HF (correct): inject once, then run blocks sequentially.
def forward(self, x, ...):
    x = x + injection  # mix in once
    for layer in self.layers:
        x = layer(x, ...)
    return self.final_norm(x)
```

A common port mistake is to unroll the stack into the top-level loop
and write:

```python
# WRONG: injection added at every block. mean_sq grows super-linearly.
for cycle in range(N):
    for i, layer in enumerate(self.layers):
        z = layer(z + injection, ...)  # ← added every block
```

The correct loop:

```python
# RIGHT: inject once per stack invocation, then iterate.
for cycle in range(N):
    z = z + injection
    for i, layer in enumerate(self.layers):
        z = layer(z, ...)
```

See ["Stack-vs-block residual"][stack-residual]
in the pitfalls reference for the full diagnostic recipe. The symptom is a
residual stream whose `mean_sq` grows super-linearly with depth in MAX,
while HF grows linearly.

Detailed reading guide: [read-modeling-code.md](read-modeling-code.md).

---

## Completion criteria (required before serving)

All must be true before `pixi run max serve` or any verification script:

- [ ] Delta list: every row implemented or explicitly marked N/A with HF
      citation
- [ ] `model_config.py` wires every key from the Phase 1 config table /
      `inspect_hf.py`
- [ ] Checkpoint metadata listed (`list_checkpoint_keys.py`), and shapes
      agree with config
- [ ] `arch.py`: `name=` equals `config.json` → `architectures[0]`,
      and `default_encoding` matches Hub dtype
- [ ] `weight_adapters.py`: every parameter bound, and the state-dict audit
      reports no unconsumed checkpoint tensors you need
- [ ] `<slug>.py`: attention, MLP/MoE, and block modules match HF `forward()`
      for block 0 at minimum, then full depth
- [ ] **Top-level forward**: walked HF's outermost `forward()` (not only
      `DecoderLayer.forward`) and confirmed how blocks are wired together. For
      recurrent / shared-weight stacks: injection state mixed in
      **once per stack invocation**, not per block (see
      ["Recurrent / shared-weight stacks"](#recurrent--shared-weight-stacks-mix-the-injection-in-once)
      above)
- [ ] You can explain, per delta, which line of HF code the MAX change mirrors

This phase ends when the module implements HF math. A module that compiles
and starts under `pixi run max serve` can still compute the wrong math.
Garbage or `&&&&` loops after a rewrite usually mean a delta is still wrong
(NoPE `freqs_cis` layout, block wiring, MoE combine). A missing delta can
also leave the output fluent, so coherent text doesn't end this phase. Stay
in the implement / divergence-hunt loop until logits match.

If any box is unchecked, you're still implementing, not verifying.

---

## Registration and imports

Copy imports and registration from the scaffold donor in your installed MAX
package, `max/pipelines/architectures/<donor>/`. For stale-import
traps and encoding/device rules, see
[pitfalls-config.md § Import and config API traps][import-traps].

[stack-residual]: pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures
[import-traps]: pitfalls-config.md#import-and-config-api-traps
