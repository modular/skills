# Implement the graph

Scaffolding copies a donor architecture (`llama3`, `qwen3`, …). **That copy is
not your model.** The donor graph computes the donor's math until you edit
every sublayer that the delta list flagged as different from Hugging Face.

Do not run `pixi run max serve`, coherence checks, or logit verification until
this phase's completion criteria pass. Serving an unmodified donor against a
foreign checkpoint loads weights into the wrong shapes and
**will fail verification** — that failure is not a tolerance problem; the port
was never finished.

---

## Anti-pattern (do not do this)

| What agents skip                                     | Why it fails                                                                        |
|------------------------------------------------------|-------------------------------------------------------------------------------------|
| Serve right after `scaffold.py`                      | Donor attention / block / MoE still runs                                            |
| Only edit `arch.py` + `model_config.py`              | Graph in `<slug>.py` still matches the donor                                        |
| Assume "Llama-compatible" means no `<slug>.py` edits | HF inheritance hides non-Llama blocks (parallel norms, sigmoid MoE, NoPE layers, …) |
| Run `compare_layers.py` before weights load cleanly  | Chasing logits when tensors are unbound or mis-mapped                               |

Phase 1 exists to produce a **delta list**. Implementing the graph is where
you **execute** that list in MAX code, one sublayer at a time, against HF
`forward()`.

---

## Work order

Implement in this order — each layer depends on the previous wiring being
correct:

1. **`model_config.py`** — wire every `config.json` key (Phase 1 config table).
2. **`list_checkpoint_keys.py`** — Hub safetensors metadata (keys, shapes,
   dtypes).
3. **`weight_adapters.py`** — HF safetensor names → MAX module names.
4. **Embedding + final norm + LM head** in `<slug>.py` / `model.py`.
5. **One decoder block** — get block 0 right before cloning the pattern.
6. **Full stack** — repeat for all layers; conditional layers (sliding vs full,
   MoE vs dense) need explicit per-layer logic matching HF.

Keep HF `modeling_<type>.py` open side-by-side. For each MAX class you edit,
trace the HF `forward()` line that corresponds to each MAX op.

---

## Component checklist

Copy this table from the Phase 1 delta list and mark each row **done** only when
MAX matches HF for that component (not when it "compiles").

| Component          | HF reference                           | MAX file / class                  | Done when                                                                                          |
|--------------------|----------------------------------------|-----------------------------------|----------------------------------------------------------------------------------------------------|
| Config / KV params | `*Config`                              | `model_config.py`                 | Every Hub key read in `initialize()`; `get_kv_params()` head counts match HF                       |
| Weight map         | checkpoint keys                        | `weight_adapters.py`              | Loads without silent drops; required tensors bound                                                 |
| Embedding          | `embed_tokens`                         | `<slug>.py`                       | Shape + dtype match; tie with LM head if config says so                                            |
| Attention          | `*Attention.forward`                   | attention class in `<slug>.py`    | Q/K/V layout, RoPE, mask, GQA repeat, softcap match HF                                             |
| MLP / MoE          | `*MLP.forward` / `*SparseMoeBlock`     | MLP or MoE class                  | Gate/up/down or expert routing matches HF (activation name matters)                                |
| Decoder block      | `*DecoderLayer.forward`                | block class                       | **Norm order and residual wiring** match HF (pre-norm vs parallel vs dual-norm)                    |
| Final norm         | `model.norm`                           | transformer wrapper               | Same norm type (RMS vs LayerNorm) and epsilon                                                      |
| LM head            | `ForCausalLM.forward` → `lm_head(...)` | head in `model.py` or `<slug>.py` | Tied embed; pre-head divisor (`h / (hidden_size / dim_model_base)`); MuP `logits_scaling`; softcap |

Norm order and block wiring are the most common "looks Llama-ish but isn't"
bugs. Read the HF block `forward()` before editing the donor block.

---

## `weight_adapters.py`

Goal: after adapters run, **every weight tensor HF expects exists on the MAX
side with the right shape**.

- Match fused vs split projections (`qkv_proj` vs separate `q_proj` / `k_proj` /
  `v_proj`).
- Match MoE expert key layout (`experts.N.gate_proj` vs grouped tensors).
- Match QKV stacking / head layout renames from the donor — delete donor-only
  renames that do not apply to your checkpoint.

Verify load before serve:

```python
# In a REPL or one-off script after editing adapters:
# load state dict through your adapter and assert no unexpected missing keys
```

See [rename-weights.md](rename-weights.md). After `load_state_dict`, run the
audit in [state-dict-audit.md](state-dict-audit.md) so silent drops surface
before serve.

---

## `<slug>.py` — the graph

This file is the port. Subclassing the donor is fine **only** for methods that
are identical to HF. When the delta list flagged a difference:

- **One method differs** — subclass donor layer, override that method.
- **Block wiring differs** — rewrite the block class; do not inherit donor
  `forward()` if norm/residual order differs.
- **New attention pattern** (MLA, sliding window per layer index, NoPE on some
  layers) — new attention module using `max.nn` primitives.

Do not copy-paste the donor `<slug>.py` and change the class name. Walk HF
`forward()` and implement what it actually does.

### Recurrent / shared-weight stacks: mix the injection in ONCE

If your model iterates the same Transformer stack multiple times (HRM, RWKV,
some encoder-decoders, "looped transformers") and mixes a separate "injection
state" (`z_H`, `z_L`, condition embedding, prefix state) into each stack
invocation, the injection happens **once before the first block** of the
stack, not at every block. HF's `Stack.forward` is the canonical pattern:

```python
# HF (correct): inject once, then run blocks sequentially.
def forward(self, x, ...):
    x = x + injection  # mix in once
    for layer in self.layers:
        x = layer(x, ...)
    return self.final_norm(x)
```

The seductive port mistake is to unroll the stack into the top-level loop
and write:

```python
# WRONG: injection added N×K times per cycle, residual stream explodes.
for cycle in range(N):
    for i, layer in enumerate(self.layers):
        z = layer(z + injection, ...)  # ← added every block
```

instead of:

```python
# RIGHT: inject once per stack invocation, then iterate.
for cycle in range(N):
    z = z + injection
    for i, layer in enumerate(self.layers):
        z = layer(z, ...)
```

See
["Stack-vs-block residual in recurrent / shared-weight architectures"](pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures)
in the pitfalls reference for the full diagnostic recipe (the symptom is a
residual stream whose `mean_sq` grows super-linearly with depth in MAX while HF
grows linearly).

Detailed reading guide: [read-modeling-code.md](read-modeling-code.md).

---

## Completion criteria (required before serving)

All must be true before `pixi run max serve` or any verification script:

- [ ] Delta list: every row implemented or explicitly marked N/A with HF
      citation
- [ ] `model_config.py` wires every key from the Phase 1 config table /
      `inspect_hf.py`
- [ ] Checkpoint metadata listed (`list_checkpoint_keys.py`); shapes agree with
      config
- [ ] `arch.py`: `name=` equals `config.json` → `architectures[0]`;
      `default_encoding` matches Hub dtype
- [ ] `weight_adapters.py`: every required MAX FQN mapped; no orphan HF keys you
      need
- [ ] `<slug>.py`: attention, MLP/MoE, and block classes match HF `forward()`
      for block 0 at minimum, then full depth
- [ ] **Top-level forward**: walked HF's outermost `forward()` (not just
      `DecoderLayer.forward`) and confirmed how blocks are wired together. For
      recurrent / shared-weight stacks: injection state mixed in
      **once per stack invocation**, not per block (see
      ["Recurrent / shared-weight stacks"](#recurrent--shared-weight-stacks-mix-the-injection-in-once)
      above)
- [ ] You can explain, per delta, which line of HF code the MAX change mirrors

**Compiles and serves is not done.** This phase means the graph implements HF
math, not that `pixi run max serve` starts. Garbage or `&&&&` loops after a
rewrite usually mean a delta is still wrong (NoPE `freqs_cis` layout, block
wiring, MoE combine) — stay in the implement / divergence-hunt loop until logits
match, not "tolerance."

If any box is unchecked, you are still implementing — not verifying.

---

## API surface

Copy imports and registration from the scaffold donor in
`modular/max/python/max/pipelines/architectures/<donor>/`. For stale-import
traps and encoding/device rules, see
[pitfalls-config.md § Import and config API traps](pitfalls-config.md#import-and-config-api-traps).
