# Build the per-layer tensor comparator

Building the comparator takes about 45 minutes. Each comparison after that
takes seconds.

The reference doesn't have to be `AutoModelForCausalLM`. Any PyTorch model
you can attach forward hooks to works the same way. That includes a
`trust_remote_code=True` modeling file, or (when debugging a quantized or
multi-GPU variant) your own already-verified MAX port's reference dumps from
the original bring-up.

Set dump directories via environment variables or script arguments. Use the
same paths in the HF dumper, MAX dumper, and comparator.

## Artifact 1: HF dumper

`hf_layer_dump.py` loads the HF reference, runs ONE prefill, and dumps
hidden states at every checkpoint to disk as FP32 numpy.

```python
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "<hf_model_id>"
OUT_DIR = os.environ.get("HF_DUMP_DIR", "./parity_dumps/hf_layers")
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(REPO)
# Load the reference in float32 when it fits: bfloat16 rounding in the
# reference shows up as layer drift that isn't in the port.
model = AutoModelForCausalLM.from_pretrained(
    REPO,
    dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

backbone = model.model  # adjust for non-decoder-only architectures
# Confirm this is the decoder stack, not a multimodal/encoder wrapper:
# hooks on the wrong module dump tensors with no MAX counterpart.
print("backbone:", type(backbone).__name__)


def save(name):
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        np.save(f"{OUT_DIR}/{name}.npy", h[0].detach().float().cpu().numpy())

    return hook


backbone.embed_tokens.register_forward_hook(save("post_embed"))
for i, layer in enumerate(backbone.layers):
    layer.register_forward_hook(save(f"layer_{i:02d}"))
backbone.norm.register_forward_hook(save("post_final_norm"))

text = tok.apply_chat_template(
    [{"role": "user", "content": "Hello!"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs, use_cache=False)

np.save(f"{OUT_DIR}/last_logits.npy", out.logits[0, -1].float().cpu().numpy())
np.save(f"{OUT_DIR}/prompt_tokens.npy", inputs.input_ids[0].cpu().numpy())
print("done", OUT_DIR)
```

Run time: ~2 to 3 min model load + ~1s prefill.

## Artifact 2: MAX dumper

This dumper has `forward()` taps to expose hidden states and a standalone
runner that bypasses `max serve` to capture every model output.

### `forward()` taps (in your port's model file)

Behind an env flag, collect an FP32 copy of the hidden state after the
embedding, after every decoder layer, and after the final norm. Return those
tensors as extra outputs of the root module's `forward()`:

```python
import os

from max.dtype import DType
from max.tensor import Tensor

_DUMP = os.environ.get("PORT_DUMP") == "1"


# inside the root module's forward():
dump_tensors: list[Tensor] = []


def _tap(t: Tensor) -> None:
    if _DUMP:
        dump_tensors.append(t.cast(DType.float32))


h = self.embed_tokens(tokens)
_tap(h)  # post-embed
for idx, layer in enumerate(self.layers):
    h = layer(...)
    _tap(h)  # layer_00, layer_01, ...
h = self.norm(h)
_tap(h)  # post-final-norm

return (last_logits, *dump_tensors) if _DUMP else (last_logits,)
```

The flag is read at trace time, so compile with `PORT_DUMP=1` set. Keep the
list local to `forward()` so a second trace starts empty.

**Cast to FP32 in `forward()`.** `np.from_dlpack` fails on BF16.
The in-model cast lets the dumper use plain numpy.

On a multi-GPU mesh, the hidden state is a distributed `Tensor`. Tap a
replicated tensor, or gather it first (`F.allgather(t, tensor_axis=0)` for a
sequence-sharded one), and the runner reads shard 0.

If the port shares layer subgraphs (`as_subgraph`), keep subgraphs on in
dump mode. The taps sit between layer calls and don't change the subgraph
signature.

### Standalone dumper

This runner uses internal pipeline APIs that shift between MAX releases. If
an import or attribute below fails, grep your installed `max.pipelines`
package for the symbol and adjust. The steps stay the same across releases:
build a context, allocate KV cache for it, prepare token inputs, and call the
compiled model.

```python
import os
import numpy as np

from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.context import SamplingParams, TextContext, TokenBuffer
# Register your architecture the same way you would for max serve.

_tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(your_pipeline_config)
pipeline_model = pipeline._pipeline_model
inner_model = pipeline_model.model  # the compiled ModuleV3 callable
kv_manager = pipeline.kv_manager

ctx = TextContext(
    max_length=n_tokens + 1,
    tokens=TokenBuffer(np.asarray(prompt_ids, dtype=np.int64)),
    sampling_params=SamplingParams(max_new_tokens=1),
)
replica_batches = [[ctx]]

kv_manager.claim(ctx)
kv_manager.alloc(ctx)
kv_inputs = kv_manager.runtime_inputs(replica_batches)
model_inputs = pipeline_model.prepare_initial_token_inputs(
    replica_batches=replica_batches,
    kv_cache_inputs=kv_inputs,
    return_n_logits=1,
)
# Call the compiled model directly. pipeline_model.execute strips
# everything but logits. Don't patch the compiled callable.
outs = inner_model(*model_inputs.buffers)
kv_manager.release(ctx)

NAMES = (
    ["last_logits", "post_embed"]
    + [f"layer_{i:02d}" for i in range(n_layers)]
    + ["post_final_norm"]
)
out_dir = os.environ.get("MAX_DUMP_DIR", "./parity_dumps/max_layers")
os.makedirs(out_dir, exist_ok=True)
for name, o in zip(NAMES, outs):
    # Outputs are Tensors. A distributed output reads shard 0.
    shard = o.local_shards[0] if o.is_distributed else o
    arr = shard.to_numpy().copy()
    if name == "last_logits":
        arr = arr[0]  # [1, vocab] to [vocab], the shape the HF dumper saves
    np.save(f"{out_dir}/{name}.npy", arr)
np.save(f"{out_dir}/prompt_tokens.npy", np.asarray(prompt_ids))
print("done", out_dir)
```

Run: `PORT_DUMP=1 pixi run python dump_max_layers.py`

Cold compile time matches serve cold compile (~5 to 25 min depending on
model). The dump itself is under one second.

### Quick inspection with `PrintHook`

The tensor dumpers above save each layer's activations to disk for offline
diffing. For a faster look that needs no model edits, `max.nn.legacy.hooks.PrintHook`
prints every layer's inputs and outputs as the model runs:

```python
from max.nn.legacy.hooks import PrintHook

hook = PrintHook()
hook.name_layers(model)  # name each Module by its attribute path
# compile and run the model
hook.remove()
```

`PrintHook` attaches to MAX `Module` trees. When the reference is another MAX
implementation of the same model, add the same hook to both and run them.
Compare the printed layer values to find the first one that disagrees. When
the reference is a `transformers` model, hook it with the PyTorch forward
hooks from Artifact 1. For multi-device (distributed) tensors,
`F.print(value, name)` from `max.functional` prints each shard
with its device.

`PrintHook` prints to the console, so use it for quick triage. For cosine
comparison across a full run, use the saved-dump comparator above. For MAX's
other built-in debugging options, see
[the MAX debugging tools](https://max.modular.com/develop/debugging/).

## Artifact 3: Comparator

```python
import numpy as np, os, glob

HF = os.environ.get("HF_DUMP_DIR", "./parity_dumps/hf_layers")
MX = os.environ.get("MAX_DUMP_DIR", "./parity_dumps/max_layers")


def cos(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


names = sorted(
    {os.path.basename(p)[:-4] for p in glob.glob(f"{HF}/*.npy")}
    & {os.path.basename(p)[:-4] for p in glob.glob(f"{MX}/*.npy")}
)

print(
    f"{'checkpoint':25s} {'cos_sim':>9s} {'mean_diff':>10s} {'max_diff':>10s}  spike_position"
)
for n in names:
    h = np.load(f"{HF}/{n}.npy")
    m = np.load(f"{MX}/{n}.npy")
    if h.shape != m.shape:
        print(f"{n:25s}  shape mismatch  hf={h.shape} max={m.shape}")
        continue
    diff = np.abs(h - m)
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    pos = ",".join(str(i) for i in idx)  # (token, dim), or (vocab,) for logits
    spike = f"({pos}) HF={h[idx]:+.2f} MAX={m[idx]:+.2f}"
    print(
        f"{n:25s} {cos(h, m):9.4f} {diff.mean():10.4f} {diff.max():10.4f}  {spike}"
    )
```

**Column meanings:**

- `cos_sim`: full-tensor cosine similarity (flattened)
- `mean_diff`: `mean(|HF - MAX|)`, average element-wise disagreement
- `max_diff`: `max(|HF - MAX|)`, largest single-element disagreement
- `spike_position`: the (token, dim) where the max diff lives, plus HF and MAX
  values there

**Common reading mistake**: `max_diff` is element-wise disagreement, not
the tensor's max-abs. A large `max_diff` where HF has a spike and MAX is
flat means HF formed an anchor MAX didn't.

## Beyond the default comparator

Add per-token and per-dim slices when the default output shows divergence:

```python
def per_token_cos(h, m):
    return [cos(h[t], m[t]) for t in range(h.shape[0])]


def per_dim_cos(h, m):
    return [cos(h[:, d], m[:, d]) for d in range(h.shape[1])]
```

## Decode-step dumps (greedy failure at index K)

When greedy verification fails at token index K, dump with prefix =
tokens 0..K-1 and compare at `--token-index K`.

**HF:** extend both `input_ids` and `attention_mask` when appending
teacher-forced tokens:

```python
ids = tok(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
for tid in prefix_token_ids:
    ids = torch.cat([ids, torch.tensor([[tid]], device=ids.device)], dim=1)
mask = torch.ones_like(ids)
with torch.no_grad():
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
```

**MAX:** build `TextContext` with the same token list, and set
`return_n_logits=1` for last-position logits only.

## HF `hidden_states` indexing

**Don't** assume `output_hidden_states[i+1]` is layer `i` output. Some
models store pre-layer inputs plus a final norm output, not per-layer
outputs. Use per-layer forward hooks when tuple semantics are unclear.
See [comparator-output-patterns.md](comparator-output-patterns.md).

Use `add_special_tokens=False` when your MAX prompt has no BOS.

## Check RoPE before debugging attention

For position-dependent divergence (t=0 matches, t≥1 bad from layer 1):

```python
pos = torch.arange(seq_len, device=model.device).unsqueeze(0)
emb = model.model.rotary_emb(hidden_states, pos)
print("inv_freq max:", model.model.rotary_emb.inv_freq.abs().max().item())
cos, sin = emb
print("cos sample:", cos[0, -1, :4].tolist())
print("sin sample:", sin[0, -1, :4].tolist())
```

If `inv_freq≈0` → cos≈1, sin≈0 → **NoPE at runtime**. Your MAX module must
not apply plain RoPE when HF uses identity rotation at runtime.

## Stateful layers (conv, SSM): incremental decode dumps

Teacher-forced **prefill-only** dumps aren't enough when the model carries
state across decode steps (conv layers, SSM blocks, etc.). A single
`execute()` with the full greedy prefix exercises prefill math but may not
match the **incremental** decode path your verification uses.

**Required pattern:**

1. Prefill the instruct prompt once.
2. Run **K single-token decode steps**, updating state + KV each step
   (mirror `prepare_next_token_inputs` and any state outputs).
3. Dump hidden states at step K (teacher-forced prefix 0..K−1 **or**
   autoregressive MAX greedy throughout).

When dump mode returns logits + tap tensors + state tensors, document the
output order in your `forward()`. Slice state outputs by count, not by position
after logits alone.

## Comparator CLI extensions

```python
parser.add_argument(
    "--token-index",
    type=int,
    default=-1,
    help="Compare only row t (last row if -1)",
)
parser.add_argument(
    "--skip",
    nargs="*",
    default=["last_logits"],
    help="Skip logits when shapes differ by vocab axis",
)
```

When comparing logits tensors, skip token slicing (1D vocab vectors).

## Validate dumpers before debugging the port

1. Run both dumpers on a model you already know works in MAX. Cosine
   similarity should be ~0.999 throughout. If not, fix the dumpers first.
2. Check `prompt_tokens.npy` from both sides. They must be identical.
3. `post_embed.npy` should be cos=1.0 (within FP rounding). If not,
   fix embedding load before comparing deeper layers.
