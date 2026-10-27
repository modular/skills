# Module build pitfalls

This page covers Phase 2 traps that surface while you write `<slug>.py` and
the ModuleV3 `forward()` path: `F.*` calls, constants, RoPE wiring, residual
placement, and subgraph sharing. It describes these pitfalls:

- Scaffold is not a port
- `F.constant` defaults to the accelerator and bfloat16
- Partial-rotary padding is interleaved
- `F.sum` keeps the reduced dim
- Stack-vs-block residual in recurrent / shared-weight architectures
- A shared subgraph assumes uniform layer signatures

## Scaffold is not a port

`scaffold.py` subclasses a donor (`llama3_modulev3`, `olmo3`, …). Until you
implement the module, `<slug>.py` runs the **donor's** attention, block
wiring, and MLP, not your model's. Serving it and running logit verification
will fail. That's expected, not a tolerance or script bug. Complete
[implement-graph.md](implement-graph.md) before any verification activity.

## `F.constant` defaults to the accelerator and bfloat16

`F.constant(value, dtype=None, device=None)` places the constant on an
accelerator when one is present, and a Python scalar with no `dtype` becomes
`bfloat16` there. An epsilon or scale created that way loses precision, and a
constant that feeds a CPU-side input (a KV-cache layer index) lands on the
wrong device. Pass both arguments:

```python
from max.driver import CPU
from max.dtype import DType
from max import functional as F
from max.tensor import Tensor

x = Tensor.ones([2, 4], dtype=DType.float32)

# Wrong: bfloat16 on the GPU
eps = F.constant(1e-6)

# Right
eps = F.constant(1e-6, DType.float32, device=x.device)
layer_idx = F.constant(0, DType.uint32, device=CPU())
```

On a mesh, pass the mesh (`device=x.mesh`) so the constant is replicated on
every device.

## Partial-rotary padding is interleaved

The base `RotaryEmbedding` produces interleaved frequency pairs:
`[cos0, sin0, cos1, sin1, ...]`. When `partial_rotary_factor < 1.0`,
pad the unrotated section with interleaved identity pairs
`[1, 0, 1, 0, ...]`, not split-half `[1, 1, ..., 0, 0, ...]`. Wrong
padding produces position-dependent errors that grow with sequence
length.

## `F.sum` keeps the reduced dim

In MAX, `F.sum(x, axis=-1)` on a `[N, H, D]` tensor returns `[N, H, 1]`,
where PyTorch returns `[N, H]`. `Tensor.sum()` and `F.mean()` behave the same
way. Squeeze after if you need rank-2.

## Stack-vs-block residual in recurrent / shared-weight architectures

Some models run the same transformer stack multiple times per forward
(HRM-style loops, shared-weight stacks, some encoder-decoders). These models
mix an injection state into the stack **once before the first block**, not
at every block inside the loop.

HF pattern (correct):

```python
def forward(self, x, injection, ...):
    x = x + injection
    for layer in self.layers:
        x = layer(x, ...)
    return self.final_norm(x)
```

Common port mistake:

```python
for cycle in range(num_cycles):
    for layer in self.layers:
        z = layer(z + injection, ...)  # injection added every block
```

Symptom: MAX hidden-state norms grow super-linearly with depth while HF grows
linearly. Greedy text diverges from token 1–2 even when weights load cleanly.
Fix: move `z = z + injection` outside the inner layer loop (once per stack
invocation). See also
[implement-graph.md](implement-graph.md#recurrent--shared-weight-stacks-mix-the-injection-in-once).

## A shared subgraph assumes uniform layer signatures

`as_subgraph(layer, name=...)` from `max.nn` traces one subgraph
per name and input signature, and calls it from every layer that uses that
name, so the compiler processes a repeated block once. Each call reads its own
weights by name. The body comes from the **first** call, including that
layer's weight shapes and any Python values baked into the trace. A later call
with different input types gets a separate subgraph. A later layer whose
weights differ from the first layer's breaks the shared body:

- Different weight names make `compile()` raise `ValueError` with
  `unable to look up weight by name: layers.1.<...>`.
- Different weight shapes, or a different per-layer Python value, compile and
  run without an error, and the layer computes wrong values.

Per-layer-variable architectures hit this when every layer shares one name:

- Per-layer variable head count (e.g.
  ``num_attention_heads_per_layer = [48, 64, 48, 64, ...]``)
- Mixed full/sliding attention with different ``freqs_cis`` last-dim
  (partial-rotary 0.5 vs full-rotary 1.0)
- Mixed dense/sparse MLP per layer (dense layer 0 + sparse rest)

Key the subgraph name on the layer's signature:

```python
for i, layer in enumerate(self.layers):
    kind = f"{heads_per[i]}_{layer_types[i]}_{mlp_types[i]}"
    h = as_subgraph(layer, name=kind)(h, ...)
```

`nemotron_h_modulev3/nemotron_h.py` names its subgraphs by layer kind the
same way. A layer that bakes a per-layer value into the trace (such as a
KV-cache layer index created with `F.constant`) can't share a subgraph with
other layers. Call that layer directly.
