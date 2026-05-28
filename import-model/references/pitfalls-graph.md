# Graph build pitfalls

In scope: Phase 2 traps surfaced while writing `model.py` and assembling
the MAX graph — `ops.*` calls, RoPE wiring, residual placement, and
subgraph grouping.

Covered:

- Scaffold is not a port
- `ops.constant` requires `device=`
- Partial-rotary padding is interleaved
- `ops.sum` keeps the reduced dim
- Stack-vs-block residual in recurrent / shared-weight architectures
- Subgraph cache assumes uniform layer shape

## Scaffold is not a port

`scaffold.py` copies a donor (`llama3`, `qwen3`, …). Until you implement
the graph, `<slug>.py` runs the **donor's** attention, block wiring, and MLP —
not your model's. Serving that graph and running logit verification will
fail; that is expected, not a tolerance or script bug. Complete
[implement-graph.md](implement-graph.md) before any verification activity.

## `ops.constant` requires `device=`

The MAX graph has no implicit "current device." Every `ops.constant`
inside a `Graph` context must specify `device=`. Without it, you'll
either get a compile error or — worse — silent wrong values.

```python
# Wrong
scale = ops.constant(1.0 / math.sqrt(d), DType.float32)

# Right
scale = ops.constant(1.0 / math.sqrt(d), DType.float32, device=x.device)
```

## Partial-rotary padding is interleaved

The base `RotaryEmbedding` produces interleaved frequency pairs:
`[cos0, sin0, cos1, sin1, ...]`. When `partial_rotary_factor < 1.0`,
pad the unrotated section with interleaved identity pairs
`[1, 0, 1, 0, ...]`, not split-half `[1, 1, ..., 0, 0, ...]`. Wrong
padding produces position-dependent errors that grow with sequence
length.

## `ops.sum` keeps the reduced dim

In MAX, `ops.sum([N, H, D], axis=-1)` returns `[N, H, 1]`, not `[N, H]`
like PyTorch. Squeeze after if you need rank-2.

## Stack-vs-block residual in recurrent / shared-weight architectures

Models that run the same transformer stack multiple times per forward
(HRM-style loops, shared-weight stacks, some encoder-decoders) mix an
injection state into the stack **once before the first block**, not at
every block inside the loop.

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
linearly; greedy text diverges from token 1–2 even when weights load cleanly.
Fix: move `z = z + injection` outside the inner layer loop (once per stack
invocation). See also
[implement-graph.md](implement-graph.md#recurrent--shared-weight-stacks-mix-the-injection-in-once).

## Subgraph cache assumes uniform layer shape

`Transformer.subgraph_layer_groups` (the optimization that compiles one
subgraph per group of identical layers and reuses it via ``ops.call``)
treats the FIRST layer's input types as the subgraph signature, then
expects every other layer in the group to call with identical operand
types. Symptom of a violation:

```text
Subgraph transformer_block_0 has wrong type for argument 29
(function type: '!mo.tensor<[2048, 64], f32, gpu:0>',
 operand type: '!mo.tensor<[2048, 128], f32, gpu:0>')
```

Per-layer-variable architectures violate this:

- Per-layer variable head count (e.g.
  ``num_attention_heads_per_layer = [48, 64, 48, 64, ...]``)
- Mixed full/sliding attention with different ``freqs_cis`` last-dim
  (partial-rotary 0.5 vs full-rotary 1.0)
- Mixed dense/sparse MLP per layer (dense layer 0 + sparse rest)

The donor likely does ``self.subgraph_layer_groups = [list(range(num_layers))]``
which is wrong for these models. Replace with signature-keyed grouping:

```python
if config.use_subgraphs:
    heads_per = config.num_attention_heads_per_layer or [num_heads] * N
    layer_types = config.layer_types or ["full_attention"] * N
    mlp_types = config.mlp_layer_types or ["sparse"] * N
    groups: dict[tuple, list[int]] = {}
    for i in range(N):
        key = (heads_per[i], layer_types[i], mlp_types[i])
        groups.setdefault(key, []).append(i)
    self.subgraph_layer_groups = list(groups.values())
else:
    self.subgraph_layer_groups = []
```

Any model with ``num_attention_heads_per_layer`` or layer-type-keyed RoPE
needs this pattern.
