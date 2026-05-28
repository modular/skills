# Weight adapter pitfalls

In scope: Phase 2 traps surfaced while writing `weight_adapters.py`,
mapping HF safetensor keys to MAX FQNs, and handling BF16 tensors.

Covered:

- Tied embeddings need shared tensor objects
- `stacked_qkv=False` keeps HF's unfused Q/K/V key names
- `lm_head.weight` is the canonical state-dict key — never rename to
  `output.weight`
- Don't trust `strict=False` weight loads
- `numpy.from_dlpack` does not support bfloat16
- Embedding row-count may exceed `vocab_size`

## Tied embeddings need shared tensor objects

If `config.tie_word_embeddings=True`, `lm_head.weight` must be the *same
tensor object* as `embed_tokens.weight`. Copying the values isn't enough
— the weight loader must alias them. Verify in your weight adapter that
when `tie_word_embeddings=True`, the lm_head key is set to the same
tensor as the embedding key.

## `stacked_qkv=False` keeps HF's unfused Q/K/V key names

`AttentionWithRope(stacked_qkv=False)` builds
`StackedLinear(stacked=False, names=["q_proj","k_proj","v_proj"])`, which
sets `_omit_module_attr_name=True`. The Q/K/V weights are therefore
exposed at `self_attn.{q,k,v}_proj.weight` — exactly what HF ships. **Do
not** rename to `qkv_proj.q/k/v.weight`; that path silently orphans the
projections under `strict=False` and logit checks return gibberish. The
fused `self_attn.qkv_proj.weight` name applies only to
`stacked_qkv=True`.

## `lm_head.weight` is the canonical state-dict key — never rename to `output.weight`

`max.nn.Transformer.__init__` sets `self.lm_head = output`, so the output
projection's weight lands at `lm_head.weight` in the loaded state dict,
**not** `output.weight`. Some donor scaffolds incorrectly rename
`lm_head.` → `output.` in `weight_adapters.convert_safetensor_state_dict`.
Under `strict=False`, this silently leaves `lm_head.weight`
zero-initialized and the model emits repeated punctuation or garbage
with no load warning.

**Rule:** in `weight_adapters.py`, treat `lm_head.weight` (and
`embed_tokens.weight`) as canonical names — pass through, do not
rewrite to a donor-internal alias. State-dict audit
(see [`state-dict-audit.md`](state-dict-audit.md)) catches this.

## Don't trust `strict=False` weight loads

When a `state_dict` load reports "missing keys" or "unexpected keys",
it's reporting silently-wrong behavior. Missing keys mean some MAX
parameters stayed at their random initialization. Unexpected keys mean
some HF tensors were dropped on the floor.

Either case will pass through to runtime as garbage output with no
error. Always make `weight_adapters.py` produce *exactly* the expected
MAX FQNs — no missing, no extra.

## `numpy.from_dlpack` does not support bfloat16

`np.from_dlpack(weight_data.data)` crashes with
`RuntimeError: Unsupported dtype in DLTensor` when the underlying
buffer is bfloat16 (or fp8, fp4, etc.). NumPy has no native bfloat16
dtype. For BF16 weight manipulation in a weight adapter (slicing,
reshaping, repacking), use torch as the DLPack bridge:

```python
import torch
from max.graph import Shape
t = torch.from_dlpack(weight_data.data)
t_sliced = t[:vocab_size].contiguous()
new_wd = WeightData(
    data=t_sliced,
    name=name,
    dtype=weight_data.dtype,
    shape=Shape((vocab_size, hidden_size)),   # MUST be Shape, not tuple
    quantization_encoding=weight_data.quantization_encoding,
)
```

The `shape` argument **must be a `max.graph.Shape`**, not a `tuple`.
The layer-loader validator compares the WeightData shape against the
graph's expected `[Dim, Dim, …]` and rejects raw tuples with
`expected=[Dim(N), Dim(D)], actual=(N, D)`. Always wrap in
`Shape((...))`.

## Embedding row-count may exceed `vocab_size`

Some HF models size their `nn.Embedding` larger than
`config.vocab_size` to make room for special image/tile/audio tokens
that the LM head doesn't predict over. Mllama ships
`nn.Embedding(vocab_size + 8, hidden_size)` with
`lm_head: Linear(hidden_size, vocab_size)` — embed is 128264 rows,
lm_head is 128256. MAX sizes `embed_tokens` at `vocab_size` and fails
to load 128264-row weights. The weight adapter must slice the first
`vocab_size` rows of `embed_tokens.weight`; the extra rows are
reserved special tokens that never appear in text-only generation,
so truncation is safe for that scope. Use the torch-DLPack pattern
above; NumPy will crash on BF16.
