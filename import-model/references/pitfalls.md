# Universal pitfalls (index)

Find your symptom below, then load the one category file it points to.

## Config and registration → [pitfalls-config.md](pitfalls-config.md)

- [`arch.py::name` must match `architectures[0]`
  exactly](pitfalls-config.md#archpyname-must-match-architectures0-exactly)
- [`default_encoding` should match the Hub checkpoint dtype](pitfalls-config.md#default_encoding-should-match-the-hub-checkpoint-dtype)
- [Memory estimator reads HF `torch_dtype`, not the adapter's post-cast dtype](pitfalls-config.md#memory-estimator-reads-hf-torch_dtype-not-the-adapters-post-cast-dtype)
- [`gelu_new` ≠ `gelu` ≠ `gelu_tanh`](pitfalls-config.md#gelu_new--gelu--gelu_tanh)
- [Import and config API traps](pitfalls-config.md#import-and-config-api-traps)

## Weight adapter → [pitfalls-weights.md](pitfalls-weights.md)

- [Tied embeddings need shared tensor objects](pitfalls-weights.md#tied-embeddings-need-shared-tensor-objects)
- [`stacked_qkv=False` keeps HF's unfused Q/K/V key names](pitfalls-weights.md#stacked_qkvfalse-keeps-hfs-unfused-qkv-key-names)
- [`lm_head.weight` is the canonical state-dict key — never rename to `output.weight`](pitfalls-weights.md#lm_headweight-is-the-canonical-state-dict-key--never-rename-to-outputweight)
- [Don't trust `strict=False` weight loads](pitfalls-weights.md#dont-trust-strictfalse-weight-loads)
- [`numpy.from_dlpack` does not support bfloat16](pitfalls-weights.md#numpyfrom_dlpack-does-not-support-bfloat16)
- [Embedding row-count may exceed `vocab_size`](pitfalls-weights.md#embedding-row-count-may-exceed-vocab_size)

## Graph build → [pitfalls-graph.md](pitfalls-graph.md)

- [Scaffold is not a port](pitfalls-graph.md#scaffold-is-not-a-port)
- [`ops.constant` requires `device=`](pitfalls-graph.md#opsconstant-requires-device)
- [Partial-rotary padding is interleaved](pitfalls-graph.md#partial-rotary-padding-is-interleaved)
- [`ops.sum` keeps the reduced dim](pitfalls-graph.md#opssum-keeps-the-reduced-dim)
- [Stack-vs-block residual in recurrent / shared-weight architectures](pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures)
- [Subgraph cache assumes uniform layer shape](pitfalls-graph.md#subgraph-cache-assumes-uniform-layer-shape)

## Serve and verify → [pitfalls-serving.md](pitfalls-serving.md)

- [Test decode at 16+ tokens, not just 1](pitfalls-serving.md#test-decode-at-16-tokens-not-just-1)
- [`trust_remote_code=True` with `.to("cuda")` can produce NaN](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan)
- [Text comparison is best with greedy](pitfalls-serving.md#text-comparison-is-best-with-greedy)
- [`ARCHITECTURES = [arch]` export is mandatory for
  `--custom-architectures`](pitfalls-serving.md#architectures--arch-export-is-mandatory-for---custom-architectures)
- [Fake HF oracle (broken reference before you compare MAX)](pitfalls-serving.md#fake-hf-oracle-broken-reference-before-you-compare-max)
