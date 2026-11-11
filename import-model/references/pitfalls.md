# Universal pitfalls (index)

Find your symptom below, then load the one category file it points to.

## Config and registration → [pitfalls-config.md](pitfalls-config.md)

- [`arch.py::name` must match `architectures[0]`
  exactly](pitfalls-config.md#archpyname-must-match-architectures0-exactly)
- [`default_encoding` should match the Hub checkpoint dtype](pitfalls-config.md#default_encoding-should-match-the-hub-checkpoint-dtype)
- [`gelu` and `gelu_new` are different GELUs](pitfalls-config.md#gelu-and-gelu_new-are-different-gelus)
- [Import and config API traps](pitfalls-config.md#import-and-config-api-traps)

## Weight adapter → [pitfalls-weights.md](pitfalls-weights.md)

- [Weight names follow the root module's attribute paths](pitfalls-weights.md#weight-names-follow-the-root-modules-attribute-paths)
- [Tied embeddings keep one copy of the shared weight](pitfalls-weights.md#tied-embeddings-keep-one-copy-of-the-shared-weight)
- [Unconsumed checkpoint tensors load without an error](pitfalls-weights.md#unconsumed-checkpoint-tensors-load-without-an-error)
- [Dtype mismatches raise unless `auto_cast` permits them](pitfalls-weights.md#dtype-mismatches-raise-unless-auto_cast-permits-them)
- [`numpy.from_dlpack` does not support bfloat16](pitfalls-weights.md#numpyfrom_dlpack-does-not-support-bfloat16)
- [Embedding row-count may exceed `vocab_size`](pitfalls-weights.md#embedding-row-count-may-exceed-vocab_size)

## Module build → [pitfalls-graph.md](pitfalls-graph.md)

- [Scaffold is not a port](pitfalls-graph.md#scaffold-is-not-a-port)
- [`F.constant` defaults to the accelerator and bfloat16](pitfalls-graph.md#fconstant-defaults-to-the-accelerator-and-bfloat16)
- [Partial-rotary padding is interleaved](pitfalls-graph.md#partial-rotary-padding-is-interleaved)
- [`F.sum` keeps the reduced dim](pitfalls-graph.md#fsum-keeps-the-reduced-dim)
- [Stack-vs-block residual in recurrent / shared-weight architectures](pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures)
- [A shared subgraph assumes uniform layer signatures](pitfalls-graph.md#a-shared-subgraph-assumes-uniform-layer-signatures)

## Serve and verify → [pitfalls-serving.md](pitfalls-serving.md)

- [Test decode at 16+ tokens, not just 1](pitfalls-serving.md#test-decode-at-16-tokens-not-just-1)
- [`trust_remote_code=True` with `.to("cuda")` can produce NaN](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan)
- [Text comparison is best with greedy](pitfalls-serving.md#text-comparison-is-best-with-greedy)
- [`ARCHITECTURES = [arch]` export is mandatory for
  `--custom-architectures`](pitfalls-serving.md#architectures--arch-export-is-mandatory-for---custom-architectures)
- [Fake HF oracle (broken reference before you compare MAX)](pitfalls-serving.md#fake-hf-oracle-broken-reference-before-you-compare-max)
