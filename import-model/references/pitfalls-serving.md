# Serve and verify pitfalls

In scope: Phase 3 traps surfaced while serving the port, comparing
against an HF oracle, and running text/logit verification.

Covered:

- Test decode at 16+ tokens, not just 1
- `trust_remote_code=True` with `.to("cuda")` can produce NaN
- Text comparison is best with greedy
- `ARCHITECTURES = [arch]` export is mandatory for `--custom-architectures`
- Fake HF oracle (broken reference before you compare MAX)

## Test decode at 16+ tokens, not just 1

`max_tokens=1` only exercises the prefill path. Several common bugs only
show up in the decode loop:

- RoPE theta wrong → outputs degrade as position grows.
- Partial-rotary padding wrong → grows with position.
- Absolute positional embedding reset → every step uses position 0.
- Sliding-window mask wrong at decode → behavior changes after window
  size.

Always run a length sweep before declaring a port done:

```bash
for n in 1 16 64 200 2000; do
  curl -s http://localhost:8000/v1/completions ... -d "{\"max_tokens\": $n, ...}"
done
```

Look for repetition, vocab collapse (same token 100 times), or non-word
runs (token IDs that decode to single characters).

## `trust_remote_code=True` with `.to("cuda")` can produce NaN

Some custom remote-code models build rotary embedding caches in their
constructor, on the device they were constructed on. The pattern
`from_pretrained(...).to("cuda")` leaves those caches on CPU, and
subsequent forward passes silently produce all-NaN logits.

Always load with `device_map="auto"` for `trust_remote_code` models on
CUDA hosts:

```python
model = AutoModelForCausalLM.from_pretrained(
    "<HF_MODEL_ID>",
    trust_remote_code=True,
    device_map="auto",
    dtype="bfloat16",
)
```

## Text comparison is best with greedy

For the final text-vs-text check, both implementations must use the same
sampling settings. Use greedy (`temperature=0.0`, `top_p=1.0`) for the
comparison. Any nonzero temperature introduces variance that will
diverge across implementations even when the models are bit-identical.

## `ARCHITECTURES = [arch]` export is mandatory for `--custom-architectures`

`pixi run max serve --custom-architectures <port_dir>` loads the slug package:
MAX appends `dirname(<port_dir>)` to `sys.path` and imports
`basename(<port_dir>)`. Pass the **slug folder** (contains `arch.py`), not its
parent — passing the parent imports the wrong module name and serve crashes
with `AttributeError: module 'custom-arch' has no attribute 'ARCHITECTURES'`.

The slug's `__init__.py` must expose a top-level list:

```python
ARCHITECTURES = [my_arch]
```

## Fake HF oracle (broken reference before you compare MAX)

A port can look green against a Hugging Face reference that never loaded
correctly. To avoid that:

- Before doing layer-by-layer parity, run a coherence sanity-check on
  the HF reference alone, on the **model card's intended prompt
  template** (read the card — PrefixLMs need conditional prefix
  tokens, instruct models need their chat template, etc.). If HF
  itself outputs gibberish on the model card's example prompt, **do
  not proceed to compare against the port** — the oracle is broken.
- Cross-check the loaded HF state-dict against the on-disk safetensor
  keys. If the model class declares separate Q/K/V but the safetensor
  has only fused gqkv, there must be a corresponding entry in
  `transformers.conversion_mapping._build_checkpoint_conversion_mapping`
  (or in the model's `_pre_load_state_dict_hook`). If there isn't,
  bump `transformers` or build a manual fuse-split reference and
  verify it generates coherent text in isolation before using it as
  an oracle.
- The MAX port's adapter and slice order are independent of this —
  they need to match the *on-disk* layout, not whatever HF chose to
  split it into. Confirm with a one-shot empirical test:
  load the on-disk fused weight, chunk it `dim=0`, and verify each
  chunk matches the corresponding loaded `*_proj.weight` in HF.

Automated gates can pass while greedy text still diverges from HF on the model
card's template — often because the HF reference never loaded correctly
(zero-init attention from unfused QKV never mapped). That is fake-oracle parity,
not a finished port. A separate class of bug is injecting state inside every
block of a recurrent stack instead of once per stack invocation — see
[Stack-vs-block residual in recurrent / shared-weight architectures](pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures).
