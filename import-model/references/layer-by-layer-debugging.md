# Reading the layer-by-layer debug output

The divergence hunt has two layers of debugging:

1. **`compare_layers.py`** — automated logit probes (what the script actually
   runs).
2. **Manual `ops.output()` taps** — true per-block tensor diffs (when logit
   probes aren't enough).

## Running `compare_layers.py`

From the repo root (via the OSS shim):

```bash
pixi run python compare_layers.py <HF_MODEL_ID> \
  --slug <your_slug> \
  --port 8000 \
  --prompt "The capital of France is"
```

Requires `pixi run max serve` with `--custom-architectures <port_dir>` (the slug
folder, not its parent).

Flags:

| Flag                            | Purpose                                                            |
|---------------------------------|--------------------------------------------------------------------|
| `--dtype bfloat16\|float32`     | Must match `--quantization-encoding` on serve (default `bfloat16`) |
| `--skip-hf-layers`              | Skip HF hidden-state stats (faster)                                |
| `--skip-multi`                  | Skip multi-position prefix probe                                   |
| `--long-prompt "..."`           | Prompt for multi-position probe (~110 tokens default)              |
| `--positions 1,5,20,50,100,200` | Prefix lengths to compare                                          |

Preflight before serve (guard + scaffold):

```bash
pixi run python check_walls.py <HF_MODEL_ID>
pixi run python run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>/
```

After serve (during the divergence hunt):

```bash
pixi run python run_oss_gates.py <HF_MODEL_ID> \
  --port-dir <port_dir>/ --phase verify --slug <slug> --port 8000
```

---

## What the script prints

### 1. HF hidden-state stats (HF-only)

```text
  layer        mean     max_abs        norm
    0      0.0001      0.4200       12.34
    ...
```

These rows are **diagnostic only** — MAX does not expose hidden states through
the completions API. Use them to confirm HF loads sanely (no NaNs, reasonable
norms). They do **not** compare against MAX.

### 2. Short-prompt top-1 logprob

```text
       check            hf           max    rel_diff  verdict
top1_logprob      -0.1234      -0.1250      0.0130  ok
```

| Verdict    | Meaning                                          |
|------------|--------------------------------------------------|
| `ok`       | Relative logprob difference &lt; 5%              |
| `DIVERGED` | Prefill logits disagree at the last prompt token |

If this row diverges at token 0, the bug is early: tokenizer, embedding,
norm order, or attention block 0.

### 3. Multi-position top-1 probe

```text
   pos     hf_id              max_text  verdict
     1      1234                  ' the'  ok
     5      5678                  ' fox'  ok
    50      9012                  ' lap'  DIVERGED
```

The script decodes HF's argmax token id and compares to MAX's greedy completion
text at each prefix length.

**First `DIVERGED` row** localizes position-dependent bugs:

| First divergence at | Likely cause                                  |
|---------------------|-----------------------------------------------|
| Position 1–5        | Tokenizer, embedding, early attention         |
| Position 5–20       | GQA repeat, QK-norm, head layout              |
| Position 50+        | RoPE style/theta, partial-RoPE padding        |
| Position 100+       | Decode-path position handling, sliding window |

See [divergences.md](divergences.md) for fixes.

---

## Manual per-block taps (when logit probes aren't enough)

When short and multi-position probes pass but output is still wrong — or you
need to know *which sublayer* diverged — add temporary `ops.output(...)` taps
in your port's `<slug>.py` at block boundaries:

- post-embed (layer 0 only)
- post-attention
- post-MLP
- post-block residual

Re-run serve, read the tapped tensors, and compare against HF
`output_hidden_states` at the same point.

### Interpreting tap diffs

If the first divergence is at:

- **post-embed** — check `weight_adapters.py` for `embed_tokens.weight`; check
  `tie_word_embeddings`.
- **post-attn (layer 0)** — attention: Q/K/V naming, RoPE style, GQA, QK-norm,
  sliding-window mask.
- **post-mlp (layer 0)** — MLP: activation variant (`gelu_new` vs `gelu_tanh`),
  gated layout, bias terms.
- **post-block-norm** but attn and MLP ok — block wiring: pre-norm vs parallel
  vs dual-norm, residual order, MuP multipliers.

### Scaling vs direction bugs

- **Similar direction, different magnitude** — missing scale factor, wrong
  softmax scale, wrong norm formulation (`x * weight` vs `x * (1 + weight)`).
- **Similar magnitude, low cos_sim** — RoPE style mismatch, head permutation,
  sign error in `rotate_half`.

---

## Iteration loop

1. Run `compare_layers.py` (or `run_oss_gates.py --phase verify`).
2. Localize from the first `DIVERGED` row (logprob row or multi-position row).
3. Open the matching HF `forward()` for that layer/component.
4. Fix one operation in `<slug>.py` or `weight_adapters.py`.
5. **Restart `pixi run max serve`** — stale compiled graphs hide fixes.
6. Re-run.

Each pass should push the first divergence later. If it doesn't, revert and
re-read HF source.

---

## Limitations

- MAX completions API has no per-layer hidden-state export — taps are manual.
- Top-1 text comparison assumes greedy decode; use `temperature=0.0` on serve.
- HF reference loads with `trust_remote_code=True` and `device_map="auto"`.
- Both sides must use the same dtype (`--dtype` must match serve encoding).

For `trust_remote_code` models that NaN with `.to("cuda")`, see
[divergences.md](divergences.md) #19 and
[pitfalls-serving.md](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan).
