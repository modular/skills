# Comparator output patterns

Read the signature table **and** per-token cos before choosing a fix. A
late-layer cliff means nothing if the HF dump indexed the wrong tensor.

## Signature table

| Pattern                                                           | Per-token shape         | Likely category                             | First check                                    |
|-------------------------------------------------------------------|-------------------------|---------------------------------------------|------------------------------------------------|
| `post_embed` cos < 0.99                                           | all tokens bad          | Embedding load, vocab sharding, wrong dtype | Weight key audit; same token IDs in both dumps |
| `post_embed` cos = 1.0, `layer_00` cos < 0.99 at **t=0**          | token 0 bad             | Layer-0 attn/MLP, not RoPE                  | Token-0 invariant (below)                      |
| `post_embed` cos = 1.0, **t=0 perfect**, **t≥1** bad from layer 1 | position-dependent      | RoPE, causal mask, KV, GQA                  | HF RoPE cos/sin; No-RoPE vs plain RoPE         |
| Slow monotonic cos decay                                          | drift everywhere        | BF16 accumulation, wrong attention scale    | Compare at FP32; check attention scale         |
| Cliff at layer N, HF spike, MAX flat                              | sink not forming        | Attention can't build register anchor       | Attn delta at layer N                          |
| Cliff at layer N, MAX spike, HF ~0                                | spurious MAX activation | Wrong norm order, bad residual, dump bug    | Re-verify HF dump is layer output, not norm    |
| `last_logits` cos ≥ 0.98, wrong argmax                            | near-miss logits        | Rank-1 subspace drift                       | Top-10 logit diff                              |
| Full-seq cos ≈ 1.0, last-token cos bad                            | masked by other rows    | Position-specific bug                       | `--token-index` / per-token loop               |

## Token-0 invariant

Under causal mask, attention output at token 0 is independent of Q, K, RoPE,
and mask because softmax over one key is 1.0:

```text
attn_out[token=0, :] = V[0, :, :] @ o_proj^T
```

1. Compute analytical `V[0] @ o_proj^T` from weights + HF input at t=0.
2. Compare to MAX `attn_out[t=0]` in dumps.
3. Mismatch → V projection, o_proj layout, or KV write bug, not Q/K/RoPE.

Run this before attention sub-taps when layer 0 attention diverges. Perfect
t=0 with bad t≥1 → RoPE application or mask, not V/o_proj.

**Exception — learned attention sinks.** The invariant assumes softmax over a
single key yields weight 1.0. Models that add a learned sink logit to the
softmax denominator (gpt-oss style) put probability mass on the sink at every
position, so `attn_out[t=0] ≠ V[0] @ o_proj^T` by design. Skip this check for
sink models; a t=0 mismatch there is not evidence of a V/o_proj bug.

Bisect taps may store scaled residuals; divide by the residual multiplier before
comparing raw attention.

## False cliffs (dump bugs, not graph bugs)

### Non-standard `hidden_states` layout

Some models store pre-layer inputs plus final norm, not per-layer outputs.
Indexing `hidden_states[i+1]` as layer `i` creates fake final-layer cliffs.
Use per-layer forward hooks.

### Decode-prefix without `attention_mask`

Extend `attention_mask` when appending teacher-forced tokens. Otherwise
`post_embed` cos collapses even when token IDs match.

### Hook overwrite on sub-forwards

Hooks can fire multiple times per forward. Prefer `output_hidden_states=True`
plus layer hooks when tuple semantics are ambiguous.

## Partial corruption

Verification can fail at token index 1+ while smoke passes:

- Build decode-step-K dumps (prefix = greedy tokens 0..K-1)
- Compare at `--token-index K`
- Bisect first layer where per-token cos drops below 0.99

See [stacked-failures.md](stacked-failures.md) when fixing one bug reveals
another.
