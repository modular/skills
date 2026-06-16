# Stacked failures

When fixing silent corruption, you often hit **multiple independent bugs** in
sequence. Fixing the first reveals the next. Don't declare victory after one
fix unless verification passes end-to-end.

## Bug classes (often in this order)

1. **Comparator / harness bug**: dumps misaligned, wrong HF tensor indexed,
   mask not extended, token recovery wrong. Symptom: catastrophic cos at
   `post_embed` or fake cliff at final layer while token IDs match.
2. **Real graph bug**: RoPE, norm order, MoE routing, etc. Symptom: after
   the comparator is trustworthy, per-token cos shows position-dependent drift.

Fixing (1) without (2) moves the failure index forward (for example, greedy
token 1 → token 2) rather than to PASS.

After graph math is proven on teacher-forced prefill:

3. **Pipeline decode / KV bug**: incremental `prepare_next_token_inputs`
   drift. Symptom: teacher-forced prefill matches HF; **pipeline incremental
   fails** at the same index.

4. **Serve / harness bug**: HTTP session or token recovery. Symptom:
   teacher-forced and **pipeline incremental both pass**; `max serve` or your
   verification check fails at the same index. Run the serve-vs-pipeline bisect
   in Step 6 of [SKILL.md](../SKILL.md).

## Example: four bugs in one port

| Order | Bug class            | Symptom                                                                  | Fix                                                                       |
|-------|----------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|
| 1     | Comparator / harness | `post_embed` cos 0.04; token IDs match on both sides                     | Extend `attention_mask` when appending teacher-forced tokens              |
| 2     | Comparator / harness | Fake cliff at final layer                                                | Dump layer outputs with forward hooks, not wrong `hidden_states` index    |
| 3     | Graph                | t=0 cos 1.0; t≥1 cliff from layer 1                                      | Fix RoPE variant (for example No-RoPE when `inv_freq` is zero at runtime) |
| 4     | Serve / harness      | Teacher-forced + pipeline incremental pass; serve greedy fails @ index K | Bisect tokenizer, chat template, HTTP token recovery                      |

## How to detect stacked failures

- After a "fix", matched tokens **increase** but verification still fails
- Comparator at the old failure index is green; a new index needs new dumps
- You fixed one subsystem but the symptom category didn't change

## Procedure

1. Validate dumpers on a **model already known to work in MAX** before
   trusting cliffs on a new port
2. After each fix: full verification run, not just re-dump at the same index
