# Fast-fail thresholds

Fast-fail runs a small direct-HTTP sample before the full evaluation. Its purpose
is to catch unusable generation, request-schema problems, and answer-extraction
failures without misrepresenting them as a complete benchmark.

`scripts/_eval_tasks.py::TASK_CONFIGS` is the source of truth.

## Current seed tasks

| Task | Seed size | Floor | Metric |
|---|---:|---:|---|
| `gsm8k` | 24 | 10% | Exact match |
| `aime` | 10 | 5% | Exact match |

Override a floor with `--fast-fail-floor`:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k,aime \
  --fast-fail-floor "gsm8k:0.15,aime:0.05"
```

Use `--no-fast-fail` when only the full lm-eval result is required.

## Why multiple-choice tasks don't fast-fail

`mmlu`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, and
`truthfulqa` use lm-eval for prompt construction, tokenization boundaries, and
metric behavior. A simplified direct scorer can produce numbers that aren't
comparable to the benchmark, especially for normalized likelihood and
TruthfulQA MC2. These tasks therefore skip the direct seed phase.

Their preflight still verifies that `/v1/completions` returns prompt log
probabilities. Start the server with `--enable-echo`.

## Change a threshold

When adding or changing a direct seed task:

1. Use a deterministic sample selection.
2. Choose a floor that detects a broken request or scorer, not normal model
   quality variation.
3. Add or update its `TaskConfig` in `scripts/_eval_tasks.py`.
4. Confirm the effective configuration:

   ```bash
   pixi run --manifest-path "$SKILL_DIR/pixi.toml" validate
   ```

5. Update the table in this file.
