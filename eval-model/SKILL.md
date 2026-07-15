---
name: eval-model
description: >
  Measures the task accuracy of text models served by MAX using standard
  benchmarks such as GSM8K, MMLU, HellaSwag, ARC, AIME, GPQA, TruthfulQA,
  WinoGrande, and BABILong. Use when benchmarking a served model, comparing it
  with model-card or reference scores, verifying that a new MAX model produces
  correct answers, or running repeatable dataset evaluations against a MAX
  OpenAI-compatible endpoint.
argument-hint: "--model <MODEL_ID> --tasks gsm8k [--host localhost]"
disable-model-invocation: true
compatibility: "Pixi, network access for datasets, and a running MAX endpoint."
---

# Evaluate a model's accuracy on MAX

A model that serves cleanly can still answer benchmark questions wrong. This
skill measures how accurately a text model behind MAX answers standard
datasets — GSM8K, MMLU, HellaSwag, ARC, AIME, GPQA, TruthfulQA, WinoGrande, and
BABILong. It checks endpoint compatibility before each run, separates serving
failures from wrong answers, and writes reproducible per-task scores you can
compare against a model card.

**Use this skill when** you're benchmarking a served model, comparing it with
model-card or reference scores, verifying that a newly imported MAX model answers
correctly, or running repeatable dataset evaluations against a MAX
OpenAI-compatible endpoint.

**Do not use this skill when** the model doesn't serve yet. Bring the model up
first (`import-model`); if it serves but generates wrong output, chase the
divergence with `debug-model` before benchmarking.

## References

| File | Read when |
|------|-----------|
| [references/examples.md](references/examples.md) | You want more command variants — remote endpoints, artifact paths, long context |
| [references/fast-fail-thresholds.md](references/fast-fail-thresholds.md) | Before changing seed floors or adding a seed task |
| [references/heartbeat-protocol.md](references/heartbeat-protocol.md) | Building a program that consumes the progress stream |

Read the reference for what you're doing, not all of them upfront.

## Set up the evaluator

Resolve the absolute path to the directory containing this `SKILL.md` and save
it as `SKILL_DIR`. Every command below depends on it, so set it from the skill's
real location — not the user's working directory and not the literal placeholder
below. If you were handed the skill path, export that directly; otherwise resolve
it from the path to this file:

```bash
export SKILL_DIR="$(cd "$(dirname /path/to/eval-model/SKILL.md)" && pwd)"
```

Confirm it points at the manifest before continuing, so a wrong value fails here
rather than midway through a run:

```bash
test -f "$SKILL_DIR/pixi.toml" && echo "SKILL_DIR ok: $SKILL_DIR"
```

Install the evaluator environment from its Pixi manifest:

```bash
pixi install --manifest-path "$SKILL_DIR/pixi.toml"
```

This environment holds the evaluator's own dependencies (lm-eval, datasets,
transformers).

Some datasets require internet access, Hugging Face authentication, or license
acceptance. Set `HF_TOKEN` before the run when the dataset requires it.

## Serve the model

The tasks fall into two groups that serve the model differently — decide which
you're running, or serve so you can run both:

- **Generation tasks** (`gsm8k`, `aime`) score a generated answer and run
  against a normally served model.
- **Multiple-choice tasks** (`mmlu`, `hellaswag`, `arc_easy`, `arc_challenge`,
  `winogrande`, `truthfulqa`, `gpqa`) score by prompt log probabilities, which
  MAX only returns when the server starts with `--enable-echo`.

Serve from an environment where MAX is installed.

For generation tasks, start the model normally:

```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct
```

For multiple-choice tasks, add `--enable-echo`:

```bash
max serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-echo
```

You can't toggle echo on a live server. If one is already running without
`--enable-echo`, stop and restart it before evaluating a multiple-choice task —
there's no way to add the capability to a running process. The evaluator's
preflight is the authoritative check: it verifies prompt log probabilities
before downloading datasets and reports a configuration error if they're
unavailable.

If a model's `generation_config.json` sets a non-default `repetition_penalty`
(the Qwen2.5 family ships `1.05`, for example), MAX auto-enables its penalty
sampling path, which is shape-incompatible with echo and crashes the serving
worker at startup with a `repetition_penalty` tensor-shape error. Models that
leave `repetition_penalty` unset (such as Llama-3.1 and Qwen3) don't trigger this
path. If you hit the crash, evaluate a model whose generation config leaves
`repetition_penalty` unset, or restrict the run to generation tasks (`gsm8k`,
`aime`).

## Run an evaluation

Invoke the evaluator through its Pixi manifest from any working directory:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k
```

For a quick endpoint and scoring check, lower both limits and use the bundled
scorer:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k \
  --seed-limit 4 \
  --full-limit 8 \
  --direct-http
```

Use `--direct-http` only for connectivity and parsing checks like this one. Its
bundled scorer is 0-shot with a custom prompt, so its numbers run far below the
few-shot chain-of-thought scores on a model card — in one measured run the same
model scored 33% under the direct seed scorer and 82% on the lm-eval full pass.
Don't compare `--direct-http` results with published benchmarks; drop the flag
for any card comparison so the default lm-eval pass runs instead.

For multiple-choice tasks on an aliased served model, pass the Hugging Face
tokenizer ID with `--tokenizer`. This is required, not optional: lm-eval needs a
real tokenizer to compute token boundaries, and without it the tokenizer
defaults to `--model` and lm-eval fails trying to load the alias as a Hugging
Face repo.

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "my-served-alias" \
  --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mmlu,hellaswag
```

The `--model` value must match an ID returned by `/v1/models`.

`--apply-chat-template` wraps each prompt in the model's chat template. Whether
to use it depends on your comparison baseline, not just on whether the model is
instruction-tuned:

- To reproduce published multiple-choice leaderboard scores (MMLU, HellaSwag,
  ARC, WinoGrande), leave it **off** — those references are measured on raw
  completion prompts. Adding it succeeds but produces numbers that can't be
  compared to the card.
- Use it only when you intend to measure the model as it's actually chatted to,
  and are comparing against a chat-template baseline.

## Interpret the result

The command uses these exit codes:

- `0`: All tasks completed, or every explicit `--target` was met.
- `1`: Setup, dependency, endpoint, dataset, or lm-eval failure.
- `2`: At least one explicit `--target` was missed.
- `3`: A direct seed task scored below its fast-fail floor.

Without `--target`, a valid full run has status `completed`; the evaluator
doesn't impose model-independent accuracy targets. A `completed` or `met` status
means the run finished cleanly — it's not by itself a verdict that the model is
correct or at parity. To turn a run into a pass/fail or a comparison, supply a
baseline yourself.

### Compare against a model card

The evaluator doesn't know any model's expected scores, so a real comparison is
a few explicit steps:

1. Look up the specific metric on the model card or reference, matching the
   variant this tool produces — for generation tasks that's few-shot
   chain-of-thought (for example, Llama-3.1-8B-Instruct reports GSM8K around
   84.5, 8-shot CoT). Note the shot count and whether a chat template was used.
2. Pass that number as `--target`. Be aware `--target` is a one-sided floor: the
   run fails (exit `2`) only when accuracy falls *below* it, so it gates rather
   than measuring a two-sided delta.
3. Read `accuracy` from the task's summary JSON and report the actual difference
   from the reference. Allow for sampling noise — the default full run is 200
   samples per task (or per subtask for grouped tasks), not the full test set,
   so treat small gaps as noise rather than regressions. Raise `--full-limit`
   toward the full dataset when you need a tighter estimate.

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k,mmlu \
  --target "gsm8k:0.84,mmlu:0.68"
```

Summaries default to `eval-results/<model-name>/`:

```text
eval-results/meta-llama_Llama-3.1-8B-Instruct/
├── eval_summary.json
├── gsm8k_summary.json
└── mmlu_summary.json
```

Use `--output-dir` to choose another location or `--run-id` to change only the
default directory label. The combined summary records the effective CLI
configuration and evaluator dependency versions needed to interpret the run.

## Choose tasks

Task behavior is as follows:

- `gsm8k` and `aime`: Direct-HTTP seed check, followed by an lm-eval full run.
  Add `--direct-http` to use the bundled exact-match scorer for the full run.
- `mmlu`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, and
  `truthfulqa`: lm-eval full run through `/v1/completions`; require
  `--enable-echo`.
- `gpqa`: lm-eval multiple-choice run through `/v1/completions`; requires
  `--enable-echo`, and the dataset can require Hugging Face access.
- `babilong`: lm-eval chat run; defaults to
  `{"max_seq_lengths": "16k"}` metadata.

Each full run defaults to 200 samples (200 per subtask for grouped tasks such as
`mmlu`; `gpqa` and `babilong` default to 100). This is a spot check, not the
whole test set — raise `--full-limit` when you need a tighter estimate, and
expect small run-to-run variation at the default size. The summary reports the
aggregate number of effective child samples for grouped tasks.

Only the generation tasks (`gsm8k`, `aime`) run the direct seed pass; the
multiple-choice tasks skip straight to the full run, so seeing no seed accuracy
for `mmlu` and friends is expected. The seed's fast-fail floor is a
plumbing gate — it catches an unusable endpoint, a malformed request, or broken
answer extraction, not normal model-quality variation. To judge quality, use
`--target` (see "Compare against a model card"), not the fast-fail floor.

ChartQA isn't registered because stock lm-eval ChartQA prompt formatting isn't
compatible with MAX image-token injection. Add a validated MAX-specific task
configuration before enabling it.

Print the effective task configuration without installing MAX or starting a
server:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" validate
```

## Evaluate reasoning models

For a MAX model whose chat template accepts `reasoning_effort`, pass one of
`low`, `medium`, `high`, or `xhigh`:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k,aime \
  --reasoning-effort low
```

The evaluator sends this value through
`chat_template_kwargs={"reasoning_effort": ...}` and automatically selects the
direct-HTTP full scorer for supported generation tasks. The preflight request
fails clearly if the endpoint doesn't accept the field.

## Monitor progress

Heartbeat records go to stdout; human-readable logs go to stderr. To archive
them, capture each stream to its own file:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k \
  > heartbeats.jsonl \
  2> eval.log
```

To watch liveness *while* the run proceeds, keep stdout on a pipe rather than a
plain file, so a consumer can read heartbeats as they arrive — the example
consumer in the heartbeat reference can't tail a file redirect alone:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" --tasks gsm8k \
  2> eval.log | tee heartbeats.jsonl | python your_consumer.py
```

Treat a missing heartbeat for twice `--heartbeat-interval` as a possible stall.
For reasoning models, a single generation can legitimately exceed the default
30-second interval, so raise `--heartbeat-interval` (and with it the stall
threshold) to avoid false alarms on slow-but-healthy runs. The evaluator emits
`done` only after writing the corresponding task summary.

## Troubleshoot failures

Use these checks:

- Connection or model error: Run `curl http://localhost:8000/v1/models` and
  pass an exact returned model ID.
- Missing prompt logprobs: Restart MAX with `--enable-echo` (it can't be
  enabled on a running server); confirm the selected runtime supports log
  probabilities.
- Serving worker crashes on startup with echo and a `repetition_penalty`
  tensor-shape error: The served model's `generation_config.json` sets a
  non-default `repetition_penalty` (common in Qwen2.5), so MAX auto-enables a
  penalty path that echo doesn't support. Use a model whose generation config
  leaves `repetition_penalty` unset (Llama-3.1, Qwen3) for multiple-choice tasks,
  or restrict the run to generation tasks (`gsm8k`, `aime`).
- `No module named 'pydantic'` from `max serve`: The serving environment has the
  slim `max` package but not `max-pipelines`. Install `max-pipelines`.
- Missing package: Run
  `pixi install --manifest-path "$SKILL_DIR/pixi.toml"` again.
- Dataset access error: Authenticate with Hugging Face and accept the dataset's
  terms.
- lm-eval task error: List tasks for the installed lm-eval version with
  `pixi run --manifest-path "$SKILL_DIR/pixi.toml" lm-eval ls tasks`, then print
  the effective task configuration with
  `pixi run --manifest-path "$SKILL_DIR/pixi.toml" validate` (the `validate` task
  runs the script's `--validate-thresholds` mode).
- Out-of-memory or context-length error: Reduce `--num-concurrent`,
  `--full-limit`, or the model's served context length as appropriate.
