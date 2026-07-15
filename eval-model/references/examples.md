# Evaluation examples

Set `SKILL_DIR` to the absolute `eval-model` skill directory, then run
`pixi install --manifest-path "$SKILL_DIR/pixi.toml"`.

## Check generation quality

From the Pixi workspace where MAX is installed, start the server:

```bash
pixi run max serve --model meta-llama/Llama-3.1-8B-Instruct
```

Run a GSM8K seed and full evaluation:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k
```

## Smoke-test the evaluator

Use small limits and the bundled full scorer:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k \
  --seed-limit 4 \
  --full-limit 8 \
  --direct-http
```

This is useful for validating connectivity and output parsing. Don't compare
small-limit results with published benchmark scores.

## Run multiple-choice tasks

Start the server with prompt echo:

```bash
pixi run max serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-echo
```

Then run lm-eval tasks. To reproduce published leaderboard scores, leave
`--apply-chat-template` off — those references use raw completion prompts:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mmlu,hellaswag,arc_easy
```

If the server exposes an alias, provide the tokenizer separately. `--tokenizer`
is required for multiple-choice tasks on an aliased model — without it lm-eval
tries to load the alias as a Hugging Face repo and fails:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "production-model" \
  --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mmlu
```

Add `--apply-chat-template` only when you want to measure the model as it's
actually chatted to, and are comparing against a chat-template baseline rather
than raw-completion leaderboard numbers.

## Compare with explicit targets

Targets are optional and should come from a model card or established baseline:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k,mmlu \
  --target "gsm8k:0.50,mmlu:0.60"
```

The command exits `2` if a full result misses a specified target.

## Evaluate a reasoning model

Use reasoning effort only when the served model's chat template accepts the
`reasoning_effort` option through `chat_template_kwargs`:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k,aime \
  --reasoning-effort low
```

The preflight request reports an error if the endpoint rejects the field.

## Evaluate long context

`babilong` defaults to a 16K sequence-length metadata value:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks babilong
```

Override the metadata when required:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks babilong \
  --lm-eval-metadata '{"max_seq_lengths": "32k"}'
```

## Capture machine-readable progress

Keep heartbeat JSONL separate from human logs:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k \
  > heartbeats.jsonl \
  2> eval.log
```

## Use a remote or prefixed endpoint

`--base-url` accepts either the server root or an OpenAI `/v1` base:

```bash
OPENAI_API_KEY="<TOKEN>" \
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --base-url "https://example.com/v1" \
  --model "<MODEL_ID>" \
  --tasks gsm8k
```

## Choose artifact paths

Use a short run label:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --run-id nightly-smoke \
  --tasks gsm8k
```

Or set the complete output directory:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --output-dir artifacts/evals/model-a \
  --tasks gsm8k
```
