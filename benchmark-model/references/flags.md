# `max benchmark` flags and datasets

`max benchmark` wraps the open-source `benchmark_serving.py` and accepts all the
same options. Run `max benchmark --help` for the exhaustive list. The groups
below cover what you actually reach for. Every run needs a running server and
either `--num-prompts` or `--num-chat-sessions`.

## Contents

- [Connecting to the server](#connecting-to-the-server)
- [Load generation](#load-generation)
- [Datasets](#datasets)
- [Output length and sampling](#output-length-and-sampling)
- [Saving results](#saving-results)
- [Stats and profiling](#stats-and-profiling)
- [Config file](#config-file)

## Connecting to the server

- `--backend`: server type, one of `modular` (default), `modular-chat`, `vllm`,
  `vllm-chat`, `sglang`, `trtllm`, and the `-chat` variants. Use `modular` for a
  `max serve` endpoint.
- `--model`: a Hugging Face ID or local path. This **must equal** the server's
  `--served-model-name` or the requests fail. Read it from `curl /v1/models`.
- `--endpoint`: either `/v1/completions` or `/v1/chat/completions` (default
  `/v1/chat/completions`). Base LMs use `/v1/completions`, which needs no chat
  template. Instruct and chat models use `/v1/chat/completions`, which needs a
  chat template.
- `--base-url`: the full base URL. This overrides `--host` and `--port` when you
  set it.
- `--host` and `--port`: default to `localhost:8000`.
- `--tokenizer`: the Hugging Face tokenizer, which defaults to `--model`. If the
  served name is an *alias* rather than a Hugging Face ID, that default can't
  resolve, so pass the model's real Hugging Face ID here or the run fails with
  "not a valid model identifier."

## Load generation

- `--num-prompts`: the number of single-turn prompts. Required unless you use
  `--num-chat-sessions`. More prompts steady the averages and lengthen the run.
- `--num-chat-sessions`: multiturn sessions, for multiturn datasets and judge
  workloads, instead of `--num-prompts`.
- `--max-concurrency`: the maximum in-flight requests. Takes a single value or a
  sweep (`1,2,4,8,16,32`). The server's `--max-batch-size` must be at least the
  top of the sweep, or the higher points just queue.
- `--request-rate`: requests per second. Takes a single value or a sweep, and
  defaults to `inf` (no limit). Use a rate sweep to measure behavior at target
  loads rather than saturating the server.
- `--seed`: the workload RNG seed, default `24301` (fixed for reproducibility).
  `--seed none` draws a fresh one, then logs and records it.
- `--workload-config`: a YAML file of workload options using hyphenated keys.
  The CLI wins over the file.

## Datasets

`--dataset-name` selects the workload. The most useful choices are the
following:

- `random`: synthetic and fully controllable. This works best for clean,
  reproducible micro-measurements. Shape it with these flags:
  - `--random-input-len` (default `1024`) and `--random-output-len` (default
    `128`). Pass a constant or a distribution: `N(mean,std)`, `U(lo,hi)`,
    `DU(lo,hi)`, `NB(n,p)`, `G(shape,scale)`, or `LN(mean,std)`. A `;` splits
    the first turn from later turns.
  - `--random-sys-prompt-ratio`, `--random-max-num-unique-sys-prompt`, and
    `--warm-shared-prefix` drive prefix-cache experiments.
  - `--random-image-count` and `--random-image-size` enable vision mode.
- `sharegpt` (default): real human and AI conversations from the Hugging Face
  Hub. This works best for a representative chat mix.
- `arxiv-summarization`: long-context summarization, sized by
  `--arxiv-summarization-input-len` (default `15000`).
- `sonnet`: poetry, sized by `--sonnet-input-len` and `--sonnet-prefix-len`.
- Code datasets: `instruct-coder`, `agentic-code`, and `nemotron-opencode` carry
  agentic and tool-call traces (toggle with `--tool-calls` or
  `--no-tool-calls`), and `code_debug` covers long context.
- Vision datasets: `vision-arena`, `local-image` (needs `--dataset-path`),
  `batch-job`, and `synthetic-pixel`.
- `synthetic`: like `random` but with synthetic token IDs, and it supports
  multiturn.
- Local overrides: some datasets accept `--dataset-path`, and `chat-judge`,
  `obfuscated-conversations`, `local-image`, and `batch-job` **require** it.

## Output length and sampling

- `--max-output-len`: the maximum tokens generated per request. This dominates
  run time.
- `--temperature`, `--top-p`, and `--top-k`: forwarded to the server. For
  reproducible throughput numbers, keep the temperature low and fixed.

## Saving results

- `--result-filename`: the JSON output path, and it creates the directories it
  needs. If you leave it unset, MAX saves nothing. Set it for anything you want
  to compare or keep.
- `--metadata key=value ...`: recorded in the JSON, for example
  `--metadata version=0.3.3 tp=1 gpu=b200`. This pays off when you diff runs
  later.
- `--log-dir`: the per-run log directory, default
  `<backend>-latency-<timestamp>`. A concurrency sweep drops a
  `results-<N>-median.json` per step here.

## Stats and profiling

- `--collect-gpu-stats` and `--no-collect-gpu-stats`: report GPU utilization and
  peak memory. NVIDIA only, and only when the benchmark runs on the **same
  instance** as the server. On by default.
- `--collect-cpu-stats` and `--collect-server-stats`: on by default.
- `--profile`: captures an Nsight Systems trace and a ranked top-N kernel
  summary. The **server must already run under `nsys launch`**. Tune the output
  with `--profile-output` and `--profile-top-n` (default 15). `--trace` and
  `--trace-file` are the lower-level form. NVIDIA only.

## Config file

`--config-file file.yaml` loads options from YAML under a top-level
`benchmark_config:` key. **Keys use `snake_case`**, so `--num-prompts` becomes
`num_prompts`. CLI flags override the file. For example:

```yaml title="bench.yaml"
benchmark_config:
  model: my_arch
  backend: modular
  endpoint: /v1/completions
  host: localhost
  port: 8000
  num_prompts: 200
  dataset_name: random
  random_input_len: 512
  random_output_len: 128
  max_output_len: 128
```

```bash
max benchmark --config-file bench.yaml --max-concurrency 1,2,4,8,16,32 \
  --result-filename results/run.json
```

Most repeatable harnesses follow one pattern: a YAML file with
`--section-name benchmark_config` plus per-run CLI overrides (`--base-url`,
`--model`, `--result-filename`, `--max-concurrency`). The file pins the workload
shape, and the CLI pins the run-specific wiring.
