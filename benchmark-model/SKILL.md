---
name: benchmark-model
description: >
  Benchmark a model served on MAX with the `max benchmark` command: measure
  throughput (tokens/sec), latency (TTFT, TPOT, inter-token latency), and GPU
  utilization by driving load against a running `max serve` endpoint. Use this
  whenever the user wants to benchmark, load-test, or measure the performance of
  a MAX model, get tokens-per-second / TTFT / TPOT numbers, run a concurrency or
  request-rate sweep, compare latency vs throughput, size a deployment, or
  produce benchmark JSON, even if they don't say "benchmark" by name. Also use
  when a `max benchmark` run fails to connect or reports zero/garbage numbers.
compatibility: Requires a pip or pixi MAX install and a running `max serve` endpoint; GPU stats (`--collect-gpu-stats`) need the benchmark to run on the same NVIDIA machine as the server.
argument-hint: "[what to measure, for example 'single-request latency' or 'peak throughput']"
---

# Benchmark a model on MAX

`max benchmark` measures a **running** model server. It's a load generator: it
sends inference requests to a live `max serve` endpoint, times them, and reports
throughput (tokens/sec) and latency (TTFT, TPOT, inter-token latency). A
benchmark combines two things: a server under test, and a workload that matches
the question you're asking.

Every run reports both throughput and latency, so there's no mode to select.
Decide what you want to learn first, then pick the workload that measures it.
Single-stream latency and peak throughput come from different workloads, so the
workload you choose *is* the measurement. Get that right and a clean number
falls out.

Use this skill when you want a performance number for a model on MAX:
tokens/sec, TTFT / TPOT, a concurrency or request-rate sweep, a
latency-vs-throughput tradeoff, or deployment sizing.

Don't use this skill when no server is running yet. `max benchmark` is a client,
so start a server with the `serve-model` skill first. To find out *where*
inference time goes at the kernel level, use `profile-model`. To check *whether
the output is correct*, treat it as a parity task (`import-model`, then
`debug-model`) rather than a benchmark.

This skill works anywhere MAX is installed (pip or pixi). Add `pixi run` in a
pixi project.

## References

The following table lists the reference files and when to read each one:

| File | Read when |
|------|-----------|
| [references/flags.md](references/flags.md) | Choosing any flag or dataset beyond the ones below |
| [references/metrics.md](references/metrics.md) | Turning the throughput and latency numbers into a conclusion |
| [references/troubleshooting.md](references/troubleshooting.md) | A run won't connect, requests fail, or the numbers look wrong |

Read the reference for what you're doing, not all of them upfront.

## 1. Check the server and read its model name

```bash
curl -s http://localhost:8000/v1/health     # 200 = ready
curl -s http://localhost:8000/v1/models      # note the served model name
```

Both checks matter before you spend a run:

- The benchmark's `--model` must equal the server's `--served-model-name`
  exactly, or every request fails. Take the value from `/v1/models` rather than
  guessing it.
- If that served name is an *alias* rather than a Hugging Face ID (no `/` in
  it), `--tokenizer` defaults to it and can't resolve, and the run dies with
  "not a valid model identifier." Pass the model's real Hugging Face ID as
  `--tokenizer`.

If nothing is serving, start a server with `max serve` (for a custom
architecture, use the `serve-model` skill). One more server setting matters:
`--max-batch-size` caps real concurrency. A sweep to `--max-concurrency 32`
against a server started with `--max-batch-size 1` queues requests instead of
batching them, so raise the server's batch size to match the sweep or the
high-concurrency points mean nothing.

Wait for `Server ready` *then* benchmark. Benchmarking during compile or warmup
produces garbage first-token times.

## 2. Pick the workload for your question

The workload is the measurement. Match it to what you want to learn:

| What you want to know | Workload |
|---|---|
| Best-case single-request latency (TTFT, TPOT) | `--max-concurrency 1`, fixed `random` shape, small `--num-prompts` |
| Peak throughput and where latency degrades | `--max-concurrency 1,2,4,8,16,32` sweep, more prompts |
| Performance under a realistic mix | `--dataset-name sharegpt`, moderate concurrency |
| Behavior at a target load | `--request-rate 1,2,4,8` sweep (requests/sec) |

A sweep answers the first two rows at once: the concurrency-1 point is the
best-case latency number, and the peak across the sweep is the throughput
number. Reach for a dedicated concurrency-1 run when you only want the latency
figure and don't want to pay for the rest of the curve.

The key knobs are the following (`references/flags.md` has the full catalog):

- `--dataset-name`: pick `random` (synthetic, shape it with
  `--random-input-len` and `--random-output-len`), `sharegpt` (real chat), or
  `arxiv-summarization` (long context). `random` works best for clean,
  reproducible micro-measurements.
- `--max-concurrency` and `--request-rate`: take a single value or a
  comma-separated *sweep* (`1,2,4,8`). A sweep is how you find the throughput
  knee.
- `--endpoint`: use `/v1/completions` for base LMs, which need no chat template,
  or `/v1/chat/completions` for instruct and chat models, which must have a chat
  template or the requests return 400.
- `--max-output-len`: sets the decode length, which dominates how long the run
  takes.
- `--num-prompts`: required for single-turn runs.

## 3. Run it, save results, add GPU stats

For best-case single-request latency, pin concurrency to 1 and keep the shape
fixed:

```bash
pixi run max benchmark --backend modular --base-url http://localhost:8000 \
  --model <served-model-name> --endpoint /v1/completions \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --max-output-len 128 --num-prompts 32 --max-concurrency 1 \
  --result-filename results/latency.json --collect-gpu-stats
```

For peak throughput and the latency knee, sweep concurrency and send more
prompts:

```bash
pixi run max benchmark --backend modular --base-url http://localhost:8000 \
  --model <served-model-name> --endpoint /v1/completions \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --max-output-len 128 --num-prompts 200 \
  --max-concurrency 1,2,4,8,16,32 \
  --result-filename results/throughput.json --collect-gpu-stats
```

Note these three things about saving and instrumenting a run:

- `--result-filename`: writes metrics to JSON and creates the directories it
  needs. Set it whenever you want to track or compare runs; without it, MAX
  saves nothing. `--metadata key=value` stamps the JSON, for example
  `--metadata tp=1 gpu=b200`. A sweep also drops a `results-<N>-median.json`
  per step under `--log-dir`.
- `--collect-gpu-stats`: adds GPU utilization and peak memory. This works only
  when the benchmark runs on the **same machine** as the server (NVIDIA).
- For version-controlled configs, put options under a `benchmark_config:` key in
  a YAML file and pass `--config-file file.yaml`. Keys use `snake_case`, and CLI
  flags override the file.

## 4. Read the metrics

The run prints throughput and latency, and a sweep prints one row per point. The
headline numbers are the following:

- Output token throughput (tok/s): the main throughput number.
- TTFT (time to first token): prefill responsiveness. Watch p50 and p99.
- TPOT and ITL (time per output token and inter-token latency): decode speed.
- GPU utilization and peak memory: reported with `--collect-gpu-stats`.

For how to turn these numbers into a conclusion, and the latency-vs-throughput
tradeoff a sweep reveals, see `references/metrics.md`.

## Troubleshooting

Match the symptom against `references/troubleshooting.md`, which covers
connection failures, model-name mismatches, tokenizer-alias errors,
chat-template 400s, flat throughput from a batch-size cap, and warmup-skewed
first-token times. Confirm that `curl /v1/health` returns 200 before you check
anything else.
