# Reading `max benchmark` results

A benchmark run prints the metrics below, and a `--max-concurrency` or
`--request-rate` sweep prints one row per point. This page shows you how to turn
those numbers into a conclusion.

## The metrics

Each run reports the following metrics:

- Request throughput: completed requests per second. This helps most for
  short-response workloads; for generation, output token throughput matters
  more.
- Input token throughput: input tokens processed per second, which measures
  prefill work.
- Output token throughput (tok/s): generated tokens per second, aggregated
  across all concurrent requests. This is the headline throughput number.
- TTFT (time to first token): the time from request start to the first token,
  which is *prefill* latency, or how fast the model responds before it streams.
  Watch p50 and p99, because p99 is what tail users feel.
- TPOT (time per output token): the average decode time per token after the
  first. `1 / TPOT` approximates per-stream decode tok/s.
- ITL (inter-token latency): the time between consecutive tokens or chunks. It
  tracks close to TPOT, and its p99 exposes stalls and jitter in streaming.
- GPU utilization and peak GPU memory: reported with `--collect-gpu-stats`.
  The console prints them in the same percentile table as latency and
  throughput; the JSON `gpu_stats` group keeps the per-GPU lists. Low
  utilization at high concurrency means something other than compute
  bottlenecks the run, such as batching, tokenization, or host work.

Multiturn workloads add a per-turn cached token rate and per-turn KV cache
retention, which show prefix-cache effectiveness across turns.

In the result JSON, the aggregate `output_throughput` and
`total_token_throughput` fields can hold `null` for a single-stream
(concurrency 1) run. Read the per-stream `mean_output_throughput` instead, or
derive the aggregate from `total_output_tokens / duration`.

## What a sweep tells you about latency and throughput

Single-stream runs (`--max-concurrency 1`) give the *best-case* latency, with no
queuing and no batching contention. Quote that number for "how fast is one
request," but remember that it leaves the GPU underused.

As concurrency rises, the server batches requests, so output token throughput
climbs while per-request latency degrades. TTFT and ITL climb because each
request shares the GPU and may queue. A sweep maps that curve, and it gives you
three useful readings:

- Peak throughput: the highest output tok/s across the sweep, and the
  concurrency where it plateaus. Past the plateau, you add latency and gain no
  throughput.
- The latency knee: the concurrency where TTFT and ITL p99 start climbing
  steeply. For an SLA such as "TTFT p99 under 500 ms," pick the highest
  concurrency that still meets it, which gives you the best throughput within
  budget.
- Flat throughput across the sweep: a red flag. The server's `--max-batch-size`
  sits below the sweep, so requests queue instead of batching. Raise it and
  re-serve.

## Getting numbers you can trust

Follow these practices so your numbers hold up:

- Warm up: benchmark only after `Server ready`, and after a few requests have
  gone through. The first request pays one-time costs such as cache priming,
  which distort TTFT.
- Fix the seed (default `24301`) and the workload shape so runs stay comparable.
  Use `--dataset-name random` with explicit `--random-input-len` and
  `--random-output-len` for the cleanest apples-to-apples comparison.
- Send enough prompts. Too few make the averages noisy. 32 prompts suit a quick
  latency check, and 200 or more give stable throughput numbers.
- Save and stamp the run: pass
  `--result-filename run.json --metadata gpu=b200 tp=1 build=<nightly>` so a run
  describes itself and you can diff it later.
- Isolate the variable. When you compare two configs (dtype, batch size, or TP
  degree), change one thing at a time and keep the workload identical.

## A quick decision guide

Match your question to a workload:

- "How fast is a single request?" Run `--max-concurrency 1` with a fixed
  `random` shape, then report TTFT p50/p99 and TPOT.
- "What's the max this GPU can push?" Sweep `--max-concurrency 1,2,4,8,16,32`,
  then report peak output tok/s and the concurrency it needs.
- "Can it hold my SLA at load?" Run the same sweep, then read the highest
  concurrency whose TTFT and ITL p99 still meet the target.
- "How does it do on real traffic?" Run `--dataset-name sharegpt` at moderate
  concurrency.
