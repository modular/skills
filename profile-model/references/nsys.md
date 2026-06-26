# Kernel breakdown on NVIDIA with Nsight Systems (`nsys`)

Use this to answer *"where does my model spend time?"* on NVIDIA GPUs. `nsys`
captures the full timeline — CUDA API calls, GPU kernels, memory ops — and,
with MAX profiling markers enabled, NVTX ranges that label the prefill / decode
/ sampling phases.

Confirm `nsys` is available first: `which nsys`. If missing but CUDA is
installed, `export PATH=/usr/local/cuda/bin:$PATH`; otherwise
`sudo apt-get install nsight-systems` or download from NVIDIA. Use version
2024.4.2 or newer (the version the MAX benchmark images pin); older `nsys`
releases have known issues. Check with `nsys --version`.

## Enabling MAX markers

MAX only emits NVTX markers when you ask it to. Prefer the
`MODULAR_ENABLE_PROFILING=detailed` environment variable: it needs no code or
command change and travels with the process when you wrap `max serve` under
`nsys`. Values are `off` (default), `on` (kernel-correlation markers), and
`detailed` (adds Python-level NVTX); use `detailed`.

Two alternatives override the environment variable when set: the
`max serve --gpu-profiling detailed` CLI flag, and
`InferenceSession.gpu_profiling("detailed")` in Python (call it before
`load()`).

## Fast path: `max generate --profile`

The lowest-friction breakdown. `max generate` and `max benchmark` accept
`--profile`: this captures the timed run to an `.nsys-rep` and prints a ranked
top-N GPU-kernel table to the terminal. `cudaProfilerStart/Stop` bounds the
capture, so it excludes warmup and compile automatically.

```bash
pixi run max generate \
  --model-path modularai/Llama-3.1-8B-Instruct-GGUF \
  --prompt "Explain GPU profiling in one sentence." \
  --max-length 64 --num-warmups 1 \
  --profile
```

Output ends with a GPU-kernel table plus a Python/CPU table. For example, on
Llama-3.1-8B-GGUF:

```text
=== Top 15 GPU kernels (98.9% of GPU time, 71.35 ms total) ===
     %       total     calls  kernel
 33.6%    23.96 ms       672  gemv_split_k_bfloat16_bfloat16_bfloat16_256_f519d2c2
 20.3%    14.52 ms       651  gemv_split_k_bfloat16_bfloat16_bfloat16_128_e7d08336
 11.2%     8.02 ms       672  gemv_split_k_bfloat16_bfloat16_bfloat16_128_74024e6c
  4.7%     3.38 ms       672  sm100_mha_1q_depth128_bfloat16_bfloat16_nqh32_nkvh8_…
  3.8%     2.70 ms      1430  rms_norm_gpu_warp_tiling_bfloat16_…
  ...
Full profile saved to: /home/.../quickstart/max-profile.nsys-rep
Open with: nsys-ui /home/.../quickstart/max-profile.nsys-rep
```

The decode-heavy `gemv_split_k` GEMVs dominating, with attention (`sm100_mha`)
and `rms_norm` below them, is the expected steady-state mix for an LLM.

Options: `--profile-output PATH` (where the `.nsys-rep` goes),
`--profile-top-n N` (rows in each table, default 15). Without `nsys` or a GPU,
`--profile` still prints a CPU-only `cProfile` summary.

> [!NOTE]
> Requires the nightly `modular` build (see SKILL.md).

This is usually enough to answer "what dominates GPU time." Use the manual
flows below only when you need a custom trace window or you're profiling a
long-running server.

## Profile a standalone script

```bash
MODULAR_ENABLE_PROFILING=detailed \
  nsys profile --trace=cuda,osrt,nvtx \
  --cuda-memory-usage=true \
  --output=profile --force-overwrite=true \
  pixi run python my_program.py
```

Produces `profile.nsys-rep`.

## Profile a serving endpoint

Use `nsys launch` (starts the server but defers collection) + `nsys start` /
`nsys stop` around the benchmark, following
[the durable algorithm](../SKILL.md#the-durable-algorithm) (warm caches first,
keep the window short, fail fast on a dead server).

1. Start the server under `nsys launch`. Check socket count first with
   `lscpu | grep '^Socket(s):'`. On a multi-socket box (more than one), bind
   NUMA (`numactl --cpunodebind=N --membind=N`) for stable numbers; on a
   single-socket node, run `nsys` without it.

   ```bash
   MODULAR_ENABLE_PROFILING=detailed \
     nsys launch --trace=cuda,nvtx,osrt \
     --cuda-memory-usage=true --trace-fork-before-exec=true \
     pixi run max serve --model-path <model>
   ```

2. Wait for `🚀 Server ready on http://0.0.0.0:8000`. (Fail fast: if the
   process dies or the log shows an early error, stop and surface it.)

3. In a second terminal in the same env, start collection:

   ```bash
   nsys start --force-overwrite=true --output=server_profile \
     --session=$(nsys sessions list -p false | awk '{print $1}')
   ```

4. Run a short benchmark at concurrency 1 (long enough to span the window):

   ```bash
   pixi run max benchmark --model <model> --backend modular \
     --endpoint /v1/chat/completions --dataset-name sharegpt \
     --num-prompts 50 --max-concurrency 1 --max-benchmark-duration-s 12
   ```

5. Stop collection:

   ```bash
   nsys stop --session=$(nsys sessions list -p false | awk '{print $1}')
   ```

Yields `server_profile.nsys-rep`.

### Confirm the markers landed

Before trusting the timeline, verify the NVTX phase ranges are actually present
— a profile captured without `MODULAR_ENABLE_PROFILING=detailed` looks valid but
has no phase labels. Check from the terminal:

```bash
nsys stats --report nvtx_pushpop_sum server_profile.nsys-rep
```

A correctly-marked capture lists NVTX ranges with names like `prefill`,
`decode`, and `sampling`. An empty report (or "no NVTX data") means the markers
didn't land — re-run with `MODULAR_ENABLE_PROFILING=detailed` set. In the GUI,
the same markers appear as a labeled phase track above the kernel stream.

## Reading the result

- **Terminal summary** is fastest: `nsys stats profile.nsys-rep` re-prints the
  per-kernel table from any `.nsys-rep`. For a readable, demangled rollup (names
  stripped of C++/Mojo template noise — useful for grouping by kernel family),
  use
  `nsys stats --report cuda_gpu_kern_sum:base --timeunit msec profile.nsys-rep`.
- **GUI**: open in the Nsight Systems app — `nsys-ui profile.nsys-rep`
  locally, or `scp` the file to your laptop and double-click it. The timeline
  shows NVTX phase ranges over the kernel stream, so you can see whether
  prefill or decode dominates and where the gaps are.
- A kernel name like `gemv_split_k_*` dominating decode is expected (decode is
  GEMV-heavy). If a single elementwise/copy kernel dominates, that's a candidate
  for an `ncu` single-kernel deep dive.

## Common failures

- **Profile has no NVTX / no phase labels** → you forgot
  `MODULAR_ENABLE_PROFILING=detailed` (or `--gpu-profiling detailed`).
- **"Process launch is not allowed in this state"** → a stale `nsys` session.
  `nsys sessions list`; cancel/shutdown stale sessions before relaunching.
- **Empty / tiny `.nsys-rep`** → the benchmark finished before the window
  opened, or the window never opened. Start the benchmark just after `nsys
  start`, and make the benchmark longer than the capture.
- **Kernels appear as one opaque graph blob, or per-kernel time is missing** →
  CUDA device-graph capture is on (the default for `max serve`), so the timeline
  shows graph *replays* instead of individual kernels. Serve/generate with
  `--no-device-graph-capture` so `nsys` sees each kernel (this is also why
  `custom-architectures.md` uses that flag).
