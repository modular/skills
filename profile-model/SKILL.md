---
name: profile-model
description: >
  Profile a model running on MAX to find where it spends time and whether the
  GPU is saturated. Use when the user asks to "profile my model," "where is my
  model spending time," "why is inference slow," "is my GPU being utilized,"
  "how much GPU am I using," "get a kernel breakdown," "capture an
  nsys/rocprof/ncu trace of max serve," or wants to measure MAX inference
  performance. Works for any model MAX can run — built-in architectures and
  custom ones loaded with --custom-architectures — from a pip or pixi install
  (max generate, max serve, or a Python script) on NVIDIA or AMD GPUs. Decide
  cheapest-first: a GPU utilization check, then a kernel breakdown, then a
  single-kernel deep dive only when one kernel dominates.
compatibility: Requires a pip/pixi MAX install and a GPU (NVIDIA or AMD). The kernel-breakdown and deep-dive steps also need the vendor profiler (nsys/ncu for NVIDIA, rocprofv3 for AMD).
argument-hint: "[model-path] [serve/generate flags]"
---

# Profile a model on MAX

This skill answers three questions about any model you can run on MAX, whether
it's a built-in architecture or a custom one loaded with
`--custom-architectures`:

1. **Is my GPU actually being used?** (utilization, memory, clocks)
2. **Where does my model spend the most time?** (which kernels / phases)
3. **Why is *this* kernel slow?** (occupancy, stalls, roofline)

It works from a `pip` or `pixi` install of MAX, driving the public `max` CLI or
a Python script.

## How to decide what to run

Each question needs a deeper, more expensive capture than the last. Work from
the cheapest check toward the most invasive, and let each result decide whether
going deeper is even worth it — don't capture more than the question needs.

1. **Start with the utilization check**. If the GPU is idle or lightly loaded
   while the workload runs, the bottleneck is host-side (CPU, transfers, small
   batch) — stop and report that. A kernel trace won't tell you anything a
   busy GPU wouldn't.
2. **If the GPU is busy, capture a kernel breakdown**. This is the common case
   and usually the final answer: it shows which kernels dominate GPU time.
3. **Only if one kernel dominates and you need to know *why*, do a single-kernel
   deep dive** on that one kernel. Skip this unless a breakdown has already
   pointed at a specific kernel — it replays the kernel many times and is slow.

| What you're answering    | Tool                                                        | Cost                        |
|--------------------------|-------------------------------------------------------------|-----------------------------|
| Is the GPU busy?         | `max.profiler.gpu` (pure Python, NVIDIA + AMD)              | seconds, no extra installs  |
| Where does time go?      | `max ... --profile`, or `nsys` (NVIDIA) / `rocprofv3` (AMD) | a couple of profiled runs   |
| Why is one kernel slow?  | Nsight Compute (`ncu`), NVIDIA only                         | one slow capture per kernel |

Read the reference for the step you're on rather than loading all of them:

- **Utilization check** →
  [`references/utilization-api.md`](references/utilization-api.md) (and the
  bundled `scripts/gpu_snapshot.py`)
- **Kernel breakdown, NVIDIA** → [`references/nsys.md`](references/nsys.md)
- **Kernel breakdown, AMD** →
  [`references/rocprofv3.md`](references/rocprofv3.md)
- **Single-kernel deep dive (NVIDIA)** →
  [`references/ncu.md`](references/ncu.md)

### Models loaded with `--custom-architectures`

Profiling itself works the same for a custom architecture as for a built-in
one. What differs is loading: a custom architecture has extra prerequisites and
failure modes — import compatibility with your installed MAX version,
`PYTHONPATH` traps, architecture-name collisions, base models with no chat
template, and a `max serve` limitation for custom embeddings models. Those stop
the model from loading *before* any profiling can run. If you're profiling a
model passed via `--custom-architectures`, read
[`references/custom-architectures.md`](references/custom-architectures.md)
**first**.

## Reading the result (what the kernel mix tells you)

Profiling answers two things: *is the GPU saturated* (the utilization check) and
*where does time go* (the kernel breakdown). Map what you see back to a
diagnosis:

- **Decode dominated by `gemv_split_k_*` GEMVs, with `*_mha_*` attention and
  `rms_norm` below** — this is the normal, healthy shape for a *dense* LLM token
  generation (decode is memory-bound GEMV). Nothing to chase.
- **Sparse-MoE decode dominated by grouped / block-scaled matmul plus routing
  kernels** — the per-expert grouped matmul (`*grouped*` / `block_scaled_*`
  matmul) is the dominant kernel instead of a dense GEMM, alongside top-k
  routing (`topk_*`, `moe_create_indices`) and expert gather/scatter; on
  multi-GPU, EP `dispatch`/`combine` collectives appear too. This is the healthy
  shape for a Mixture-of-Experts model — routing and gather overhead is
  expected, and only worth chasing if it rivals the matmul itself.
- **Low-bit weights (FP8 / FP4 / NVFP4) add quant-prep kernels** — alongside the
  matmul you'll see dynamic activation-quantization and scale-layout kernels
  (`*quantize*`, `*block_scales*`) and block-scaled matmul variants rather than
  plain `gemv`/`gemm`. Time spent in quant/dequant prep is normal for a low-bit
  model; flag it only if it dwarfs the matmul it feeds. Kernel *prefixes* are
  GPU-arch-specific (`sm100_*` on Blackwell, different on Hopper / MI) — match
  on the kernel *family*, not the exact name.
- **A custom architecture *not* hitting the expected fused kernel** (for example
  attention showing as generic `elementwise`/`matmul` instead of a `*_mha_*`
  kernel, or norms/RoPE unfused) — a wiring signal: the graph may not be built
  the way you think, even if logits pass. Worth a look during bring-up.
- **Pathologically low decode tok/s + an outsized prefill-shaped kernel mix
  repeating every step** — classic O(n²) re-prefill (no/broken KV cache). This
  is a correctness-adjacent bug, not just slowness.
- **Low `gpu_usage_percent` while a workload runs** — host-bound, launch/sync
  overhead, or batch too small; the GPU isn't the bottleneck.
- **Throttle reasons present** — the GPU is clock-limited; numbers aren't
  representative until you address thermal/power.

## The durable algorithm

Tooling and MAX packaging change often; this shape is the part worth
preserving. Whichever capture you run, follow it:

1. **Detect the environment before committing to a tool**. Run `nvidia-smi` or
   `rocm-smi` to learn the vendor — the kernel-breakdown and deep-dive tools
   differ by vendor. Confirm the profiler is installed (`which nsys` /
   `which rocprofv3` / `which ncu`); if it's missing, tell the user the exact
   install line (see the reference) rather than failing midway.
2. **Confirm before any long-lived run**. Profiling `max serve` warms a
   compile cache (cold compile can take minutes), launches a server, and runs a
   benchmark. Before doing that, show the user the model, the flags, and the
   planned commands, and wait for confirmation. A one-shot
   `max generate --profile` on a tiny model is cheap enough to skip this.
3. **Warm caches first, profile second**. The first run of a model pays
   one-time compile and weight-load costs that drown out the real kernel time.
   Do an unprofiled warm-up run, *then* the profiled run, so the capture
   reflects steady-state inference — not compilation.
4. **Keep the capture window small**. A 10-second window at concurrency 1 is
   enough to see the kernel mix. Long captures produce huge trace files that
   are slow to open and no more informative.
5. **Fail fast**. After launching a server, check within a few seconds that the
   process is alive and the log has no early error (model not found, OOM, bad
   flag) before you start polling for `/health`. Don't wait out a full timeout
   on a server that already died.
6. **Verify the artifact, then report**. Confirm the output file exists and is
   non-trivial in size before claiming success. Report the artifact path, how
   to open it, and the headline numbers (top kernels, or peak GPU utilization).
7. **Clean up only what you started**. Kill the server / benchmark process
   group you launched. Avoid broad `pkill -f max` on a shared box — you may
   stop someone else's run.

## Nightly vs stable

Profiling features land in nightly before stable, so this skill targets the
nightly `modular` build. Install it with pixi:

```bash
pixi init quickstart -c https://conda.modular.com/max-nightly/ -c conda-forge
cd quickstart && pixi add modular
pixi run max --version          # expect a *.dev build
```

The utilization API (`GPUDiagContext`, `BackgroundRecorder`) ships in the
**conda** `modular` package, so install via pixi or conda rather than a plain
`pip install modular` wheel, which may not expose `max.profiler.gpu`. If a
command below 404s or an import fails, confirm you're on a recent nightly
build.

## Install notes

- MAX itself: a project with the nightly `modular` package installed via pixi
  (see [Nightly vs stable](#nightly-vs-stable) above). All
  `max` CLI commands below assume you can run `pixi run max ...` in that
  project.
- `nsys` and `ncu` ship with the CUDA Toolkit. If `which nsys` fails but CUDA is
  present, `export PATH=/usr/local/cuda/bin:$PATH`. Otherwise install Nsight
  Systems / Nsight Compute from NVIDIA, or
  `sudo apt-get install nsight-systems`.
- `rocprofv3` ships with ROCm (`/opt/rocm/bin`). Add it to PATH if needed.
- The utilization API needs no extra tooling beyond `modular` — it talks to
  NVML / ROCm SMI directly through `max.profiler.gpu`.

## Fast paths

The lowest-friction commands, each expanded in its reference file:

- **"Is my GPU being used?"** → run `scripts/gpu_snapshot.py` alongside a short
  benchmark. Pure Python, works on NVIDIA and AMD.
- **"Give me a kernel breakdown, fast."** → `pixi run max generate --model <m>
  --prompt "hello" --num-warmups 1 --profile`. Prints a ranked top-N GPU-kernel
  table
  and writes a `.nsys-rep` (NVIDIA). Falls back to a CPU summary without a GPU.
- **"Profile my serving benchmark."** → the serving-endpoint flow with
  `MODULAR_ENABLE_PROFILING=detailed` + `nsys launch` / `rocprofv3`.
