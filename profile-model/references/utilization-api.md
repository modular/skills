# GPU utilization snapshot (`max.profiler.gpu`)

Use this to answer *"is the GPU actually working, or is it idle / memory-bound
/ throttling?"* It's the cheapest check: pure Python, no profiler install, and
it works on both NVIDIA (via NVML) and AMD (via ROCm SMI).

> [!NOTE]
> Requires the nightly `modular` package installed via pixi or conda rather
> than a plain pip wheel, which may not expose `max.profiler.gpu`. If the
> import fails with `ModuleNotFoundError: max.profiler.gpu`, you're on the
> wrong build.

## What the API gives you

`max.profiler.gpu` exposes two entry points:

- **`GPUDiagContext`**: a context manager. `get_stats()` returns a one-shot
  snapshot: `{gpu_id: GPUStats}`. GPU ids carry a vendor prefix (`nv0`, `nv1`,
  `amd0`, …).
- **`BackgroundRecorder(interval=1.0)`**: samples in a background process while
  your workload runs. After the `with` block exits, `.stats` is a time-series
  list of those snapshots.

`GPUStats` fields:

- `memory` (`MemoryStats`): `used_bytes`, plus total/free/reserved.
- `utilization` (`UtilizationStats`): `gpu_usage_percent` (0–100 compute
  busy) and `memory_activity_percent` (memory-controller activity, or `None`
  if the vendor doesn't report it).
- `clocks` (`ClockStats | None`): current/max clocks and throttle reasons.

The headline number for "is my GPU busy?" is `utilization.gpu_usage_percent`.
If it's pegged near 100 the GPU is compute-bound; if it's low while a workload
runs, you're bottlenecked elsewhere (CPU, host-device transfer, small batch,
synchronization).

## One-shot snapshot

```python
from max.profiler.gpu import GPUDiagContext

with GPUDiagContext() as ctx:
    for gpu_id, s in ctx.get_stats().items():
        # used_bytes / 1e9 is approximate GB (decimal); use / 2**30 for GiB.
        print(f"{gpu_id}: {s.utilization.gpu_usage_percent}% compute, "
              f"{s.memory.used_bytes / 1e9:.1f} GB used")
```

This prints one line per GPU, for example `nv0: 87% compute, 14.2 GB used`.

## Sampling during a workload

Wrap the workload (for example an `InferenceSession` generate loop, or a
`time.sleep()` that spans a separate benchmark process) in a
`BackgroundRecorder`:

```python
from max.profiler.gpu import BackgroundRecorder

with BackgroundRecorder(interval=0.5) as rec:
    run_my_inference()          # or sleep while `max benchmark` runs elsewhere

# rec.stats is a list of {gpu_id: GPUStats} snapshots, one per interval.
```

## Use the bundled script

`scripts/gpu_snapshot.py` packages the recorder + a summary so you don't have
to rewrite it each time. It records for a fixed duration (default 15s) and
prints, per GPU, the peak / mean compute utilization, peak memory used, and any
throttle reasons seen.

```bash
# Terminal A: start the model serving (warm it up first — see
# ../SKILL.md "The durable algorithm").
pixi run max serve --model-path <model>

# Terminal B: start a short load, and record while it runs.
pixi run python scripts/gpu_snapshot.py --duration 15 &
pixi run max benchmark --model <model> --backend modular \
  --endpoint /v1/chat/completions --dataset-name sharegpt \
  --num-prompts 50 --max-concurrency 1 --max-benchmark-duration-s 12
```

For a standalone (non-serving) workload, point the script at your script
instead and it records for its lifetime:

```bash
pixi run python scripts/gpu_snapshot.py --duration 30 --run "pixi run python my_program.py"
```

## Reading the result

- **High `gpu_usage_percent` (80–100%)** during steady state: the GPU is the
  bottleneck. Move on to a kernel breakdown to see *which kernels* dominate.
- **Low `gpu_usage_percent` with bursts**: likely launch/sync overhead, tiny
  batch, or host-side work between kernels. A kernel-breakdown timeline shows
  the gaps.
- **`memory.used_bytes` near total**: you're near OOM; KV-cache / batch size
  may be the real constraint.
- **Throttle reasons present** (`hw_thermal_slowdown`, `sw_power_cap`, …): the
  GPU is clock-limited, not work-limited — numbers won't be representative.
