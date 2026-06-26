# Single-kernel deep dive with Nsight Compute (`ncu`)

Use this only **after** a kernel breakdown has told you *which* kernel is slow.
`ncu` answers *why*: compute vs memory throughput, occupancy, warp-stall
reasons, cache hit rates, roofline position. It's NVIDIA-only and expensive —
each profiled kernel launch is replayed many times, so you target one kernel,
not a whole run.

Confirm `ncu` is on PATH (`which ncu`; it ships with the CUDA Toolkit, often at
`/opt/nvidia/nsight-compute/<ver>/ncu` or `/usr/local/cuda/bin/ncu`). On a
shared box, lock clocks for reproducible numbers if you have permission, and
pick an idle GPU (`nvidia-smi`).

To profile a kernel, point `ncu` at a short `max generate` run and filter to the
kernel of interest by name.

## Target one kernel by name

Take the kernel name (or a distinctive substring) from the top-N table the
kernel breakdown produced ([nsys.md](nsys.md) on NVIDIA), and use
`--kernel-name-base` / `-k` to filter, with `-c` to cap how many launches are
profiled (keep it small — each is slow):

```bash
ncu --target-processes all \
  -k regex:"gemv_split_k" \
  -c 3 \
  --set full \
  -o gemv_profile \
  pixi run max generate \
    --model <model> --prompt "hello" --num-warmups 1 --max-new-tokens 8
```

- `-k regex:"…"`: only profile kernels whose name matches. Without this, `ncu`
  tries to profile *every* kernel and the run takes forever.
- `-c 3`: stop after 3 matching launches; one steady-state launch is usually
  enough.
- `--set full`: collect the full metric set (throughput, occupancy, stalls,
  memory). Use `--set basic` for a faster, lighter pass.
- `-o gemv_profile`: writes `gemv_profile.ncu-rep`.

A short generate (small `--max-new-tokens`) is enough to hit decode kernels; you
don't need a full server.

## Reading the result

The names in quotes and backticks below are the literal section and metric
labels `ncu` reports, so you can match them directly in the output.

- **Terminal**: `ncu --import gemv_profile.ncu-rep --page details` (or
  `--page raw`) prints the metrics. The `GPU Speed Of Light` section tells you
  immediately whether the kernel is compute-bound or memory-bound.
- **GUI**: open `gemv_profile.ncu-rep` in the Nsight Compute app (copy to your
  laptop if you profiled remotely). The roofline and warp-stall sampling views
  are the most actionable.
- Typical reads: low `Achieved Occupancy` → launch config / register or
  shared-memory pressure; high `Memory Throughput` near peak → memory-bound
  (the expected state for decode GEMV); the dominant stall reason points at the
  specific bottleneck (`Stall Long Scoreboard` = memory latency, and so on).

## Common failures

- **Run never finishes** → you forgot `-k` and `ncu` is profiling every kernel,
  replaying each one. Filter by name and add `-c`.
- **"No kernels were profiled"** → the name regex didn't match. Re-check the
  exact substring from the kernel-breakdown table; names are long and templated.
- **Permission / counter errors** → NCU needs GPU performance-counter access;
  on locked-down nodes run with elevated permissions or ask the admin to enable
  it.
