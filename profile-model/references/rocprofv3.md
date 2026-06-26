# Kernel breakdown on AMD with `rocprofv3`

Use this to answer *"where does my model spend time?"* on AMD GPUs (MI300,
MI355, and similar). `rocprofv3` captures a kernel trace plus marker
(NVTX-equivalent) trace and writes a Perfetto `.pftrace` you open at
<https://ui.perfetto.dev>.

Confirm the tool first: `command -v rocprofv3` and `rocprofv3 --version`. It
ships with ROCm under `/opt/rocm/bin` — add that to PATH if missing. Enable MAX
markers with `MODULAR_ENABLE_PROFILING=detailed` (so the marker trace labels
the prefill/decode phases).

## The three-pass capture

`rocprofv3 --collection-period` takes `delay:duration:rate` — the seconds to
wait before recording, the seconds to record, and the sampling multiplier. Open
the capture window *after* the server is warm and serving, so you need the warm
startup time first. Hence three passes:

`R2` below is the warm startup time: the seconds from launching `max serve`
until `/health` returns 200. The measure pass exists only to find it.

1. **Warm pass** — start `max serve`, run a short benchmark, tear down. This
   populates the compile cache so the next startup is fast.
2. **Measure pass** — start `max serve` again (no profiler), time how long until
   `/health` returns 200. That time is `R2`. Run a short benchmark, tear down.
3. **Profile pass** — start `max serve` under `rocprofv3` with the window
   opening at about `R2 + 10` seconds for a 10-second capture. The profiled
   server and the benchmark run as **two concurrent processes**: start the
   server (it blocks the shell), then drive the benchmark from a second shell.

   In the first shell, start the profiled server (`$((R2 + 10)):10:1` is the
   `delay:duration:rate` window — wait `R2 + 10` s, record for 10 s):

   ```bash
   MODULAR_ENABLE_PROFILING=detailed \
     rocprofv3 --kernel-trace --marker-trace \
     --collection-period $((R2 + 10)):10:1 \
     --output-format pftrace -d ./rocprof_out -- \
     pixi run max serve --model-path <model>
   ```

   In a **second shell**, start the benchmark ~2s before the window opens and
   let it run past the end of the 10s window:

   ```bash
   sleep $((R2 + 8))
   pixi run max benchmark --model <model> --backend modular \
     --endpoint /v1/chat/completions --dataset-name sharegpt \
     --num-prompts 100 --max-concurrency 1 --max-benchmark-duration-s 20
   ```

## Core rules

- Launch `max serve` **directly** — do not wrap it in `rocprofv3 --attach`, and
  do not use any `--run_under` indirection. Attach/run-under produce tiny or
  empty `.pftrace` files here.
- Keep benchmark concurrency at 1 — a clean single-stream trace is far easier
  to read than 600 concurrent requests.
- Run the three passes in order; skipping the warm pass means the capture
  window lands during compilation, not inference.
- Clean up only the process group you started.

## Verify and report

After the run, find the trace and check it's real before claiming success:

```bash
find ./rocprof_out -name "*.pftrace" -size +1k -exec ls -lh {} \;
```

A valid Perfetto file is non-trivial in size and begins with the Perfetto
magic header. Report every `.pftrace` path and size, and tell the user to open
it at <https://ui.perfetto.dev> (drag-and-drop) — the marker track shows the
phase labels over the kernel timeline.

## Common failures

- **Tiny / missing `.pftrace`** → you used `--attach` or `--run_under`, or the
  collection window opened before the server was serving. Re-run the three-pass
  flow with `max serve` launched directly and recompute `R2`.
- **No marker/phase track** → you didn't set
  `MODULAR_ENABLE_PROFILING=detailed`, or you omitted `--marker-trace`.
