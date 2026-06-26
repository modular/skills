#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Record GPU utilization/memory while a workload runs, then print a summary.

The GPU utilization check of the profile-model skill. Uses max.profiler.gpu
(NVML on NVIDIA, ROCm SMI on AMD) so it needs no profiler install beyond the
`modular` package.

Two modes:
  * --run "<cmd>"   : launch the command, record for its lifetime, summarize.
  * (no --run)      : record for --duration seconds while you drive load from
                      another terminal (e.g. `max benchmark`).

Examples:
  python gpu_snapshot.py --duration 15
  python gpu_snapshot.py --run "pixi run python my_program.py"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _summarize(samples: list[dict]) -> None:
    """Print peak/mean utilization and peak memory per GPU."""
    if not samples:
        print("No samples collected — was the workload too short?")
        return

    # Collect per-GPU time series.
    per_gpu: dict[str, dict] = {}
    for snapshot in samples:
        for gpu_id, s in snapshot.items():
            g = per_gpu.setdefault(
                gpu_id, {"util": [], "mem": [], "throttles": set()}
            )
            g["util"].append(s.utilization.gpu_usage_percent)
            g["mem"].append(s.memory.used_bytes)
            clocks = getattr(s, "clocks", None)
            for reason in getattr(clocks, "throttle_reasons", None) or []:
                # gpu_idle is normal between samples; don't alarm on it.
                if reason != "gpu_idle":
                    g["throttles"].add(reason)

    print(f"\n=== GPU utilization over {len(samples)} samples ===")
    print(f"{'gpu':<6}{'peak%':>7}{'mean%':>7}{'peak mem (GB)':>15}  throttles")
    for gpu_id, g in sorted(per_gpu.items()):
        util = g["util"]
        peak = max(util)
        mean = sum(util) / len(util)
        peak_mem = max(g["mem"]) / 1e9
        throttles = ", ".join(sorted(g["throttles"])) or "-"
        print(f"{gpu_id:<6}{peak:>7}{mean:>7.0f}{peak_mem:>15.1f}  {throttles}")

    # One-line verdict to orient the reader.
    best = max(per_gpu.values(), key=lambda g: max(g["util"]))
    peak = max(best["util"])
    if peak >= 80:
        print(
            "\nVerdict: GPU is busy at peak — likely compute-bound. "
            "Capture a kernel breakdown (nsys/rocprofv3) to see which "
            "kernels dominate."
        )
    elif peak >= 30:
        print(
            "\nVerdict: moderate GPU use — gaps suggest launch/sync overhead "
            "or small batch. A kernel-breakdown timeline will show the idle "
            "gaps."
        )
    else:
        print(
            "\nVerdict: GPU mostly idle during the window — the bottleneck is "
            "probably host-side, transfer-bound, or the workload hadn't "
            "ramped up. Check that load was actually running."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Seconds to record (ignored when --run finishes "
        "sooner). Default 15.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval in seconds. Default 0.5.",
    )
    ap.add_argument(
        "--run",
        default=None,
        help="Optional command to launch; recording stops when it "
        "exits or --duration elapses, whichever is first.",
    )
    args = ap.parse_args()

    try:
        from max.profiler.gpu import BackgroundRecorder
    except ImportError:
        print(
            "ERROR: could not import max.profiler.gpu. Install the `modular` "
            "package (pixi add modular / pip install modular) and run inside "
            "that environment.",
            file=sys.stderr,
        )
        return 1

    proc = None
    if args.run:
        proc = subprocess.Popen(args.run, shell=True)

    with BackgroundRecorder(interval=args.interval) as rec:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                break
            time.sleep(
                min(args.interval, max(0.0, deadline - time.monotonic()))
            )

    if proc is not None and proc.poll() is None:
        proc.terminate()

    _summarize(rec.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
