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
"""Smoke-test skill scripts (no GPU, minimal Hub access).

Usage::

    pixi run python test_scripts.py
    pixi run python test_scripts.py --hf-id Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_HF_DEFAULT = "Qwen/Qwen3-1.7B"


def run(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd or _SCRIPT_DIR,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def check(name: str, code: int, out: str, expect: str = "") -> None:
    if code != 0:
        raise SystemExit(f"FAIL {name} (exit {code}):\n{out[-2000:]}")
    if expect and expect not in out:
        raise SystemExit(
            f"FAIL {name}: expected {expect!r} in output:\n{out[-2000:]}"
        )
    print(f"PASS {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-id", default=_HF_DEFAULT)
    args = ap.parse_args()
    py = sys.executable
    hf = args.hf_id

    code, out = run([py, "list_native_archs.py", "--match", "Qwen3ForCausalLM"])
    check("list_native_archs --match", code, out, "Qwen3ForCausalLM")

    code, out = run([py, "check_walls.py", hf])
    check("check_walls", code, out)

    code, out = run([py, "list_checkpoint_keys.py", hf, "--summary"])
    check("list_checkpoint_keys", code, out, "dominant_dtype")

    code, out = run([py, "inspect_hf.py", hf])
    check("inspect_hf", code, out, "HF inspection")

    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(tmp) / "ports"
        code, out = run(
            [
                py,
                "scaffold.py",
                hf,
                "--start-from",
                "llama3",
                "--output-dir",
                str(out_root),
                "--slug",
                "test_qwen3_port",
            ]
        )
        check("scaffold", code, out, "Scaffold created")
        port_dir = out_root / "test_qwen3_port"
        if not (port_dir / "arch.py").is_file():
            raise SystemExit(f"FAIL scaffold: missing {port_dir / 'arch.py'}")

        code, out = run(
            [
                py,
                "run_oss_gates.py",
                hf,
                "--port-dir",
                str(port_dir),
            ]
        )
        check("run_oss_gates preflight", code, out)

    code, out = run([py, "compare_layers.py", "--help"])
    check("compare_layers --help", code, out, "usage")

    print("\nAll script smoke tests passed.")


if __name__ == "__main__":
    main()
