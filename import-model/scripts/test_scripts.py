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
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

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


def _load_scaffold() -> Any:
    """Import ``scaffold.py`` as a module for direct unit tests."""
    spec = importlib.util.spec_from_file_location(
        "_scaffold", _SCRIPT_DIR / "scaffold.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_scaffold"] = module
    spec.loader.exec_module(module)
    return module


def test_planner_resolution() -> None:
    """Cover every way a donor arch.py can declare ``memory_planner``.

    The Call form matters most: eight architectures configure their planner
    via ``with_activation_reservation(...)``, and a resolver that only
    understood bare names silently dropped the reservation.
    """
    scaffold = _load_scaffold()
    archs = (
        _SCRIPT_DIR / "../../../../max/python/max/pipelines/architectures"
    ).resolve()
    if not archs.is_dir():
        print("SKIP planner resolution (architectures tree not present)")
        return

    cases = [
        # donor, status, name to import, substring expected in the value
        ("llama3", "resolved", "PagedMemoryPlanner", "PagedMemoryPlanner"),
        # Call form: the reservation and its arguments must survive.
        ("gemma3", "resolved", "PagedMemoryPlanner", "with_activation"),
        ("qwen2_5vl", "resolved", "PagedMemoryPlanner", "5 * 1024**3"),
        # Per-architecture planner: import comes from the donor's own module.
        ("deepseekV3", "resolved", "DeepseekV3MemoryPlanner", "DeepseekV3"),
        # No planner at all: correct for diffusion, must not be invented.
        ("flux2", "absent", "", ""),
    ]
    for donor, status, root_name, expected in cases:
        if not (archs / donor / "arch.py").is_file():
            print(f"SKIP planner resolution for {donor} (donor absent)")
            continue
        got = scaffold.find_donor_memory_planner(archs / donor, donor)
        if got.status != status:
            raise SystemExit(
                f"FAIL planner resolution {donor}: expected status "
                f"{status!r}, got {got.status!r}"
            )
        if got.root_name != root_name:
            raise SystemExit(
                f"FAIL planner resolution {donor}: expected import "
                f"{root_name!r}, got {got.root_name!r}"
            )
        if expected and expected not in got.expr:
            raise SystemExit(
                f"FAIL planner resolution {donor}: expected {expected!r} in "
                f"rendered value {got.expr!r}"
            )
        print(f"PASS planner resolution {donor} ({status})")

    # An absent planner must leave the field out, not default it.
    absent = scaffold.find_donor_memory_planner(archs / "flux2", "flux2")
    rendered = scaffold.render_arch(
        slug="p",
        arch_name="X",
        short="X",
        hf_id="org/m",
        donor_planner=absent,
    )
    # The TODO text names the keyword, so only count real code lines.
    passed = [
        line
        for line in rendered.splitlines()
        if "memory_planner=" in line and not line.lstrip().startswith("#")
    ]
    if passed:
        raise SystemExit(
            "FAIL planner resolution: scaffolded a memory_planner for a donor "
            f"that declares none: {passed}"
        )
    print("PASS planner resolution omits memory_planner when donor has none")


def test_generated_arch_lints() -> None:
    """The generated arch.py must satisfy ruff's import sorting (I001).

    The planner import can come from anywhere in the tree, so it has to be
    merged into a sorted block rather than spliced at a fixed line.
    """
    if shutil.which("ruff") is None:
        print("SKIP generated arch.py lint (ruff not on PATH)")
        return
    scaffold = _load_scaffold()
    archs = (
        _SCRIPT_DIR / "../../../../max/python/max/pipelines/architectures"
    ).resolve()
    if not archs.is_dir():
        print("SKIP generated arch.py lint (architectures tree not present)")
        return

    # deepseekV3's planner sorts before other max.pipelines imports, so it is
    # the case a fixed-position splice gets wrong.
    for donor in ("llama3", "deepseekV3", "gemma3", "flux2"):
        if not (archs / donor / "arch.py").is_file():
            continue
        planner = scaffold.find_donor_memory_planner(archs / donor, donor)
        rendered = scaffold.render_arch(
            slug="p",
            arch_name="X",
            short="X",
            hf_id="org/m",
            donor_planner=planner,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arch.py"
            path.write_text(rendered, encoding="utf-8")
            # E501 is in the repo's ruff ignore list, so check I and F only.
            code, out = run(
                [
                    "ruff",
                    "check",
                    "--isolated",
                    "--select",
                    "I,F",
                    "--line-length",
                    "80",
                    str(path),
                ]
            )
            if code != 0:
                raise SystemExit(
                    f"FAIL generated arch.py lint (donor {donor}):\n{out}"
                )
        print(f"PASS generated arch.py lint ({donor})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-id", default=_HF_DEFAULT)
    args = ap.parse_args()
    py = sys.executable
    hf = args.hf_id

    test_planner_resolution()
    test_generated_arch_lints()

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
        arch_py = (port_dir / "arch.py").read_text(encoding="utf-8")
        if "memory_planner=PagedMemoryPlanner" not in arch_py:
            raise SystemExit(
                "FAIL scaffold: arch.py is missing "
                "memory_planner=PagedMemoryPlanner (donor: llama3)"
            )
        print("PASS scaffold arch.py memory_planner")

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
