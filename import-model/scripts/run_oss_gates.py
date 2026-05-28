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
"""Preflight gates for the port workflow (walls, checkpoint metadata, arch registration).

Does **not** replace import/graph/adapter smoke in serve-and-iterate.md.

Phases:

- ``preflight`` (default): wall scan + ``arch.py`` name/encoding vs Hub config.
- ``verify`` (requires ``--port``): multi-position logit probe via running
  ``pixi run max serve``.

Usage::

    pixi run python run_oss_gates.py <HF_ID> --port-dir <port_dir>/
    pixi run python run_oss_gates.py <HF_ID> --port-dir <port_dir>/ \\
        --phase verify --slug my_slug --port 8000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .check_walls import scan_config
    from .checkpoint_metadata import fetch_repo_tensors
    from .dtype_utils import canonical_native_dtype, encoding_from_config_dict
    from .hub_config import architecture_class, load_hub_config
except ImportError:
    # Standalone invocation: `python /path/to/run_oss_gates.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_walls import scan_config  # type: ignore[no-redef]
    from checkpoint_metadata import fetch_repo_tensors  # type: ignore[no-redef]
    from dtype_utils import (  # type: ignore[no-redef]
        canonical_native_dtype,
        encoding_from_config_dict,
    )
    from hub_config import (  # type: ignore[no-redef]
        architecture_class,
        load_hub_config,
    )

_ARCH_NAME_RE = re.compile(r'name\s*=\s*["\']([^"\']+)["\']')
_DEFAULT_ENC_RE = re.compile(r'default_encoding\s*=\s*["\']([^"\']+)["\']')


@dataclass
class GateResult:
    gate: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str


def _parse_arch_py(port_dir: Path) -> tuple[str | None, str | None]:
    arch = port_dir / "arch.py"
    if not arch.is_file():
        return None, None
    text = arch.read_text()
    name_m = _ARCH_NAME_RE.search(text)
    enc_m = _DEFAULT_ENC_RE.search(text)
    return (
        name_m.group(1) if name_m else None,
        enc_m.group(1) if enc_m else None,
    )


def gate_checkpoint_metadata(hf_id: str, port_dir: Path) -> GateResult:
    """Verify the Hub repo exposes safetensors metadata and dtype matches arch."""
    try:
        summary = fetch_repo_tensors(hf_id)
    except ValueError as exc:
        return GateResult("checkpoint_meta", "FAIL", str(exc))

    dominant = summary.dominant_dtype()
    _, enc = _parse_arch_py(port_dir)
    if enc and dominant and enc != dominant:
        return GateResult(
            "checkpoint_meta",
            "WARN",
            f"dominant checkpoint dtype {dominant!r} != arch default_encoding {enc!r}",
        )
    detail = f"{len(summary.tensors)} tensors, dominant={dominant}"
    if summary.sharded:
        detail += ", sharded"
    return GateResult("checkpoint_meta", "PASS", detail)


def gate_walls(cfg: dict) -> GateResult:
    findings = scan_config(cfg)
    blocks = [f for f in findings if f.level == "block"]
    if blocks:
        return GateResult("walls", "FAIL", blocks[0].message)
    warns = [f for f in findings if f.level == "warn"]
    if warns:
        return GateResult("walls", "WARN", warns[0].message)
    return GateResult("walls", "PASS", "no wall signals")


def gate_arch_name(cfg: dict, port_dir: Path) -> GateResult:
    expected = architecture_class(cfg)
    found, _ = _parse_arch_py(port_dir)
    if found is None:
        return GateResult(
            "arch_name", "FAIL", f"missing arch.py under {port_dir}"
        )
    if found != expected:
        return GateResult(
            "arch_name",
            "FAIL",
            f"arch.py name={found!r} != config architectures[0]={expected!r}",
        )
    return GateResult("arch_name", "PASS", found)


def gate_encoding(cfg: dict, port_dir: Path) -> GateResult:
    hub_raw = encoding_from_config_dict(cfg)
    expected = canonical_native_dtype(hub_raw) if hub_raw else "bfloat16"
    _, found = _parse_arch_py(port_dir)
    if found is None:
        return GateResult("encoding", "SKIP", "no arch.py")
    if found != expected:
        return GateResult(
            "encoding",
            "WARN",
            f"default_encoding={found!r} != Hub-native {expected!r}",
        )
    return GateResult("encoding", "PASS", found)


def gate_verify_logprobs(
    hf_id: str,
    slug: str,
    port: int,
    dtype: str,
) -> GateResult:
    try:
        from .compare_layers import probe_multi_position
    except ImportError:
        from compare_layers import probe_multi_position

    diverged = probe_multi_position(
        hf_id,
        slug,
        port,
        dtype=dtype,
        quiet=True,
    )
    if diverged:
        pos, _hf, _mx = diverged[0]
        return GateResult(
            "logits_multi",
            "FAIL",
            f"first top-1 mismatch at prefix length {pos}",
        )
    return GateResult("logits_multi", "PASS", "all probed positions match")


def run_preflight(hf_id: str, port_dir: Path) -> list[GateResult]:
    cfg = load_hub_config(hf_id)
    return [
        gate_walls(cfg),
        gate_checkpoint_metadata(hf_id, port_dir),
        gate_arch_name(cfg, port_dir),
        gate_encoding(cfg, port_dir),
    ]


def run_verify(
    hf_id: str, port_dir: Path, slug: str, port: int, dtype: str
) -> list[GateResult]:
    results = run_preflight(hf_id, port_dir)
    if any(r.status == "FAIL" for r in results):
        results.append(GateResult("logits_multi", "SKIP", "preflight failed"))
        return results
    results.append(gate_verify_logprobs(hf_id, slug, port, dtype))
    return results


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id")
    parser.add_argument(
        "--port-dir",
        type=Path,
        required=True,
        help="Slug directory containing arch.py (same path as --custom-architectures)",
    )
    parser.add_argument(
        "--phase", choices=("preflight", "verify"), default="preflight"
    )
    parser.add_argument(
        "--slug", help="MAX model slug (verify phase; default: port-dir name)"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float32"]
    )
    parser.add_argument("--json-out", type=Path, help="Write results JSON")


def main(args: argparse.Namespace) -> int:
    port_dir = args.port_dir.resolve()
    if not port_dir.is_dir():
        sys.exit(f"port-dir not found: {port_dir}")

    slug = args.slug or port_dir.name
    if args.phase == "preflight":
        results = run_preflight(args.hf_id, port_dir)
    else:
        results = run_verify(args.hf_id, port_dir, slug, args.port, args.dtype)

    first_fail = next((r.gate for r in results if r.status == "FAIL"), None)
    overall = "FAIL" if first_fail else "PASS"

    payload = {
        "hf_id": args.hf_id,
        "port_dir": str(port_dir),
        "phase": args.phase,
        "overall": overall,
        "first_failing_gate": first_fail,
        "gates": [r.__dict__ for r in results],
    }

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

    for r in results:
        print(f"{r.gate:16} {r.status:4}  {r.detail}")
    print(f"\noverall: {overall}")

    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
