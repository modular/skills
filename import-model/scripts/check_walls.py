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
"""Scan a Hub ``config.json`` for architecture patterns that block OSS ports.

Implements the signals from ``references/recognize-walls.md``. Exit 0 when no
blockers are found, 1 when warnings only, 2 when at least one hard blocker.

Usage::

    pixi run python check_walls.py <HF_MODEL_ID>
    pixi run python check_walls.py <HF_MODEL_ID> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .dtype_utils import encoding_from_config_dict
    from .hub_config import load_hub_config
except ImportError:
    # Standalone invocation: `python /path/to/check_walls.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dtype_utils import encoding_from_config_dict  # type: ignore[no-redef]
    from hub_config import load_hub_config  # type: ignore[no-redef]


@dataclass
class Finding:
    level: str  # "block" | "warn"
    code: str
    message: str


def _nested(cfg: dict, *keys: str) -> object:
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def scan_config(cfg: dict) -> list[Finding]:
    """Return wall findings for a parsed Hub config."""
    out: list[Finding] = []

    pos = str(cfg.get("position_embedding_type") or "").lower()
    if pos == "alibi":
        out.append(
            Finding(
                "block",
                "alibi",
                "position_embedding_type=alibi — no first-class ALiBi path in MAX 26.x",
            )
        )

    ssm_keys = (
        "state_size",
        "conv_kernel",
        "time_step_rank",
        "d_state",
        "expand",
        "ssm_state_size",
    )
    hit_ssm = [k for k in ssm_keys if cfg.get(k) is not None]
    model_type = str(cfg.get("model_type") or "").lower()
    if hit_ssm or model_type in ("mamba", "mamba2", "rwkv", "jamba", "bamba"):
        out.append(
            Finding(
                "warn",
                "ssm_or_recurrence",
                f"SSM/recurrence signals ({', '.join(hit_ssm) or model_type}) — "
                "verify MAX has a native arch before porting",
            )
        )

    qcfg = cfg.get("quantization_config")
    hub_dtype = encoding_from_config_dict(cfg)
    if isinstance(qcfg, dict) and hub_dtype is None:
        out.append(
            Finding(
                "warn",
                "quant_only",
                "quantization_config present and no torch_dtype/dtype — "
                "weights may be FP8/FP4-only",
            )
        )

    if cfg.get("architectures") and not isinstance(
        cfg.get("architectures"), list
    ):
        out.append(
            Finding(
                "warn",
                "architectures_shape",
                "architectures is not a list — verify config.json manually",
            )
        )

    num_params = cfg.get("num_parameters") or cfg.get("total_params")
    if isinstance(num_params, (int, float)) and num_params > 30_000_000_000:
        out.append(
            Finding(
                "warn",
                "very_large",
                f"~{num_params / 1e9:.0f}B params — local CPU bring-up unlikely; plan GPU tier",
            )
        )

    return out


def _exit_code(findings: list[Finding]) -> int:
    if any(f.level == "block" for f in findings):
        return 2
    if findings:
        return 1
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id", help="HuggingFace model ID")
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON findings"
    )


def main(args: argparse.Namespace) -> int:
    cfg = load_hub_config(args.hf_id)
    findings = scan_config(cfg)
    code = _exit_code(findings)

    if args.json:
        print(
            json.dumps(
                {
                    "hf_id": args.hf_id,
                    "exit_code": code,
                    "findings": [f.__dict__ for f in findings],
                },
                indent=2,
            )
        )
    else:
        if not findings:
            print(f"OK: no wall signals in config for {args.hf_id!r}")
        for f in findings:
            tag = "BLOCK" if f.level == "block" else "WARN"
            print(f"{tag} [{f.code}] {f.message}")
        if code == 2:
            print(
                "\nSee references/recognize-walls.md — do not scaffold until resolved.",
                file=sys.stderr,
            )

    return code


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
