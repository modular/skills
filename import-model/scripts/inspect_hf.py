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
"""Inspect a HuggingFace model: config, modeling-code location, deltas.

Fetches the repo's raw ``config.json`` from the Hub and maps every key to the
MAX API surface (``pipeline_config.model.huggingface_config``, ``arch.py``,
``model_config.py``).

Usage:
    pixi run python inspect_hf.py <HF_MODEL_ID>
    pixi run python inspect_hf.py <HF_MODEL_ID> --output report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .hub_config import hub_config_url, load_hub_config
    from .max_arch_paths import list_native_arch_mapping
except ImportError:
    # Standalone invocation: `python /path/to/inspect_hf.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hub_config import (  # type: ignore[no-redef]
        hub_config_url,
        load_hub_config,
    )
    from max_arch_paths import (
        list_native_arch_mapping,  # type: ignore[no-redef]
    )

# Keys whose MAX destination is not a 1:1 huggingface_config attribute.
MAX_API_OVERRIDES: dict[str, str] = {
    "architectures": "`SupportedArchitecture.name` ← `architectures[0]` in `arch.py`",
    "torch_dtype": "`SupportedArchitecture.default_encoding` / `supported_encodings` in `arch.py`",
    "dtype": "`SupportedArchitecture.default_encoding` / `supported_encodings` in `arch.py`",
    "model_type": "HF modeling path only (`transformers.models.<model_type>.modeling_<model_type>`)",
}


def max_api_for(key: str) -> str:
    if key in MAX_API_OVERRIDES:
        return MAX_API_OVERRIDES[key]
    return (
        f"`pipeline_config.model.huggingface_config.{key}` "
        f"→ read in `MyConfig.initialize()` / set on `MyConfig`"
    )


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif value is None:
        text = "null"
    else:
        text = json.dumps(value)
    if len(text) > 100:
        text = text[:97] + "..."
    return text


def config_mapping_lines(cfg: dict) -> list[str]:
    lines = [
        "## config.json → MAX API",
        "",
        "| Key | Value | MAX API |",
        "|-----|-------|---------|",
    ]
    for key in sorted(cfg):
        value = format_value(cfg[key])
        api = max_api_for(key)
        lines.append(f"| `{key}` | {value} | {api} |")
    lines.append("")
    return lines


def native_arch_lines(arch_class: str, hf_id: str) -> list[str]:
    """Guard: is architectures[0] already registered in MAX?"""
    mapping = list_native_arch_mapping()
    lines = ["## Guard — Native in MAX?", ""]
    if not mapping:
        lines.extend(
            [
                "Could not read the MAX registry (MAX not installed in this Python env).",
                "Install MAX with pixi, then run:",
                f"`pixi run python list_native_archs.py --match {arch_class}`",
                "",
            ]
        )
        return lines

    if mapping.get(arch_class):
        lines.extend(
            [
                f"`{arch_class}` is **already registered** in MAX.",
                "",
                "Stop here — no port needed. Serve the Hub checkpoint:",
                "",
                "```bash",
                f"pixi run max serve --model {hf_id}",
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"`{arch_class}` is **not** in the MAX registry — continue with Phase 1.",
                "",
                "```bash",
                f"pixi run python list_native_archs.py --match {arch_class}  # exit 1",
                "```",
                "",
            ]
        )
    return lines


def build_report(hf_id: str, cfg: dict) -> str:
    arch = (cfg.get("architectures") or ["Unknown"])[0]
    model_type = cfg.get("model_type", "unknown")

    lines = [
        f"# HF inspection: `{hf_id}`",
        "",
        f"- **Config source:** [`config.json`]({hub_config_url(hf_id)})",
        f"- **Architecture class:** `{arch}` (`architectures[0]`)",
        f"- **model_type:** `{model_type}`",
        "",
    ]
    lines.extend(native_arch_lines(arch, hf_id))
    lines.extend(config_mapping_lines(cfg))
    lines.extend(
        [
            "## Where the modeling code lives",
            "",
            "```python",
            "import importlib",
            f"mod = importlib.import_module('transformers.models.{model_type}.modeling_{model_type}')",
            "print(mod.__file__)",
            "```",
            "",
            "Read in this order: model `__init__`, attention `forward`, MLP `forward`, "
            "block class, final LM head.",
            "",
            "## Checklist (complete before implementing the graph)",
            "",
            "- [ ] Confirmed `architectures[0]` not already in MAX (see guard above)",
            "- [ ] Delta list written (one row per HF vs donor difference)",
            "- [ ] Every config.json key has a plan in `model_config.py` / nn layers",
            "- [ ] Read attention `forward` and noted Q/K/V layout, RoPE style, mask",
            "- [ ] Read MLP `forward` and noted gated/non-gated shape and activation",
            "- [ ] Read block class and noted pre-norm / post-norm / peri-LN pattern",
            "- [ ] Each delta implemented in `<slug>.py` (not serving donor shim)",
            "",
        ]
    )
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id", help="HuggingFace model ID")
    parser.add_argument(
        "--output", type=Path, help="Write report here (default: stdout)"
    )


def main(args: argparse.Namespace) -> int:
    try:
        cfg = load_hub_config(args.hf_id)
    except Exception as exc:
        sys.exit(f"Failed to fetch config.json for {args.hf_id!r}: {exc}")
    report = build_report(args.hf_id, cfg)

    if args.output:
        args.output.write_text(report)
        print(f"Wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
