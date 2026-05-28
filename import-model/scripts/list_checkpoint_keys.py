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
"""List Hub safetensors keys, shapes, and dtypes without downloading weights.

Uses ``huggingface_hub.get_safetensors_metadata`` (HTTP header / index parsing
only). Run this right after reading ``config.json`` and before writing
``weight_adapters.py``.

Usage::

    pixi run python list_checkpoint_keys.py <HF_MODEL_ID>
    pixi run python list_checkpoint_keys.py <HF_MODEL_ID> --summary
    pixi run python list_checkpoint_keys.py <HF_MODEL_ID> \\
        --prefix model.language_model. --limit 40
    pixi run python list_checkpoint_keys.py <HF_MODEL_ID> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .checkpoint_metadata import fetch_repo_tensors, filter_tensors
except ImportError:
    # Standalone invocation: `python /path/to/list_checkpoint_keys.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from checkpoint_metadata import (  # type: ignore[no-redef]
        fetch_repo_tensors,
        filter_tensors,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id", help="HuggingFace model ID")
    parser.add_argument(
        "--prefix", help="Only show keys starting with this prefix"
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Drop keys starting with this prefix (repeatable)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print dtype/param summary only"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max rows to print (0 = all)"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")


def main(args: argparse.Namespace) -> int:
    try:
        summary = fetch_repo_tensors(args.hf_id)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    tensors = filter_tensors(
        summary.tensors,
        prefix=args.prefix,
        exclude_prefixes=tuple(args.exclude_prefix),
    )

    if args.json:
        payload = {
            "hf_id": summary.hf_id,
            "sharded": summary.sharded,
            "dtype_counts": summary.dtype_counts,
            "dominant_dtype": summary.dominant_dtype(),
            "total_parameters": summary.total_parameters,
            "tensors": [t.__dict__ for t in tensors],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"repo: {summary.hf_id}")
    print(f"sharded: {summary.sharded}")
    print(f"dtype_counts: {summary.dtype_counts}")
    print(f"dominant_dtype: {summary.dominant_dtype()}")
    print(f"total_parameters: {summary.total_parameters:,}")

    if args.summary:
        return 0

    print()
    print(f"{'tensor':60}  {'dtype':>10}  shape")
    print("-" * 90)
    rows = tensors if args.limit <= 0 else tensors[: args.limit]
    for t in rows:
        shape = "x".join(str(x) for x in t.shape)
        print(f"{t.name:60}  {t.dtype:>10}  [{shape}]")
    if args.limit > 0 and len(tensors) > args.limit:
        print(
            f"... {len(tensors) - args.limit} more keys (use --limit 0 for all)"
        )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
