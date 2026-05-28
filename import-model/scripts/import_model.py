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
"""Unified CLI dispatcher for the import-model helper scripts.

Each subcommand maps 1:1 to one of the standalone scripts in this directory.
Argument names, defaults, help text, and exit codes are inherited from the
underlying script's ``add_arguments`` / ``main`` pair.

Subcommands:

- ``inspect <HF_ID> [--output ...]``                   -> inspect_hf.py
- ``scaffold <HF_ID> --start-from <slug> --output-dir ...``  -> scaffold.py
- ``list-archs [--match <Class>]``                     -> list_native_archs.py
- ``check-walls <HF_ID> [--json]``                     -> check_walls.py
- ``list-keys <HF_ID> [--summary | --prefix ... ...]`` -> list_checkpoint_keys.py
- ``gates <HF_ID> --port-dir ... [--phase ...]``       -> run_oss_gates.py
- ``compare <HF_ID> --slug ... --port ...``            -> compare_layers.py

Each subcommand also remains usable as a standalone script:

    pixi run python /path/to/scripts/scaffold.py <HF_ID> --start-from llama3 ...

is equivalent to:

    pixi run python /path/to/scripts/import_model.py scaffold <HF_ID> --start-from llama3 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from . import (
        check_walls,
        compare_layers,
        inspect_hf,
        list_checkpoint_keys,
        list_native_archs,
        run_oss_gates,
        scaffold,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_walls  # type: ignore[no-redef]
    import compare_layers  # type: ignore[no-redef]
    import inspect_hf  # type: ignore[no-redef]
    import list_checkpoint_keys  # type: ignore[no-redef]
    import list_native_archs  # type: ignore[no-redef]
    import run_oss_gates  # type: ignore[no-redef]
    import scaffold  # type: ignore[no-redef]


_SUBCOMMANDS = [
    (
        "inspect",
        inspect_hf,
        "Inspect a HuggingFace model: config, MAX API mapping.",
    ),
    (
        "scaffold",
        scaffold,
        "Generate a custom-arch port skeleton from a MAX donor.",
    ),
    (
        "list-archs",
        list_native_archs,
        "List HF architecture classes registered in MAX.",
    ),
    (
        "check-walls",
        check_walls,
        "Scan a Hub config.json for OSS-port blockers.",
    ),
    (
        "list-keys",
        list_checkpoint_keys,
        "List Hub safetensors keys, shapes, dtypes.",
    ),
    (
        "gates",
        run_oss_gates,
        "Run OSS preflight / verify gates against a port.",
    ),
    (
        "compare",
        compare_layers,
        "Compare HF reference vs MAX serve at a prompt.",
    ),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="import_model",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcommand", required=True, metavar="COMMAND")
    for name, module, helptext in _SUBCOMMANDS:
        sp = sub.add_parser(name, help=helptext, description=module.__doc__)
        module.add_arguments(sp)
        sp.set_defaults(func=module.main)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
