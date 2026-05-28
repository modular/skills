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
"""List HuggingFace architecture classes registered in installed MAX.

Use before scaffolding: if the model's config.json::architectures[0] appears
here, MAX already supports it and you can run `pixi run max serve` directly.

Usage:
    pixi run python list_native_archs.py
    pixi run python list_native_archs.py --match LlamaForCausalLM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .max_arch_paths import list_native_arch_mapping
except ImportError:
    # Standalone invocation: `python /path/to/list_native_archs.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from max_arch_paths import (
        list_native_arch_mapping,  # type: ignore[no-redef]
    )

MAX_NOT_INSTALLED_MSG = """\
MAX is not installed in this Python environment (or no architectures could be discovered).
Install MAX with pixi — not `pip install modular` (that is the Mojo language package, not MAX):
  https://docs.modular.com/max/get-started
Then rerun inside that environment, for example:
  pixi run python list_native_archs.py --match <ArchitecturesClassFromConfig>
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--match",
        metavar="ARCH_CLASS",
        help="Exit 0 if this HF architectures[0] value is registered, else 1.",
    )


def main(args: argparse.Namespace) -> int:
    mapping = list_native_arch_mapping()
    if not mapping:
        print(MAX_NOT_INSTALLED_MSG, file=sys.stderr)
        return 2

    if args.match:
        slug = mapping.get(args.match)
        if slug:
            print(f"{args.match}\t{slug}")
            return 0
        return 1

    for arch_class, slug in sorted(mapping.items()):
        print(f"{arch_class}\t{slug}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
