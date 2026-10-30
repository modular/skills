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
Each row is ``arch name<TAB>slug``. A name that several architecture
directories register (``Qwen3ForCausalLM`` in ``qwen3`` and
``qwen3_embedding``) gets one row per directory.

``--donors`` lists the ModuleV3 architectures that ``scaffold.py --start-from``
accepts, one ``slug<TAB>arch name<TAB>style`` row each. The style is
``registered`` for a donor that registers its modules by role, whose port
registers a ``DECODER`` module and has no pipeline model, and
``pipeline_model`` for a donor whose port subclasses its pipeline model.

Usage:
    pixi run python list_native_archs.py
    pixi run python list_native_archs.py --match LlamaForCausalLM
    pixi run python list_native_archs.py --donors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .donor import list_donors
    from .max_arch_paths import list_native_arch_mapping
except ImportError:
    # Standalone invocation: `python /path/to/list_native_archs.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from donor import list_donors  # type: ignore[no-redef]
    from max_arch_paths import (  # type: ignore[no-redef]
        list_native_arch_mapping,
    )

MAX_NOT_INSTALLED_MSG = """\
MAX is not installed in this Python environment (or no architectures could be discovered).
Install MAX with pixi, not pip:
  https://max.modular.com/get-started
Then rerun inside that environment, for example:
  pixi run python list_native_archs.py --match <ArchitecturesClassFromConfig>
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--match",
        metavar="ARCH_CLASS",
        help="Print the slugs that register this HF architectures[0] value "
        "and exit 0, or exit 1 if none does.",
    )
    group.add_argument(
        "--donors",
        action="store_true",
        help="List ModuleV3 donor slugs for scaffold.py --start-from, "
        "with each donor's style (registered or pipeline_model).",
    )


def main(args: argparse.Namespace) -> int:
    if args.donors:
        donors = list_donors()
        if not donors:
            print(MAX_NOT_INSTALLED_MSG, file=sys.stderr)
            return 2
        for donor in donors:
            style = "registered" if donor.registered else "pipeline_model"
            print(f"{donor.slug}\t{donor.name}\t{style}")
        return 0

    mapping = list_native_arch_mapping()
    if not mapping:
        print(MAX_NOT_INSTALLED_MSG, file=sys.stderr)
        return 2

    if args.match:
        slugs = mapping.get(args.match, [])
        for slug in slugs:
            print(f"{args.match}\t{slug}")
        return 0 if slugs else 1

    for arch_class, slugs in sorted(mapping.items()):
        for slug in slugs:
            print(f"{arch_class}\t{slug}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
