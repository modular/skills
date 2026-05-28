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
"""Discover native MAX architectures without importing ``max.pipelines``."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ARCH_NAME_RE = re.compile(
    r"""SupportedArchitecture\s*\(\s*[\s\S]*?name\s*=\s*["']([^"']+)["']""",
    re.MULTILINE,
)
# Fallback: first ``name="..."`` in arch.py (older / minimal arch shells).
_SIMPLE_NAME_RE = re.compile(r"""name\s*=\s*["']([^"']+)["']""")


def architectures_root() -> Path | None:
    """Return ``max/pipelines/architectures`` on disk, or ``None`` if MAX is missing."""
    try:
        import max
    except ImportError:
        return None
    root = Path(max.__path__[0]) / "pipelines" / "architectures"
    return root if root.is_dir() else None


def _read_arch_name(arch_py: Path) -> str | None:
    if not arch_py.is_file():
        return None
    text = arch_py.read_text(encoding="utf-8", errors="replace")
    m = _ARCH_NAME_RE.search(text)
    if m:
        return m.group(1)
    for m in _SIMPLE_NAME_RE.finditer(text):
        candidate = m.group(1)
        if candidate.endswith("ForCausalLM") or "For" in candidate:
            return candidate
    return None


def list_native_arch_mapping() -> dict[str, str]:
    """Return ``{HF architectures[0] class name: directory slug}``."""
    root = architectures_root()
    if root is None:
        return {}

    mapping: dict[str, str] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        name = _read_arch_name(entry / "arch.py")
        if name:
            mapping[name] = entry.name
    return mapping


def find_arch_dir(slug: str) -> Path:
    """Return ``architectures/<slug>/`` or exit with a helpful message."""
    root = architectures_root()
    if root is None:
        sys.exit(
            "MAX is not installed in this Python environment. "
            "Install MAX with pixi (https://docs.modular.com/max/get-started), "
            "not pip install modular."
        )
    candidate = root / slug
    if candidate.is_dir():
        return candidate
    known = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())[:12])
    sys.exit(
        f"Architecture directory {slug!r} not found under {root}. "
        f"List slugs with: pixi run python list_native_archs.py\n"
        f"(sample slugs: {known}...)"
    )
