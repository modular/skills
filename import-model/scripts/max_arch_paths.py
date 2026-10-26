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
"""Discover native MAX architectures without importing ``max.pipelines``.

Everything here reads the architecture sources with ``ast``, so it works in
an environment that can't import MAX's compiled extensions.
"""

from __future__ import annotations

import ast
import builtins
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

ARCH_PACKAGE = "max.pipelines.architectures"

# Every ModuleV3 pipeline model base in ``max.pipelines.lib`` carries this
# prefix: ``ModuleV3PipelineModelWithKVCache``,
# ``ModuleV3MultiGraphPipelineModelWithKVCache``.
_MODULEV3_BASE_PREFIX = "ModuleV3"
# ModuleV3 models without a KV cache subclass ``PipelineModel`` directly and
# build their module in this method.
_MODULEV3_BUILD_METHOD = "_instantiate_module"
# Bound on base-class hops across architecture packages. Real chains are
# two or three deep (phi3_modulev3 -> llama3_modulev3 -> lib base).
_MAX_CLASS_DEPTH = 8


def architectures_root() -> Path | None:
    """Return ``max/pipelines/architectures`` on disk, or ``None`` if MAX is missing."""
    try:
        import max
    except ImportError:
        return None
    # ``max`` is a namespace package, so any of its portions can come
    # first. Use the one that holds the architectures.
    for portion in max.__path__:
        root = Path(portion) / "pipelines" / "architectures"
        if root.is_dir():
            return root
    return None


def list_native_arch_mapping(root: Path | None = None) -> dict[str, list[str]]:
    """Return ``{HF architectures[0] class name: [directory slug, ...]}``.

    Reads every name each ``arch.py`` registers (see
    :func:`registered_names`). One name can have several slugs: ``qwen3`` and
    ``qwen3_embedding`` both register ``Qwen3ForCausalLM``, for text
    generation and embeddings. Slugs are sorted. ``root`` defaults to the
    installed MAX's architectures directory.
    """
    root = root or architectures_root()
    if root is None:
        return {}

    mapping: dict[str, list[str]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        parsed = parse_arch_py(entry)
        if parsed is None:
            continue
        for name in registered_names(parsed.tree):
            mapping.setdefault(name, []).append(entry.name)
    return mapping


class ImportedName(NamedTuple):
    """Where a ``from ... import`` binding comes from."""

    module: str
    """The absolute module the name is imported from."""
    name: str
    """The name inside ``module``, before any ``as`` alias."""


def module_imports(tree: ast.Module, package: str) -> dict[str, ImportedName]:
    """Map every name bound by a ``from ... import`` to where it comes from.

    ``package`` is the dotted package that holds the parsed file, so relative
    imports resolve against it: inside ``max.pipelines.architectures.olmo3``,
    ``from .model import Olmo3Model`` maps ``Olmo3Model`` to
    ``ImportedName("max.pipelines.architectures.olmo3.model", "Olmo3Model")``.
    An alias keeps the original name: ``from . import weight_adapters as wa``
    maps ``wa`` to ``ImportedName("<package>", "weight_adapters")``.
    """
    parts = package.split(".")
    imports: dict[str, ImportedName] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            module = node.module or ""
        else:
            prefix = ".".join(parts[: len(parts) - (node.level - 1)])
            module = f"{prefix}.{node.module}" if node.module else prefix
        for alias in node.names:
            imports[alias.asname or alias.name] = ImportedName(
                module, alias.name
            )
    return imports


def free_names(
    nodes: Iterable[ast.AST], bound: Iterable[str] = ()
) -> list[ast.Name]:
    """Every read of a module-scope name in ``nodes``, in source order.

    A name the code binds itself (an assignment target, a comprehension
    variable, a lambda or function argument) is local, and so is every name
    in ``bound``. Builtins such as ``set`` need no import, so they are
    skipped too. What remains are the names an import or a module-level
    definition brought into scope.
    """
    walked = [child for node in nodes for child in ast.walk(node)]
    local = set(bound)
    for child in walked:
        if isinstance(child, ast.Name) and not isinstance(child.ctx, ast.Load):
            local.add(child.id)
        elif isinstance(child, ast.arg):
            local.add(child.arg)
    reads = [
        child
        for child in walked
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id not in local
        and not hasattr(builtins, child.id)
    ]
    return sorted(reads, key=lambda n: (n.lineno, n.col_offset))


@dataclass(frozen=True)
class ParsedModule:
    """One parsed source file under ``max/pipelines/architectures``."""

    module: str
    tree: ast.Module
    imports: dict[str, ImportedName]
    source: str
    """The file's text, for :func:`ast.get_source_segment`."""

    def class_def(self, name: str) -> ast.ClassDef | None:
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def defines(self, name: str) -> bool:
        """Whether ``name`` is bound at module level by a def or assignment."""
        for node in self.tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name == name:
                    return True
            elif isinstance(node, ast.Assign):
                if any(
                    isinstance(t, ast.Name) and t.id == name
                    for t in node.targets
                ):
                    return True
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and (
                    node.target.id == name
                ):
                    return True
        return False


def parse_arch_module(root: Path, module: str) -> ParsedModule | None:
    """Parse ``max.pipelines.architectures.<...>`` from the tree at ``root``."""
    if not module.startswith(f"{ARCH_PACKAGE}."):
        return None
    rel = module[len(ARCH_PACKAGE) + 1 :].split(".")
    path = root.joinpath(*rel).with_suffix(".py")
    package = module.rsplit(".", 1)[0]
    if not path.is_file():
        path = root.joinpath(*rel, "__init__.py")
        package = module
    if not path.is_file():
        return None
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return ParsedModule(module, tree, module_imports(tree, package), source)


def resolve_class(
    root: Path, parsed: ParsedModule, name: str, depth: int = 0
) -> tuple[ParsedModule, ast.ClassDef] | None:
    """Find the ``ClassDef`` that ``name`` refers to inside ``parsed``.

    Follows ``from ... import`` chains (including package re-exports and
    ``as`` aliases) to the file that defines the class, as long as they stay
    inside the architectures tree.
    """
    node = parsed.class_def(name)
    if node is not None:
        return parsed, node
    imported = parsed.imports.get(name)
    if imported is None or depth >= _MAX_CLASS_DEPTH:
        return None
    target = parse_arch_module(root, imported.module)
    if target is None:
        return None
    return resolve_class(root, target, imported.name, depth + 1)


def base_name(expr: ast.expr) -> str | None:
    """``Foo`` for ``Foo``, ``Foo[Ctx]`` and ``pkg.Foo`` base expressions."""
    if isinstance(expr, ast.Subscript):
        expr = expr.value
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def class_lineage(
    root: Path, parsed: ParsedModule, node: ast.ClassDef, depth: int = 0
) -> Iterator[tuple[ParsedModule, ast.ClassDef]]:
    """Yield ``node`` and then its bases found in the architectures tree.

    Depth-first, child before parent. Bases defined outside the tree
    (``max.pipelines.lib``, mixins) are skipped. Callers that care about
    those check :func:`base_name` on each yielded class.
    """
    yield parsed, node
    if depth >= _MAX_CLASS_DEPTH:
        return
    for base in node.bases:
        name = base_name(base)
        if name is None:
            continue
        found = resolve_class(root, parsed, name)
        if found is not None:
            yield from class_lineage(root, *found, depth + 1)


def supported_architecture_calls(tree: ast.Module) -> Iterator[ast.Call]:
    """Yield every ``SupportedArchitecture(...)`` call in an ``arch.py``."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SupportedArchitecture"
        ):
            yield node


# ``replace`` is ``dataclasses.replace``, which derives one registration
# from another: ``replace(mimo_v2_arch, name="MiMoV2DFlashContext...")``.
_REGISTRATIONS = frozenset({"SupportedArchitecture", "replace"})


def registered_names(tree: ast.Module) -> list[str]:
    """Every architecture name an ``arch.py`` registers, in source order.

    Counts ``SupportedArchitecture(name=...)`` and registrations derived
    with ``dataclasses.replace(base, name=...)``.
    """
    names: dict[str, None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id in _REGISTRATIONS)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
            )
        ):
            name = string_keyword(node, "name")
            if name is not None:
                names.setdefault(name)
    return list(names)


def call_keyword(call: ast.Call, keyword: str) -> ast.expr | None:
    """The value ``call`` passes for ``keyword``, if it passes one."""
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def string_keyword(call: ast.Call, keyword: str) -> str | None:
    """The string literal ``call`` passes for ``keyword``, if any."""
    value = call_keyword(call, keyword)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def source_offset(source: str, lineno: int, col_offset: int) -> int:
    """Character offset in ``source`` of an AST ``(lineno, col_offset)``.

    AST columns count UTF-8 bytes, so a line with non-ASCII text before the
    node shifts the column.
    """
    lines = source.splitlines(keepends=True)
    before = sum(len(line) for line in lines[: lineno - 1])
    return before + len(lines[lineno - 1].encode()[:col_offset].decode())


def parse_arch_py(arch_dir: Path) -> ParsedModule | None:
    """Parse ``<arch_dir>/arch.py`` with imports resolved to absolute names."""
    return parse_arch_module(
        arch_dir.parent, f"{ARCH_PACKAGE}.{arch_dir.name}.arch"
    )


def is_modulev3_lineage(
    lineage: Iterable[tuple[ParsedModule, ast.ClassDef]],
) -> bool:
    """Whether a pipeline model's lineage reaches a ModuleV3 base.

    ``lineage`` is what :func:`class_lineage` yields for the pipeline model.
    True when one of those classes subclasses a ``ModuleV3*PipelineModel*``
    base in ``max.pipelines.lib`` or defines ``_instantiate_module``.
    """
    for _, node in lineage:
        if any(
            isinstance(stmt, ast.FunctionDef)
            and stmt.name == _MODULEV3_BUILD_METHOD
            for stmt in node.body
        ):
            return True
        for base in node.bases:
            name = base_name(base)
            if (
                name
                and name.startswith(_MODULEV3_BASE_PREFIX)
                and "PipelineModel" in name
            ):
                return True
    return False


def find_arch_dir(slug: str) -> Path:
    """Return ``architectures/<slug>/`` or exit with a helpful message."""
    root = architectures_root()
    if root is None:
        sys.exit(
            "MAX is not installed in this Python environment. "
            "Install MAX with pixi (https://max.modular.com/get-started), "
            "not pip."
        )
    candidate = root / slug
    if candidate.is_dir():
        return candidate
    known = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())[:12])
    sys.exit(
        f"Architecture directory {slug!r} not found under {root}. "
        f"List donor slugs with: pixi run python list_native_archs.py "
        f"--donors\n"
        f"(sample slugs: {known}...)"
    )
