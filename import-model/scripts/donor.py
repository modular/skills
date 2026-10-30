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
"""Resolve a ModuleV3 donor architecture for ``scaffold.py``.

Builds on the ``ast`` layer in ``max_arch_paths``. :func:`load_donor`
parses the donor's ``arch.py`` and picks its ``SupportedArchitecture`` call.
A donor comes in one of two styles:

- **registered**: the call sets ``modules={GraphRole.DECODER: <Module>}``
  and ``config=``. The framework builds the pipeline model, so the donor
  has none, and its decoder takes ``DecoderInputs``.
- **pipeline model**: the call sets ``pipeline_model=`` to a ModuleV3
  pipeline model, whose class lineage the donor walks to find the root
  module and the config class.

Everything else the scaffold asks (the carried keywords, the root module,
the config class) is a cached property of the returned :class:`Donor`,
resolved on first use. The properties the subclass skeleton needs raise
:class:`DonorError` for donors it can't subclass, so ``--full-copy`` never
touches them. :func:`list_donors` lists every architecture that :func:`load_donor`
accepts.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

try:
    from .max_arch_paths import (
        ARCH_PACKAGE,
        ImportedName,
        ParsedModule,
        architectures_root,
        call_keyword,
        class_lineage,
        free_names,
        is_modulev3_lineage,
        parse_arch_module,
        parse_arch_py,
        resolve_class,
        source_offset,
        string_keyword,
        supported_architecture_calls,
    )
except ImportError:
    # Standalone invocation: `python /path/to/scaffold.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from max_arch_paths import (  # type: ignore[no-redef]
        ARCH_PACKAGE,
        ImportedName,
        ParsedModule,
        architectures_root,
        call_keyword,
        class_lineage,
        free_names,
        is_modulev3_lineage,
        parse_arch_module,
        parse_arch_py,
        resolve_class,
        source_offset,
        string_keyword,
        supported_architecture_calls,
    )


class DonorError(ValueError):
    """The donor's sources don't expose what the scaffold needs.

    The message is ready to show the user as-is.
    """


@dataclass(frozen=True)
class CarriedName:
    """One module-scope name a carried keyword value reads."""

    name: str
    """The name as the donor's ``arch.py`` binds it."""
    source: ImportedName
    """Where the port imports it from."""
    spans: tuple[tuple[int, int], ...]
    """``(start, end)`` of each occurrence, as offsets into the value text."""
    role: Literal["model", "config", "module"] | None = None
    """``model``, ``config`` or ``module`` when the name is the class the
    donor passes as ``pipeline_model=``, ``config=`` or its ``DECODER`` in
    ``modules=``. The port has its own class for each, so the renderer
    points these references at the port's classes."""


@dataclass(frozen=True)
class DonorKeyword:
    """How the donor's ``arch.py`` sets one ``SupportedArchitecture`` keyword.

    ``expr`` is the value's source text, verbatim, and ``names`` lists every
    module-scope name it reads with the import that provides it. ``expr``
    is ``None`` when a name could not be followed (for example, one bound
    by a plain ``import x.y``). That keyword needs a human: guessing a
    default would change serving behavior with no error or warning.
    """

    expr: str | None
    names: tuple[CarriedName, ...] = ()


def _root_module_name(fn: ast.FunctionDef) -> str | None:
    """Name of the class ``_instantiate_module`` constructs and returns.

    Single-device donors assign the root module to a local
    (``nn_model = Olmo3(...)``), move it to the device, and return the local.
    Multi-device donors return the constructor call from inside
    ``default_device(mesh)`` (``return Gemma3(...)``).
    """
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            return node.value.func.id
    returned = {
        node.value.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    }
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and any(
                isinstance(t, ast.Name) and t.id in returned
                for t in node.targets
            )
        ):
            return node.value.func.id
    return None


def _registered_decoder(call: ast.Call) -> ast.Name | None:
    """The class ``call`` registers for the ``DECODER`` role, if any.

    Reads ``modules={GraphRole.DECODER: Cls}``. The key can also be the
    role's string value, since ``GraphRole`` is a ``str`` enum.
    """
    value = call_keyword(call, "modules")
    if not isinstance(value, ast.Dict):
        return None
    for key, entry in zip(value.keys, value.values, strict=True):
        is_decoder = (
            isinstance(key, ast.Attribute) and key.attr == "DECODER"
        ) or (isinstance(key, ast.Constant) and key.value == "decoder")
        if is_decoder and isinstance(entry, ast.Name):
            return entry
    return None


def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    """The method ``name`` defined in the body of ``node``, if any."""
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            return stmt
    return None


def _assigned(node: ast.ClassDef) -> dict[str, ast.expr]:
    """Class attributes ``node`` assigns in its body, by name."""
    assigned: dict[str, ast.expr] = {}
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if isinstance(stmt.target, ast.Name):
                assigned[stmt.target.id] = stmt.value
    return assigned


@dataclass(frozen=True)
class Donor:
    """A ModuleV3 donor architecture, resolved once per scaffold run."""

    slug: str
    root: Path
    """The architectures directory that holds the donor package."""
    arch: ParsedModule
    """The donor's ``arch.py``."""
    call: ast.Call
    """The ``SupportedArchitecture`` call the port is built from.

    The first call that registers a ``DECODER`` module or whose
    ``pipeline_model=`` resolves to a ModuleV3 pipeline model. An
    ``arch.py`` can register more than one task (text generation and
    embeddings), and mixing their keywords would register a port that
    neither call describes.
    """
    name: str
    """The architecture name :attr:`call` registers."""
    lineage: tuple[tuple[ParsedModule, ast.ClassDef], ...] = ()
    """The ``pipeline_model=`` class, then its bases in the tree. Empty for
    a registered donor."""
    decoder: tuple[ParsedModule, ast.ClassDef] | None = None
    """The class a registered donor's ``modules=`` names for ``DECODER``."""

    @property
    def registered(self) -> bool:
        """Whether the donor registers its modules by role.

        A registered donor has no pipeline model: the framework builds one
        from ``modules=`` and ``config=``, and constructs the decoder as
        ``decoder_cls(config)``.
        """
        return self.decoder is not None

    def _fail(self, detail: str) -> DonorError:
        return DonorError(
            f"Could not introspect donor {self.slug}: {detail}. "
            f"Use --full-copy mode for this donor."
        )

    @property
    def model(self) -> ImportedName:
        """The pipeline model class and the module that defines it.

        Raises:
            DonorError: The donor is registered, so it has none.
        """
        if not self.lineage:
            raise DonorError(
                f"{self.slug} registers its modules by role and has no "
                f"pipeline model"
            )
        parsed, node = self.lineage[0]
        return ImportedName(parsed.module, node.name)

    @cached_property
    def keywords(self) -> dict[str, DonorKeyword]:
        """Every keyword :attr:`call` sets, resolved, in source order.

        A keyword the donor leaves at its default is absent from the result.
        For ``memory_planner`` that is correct for architectures that do
        their own memory estimation (diffusion, embedding).
        """
        return {
            kw.arg: self._resolve(kw.value)
            for kw in self.call.keywords
            if kw.arg is not None
        }

    def _resolve(self, value: ast.expr) -> DonorKeyword:
        """Resolve one keyword value to its source text and the names it reads.

        Follows the import that brought each referenced name into scope, or
        the donor's own ``arch.py`` for a module-level constant. The text is
        preserved verbatim, so configured values such as
        ``PagedMemoryPlanner.with_activation_reservation(0)`` carry their
        arguments across to the port.
        """
        source = self.arch.source
        assert value.end_lineno is not None and value.end_col_offset is not None
        start = source_offset(source, value.lineno, value.col_offset)
        end = source_offset(source, value.end_lineno, value.end_col_offset)
        roles: dict[str, Literal["model", "config", "module"]] = {}
        model = call_keyword(self.call, "pipeline_model")
        config = call_keyword(self.call, "config")
        decoder = _registered_decoder(self.call)
        if isinstance(model, ast.Name):
            roles[model.id] = "model"
        if isinstance(config, ast.Name):
            roles[config.id] = "config"
        if decoder is not None:
            roles[decoder.id] = "module"

        spans: dict[str, list[tuple[int, int]]] = {}
        for node in free_names([value]):
            assert node.end_lineno is not None
            assert node.end_col_offset is not None
            spans.setdefault(node.id, []).append(
                (
                    source_offset(source, node.lineno, node.col_offset) - start,
                    source_offset(source, node.end_lineno, node.end_col_offset)
                    - start,
                )
            )
        names = []
        for name, occurrences in spans.items():
            imported = self.arch.imports.get(name)
            if imported is None and self.arch.defines(name):
                imported = ImportedName(self.arch.module, name)
            if imported is None:
                return DonorKeyword(expr=None)
            names.append(
                CarriedName(name, imported, tuple(occurrences), roles.get(name))
            )
        # Multi-line values keep the donor's indentation verbatim: the
        # donor's keyword sits at the same depth inside its own
        # SupportedArchitecture call, so the text transfers as-is.
        return DonorKeyword(expr=source[start:end], names=tuple(names))

    @cached_property
    def safetensors_adapter(self) -> ImportedName | None:
        """The safetensors weight adapter function and its module.

        Reads the ``WeightsFormat.safetensors`` entry of
        ``weight_adapters={...}`` and follows the import that brought the
        referenced name into scope.
        """
        imports = self.arch.imports
        value = call_keyword(self.call, "weight_adapters")
        if not isinstance(value, ast.Dict):
            return None
        for key, entry in zip(value.keys, value.values, strict=True):
            if not (
                isinstance(key, ast.Attribute) and key.attr == "safetensors"
            ):
                continue
            if isinstance(entry, ast.Attribute) and isinstance(
                entry.value, ast.Name
            ):
                # e.g. weight_adapters.convert_safetensor_state_dict
                imported = imports.get(entry.value.id)
                if imported:
                    return ImportedName(
                        f"{imported.module}.{imported.name}", entry.attr
                    )
            elif isinstance(entry, ast.Name):
                # e.g. convert_safetensor_state_dict imported directly
                return imports.get(entry.id)
        return None

    @cached_property
    def _instantiate(self) -> tuple[ParsedModule, ast.FunctionDef] | None:
        for parsed, node in self.lineage:
            method = _method(node, "_instantiate_module")
            if method is not None:
                return parsed, method
        return None

    @cached_property
    def _root_module(self) -> tuple[ParsedModule, ast.ClassDef] | None:
        if self.decoder is not None:
            return self.decoder
        if self._instantiate is None:
            return None
        parsed, fn = self._instantiate
        name = _root_module_name(fn)
        return resolve_class(self.root, parsed, name) if name else None

    def _no_instantiate(self) -> DonorError:
        return self._fail(
            f"{self.model.name} defines no _instantiate_module "
            f"(multi-module donors compile each tower separately)"
        )

    @property
    def instantiate(self) -> tuple[ParsedModule, ast.FunctionDef]:
        """The nearest ``_instantiate_module`` in the lineage.

        Raises:
            DonorError: No class in the lineage defines it. Multi-module
                donors compile each tower separately.
        """
        if self._instantiate is None:
            raise self._no_instantiate()
        return self._instantiate

    @property
    def module(self) -> ImportedName:
        """The root module class: a registered donor's ``DECODER``, or the
        class ``_instantiate_module`` builds.

        Raises:
            DonorError: The donor defines no ``_instantiate_module``, or
                the class it builds can't be found in the tree.
        """
        if self.decoder is not None:
            parsed, node = self.decoder
            return ImportedName(parsed.module, node.name)
        if self._instantiate is None:
            raise self._no_instantiate()
        if self._root_module is None:
            raise self._fail(
                f"could not find the root module class that "
                f"{self.model.name}._instantiate_module constructs"
            )
        parsed, node = self._root_module
        return ImportedName(parsed.module, node.name)

    @property
    def module_stem(self) -> str | None:
        """File stem of the root module inside the donor package.

        ``llama3`` for ``llama3_modulev3``. ``None`` for donors whose root
        module lives in another package (``phi3_modulev3``) and for
        multi-module donors that define no ``_instantiate_module``.
        """
        if self._root_module is None:
            return None
        module = self._root_module[0].module
        prefix = f"{ARCH_PACKAGE}.{self.slug}."
        stem = module.removeprefix(prefix)
        if not module.startswith(prefix) or "." in stem:
            return None
        return stem

    @cached_property
    def config(self) -> ImportedName:
        """The donor's config class.

        A registered donor's is its ``config=``. A pipeline model donor's is
        the class its ``_create_model_config`` builds: the llama3_modulev3
        family builds ``config_class`` (``OlmoModel`` sets ``OlmoConfig``
        there), and every other donor builds ``model_config_cls``. The
        nearest class in the lineage that sets either one wins.

        Raises:
            DonorError: The donor names no config class the tree defines.
        """
        if self.registered:
            value = call_keyword(self.call, "config")
            found = (
                resolve_class(self.root, self.arch, value.id)
                if isinstance(value, ast.Name)
                else None
            )
            if found is None:
                raise self._fail(
                    "could not resolve the config= class its arch.py registers"
                )
            return ImportedName(found[0].module, found[1].name)
        for parsed, node in self.lineage:
            assigned = _assigned(node)
            for attr in ("config_class", "model_config_cls"):
                value = assigned.get(attr)
                if isinstance(value, ast.Name):
                    found = resolve_class(self.root, parsed, value.id)
                    if found is not None:
                        return ImportedName(found[0].module, found[1].name)
        raise self._fail(
            f"could not resolve the config class {self.model.name} builds"
        )

    def config_defines(self, method: str) -> bool:
        """Whether the config class or one of its bases in the tree
        defines ``method``.

        Bases outside the architectures tree (in ``max.pipelines.lib``)
        aren't read.
        """
        parsed = parse_arch_module(self.root, self.config.module)
        node = parsed.class_def(self.config.name) if parsed else None
        if parsed is None or node is None:
            return False
        return any(
            _method(cls, method) is not None
            for _, cls in class_lineage(self.root, parsed, node)
        )

    @cached_property
    def config_names(self) -> frozenset[str]:
        """Every identifier and string the config class's source files mention.

        Covers the module that defines the config class and the modules of
        its bases in the tree. A ``config.json`` key absent from this set is
        one the donor's config never reads, so the port must handle it.
        Helpers outside the architectures tree (such as ``get_rope_theta``)
        aren't scanned.
        """
        parsed = parse_arch_module(self.root, self.config.module)
        node = parsed.class_def(self.config.name) if parsed else None
        if parsed is None or node is None:
            return frozenset()
        names: set[str] = set()
        for module, _ in class_lineage(self.root, parsed, node):
            for child in ast.walk(module.tree):
                if isinstance(child, ast.Attribute):
                    names.add(child.attr)
                elif isinstance(child, ast.Name):
                    names.add(child.id)
                elif isinstance(child, ast.keyword) and child.arg:
                    names.add(child.arg)
                elif isinstance(child, ast.Constant) and isinstance(
                    child.value, str
                ):
                    names.add(child.value)
        return frozenset(names)

    @property
    def builds_config(self) -> bool:
        """Whether the donor's module is built from a config object.

        Always true for a registered donor. ``qwen3_embedding_modulev3``'s
        ``_create_model_config`` returns ``None`` and builds its module
        from ``huggingface_config``, so the port has no config to re-type.
        """
        for _, node in self.lineage:
            create = _method(node, "_create_model_config")
            if create is not None:
                return not (
                    isinstance(create.returns, ast.Constant)
                    and create.returns.value is None
                )
        return True


def _resolve_donor(donor_dir: Path) -> Donor | None:
    """Resolve ``donor_dir`` as a ModuleV3 donor, or return ``None``."""
    arch = parse_arch_py(donor_dir)
    if arch is None:
        return None
    root = donor_dir.parent
    for call in supported_architecture_calls(arch.tree):
        name = string_keyword(call, "name")
        if name is None:
            continue
        decoder = _registered_decoder(call)
        if decoder is not None:
            found = resolve_class(root, arch, decoder.id)
            if found is not None:
                return Donor(
                    donor_dir.name, root, arch, call, name, decoder=found
                )
            continue
        model = call_keyword(call, "pipeline_model")
        if not isinstance(model, ast.Name):
            continue
        found = resolve_class(root, arch, model.id)
        if found is None:
            continue
        lineage = tuple(class_lineage(root, *found))
        if is_modulev3_lineage(lineage):
            return Donor(donor_dir.name, root, arch, call, name, lineage)
    return None


def load_donor(donor_dir: Path) -> Donor:
    """Resolve ``donor_dir`` as a ModuleV3 donor.

    Raises:
        DonorError: ``donor_dir`` is not a ModuleV3 architecture. The
            message lists the architectures that are.
    """
    donor = _resolve_donor(donor_dir)
    if donor is None:
        donors = ", ".join(
            slug for slug, _ in list_modulev3_donors(donor_dir.parent)
        )
        raise DonorError(
            f"{donor_dir.name} is not a ModuleV3 architecture. "
            f"Pass --start-from one of: {donors}"
        )
    return donor


def list_donors(root: Path | None = None) -> list[Donor]:
    """Return every ModuleV3 architecture directory, resolved as a donor.

    A directory qualifies when :func:`load_donor` accepts it. ``root``
    defaults to the installed MAX's architectures directory. Pass a
    source-tree path to inspect a checkout.
    """
    root = root or architectures_root()
    if root is None:
        return []
    donors = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        # load_donor builds its error message from this list, so calling
        # it here would recurse on every rejected directory.
        donor = _resolve_donor(entry)
        if donor is not None:
            donors.append(donor)
    return donors


def list_modulev3_donors(root: Path | None = None) -> list[tuple[str, str]]:
    """Return ``(slug, arch name)`` for every donor :func:`list_donors` finds."""
    return [(donor.slug, donor.name) for donor in list_donors(root)]
