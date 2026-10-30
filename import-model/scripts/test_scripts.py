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
"""Tests for the import-model scripts.

Run from this directory::

    pixi run test-scripts                       # everything
    pytest test_scripts.py -m "not smoke"       # no MAX install, no network

Tests that read donor sources use the MAX source tree next to this skill and
skip when it isn't there. ``smoke`` tests run each script as a subprocess.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import email.message
import io
import shutil
import subprocess
import sys
import types
import urllib.error
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import check_port
import compare_layers
import donor as donor_mod
import full_copy
import max_arch_paths
import run_oss_gates
import scaffold
import templates

_ARCHS = (
    _SCRIPT_DIR / "../../../../max/python/max/pipelines/architectures"
).resolve()
_ARCH_PKG = "max.pipelines.architectures"

needs_tree = pytest.mark.skipif(
    not _ARCHS.is_dir(), reason="MAX architectures source tree not present"
)
_DONORS = donor_mod.list_modulev3_donors(_ARCHS) if _ARCHS.is_dir() else []


def _donor(slug: str) -> donor_mod.Donor:
    if not (_ARCHS / slug / "arch.py").is_file():
        pytest.skip(f"donor {slug} not in this tree")
    return donor_mod.load_donor(_ARCHS / slug)


def _lint(files: dict[str, str], package: str, tmp_path: Path) -> None:
    """Fail on ruff's import-sorting and pyflakes findings in ``files``.

    E501 is in the repo's ruff ignore list, so only I and F are checked.
    Without ruff on PATH, the check is skipped with a warning.
    """
    if shutil.which("ruff") is None:
        warnings.warn("ruff not on PATH; skipping generated-code lint")
        return
    port = tmp_path / package
    port.mkdir(exist_ok=True)
    for name, content in files.items():
        (port / name).write_text(content, encoding="utf-8")
    proc = subprocess.run(
        ["ruff", "check", "--isolated", "--select", "I,F", str(port)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _render_arch(
    keywords: dict[str, donor_mod.DonorKeyword], arch_name: str
) -> tuple[str, list[str]]:
    return templates.render_arch(
        slug="p",
        arch_name=arch_name,
        short=scaffold.short_name(arch_name),
        hf_id="org/m",
        donor_keywords=keywords,
    )


def _render_port(slug: str, arch_name: str) -> templates.Skeleton:
    """Render the subclass skeleton ``scaffold.py`` writes for ``slug``."""
    return templates.render_skeleton(
        slug=f"port_{slug.lower()}",
        short=scaffold.short_name(arch_name),
        arch_name=arch_name,
        hf_id="org/m",
        donor=_donor(slug),
    )


def _call_keywords(rendered: str) -> list[str]:
    """Keywords of the one ``SupportedArchitecture(...)`` call in ``rendered``."""
    calls = list(
        max_arch_paths.supported_architecture_calls(ast.parse(rendered))
    )
    assert len(calls) == 1, rendered
    return [kw.arg for kw in calls[0].keywords if kw.arg is not None]


def _imports_from(tree: ast.Module) -> dict[str, str]:
    """Map each ``from X import Name [as alias]`` in ``tree`` to ``X.Name``."""
    return {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 0
        for alias in node.names
    }


def _class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def test_architectures_root_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``architectures_root`` finds the architectures in any ``max`` portion.

    ``max`` is a namespace package, so the portion that holds
    ``pipelines/architectures`` isn't always ``max.__path__[0]``.
    """
    empty = tmp_path / "empty_portion"
    real = tmp_path / "real_portion"
    empty.mkdir()
    (real / "pipelines" / "architectures").mkdir(parents=True)
    fake = types.ModuleType("max")
    fake.__path__ = [str(empty), str(real)]
    monkeypatch.setitem(sys.modules, "max", fake)
    assert max_arch_paths.architectures_root() == (
        real / "pipelines" / "architectures"
    )


@needs_tree
@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("llama3_modulev3", True),
        # No model file of its own: inherits llama3_modulev3's pipeline model.
        ("phi3_modulev3", True),
        ("granite_modulev3", True),
        ("deepseekV3_modulev3", True),
        # No KV cache: subclasses PipelineModel, not a ModuleV3 base.
        ("mpnet_modulev3", True),
        # Architectures without a ModuleV3 pipeline model.
        ("llama3", False),
        ("mpnet", False),
        ("gemma3", False),
        ("deepseekV3", False),
    ],
)
def test_modulev3_detection(slug: str, expected: bool) -> None:
    """Only ModuleV3 architectures are donors."""
    if not (_ARCHS / slug).is_dir():
        pytest.skip(f"{slug} not in this tree")
    try:
        donor_mod.load_donor(_ARCHS / slug)
        loaded = True
    except donor_mod.DonorError:
        loaded = False
    assert loaded == expected


@needs_tree
def test_list_modulev3_donors() -> None:
    donors = dict(_DONORS)
    assert "llama3" not in donors


@needs_tree
def test_native_mapping_reads_every_registration() -> None:
    """Every registered name maps to every slug that registers it.

    ``qwen3`` and ``qwen3_embedding`` both register ``Qwen3ForCausalLM``.
    ``qwen3`` also registers ``Qwen3MoeForCausalLM`` in a second call.
    ``replace(base, name=...)`` registers names too.
    """
    mapping = max_arch_paths.list_native_arch_mapping(_ARCHS)
    assert {"qwen3", "qwen3_embedding"} <= set(mapping["Qwen3ForCausalLM"])
    assert mapping.get("Qwen3MoeForCausalLM") == ["qwen3"]
    assert mapping.get("Eagle3MHADeepseekV3DraftModel") == ["eagle3_deepseekV3"]
    assert mapping.get("MiMoV2DFlashContextForCausalLM") == ["dflash_mimo_v2"]


@needs_tree
@pytest.mark.parametrize(
    ("slug", "root_name", "expected"),
    [
        ("llama3_modulev3", "PagedMemoryPlanner", "PagedMemoryPlanner"),
        # Call form: the reservation and its arguments must survive.
        ("gemma3multimodal_modulev3", "PagedMemoryPlanner", "with_activation"),
        # Per-architecture planner imported from another donor package.
        ("deepseekV3_modulev3", "DeepseekV3MemoryPlanner", "DeepseekV3"),
    ],
)
def test_planner_resolution(slug: str, root_name: str, expected: str) -> None:
    """A donor's ``memory_planner`` carries over with its arguments.

    Eight architectures configure their planner with the Call form
    ``with_activation_reservation(...)``. A resolver that only understood
    bare names dropped the reservation without a warning.
    """
    planner = _donor(slug).keywords["memory_planner"]
    assert planner.expr is not None and expected in planner.expr
    assert [carried.name for carried in planner.names] == [root_name]


@needs_tree
def test_absent_planner_is_not_invented() -> None:
    """A donor without a planner (an embedding model) gets a TODO only."""
    keywords = _donor("mpnet_modulev3").keywords
    assert "memory_planner" not in keywords
    rendered, _ = _render_arch(keywords, "X")
    # The TODO text names the keyword, so count only non-comment lines.
    assert not [
        line
        for line in rendered.splitlines()
        if "memory_planner=" in line and not line.lstrip().startswith("#")
    ]


@needs_tree
@pytest.mark.parametrize("own_name", [False, True], ids=["Qwen3", "own"])
@pytest.mark.parametrize("slug", [slug for slug, _ in _DONORS])
def test_arch_carries_donor_keywords(
    slug: str, own_name: bool, tmp_path: Path
) -> None:
    """The rendered arch.py sets every keyword the donor's arch.py sets.

    The port writes the keywords in ``PORT_KEYWORDS`` itself. Every other
    keyword carries over from the donor, so serving settings such as
    ``reasoning_parser=`` reach the port. Rendering under the donor's own
    name gives the port's classes the donor's class names, which a carried
    import must not shadow.
    """
    donor = _donor(slug)
    arch_name = (
        donor.name.removesuffix("_ModuleV3") if own_name else "Qwen3ForCausalLM"
    )
    rendered, _ = _render_arch(donor.keywords, arch_name)
    got = _call_keywords(rendered)
    port = templates.PORT_KEYWORDS
    assert set(got) - port == set(donor.keywords) - port
    assert port <= set(got)
    assert len(got) == len(set(got))
    _lint({"arch.py": rendered}, "p", tmp_path)


@needs_tree
@pytest.mark.parametrize("short", ["Qwen3", "NemotronH"])
def test_nemotron_carry(short: str) -> None:
    """Nemotron keeps its parsers, and ``batching=`` reads the port's model.

    The allowlist this replaced dropped both parsers. The donor's
    ``batching=NemotronHModel.batch_processor_cls`` has to name the port's
    class, so a ``batch_processor_cls`` override in the port takes effect.
    """
    rendered, _ = _render_arch(
        _donor("nemotron_h_modulev3").keywords, f"{short}ForCausalLM"
    )
    assert {"reasoning_parser", "tool_parser"} <= set(_call_keywords(rendered))
    assert f"batching={short}Model.batch_processor_cls," in rendered
    assert "nemotron_h_modulev3.model import" not in rendered


_SYNTHETIC_ARCH = """\
from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext as Ctx
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import FakeModel
from .model_config import FakeConfig

_PARSER = "fake"

fake_arch = SupportedArchitecture(
    name="FakeForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["org/fake"],
    pipeline_model=FakeModel,
    config=FakeConfig,
    context_type=Ctx,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert,
    },
    batching=FakeModel.batch_processor_cls,
    tool_parser=_PARSER,
    required_arguments={k: False for k in ("enable_chunked_prefill",)},
    rope_adapter=weight_adapters.rope_fix,
)

fake_embed_arch = SupportedArchitecture(
    name="FakeForCausalLM",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=["org/fake"],
    pipeline_model=MissingModel,
    required_arguments={"enable_prefix_caching": False},
)
"""

_SYNTHETIC_MODEL = """\
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache


class FakeModel(ModuleV3PipelineModelWithKVCache):
    pass
"""


def test_carried_names_resolve(tmp_path: Path) -> None:
    """Carried keywords keep aliases, stay in one call, and avoid clashes.

    A synthetic donor covers what no in-tree ModuleV3 donor does yet: an
    aliased import, a module-level constant, a comprehension variable, a
    name the port also binds (``weight_adapters``), and a second
    ``SupportedArchitecture`` call for another task.
    """
    fake = tmp_path / "archs" / "fake"
    fake.mkdir(parents=True)
    (fake / "arch.py").write_text(_SYNTHETIC_ARCH, encoding="utf-8")
    (fake / "model.py").write_text(_SYNTHETIC_MODEL, encoding="utf-8")
    donor = donor_mod.load_donor(fake)
    rendered, warnings = templates.render_arch(
        slug="p",
        arch_name="FakeForCausalLM",
        short="Fake",
        hf_id="org/m",
        donor_keywords=donor.keywords,
    )
    pkg = f"{_ARCH_PKG}.fake"
    for text in (
        "from max.pipelines.context import TextContext as Ctx",
        "context_type=Ctx,",
        f"from {pkg}.arch import _PARSER",
        "tool_parser=_PARSER,",
        'required_arguments={k: False for k in ("enable_chunked_prefill",)},',
        "weight_adapters as donor_weight_adapters,",
        "rope_adapter=donor_weight_adapters.rope_fix,",
        "batching=FakeModel.batch_processor_cls,",
        "task=PipelineTask.TEXT_GENERATION,",
    ):
        assert text in rendered
    assert "enable_prefix_caching" not in rendered
    assert "EMBEDDINGS" not in rendered
    assert f"{pkg}.model import" not in rendered
    assert warnings == []
    _lint({"arch.py": rendered}, "p", tmp_path)


def test_unresolved_keyword_warns() -> None:
    """An unresolved keyword renders a TODO and a warning, never a default.

    ``task`` is required, so a guessed ``PipelineTask.TEXT_GENERATION``
    would register an embedding port as text generation without an error.
    """
    unresolved = donor_mod.DonorKeyword(expr=None)
    rendered, warnings = _render_arch(
        {"task": unresolved, "tool_parser": unresolved}, "X"
    )
    got = _call_keywords(rendered)
    for keyword in ("task", "tool_parser"):
        assert keyword not in got
        assert (
            f"# TODO(port): copy the donor's {keyword}= line here." in rendered
        )
        assert any(f"arch.py sets {keyword}," in w for w in warnings)


@needs_tree
@pytest.mark.parametrize(
    ("slug", "arch_name", "module_path", "model_path"),
    [
        (
            "llama3_modulev3",
            "Qwen3ForCausalLM",
            f"{_ARCH_PKG}.llama3_modulev3.llama3.Llama3",
            f"{_ARCH_PKG}.llama3_modulev3.model.Llama3Model",
        ),
        # Inherits both classes from llama3_modulev3.
        (
            "phi3_modulev3",
            "Phi4ForCausalLM",
            f"{_ARCH_PKG}.llama3_modulev3.llama3.Llama3",
            f"{_ARCH_PKG}.phi3_modulev3.model.Phi3Model",
        ),
    ],
)
def test_scaffold_modulev3_donors(
    slug: str, arch_name: str, module_path: str, model_path: str, tmp_path: Path
) -> None:
    """The skeleton subclasses the donor's root module and pipeline model."""
    port = f"port_{slug.lower()}"
    short = scaffold.short_name(arch_name)
    files = _render_port(slug, arch_name).files
    trees = {name: ast.parse(text) for name, text in files.items()}

    module_base = _class_def(trees[f"{port}.py"], short).bases[0]
    assert isinstance(module_base, ast.Name)
    assert _imports_from(trees[f"{port}.py"])[module_base.id] == module_path

    model_cls = _class_def(trees["model.py"], f"{short}Model")
    model_base = model_cls.bases[0]
    assert isinstance(model_base, ast.Name)
    assert _imports_from(trees["model.py"])[model_base.id] == model_path
    methods = {
        node.name: node
        for node in model_cls.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_build_graph" not in methods
    built = {
        node.func.id
        for node in ast.walk(methods["_instantiate_module"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert short in built

    for expected in (
        f'name="{arch_name}"',
        "memory_planner=PagedMemoryPlanner",
        "supports_overlap_scheduler=False",
        "supports_device_graph_capture=False",
    ):
        assert expected in files["arch.py"]
    _lint(files, port, tmp_path)


@needs_tree
@pytest.mark.parametrize(
    ("slug", "expected", "builds_config"),
    [
        (
            "qwen3_embedding_modulev3",
            ["task=PipelineTask.EMBEDDINGS_GENERATION"],
            False,
        ),
        (
            "mpnet_modulev3",
            [
                "task=PipelineTask.EMBEDDINGS_GENERATION",
                'required_arguments={"enable_prefix_caching": False}',
            ],
            True,
        ),
        (
            "nemotron_h_modulev3",
            [
                "tokenizer=ReasoningTextTokenizer",
                "checkpoints_recurrent_state=True",
            ],
            True,
        ),
        (
            "deepseekV3_modulev3",
            ["requires_max_batch_context_length=True"],
            True,
        ),
    ],
)
def test_scaffold_task_donors(
    slug: str, expected: list[str], builds_config: bool, tmp_path: Path
) -> None:
    """Embedding and reasoning donors keep their registration settings.

    ``qwen3_embedding_modulev3``'s ``_create_model_config`` returns
    ``None``, so its port has no config to re-type.
    """
    files = _render_port(slug, "Qwen3ForCausalLM").files
    for text in expected:
        assert text in files["arch.py"]
    assert ("from_donor(" in files["model.py"]) == builds_config
    _lint(files, f"port_{slug.lower()}", tmp_path)


def test_rope_theta_rewrite() -> None:
    """Hugging Face ``rope_theta`` reads go through ``get_rope_theta()``.

    A MAX config's own ``rope_theta`` field and assignments stay as they are.
    """
    rewrite = templates._RopeThetaRewrite()
    tree = rewrite.visit(
        ast.parse(
            "a = huggingface_config.rope_theta\n"
            "b = self.huggingface_config.rope_theta\n"
            "c = config.rope_theta\n"
            "hf_config.rope_theta = 1.0\n"
        )
    )
    assert rewrite.rewrote
    assert ast.unparse(tree).splitlines() == [
        "a = get_rope_theta(huggingface_config)",
        "b = get_rope_theta(self.huggingface_config)",
        "c = config.rope_theta",
        "hf_config.rope_theta = 1.0",
    ]


@needs_tree
def test_scaffold_reads_rope_theta_with_get_rope_theta(tmp_path: Path) -> None:
    """The Qwen3 embedding donor's ``rope_theta`` read doesn't reach the port.

    The donor reads ``huggingface_config.rope_theta``, which Transformers 5
    moves into ``rope_parameters``.
    """
    files = _render_port("qwen3_embedding_modulev3", "Qwen3ForCausalLM").files
    model = files["model.py"]
    assert "huggingface_config.rope_theta" not in model
    assert "get_rope_theta(huggingface_config)" in model
    imports = _imports_from(ast.parse(model))
    assert imports["get_rope_theta"] == (
        "max.pipelines.lib.pipeline_variants.utils.get_rope_theta"
    )
    _lint(files, "port_qwen3_embedding_modulev3", tmp_path)


def test_format_port_formats_long_lines(tmp_path: Path) -> None:
    """``format_port`` leaves the port passing ``ruff format --check``."""
    beside = Path(sys.executable).parent / "ruff"
    ruff = str(beside) if beside.is_file() else shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff not installed")
    (tmp_path / "model.py").write_text(
        "x = dict(" + ", ".join(f"key_{i}={i}" for i in range(20)) + ")\n"
    )
    assert scaffold.format_port(tmp_path) is None
    check = subprocess.run(
        [ruff, "format", "--check", str(tmp_path)],
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0


def test_format_port_without_ruff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ruff, ``format_port`` says how to format the port by hand."""
    monkeypatch.setattr(scaffold.shutil, "which", lambda _name: None)
    monkeypatch.setattr(scaffold.sys, "executable", str(tmp_path / "python"))
    problem = scaffold.format_port(tmp_path)
    assert problem is not None
    assert f"ruff format {tmp_path}" in problem


@needs_tree
@pytest.mark.parametrize(
    ("slug", "flag"),
    [
        ("qwen3_embedding_modulev3", " --task embeddings_generation"),
        ("llama3_modulev3", ""),
    ],
)
def test_serve_task_flag(slug: str, flag: str) -> None:
    """The printed serve command names a task other than text generation."""
    assert scaffold.serve_task_flag(_donor(slug)) == flag


@needs_tree
def test_config_names_mark_unread_keys() -> None:
    """The donor's config sources mention the keys it reads, not novel ones.

    ``llama3_modulev3`` reads ``hidden_size`` and ``rope_theta``, and ignores
    ERNIE 4.5's ``use_bias``, which ``inspect_hf.py --start-from`` flags.
    """
    names = _donor("llama3_modulev3").config_names
    assert {"hidden_size", "rope_theta", "num_key_value_heads"} <= names
    assert "use_bias" not in names


@needs_tree
def test_multi_module_donor_raises() -> None:
    """A donor with no ``_instantiate_module`` points at ``--full-copy``."""
    multi = next(
        (slug for slug, _ in _DONORS if _donor(slug)._instantiate is None),
        None,
    )
    if multi is None:
        pytest.skip("no multi-module donor in this tree")
    with pytest.raises(donor_mod.DonorError, match="--full-copy"):
        _render_port(multi, "Qwen3ForCausalLM")


@needs_tree
def test_from_donor_carries_fields() -> None:
    """The rendered ``from_donor`` keeps donor state and the port's defaults.

    The stand-in donor config uses ``__slots__`` and sets a field after
    construction the way ``finalize`` does. The port adds a novel field.
    """
    rendered = _render_port("llama3_modulev3", "Llama3ForCausalLM").files[
        "model_config.py"
    ]
    port = _class_def(ast.parse(rendered), "Llama3Config")
    port.body.insert(0, ast.parse("embedding_multiplier: float = 1.0").body[0])
    # The rendered file postpones annotations, and they name the class
    # before its body finishes.
    source = f"from __future__ import annotations\n\n{ast.unparse(port)}\n"

    @dataclasses.dataclass(kw_only=True, slots=True)
    class DonorLlama3Config:
        hidden_size: int
        return_logits: str = "last"

    namespace: dict[str, Any] = dict(
        dataclass=dataclasses.dataclass,
        fields=dataclasses.fields,
        AutoConfig=object,
        DonorLlama3Config=DonorLlama3Config,
    )
    exec(compile(source, "model_config.py", "exec"), namespace)
    donor_config = DonorLlama3Config(hidden_size=8)
    donor_config.return_logits = "all"
    cfg = namespace["Llama3Config"].from_donor(donor_config, object())
    assert (cfg.hidden_size, cfg.return_logits, cfg.embedding_multiplier) == (
        8,
        "all",
        1.0,
    )


@pytest.mark.parametrize(
    "donor_adapter",
    [max_arch_paths.ImportedName("_stub_donor", "convert"), None],
    ids=["delegating", "pass-through"],
)
def test_weight_adapter_forwards_pipeline_kwargs(
    donor_adapter: max_arch_paths.ImportedName | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rendered adapters accept every way the pipeline calls them.

    KV-cache pipelines call ``adapter(state, huggingface_config=...,
    pipeline_config=...)``, and encoder pipelines call ``adapter(state)``.
    Donor adapters take different signatures, so the delegating adapter
    must forward the keywords it receives unchanged.
    """
    calls: list[tuple[Any, dict[str, Any]]] = []

    def donor_convert(state_dict: Any, **kwargs: Any) -> Any:
        calls.append((state_dict, kwargs))
        return state_dict

    weights = types.ModuleType("max.graph.weights")
    weights.WeightData = object  # type: ignore[attr-defined]
    weights.Weights = object  # type: ignore[attr-defined]
    stub = types.ModuleType("_stub_donor")
    stub.convert = donor_convert  # type: ignore[attr-defined]
    for name, module in {
        "max": types.ModuleType("max"),
        "max.graph": types.ModuleType("max.graph"),
        "max.graph.weights": weights,
        "_stub_donor": stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source, warnings = templates.render_weight_adapter(
        donor_slug="stub", donor_adapter=donor_adapter
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, "weight_adapters.py", "exec"), namespace)
    convert = namespace["convert_safetensor_state_dict"]
    convert({}, huggingface_config="hf", pipeline_config="pc")
    convert({})
    if donor_adapter is None:
        assert len(warnings) == 1 and "pass-through" in warnings[0]
    else:
        assert warnings == []
        assert calls == [
            ({}, {"huggingface_config": "hf", "pipeline_config": "pc"}),
            ({}, {}),
        ]


_SNAPSHOT_CASES = [
    ("llama3_modulev3", "Qwen3ForCausalLM"),
    ("phi3_modulev3", "Phi4ForCausalLM"),
]


@needs_tree
@pytest.mark.parametrize(("slug", "arch_name"), _SNAPSHOT_CASES)
def test_render_snapshot(
    slug: str, arch_name: str, request: pytest.FixtureRequest
) -> None:
    """Compare each rendered skeleton with ``<snapshot-dir>/<donor>-<arch>/``.

    With ``--update-snapshot``, write the snapshot instead.
    """
    snapshot_dir = request.config.getoption("--snapshot-dir")
    if snapshot_dir is None:
        pytest.skip("pass --snapshot-dir to compare rendered skeletons")
    case = snapshot_dir / f"{slug}-{arch_name}"
    files = _render_port(slug, arch_name).files
    if request.config.getoption("--update-snapshot"):
        case.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (case / name).write_text(content, encoding="utf-8")
        return
    saved = {p.name: p.read_text(encoding="utf-8") for p in case.glob("*")}
    diff = "".join(
        line
        for name in sorted(saved.keys() | files.keys())
        for line in difflib.unified_diff(
            saved.get(name, "").splitlines(keepends=True),
            files.get(name, "").splitlines(keepends=True),
            f"snapshot/{name}",
            f"rendered/{name}",
        )
    )
    assert not diff, diff


def test_write_port_leaves_nothing_on_failure(tmp_path: Path) -> None:
    """A failed write leaves neither the port nor a temporary directory."""

    def fail(tmp: Path) -> None:
        (tmp / "arch.py").write_text("partial", encoding="utf-8")
        raise RuntimeError("halfway")

    dst = tmp_path / "ports" / "my_port"
    with pytest.raises(RuntimeError, match="halfway"):
        scaffold.write_port(dst, fail)
    assert list((tmp_path / "ports").iterdir()) == []

    def succeed(tmp: Path) -> None:
        (tmp / "arch.py").write_text("done", encoding="utf-8")

    scaffold.write_port(dst, succeed)
    assert (dst / "arch.py").read_text(encoding="utf-8") == "done"


def test_slug_from_hf_id() -> None:
    assert scaffold.slug_from_hf_id("mistralai/Mistral-7B") == "mistral_7b"
    # A digit-leading repo name is not a Python identifier on its own.
    assert scaffold.slug_from_hf_id("org/1b-llama") == "_1b_llama"


@needs_tree
@pytest.mark.parametrize(
    ("slug", "kept"),
    [
        # Root module in graph.py: a blanket rename of ``graph`` would
        # rewrite ``max.graph`` imports.
        ("mpnet_modulev3", "from max.graph import"),
        # Root module in llama3.py, and compares ``rope_type == "llama3"``.
        ("llama3_modulev3", '"llama3"'),
    ],
)
def test_full_copy(slug: str, kept: str, tmp_path: Path) -> None:
    """``--full-copy`` renames only the port's own module and exports it."""
    donor = _donor(slug)
    port = f"port_{slug.lower()}"
    dst = tmp_path / port
    dst.mkdir()
    written = full_copy.write_full_copy(
        donor=donor, dst=dst, slug=port, arch_name="Qwen3ForCausalLM"
    )
    assert full_copy.export_architectures(dst, "Qwen3ForCausalLM")
    texts = {
        p.relative_to(dst).as_posix(): p.read_text() for p in dst.rglob("*.py")
    }
    assert Path(f"{port}.py") in written
    joined = "\n".join(texts.values())
    assert kept in joined
    assert f"from .{donor.module_stem} import" not in joined
    assert f"max.{port}" not in joined
    assert "ARCHITECTURES = [" in texts["__init__.py"]
    assert run_oss_gates._parse_arch_py(dst)[0] == "Qwen3ForCausalLM"


def test_rename_registration_targets_one_call() -> None:
    text = (
        'a = SupportedArchitecture(\n    name="Old",\n)\n'
        'b = SupportedArchitecture(name="Other")\n'
    )
    renamed = full_copy.rename_registration(text, "Old", "New")
    assert 'name="New"' in renamed and 'name="Other"' in renamed
    assert full_copy.rename_registration(text, "Missing", "New") == text


def test_export_architectures_includes_replace_derived(tmp_path: Path) -> None:
    """A ``dataclasses.replace`` copy registers too, and reaches ARCHITECTURES."""
    (tmp_path / "arch.py").write_text(
        "import dataclasses\n\n"
        "a_arch = SupportedArchitecture(name='A')\n"
        "b_arch = dataclasses.replace(a_arch, name='B')\n",
        encoding="utf-8",
    )
    assert full_copy.export_architectures(tmp_path, "B")
    init = (tmp_path / "__init__.py").read_text(encoding="utf-8")
    assert "from .arch import b_arch" in init
    assert "ARCHITECTURES = [b_arch]" in init


def test_export_architectures_skips_the_donor_s_other_registration(
    tmp_path: Path,
) -> None:
    """Only the port's renamed registration reaches ``ARCHITECTURES``.

    ``--custom-architectures`` registers every list entry with
    ``allow_override=True``, so a donor's unrenamed registration would
    shadow its built-in architecture.
    """
    (tmp_path / "arch.py").write_text(
        "a_arch = SupportedArchitecture(name='A')\n"
        "b_arch = SupportedArchitecture(name='B')\n",
        encoding="utf-8",
    )
    assert full_copy.export_architectures(tmp_path, "A")
    init = (tmp_path / "__init__.py").read_text(encoding="utf-8")
    assert "from .arch import a_arch" in init
    assert "ARCHITECTURES = [a_arch]" in init
    assert "b_arch" not in init


def test_build_lazily_runs_the_distributed_runtime_hook() -> None:
    """The build runs ``_init_distributed_runtime()`` before the module build.

    A donor can set what ``_instantiate_module()`` reads in that hook
    (deepseekV3_modulev3 sets its expert-parallel batch manager there), so
    the hook must run in the same order ``load_model()`` runs it.
    """
    from max.dtype import DType

    class Stub:
        def _load_state_dict(self) -> dict:
            return {}

        def _create_model_config(self, state_dict: dict) -> dict:
            return {}

        def _prepare_state_dict(
            self, state_dict: dict, model_config: dict
        ) -> dict:
            return state_dict

        def _module_default_dtype(
            self, state_dict: dict, model_config: dict
        ) -> DType:
            return DType.float32

        def _instantiate_module(self, model_config: dict) -> str:
            return "built"

    class WithHook(Stub):
        def _init_distributed_runtime(self, model_config: dict) -> None:
            self.ep_batch_manager = None

        def _instantiate_module(self, model_config: dict) -> str:
            assert self.ep_batch_manager is None
            return "built"

    with pytest.raises(check_port._Built) as built:
        check_port._build_lazily(WithHook())
    assert built.value.module == "built"

    # A donor outside the ModuleV3 base defines no hook; the build
    # still runs.
    with pytest.raises(check_port._Built) as built:
        check_port._build_lazily(Stub())
    assert built.value.module == "built"


@pytest.mark.parametrize(
    ("hf_text", "mx_text", "expected"),
    [
        ("\n", "\n", True),
        ("\n", "", False),
        (" Paris", "Paris", True),
        (" Paris", " London", False),
        (" ", "\n", False),
    ],
)
def test_compare_layers_token_text(
    hf_text: str, mx_text: str, expected: bool
) -> None:
    """The probe compares whitespace tokens and tolerates leading spaces.

    The server returns raw completion text. A ``"\\n"`` prediction must
    match HF's ``"\\n"``, and a leading-space difference between the
    tokenizer's decode and the server's text must not count as a mismatch.
    """
    assert compare_layers.same_token_text(hf_text, mx_text) is expected


def _http_error(body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://localhost:8000/v1/completions",
        400,
        "Bad Request",
        email.message.Message(),
        io.BytesIO(body),
    )


def test_server_error_reports_the_server_message() -> None:
    """A 400 shows MAX's own message, plus the flags the overlap case needs.

    MAX ignores ``--no-enable-overlap-scheduler`` without ``--force``, and
    its message names only the first flag.
    """
    overlap = _http_error(
        b'{"error": {"code": "400", "message": "Log probabilities are not '
        b'supported with the overlap scheduler."}}'
    )
    message = compare_layers.server_error(overlap)
    assert message.startswith("HTTP 400: Log probabilities")
    assert "--no-enable-overlap-scheduler --force" in message
    assert compare_layers.server_error(_http_error(b"<html>")) == (
        "HTTP 400 Bad Request"
    )


def test_verify_gate_fails_on_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verify gate reports a rejected request as a failed gate."""

    def reject(*args: Any, **kwargs: Any) -> None:
        raise _http_error(b'{"error": {"message": "model not found"}}')

    monkeypatch.setattr(run_oss_gates, "probe_multi_position", reject)
    result = run_oss_gates.gate_verify_logprobs(
        "org/m", "org/m", 8000, "float32"
    )
    assert result.status == "FAIL"
    assert result.detail == "HTTP 400: model not found"


def test_canonical_native_dtype() -> None:
    assert run_oss_gates.canonical_native_dtype("BF16") == "bfloat16"
    # A Hub string that names no concrete dtype must not pass through.
    assert run_oss_gates.canonical_native_dtype("auto") is None
    # An unknown concrete dtype stays so the gate can report it.
    assert run_oss_gates.canonical_native_dtype("float64") == "float64"


def test_gate_encoding_skips_abstract_hub_dtype(tmp_path: Path) -> None:
    """``torch_dtype: "auto"`` names no dtype, so the encoding gate abstains."""
    (tmp_path / "arch.py").write_text(
        'a = SupportedArchitecture(name="X", default_encoding="bfloat16")\n',
        encoding="utf-8",
    )
    result = run_oss_gates.gate_encoding({"torch_dtype": "auto"}, tmp_path)
    assert result.status == "SKIP"
    assert "auto" in result.detail


@needs_tree
def test_gates_read_rendered_arch(tmp_path: Path) -> None:
    """The preflight gates read the name and encoding the skeleton writes."""
    files = _render_port("llama3_modulev3", "Qwen3ForCausalLM").files
    (tmp_path / "arch.py").write_text(files["arch.py"], encoding="utf-8")
    assert run_oss_gates._parse_arch_py(tmp_path) == (
        "Qwen3ForCausalLM",
        "bfloat16",
    )


def _run(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, *args], cwd=_SCRIPT_DIR, capture_output=True, text=True
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-2000:]
    return out


@pytest.fixture
def hf_id(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--hf-id")


@pytest.mark.smoke
def test_list_native_archs_smoke() -> None:
    out = _run("list_native_archs.py", "--match", "Qwen3ForCausalLM")
    assert "Qwen3ForCausalLM\tqwen3\n" in out
    assert "Qwen3ForCausalLM\tqwen3_embedding\n" in out


_HUB_SMOKE: list[tuple[str, Callable[[str], list[str]], str]] = [
    ("check_walls", lambda hf: ["check_walls.py", hf], ""),
    (
        "list_checkpoint_keys",
        lambda hf: ["list_checkpoint_keys.py", hf, "--summary"],
        "dominant_dtype",
    ),
    ("inspect_hf", lambda hf: ["inspect_hf.py", hf], "HF inspection"),
    ("compare_layers", lambda hf: ["compare_layers.py", "--help"], "usage"),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("args", "expect"),
    [(args, expect) for _, args, expect in _HUB_SMOKE],
    ids=[name for name, _, _ in _HUB_SMOKE],
)
def test_hub_scripts_smoke(
    args: Callable[[str], list[str]], expect: str, hf_id: str
) -> None:
    assert expect in _run(*args(hf_id))


@pytest.mark.smoke
def test_scaffold_smoke(hf_id: str, tmp_path: Path) -> None:
    """Scaffold a port from a donor, then run the preflight gates on it."""
    out = _run(
        "scaffold.py",
        hf_id,
        "--start-from",
        "llama3_modulev3",
        "--output-dir",
        str(tmp_path),
        "--slug",
        "test_port",
    )
    assert "Scaffold created" in out
    port = tmp_path / "test_port"
    assert "memory_planner=PagedMemoryPlanner" in (port / "arch.py").read_text()
    _run("run_oss_gates.py", hf_id, "--port-dir", str(port))
    report = _run("check_port.py", hf_id, "--port-dir", str(port))
    assert "missing (compile raises KeyError): 0" in report
