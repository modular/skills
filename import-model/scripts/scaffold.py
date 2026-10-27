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
"""Generate a custom-arch port skeleton by subclassing a ModuleV3 donor.

Donors are the ModuleV3 architectures under ``max.pipelines.architectures``
(``olmo3``, ``llama3_modulev3``, ``gemma3_modulev3``, ...). List them with
``list_native_archs.py --donors``.

Default mode (recommended): produces a small subclass skeleton that inherits
the donor's behavior and exposes the deltas a port typically overrides:

- ``model_config.py``: the donor's config plus the port's novel HF fields.
- ``<slug>.py``: the port's root ``max.nn.Module``, a subclass
  of the donor's root module. Override ``__init__`` and ``forward()``.
- ``model.py``: the donor's pipeline model with ``_instantiate_module``
  building the port's root module.
- ``weight_adapters.py``: delegates to the donor's safetensors adapter.

After scaffold, edit only the spots marked ``TODO`` for the deltas your
model has.

``--full-copy`` mode: copy every donor file verbatim and sed-rename
slug/class references. Use it only when the port needs a module tree with a
different shape from every donor (Gemma3, Llama4). Most ports don't.

Usage:
    pixi run python scaffold.py <HF_MODEL_ID> --start-from olmo3 --output-dir <output_dir>
    pixi run python scaffold.py <HF_MODEL_ID> --start-from llama3_modulev3 --output-dir <output_dir>
    pixi run python scaffold.py <HF_MODEL_ID> --start-from olmo3 --output-dir <output_dir> --full-copy
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

try:
    from .donor import Donor, DonorError, load_donor
    from .full_copy import export_architectures, write_full_copy
    from .hub_config import architecture_class, load_hub_config
    from .max_arch_paths import find_arch_dir
    from .templates import render_skeleton
except ImportError:
    # Standalone invocation: `python /path/to/scaffold.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from donor import (  # type: ignore[no-redef]
        Donor,
        DonorError,
        load_donor,
    )
    from full_copy import (  # type: ignore[no-redef]
        export_architectures,
        write_full_copy,
    )
    from hub_config import (  # type: ignore[no-redef]
        architecture_class,
        load_hub_config,
    )
    from max_arch_paths import find_arch_dir  # type: ignore[no-redef]
    from templates import render_skeleton  # type: ignore[no-redef]


def slug_from_hf_id(hf_id: str) -> str:
    """Dir and module name for the port, from the Hub repo id.

    A repo name starting with a digit would make an invalid Python
    identifier (``_arch`` import, ``<slug>.py`` module), so the slug gets
    a ``_`` prefix.
    """
    slug = hf_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    if slug[0].isdigit():
        slug = "_" + slug
    return slug


def short_name(arch_name: str) -> str:
    """Strip common HF arch suffixes to get a short camel name.

    ``MiniCPMForCausalLM`` -> ``MiniCPM``
    ``LlamaForCausalLM``   -> ``Llama``
    ``Phi3ForCausalLM``    -> ``Phi3``
    """
    for suffix in (
        "ForCausalLM",
        "ForConditionalGeneration",
        "ForSequenceClassification",
        "Model",
    ):
        if arch_name.endswith(suffix):
            return arch_name[: -len(suffix)]
    return arch_name


def write_port(dst: Path, write: Callable[[Path], None]) -> None:
    """Run ``write`` on a temporary sibling of ``dst``, then move it to ``dst``.

    A failure partway through leaves no ``dst`` behind, so rerunning the
    scaffold doesn't stop at "Output already exists".
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
    try:
        write(tmp)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def format_port(dst: Path) -> str | None:
    """Format the port with ``ruff format``. Returns why it couldn't, if so.

    Scaffolded bodies come from ``ast.unparse``, which writes each statement
    on one line, so the port fails ``ruff format --check`` until formatted.
    """
    # Prefer the ruff installed next to this interpreter, the one the
    # skill's pixi environment pins, over any other ruff on PATH.
    beside = Path(sys.executable).parent / "ruff"
    ruff = str(beside) if beside.is_file() else shutil.which("ruff")
    if ruff is None:
        return (
            f"ruff is not installed, so the port is unformatted. Install it "
            f"and run `ruff format {dst}` before the smoke gate."
        )
    proc = subprocess.run(
        [ruff, "format", "--quiet", str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return f"`ruff format {dst}` failed:\n{proc.stderr.strip()}"
    return None


def serve_task_flag(donor: Donor) -> str:
    """The ``--task`` flag ``max serve`` needs for a port of ``donor``.

    ``max serve`` picks text generation for an architecture name that more
    than one task registers, such as ``Qwen3ForCausalLM``, so a port for
    another task has to name its own.
    """
    task = donor.keywords.get("task")
    expr = task.expr if task is not None else None
    if expr is None or not expr.startswith("PipelineTask."):
        return ""
    name = expr.removeprefix("PipelineTask.")
    return "" if name == "TEXT_GENERATION" else f" --task {name.lower()}"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id", help="HuggingFace model ID")
    parser.add_argument(
        "--start-from",
        required=True,
        help="ModuleV3 donor slug under max.pipelines.architectures "
        "(e.g. olmo3, llama3_modulev3). List them with "
        "list_native_archs.py --donors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to create the new arch under",
    )
    parser.add_argument(
        "--slug",
        help="Slug for the new arch (default: derived from HF model ID)",
    )
    parser.add_argument(
        "--arch-name",
        help="HF architectures[0] for arch.py::name (default: from config.json)",
    )
    parser.add_argument(
        "--full-copy",
        action="store_true",
        help="Copy every donor file verbatim and sed-rename. Use only if the "
        "port needs a module tree no donor has (Gemma3, Llama4 shapes).",
    )


def main(args: argparse.Namespace) -> int:
    cfg = load_hub_config(args.hf_id)
    try:
        arch_from_config = architecture_class(cfg)
    except ValueError:
        arch_from_config = None
    arch_name: str | None = args.arch_name or arch_from_config
    if not arch_name:
        sys.exit(
            f"{args.hf_id} config.json has no architectures[0]. Pass --arch-name explicitly."
        )

    slug = args.slug or slug_from_hf_id(args.hf_id)

    src = find_arch_dir(args.start_from)
    dst = args.output_dir / slug
    if dst.exists():
        sys.exit(f"Output already exists: {dst}")
    try:
        donor = load_donor(src)
        if args.full_copy:
            print(f"Copying {src} -> {dst}  (--full-copy mode)")
            progress: list[str] = []

            def write(tmp: Path) -> None:
                copied = write_full_copy(
                    donor=donor, dst=tmp, slug=slug, arch_name=arch_name
                )
                progress.extend(f"wrote {path}" for path in copied)
                if export_architectures(tmp, arch_name):
                    progress.append("added ARCHITECTURES to __init__.py")

            write_port(dst, write)
            for line in progress:
                print(f"  {line}")
        else:
            # Rendering resolves the donor's root module and config class,
            # which raises DonorError for multi-module donors before
            # anything is written.
            skeleton = render_skeleton(
                slug=slug,
                short=short_name(arch_name),
                arch_name=arch_name,
                hf_id=args.hf_id,
                donor=donor,
            )

            def write(tmp: Path) -> None:
                for name, content in skeleton.files.items():
                    (tmp / name).write_text(content)

            write_port(dst, write)
            print(f"Scaffolding subclass skeleton at {dst}")
            print(
                f"  donor classes: module={donor.module.name}, "
                f"model={donor.model.name}, config={donor.config.name}"
            )
            for name in skeleton.files:
                print(f"  wrote {name}")
            for warning in skeleton.warnings:
                print(f"WARNING: {warning}", file=sys.stderr)
    except DonorError as exc:
        sys.exit(str(exc))

    if (problem := format_port(dst)) is not None:
        print(f"WARNING: {problem}", file=sys.stderr)

    print()
    print(f"Scaffold created at: {dst}")
    print()
    print("Next:")
    if args.full_copy:
        print(f"  1. Open {dst}/model_config.py: wire every config.json key")
        print(f"  2. Open {dst}/weight_adapters.py: map your checkpoint keys")
        module_file = dst / f"{slug}.py"
        if module_file.is_file():
            print(
                f"  3. Open {module_file}: implement every delta in the "
                f"root module's __init__ and forward()"
            )
        else:
            print(
                f"  3. The donor reuses another architecture's root module. "
                f"Add {slug}.py with the port's Module and build it in "
                f"{dst}/model.py::_instantiate_module"
            )
        print(
            f"  4. Open {dst}/arch.py: verify name={arch_name!r} and default_encoding"
        )
    else:
        print(
            "  1. Inspect the HF reference modeling code. Record deltas vs the donor."
        )
        print(
            f"  2. Open {dst}/model_config.py: add novel HF fields and set them in from_donor"
        )
        print(
            f"  3. Open {dst}/{slug}.py: override __init__ / forward() for structural deltas"
        )
        print(
            f"  4. Open {dst}/weight_adapters.py: add rewrites if HF keys differ"
        )
        print(f"  5. Open {dst}/arch.py: verify encoding, repo_ids")
    print()
    print("Don't serve until the smoke gate passes.")
    print(f"  port_dir: {dst.resolve()}")
    print("Then:")
    print(
        f"  pixi run max serve --model-path {args.hf_id} "
        f"--custom-architectures {dst.resolve()}{serve_task_flag(donor)}"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
