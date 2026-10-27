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
"""Check a port's parameters against its adapted checkpoint, without compiling.

Imports the port the way ``max serve --custom-architectures`` does, builds
the pipeline that ``max serve`` would build, and stops in ``load_model()``
right after the port constructs its root module under ``F.lazy()``. It then
compares the module's parameters with the tensors the port's weight adapter
returns for the real checkpoint:

- missing: a parameter with no tensor. ``compile()`` raises ``KeyError``.
- shape or dtype mismatch: ``compile()`` raises ``ValueError``.
- unconsumed: a tensor no parameter reads. ``compile()`` ignores it, which
  hides a wrong rename. See references/state-dict-audit.md.

Exits 1 when a parameter is missing or mismatched. Runs in seconds, so run
it before every ``max serve``.

Usage::

    pixi run python scripts/check_port.py <HF_MODEL_ID> --port-dir <port_dir>
    pixi run python scripts/check_port.py <HF_MODEL_ID> --port-dir <port_dir> \\
        --task embeddings_generation
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from max import functional as F
from max.dtype import DType
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.lib.pipeline_args import PipelineArgs
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.types import PipelineTask
from max.tensor import default_dtype

_CASTABLE = frozenset({DType.float32, DType.bfloat16})


class _Built(BaseException):
    """Carries the lazily built module out of ``load_model()``.

    A ``BaseException``, so a pipeline's ``except Exception`` cleanup can't
    swallow it.
    """

    def __init__(self, module: Any, state_dict: dict[str, Any]) -> None:
        super().__init__("module built")
        self.module = module
        self.state_dict = state_dict


def _build_lazily(self: Any) -> Any:
    """Runs ``load_model()``'s steps up to the module build, then stops."""
    state_dict = self._load_state_dict()
    model_config = self._create_model_config(state_dict)
    state_dict = self._prepare_state_dict(state_dict, model_config)
    # A donor can set what _instantiate_module reads (an expert-parallel
    # batch manager) in this hook; load_model() runs it before the build.
    init_distributed = getattr(self, "_init_distributed_runtime", None)
    if init_distributed is not None:
        init_distributed(model_config)
    dtype = self._module_default_dtype(state_dict, model_config)
    with F.lazy(), default_dtype(dtype):
        module = self._instantiate_module(model_config)
    raise _Built(module, state_dict)


def _import_port(port_dir: Path) -> Any:
    """Imports the port as ``max serve`` does: parent on sys.path, slug as module."""
    sys.path.insert(0, str(port_dir.parent))
    package = importlib.import_module(port_dir.name)
    archs = getattr(package, "ARCHITECTURES", None)
    if not archs:
        sys.exit(f"{port_dir.name}/__init__.py exposes no ARCHITECTURES list")
    return archs


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(d) for d in tensor.shape)


def _report(module: Any, state_dict: dict[str, Any]) -> int:
    params = dict(module.parameters)
    missing = sorted(set(params) - set(state_dict))
    unconsumed = sorted(set(state_dict) - set(params))
    shapes, dtypes = [], []
    for name in sorted(set(params) & set(state_dict)):
        param, tensor = params[name], state_dict[name]
        if _shape(param) != _shape(tensor):
            shapes.append(
                f"{name}: parameter {_shape(param)}, "
                f"checkpoint {_shape(tensor)}"
            )
        elif param.dtype != tensor.dtype:
            dtypes.append((name, tensor.dtype, param.dtype))

    print(
        f"module built: {len(params)} parameters, "
        f"{len(state_dict)} adapted tensors"
    )
    for label, names in (
        ("missing (compile raises KeyError)", missing),
        ("shape mismatch (compile raises ValueError)", shapes),
        ("unconsumed (compile ignores them)", unconsumed),
    ):
        print(f"{label}: {len(names)}")
        for name in names[:20]:
            print(f"  {name}")
    print(f"dtype mismatch (compile raises ValueError): {len(dtypes)}")
    for name, have, want in dtypes[:20]:
        print(f"  {name}: checkpoint {have}, parameter {want}")
    if any({have, want} == _CASTABLE for _, have, want in dtypes):
        print(
            "  Cast float32 and bfloat16 tensors to the parameter's dtype in "
            "weight_adapters.py (references/pitfalls-weights.md)."
        )
    return 1 if missing or shapes or dtypes else 0


def main(args: argparse.Namespace) -> int:
    port_dir = args.port_dir.resolve()
    archs = _import_port(port_dir)
    print(f"import OK: {port_dir.name} registers {[a.name for a in archs]}")
    # Patch the class that defines load_model(), usually a base class in
    # max.pipelines.lib. MAX can import the port package again under its own
    # loader, which makes new port classes but reuses those bases.
    for arch in archs:
        owner = next(
            cls
            for cls in arch.pipeline_model.__mro__
            if "load_model" in vars(cls)
        )
        owner.load_model = _build_lazily

    # The same steps `max serve` takes: resolve the task by architecture name
    # unless --task names it, then build the pipeline from the factory.
    pipeline_args = PipelineArgs(
        model_path=args.hf_id,
        max_length=args.max_length,
        runtime=PipelineRuntimeConfig(
            custom_architectures=[str(port_dir)], max_batch_size=1
        ),
        **({"task": args.task} if args.task else {}),
    )
    PIPELINE_REGISTRY._import_custom_architectures([str(port_dir)])
    if pipeline_args.task == PipelineTask.UNDEFINED:
        pipeline_args = pipeline_args.model_copy(
            update={
                "task": PIPELINE_REGISTRY.retrieve_pipeline_task(
                    pipeline_args.main_architecture_name
                )
            }
        )
    print(f"task: {pipeline_args.task.value}")
    config = PipelineConfig.from_args(pipeline_args)
    retrieved = PIPELINE_REGISTRY.retrieve_factory(
        config, task=pipeline_args.task
    )
    try:
        retrieved.factory()
    except _Built as built:
        return _report(built.module, built.state_dict)
    print(
        "MAX built the pipeline without calling the port's load_model(). "
        "Check that arch.py's name matches config.json architectures[0], and "
        "pass --task when the port serves a task other than the one the "
        "name resolves to."
    )
    return 1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id", help="Hugging Face model ID or local path")
    parser.add_argument(
        "--port-dir",
        type=Path,
        required=True,
        help="The port's package directory, as passed to --custom-architectures",
    )
    parser.add_argument(
        "--task",
        help="Pipeline task, as passed to max serve --task "
        "(for example embeddings_generation)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Sequence length for the config (default: 512)",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()))
