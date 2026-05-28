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
"""Fetch HuggingFace safetensors metadata without downloading weights."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_SAFETENSORS_TO_TORCH = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


@dataclass(frozen=True)
class TensorMeta:
    """One tensor entry from a Hub safetensors repo."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    parameter_count: int


@dataclass(frozen=True)
class RepoTensorSummary:
    """Aggregated safetensors metadata for a Hub model repo."""

    hf_id: str
    sharded: bool
    tensors: tuple[TensorMeta, ...]
    dtype_counts: dict[str, int]

    @property
    def total_parameters(self) -> int:
        return sum(t.parameter_count for t in self.tensors)

    def dominant_dtype(self) -> str | None:
        if not self.dtype_counts:
            return None
        top, _ = max(self.dtype_counts.items(), key=lambda kv: kv[1])
        return top


def _torch_dtype(safetensors_dtype: str) -> str:
    return _SAFETENSORS_TO_TORCH.get(
        safetensors_dtype, safetensors_dtype.lower()
    )


def fetch_repo_tensors(hf_id: str) -> RepoTensorSummary:
    """Return tensor names, shapes, and dtypes from Hub safetensors metadata."""
    try:
        from huggingface_hub import get_safetensors_metadata
        from huggingface_hub.errors import NotASafetensorsRepoError
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub to fetch safetensors metadata"
        ) from exc

    try:
        meta = get_safetensors_metadata(hf_id)
    except NotASafetensorsRepoError as exc:
        raise ValueError(f"{hf_id!r} is not a safetensors repo: {exc}") from exc

    tensors: list[TensorMeta] = []
    for file_meta in meta.files_metadata.values():
        for name, info in file_meta.tensors.items():
            tensors.append(
                TensorMeta(
                    name=name,
                    dtype=_torch_dtype(info.dtype),
                    shape=tuple(int(x) for x in info.shape),
                    parameter_count=int(info.parameter_count),
                )
            )
    tensors.sort(key=lambda t: t.name)
    counts = Counter(t.dtype for t in tensors)
    return RepoTensorSummary(
        hf_id=hf_id,
        sharded=bool(meta.sharded),
        tensors=tuple(tensors),
        dtype_counts=dict(counts),
    )


def filter_tensors(
    tensors: Iterable[TensorMeta],
    *,
    prefix: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[TensorMeta]:
    """Return tensors whose names match ``prefix`` and none of ``exclude_prefixes``."""
    out: list[TensorMeta] = []
    for t in tensors:
        if prefix and not t.name.startswith(prefix):
            continue
        if any(t.name.startswith(p) for p in exclude_prefixes):
            continue
        out.append(t)
    return out


def diff_key_sets(
    hf_names: Iterable[str],
    max_names: Iterable[str],
    *,
    limit: int = 30,
) -> tuple[list[str], list[str]]:
    """Return (hf_only, max_only) key lists, truncated to ``limit`` each."""
    hf_set, max_set = set(hf_names), set(max_names)
    hf_only = sorted(hf_set - max_set)[:limit]
    max_only = sorted(max_set - hf_set)[:limit]
    return hf_only, max_only
