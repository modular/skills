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
"""Map HuggingFace ``config.json`` dtype fields to MAX ``default_encoding`` strings."""

from __future__ import annotations

# Hub config value → ``SupportedArchitecture.default_encoding`` in arch.py
_HUB_TO_ENCODING: dict[str, str] = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "torch.float32": "float32",
}


def _normalize_hub_dtype(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    return s


def encoding_from_config_dict(cfg: dict) -> str | None:
    """Return the Hub checkpoint dtype from top-level or ``text_config``, if set."""
    for container in (
        cfg,
        cfg.get("text_config")
        if isinstance(cfg.get("text_config"), dict)
        else {},
    ):
        if not isinstance(container, dict):
            continue
        for key in ("torch_dtype", "dtype"):
            normalized = _normalize_hub_dtype(container.get(key))
            if normalized is not None:
                return normalized
    return None


def canonical_native_dtype(hub_dtype: str) -> str:
    """Map a Hub dtype string to the ``default_encoding`` value used in ``arch.py``."""
    key = hub_dtype.strip().lower()
    return _HUB_TO_ENCODING.get(key, key)
