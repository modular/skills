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
"""Load raw ``config.json`` from the Hugging Face Hub."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_hub_config(hf_id: str) -> dict:
    """Return the repo's ``config.json`` as a plain dict."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Install huggingface_hub: pip install huggingface-hub")

    path = hf_hub_download(repo_id=hf_id, filename="config.json")
    return json.loads(Path(path).read_text())


def hub_config_url(hf_id: str, revision: str = "main") -> str:
    """Browser/raw URL for the repo's config.json."""
    return f"https://huggingface.co/{hf_id}/raw/{revision}/config.json"


def architecture_class(cfg: dict) -> str:
    """Return ``architectures[0]`` or raise if missing."""
    archs = cfg.get("architectures") or []
    if not archs:
        raise ValueError("config.json has no architectures[0]")
    return archs[0]
