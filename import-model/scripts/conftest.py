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
"""Options and markers for ``test_scripts.py``."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hf-id",
        default="Qwen/Qwen3-1.7B",
        help="Hub model the smoke tests inspect and scaffold",
    )
    parser.addoption(
        "--gemma3-hf-id",
        default="unsloth/gemma-3-270m-it",
        help="Ungated Gemma 3 checkpoint the registered-donor smoke test "
        "scaffolds from gemma3_modulev3",
    )
    parser.addoption(
        "--snapshot-dir",
        type=Path,
        help="Compare each rendered skeleton with <dir>/<donor>-<arch>/",
    )
    parser.addoption(
        "--update-snapshot",
        action="store_true",
        help="Write the rendered skeletons to --snapshot-dir",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: runs a script as a subprocess; needs MAX and the Hub",
    )
