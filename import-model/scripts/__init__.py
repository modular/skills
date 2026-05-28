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
"""Helper scripts for importing a model into MAX.

This package exposes the individual scripts (``inspect_hf``, ``scaffold``,
``check_walls``, ``list_checkpoint_keys``, ``list_native_archs``,
``run_oss_gates``, ``compare_layers``) and a unified CLI dispatcher in
``import_model``. Each script still works as a standalone entry point
(``pixi run python /path/to/scaffold.py ...``); the package layout is
additive.
"""
