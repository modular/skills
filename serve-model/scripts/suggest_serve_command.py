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
"""Suggest a `max serve` command for a custom architecture.

Reads the architecture package's ``arch.py`` (the ``SupportedArchitecture``) and
the checkpoint's ``config.json`` together, then prints a ready-to-run
``max serve`` command plus the reasoning behind each non-default flag. This is
the one-shot version of "open arch.py, open config.json, decide the flags". Run
it instead of doing that by hand so a serve takes a couple of tool calls, not a
dozen.

It is advisory: it prints a command and notes, it does not launch anything. Read
the notes: they flag the traps (default_encoding that disagrees with the
checkpoint, a name that collides with a built-in, GPU-only encodings, MoE).

Usage:
    python suggest_serve_command.py --custom-architectures PKG_DIR --model HF_ID_OR_PATH
    python suggest_serve_command.py -c PKG_DIR -m PKG_DIR/config.json   # local config

Options:
    --devices D         Override device (cpu | gpu | gpu:0 | gpu:0,1). Default: inferred.
    --port N            Serve port (default 8000; never 8001, metrics collision).
    --max-length N      Cap sequence length (default min(4096, model max)).
    --no-hub            Don't try to fetch config.json from the Hub for an HF id.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Encodings that only run on GPU; picking one forces a GPU device.
GPU_ONLY = {"float8_e4m3fn", "float8_e5m2", "float4_e2m1fnx2", "gptq"}
# torch_dtype string -> MAX encoding.
DTYPE_TO_ENCODING = {
    "float32": "float32",
    "float": "float32",
    "float16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
}
# HF quantization method -> MAX encoding.
QUANT_METHOD_TO_ENCODING = {
    "gptq": "gptq",
    "compressed-tensors": "gptq",
    "fp8": "float8_e4m3fn",
    "fbgemm_fp8": "float8_e4m3fn",
}
# Common built-in arch names. On current nightly a custom arch that reuses
# one of these names silently overrides the built-in registration; on MAX
# 26.5 and earlier the registration is refused outright.
LIKELY_BUILTINS = {
    "Gemma3ForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
}


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def parse_arch(pkg: Path) -> dict:
    """Pull name, default_encoding, supported_encodings from arch.py."""
    text = _read(pkg / "arch.py")
    out: dict = {
        "name": None,
        "default_encoding": None,
        "supported_encodings": set(),
        "chat_template": None,
        "multi_gpu": None,
    }
    m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', text)
    if m:
        out["name"] = m.group(1)
    m = re.search(r'default_encoding\s*=\s*["\']([^"\']+)["\']', text)
    if m:
        out["default_encoding"] = m.group(1)
    m = re.search(r"supported_encodings\s*=\s*\{([^}]*)\}", text, re.DOTALL)
    if m:
        out["supported_encodings"] = set(
            re.findall(r'["\']([^"\']+)["\']', m.group(1))
        )
    m = re.search(r"multi_gpu_supported\s*=\s*(True|False)", text)
    if m:
        out["multi_gpu"] = m.group(1) == "True"
    jinja = list(pkg.glob("*.jinja"))
    if jinja:
        out["chat_template"] = str(jinja[0])
    return out


def load_config(model: str, pkg: Path, use_hub: bool) -> dict | None:
    """Load the checkpoint config.json from a local path, the package dir, or the Hub."""
    p = Path(model)
    if p.name == "config.json" and p.is_file():
        return json.loads(_read(p) or "{}")
    if p.is_dir() and (p / "config.json").is_file():
        return json.loads(_read(p / "config.json") or "{}")
    if (pkg / "config.json").is_file():
        return json.loads(_read(pkg / "config.json") or "{}")
    if use_hub and "/" in model and not p.exists():
        try:
            from huggingface_hub import hf_hub_download

            cfg = hf_hub_download(repo_id=model, filename="config.json")
            return json.loads(_read(Path(cfg)) or "{}")
        except Exception as e:
            print(
                f"# note: could not fetch config.json from Hub ({e}); "
                f"proceeding from arch.py only",
                file=sys.stderr,
            )
    return None


def is_moe(cfg: dict) -> bool:
    return any(
        k in cfg
        for k in (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_num_experts",
        )
    )


def choose_encoding(arch: dict, cfg: dict | None, notes: list[str]) -> str:
    supported = arch["supported_encodings"]
    default = arch["default_encoding"] or "bfloat16"
    if not cfg:
        notes.append(
            f"No config.json seen; using arch.py default_encoding '{default}'. "
            f"Verify it matches what the checkpoint actually ships."
        )
        return default
    # Pre-quantized checkpoint wins.
    qc = cfg.get("quantization_config") or {}
    method = str(qc.get("quant_method", "")).lower()
    if method:
        enc = QUANT_METHOD_TO_ENCODING.get(method)
        if enc and (not supported or enc in supported):
            if enc != default:
                notes.append(
                    f"Checkpoint is {method}-quantized -> using '{enc}' "
                    f"(arch default was '{default}')."
                )
            return enc
    # A quantized/GPU-only default means the checkpoint itself is quantized; a
    # GPTQ/fp8 config often still reports torch_dtype (the compute dtype), so
    # don't let torch_dtype downgrade a quantized default.
    if default in GPU_ONLY:
        return default
    # Otherwise map torch_dtype.
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "").lower()
    enc = DTYPE_TO_ENCODING.get(dtype)
    if enc and (not supported or enc in supported):
        if enc != default:
            notes.append(
                f"config.json dtype is '{dtype}' but arch default_encoding is "
                f"'{default}': the default looks wrong for this checkpoint; "
                f"using '{enc}' (it's in supported_encodings)."
            )
        return enc
    if enc and supported and enc not in supported:
        notes.append(
            f"config.json dtype '{dtype}' -> '{enc}' is NOT in supported_encodings "
            f"{sorted(supported)}; falling back to default '{default}'."
        )
    return default


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "-c", "--custom-architectures", required=True, help="package directory"
    )
    ap.add_argument(
        "-m",
        "--model",
        required=True,
        help="HF id, local checkpoint dir, or config.json",
    )
    ap.add_argument("--devices", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--no-hub", action="store_true")
    args = ap.parse_args()

    pkg = Path(args.custom_architectures).expanduser().resolve()
    if not (pkg / "arch.py").is_file():
        print(f"error: {pkg}/arch.py not found", file=sys.stderr)
        return 2

    arch = parse_arch(pkg)
    cfg = load_config(args.model, pkg, use_hub=not args.no_hub)
    notes: list[str] = []

    encoding = choose_encoding(arch, cfg, notes)

    if args.devices:
        devices = args.devices
    elif encoding in GPU_ONLY:
        devices = "gpu:0"
        notes.append(
            f"'{encoding}' is GPU-only -> --devices gpu:0 (CPU won't work)."
        )
    elif encoding == "float32":
        devices = "cpu"
        notes.append(
            "float32 -> CPU is feasible (use gpu:0 instead if you have the GPU)."
        )
    else:
        devices = "gpu:0"

    model_max = None
    if cfg:
        model_max = cfg.get("max_position_embeddings") or cfg.get("n_positions")
    if args.max_length:
        max_length = args.max_length
    elif isinstance(model_max, int) and model_max > 0:
        max_length = min(4096, model_max)
        if model_max < 4096:
            notes.append(
                f"Capped --max-length at model max_position_embeddings={model_max} "
                f"(a larger value makes serve abort)."
            )
    else:
        max_length = 4096

    if cfg:
        arch0 = (cfg.get("architectures") or [None])[0]
        if arch0 and arch["name"] and arch0 != arch["name"]:
            notes.append(
                f"MISMATCH: config.json architectures[0]='{arch0}' but arch.py "
                f"name='{arch['name']}'. Make them equal or MAX won't match."
            )
    if arch["name"] in LIKELY_BUILTINS:
        notes.append(
            f"name '{arch['name']}' matches a MAX built-in. On current "
            "nightly your custom arch overrides the built-in registration; "
            'on MAX 26.5 and earlier serve fails with "Refusing to override '
            'existing architecture". Rename the arch (e.g. '
            f"'{arch['name'].replace('For', 'CustomFor')}') and set the "
            "checkpoint's config.json architectures[0] to match unless the "
            "override is deliberate."
        )

    trust = bool(cfg and cfg.get("auto_map"))
    if trust:
        notes.append(
            "config.json has auto_map (custom modeling files) -> --trust-remote-code."
        )
    moe = bool(cfg and is_moe(cfg))
    if moe:
        notes.append(
            "MoE checkpoint -> consider --device-graph-capture and a larger --max-length."
        )

    # If --model pointed at a config.json (offline parse), the servable path is
    # its parent checkpoint directory, not the file itself.
    model_path = args.model
    _mp = Path(args.model)
    if _mp.name == "config.json" and _mp.is_file():
        model_path = str(_mp.parent)
    served = pkg.name
    cmd = [
        "max serve",
        f"--model-path {model_path}",
        f"--custom-architectures {pkg}",
        f"--devices {devices}",
        f"--quantization-encoding {encoding}",
        f"--max-length {max_length}",
        "--max-batch-size 1",
        f"--served-model-name {served}",
        f"--port {args.port}",
    ]
    notes.append(
        "Included --max-batch-size 1: some custom models assert it is set "
        "(serve crashes with 'Expected max_batch_size to be set' otherwise). "
        "Raise it for real throughput."
    )
    if trust:
        cmd.append("--trust-remote-code")
    if arch["chat_template"]:
        cmd.append(f"--chat-template {arch['chat_template']}")
    if moe:
        cmd.append("--device-graph-capture")

    print("# Suggested serve command (review the notes, then run):")
    print(" \\\n  ".join(cmd))
    print()
    print(
        f"# arch name={arch['name']}  default_encoding={arch['default_encoding']}  "
        f"chosen_encoding={encoding}  devices={devices}  max_length={max_length}"
    )
    if notes:
        print("# notes:")
        for n in notes:
            print(f"#  - {n}")
    else:
        print("# notes: none; defaults look right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
