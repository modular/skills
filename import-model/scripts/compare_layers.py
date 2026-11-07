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
"""Compare HF reference vs MAX serve at a prompt (divergence-hunt helper).

MAX does not expose per-layer hidden states through the OpenAI completions API.
This script therefore:

1. (Optional) Runs HF and prints per-layer hidden-state stats (HF-only diagnostic).
2. Compares top-1 next-token logprob at a short prompt (prefill).
3. Compares top-1 token id at several prefix lengths (catches RoPE / position bugs).

For per-layer HF-vs-MAX diffs, build the debug-model skill's tensor-dump
comparators (see references/layer-by-layer-debugging.md).

Requires:
- ``pixi run max serve --model-path <HF_ID> --custom-architectures <port_dir>``
  running on ``--port``. The probes read logprobs, which the overlap
  scheduler rejects; for an architecture whose ``arch.py`` leaves
  ``supports_overlap_scheduler`` on, also pass
  ``--no-enable-overlap-scheduler --force``.
- transformers installed for the HF side.

Usage::

    pixi run python compare_layers.py <HF_MODEL_ID> --port 8000

The probe sends the HF model ID as the API ``model`` name, the name
``max serve`` uses by default. If you started the server with
``--served-model-name``, pass the same name to this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore, where the gentle waves lap "
    "against smooth, sun-bleached stones. Every morning at dawn, the "
    "fisherman rowed his small wooden boat across the glassy surface of "
    "the lake, casting his line into the deep blue water and waiting "
    "patiently for the first bite. Over the course of many years, he had "
    "learned that patience was the single most valuable virtue a fisherman "
    "could possess, and he practiced it diligently."
)
DEFAULT_POSITIONS = (1, 5, 20, 50, 100, 200)
# A MAX top-1 token within this many nats of HF's top-1 counts as a near-tie.
# bfloat16 serving and device differences move top logprobs by about 0.05
# nats, enough to flip the order of two close tokens in a correct model.
NEAR_TIE_NATS = 0.1
# The prefill check passes when MAX's top-1 logprob is within this fraction
# of HF's. A correct bfloat16 port lands near 0.01-0.02 against a float32
# reference, and a wrong RoPE layout lands near 0.09.
PREFILL_REL_TOL = 0.05


@dataclass
class Stats:
    mean: float
    max_abs: float
    norm: float


def tensor_stats(t: object) -> Stats:
    t = t.detach().float()  # type: ignore[attr-defined]
    return Stats(
        mean=float(t.mean()),
        max_abs=float(t.abs().max()),
        norm=float(t.norm()),
    )


def load_hf_model(hf_id: str, dtype: str) -> tuple[object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return tok, model


def run_hf(hf_id: str, prompt: str, dtype: str) -> tuple[list, float, int]:
    """Return (layer hidden states, hf top-1 logprob, hf top-1 token id)."""
    import torch

    tok, model = load_hf_model(hf_id, dtype)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
    logits = out.logits[0, -1]
    top1_id = int(logits.argmax())
    log_probs = torch.log_softmax(logits, dim=-1)
    hf_top1_logprob = float(log_probs[top1_id])
    return [h.cpu() for h in out.hidden_states], hf_top1_logprob, top1_id


def hf_prefill_top1(hf_id: str, prompt: str, dtype: str) -> tuple[float, int]:
    """Return HF's top-1 logprob and token id for the token after ``prompt``."""
    import torch

    tok, model = load_hf_model(hf_id, dtype)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top1_id = int(logits.argmax())
    return float(torch.log_softmax(logits, dim=-1)[top1_id]), top1_id


def prefill_rel_diff(hf_logprob: float, max_logprob: float) -> float:
    """Relative difference between HF's and MAX's top-1 prefill logprobs."""
    return abs(hf_logprob - max_logprob) / max(abs(hf_logprob), 1e-6)


def hf_top_logprobs_at(
    model: object, input_ids: object, pos: int, k: int = 5
) -> list[tuple[int, float]]:
    """Return HF's top ``k`` next-token ids and logprobs after ``pos`` tokens."""
    import torch

    prefix = input_ids[:, :pos]
    with torch.no_grad():
        logits = model(prefix).logits[0, -1].float()
    top = torch.topk(torch.log_softmax(logits, dim=-1), k)
    return [
        (int(i), float(v)) for v, i in zip(top.values, top.indices, strict=True)
    ]


def same_token_text(hf_text: str, mx_text: str) -> bool:
    """Whether a decoded HF token and the server's completion text agree.

    Compares the raw text first, so whitespace tokens like "\\n" match
    without stripping. The stripped fallback absorbs leading-space
    differences between the tokenizer's decode and the server's text.
    """
    return hf_text == mx_text or bool(
        hf_text.strip() and hf_text.strip() == mx_text.strip()
    )


# A server that runs the overlap scheduler rejects logprobs requests. MAX
# turns it on for any architecture whose arch.py doesn't set
# supports_overlap_scheduler=False, and only turns it off with both flags.
_OVERLAP_FLAGS = "--no-enable-overlap-scheduler --force"


def server_error(exc: urllib.error.HTTPError) -> str:
    """The error message in a MAX server's HTTP error response.

    Falls back to the status line when the body isn't MAX's JSON error.
    """
    try:
        message = json.loads(exc.read())["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return f"HTTP {exc.code} {exc.reason}"
    if "overlap scheduler" in message:
        message += f" Restart max serve with {_OVERLAP_FLAGS}."
    return f"HTTP {exc.code}: {message}"


def fetch_max_completion(
    port: int,
    model_name: str,
    prompt: str,
    *,
    top_k: int = 5,
    max_tokens: int = 1,
) -> dict:
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": top_k,
            "echo": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        choice = json.loads(r.read())["choices"][0]
    lp_block = choice.get("logprobs") or {}
    token_lps = lp_block.get("token_logprobs") or []
    top_dict = (lp_block.get("top_logprobs") or [{}])[0]
    text = choice.get("text") or ""
    return {
        "text": text,
        "top1_logprob": float(token_lps[0]) if token_lps else float("nan"),
        "top5": sorted(top_dict.items(), key=lambda kv: kv[1], reverse=True)[
            :top_k
        ],
    }


def fetch_max_logprobs(
    port: int, model_name: str, prompt: str, top_k: int = 5
) -> dict:
    mx = fetch_max_completion(port, model_name, prompt, top_k=top_k)
    return {
        "top1_text": mx["text"],
        "top1_logprob": mx["top1_logprob"],
        "top5": mx["top5"],
    }


def probe_multi_position(
    hf_id: str,
    model_name: str,
    port: int,
    *,
    dtype: str = "float32",
    prompt: str = DEFAULT_LONG_PROMPT,
    positions: tuple[int, ...] = DEFAULT_POSITIONS,
    quiet: bool = False,
) -> list[tuple[int, int, str]]:
    """Return list of (pos, hf_top1_id, max_top1_text) where top-1 diverged."""
    tok, model = load_hf_model(hf_id, dtype)
    ids = tok(prompt, return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    diverged: list[tuple[int, int, str]] = []

    if not quiet:
        print(
            f"Multi-position probe ({len(positions)} points, seq_len={seq_len})"
        )
        print(f"{'pos':>6}  {'hf_id':>8}  {'max_text':>20}  verdict")

    for pos in positions:
        if pos > seq_len:
            if not quiet:
                print(f"{pos:>6}  (skip: past seq_len {seq_len})")
            continue
        prefix = tok.decode(
            ids[0, :pos].tolist(), skip_special_tokens=True
        )
        if not prefix.strip():
            # A BOS-only prefix decodes to no text: the server would answer
            # a fresh context, while HF predicts from after the BOS token.
            if not quiet:
                print(f"{pos:>6}  (skip: prefix decodes to no text)")
            continue
        ranked = hf_top_logprobs_at(model, ids.to(model.device), pos)
        hf_id_at, hf_top_lp = ranked[0]
        mx_text = fetch_max_completion(port, model_name, prefix, top_k=1)[
            "text"
        ]
        match = same_token_text(tok.decode([hf_id_at]), mx_text)
        verdict = "ok" if match else "DIVERGED"
        if not match and any(
            hf_top_lp - lp <= NEAR_TIE_NATS
            and same_token_text(tok.decode([tid]), mx_text)
            for tid, lp in ranked[1:]
        ):
            match = True
            verdict = "near-tie"
        if not quiet:
            print(f"{pos:>6}  {hf_id_at:>8}  {mx_text!r:>20}  {verdict}")
        if not match:
            diverged.append((pos, hf_id_at, mx_text))

    return diverged


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id", help="HuggingFace model ID")
    parser.add_argument(
        "--served-model-name",
        help="Model name the server answers to (default: the HF model ID)",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Short prompt for single-step logprob check",
    )
    parser.add_argument(
        "--long-prompt",
        default=DEFAULT_LONG_PROMPT,
        help="Long prompt for multi-position probe",
    )
    parser.add_argument(
        "--positions",
        default=",".join(map(str, DEFAULT_POSITIONS)),
        help="Comma-separated prefix lengths for multi-position probe",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="HF reference dtype. A float32 reference keeps HF's own rounding "
        "out of the comparison.",
    )
    parser.add_argument("--port", type=int, default=8000, help="MAX serve port")
    parser.add_argument(
        "--skip-hf-layers",
        action="store_true",
        help="Skip HF hidden-state stats dump",
    )
    parser.add_argument(
        "--skip-multi",
        action="store_true",
        help="Skip multi-position prefix probe",
    )


def main(args: argparse.Namespace) -> int:
    positions = tuple(
        int(x.strip()) for x in args.positions.split(",") if x.strip()
    )

    print(f"HF reference: {args.hf_id!r}")
    model_name = args.served_model_name or args.hf_id
    print(f"MAX model name: {model_name!r} on localhost:{args.port}")
    print(f"dtype: {args.dtype}")
    print()

    exit_code = 0

    if not args.skip_hf_layers:
        print(
            "Running HF (per-layer stats are HF-only: MAX has no API for these yet)..."
        )
        hf_states, hf_top1_logprob, hf_top1_id = run_hf(
            args.hf_id, args.prompt, args.dtype
        )
        print(f"{'layer':>5}  {'mean':>10}  {'max_abs':>10}  {'norm':>10}")
        for i, h in enumerate(hf_states):
            s = tensor_stats(h)
            print(
                f"{i:>5}  {s.mean:>10.4f}  {s.max_abs:>10.4f}  {s.norm:>10.2f}"
            )
        print()
    else:
        hf_top1_logprob, hf_top1_id = hf_prefill_top1(
            args.hf_id, args.prompt, args.dtype
        )

    print("Querying MAX logits via /v1/completions logprobs...")
    try:
        mx = fetch_max_logprobs(args.port, model_name, args.prompt)
    except urllib.error.HTTPError as exc:
        print(
            f"error: MAX server on port {args.port} returned "
            f"{server_error(exc)}",
            file=sys.stderr,
        )
        return 1
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(
            f"error: MAX server unreachable on port {args.port}: {exc}",
            file=sys.stderr,
        )
        print(
            "Start serve first, e.g.\n"
            f"  pixi run max serve --model-path {args.hf_id} "
            f"--custom-architectures <port_dir>  # port folder, not its parent",
            file=sys.stderr,
        )
        return 1

    mx_lp = mx["top1_logprob"]
    if math.isnan(mx_lp):
        print(
            "error: MAX returned no token_logprobs in the completion response.",
            file=sys.stderr,
        )
        return 1

    rel_diff = prefill_rel_diff(hf_top1_logprob, mx_lp)
    match = "ok" if rel_diff < PREFILL_REL_TOL else "DIVERGED"

    print()
    print(f"{'check':>12}  {'hf':>12}  {'max':>12}  {'rel_diff':>10}  verdict")
    print("-" * 62)
    print(
        f"{'top1_logprob':>12}  {hf_top1_logprob:>12.4f}  {mx_lp:>12.4f}  "
        f"{rel_diff:>10.4f}  {match}"
    )
    print(f"HF top-1 token id: {hf_top1_id}")
    print(f"MAX top-1 text: {mx['top1_text']!r}")
    if mx["top5"]:
        print(
            "MAX top-5 logprobs:",
            ", ".join(f"{t!r}:{lp:.3f}" for t, lp in mx["top5"]),
        )

    if match == "DIVERGED":
        exit_code = 1

    if not args.skip_multi:
        print()
        try:
            diverged = probe_multi_position(
                args.hf_id,
                model_name,
                args.port,
                dtype=args.dtype,
                prompt=args.long_prompt,
                positions=positions,
            )
        except urllib.error.HTTPError as exc:
            print(
                f"error: multi-position probe failed: {server_error(exc)}",
                file=sys.stderr,
            )
            return 1
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"error: multi-position probe failed: {exc}", file=sys.stderr)
            return 1
        if diverged:
            pos, hf_tid, mx_txt = diverged[0]
            print()
            print(
                f"First top-1 mismatch at prefix length {pos} "
                f"(hf_id={hf_tid}, max={mx_txt!r})"
            )
            print(
                "Likely RoPE, partial-RoPE, sliding-window, or NoPE bug. See divergences.md"
            )
            exit_code = 1

    if exit_code:
        print()
        print(
            "Logits diverge. Match the symptom in references/divergences.md, then"
        )
        print(
            "localize the layer with the debug-model skill's per-layer comparators."
        )
    return exit_code


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
