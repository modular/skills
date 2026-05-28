#!/usr/bin/env python3
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

1. (Optional) Runs HF and prints per-layer hidden-state stats — HF-only diagnostic.
2. Compares top-1 next-token logprob at a short prompt (prefill).
3. Compares top-1 token id at several prefix lengths (catches RoPE / position bugs).

For true HF-vs-MAX tensor diffs inside each block, add ``ops.output(...)`` taps
in your port's ``<slug>.py`` (see references/layer-by-layer-debugging.md).

Requires:
- ``pixi run max serve --model-path <HF_ID> --custom-architectures <port_dir>``
  running on ``--port``.
- transformers installed for the HF side.

Usage::

    pixi run python compare_layers.py <HF_MODEL_ID> \\
        --slug <your_slug> --port 8000
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


def hf_top1_id_at(model: object, input_ids: object, pos: int) -> int:
    import torch

    prefix = input_ids[:, :pos]
    with torch.no_grad():
        logits = model(prefix).logits[0, -1]
    return int(logits.argmax())


def fetch_max_completion(
    port: int,
    slug: str,
    prompt: str,
    *,
    top_k: int = 5,
    max_tokens: int = 1,
) -> dict:
    payload = json.dumps(
        {
            "model": slug,
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
    text = (choice.get("text") or "").strip()
    return {
        "text": text,
        "top1_logprob": float(token_lps[0]) if token_lps else float("nan"),
        "top5": sorted(top_dict.items(), key=lambda kv: kv[1], reverse=True)[
            :top_k
        ],
    }


def fetch_max_logprobs(
    port: int, slug: str, prompt: str, top_k: int = 5
) -> dict:
    mx = fetch_max_completion(port, slug, prompt, top_k=top_k)
    return {
        "top1_text": mx["text"],
        "top1_logprob": mx["top1_logprob"],
        "top5": mx["top5"],
    }


def max_top1_text_at(
    tok: object, port: int, slug: str, input_ids: object, pos: int
) -> str:
    prompt = tok.decode(input_ids[0, :pos].tolist(), skip_special_tokens=True)
    return fetch_max_completion(port, slug, prompt, top_k=1)["text"]


def probe_multi_position(
    hf_id: str,
    slug: str,
    port: int,
    *,
    dtype: str = "bfloat16",
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
                print(f"{pos:>6}  (skip — past seq_len {seq_len})")
            continue
        hf_id_at = hf_top1_id_at(model, ids.to(model.device), pos)
        mx_text = max_top1_text_at(tok, port, slug, ids, pos)
        hf_text = tok.decode([hf_id_at])
        match = hf_text == mx_text or (
            mx_text and hf_text.strip() == mx_text.strip()
        )
        verdict = "ok" if match else "DIVERGED"
        if not quiet:
            print(f"{pos:>6}  {hf_id_at:>8}  {mx_text!r:>20}  {verdict}")
        if not match:
            diverged.append((pos, hf_id_at, mx_text))

    return diverged


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("hf_id", help="HuggingFace model ID")
    parser.add_argument(
        "--slug", required=True, help="Slug registered in your custom arch"
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
        "--dtype", default="bfloat16", choices=["float32", "bfloat16"]
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
    print(f"MAX slug: {args.slug!r} on localhost:{args.port}")
    print(f"dtype: {args.dtype}")
    print()

    exit_code = 0

    if not args.skip_hf_layers:
        print(
            "Running HF (per-layer stats are HF-only — MAX has no API for these yet)..."
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
        import torch

        tok, model = load_hf_model(args.hf_id, args.dtype)
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        hf_top1_id = int(logits.argmax())
        hf_top1_logprob = float(torch.log_softmax(logits, dim=-1)[hf_top1_id])

    print("Querying MAX logits via /v1/completions logprobs...")
    try:
        mx = fetch_max_logprobs(args.port, args.slug, args.prompt)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(
            f"error: MAX server unreachable on port {args.port}: {exc}",
            file=sys.stderr,
        )
        print(
            "Start serve first, e.g.\n"
            f"  pixi run max serve --model-path {args.hf_id} "
            f"--custom-architectures <port_dir>  # slug folder, not its parent",
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

    abs_diff = abs(hf_top1_logprob - mx_lp)
    rel_diff = abs_diff / max(abs(hf_top1_logprob), 1e-6)
    match = "ok" if rel_diff < 0.05 else "DIVERGED"

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
                args.slug,
                args.port,
                dtype=args.dtype,
                prompt=args.long_prompt,
                positions=positions,
            )
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
                "Likely RoPE, partial-RoPE, sliding-window, or NoPE bug — see divergences.md"
            )
            exit_code = 1

    if exit_code:
        print()
        print(
            "Logits diverge. Use divergences.md symptom table; add ops.output() taps"
        )
        print("in your port for layer-local comparison.")
    return exit_code


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    sys.exit(main(p.parse_args()) or 0)
