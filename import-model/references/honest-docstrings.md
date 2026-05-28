# Honest docstrings after scaffold-by-copy

`scaffold.py` (and every "copy a donor + rename" approach) is a silent
failure for **comments and docstrings**. The class names get renamed.
The text that records *what the file claims to do* — module docstrings,
class docstrings, code comments — does not. After a sed-rename,
``my_large_moe.py`` can still open with a docstring describing Qwen3; the
new class can claim "single-GPU, multi-GPU TP, and DP+EP" when the port
only supports multi-GPU.

Nothing in the test loop catches this. ``max serve`` doesn't read
docstrings. ``compare_layers.py`` doesn't read docstrings. Only a human
reader catches it — usually a reader who isn't you, weeks later, who
mistakes a lie for documentation and acts on it.

This reference lays down two rules that prevent the failure by
construction:

1. A positive shape every module docstring should follow.
2. A mechanical check to run before declaring scaffold + implementation
   done.

## The three-sentence module docstring

Every file in your port directory opens with a module docstring of this
shape:

1. **What this file is**, in your port's terms — not the donor's.
   ("``my_large_moe`` text-generation graph for multi-GPU serve.")
2. **What it's derived from**, named explicitly. ("Structurally mirrors
   ``max.pipelines.architectures.qwen3.qwen3`` for the distributed-
   transformer skeleton.")
3. **Deltas applied**, as a bullet list — only what you actually
   implemented. ("Parallel decoder block. LayerNorm without bias (not
   RMSNorm). Interleaved RoPE. Sigmoid+renorm router. NoPE on full-attn
   layers via identity ``freqs_cis``.")

The donor is *acknowledged* ("derived from qwen3") but not *claimed*
("this is a qwen3 model"). Sed-rename swaps names in the donor's text;
the fix is to rewrite the docstring from scratch using this shape.

## Worked example: module docstring

### Lying (after sed-rename of `qwen3/qwen3.py`)

```python
"""Build a Qwen3 model that supports single-GPU, multi-GPU TP, and DP+EP."""
```

The class in this file is now ``MyLargeMoE``. It does not implement Qwen3.
The released BF16 weights do not fit on one GPU — single-GPU is **not** a
supported mode. Both claims are false.

### Honest rewrite

```python
"""MyLargeMoE text-generation graph for multi-GPU serve.

Structurally mirrors ``max.pipelines.architectures.qwen3.qwen3`` for
the distributed-transformer skeleton (LayerList + per-block ``.shard()``
+ ``forward_sharded_layers`` + ``Allreduce``). Single-GPU is not a
supported mode for this checkpoint; multi-GPU is the only viable path.

Deltas layered on top of the qwen3 skeleton:

- **Parallel decoder block** with a single ``input_layernorm`` and
  ``x' = x + attn(norm(x)) + ffn(norm(x))`` (no post-attention norm).
- **LayerNorm without bias** (not RMSNorm). Read ``layer_norm_eps`` from
  config when ``rms_norm_eps`` is absent.
- **Interleaved RoPE** with ``interleaved=True`` on the rotary embedding
  (verify against HF ``rotate_half`` — see [divergences.md](divergences.md)).
- **NoPE on full-attention layers** — identity ``freqs_cis`` (cos=1,
  sin=0) per ``config.layer_types[]`` where HF skips rotation.
- **Sigmoid + renorm-by-sum router** (not softmax-then-topk like Qwen3).
"""
```

## Worked example: class docstring

### Lying

```python
class MyLargeMoE(DistributedLogitsPostprocessMixin, Module):
    """Unified Qwen3 model supporting single-GPU, TP, and DP+EP inference."""
```

Wrong on name, GPU modes, and model family.

### Honest rewrite

```python
class MyLargeMoE(DistributedLogitsPostprocessMixin, Module):
    """MyLargeMoE causal-LM graph for multi-GPU inference.

    Builds the embedding, :class:`MyLargeMoETransformerBlock` stack, final
    norm, and LM head. Sharding and KV cache plumbing live in the block
    layer; this class wires layers and per-layer RoPE tables (real vs
    identity for NoPE).
    """
```

## Comment-level lies

```python
# Per-head RMSNorm for Q and K (Qwen3-specific)
self.q_norm = RMSNorm(...)
self.k_norm = RMSNorm(...)
```

If your model has no QK-norm, delete the dead code or explain why fields
remain (e.g. donor sharding plumbing). Do not leave the Qwen3-specific
comment.

## Behavioral lies that survive a name grep

- ``"""Supports single-GPU, multi-GPU TP, and DP+EP inference."""`` in a
  multi-GPU-only file
- ``# Allreduce (only used in TP mode)`` when there is no non-TP mode
- ``# Apply post-attention layer norm`` in a parallel block with no
  post-attention norm

Grepping for donor names does not catch these. Read every docstring and
comment.

## Mandatory audit before declaring the implementation done

Run from ``<port_dir>/`` (the slug folder with ``arch.py``):

```bash
pixi run rg -i -n \
  'qwen|llama|mistral|cohere|gemma|phi|deepseek|exaone|olmo|granite|qwen3|mixtral|single-GPU|single GPU|RMSNorm|QK-norm' \
  .
```

Classify each match:

- **OK** — explicit lineage ("Structurally mirrors qwen3 …"). Leave it.
- **Lie** — donor behavior that does not apply ("Qwen3-specific QK-norm"
  with no QK-norm; "Supports single-GPU" in a multi-GPU-only file).
  Rewrite or delete.
- **Stale** — wrong line numbers, names, or paths. Update or remove.

Then read every module and class docstring without grepping.

## What "done" looks like

Attest explicitly when implementation is complete:

> Scaffolding + implementation complete. Module docstrings follow the
> three-sentence pattern (qwen3 under "derived from", not claimed).
> ``rg -i 'qwen' <port_dir>`` returns N hits, all legitimate lineage
> references.

A completion claim without that attestation is incomplete.

## Why not ban sed-rename?

Donor file structure is genuinely useful. The discipline issue is
*post-rename review*, not the rename. "Audit after sed-rename" is a
checkpoint with concrete artifacts (grep output, rewritten docstrings).
