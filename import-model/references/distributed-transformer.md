# Multi-GPU distribution-shape patterns

Most decoder ports past ~30B BF16 are multi-GPU. The donor table in
[map-to-max.md](map-to-max.md) lists archs by *attention/MLP shape*
(GQA vs MLA vs MoE) but not by *distribution shape* (single-GPU,
tensor-parallel, DP+EP). Different distribution shapes use different
framework base classes, different file patterns, and have different
debug-time pitfalls. Picking the wrong distribution-shape donor is the
single most expensive routing mistake in this workflow — you discover
it when the first ``max serve`` crashes in ``_unflatten_kv_inputs``
on a multi-GPU launch.

This reference covers the decision rule, the donor mapping, and the
non-obvious pitfalls.

## Decision rule (run at plan-and-veto)

Estimate roughly how much HBM the weights need, then divide by your
target GPU's HBM minus cushion for KV cache + activations + compile
overhead. Cushion is ~30–40% of HBM in practice.

```text
weight_bytes ≈ total_params × bytes_per_param   (BF16 = 2, FP8 = 1, NVFP4 ≈ 0.5)
gpus_needed ≈ ceil(weight_bytes / (gpu_hbm × 0.6))
```

Examples on a ~180 GB HBM GPU:

| Model                         | Weight bytes | gpus_needed | Distribution shape                |
|-------------------------------|--------------|-------------|-----------------------------------|
| Llama-3-8B BF16               | ~16 GB       | 1           | Single-GPU dense                  |
| Llama-3-70B BF16              | ~140 GB      | 2–4         | Multi-GPU TP                      |
| Mixtral-8x7B BF16             | ~95 GB       | 1–2         | Single-GPU MoE (or small TP)      |
| Qwen3-30B-A3B BF16            | ~62 GB       | 1           | Single-GPU MoE (or TP for speed)  |
| Large MoE (~200B active) BF16 | ~400+ GB     | 4–8         | Multi-GPU MoE (DP+EP recommended) |
| DeepSeek-V3 BF16              | ~1.3 TB      | 8–16        | Multi-GPU MoE + MLA               |

The rule covers ~95% of cases. The edge case is when KV cache dominates
(extremely long context) and pushes a model into multi-GPU even if the
weights fit. If ``max_position_embeddings × num_kv_heads × head_dim ×
dtype × num_layers × batch ≈ HBM cushion``, recompute with the KV
cache included.

## Donor mapping by distribution shape

Pick the donor whose distribution shape matches yours, not just whose
attention/MLP shape matches.

| Your shape                  | Donor                                              | Framework base                                                             | When to use                                                                             |
|-----------------------------|----------------------------------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Single-GPU dense            | ``llama3``, ``mistral``, ``cohere`` (Command-R v1) | ``Transformer``                                                            | ≤30B BF16 typically; one GPU                                                            |
| Single-GPU MoE              | ``qwen3`` (dense + MoE auto-detect), ``mixtral``   | ``Transformer`` (some variants use distributed base even at single device) | MoE that fits on one GPU                                                                |
| Multi-GPU TP (dense)        | ``llama3`` with sharding wired, or ``qwen3``       | ``DistributedLogitsPostprocessMixin + Module``                             | 70B–200B dense across 2–8 GPUs                                                          |
| Multi-GPU TP (MoE)          | ``qwen3``, ``mixtral`` (distributed mode)          | ``DistributedLogitsPostprocessMixin + Module``                             | MoE where attention TP + uniform expert sharding works                                  |
| Multi-GPU **DP + EP** (MoE) | ``qwen3`` (Qwen3-30B-A3B path), ``deepseekV3``     | ``DistributedLogitsPostprocessMixin + Module`` + ``EPBatchManager``        | Large MoE where data-parallel attention + expert-parallel MoE is the right partitioning |
| MLA + MoE                   | ``deepseekV3`` only                                | Same base as above                                                         | MLA latent-KV families                                                                  |

The framework base class is the load-bearing distinction. A
``Transformer``-based file expects single-device tensors and a single
``freqs_cis`` per call. A ``DistributedLogitsPostprocessMixin + Module``
file expects per-device tensor lists everywhere and explicit allreduce
between attention/MLP. They are not interchangeable, and a sed-rename
between the two will fail at ``max serve`` startup with
``_unflatten_kv_inputs`` mismatching the KV cache input count.

## What changes when you go multi-GPU

If your single-GPU port is working and you need to scale up, **you
cannot just pass ``--devices gpu:0,1,2,3``**. The graph itself needs
rewriting. The work is mechanical but non-trivial:

### File-level

- Class base swap: ``class MyModel(Transformer)`` →
  ``class MyModel(DistributedLogitsPostprocessMixin, Module)``.
- Block constructed once, sharded explicitly: each block creates
  ``self.self_attn = Attention(...)``, calls ``self.self_attn.shard
  (devices)`` to get a per-device list, and stores both
  (``self.self_attn`` + ``self.self_attn_shards``).
- Layer list: ``self.layers = LayerList([...])``, not a Python list.
- Forward signature: every tensor argument becomes a list-of-tensors,
  one per device. ``def __call__(self, layer_idx, xs, kv_collections,
  freqs_cis, input_row_offsets, signal_buffers)`` where each of those
  (except ``layer_idx``) is a list.

### Per-block forward

```python
def __call__(self, layer_idx, xs, kv_collections, freqs_cis,
             input_row_offsets, signal_buffers):
    # 1. Norm each device's hidden state in parallel.
    norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

    # 2. Attention on each device's shard of heads.
    attn_outs = [shard(layer_idx, norm_xs[i], kv_collections[i],
                       freqs_cis[i], input_row_offsets[i])
                 for i, shard in enumerate(self.self_attn_shards)]

    # 3. TP mode: allreduce across attention shards.
    if not self.use_dp and len(self.devices) > 1:
        attn_outs = self.allreduce(attn_outs, signal_buffers)

    # 4. MLP / MoE on each shard.
    mlp_outs = forward_sharded_layers(self.mlp_shards, post_attn_xs)
    if not self.use_dp and len(self.devices) > 1:
        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

    # 5. Residuals per-device.
    return [x + a + m for x, a, m in zip(xs, attn_outs, mlp_outs)]
```

The pattern is the same regardless of architecture. What changes
between ports is *what* the attention and MLP do, not the
shard-and-allreduce skeleton.

### Embeddings + LM head

- ``self.embed_tokens = VocabParallelEmbedding(...)``: shards the vocab
  across devices; returns a per-device hidden-state list directly.
- ``self.lm_head = ColumnParallelLinear(...)``: shards output channels;
  needs signal_buffers for the final allreduce.
- Tied embeddings: ``ColumnParallelLinear(tied_weight=embed_tokens.weight,
  ...)`` — share the tensor identity, not a copy.

### Inputs

- ``signal_buffers`` is required even for nominally-TP-only models if
  the embedding or LM head are vocab-parallel (they always are in this
  base class). Mark the slug ``class MyModel(AlwaysSignalBuffersMixin,
  LlamaModelBase)`` in ``model.py``.
- The ``input_types()`` method now returns ``base_inputs +
  signal_buffer_types + flattened_kv_types`` — three concatenated
  groups, in that order.
- ``_build_graph`` unpacks accordingly:
  ``tokens, input_row_offsets, return_n_logits, *variadic = graph.inputs``
  then peel signal buffers and KV cache inputs by count.

## Pitfalls specific to multi-GPU MoE

Common failure modes on multi-GPU MoE ports:

1. **``linear_cls = functools.partial(Linear, quant_config=...)`` is a
   trap.** Binding ``quant_config`` into ``linear_cls`` makes every
   ``Linear`` constructed via that partial inherit the global config —
   even when the consuming module wants ``quant_config=None`` (e.g.
   selective per-layer quantization). Solution: don't bind. Pass
   ``Linear`` itself; have each call site specify ``quant_config``
   explicitly.

2. **Selective quantization needs per-layer routing.** ``QuantConfig``
   carries ``attn_quantized_layers`` and ``mlp_quantized_layers`` sets.
   When building each block, consult them:

   ```python
   qc = config.quant_config
   attn_quant_config = (
       qc if qc is not None and layer_idx in qc.attn_quantized_layers
       else None
   )
   ```

   Without this, a model that quantizes only MoE (leaving attention
   bf16) will have a graph expecting FP8 weights at attention positions
   and crash at load.

3. **Dispatch dtype ≠ unquantized dtype.** For an FP8 model,
   ``config.dtype == DType.float8_e4m3fn`` is the *dispatch* dtype. The
   un-quantized sections (attention if not in ``attn_quantized_layers``,
   the MoE router gate) are bf16 on disk. Pass an explicit
   ``DType.bfloat16`` to those Linears, not ``config.dtype``.

4. **HF wraps multimodal configs.** Vision or conditional-generation
   config types nest the text backbone
   under ``.text_config``. Framework methods (``calculate_max_seq_len``,
   ``get_kv_params``, ``parse_quant_config``) walk the *parent* config
   by default. Override on the model class to unwrap before delegating.

5. **Quantization config ignore-list prefix mismatch.** Compressed-
   tensors HF configs store ignore entries with the original prefix
   (``model.language_model.layers.X.self_attn.q_proj``) but MAX's
   ``parse_quant_config`` checks against the post-strip prefix
   (``model.layers.X.self_attn.q_proj``). Rewrite the ignore list
   before delegating to ``parse_quant_config`` if your model is
   multimodal-wrapped.

6. **Per-device freqs_cis lists.** For NoPE on full-attention layers
   (some MoE and Gemma families), build two per-device lists at graph entry —
   ``real_freqs_cis`` from the rotary embedding and
   ``identity_freqs_cis`` (cos=1, sin=0). Select per layer by
   ``layer_types[i]``. The identity table must match the layout of
   ``rope.freqs_cis`` exactly, including the ``max_seq_len * 2`` row
   count (the rotary embedding pre-allocates 2× for decode positions
   past prefill).

7. **GPU memory zombies after ``pkill max serve``.** A multiprocessing
   spawn worker can survive ``kill -9`` and hold HBM as a defunct
   (Z-state) process. Symptom: subsequent serve attempts see only
   far less free HBM than expected on each device.
