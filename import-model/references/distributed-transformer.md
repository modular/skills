# Multi-GPU distribution-shape patterns

Most decoder ports past ~30B BF16 are multi-GPU. The donor table in
[map-to-max.md](map-to-max.md) lists archs by *attention/MLP shape*
(GQA vs MLA vs MoE). This reference covers *distribution shape*
(single-GPU, tensor-parallel, DP+EP). A ModuleV3 port distributes by
placing the module on a `DeviceMesh` and tagging weights with placements, so
the same `forward()` serves one GPU or many. The decisions are which mesh
axes the port uses and which weights shard over them.

## Decision rule (run at plan-and-veto)

Estimate roughly how much HBM the weights need, then divide by your
target GPU's HBM minus cushion for KV cache + activations + compile
overhead. Cushion is ~30–40% of HBM in practice.

```text
weight_bytes ≈ total_params × bytes_per_param   (BF16 = 2, FP8 = 1, NVFP4 ≈ 0.5)
gpus_needed ≈ ceil(weight_bytes / (gpu_hbm × 0.6))
```

The table applies the rule to a GPU with ~180 GB of HBM:

| Model                         | Weight bytes | gpus_needed | Distribution shape                |
|-------------------------------|--------------|-------------|-----------------------------------|
| Llama-3-8B BF16               | ~16 GB       | 1           | Single-GPU dense                  |
| Llama-3-70B BF16              | ~140 GB      | 2–4         | Multi-GPU TP                      |
| Mixtral-8x7B BF16             | ~95 GB       | 1–2         | Single-GPU MoE (or small TP)      |
| Qwen3-30B-A3B BF16            | ~62 GB       | 1           | Single-GPU MoE (or TP for speed)  |
| Large MoE (~200B total) BF16  | ~400+ GB     | 4–8         | Multi-GPU MoE (DP+EP recommended) |
| DeepSeek-V3 BF16              | ~1.3 TB      | 8–16        | Multi-GPU MoE + MLA               |

The rule breaks down when KV cache dominates
(extremely long context) and pushes a model into multi-GPU even if the
weights fit. If ``2 × max_position_embeddings × num_kv_heads × head_dim ×
bytes_per_element × num_layers × batch ≈ HBM cushion``, recompute with the
KV cache included.

## Donor mapping by distribution shape

Pick the donor whose distribution shape matches yours, as well as its
attention/MLP shape.

| Your shape                  | Donor                                      | Mesh                             | When to use                                                                             |
|-----------------------------|--------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------|
| Single-GPU dense            | `llama3_modulev3`, `olmo3`                 | `.to(self.devices[0])`           | ≤30B BF16 typically, one GPU                                                            |
| Single-GPU MoE              | `gpt_oss_modulev3`                         | `.to(self.devices[0])`           | MoE that fits on one GPU                                                                |
| Multi-GPU TP (dense)        | `gemma3_modulev3`                          | 1-D `("tp",)`                    | 70B–200B dense across 2–8 GPUs                                                          |
| Multi-GPU TP (MoE)          | `deepseekV3_modulev3`                      | 1-D `("tp",)`                    | MoE where attention TP + uniform expert sharding works                                  |
| Multi-GPU **DP + EP** (MoE) | `deepseekV3_modulev3`, `kimik2_5_modulev3` | 1-D `("dp",)` + `EPBatchManager` | Large MoE where data-parallel attention + expert-parallel MoE is the right partitioning |
| MLA + MoE                   | `deepseekV3_modulev3`                      | `("tp",)` or `("dp",)`           | MLA latent-KV families                                                                  |

Set `multi_gpu_supported=True` in `arch.py` for a port that shards.

## The mesh

`model.py` builds the mesh and constructs the module inside
`default_device(mesh)`, so every weight is created on the mesh. The axis name
tells the parallel layers which mesh axis to shard over: `"tp"` for tensor
parallelism, `"dp"` for data parallelism
(`max.nn.common_layers.mesh_axis`):

```python
from max.tensor.sharding import DeviceMesh
from max.tensor import default_device


def _instantiate_module(self, model_config: MyConfig) -> MyModel:
    n_devices = len(self.devices)
    mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
    with default_device(mesh):
        return MyModel(model_config, self.kv_params)
```

`Module.to()` moves a module to one device and raises for a multi-device
mesh, so a sharded port can't build on one device and move afterward.

`gemma3_modulev3/model.py` builds the mesh here. `deepseekV3_modulev3/model.py`
builds it in `_create_model_config()` and stores it on the config, so modules
that need the mesh during construction can read `config.mesh`. It picks the
`"dp"` axis name when `data_parallel_degree > 1`.

## Sharded layers

Tag each weight with how it shards. A layer built under `default_device(mesh)`
creates each shard on its device:

- **Column-parallel** (Q/K/V projections, MLP gate and up, LM head):
  `ColumnParallelLinear(in_dim, out_dim, bias=False)`, or
  `col_parallel(Linear(...))` on an existing layer. Each device holds a slice
  of the output features.
- **Row-parallel** (attention output, MLP down): `RowParallelLinear(...)` or
  `row_parallel(Linear(...))`. Each device holds a slice of the input
  features and produces a partial sum.
- **Vocab-parallel embedding**: `VocabParallelEmbedding(vocab_size,
  dim=hidden_size)` from `max.nn.common_layers.embedding`.

All of these live in `max.nn.common_layers.linear` unless noted.
`gemma3_modulev3/layers/attention.py` is the reference for a TP attention
module.

## Collectives

The sharding solver tracks each tensor's placement (`Replicated`, `Sharded`,
`Partial`) and inserts the collective an op needs. In
`gemma3_modulev3/layers/transformer_block.py`, the row-parallel attention
output is a `Partial` sum, and the post-attention norm that consumes it
inserts the all-reduce. The block's `forward()` has no explicit collective.

Call a collective directly when the port controls where the reduction
happens:

- `F.allreduce_sum(t)`: `Partial` → `Replicated`.
- `F.reduce_scatter(t, scatter_axis=0)`: `Partial` → `Sharded`.
- `F.allgather(t, tensor_axis=-1)`: `Sharded` → `Replicated` (tied LM head in
  `gemma3_modulev3/gemma3.py`).
- `F.transfer_to(t, mapping)`: any placement change.

`deepseekV3_modulev3/layers/transformer_block.py` switches between these per
parallelism mode (TP attention with TP or EP MoE, DP attention with EP MoE).

## Expert parallelism

EP routes tokens to experts on other devices through NVSHMEM communication
buffers. The pipeline model sets it up in `_init_distributed_runtime()`.
That method builds an `EPBatchManager` and stores its input types in
`self._modulev3_extra_input_types`, which the base class appends to the
compile inputs. It also initializes the communication buffers. The MoE module
receives the buffers as extra `forward()` inputs.
`deepseekV3_modulev3/model.py` is the reference.

## Per-device work

Some steps run independently on each device, such as a gather over each data
parallel replica's own rows. Wrap the per-device function with
`F.functional()`. With no sharding rule, it runs on each device's shard. To
call a helper that takes per-device lists, pass
`[TensorValue(s) for s in t.local_shards]` and rebuild the result with
`Tensor.from_shard_values(values, mapping)`. `gather_last_tokens()` and
`split_replicated_batch()` in `deepseekV3_modulev3/deepseekV3.py` show both.

## Pitfalls specific to multi-GPU MoE

These failure modes are common on multi-GPU MoE ports:

1. **Selective quantization needs per-layer routing.** ``QuantConfig``
   carries ``attn_quantized_layers`` and ``mlp_quantized_layers`` sets.
   When building each block, consult them:

   ```python
   qc = config.quant_config
   attn_quant_config = (
       qc if qc is not None and layer_idx in qc.attn_quantized_layers else None
   )
   ```

   Without this, a model that quantizes only MoE (leaving attention
   bf16) builds attention parameters that expect FP8 weights and fails
   `compile()` on the dtype check.

2. **Dispatch dtype differs from unquantized dtype.** For an FP8 model,
   ``config.dtype == DType.float8_e4m3fn`` is the *dispatch* dtype. The
   un-quantized sections (norms, biases, embeddings, attention if not in
   ``attn_quantized_layers``, the MoE router gate) are bf16 on disk.
   Override `_module_default_dtype()` to return `DType.bfloat16` so those
   parameters build as bf16, as `deepseekV3_modulev3/model.py` does.

3. **HF wraps multimodal configs.** Vision or conditional-generation
   config types nest the text backbone
   under ``.text_config``. Framework methods (``calculate_max_seq_len`` on
   the config class, ``get_kv_params`` on the pipeline model class) read the
   *parent* config by default. Override them to pass ``.text_config``, as
   ``kimik2_5_modulev3/model.py::get_kv_params`` does.

4. **Quantization config ignore-list prefix mismatch.** Compressed-
   tensors HF configs store ignore entries with the original prefix
   (``model.language_model.layers.X.self_attn.q_proj``). MAX's
   ``parse_quant_config`` checks them against ``ignored_modules_prefix``,
   which defaults to ``model.`` (``model.layers.X.self_attn.q_proj``). For a
   multimodal-wrapped model, pass
   ``ignored_modules_prefix="model.language_model."``, as
   ``gemma4/model_config.py`` does.

5. **RoPE tables on the mesh.** Move `freqs_cis` to the mesh before use
   (`rope.freqs_cis.cast(dtype).to(mesh)`, as
   `Gemma3TextModel.prepare_freq_cis()` does). For NoPE layers, build an
   identity table (cos=1, sin=0) next to the real one and select it per layer
   with the field HF's attention checks (``layer_types[i]`` or
   ``no_rope_layers[i]``, see
   [divergences.md](divergences.md#17-nope--skip-rope-layers-via-identity-freqs_cis)).
   The identity table must match the layout of ``rope.freqs_cis``, including the
   ``max_seq_len * 2`` row count (the rotary embedding pre-allocates 2× for
   decode positions past prefill).

6. **GPU memory zombies after ``pkill max serve``.** A multiprocessing
   spawn worker can survive ``kill -9`` and hold HBM as a defunct
   (Z-state) process. Symptom: later serve attempts see less
   free HBM than expected on each device.
