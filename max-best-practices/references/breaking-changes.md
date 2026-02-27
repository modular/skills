# MAX Breaking Changes Reference

> Updated for MAX v26.1.0.0.0+


## v24.2.1 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `max.graph.ops` | - | - | You can now import more MAX Graph functions from `max.gra... |

| `List` | - | - | And, of course, lots of Mojo updates, including implicit ... |
## v24.1 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `OlmoForCausalLM` | - | - | `OlmoForCausalLM` (such as `allenai/OLMo-1B-0724-hf`) |
| `GraniteForCausalLM` | - | - | `GraniteForCausalLM` (such as `ibm-granite/granite-3.1-8b... |
| `Phi3ForCausalLM` | - | - | `Phi3ForCausalLM` (for Microsoft Phi-3 models) |
| `max.pipelines.dataprocessing.tokenizer` | - | - | The `max.pipelines.dataprocessing.tokenizer` and `max.pip... |
| `PipelineConfig.architecture` | - | - | The previously deprecated `PipelineConfig.architecture` f... |
| `--devices` | - | - | The `--devices` CLI argument now supports a comma-separat... |
| `max benchmark` | - | - | The `max benchmark` tool, which runs MLPerf benchmarks on... |
| `max visualize` | - | - | The `max visualize` tool, which allows you to visualize y... |
| `mojo` | - | - | The latest version of Mojo, the standard library, and the... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `--huggingface-revision` | Added `--huggingface-revision` option, to allow selecting... | |

## v24.3 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `None` | - | - | Added support for named dynamic dimensions. This means yo... |
| `Type` | `AnyMoType` | - | `AnyMoType` renamed to `Type`, `MOTensor` renamed to `Ten... |
| `ElementType` | - | - | Removed `ElementType` in favor of using `DType`. |
| `TypeTuple` | - | - | Removed `TypeTuple` in favor of using `List[Type]`. |
| `Module` | - | - | Removed the `Module` type so you can now start building a... |
| `max.ops` | - | - | Some new ops in `max.ops`, including support for custom o... |
| `TorchInputSpec` | - | - | `TorchInputSpec` now supports named dynamic dimensions (p... |
| `InferenceSession.load_model()` | - | - | `InferenceSession.load_model()` was renamed to `load()`. |
| `max.engine.engine` | - | - | `max.engine.engine` module was renamed to `max.engine.info`. |
| `CommonLoadOptions` | - | - | Removed the Python `CommonLoadOptions`, `TorchLoadOptions... |
| `LoadOptions` | - | - | Removed the Mojo `LoadOptions` type. See the note above a... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `InputSpec` | Redesigned `InferenceSession.load()` to replace the confu... | |
| `ShapeElement` | New `ShapeElement` type to allow for named dynamic dimens... | |
| `dimNames` | `M_newTorchInputSpec()` now supports named dynamic dimens... | |

## v24.4 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `EngineNumpyView.data()` | - | - | `EngineNumpyView.data()` and `EngineTensorView.data()` fu... |
| `TensorMap` | - | - | `TensorMap` now conforms to `CollectionElement` trait to ... |
| `custom_nv()` | - | - | `custom_nv()` was removed, and its functionality moved in... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `max.graph.checkpoint` | Added `max.graph.checkpoint` package to save and load mod... | |
| `BFloat16Encoding` | Added quantization encodings `BFloat16Encoding`, `Q4_0Enc... | |
| `QuantizationEncoding` | Added the `QuantizationEncoding` trait so you can build c... | |
| `Graph.quantize()` | Added `Graph.quantize()` to create a quantized tensor node. | |
| `qmatmul()` | Added `qmatmul()` to perform matrix-multiplication with a... | |
| `avg_pool()` | `avg_pool()` | |
| `max_pool()` | `max_pool()` | |
| `conv2d()` | `conv2d()` | |
| `conv3d()` | `conv3d()` | |
| `layer_norm()` | `layer_norm()` | |
| `tile()` | `tile()` | |
| `select()` | `select()` | |
| `layer()` | Added a `layer()` context manager and `current_layer()` f... | |
| `format_system_stack()` | Added `format_system_stack()` function to format the stac... | |
| `TensorMap.keys()` | Added `TensorMap.keys()` to get all the tensor key names. | |
| `M_cloneCompileConfig()` | `M_cloneCompileConfig()` | |
| `M_copyAsyncTensorMap()` | `M_copyAsyncTensorMap()` | |
| `M_tensorMapKeys()` | `M_tensorMapKeys()` and `M_deleteTensorMapKeys()` | |
| `M_setTorchLibraries()` | `M_setTorchLibraries()` | |

## v24.6 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `llama3` | - | - | Upgraded MAX models As we continue to build our Python-ba... |
| `/completion` | - | - | An OpenAI-compatible server with streaming `/chat/complet... |
| `TextGeneration` | - | - | Supports most `TextGeneration` Hugging Face Hub models. |
| `AttentionWithRope` | - | - | Attention layers (including highly optimized implementati... |
| `Transformer` | - | - | Transformers such as `Transformer` and `TransformerBlock`... |
| `RMSNorm` | - | - | Common layers such as `RMSNorm` , `Embedding`, and `Seque... |
| `fused_qk_ragged_rope` | - | - | Low-level wrappers over optimized kernels like `fused_qk_... |
| `TextGenerationPipeline` | - | - | An optimized `TextGenerationPipeline` that can be combine... |
| `HFTextGenerationPipeline` | - | - | A generic `HFTextGenerationPipeline` that can run any Hug... |
| `session.load()` | - | - | Models now accept weights via a weights registry, which i... |
| `LlamaForCausalLM` | - | - | Significant performance upgrades for Llama 3, and expande... |
| `MistralForCausalLM` | - | - | Mistral NeMo (and other `MistralForCausalLM` models) |
| `top-k` | - | - | Using a large setting for `top-k` with the Llama 3.1 mode... |
| `max_seq_len` | - | - | The models currently use a smaller default context window... |
| `LlamaForCausalLM` | - | - | Some variants of the supported core models (like `LlamaFo... |
| `Model` | - | - | The `Model` object is now callable to run an inference. |
| `execute()` | - | - | The `execute()` function currently doesn't accept keyword... |
| `execute_legacy()` | - | - | `execute_legacy()` preserves the semantics of `execute()`... |
| `execute()` | - | - | Calling `execute()` with positional arguments still works... |
| `max.driver.Device` | - | - | Expanded functionality for `max.driver.Device`, with new ... |
| `driver.Tensor` | - | - | `driver.Tensor` (and the `InferenceSession.load()` argume... |
| `driver.Tensor` | - | - | `driver.Tensor` has new methods, such as `from_dlpack()`,... |
| `max.graph` | - | - | The `max.graph` APIs now include preliminary support for ... |
| `DType` | - | - | `DType` has new methods/properties, `align`, `size_in_byt... |
| `Value` | - | - | `Value` constructor accepts more types for `value`. |
| `ops.scalar()` | - | - | Instead of `ops.scalar()`, use `ops.constant()`. |
| `is_static()` | - | - | Instead of `is_static()` and `is_symbolic()`, use `isinst... |
| `matmul` | - | - | The custom op APIs will allow you to extend MAX Engine wi... |
| `NDBuffer` | - | - | The new API requires far less adornment at the definition... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `CUDA()` | Added `CUDA()` device to open an NVIDIA GPU. | |
| `BufferValue` | Added `BufferValue`, which allows for device-resident ten... | |
| `message` | `TensorValue.rebind()` accepts a new `message` argument. | |
| `align` | The `Weight` constructor arguments changed; added `align`... | |

## v25.1 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `Llama-3.2-11B-Vision-Instruct` | - | - | Image inputs for multimodal models such as `Llama-3.2-11B... |
| `response_format` | - | - | Structured output (also known as constrained decoding), w... |
| `/v1/embeddings` | - | - | Added support for the `/v1/embeddings` API endpoint, allo... |
| `--cache-strategy=paged` | - | - | Added support for PagedAttention, which improves memory e... |
| `--enable-prefix-caching` | - | - | Added support for prefix caching in all cases where Paged... |
| `max-pipelines generate` | - | - | `max-pipelines generate` now accepts image input with `--... |
| `LlavaForConditionalGeneration` | - | - | Added an experimental Pixtral pipeline you can run as fol... |
| `MllamaForConditionalGeneration` | - | - | Added an experimental Llama Vision pipeline you can run a... |
| `Qwen2ForCausalLM` | - | - | Added support for the `Qwen2ForCausalLM` model architectu... |
| `examples/offline-inference/basic.py` | - | - | Added support for offline batched inference for text-base... |
| `--max-cache-batch-size` | - | - | The `--max-cache-batch-size` flag has been deprecated in ... |
| `custom()` | - | - | You can now write custom operations (ops) in Mojo, and ad... |
| `Graph.add_weight()` | - | - | `Graph.add_weight()` now takes an explicit `device` argum... |
| `max.graph.Weight` | - | - | `max.graph.Weight` now inherits from `TensorValue`, allow... |
| `TextTokenizer.new_context()` | - | - | `TextTokenizer.new_context()` now supports tool definitio... |
| `num_steps` | - | - | Removed the default `num_steps` value for `TokenGenerator... |
| `@compiler.register` | - | - | The `@compiler.register` decorator is applied to a Mojo s... |
| `foreach()` | - | - | The `foreach()` function, which efficiently executes an e... |
| `custom()` | - | - | The `custom()` and `inplace_custom()` functions allow you... |
| `InferenceSession` | - | - | The `InferenceSession` constructor accepts the custom op ... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `nn.Conv1D` | Added `nn.Conv1D` for audio models like Whisper. | |

## v24.5 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `magic` | - | - | Mojo and MAX are magical! We've created a new package and... |
| `M_setTorchLibraries` | - | - | `M_setTorchLibraries` -> `M_setTorchLibraryPath` |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `repeat_interleave` | Added `repeat_interleave` graph op. | |
| `M_setTorchMetadataLibraryPath` | `M_setTorchMetadataLibraryPath` | |
| `M_setTorchMetadataLibraryPtr` | `M_setTorchMetadataLibraryPtr` | |

## v26.2 (Nightly)

### Breaking API Changes

| Change | Description |
|--------|-------------|
| KVCache scheduler | Requires replica index for KV cache operations |
| Large runtime layouts | Uses int64 indices for large tensor layouts |
| SM100 kernels | Unified pipeline storage and naming for Blackwell |
| `DeviceRef.from_device()` | **Deprecated** - still works but prefer `DeviceRef.CPU()` or `DeviceRef.GPU()` |
| `PipelineConfig.max_length` | **Removed** - moved to `MAXModelConfig.max_length` (access as `config.model.max_length`) |
| `PipelineModel(encoding=...)` | `encoding` parameter **removed** - automatically inferred from `quantization_encoding` |
| Device-graph capture API | Requires explicit graph key: `model.capture(graph_key, *inputs)` instead of `model.capture(*inputs)` |
| `max.nn` namespace | Graph-based API restored as default; eager module API moved to `max.nn.module_v3`; `max.tensor/functional/random` moved to `max.experimental` |
| `ops.custom()` signature | `device` is now required positional arg: `ops.custom(name, device, values, out_types)` |
| `TensorType` | Now requires `device` parameter: `TensorType(dtype, shape, device=DeviceRef.CPU())` |
| Custom op kernel imports | Use `from tensor import InputTensor, OutputTensor, foreach` (not `from max.tensor`) |
| Graph `custom_extensions` | Must be passed during Graph construction (not after) |
| `foreach` callback signature | Changed from `fn[width: Int, element_alignment: Int](idx)` to `fn[width: Int](idx)` |
| `ops.neg()` | Renamed to `ops.negate()` |
| `ops.reduce_mean()` | Renamed to `ops.mean()` with different API |
| `ops.clamp()` | Not available - use `ops.max(ops.min(x, high), low)` workaround |
| `AlgebraicDim` | Use strings in TensorType shape instead: `["batch", 128]` |
| `Graph.verify()` | Removed - no longer available |

### New Features (v26.2 Nightly)

| Feature | Description |
|---------|-------------|
| Overlap scheduling | Enabled by default for select architectures; reduces CPU overhead. Disable with `--no-enable-overlap-scheduler --force`. Incompatible with structured outputs and CPU models. |
| `DeviceEvent(enable_timing=True)` | GPU event timing support; use `start.elapsed_time(end)` for GPU time measurement |
| Global MLIR context | Now active by default; eliminates per-graph context plumbing requirement |
| `max.graph.ops.prod` | New op for computing element products along axes |
| `MODULAR_NVPTX_COMPILER_PATH` | Env var to point to system `ptxas` instead of bundled `libnvptxcompiler` |
| CUDA 13.1 | Upgraded from CUDA 12.9; **minimum NVIDIA driver raised to 580** (Turing/sm_75+) |
| AMD RDNA consumer GPUs | Common MAX models now run on AMD RDNA consumer GPUs |
| Olmo3ForCausalLM | New architecture support |
| Qwen3-30B-A3B MOE | Multi-GPU tensor parallelism support |
| Multi-GPU TP for GPT-OSS | Tensor parallelism for GPT-OSS architecture |
| Legacy Gemma 3 multimodal | **Removed** |

**Critical: Version Alignment**

MAX Python package and Mojo must be version-aligned. Mismatched versions cause kernel compilation failures.

### Quick Version Check

```bash
# Manual check - versions must align
mojo --version          # e.g., Mojo 0.25.7.0 → need MAX 25.7
pip show max | grep Version  # e.g., 25.7.0
```

### Version Alignment Rules

| Mojo Version | Required MAX Version |
|--------------|---------------------|
| `0.25.7` | `25.7.x` |
| `0.26.1.0.0` | `26.1.0.x` |
| `0.26.2` | `26.2.x` |

### Common Mismatch Errors

**Error: `no matching function in call to 'foreach'`**
```
note: callee parameter 'func' has 'fn[width: Int, element_alignment: Int](IndexList[rank])' type,
      but value has type 'fn[width: Int](idx: IndexList[rank])'
```
- **Cause:** Using nightly callback signature (`fn[width: Int]`) with stable MAX (requires `fn[width: Int, element_alignment: Int]`)
- **Fix:** Check version alignment, use correct signature for your version

**Error: `DeviceRef has no attribute 'from_device'`**
- **Cause:** Using stable API with nightly MAX
- **Fix:** Use `DeviceRef.CPU()` or `DeviceRef.GPU()` on nightly

**Error: `Buffer not found in max.driver`**
- **Cause:** Using nightly API (`Buffer`) with stable MAX
- **Fix:** Use `max.driver.Tensor` on stable v26.1.0.0.0

### How to Fix Mismatches

**Option 1: Use pixi (recommended)**
```bash
# pixi manages both Mojo and MAX versions together
pixi shell  # Always work inside the shell
pixi list | grep -E "^(max|mojo)"  # Verify aligned versions
```

**Option 2: Align pip install with Mojo**
```bash
# For stable Mojo 0.25.7:
pip install max==25.7.0

# For nightly:
pip install --upgrade max --index-url https://whl.modular.com/nightly/simple/
```

**Option 3: Remove conflicting global installs**
```bash
pip uninstall max  # Remove global install
# Then use pixi exclusively
```

## v26.1.0.0.0

| Change | Description |
|--------|-------------|
| `--max-batch-size` semantics | Now per-replica with data parallelism (was aggregate) |
| `--max-ce-batch-size` | Deprecated, use `--max-batch-size` |
| `max.driver.Tensor` | Renamed to `max.driver.Buffer` |
| `prefill_chunk_size` | Renamed to `max_batch_input_tokens` |
| `max_batch_context_length` | Renamed to `max_batch_total_tokens` |
| `--kvcache-ce-watermark` | New option for KVCache scheduling (default 0.95) |
| Llama 3.2 Vision | **Removed** - use Pixtral, InternVL, or Qwen2.5-VL |
| Gemma3 Vision | Added support for 12B and 27B multimodal variants |
| `accelerator_count()` | Now returns non-zero on Apple silicon |
| Stream API | All streams non-blocking (`blocking` arg removed) |
| V1 layer classes | **Removed** (`Conv2dV1`, `LinearV1`, etc.) |
| Python wheels | URL changed to `https://whl.modular.com/nightly/simple/` |
| `max.engine.MojoValue` | Removed |
| `custom_ops_path` | Removed from `InferenceSession.__init__` |

## v26.1.0.0.0 (Stable)

| Change | Description |
|--------|-------------|
| `--do-penalties` | Renamed to `--enable-penalties` (now default) |
| Removed `Conv2dV1`, `LinearV1`, etc. | Use `Conv2d`, `Linear` instead |
| `max.engine.MojoValue` | Removed |
| `custom_ops_path` in InferenceSession | Removed |
| `foreach` callback signature | Requires `element_alignment: Int`: `fn[width: Int, element_alignment: Int](idx)` |
| `max.driver.Buffer` | Use `max.driver.Tensor` (Buffer renamed to Tensor) |
| `KVCacheStrategy` import | Import from `max.kv_cache.registry` or `max.nn.kv_cache` |

## v25.6

| Change | Description |
|--------|-------------|
| `KVCacheStrategy.CONTINUOUS` | Deprecated, use `PAGED` |
| `ContinuousBatchingKVCacheManager` | Removed |
| `InputContext` | Replaced by `TextGenerationContext`, `EmbeddingsContext` |
| `llguidance` | Replaced `XGrammar` for structured output |

## v25.5

| Change | Description |
|--------|-------------|
| `torch` dependency | Removed from MAX package |
| `PipelineEngine.HUGGINGFACE` | Removed (HuggingFace fallback) |

## v25.4

| Change | Description |
|--------|-------------|
| `max.nn` deprecated layers | Marked as `V1`, new layers are default |
| `ops.select` | Renamed to `ops.where` |
| `MojoCallContextPtr` | Replaced by `DeviceContextPtr` |
| Custom ops | Now use `InputTensor`/`OutputTensor` (not `ManagedTensorSlice`) |

## v25.3

| Change | Description |
|--------|-------------|
| `max-pipelines` CLI | Renamed to `max` |
| `--use-gpu` | Deprecated, use `--devices gpu:0` |
| `--devices=gpu-N` | Changed to `--devices gpu:0,1,2,3` |

## v25.2

| Change | Description |
|--------|-------------|
| `--huggingface-repo-id` | Removed, use `--model` |
| `Model.execute()` | Signature changed for GPU support |
| TorchScript models | Removed support |

## CLI Flag Renames Summary

| Old Flag | New Flag | Version |
|----------|----------|---------|
| `--model-path` | `--model` (preferred; `--model-path` still accepted) | v26.1+ |
| `--use-gpu` | `--devices gpu:0` | v25.3 |
| `--max-ce-batch-size` | `--max-batch-size` | v26.1.0.0.0 |
| `--do-penalties` | `--enable-penalties` | v26.1.0.0.0 |
| `--prefill-chunk-size` | `--max-batch-input-tokens` | v26.1.0.0.0 |
| `--max-batch-context-length` | `--max-batch-total-tokens` | v26.1.0.0.0 |
| `--ignore-eos` | Use HTTP request payload | v25.6 |

## API Renames Summary

| Old API | New API | Version |
|---------|---------|---------|
| `PipelineConfig.max_length` | `config.model.max_length` (via `MAXModelConfig`) | v26.2 |
| `DeviceRef.from_device(device)` | `DeviceRef.CPU()` / `DeviceRef.GPU()` (deprecated, not removed) | v26.2 |
| `ops.custom(name, values, ...)` | `ops.custom(name, device, values, ...)` | v26.2 |
| `from max.tensor import InputTensor` | `from tensor import InputTensor` | v26.2 |
| `TensorType(dtype, shape)` | `TensorType(dtype, shape, device=...)` | v26.2 |
| `max.driver.Tensor` | `max.driver.Buffer` | v26.1.0.0.0 |
| `prefill_chunk_size` | `max_batch_input_tokens` | v26.1.0.0.0 |
| `max_batch_context_length` | `max_batch_total_tokens` | v26.1.0.0.0 |
| `InputContext` | `TextGenerationContext` | v25.6 |
| `ops.select` | `ops.where` | v25.4 |
| `ops.neg()` | `ops.negate()` | v26.2 |
| `ops.reduce_mean()` | `ops.mean()` | v26.2 |
| `ops.clamp()` | `ops.max(ops.min(x, high), low)` (workaround) | v26.2 |
| `AlgebraicDim("name")` | Strings in shape: `["name", 128]` | v26.2 |
| `Graph.verify()` | Removed (no replacement) | v26.2 |

## Vision Model Support

| Model | Stable (v26.1.0.0.0) | Nightly (v26.2) |
|-------|----------------|-----------------|
| Llama 3.2 Vision | Supported | **Removed** |
| Gemma3 Vision (12B, 27B) | Not available | Supported |
| Pixtral | Supported | Supported |
| InternVL | Supported | Supported |
| Qwen2.5-VL | Supported | Supported |
