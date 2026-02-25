# Max Best Practices Error Index

This index maps common error messages to relevant patterns.

## Error Message Lookup

| Error Message | Pattern | Category |
|--------------|---------|----------|
| `API error` | [serve-api](../patterns/serve-api.md) | serve |
| `Cannot compile module with uninitialized parameters` | [model-implementation](../patterns/model-implementation.md) | model |
| `Could not find mojo kernel` | [engine-operations](../patterns/engine-operations.md) | engine |
| `DLPack` | [engine-weights](../patterns/engine-weights.md) | engine |
| `DeviceRef has no attribute 'from_device'` | [engine-operations](../patterns/engine-operations.md) | engine |
| `Expected TensorValue but got Tensor` | [model-implementation](../patterns/model-implementation.md) | model |
| `Failed to resolve module path for MOGGKernelAPI` | [engine-operations](../patterns/engine-operations.md) | engine |
| `GGUF` | [model-loading](../patterns/model-loading.md) | model |
| `GPTQ` | [engine-quantization](../patterns/engine-quantization.md) | engine |
| `GPU OOM` | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu |
| `Graph` | [graph-construction](../patterns/graph-construction.md) | graph |
| `HuggingFace` | [model-loading](../patterns/model-loading.md) | model |
| `Invalid LoRA adapter path` | [engine-lora](../patterns/engine-lora.md) | engine |
| `KV cache` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `KeyError` | [model-implementation](../patterns/model-implementation.md) | model |
| `LoRA` | [serve-api](../patterns/serve-api.md) | serve |
| `LoRA adapter contains unsupported target modules` | [engine-lora](../patterns/engine-lora.md) | engine |
| `LoRA adapter name already exists` | [engine-lora](../patterns/engine-lora.md) | engine |
| `LoRA functionality is not enabled on this server` | [engine-lora](../patterns/engine-lora.md) | engine |
| `LoRA of rank exceeds maximum rank` | [engine-lora](../patterns/engine-lora.md) | engine |
| `LoRA only supports files in safetensors format` | [engine-lora](../patterns/engine-lora.md) | engine |
| `Module has no attribute 'forward'` | [model-implementation](../patterns/model-implementation.md) | model |
| `NCCL` | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu |
| `OOM` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `Shape mismatch` | [model-implementation](../patterns/model-implementation.md) | model |
| `TTFT` | [deploy-production](../patterns/deploy-production.md) | serve |
| `TensorType` | [graph-construction](../patterns/graph-construction.md) | graph |
| `TensorType missing required argument 'device'` | [engine-operations](../patterns/engine-operations.md) | engine |
| `Weight not found in state dict` | [model-implementation](../patterns/model-implementation.md) | model |
| `accuracy` | [engine-quantization](../patterns/engine-quantization.md) | engine |
| `architecture` | [model-loading](../patterns/model-loading.md) | model |
| `batch` | [perf-inference](../patterns/perf-inference.md) | perf |
| `batch size exceeds` | [serve-configuration](../patterns/serve-configuration.md) | serve |
| `buffer` | [engine-weights](../patterns/engine-weights.md) | engine |
| `cache hit` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `communication` | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu |
| `compilation` | [graph-construction](../patterns/graph-construction.md) | graph |
| `connection closed` | [serve-api](../patterns/serve-api.md) | serve |
| `container` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `context length exceeded` | [serve-configuration](../patterns/serve-configuration.md) | serve |
| `custom() missing required argument 'device'` | [engine-operations](../patterns/engine-operations.md) | engine |
| `deployment failed` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `device` | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu |
| `device mismatch` | [engine-weights](../patterns/engine-weights.md) | engine |
| `disconnect` | [serve-api](../patterns/serve-api.md) | serve |
| `docker` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `error propagation` | [serve-api](../patterns/serve-api.md) | serve |
| `float8` | [engine-quantization](../patterns/engine-quantization.md) | engine |
| `fp8` | [engine-quantization](../patterns/engine-quantization.md) | engine |
| `function calling` | [serve-api](../patterns/serve-api.md) | serve |
| `health check` | [serve-api](../patterns/serve-api.md) | serve |
| `image` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `kernel compilation failed` | [engine-operations](../patterns/engine-operations.md) | engine |
| `kubernetes` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `kv-cache` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `latency` | [perf-inference](../patterns/perf-inference.md) | perf |
| `latency` | [deploy-production](../patterns/deploy-production.md) | serve |
| `load_state_dict` | [model-implementation](../patterns/model-implementation.md) | model |
| `loading failed` | [model-loading](../patterns/model-loading.md) | model |
| `memory exhausted` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `metrics` | [deploy-production](../patterns/deploy-production.md) | serve |
| `missing keys in state_dict` | [model-implementation](../patterns/model-implementation.md) | model |
| `model not found` | [model-loading](../patterns/model-loading.md) | model |
| `model too large` | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu |
| `module` | [graph-construction](../patterns/graph-construction.md) | graph |
| `monitoring` | [deploy-production](../patterns/deploy-production.md) | serve |
| `mount` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `multi-GPU` | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu |
| `no matching function in call to 'forward'` | [model-implementation](../patterns/model-implementation.md) | model |
| `out of memory` | [serve-configuration](../patterns/serve-configuration.md) | serve |
| `out of memory` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `page size` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `pod` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `precision loss` | [engine-quantization](../patterns/engine-quantization.md) | engine |
| `preemption` | [serve-api](../patterns/serve-api.md) | serve |
| `prefill` | [perf-inference](../patterns/perf-inference.md) | perf |
| `prefix caching` | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve |
| `quantization` | [engine-quantization](../patterns/engine-quantization.md) | engine |
| `request cancelled` | [serve-api](../patterns/serve-api.md) | serve |
| `scale` | [engine-quantization](../patterns/engine-quantization.md) | engine |
| `scheduling error` | [serve-configuration](../patterns/serve-configuration.md) | serve |
| `shape mismatch` | [graph-construction](../patterns/graph-construction.md) | graph |
| `sharding` | [engine-weights](../patterns/engine-weights.md) | engine |
| `slow inference` | [perf-inference](../patterns/perf-inference.md) | perf |
| `stream` | [serve-api](../patterns/serve-api.md) | serve |
| `streaming` | [serve-api](../patterns/serve-api.md) | serve |
| `structured output` | [serve-api](../patterns/serve-api.md) | serve |
| `swapping` | [perf-inference](../patterns/perf-inference.md) | perf |
| `symbolic dimension` | [graph-construction](../patterns/graph-construction.md) | graph |
| `telemetry` | [deploy-production](../patterns/deploy-production.md) | serve |
| `tensor parallel` | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu |
| `throughput` | [perf-inference](../patterns/perf-inference.md) | perf |
| `throughput` | [deploy-production](../patterns/deploy-production.md) | serve |
| `timeout` | [serve-api](../patterns/serve-api.md) | serve |
| `token` | [model-loading](../patterns/model-loading.md) | model |
| `token budget` | [serve-api](../patterns/serve-api.md) | serve |
| `token limit exceeded` | [serve-configuration](../patterns/serve-configuration.md) | serve |
| `transfer` | [engine-weights](../patterns/engine-weights.md) | engine |
| `unable to locate module 'max'` | [engine-operations](../patterns/engine-operations.md) | engine |
| `unexpected keys in state_dict` | [model-implementation](../patterns/model-implementation.md) | model |
| `unsupported model` | [model-loading](../patterns/model-loading.md) | model |
| `volume` | [deploy-production](../patterns/deploy-production.md) | deploy |
| `weight` | [engine-weights](../patterns/engine-weights.md) | engine |
| `worker` | [deploy-production](../patterns/deploy-production.md) | serve |

## Patterns by Error Count

| Pattern | Category | Error Patterns Covered |
|---------|----------|----------------------|
| [model-implementation](../patterns/model-implementation.md) | model | 10 |
| [deploy-production](../patterns/deploy-production.md) | deploy, serve | 15 |
| [serve-api](../patterns/serve-api.md) | serve | 14 |
| [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | 8 |
| [engine-operations](../patterns/engine-operations.md) | engine | 7 |
| [engine-quantization](../patterns/engine-quantization.md) | engine | 7 |
| [model-loading](../patterns/model-loading.md) | model | 7 |
| [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | 7 |
| [engine-lora](../patterns/engine-lora.md) | engine | 6 |
| [engine-weights](../patterns/engine-weights.md) | engine | 6 |
| [graph-construction](../patterns/graph-construction.md) | graph | 6 |
| [perf-inference](../patterns/perf-inference.md) | perf | 6 |
| [serve-configuration](../patterns/serve-configuration.md) | serve | 5 |

---
