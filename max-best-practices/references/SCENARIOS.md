# Max Best Practices Scenario Index

This index maps common tasks and scenarios to relevant patterns.

## Task Lookup

| Task/Scenario | Pattern | Category | Impact |
|--------------|---------|----------|--------|
| Add a custom architecture to MAX Serve | [model-implementation](../patterns/model-implementation.md) | model | HIGH |
| Benchmark production deployment | [deploy-production](../patterns/deploy-production.md) | deploy | HIGH |
| Build MAX Graph from scratch | [graph-construction](../patterns/graph-construction.md) | graph | HIGH |
| Build a transformer model with MAX nn | [model-implementation](../patterns/model-implementation.md) | model | HIGH |
| Build complete custom model with MAX | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Check if model is supported | [model-loading](../patterns/model-loading.md) | model | HIGH |
| Choose KV cache strategy | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | CRITICAL |
| Compile a model for inference | [model-implementation](../patterns/model-implementation.md) | model | HIGH |
| Configure Docker container | [deploy-production](../patterns/deploy-production.md) | deploy | HIGH |
| Configure Float8 quantization | [engine-quantization](../patterns/engine-quantization.md) | engine | HIGH |
| Configure HF token for gated models | [model-loading](../patterns/model-loading.md) | model | HIGH |
| Configure KV cache for MAX Serve | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | CRITICAL |
| Configure MAX Serve for high throughput | [serve-configuration](../patterns/serve-configuration.md) | serve | CRITICAL |
| Configure chunked prefill | [perf-inference](../patterns/perf-inference.md) | perf | HIGH |
| Configure health endpoints | [serve-api](../patterns/serve-api.md) | serve | HIGH |
| Configure inference session | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Configure lazy context | [graph-construction](../patterns/graph-construction.md) | graph | HIGH |
| Configure metrics collection | [deploy-production](../patterns/deploy-production.md) | serve | HIGH |
| Configure preemption behavior | [serve-api](../patterns/serve-api.md) | serve | CRITICAL |
| Configure scale granularity | [engine-quantization](../patterns/engine-quantization.md) | engine | HIGH |
| Configure scheduling priorities | [serve-configuration](../patterns/serve-configuration.md) | serve | CRITICAL |
| Configure tensor parallelism | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | CRITICAL |
| Configure token budgets | [serve-api](../patterns/serve-api.md) | serve | HIGH |
| Configure volume mounts | [deploy-production](../patterns/deploy-production.md) | deploy | HIGH |
| Configure weight sharding strategy | [engine-weights](../patterns/engine-weights.md) | engine | HIGH |
| Create custom MAX operation | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Create mixed Mojo/Python model project | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Create reusable modules | [graph-construction](../patterns/graph-construction.md) | graph | HIGH |
| Create weight adapter | [engine-weights](../patterns/engine-weights.md) | engine | HIGH |
| Debug performance issues | [deploy-production](../patterns/deploy-production.md) | serve | HIGH |
| Debug streaming failures | [serve-api](../patterns/serve-api.md) | serve | CRITICAL |
| Define custom layers with MAX Module | [model-implementation](../patterns/model-implementation.md) | model | HIGH |
| Deploy MAX Serve in production | [deploy-production](../patterns/deploy-production.md) | deploy | HIGH |
| Deploy large model on multiple GPUs | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | CRITICAL |
| Deploy on AMD MI300X | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | CRITICAL |
| Deploy on AWS/Azure/GCP | [deploy-production](../patterns/deploy-production.md) | deploy | HIGH |
| Enable H100 FP8 | [engine-quantization](../patterns/engine-quantization.md) | engine | HIGH |
| Enable in-flight batching | [perf-inference](../patterns/perf-inference.md) | perf | HIGH |
| Enable prefix caching | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | CRITICAL |
| Enable ragged batching | [serve-configuration](../patterns/serve-configuration.md) | serve | CRITICAL |
| Enable streaming responses | [serve-api](../patterns/serve-api.md) | serve | HIGH |
| Fix OOM during inference | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | CRITICAL |
| Fix batch size errors | [serve-configuration](../patterns/serve-configuration.md) | serve | CRITICAL |
| Fix custom ops API error | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Fix graph compilation errors | [graph-construction](../patterns/graph-construction.md) | graph | HIGH |
| Fix model loading errors | [model-loading](../patterns/model-loading.md) | model | HIGH |
| Fix multi-GPU communication | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | CRITICAL |
| Fix quantization accuracy issues | [engine-quantization](../patterns/engine-quantization.md) | engine | HIGH |
| Handle client disconnects | [serve-api](../patterns/serve-api.md) | serve | CRITICAL |
| Handle request cancellation | [serve-api](../patterns/serve-api.md) | serve | CRITICAL |
| Implement a new model architecture in MAX | [model-implementation](../patterns/model-implementation.md) | model | HIGH |
| Implement function calling | [serve-api](../patterns/serve-api.md) | serve | HIGH |
| Implement pipeline model | [graph-construction](../patterns/graph-construction.md) | graph | HIGH |
| Improve throughput | [perf-inference](../patterns/perf-inference.md) | perf | HIGH |
| Integrate Mojo layers with Python serving | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Load GGUF quantized model | [model-loading](../patterns/model-loading.md) | model | HIGH |
| Load HuggingFace weights into a MAX model | [model-implementation](../patterns/model-implementation.md) | model | HIGH |
| Load and unload LoRA adapters at runtime | [engine-lora](../patterns/engine-lora.md) | engine | MEDIUM |
| Load model from HuggingFace | [model-loading](../patterns/model-loading.md) | model | HIGH |
| Manage LoRA adapters | [serve-api](../patterns/serve-api.md) | serve | HIGH |
| Manage request timeouts | [serve-api](../patterns/serve-api.md) | serve | CRITICAL |
| Monitor MAX Serve deployment | [deploy-production](../patterns/deploy-production.md) | serve | HIGH |
| Monitor worker lifecycle | [deploy-production](../patterns/deploy-production.md) | serve | HIGH |
| Optimize cache hit rate | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | CRITICAL |
| Optimize for NVIDIA Hopper | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | CRITICAL |
| Optimize inference performance | [perf-inference](../patterns/perf-inference.md) | perf | HIGH |
| Port a PyTorch model to MAX | [model-implementation](../patterns/model-implementation.md) | model | HIGH |
| Propagate errors correctly | [serve-api](../patterns/serve-api.md) | serve | CRITICAL |
| Reduce latency | [perf-inference](../patterns/perf-inference.md) | perf | HIGH |
| Reduce model memory usage | [engine-quantization](../patterns/engine-quantization.md) | engine | HIGH |
| Register new model architecture | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Run offline batch inference | [engine-operations](../patterns/engine-operations.md) | engine | HIGH |
| Select GPU devices | [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | CRITICAL |
| Serve model with LoRA adapters | [engine-lora](../patterns/engine-lora.md) | engine | MEDIUM |
| Set optimal batch size | [serve-configuration](../patterns/serve-configuration.md) | serve | CRITICAL |
| Set up KV cache swapping | [perf-inference](../patterns/perf-inference.md) | perf | HIGH |
| Set up Kubernetes deployment | [deploy-production](../patterns/deploy-production.md) | deploy | HIGH |
| Set up telemetry | [deploy-production](../patterns/deploy-production.md) | serve | HIGH |
| Shard weights across GPUs | [engine-weights](../patterns/engine-weights.md) | engine | HIGH |
| Track TTFT and latency | [deploy-production](../patterns/deploy-production.md) | serve | HIGH |
| Transfer buffers between devices | [engine-weights](../patterns/engine-weights.md) | engine | HIGH |
| Tune KV cache memory | [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | CRITICAL |
| Use DLPack interop | [engine-weights](../patterns/engine-weights.md) | engine | HIGH |
| Use GPTQ 4-bit inference | [engine-quantization](../patterns/engine-quantization.md) | engine | HIGH |
| Use different LoRA adapters per request | [engine-lora](../patterns/engine-lora.md) | engine | MEDIUM |
| Use structured output | [serve-api](../patterns/serve-api.md) | serve | HIGH |
| Use symbolic dimensions | [graph-construction](../patterns/graph-construction.md) | graph | HIGH |

## Patterns by Scenario Count

| Pattern | Category | Scenarios Covered |
|---------|----------|-------------------|
| [engine-operations](../patterns/engine-operations.md) | engine | 8 |
| [model-implementation](../patterns/model-implementation.md) | model | 7 |
| [deploy-production](../patterns/deploy-production.md) | deploy, serve | 12 |
| [serve-api](../patterns/serve-api.md) | serve | 12 |
| [engine-quantization](../patterns/engine-quantization.md) | engine | 6 |
| [graph-construction](../patterns/graph-construction.md) | graph | 6 |
| [multigpu-scaling](../patterns/multigpu-scaling.md) | multigpu | 6 |
| [perf-inference](../patterns/perf-inference.md) | perf | 6 |
| [serve-kv-cache](../patterns/serve-kv-cache.md) | serve | 6 |
| [engine-weights](../patterns/engine-weights.md) | engine | 5 |
| [model-loading](../patterns/model-loading.md) | model | 5 |
| [serve-configuration](../patterns/serve-configuration.md) | serve | 5 |
| [engine-lora](../patterns/engine-lora.md) | engine | 3 |

---
