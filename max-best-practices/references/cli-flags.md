# MAX CLI Flags Reference

> `max serve` and `max generate` flags for MAX v26.1.0.0.0+

## Model Selection

| Flag | Description | Example |
|------|-------------|---------|
| `--model` | HuggingFace repo or local path | `google/gemma-3-27b-it` |
| `--huggingface-revision` | Specific branch or commit | `main` |
| `--quantization-encoding` | Weight format | `bfloat16`, `float32`, `q4_k` |
| `--trust-remote-code` | Allow custom model code from HF | (flag, no value) |

> **Note:** `--model-path` is accepted as a legacy alias for `--model`. Docker containers also accept `--model-path`.

## Device Configuration

| Flag | Description | Example |
|------|-------------|---------|
| `--devices` | GPU selection | `gpu:0,1,2,3` |
| `--data-parallel-degree` | Number of DP replicas | `2` |

## Batch Configuration

| Flag | Description | Default |
|------|-------------|---------|
| `--max-batch-size` | Max requests per batch (per replica) | Auto |
| `--max-batch-input-tokens` | Max tokens in prefill batch | Auto |
| `--max-batch-total-tokens` | Max total tokens in batch | Auto |

## KV Cache

| Flag | Description | Default |
|------|-------------|---------|
| `--kv-cache-page-size` | Page size (multiple of 128) | `128` |
| `--enable-prefix-caching` | Enable prefix cache | `false` |
| `--kvcache-ce-watermark` | CE scheduling threshold (see below) | `0.95` |
| `--enable-kvcache-swapping-to-host` | Host memory offload | `false` |
| `--host-kvcache-swap-space-gb` | Host swap space size | - |

### `--kvcache-ce-watermark` Explained

**CE = Context Encoding** (the prefill phase where prompt tokens are processed).

This flag controls when the scheduler prioritizes **decode batches** over **prefill batches**.

| Value | Behavior |
|-------|----------|
| `0.95` (default) | When KV cache is 95% full, prioritize decodes over new prefills |
| `0.80` | More aggressive: start prioritizing decodes earlier |
| `1.0` | Only prioritize decodes when KV cache is completely full |

**When to tune:**
- **Lower** (0.80-0.90): If you see high TTFT (time to first token) due to decode backpressure
- **Higher** (0.95-1.0): If you have plenty of KV cache headroom and want maximum throughput
- **Default (0.95)**: Good balance for most workloads

## LoRA Adapters

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-lora` | Enable LoRA adapter support | `false` |
| `--max-num-loras` | Max adapters loaded concurrently | `8` |
| `--max-lora-rank` | Maximum LoRA rank supported | `64` |
| `--lora-paths` | Adapter paths (name=path format) | - |

**Example:**
```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-num-loras 4 \
  --max-lora-rank 32 \
  --lora-paths "sql=./adapters/sql-lora,code=./adapters/code-lora"
```

**Adapter format:** HuggingFace PEFT format with `adapter_config.json` and `adapter_model.safetensors`.

## Serving Features

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-structured-output` | JSON schema enforcement | `false` |
| `--enable-penalties` | Repetition penalties | `true` |
| `--enable-in-flight-batching` | In-flight batching | `false` |
| `--enable-chunked-prefill` | Split long prompts across batches | `false` |
| `--chat-template` | Custom Jinja2 template | - |
| `--served-model-name` | Model name alias | - |

## Performance

| Flag | Description | Default |
|------|-------------|---------|
| `--num-warmups` | Warmup steps | `0` |
| `--gpu-profiling` | Profiling level | `none` |
| `--trace` | Enable nsys tracing | `false` |
| `--trace-file` | Trace output path | - |

## Network

| Flag | Description | Default |
|------|-------------|---------|
| `--port` | HTTP port | `8000` |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for gated models |
| `MAX_SERVE_ALLOWED_IMAGE_ROOTS` | Allowed directories for file:// image URIs |
| `MAX_SERVE_SCHEDULER_STATS_LOG_INTERVAL_S` | Stats logging interval (0 = all batches) |

## Example Commands

```bash
# Basic serving (quickstart)
max serve --model google/gemma-3-27b-it

# Serve GGUF model
max serve --model modularai/Llama-3.1-8B-Instruct-GGUF

# Multi-GPU with tensor parallelism
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3 \
  --quantization-encoding bfloat16

# High-throughput configuration
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --enable-in-flight-batching \
  --max-batch-size 64

# Vision model (requires --trust-remote-code)
max serve --model google/gemma-3-27b-it --trust-remote-code

# Benchmarking
max benchmark --model meta-llama/Llama-3.1-8B-Instruct \
  --collect-gpu-stats

# Test the endpoint
curl -s http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```
