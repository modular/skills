---
title: MAX Serve Configuration
description: Comprehensive configuration for batch processing, ragged batching, scheduling, and environment settings
impact: CRITICAL
category: serve
tags: [batch, ragged-batching, scheduling, environment, configuration]
error_patterns:
  - "batch size exceeds"
  - "out of memory"
  - "token limit exceeded"
  - "scheduling error"
  - "context length exceeded"
scenarios:
  - "Configure MAX Serve for high throughput"
  - "Set optimal batch size"
  - "Enable ragged batching"
  - "Configure scheduling priorities"
  - "Fix batch size errors"
consolidates:
  - serve-batch-config.md
  - engine-ragged-batching.md
  - serve-scheduling-priority.md
  - serve-environment-config.md
---

# MAX Serve Configuration

**Category:** serve | **Impact:** CRITICAL

Comprehensive guide to configuring MAX Serve for optimal throughput, latency, and resource utilization. Covers batch configuration, ragged tensor batching for variable sequences, CE/TG scheduling priorities, and environment-based configuration.

---

## Quickstart

```bash
# Serve a model (OpenAI-compatible endpoint on port 8000)
max serve --model google/gemma-3-27b-it

# Test the endpoint
curl -s http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

> **Note:** The CLI flag is `--model` (not `--model-path`). The legacy `--model-path` alias still works. For gated models (e.g., Llama), set `HF_TOKEN` first.

---

## Core Concepts

### Batch Configuration

Configure batch size and token limits for optimal GPU utilization. Proper batch configuration can achieve 2-5x throughput improvement.

**Pattern:**

```bash
# Explicit batch size for high throughput
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 64 \
  --max-batch-input-tokens 8192
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-batch-size` | Model-dependent | Maximum requests per batch |
| `--max-batch-input-tokens` | None (auto) | Target tokens per CE batch |
| `--max-batch-total-tokens` | None | Total context limit across batch |

**v26.1 Change:** With `--data-parallel-degree`, batch size is now per-replica (was aggregate in v25.x). This aligns with vLLM and similar frameworks.

### Batch Size vs Concurrent Requests

When `batch_size < total_concurrent_requests`, the scheduler queues excess requests. Understanding this behavior is critical for capacity planning.

**Scenario: `--max-batch-size 16` with 64 concurrent requests:**
- 16 requests are processed per batch
- 48 requests wait in the queue
- Scheduler fills batches greedily from the queue
- Higher concurrency = higher queue latency but better GPU utilization

**Minimum batch size recommendations:**

| Workload Type | Min Batch Size | Rationale |
|---------------|----------------|-----------|
| Interactive chat | 4-8 | Low latency priority |
| Batch inference | 32-64 | Throughput priority |
| Mixed workload | 16-32 | Balance latency/throughput |

**Small batch guidance:**
- Batch size < 4: Poor GPU utilization, avoid for production
- Batch size 4-8: Acceptable for latency-sensitive applications
- Scheduler behavior: When batch is partially filled and timeout expires, scheduler dispatches incomplete batch

```bash
# Interactive: smaller batches for lower latency
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 8 \
  --enable-in-flight-batching

# Throughput: larger batches for better GPU utilization
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 64 \
  --max-batch-input-tokens 16384
```

> **WARNING: Batch Size Semantics Change (v26.1+)**
>
> Starting in v26.1, `--max-batch-size` is **per-replica**, not aggregate (aligns with vLLM).
>
> **Formula:** `total_capacity = max_batch_size × data_parallel_degree`
>
> **Example:** `--max-batch-size 32 --data-parallel-degree 4` = 128 total concurrent requests
>
> **Migration from v25.x:** If you had `--max-batch-size 128` aggregate with DP=4, use `--max-batch-size 32` in v26.1+

### Ragged Tensor Batching

Ragged tensor batching concatenates variable-length sequences into a single flat tensor with offset indices, eliminating padding waste. This reduces memory by 30-60% for variable-length batches.

**Concept:**

```
Padded batching (wasteful):
  Sequence 1: [A, B, C, PAD, PAD, PAD]  # 3 tokens, 3 padding
  Sequence 2: [D, E, F, G, H, I]        # 6 tokens, 0 padding
  Sequence 3: [J, K, PAD, PAD, PAD, PAD] # 2 tokens, 4 padding
  Total: 18 elements (11 actual tokens, 7 padding = 39% waste)

Ragged batching (efficient):
  tokens:            [A, B, C, D, E, F, G, H, I, J, K]  # 11 elements
  input_row_offsets: [0, 3, 9, 11]                      # batch_size + 1
  Total: 11 elements (0% waste)
```

**Pattern:**

```python
import numpy as np
from max.driver import Buffer

def prepare_ragged_batch(contexts, device):
    """Prepare inputs using ragged tensor format."""
    # Compute row offsets: cumulative sum of sequence lengths
    # Shape: [batch_size + 1], dtype: uint32
    input_row_offsets = np.cumsum(
        [0] + [ctx.tokens.active_length for ctx in contexts],
        dtype=np.uint32,
    )

    # Concatenate all tokens into flat array - no padding
    # Shape: [total_seq_len], dtype: int64
    tokens = np.concatenate([ctx.tokens.active for ctx in contexts])

    return {
        "tokens": Buffer.from_numpy(tokens).to(device),
        "input_row_offsets": Buffer.from_numpy(input_row_offsets).to(device),
    }
```

**Key Constraints:**
- `input_row_offsets` dtype must be `uint32`
- `input_row_offsets` length must be `batch_size + 1`
- First offset must be 0, last offset must equal `total_seq_len`
- Offsets must be monotonically increasing
- Requires `cache_strategy="paged"` for attention layers

### Scheduling Priority (CE vs TG)

MAX Serve's scheduler prioritizes Context Encoding (CE) vs Token Generation (TG) based on configuration, impacting latency and throughput tradeoffs.

**Scheduling Logic:**

```python
def _identify_priority(self, replica_idx: int) -> RequestType:
    # No CE requests -> prioritize TG (keep generating)
    if len(self.replicas[replica_idx].ce_reqs) == 0:
        return RequestType.TG

    # No TG requests -> prioritize CE (start new requests)
    if len(self.replicas[replica_idx].tg_reqs) == 0:
        return RequestType.CE

    # In-flight batching enabled -> prioritize TG (lower latency)
    if self.scheduler_config.enable_in_flight_batching:
        return RequestType.TG

    # Default -> prioritize CE (higher throughput)
    return RequestType.CE
```

**Configuration Options:**

| Setting | Effect | Use Case |
|---------|--------|----------|
| Default (CE priority) | Higher throughput, variable latency | Batch processing |
| `--enable-in-flight-batching` | Lower TTFT, balanced | Interactive apps |

### Environment Configuration

MAX Serve supports comprehensive configuration via environment variables with the `MAX_SERVE_` prefix.

**Core Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SERVE_HOST` | `0.0.0.0` | Server bind address |
| `MAX_SERVE_PORT` | `8000` | API server port |
| `MAX_SERVE_METRICS_ENDPOINT_PORT` | `8001` | Prometheus metrics port |
| `MAX_SERVE_OFFLINE_INFERENCE` | `False` | Disable API server |

**Worker Health:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SERVE_MW_TIMEOUT` | None | Model worker startup timeout (seconds) |
| `MAX_SERVE_MW_HEALTH_FAIL` | `60.0` | Heartbeat failure threshold |
| `MAX_SERVE_USE_HEARTBEAT` | `False` | Enable health monitoring |

**Telemetry:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SERVE_METRIC_LEVEL` | `BASIC` | NONE, BASIC, or DETAILED |
| `MAX_SERVE_METRIC_RECORDING_METHOD` | `PROCESS` | NOOP, SYNC, ASYNCIO, PROCESS |

---

## Common Patterns

### High-Throughput Batch Processing

**When:** Processing large volumes of requests where throughput matters more than latency

**Do:**
```bash
# Optimized for throughput
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 64 \
  --max-batch-input-tokens 16384 \
  --max-batch-total-tokens 131072
```

**Don't:**
```bash
# No batch configuration - suboptimal throughput
max serve --model meta-llama/Llama-3.1-8B-Instruct
```

### Interactive Chat Applications

**When:** Building chat interfaces where time-to-first-token matters

**Do:**
```bash
# Prioritize TG for consistent TTFT
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-in-flight-batching \
  --max-batch-size 32 \
  --max-batch-input-tokens 4096
```

**Don't:**
```bash
# Default CE priority causes variable TTFT
max serve --model meta-llama/Llama-3.1-8B-Instruct
```

### Production Environment Configuration

**When:** Deploying to production with container orchestration

**Do:**
```bash
# production.env
MAX_SERVE_HOST=0.0.0.0
MAX_SERVE_PORT=8000
MAX_SERVE_METRICS_ENDPOINT_PORT=8001

# Worker health
MAX_SERVE_MW_TIMEOUT=600
MAX_SERVE_USE_HEARTBEAT=true
MAX_SERVE_MW_HEALTH_FAIL=60

# Telemetry
MAX_SERVE_METRIC_LEVEL=BASIC
MAX_SERVE_METRIC_RECORDING_METHOD=PROCESS

# Logging
MODULAR_STRUCTURED_LOGGING=true
```

```bash
docker run --gpus 1 --env-file production.env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  modular/max-nvidia-full:latest \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

**Don't:**
```python
# Configuration buried in code, hard to change per environment
settings = Settings(host="localhost", port=8000, metric_level="DETAILED")
```

### Ragged Tensor Batching

**When:** Variable-length sequences (eliminates padding waste)

Ragged batching concatenates variable-length sequences into a flat tensor with offset indices:

```
Padded (wasteful):
  Seq 1: [A, B, C, PAD, PAD, PAD]  # 3 tokens + 3 padding
  Seq 2: [D, E, F, G, H, I]        # 6 tokens
  Total: 18 elements (39% waste)

Ragged (efficient):
  tokens: [A, B, C, D, E, F, G, H, I]  # 9 elements
  input_row_offsets: [0, 3, 9]         # batch_size + 1
  Total: 9 elements (0% waste)
```

**Prepare ragged batch:**
```python
def prepare_ragged_batch(contexts, device):
    # Cumulative sum of sequence lengths
    input_row_offsets = np.cumsum(
        [0] + [ctx.tokens.active_length for ctx in contexts],
        dtype=np.uint32,
    )
    # Concatenate all tokens - no padding
    tokens = np.concatenate([ctx.tokens.active for ctx in contexts])

    return {
        "tokens": Buffer.from_numpy(tokens).to(device),
        "input_row_offsets": Buffer.from_numpy(input_row_offsets).to(device),
    }
```

**Memory savings:** 30-60% for variable-length batches. Ragged batching is the default for MAX pipelines.

### CE vs TG Scheduling Priority

**When:** Balancing throughput vs latency

The scheduler prioritizes Context Encoding (CE) or Token Generation (TG) based on configuration:

```python
def _identify_priority(self, replica_idx):
    if len(self.replicas[replica_idx].ce_reqs) == 0:
        return RequestType.TG  # No CE, keep generating
    if len(self.replicas[replica_idx].tg_reqs) == 0:
        return RequestType.CE  # No TG, start new requests
    if self.scheduler_config.enable_in_flight_batching:
        return RequestType.TG  # In-flight: prioritize latency
    return RequestType.CE  # Default: prioritize throughput
```

| Setting | Effect | Use Case |
|---------|--------|----------|
| Default (CE priority) | Higher throughput, variable TTFT | Batch processing |
| `--enable-in-flight-batching` | Lower TTFT, balanced | Interactive chat |

### Data Parallel Split with Ragged Tensors

**When:** Distributing ragged batches across multiple GPUs

**Do:**
```python
from max.graph import ops

def split_batch(devices, input, input_row_offsets, data_parallel_splits):
    """Split ragged batch across multiple devices."""
    split_input = []
    split_offsets = []

    for i, device in enumerate(devices):
        start_offset = input_row_offsets[data_parallel_splits[i]]
        end_offset = input_row_offsets[data_parallel_splits[i + 1]]

        # Slice tokens for this device
        token_slice = ops.slice_tensor(input, [slice(start_offset, end_offset)])

        # Slice and renormalize offsets (subtract start_offset)
        offset_slice = ops.slice_tensor(
            input_row_offsets,
            [slice(data_parallel_splits[i], data_parallel_splits[i + 1] + 1)]
        ) - start_offset

        split_input.append(token_slice.to(device))
        split_offsets.append(offset_slice.to(device))

    return split_input, split_offsets
```

### Token Budget Configuration

**When:** Tuning throughput/latency tradeoff for your workload

Two budgets control batch composition:

| Budget | CLI Flag | Default | Purpose |
|--------|----------|---------|---------|
| Active Token Budget | `--max-batch-input-tokens` | None (auto) | Limits tokens processed per CE batch |
| Total Context Budget | `--max-batch-total-tokens` | None | Limits total context across batch |

**Chunking behavior:** When active token budget is exceeded and chunked prefill is enabled, large prompts are split across multiple batches. This improves GPU utilization for variable-length inputs.

**Tuning by model size:**

| Model Size | Recommended `--max-batch-input-tokens` |
|------------|---------------------------------------|
| Small (7-8B) | 4096-8192 |
| Medium (13-14B) | 8192-12288 |
| Large (70B+) | 16384-32768 |

**Long-context configuration:**
```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-sequence-length 65536 \
  --max-batch-input-tokens 16384 \
  --max-batch-total-tokens 65536 \
  --enable-chunked-prefill
```

**Configuration constraints:**
- `max_batch_total_tokens` must be >= `max_seq_len`
- `max_batch_size` must be <= `target_tokens_per_batch_ce`

### Multi-Model Serving

MAX Serve runs one model per process instance. To serve multiple models:

```bash
# Run separate instances on different ports
max serve --model-path model-a --port 8000 &
max serve --model-path model-b --port 8001 &
```

Use a reverse proxy (nginx, envoy) to route requests to the correct instance based on the model name in the request.

---

## Decision Guide

| Scenario | Configuration | Key Flags |
|----------|---------------|-----------|
| Batch processing | CE priority (default) | `--max-batch-size 64` |
| Interactive chat | TG priority | `--enable-in-flight-batching` |
| Long context | Token budget control | `--max-batch-total-tokens` |
| Variable sequences | Ragged batching | Automatic with PAGED KV cache |
| Production deployment | Environment-based | `MAX_SERVE_*` variables |
| Large model throughput | Higher token budget | `--max-batch-input-tokens 16384+` |

---

## Quick Reference

- **Batch size**: Set `--max-batch-size` based on GPU memory and latency requirements
- **Ragged batching**: Default for MAX pipelines, reduces memory 30-60% for variable sequences
- **CE priority**: Default, optimizes throughput for batch workloads
- **TG priority**: Enable `--enable-in-flight-batching` for interactive applications
- **Environment config**: Use `MAX_SERVE_*` prefix for production deployments
- **Chunked prefill**: Enable for long context workloads with `--enable-chunked-prefill`
- **CE watermark**: `--kvcache-ce-watermark` (default 0.95) controls prefill scheduling threshold

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `batch size exceeds` | Request exceeds max-batch-size | Increase `--max-batch-size` or reduce request |
| `out of memory` | Total tokens exceed capacity | Reduce `--max-batch-total-tokens` or add GPUs |
| `sequence length exceeds` | Input too long | Increase `--max-sequence-length` or truncate input |
| `model not found` | Invalid model path | Check path format, use HF format or local path |
| `slow startup` | Cold compilation | Graph caching is automatic; subsequent runs use warm cache |
| `request timeout` | Request processing too slow | Enable chunked prefill, check batch settings |
| `scheduling error` | CE/TG priority conflict | Adjust `--kvcache-ce-watermark`; reduce batch size |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|------------------|
| **Batch size semantics** | Per-replica (changed from aggregate in v25.x) | Per-replica |
| **Token limit flag** | `--max-batch-total-tokens` | `--max-batch-total-tokens` |
| **Prefill chunk size** | `--max-batch-input-tokens` | `--max-batch-input-tokens` |
| **CE batch size CLI** | `--max-batch-size` | `--max-batch-size` |
| **Scheduling control** | `--kvcache-ce-watermark` (default 0.95) | `--kvcache-ce-watermark` |
| **Data parallel batch** | Per-replica × degree | Per-replica × degree |

**Stable (v26.1):**
```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 8 \
  --max-batch-total-tokens 131072 \
  --data-parallel-degree 4 \
  --kvcache-ce-watermark 0.8
# Total capacity: 8 × 4 = 32 requests (per-replica × degree)
```

**Nightly (v26.2+):**
```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 8 \
  --max-batch-total-tokens 131072 \
  --data-parallel-degree 4 \
  --kvcache-ce-watermark 0.8
# Total capacity: 8 × 4 = 32 requests (per-replica × degree)
```

**Migration Notes from v25.x:**
- Batch size changed to **per-replica** in v26.1; multiply by data parallel degree for total capacity
- `--kvcache-ce-watermark` flag controls prefill priority threshold (default 0.95)
- `--max-ce-batch-size` is deprecated; use `--max-batch-size` instead

---

## Related Patterns

- [`serve-kv-cache.md`](serve-kv-cache.md) — KV cache strategies and memory management
- [`deploy-production.md`](deploy-production.md) — Metrics and telemetry configuration
- [`serve-api.md`](serve-api.md) — API features including streaming and token budgets

---

## References

- [MAX Serve Configuration](https://docs.modular.com/max/serve)
- [MAX Pipelines Architecture](https://docs.modular.com/max/api/python/pipelines/)
