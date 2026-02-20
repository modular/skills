---
title: MAX Serve Monitoring and Telemetry
description: Metric levels, telemetry configuration, worker lifecycle, and disaggregated inference
impact: HIGH
category: serve
tags: [metrics, telemetry, monitoring, workers, disaggregated]
error_patterns:
  - "metrics"
  - "telemetry"
  - "monitoring"
  - "worker"
  - "TTFT"
  - "latency"
  - "throughput"
scenarios:
  - "Monitor MAX Serve deployment"
  - "Configure metrics collection"
  - "Set up telemetry"
  - "Debug performance issues"
  - "Track TTFT and latency"
  - "Monitor worker lifecycle"
consolidates:
  - serve-metric-levels.md
  - serve-metrics-telemetry.md
  - serve-model-worker-lifecycle.md
  - serve-disaggregated-inference.md
---

# MAX Serve Monitoring and Telemetry

**Category:** serve | **Impact:** HIGH

Comprehensive guide to monitoring MAX Serve deployments including metric levels, telemetry configuration, worker lifecycle management, and disaggregated inference architecture.

---

## Core Concepts

### Metric Levels

MAX Serve provides three metric granularity levels via `MAX_SERVE_METRIC_LEVEL`. Choosing the wrong level can degrade production performance or leave you blind during debugging.

**Level Comparison:**

| Level | Value | Metrics Collected | Performance Impact | Use Case |
|-------|-------|-------------------|-------------------|----------|
| `NONE` | 0 | No metrics | Zero overhead | Minimal latency requirements |
| `BASIC` | 10 | Request counts, TTFT, input/output tokens | Minimal (<1%) | Production deployments |
| `DETAILED` | 20 | ITL, batch size, cache hit rates, preemption | Moderate (5-15%) | Debugging, optimization |

**Metrics by Level:**

| Metric | Level | Description |
|--------|-------|-------------|
| `maxserve.request_count` | BASIC | HTTP request count |
| `maxserve.request_time` | BASIC | Request latency (ms) |
| `maxserve.time_to_first_token` | BASIC | TTFT latency (ms) |
| `maxserve.num_input_tokens` | BASIC | Input token count |
| `maxserve.num_output_tokens` | BASIC | Output token count |
| `maxserve.model_load_time` | BASIC | Model load time (ms) |
| `maxserve.num_requests_queued` | BASIC | Requests waiting |
| `maxserve.num_requests_running` | BASIC | Requests processing |
| `maxserve.itl` | DETAILED | Inter-token latency (ms) |
| `maxserve.batch_size` | DETAILED | Batch size distribution |
| `maxserve.batch_execution_time` | DETAILED | Batch execution time |
| `maxserve.cache.hit_rate` | DETAILED | Prefix cache hit rate |
| `maxserve.cache.preemption_count` | DETAILED | Memory preemption events |
| `maxserve.cache.num_used_blocks` | DETAILED | KV cache block usage |
| `maxserve.cache.num_total_blocks` | DETAILED | Total KV cache blocks |

### Recording Methods

| Method | Description | Overhead | Use Case |
|--------|-------------|----------|----------|
| `NOOP` | No recording | None | Disable telemetry entirely |
| `SYNC` | Synchronous in-process | Highest | Testing only |
| `ASYNCIO` | Async in main process | Low | Development |
| `PROCESS` | Separate process | Minimal | Production (isolated) |

### Model Worker Lifecycle

MAX Serve uses a factory pattern with async context management for model worker lifecycle. Configure timeouts and heartbeats for production reliability.

**Pattern - Model Worker Startup:**

```python
@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory,
    pipeline_config: PipelineConfig,
    settings: Settings,
    metric_client: MetricClient,
    scheduler_zmq_configs: SchedulerZmqConfigs,
) -> AsyncGenerator[ProcessManager]:
    worker_name = "MODEL_" + str(uuid.uuid4())

    # Use spawn context for GPU models (fork is unsafe)
    mp = multiprocessing.get_context("spawn")

    async with subprocess_manager("Model Worker") as proc:
        alive = mp.Event()
        proc.start(...)

        # Wait for model to be ready
        await proc.ready(alive, timeout=settings.mw_timeout_s)

        # Enable heartbeat monitoring for production
        if settings.use_heartbeat:
            proc.watch_heartbeat(alive, timeout=settings.mw_health_fail_s)

        yield proc
```

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SERVE_MW_TIMEOUT` | None | Model worker startup timeout (seconds) |
| `MAX_SERVE_USE_HEARTBEAT` | False | Enable periodic heartbeat checks |
| `MAX_SERVE_MW_HEALTH_FAIL` | 60.0 | Max seconds without heartbeat |

### Disaggregated Inference

Disaggregated inference separates prefill (context encoding) and decode (token generation) into different workers for high-throughput scenarios.

**Architecture:**

```
                       Dispatcher
                          |
           +--------------+--------------+
           |                             |
    Prefill Workers              Decode Workers
    (Context Encoding)           (Token Generation)
           |                             |
           +--[KV Transfer]-->-----------+
```

**Trade-offs:**

| Aspect | Unified | Disaggregated |
|--------|---------|---------------|
| Latency | Lower for small requests | Higher (transfer overhead) |
| Throughput | Good | Better at scale |
| Complexity | Simple | More operational overhead |
| GPU utilization | Variable | More consistent |

**Detailed Component Architecture:**

```
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    │   (TCP/HTTP)        │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │    Dispatcher       │
                    │  - Request routing  │
                    │  - KV transfer mgmt │
                    │  - Queue management │
                    └─────────┬───────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
┌────────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│ Prefill Worker 0│  │ Prefill Worker 1│  │ Decode Workers │
│ GPU 0           │  │ GPU 1           │  │ GPU 2,3        │
│ - Long prompts  │  │ - Long prompts  │  │ - Token gen    │
│ - Context encode│  │ - Context encode│  │ - Cont. batch  │
└────────┬────────┘  └────────┬────────┘  └───────▲────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                     KV Cache Transfer (RDMA/NVLink)
```

**Prefill Worker Characteristics:**

| Property | Value | Notes |
|----------|-------|-------|
| Primary workload | Context encoding | First token generation |
| Compute pattern | Compute-bound | High FLOPS utilization |
| Memory access | Sequential | Full attention over prompt |
| Batch strategy | Small batches | Memory for long contexts |
| GPU memory | Higher KV allocation | Full context must fit |
| Optimal GPU | High compute | H100, MI300X |

**Disaggregated Configuration:**

Use the `--pipeline-role` CLI flag to configure workers:

```bash
# Prefill-only worker
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --pipeline-role PrefillOnly \
  --max-batch-size 8 \
  --max-batch-input-tokens 32768 \
  --enable-chunked-prefill

# Decode-only worker
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --pipeline-role DecodeOnly \
  --max-batch-size 256 \
  --enable-in-flight-batching
```

**Decode Worker Characteristics:**

| Property | Value | Notes |
|----------|-------|-------|
| Primary workload | Token generation | Autoregressive decode |
| Compute pattern | Memory-bound | Low arithmetic intensity |
| Memory access | Random | KV cache lookups |
| Batch strategy | Large batches | More concurrent sequences |
| GPU memory | Smaller per-seq | KV transfers from prefill |
| Optimal GPU | High bandwidth | H100, MI300X |

**KV Transfer Mechanisms:**

| Mechanism | Bandwidth | Latency | Use Case |
|-----------|-----------|---------|----------|
| PCIe 4.0 | 32 GB/s | ~1-10ms | Single-node, different PCIe domains |
| PCIe 5.0 | 64 GB/s | ~0.5-5ms | Single-node, latest hardware |
| NVLink | 450-900 GB/s | ~0.1ms | Same-node, NVIDIA GPUs |
| RDMA (InfiniBand) | 100-400 GB/s | ~1-5μs | Multi-node |
| NVSwitch | 1.8 TB/s | ~0.1ms | DGX systems |

**When to Use Disaggregated:**

```
Use Disaggregated When:
├─ GPU count >= 4
├─ Average prompt length > 2000 tokens
├─ High request volume (>100 req/s)
├─ RAG workloads (long context, short generation)
└─ Latency SLA allows ~10-50ms overhead

Stay with Unified When:
├─ GPU count < 4
├─ Short prompts (<500 tokens)
├─ Low volume or latency-critical
├─ Interactive chat (minimal TTFT critical)
└─ Simple deployment preferred
```

**Monitoring Disaggregated Deployments:**

<!-- NOTE: Disaggregated-specific metrics are not yet documented in public docs.
     The standard maxserve.* metrics apply to all deployment modes.
     Monitor the standard TTFT, request count, and cache metrics below. -->

Monitor disaggregated deployments using the same standard metrics (TTFT, batch size, cache metrics). Pay special attention to:
- `maxserve.time_to_first_token` — higher with disaggregated due to KV transfer overhead
- `maxserve.num_requests_queued` — separate prefill/decode queue pressure
- `maxserve.cache.num_used_blocks` — cache utilization across workers

---

## Common Patterns

### Production Metrics Configuration

**When:** Deploying to production with monitoring infrastructure

**Do:**
```bash
# Production: BASIC metrics with process isolation
export MAX_SERVE_METRIC_LEVEL=BASIC
export MAX_SERVE_METRIC_RECORDING_METHOD=PROCESS
export MAX_SERVE_METRICS_ENDPOINT_PORT=8001

max serve --model meta-llama/Llama-3.1-8B-Instruct
```

**Don't:**
```bash
# DETAILED level adds per-token measurements that accumulate overhead
export MAX_SERVE_METRIC_LEVEL=DETAILED
export MAX_SERVE_METRIC_RECORDING_METHOD=SYNC  # Synchronous = blocking

max serve --model meta-llama/Llama-3.1-8B-Instruct
# Result: 5-15% latency increase from ITL measurements on every token
```

### Debug/Optimization Metrics

**When:** Debugging performance issues or optimizing configuration

**Do:**
```bash
# Development/Debugging: DETAILED metrics for optimization
export MAX_SERVE_METRIC_LEVEL=DETAILED
export MAX_SERVE_METRIC_RECORDING_METHOD=ASYNCIO
export MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR=20  # Buffer detailed metrics

max serve --model meta-llama/Llama-3.1-8B-Instruct
```

### Worker Timeout Configuration

**When:** Deploying large models (70B+ parameters)

**Do:**
```bash
# Configure timeouts based on model size
export MAX_SERVE_MW_TIMEOUT=600       # 10 min startup timeout
export MAX_SERVE_USE_HEARTBEAT=true   # Enable health monitoring
export MAX_SERVE_MW_HEALTH_FAIL=60    # 60s heartbeat threshold

max serve --model meta-llama/Llama-3.1-70B-Instruct
```

**Don't:**
```bash
# Model may hang indefinitely on large model loads
max serve --model meta-llama/Llama-3.1-70B-Instruct
```

### Grafana Dashboard Queries

**When:** Building observability dashboards

```promql
# P99 TTFT (Time to First Token)
histogram_quantile(0.99, rate(maxserve_time_to_first_token_milliseconds_bucket[5m]))

# Token throughput (tokens/sec)
rate(maxserve_num_output_tokens_total[5m])

# Request throughput
rate(maxserve_request_count_total[5m])

# Batch size distribution
histogram_quantile(0.95, rate(maxserve_batch_size_bucket[5m]))

# Cache hit rate (requires DETAILED)
rate(maxserve_cache_hit_rate_sum[5m]) / rate(maxserve_cache_hit_rate_count[5m])

# Preemption rate (memory pressure indicator)
rate(maxserve_cache_preemption_count_total[5m])
```

### Disaggregated Inference Setup

**When:** High-throughput deployments with 4+ GPUs and long prompts

**Do:**
```bash
# Dispatcher bind address
export MAX_SERVE_DI_BIND_ADDRESS="tcp://127.0.0.1:5555"

# Deployment architecture example:
# GPUs 0-1: Prefill workers (handle prompt encoding)
# GPUs 2-3: Decode workers (handle token generation)
#
# Prefill optimized for:
# - Large batch input tokens
# - Chunked prefill
# - Memory for long contexts
#
# Decode optimized for:
# - Low latency per token
# - Continuous batching
# - In-flight batching
```

**Don't:**
```bash
# Unified scheduler may bottleneck at scale with many GPUs
max serve --model meta-llama/Llama-3.1-70B-Instruct \
  --devices gpu:0,1,2,3,4,5,6,7
```

---

## Decision Guide

| Scenario | Metric Level | Recording Method | Notes |
|----------|--------------|------------------|-------|
| Production | BASIC | PROCESS | Minimal overhead, isolated |
| Development | DETAILED | ASYNCIO | Full visibility |
| Debugging | DETAILED | ASYNCIO | Temporarily enable |
| Latency-critical | NONE or BASIC | PROCESS | Minimize overhead |
| Capacity planning | DETAILED | PROCESS | Monitor cache metrics |

| Scenario | Architecture | When to Use |
|----------|--------------|-------------|
| Standard deployment | Unified | <4 GPUs, mixed workloads |
| High-scale production | Disaggregated | 4+ GPUs, long prompts |
| RAG workloads | Disaggregated | Long context, short generation |

---

## Quick Reference

- **BASIC metrics**: <1% overhead, recommended for production
- **DETAILED metrics**: 5-15% overhead, use for debugging only
- **PROCESS recording**: Isolates metrics collection from model worker
- **MW_TIMEOUT**: Set based on model size (~30s per 10B parameters)
- **Heartbeat**: Enable for production to detect hung workers
- **Disaggregated**: Use for 4+ GPU deployments with long prompts

---

## Prometheus Endpoint

```bash
# Metrics exposed at dedicated port
curl http://localhost:8001/metrics

# Example Prometheus output
maxserve_time_to_first_token_milliseconds_bucket{le="100.0"} 42
maxserve_request_count_total{code="200",path="/v1/chat/completions"} 1234
maxserve_batch_size_bucket{le="16"} 89
```

---

## Best Practices

1. **Production**: Use `BASIC` level with `PROCESS` recording
2. **Large models**: Set `MW_TIMEOUT` based on model size
3. **Reliability**: Enable heartbeat for production
4. **Debugging**: Temporarily enable `DETAILED`, then disable
5. **Alerting**: Monitor preemption counts and TTFT P99
6. **Buffer factor**: Set `MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR=20` to batch detailed metrics
7. **Spawn context**: Use `spawn` multiprocessing context (not fork) for GPU models

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `metrics endpoint not responding` | Metrics disabled or wrong port | Set `MAX_SERVE_METRIC_LEVEL=BASIC`; check port 8001 |
| `Prometheus scrape timeout` | Too many detailed metrics | Use BASIC mode in production; reduce scrape frequency |
| `health check failing` | Server still initializing | Increase probe timeout; use `/health` endpoint |
| `missing TTFT metric` | Streaming not enabled | TTFT only measured for streaming requests |
| `high preemption count` | KV cache pressure | Increase cache size; reduce max batch size |
| `worker crash not detected` | No health monitoring | Configure readiness/liveness probes; use process manager |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|------------------|
| **Metric levels** | NONE, BASIC, DETAILED | NONE, BASIC, DETAILED |
| **Health endpoint** | `/health` (v25.5+) | `/health` |
| **Metrics port** | 8001 (default) | 8001 (default) |
| **Disaggregated inference** | `--pipeline-role` flag | `--pipeline-role` flag |
| **CE watermark** | `--kvcache-ce-watermark` (default 0.95) | `--kvcache-ce-watermark` |
| **Batch size semantics** | Per-replica (changed in v26.1) | Per-replica |

**Stable (v26.1):**
```bash
# Standard monitoring configuration
export MAX_SERVE_METRIC_LEVEL=BASIC
export MAX_SERVE_METRICS_ENDPOINT_PORT=8001
export MAX_SERVE_USE_HEARTBEAT=true

max serve --model meta-llama/Llama-3.1-8B-Instruct
```

**Nightly (v26.2+):**
```bash
# Same configuration as stable
export MAX_SERVE_METRIC_LEVEL=BASIC
export MAX_SERVE_METRICS_ENDPOINT_PORT=8001
export MAX_SERVE_USE_HEARTBEAT=true

max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --kvcache-ce-watermark 0.8
```

**Notes:**
- Core monitoring APIs are stable across both versions
- `--kvcache-ce-watermark` available in both v26.1 and nightly (default 0.95)
- Batch size semantics: per-replica in both v26.1 and nightly (changed from aggregate in v25.x)
- Disaggregated inference available in both versions via `--pipeline-role`

---

## Related Patterns

- [`serve-configuration.md`](serve-configuration.md) — Environment configuration
- [`serve-kv-cache.md`](serve-kv-cache.md) — KV cache monitoring
- [`serve-request-lifecycle.md`](serve-request-lifecycle.md) — Error and preemption handling

---

## References

- [MAX Serve Configuration](https://docs.modular.com/max/serve)
- [MAX Serve Observability](https://docs.modular.com/max/serve)
- Source: `max/serve/config.py`, `max/serve/telemetry/metrics.py`
