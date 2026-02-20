---
title: Request Lifecycle Management
description: Request cancellation, preemption handling, and error propagation patterns
impact: CRITICAL
category: serve
tags: [cancellation, preemption, errors, reliability]
error_patterns:
  - "request cancelled"
  - "preemption"
  - "timeout"
  - "disconnect"
  - "error propagation"
  - "stream"
  - "connection closed"
scenarios:
  - "Handle request cancellation"
  - "Configure preemption behavior"
  - "Propagate errors correctly"
  - "Debug streaming failures"
  - "Handle client disconnects"
  - "Manage request timeouts"
consolidates:
  - serve-request-cancellation.md
  - serve-preemption-handling.md
  - serve-error-propagation.md
---

# Request Lifecycle Management

**Category:** serve | **Impact:** CRITICAL

Comprehensive guide to handling request lifecycle events in MAX Serve including cancellation, preemption under memory pressure, and error propagation. Understanding these mechanisms is essential for building reliable production deployments.

---

## Core Concepts

### Request Cancellation

MAX Serve automatically handles request cancellations when clients disconnect, timeout, or abort. This frees GPU resources and prevents resource leaks.

**Cancellation Triggers:**
- Client disconnects during streaming
- Client timeout while waiting for response
- Exception during stream processing
- Async generator closed early
- Client process terminates unexpectedly

**Cancellation Flow:**

```
Client         API Worker         Model Worker
  |                |                    |
  |--[Disconnect]->|                    |
  |                |                    |
  |                |  (response arrives |
  |                |   but no pending   |
  |                |   queue exists)    |
  |                |                    |
  |                |--[Cancel Queue]--->|
  |                |                    |
  |                |     release_request()
  |                |     free KV cache  |
  |                |                    |
  |                |<--[Cancelled]------|
```

**Pattern - Scheduler Cancellation Handling:**

```python
def run_iteration(self) -> SchedulerProgress:
    # ... batch execution ...

    # Process cancellations after each iteration
    for cancelled_id in get_cancelled_reqs(self.cancel_queue):
        if self.batch_constructor.contains(cancelled_id):
            # Release request from batch (frees KV cache)
            self.batch_constructor.release_request(cancelled_id)
            # Notify API layer
            self.response_queue.put_nowait(
                {cancelled_id: SchedulerResult.cancelled()}
            )

    return SchedulerProgress.MADE_PROGRESS
```

### Preemption Handling

MAX Serve preempts requests when KV cache memory is exhausted. Configure watermarks to avoid performance degradation.

**Preemption Reasons:**

```python
class PreemptionReason(str, Enum):
    KV_CACHE_MEMORY = "kv_cache_memory"
    MAX_NUM_LORAS = "max_num_loras"
```

**Preemption Mechanism:**

```python
def _add_ce_requests(self, batch, replica_idx):
    # Check watermark before adding CE request
    pct_blocks_used = self.paged_cache.get_pct_used_blocks_after_allocation(ctx)

    if pct_blocks_used > self.scheduler_config.kvcache_ce_watermark:
        # Too close to memory limit - return request to queue
        self._return_to_request_queue(ctx, replica_idx)
        break

    try:
        self.paged_cache.alloc(ctx, num_steps=1)
    except InsufficientBlocksError:
        # Preempt if allocation fails
        self._preempt_request(context, replica_idx, PreemptionReason.KV_CACHE_MEMORY)
```

**Watermark Configuration:**

```bash
# Lower watermark leaves headroom for TG requests
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --kvcache-ce-watermark 0.85 \
  --device-memory-utilization 0.9
```

### Error Propagation

MAX Serve propagates errors from the model worker through the scheduler to the API layer, ensuring clients receive proper error responses.

**Error Flow:**

```
Model Worker           Scheduler              API Layer
     |                     |                     |
     |--[Exception]------->|                     |
     |                     |                     |
     |  SchedulerResult    |                     |
     |  .from_error(exc)   |                     |
     |                     |                     |
     |     [SchedulerError with traceback]       |
     |                     |-------------------->|
     |                     |                     |
     |                     |  RuntimeError with  |
     |                     |  remote traceback   |
```

**Pattern - SchedulerError Structure:**

```python
class SchedulerError(msgspec.Struct):
    """Captures exception details for cross-process communication."""
    error_type: str      # Exception class name (e.g., 'RuntimeError')
    error_message: str   # The exception message
    traceback_str: str   # Full traceback for debugging

    @classmethod
    def from_exception(cls, exc: BaseException) -> SchedulerError:
        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_str="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )
```

---

## Common Patterns

### Proper Error Handling in Scheduler

**When:** Implementing custom schedulers or understanding error flow

**Do:**
```python
def _schedule(self, inputs: TextGenerationInputs) -> int:
    batch_request_ids = [ctx.request_id for ctx in inputs.flat_batch]

    try:
        responses = self.pipeline.execute(inputs)
    except Exception as exc:
        logger.exception("Exception during pipeline execution")

        # Send error results to ALL requests in the batch
        self.response_queue.put_nowait(
            {
                req_id: SchedulerResult.from_error(exc)
                for req_id in batch_request_ids
            }
        )

        # Release all requests from batch constructor
        for req_id in batch_request_ids:
            if self.batch_constructor.contains(req_id):
                self.batch_constructor.release_request(req_id)

        return len(batch_request_ids)

    # Normal response handling...
```

**Don't:**
```python
# WRONG: Errors are silently ignored, requests hang forever
def _schedule(self, inputs):
    try:
        responses = self.pipeline.execute(inputs)
    except Exception:
        pass  # Silent failure - clients never receive response
        return 0
```

### API Layer Error Checking

**When:** Processing scheduler results in API layer

**Do:**
```python
async def stream(self, req_id, data):
    """Stream results with proper error propagation."""
    with self.open_channel(req_id, data) as queue:
        while True:
            item = await queue.get()

            # Check for error FIRST - propagate pipeline failures
            if item.error is not None:
                raise RuntimeError(
                    f"Pipeline error ({item.error.error_type}): "
                    f"{item.error.error_message}\n\n"
                    f"Remote traceback:\n{item.error.traceback_str}"
                )

            if item.result is None:
                break

            yield item.result

            if item.is_done:
                break
```

### Preemption Mitigation

**When:** Experiencing high preemption rates under load

**Do:**
```bash
# Option 1: Reduce batch size
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 32  # Reduced from 64

# Option 2: Lower device memory utilization
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --device-memory-utilization 0.8

# Option 3: Enable host swapping (extends cache to CPU)
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-kvcache-swapping-to-host

# Option 4: Reduce max sequence length
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-length 4096  # Reduced from 8192
```

**Don't:**
```bash
# Default watermark (0.95) may cause frequent preemptions under load
max serve --model meta-llama/Llama-3.1-8B-Instruct
```

### OOM Error Handling

**When:** Handling GPU out-of-memory errors gracefully

**Do:**
```python
class OOMError(RuntimeError):
    """Custom exception with helpful guidance for OOM errors."""

    def __init__(self, _: str = ""):
        super().__init__("""
GPU ran out of memory during model execution.

Suggested solutions:
1. Reduce --device-memory-utilization to a smaller value
2. Reduce batch size with --max-batch-size parameter
3. Reduce sequence length with --max-length parameter
4. Reduce max batch input tokens with --max-batch-input-tokens parameter
""")

def detect_and_wrap_oom(exception: Exception) -> None:
    """Detect OOM errors and wrap with helpful guidance."""
    if isinstance(exception, ValueError) and "OUT_OF_MEMORY" in str(exception):
        raise OOMError() from exception
```

### ZMQ Communication Architecture

**When:** Understanding inter-process communication between API workers and model workers

MAX Serve uses ZMQ (ZeroMQ) for low-latency communication:

```
API Worker                    Model Worker
    |                              |
    |--[Request Queue (PUSH)]----->|
    |                              |
    |<-[Response Queue (PULL)]-----|
    |                              |
    |--[Cancel Queue (PUSH)]------>|
```

**Queue configuration pattern:**
```python
class SchedulerZmqConfigs:
    def __init__(self, pipeline_task, context_type):
        self.request_queue_config = ZmqConfig[BaseContext](context_type)
        self.response_queue_config = ZmqConfig[dict[RequestID, SchedulerResult]](...)
        self.cancel_queue_config = ZmqConfig[list[RequestID]](list[RequestID])

    def api_worker_queues(self):
        return (
            self.request_queue_config.push(),   # Send requests
            self.response_queue_config.pull(),  # Receive responses
            self.cancel_queue_config.push(),    # Send cancellations
        )

    def model_worker_queues(self):
        return (
            self.request_queue_config.pull(),   # Receive requests
            self.response_queue_config.push(),  # Send responses
            self.cancel_queue_config.pull(),    # Receive cancellations
        )
```

**Cancellation via response collection:**
```python
async def response_worker(self):
    cancelled = set()
    for request_id, response in response_dict.items():
        if request_id in self.pending_out_queues:
            await self.pending_out_queues[request_id].put(response)
        else:
            cancelled.add(request_id)  # Client disconnected

    if cancelled:
        self.cancel_queue.put_nowait(list(cancelled))
```

**Best practices:**
- Use unique request IDs to prevent race conditions
- Handle cancellation to free GPU resources from disconnected clients
- Implement proper error propagation from model worker to API
- Use ZMQ IPC for local communication (lower latency than TCP)

---

## Decision Guide

| Scenario | Action | Configuration |
|----------|--------|---------------|
| High cancellation rate (>5%) | Review client timeouts | Check client configuration |
| High preemption rate (>10/min) | Reduce memory pressure | Lower batch size or utilization |
| Frequent OOM errors | Increase memory headroom | `--device-memory-utilization 0.8` |
| Silent request failures | Enable error propagation | Ensure SchedulerResult.error is checked |
| Long-running requests | Enable host swapping | `--enable-kvcache-swapping-to-host` |

---

## Monitoring

### Memory Pressure Indicators

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| `cache.preemption_count` | 0-10/min | 10-50/min | >50/min |
| `cache.num_used_blocks` | <80% | 80-95% | >95% |
| `cache.hit_rate` | >50% | 25-50% | <25% |

### Cancellation Monitoring

| Indicator | Healthy | Warning | Action |
|-----------|---------|---------|--------|
| Cancellation rate | <5% of requests | 5-20% | Review client timeouts |
| Orphaned responses | 0 per minute | >10 per minute | Check client stability |
| KV cache release delay | <100ms | >1s | Check scheduler load |

```bash
# Check preemption metrics
curl http://localhost:8001/metrics | grep preemption

# Example output (high preemption = problem)
maxserve_cache_preemption_count_total 127
```

---

## Quick Reference

- **Cancellation**: Automatic via cancel queue, frees KV cache blocks
- **Preemption**: Triggered when `pct_blocks_used > kvcache_ce_watermark`
- **Default watermark**: 0.95 (tune to 0.85-0.90 for stability)
- **Error propagation**: Always check `SchedulerResult.error` before `result`
- **Resource cleanup**: KV cache, batch slots, and network resources freed on cancel/error
- **Partial results**: Already-sent tokens delivered, `is_done=True` on completion

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `request cancelled` | Client disconnected during processing | Normal behavior, resources are freed |
| `preemption` | Memory pressure caused eviction | Reduce batch size or add GPU memory |
| `timeout` | Request took too long | Check batch settings, enable chunked prefill |
| `stream error` | Network issue during streaming | Client should handle reconnection |
| `generation failed` | Internal error during generation | Check logs, may be OOM or model issue |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|--------------------|
| **Cancellation** | Automatic on disconnect | Automatic on disconnect (unchanged) |
| **Preemption** | KV cache memory, LoRA capacity | KV cache memory, LoRA capacity |
| **CE watermark** | `--kvcache-ce-watermark` (default 0.95) | `--kvcache-ce-watermark` |

**Stable (v26.1):**
```bash
# CE watermark controls when prefill is scheduled (lower = more headroom for TG)
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --kvcache-ce-watermark 0.85 \
  --device-memory-utilization 0.9
```

**Nightly (v26.2+):**
```bash
# Same flags available
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --kvcache-ce-watermark 0.85 \
  --device-memory-utilization 0.9
```

**Notes:**
- Request cancellation on client disconnect is stable
- Preemption handling for KV cache memory is stable
- Error propagation patterns are stable
- `--kvcache-ce-watermark` available since v26.1 (default 0.95)
- Scheduler cancellation and release patterns are stable

---

## Related Patterns

- [`serve-kv-cache.md`](serve-kv-cache.md) — KV cache memory management
- [`serve-monitoring.md`](serve-monitoring.md) — Metrics for monitoring health
- [`serve-configuration.md`](serve-configuration.md) — Batch and scheduling configuration

---

## References

- [MAX Serve Configuration](https://docs.modular.com/max/serve)
- Source: `max/serve/scheduler/text_generation_scheduler.py`, `max/interfaces/scheduler.py`
