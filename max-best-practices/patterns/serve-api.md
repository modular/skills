---
title: MAX Serve API & Request Lifecycle
description: Streaming, token budgets, structured output, function calling, LoRA adapters, health endpoints, request cancellation, preemption, and error propagation
impact: CRITICAL
category: serve
tags: [streaming, tokens, structured-output, function-calling, lora, health, cancellation, preemption, errors, reliability]
error_patterns:
  - "streaming"
  - "token budget"
  - "structured output"
  - "function calling"
  - "LoRA"
  - "health check"
  - "API error"
  - "request cancelled"
  - "preemption"
  - "timeout"
  - "disconnect"
  - "error propagation"
  - "connection closed"
scenarios:
  - "Enable streaming responses"
  - "Configure token budgets"
  - "Use structured output"
  - "Implement function calling"
  - "Manage LoRA adapters"
  - "Configure health endpoints"
  - "Handle request cancellation"
  - "Configure preemption behavior"
  - "Propagate errors correctly"
  - "Debug streaming failures"
  - "Handle client disconnects"
  - "Manage request timeouts"
consolidates:
  - serve-streaming.md
  - serve-token-budget.md
  - serve-structured-output.md
  - serve-function-calling.md
  - serve-lora-lifecycle.md
  - serve-health-endpoints.md
  - serve-request-cancellation.md
  - serve-preemption-handling.md
  - serve-error-propagation.md
  - serve-request-lifecycle.md
---
<!-- PATTERN QUICK REF
WHEN: Building MAX Serve APIs, handling request lifecycle, endpoint configuration, error handling
KEY_TYPES: OpenAI client, SchedulerResult, SchedulerError, PreemptionReason, OOMError, ZmqConfig
SYNTAX:
  - client.chat.completions.create(model=..., stream=True) for streaming
  - response_format={"type": "json_schema", ...} for structured output
  - tools=[{"type": "function", ...}] for function calling
  - max serve --enable-lora --no-enable-prefix-caching for LoRA
  - max serve --kvcache-ce-watermark 0.85 for preemption tuning
PITFALLS:
  - LoRA incompatible with prefix caching (must use --no-enable-prefix-caching)
  - Silent error swallowing causes requests to hang forever — always check SchedulerResult.error
  - Default watermark 0.95 causes frequent preemptions under load — tune to 0.85-0.90
  - Only Llama 3 QKVO layers supported for LoRA (no MLP modules)
RELATED: serve-configuration, serve-kv-cache, deploy-production
-->

# MAX Serve API & Request Lifecycle

**Category:** serve | **Impact:** CRITICAL

Comprehensive guide to MAX Serve API features and request lifecycle management. Covers streaming responses, token budgets, structured output, function calling, LoRA adapter management, health endpoints, request cancellation, preemption handling, and error propagation.

---

## Core Concepts

### Streaming Responses

Enable streaming for lowest time-to-first-token (TTFT) and better user experience.

**Pattern:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Streaming response
stream = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Benefits:**
- Immediate first token delivery
- Better user experience for chat interfaces
- Chunked token responses (v26.1+)

### Token Budgets

Token budgets control how many tokens are processed per batch. Understanding `--max-batch-input-tokens` (active token budget) and `--max-batch-total-tokens` is essential for optimizing throughput and latency.

**Budget Overview:**

| Budget | CLI Flag | Default | Purpose |
|--------|----------|---------|---------|
| Active Token Budget | `--max-batch-input-tokens` | None (auto) | Limits tokens processed per CE batch |
| Total Context Budget | `--max-batch-total-tokens` | None | Limits total context across batch |

**Throughput vs Latency Tradeoffs:**

| Configuration | Throughput | Latency | Use Case |
|--------------|------------|---------|----------|
| High `--max-batch-input-tokens` | Higher | Higher TTFT | Batch processing |
| Low `--max-batch-input-tokens` | Lower | Lower TTFT | Interactive apps |
| With `--max-batch-total-tokens` | Controlled | Predictable | Memory-constrained |

**Tuning for Model Sizes:**

| Model Size | Recommended `--max-batch-input-tokens` | Notes |
|------------|---------------------------------------|-------|
| Small (7-8B params) | 4096-8192 | Good default for most GPUs |
| Medium (13-14B params) | 8192-12288 | Balance memory and throughput |
| Large (70B+ params) | 16384-32768 | Higher budget utilizes compute |

### Structured Output

Enable structured output for enforced JSON schema compliance.

**Pattern:**

```bash
# Enable structured output on server
max serve --model ... --enable-structured-output
```

```python
# Use response_format in request
response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Extract entities"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "entities",
            "schema": {
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}},
                    "locations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["names", "locations"]
            }
        }
    }
)
```

### Function Calling

MAX supports OpenAI-compatible function calling / tool use for agentic workflows.

**Pattern:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Weather in SF?"}],
    tools=tools,
    tool_choice="auto"
)
```

**Note:** Function calling works with models that support tool use (Llama 3.1+, Mistral).

**Complete Function Schema Example:**

```python
# Complex tool with nested schema and validation
tools = [{
    "type": "function",
    "function": {
        "name": "search_products",
        "description": "Search for products in the catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text"
                },
                "category": {
                    "type": "string",
                    "enum": ["electronics", "clothing", "home", "books"],
                    "description": "Product category filter"
                },
                "price_range": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "number", "minimum": 0},
                        "max": {"type": "number", "minimum": 0}
                    },
                    "required": ["min", "max"]
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "price_asc", "price_desc", "rating"],
                    "default": "relevance"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
}]
```

**Handling Tool Call Responses:**

```python
import json

def process_tool_calls(response):
    """Process model's function calls and execute tools."""
    messages = [{"role": "user", "content": "Search for laptops under $1000"}]

    response = client.chat.completions.create(
        model="llama",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Check if model wants to call a function
    if response.choices[0].message.tool_calls:
        tool_calls = response.choices[0].message.tool_calls

        # Add assistant message with tool calls
        messages.append(response.choices[0].message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Execute the function (your implementation)
            result = execute_function(function_name, arguments)

            # Add tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Get final response with tool results
        final_response = client.chat.completions.create(
            model="llama",
            messages=messages,
            tools=tools
        )
        return final_response

    return response
```

**Tool Choice Options:**

| Option | Behavior |
|--------|----------|
| `"auto"` | Model decides whether to call a function |
| `"none"` | Model generates text response only |
| `"required"` | Model must call at least one function |
| `{"type": "function", "function": {"name": "..."}}` | Force specific function |

**Function Calling Errors:**

Common error scenarios:
- Invalid tool schema: Returns 400 with details about the schema validation failure
- Unknown tool_choice: Returns 400 if value is not `auto`, `none`, `required`, or a tool object
- Unsupported model: Returns error if the model doesn't support function calling

### LoRA Adapter Lifecycle

LoRA (Low-Rank Adaptation) adapters enable serving multiple fine-tuned model variants from a single base model. MAX Serve manages adapters through an LRU cache.

**Key Constraints:**
- LoRA adapters are **not compatible with prefix caching** (use `--no-enable-prefix-caching`)
- Currently only supports **Llama 3 models** with **QKVO layers** (q_proj, k_proj, v_proj, o_proj)
- Adapters must use **safetensors** weight format trained with **PEFT**
- `max_num_loras` controls how many adapters can be active simultaneously (default: 1)
- Protected adapters (in use by TG requests) cannot be evicted
- Memory scales linearly: each adapter adds `2 * rank * hidden_size * num_layers` parameters

**Server Configuration:**

```bash
# Enable LoRA with capacity limits (must disable prefix caching)
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --no-enable-prefix-caching \
  --max-num-loras 4 \
  --max-lora-rank 16 \
  --lora-paths "adapter1=/path/to/adapter1" "adapter2=/path/to/adapter2"
```

**Memory Impact per Adapter (approximate for 8B model):**
- rank=8:  ~25MB per adapter
- rank=16: ~50MB per adapter
- rank=32: ~100MB per adapter
- rank=64: ~200MB per adapter

**API Reference:**

```bash
# Load adapter at runtime
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "my_adapter", "lora_path": "/path/to/adapter"}'

# Unload adapter to free memory
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "my_adapter"}'

# List loaded adapters (included in models list)
curl http://localhost:8000/v1/models

# Use adapter for inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my_adapter", "messages": [{"role": "user", "content": "Hello"}]}'
```

**LoRA Adapter Configuration Schema:**

The adapter directory must contain `adapter_config.json`:

```json
{
  "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "r": 16,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

> **Note:** Only attention modules (q_proj, k_proj, v_proj, o_proj) are currently supported.
> MLP modules (gate_proj, up_proj, down_proj) are not yet supported.

| Field | Required | Description |
|-------|----------|-------------|
| `r` | Yes | LoRA rank (8, 16, 32, 64) |
| `lora_alpha` | Yes | Scaling factor (typically 2*r) |
| `target_modules` | Yes | Layers to apply LoRA |
| `lora_dropout` | No | Dropout rate (default 0.0) |
| `bias` | No | `"none"`, `"all"`, or `"lora_only"` |
| `task_type` | No | `"CAUSAL_LM"` for text generation |

### Health Endpoints

Use `/health` endpoint for Kubernetes readiness probes and load balancer checks.

**Pattern:**

```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

**Note:** `/health` endpoint added in v25.5 for lm-eval and similar tools.

---

## Request Lifecycle

Understanding how MAX Serve handles request cancellation, preemption, and error propagation is essential for building reliable production deployments.

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

### ZMQ Communication Architecture

MAX Serve uses ZMQ (ZeroMQ) for low-latency inter-process communication between API workers and model workers:

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

## Common Patterns

### Token Budget for Long-Context Throughput

**When:** Serving 70B+ models with 128K context

**Do:**
```bash
# Increase budget for large model with long context
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --max-batch-size 64 \
  --max-batch-input-tokens 16384 \
  --max-batch-total-tokens 131072
```

**Don't:**
```bash
# Default 8192 tokens may underutilize GPU for long-context workloads
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --max-batch-size 64
```

### Token Budget for Interactive Latency

**When:** Building responsive chat applications

**Do:**
```bash
# Lower budget for responsive chat
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 32 \
  --max-batch-input-tokens 4096 \
  --enable-in-flight-batching
```

### LoRA Lifecycle Management

**When:** Multi-tenant deployments with customer-specific adapters

**Do:**
```python
import requests

class LoRALifecycleManager:
    """Manages LoRA adapter lifecycle with capacity awareness."""

    def __init__(self, base_url: str, max_active: int = 4):
        self.base_url = base_url
        self.max_active = max_active
        self.loaded_adapters: set[str] = set()

    def load_adapter(self, name: str, path: str) -> bool:
        """Load adapter with capacity check."""
        if name in self.loaded_adapters:
            return True

        response = requests.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": path}
        )

        if response.status_code == 200:
            self.loaded_adapters.add(name)
            return True
        return False

    def unload_adapter(self, name: str) -> bool:
        """Explicitly unload adapter to free memory."""
        response = requests.post(
            f"{self.base_url}/v1/unload_lora_adapter",
            json={"lora_name": name}
        )

        if response.status_code == 200:
            self.loaded_adapters.discard(name)
            return True
        return False

# Usage
manager = LoRALifecycleManager("http://localhost:8000", max_active=4)

# Pre-load high-priority adapters at startup
for adapter in ["premium_customer_1", "premium_customer_2"]:
    manager.load_adapter(adapter, f"/adapters/{adapter}")
```

**Don't:**
```python
# BAD: Loading unlimited adapters without lifecycle management
# This exhausts GPU memory as each adapter consumes ~50-200MB
for adapter in adapters:
    requests.post(
        "http://localhost:8000/v1/load_lora_adapter",
        json={"lora_name": adapter, "lora_path": f"/adapters/{adapter}"}
    )
```

### Health Check Configuration

**When:** Kubernetes or load balancer deployment

**Do:**
```yaml
# Use dedicated health endpoint
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

**Don't:**
```yaml
# Using inference endpoint for health - expensive
readinessProbe:
  httpGet:
    path: /v1/models
```

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

---

## Decision Guide

| Feature / Scenario | When to Use | Configuration |
|---------------------|-------------|---------------|
| Streaming | Chat interfaces, interactive apps | `stream=True` in request |
| High token budget | Batch processing, long context | `--max-batch-input-tokens 16384` |
| Low token budget | Interactive latency | `--max-batch-input-tokens 4096` |
| Structured output | API integrations, data extraction | `--enable-structured-output` |
| Function calling | Agentic workflows, tool use | Model must support tools |
| LoRA adapters | Multi-tenant personalization | `--enable-lora --no-enable-prefix-caching --max-num-loras N` |
| Health endpoint | K8s, load balancers | `/health` |
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

- **Streaming**: Enable with `stream=True` for immediate TTFT
- **Token budget**: 4096-8192 for interactive, 16384+ for batch
- **Chunked prefill**: Enable for long context with `--enable-chunked-prefill`
- **Structured output**: Requires `--enable-structured-output` server flag
- **Function calling**: Llama 3.1+ and Mistral supported; check model compatibility
- **LoRA**: Incompatible with prefix caching; only Llama 3 QKVO layers supported
- **Health endpoint**: Use `/health` not `/v1/models` for probes
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
| `streaming error: connection reset` | Client disconnected mid-stream | Handle `ClientDisconnectedError` in error handler |
| `structured output validation failed` | JSON doesn't match schema | Verify schema is valid JSON Schema; check model supports structured output |
| `LoRA adapter not found` | Wrong adapter path or ID | Use adapter name from `adapter_config.json`, not file path |
| `function call not supported` | Model doesn't support tools | Check model compatibility; use Llama 3.1+ or Mistral |
| `max_tokens exceeded` | Request exceeds model context | Reduce `max_tokens` or use `--max-batch-total-tokens` server flag |
| `429 Too Many Requests` | Batch queue full | Increase `--max-batch-size` or add rate limiting on client |
| `request cancelled` | Client disconnected during processing | Normal behavior, resources are freed |
| `preemption` | Memory pressure caused eviction | Reduce batch size or add GPU memory |
| `timeout` | Request took too long | Check batch settings, enable chunked prefill |
| `stream error` | Network issue during streaming | Client should handle reconnection |
| `generation failed` | Internal error during generation | Check logs, may be OOM or model issue |

**LoRA-Specific Errors:**

| Scenario | HTTP Status | Cause |
|----------|-------------|-------|
| Invalid adapter path | 400 | Path does not exist or is not a valid adapter |
| Invalid adapter config | 400 | Missing `adapter_config.json` or missing required fields |
| Rank exceeds limit | Error | Adapter rank exceeds `--max-lora-rank` |
| Unsupported target modules | Error | Adapter uses modules other than q_proj, k_proj, v_proj, o_proj |
| Adapter not found | 404 | Adapter name not loaded when used for inference |
| Prefix caching conflict | Error | LoRA requires `--no-enable-prefix-caching` |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|--------------------|
| **Streaming** | `stream=True` | `stream=True` (unchanged) |
| **Structured output** | `--enable-structured-output` + `response_format` | Same |
| **Function calling** | Llama 3.1+ supported | Llama 3.1+ supported (unchanged) |
| **LoRA adapters** | Supported (Llama 3 only, QKVO) | Supported (unchanged) |
| **LoRA default max_num_loras** | 1 (changed from 100 in v25.x) | 1 |
| **Prefix caching** | Enabled by default | Enabled by default |
| **Cancellation** | Automatic on disconnect | Automatic on disconnect (unchanged) |
| **Preemption** | KV cache memory, LoRA capacity | KV cache memory, LoRA capacity |
| **CE watermark** | `--kvcache-ce-watermark` (default 0.95) | `--kvcache-ce-watermark` |

**Notes:**
- OpenAI-compatible API endpoints are stable across versions
- Streaming responses enhanced with chunked tokens in v26.1+
- Structured output (`response_format`) is stable
- Function calling for Llama 3.1+ and Mistral is stable
- LoRA adapter hot-swapping is stable
- Health endpoints (`/health`) are stable
- Request cancellation on client disconnect is stable
- Preemption handling for KV cache memory is stable
- Error propagation patterns are stable
- Scheduler cancellation and release patterns are stable

---

## Related Patterns

- [`serve-configuration.md`](serve-configuration.md) — Batch and scheduling configuration
- [`serve-kv-cache.md`](serve-kv-cache.md) — KV cache memory management
- [`deploy-production.md`](deploy-production.md) — Metrics and observability

---

## References

- [MAX Serve Configuration](https://docs.modular.com/max/serve)
- [OpenAI API Compatibility](https://docs.modular.com/max/cli/serve)
- Source: `max/serve/scheduler/text_generation_scheduler.py`, `max/interfaces/scheduler.py`
