---
title: MAX Serve API Features
description: Streaming, token budgets, structured output, function calling, LoRA adapters, and health endpoints
impact: HIGH
category: serve
tags: [streaming, tokens, structured-output, function-calling, lora, health]
error_patterns:
  - "streaming"
  - "token budget"
  - "structured output"
  - "function calling"
  - "LoRA"
  - "health check"
  - "API error"
scenarios:
  - "Enable streaming responses"
  - "Configure token budgets"
  - "Use structured output"
  - "Implement function calling"
  - "Manage LoRA adapters"
  - "Configure health endpoints"
consolidates:
  - serve-streaming.md
  - serve-token-budget.md
  - serve-structured-output.md
  - serve-function-calling.md
  - serve-lora-lifecycle.md
  - serve-health-endpoints.md
---

# MAX Serve API Features

**Category:** serve | **Impact:** HIGH

Comprehensive guide to MAX Serve API features including streaming responses, token budgets, structured output, function calling, LoRA adapter management, and health endpoints.

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

**LoRA Common Errors:**

| Scenario | HTTP Status | Cause |
|----------|-------------|-------|
| Invalid adapter path | 400 | Path does not exist or is not a valid adapter |
| Invalid adapter config | 400 | Missing `adapter_config.json` or missing required fields |
| Rank exceeds limit | Error | Adapter rank exceeds `--max-lora-rank` |
| Unsupported target modules | Error | Adapter uses modules other than q_proj, k_proj, v_proj, o_proj |
| Adapter not found | 404 | Adapter name not loaded when used for inference |
| Prefix caching conflict | Error | LoRA requires `--no-enable-prefix-caching` |

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

---

## Decision Guide

| Feature | When to Use | Configuration |
|---------|-------------|---------------|
| Streaming | Chat interfaces, interactive apps | `stream=True` in request |
| High token budget | Batch processing, long context | `--max-batch-input-tokens 16384` |
| Low token budget | Interactive latency | `--max-batch-input-tokens 4096` |
| Structured output | API integrations, data extraction | `--enable-structured-output` |
| Function calling | Agentic workflows, tool use | Model must support tools |
| LoRA adapters | Multi-tenant personalization | `--enable-lora --no-enable-prefix-caching --max-num-loras N` |
| Health endpoint | K8s, load balancers | `/health` |

---

## Quick Reference

- **Streaming**: Enable with `stream=True` for immediate TTFT
- **Token budget**: 4096-8192 for interactive, 16384+ for batch
- **Chunked prefill**: Enable for long context with `--enable-chunked-prefill`
- **Structured output**: Requires `--enable-structured-output` server flag
- **Function calling**: Llama 3.1+ and Mistral supported; check model compatibility
- **LoRA**: Incompatible with prefix caching; only Llama 3 QKVO layers supported
- **Health endpoint**: Use `/health` not `/v1/models` for probes

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

**Stable (v26.1):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Streaming response
stream = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

# Function calling (Llama 3.1+)
response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{"type": "function", "function": {"name": "get_weather"}}]
)
```

**Nightly (v26.2+):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Same APIs, chunked token responses in v26.1+
stream = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True  # Now returns chunked tokens
)
```

**Notes:**
- OpenAI-compatible API endpoints are stable across versions
- Streaming responses enhanced with chunked tokens in v26.1+
- Structured output (`response_format`) is stable
- Function calling for Llama 3.1+ and Mistral is stable
- LoRA adapter hot-swapping is stable
- Health endpoints (`/health`) are stable

---

## Related Patterns

- [`serve-configuration.md`](serve-configuration.md) — Batch and scheduling configuration
- [`serve-kv-cache.md`](serve-kv-cache.md) — Memory management for KV cache
- [`serve-monitoring.md`](serve-monitoring.md) — Metrics and observability

---

## References

- [MAX Serve Configuration](https://docs.modular.com/max/serve)
- [OpenAI API Compatibility](https://docs.modular.com/max/cli/serve)
