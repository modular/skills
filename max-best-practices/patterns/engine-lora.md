---
title: LoRA Adapter Serving
description: Serving LoRA adapters with MAX Serve for multi-tenant fine-tuned model inference
impact: MEDIUM
category: engine
tags: [lora, adapters, serving, inference]
error_patterns:
  - "LoRA functionality is not enabled on this server"
  - "LoRA adapter name already exists"
  - "Invalid LoRA adapter path"
  - "LoRA of rank exceeds maximum rank"
  - "LoRA only supports files in safetensors format"
  - "LoRA adapter contains unsupported target modules"
scenarios:
  - "Serve model with LoRA adapters"
  - "Load and unload LoRA adapters at runtime"
  - "Use different LoRA adapters per request"
---

# LoRA Adapter Serving

Best practices for serving LoRA (Low-Rank Adaptation) adapters with MAX Serve.

## Overview

MAX Serve supports serving multiple LoRA adapters from a single base model. Adapters are managed through an LRU cache, loaded/unloaded at runtime via REST API, and selected per-request by specifying the adapter name as the `model` parameter.

**Supported architectures:** Llama 3 family (including Llama 3.1, 3.2, 3.3)

**Supported target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention only; MLP modules like `gate_proj`, `up_proj`, `down_proj` are not yet supported)

**Weight format:** safetensors only

---

## Server Configuration

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--enable-lora` | bool | `false` | Enables LoRA adapter support |
| `--lora-paths` | list[str] | `[]` | Adapters to pre-load at startup (format: `name=/path` or just `/path`) |
| `--max-lora-rank` | int | `16` | Maximum rank of any LoRA adapter that can be loaded |
| `--max-num-loras` | int | `1` | Maximum number of adapters active simultaneously in a batch |

### Starting the Server with LoRA

```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-num-loras 4 \
  --max-lora-rank 16 \
  --lora-paths "my_adapter=/path/to/adapter1" "another=/path/to/adapter2"
```

Without `--enable-lora`, the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints return HTTP 501.

---

## REST API

### Load an Adapter

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "my_adapter", "lora_path": "/path/to/adapter"}'
```

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_name` | string | Yes | Name identifier for the adapter |
| `lora_path` | string | Yes | Local filesystem path to the adapter directory |

**Responses:**

| Status | Meaning |
|--------|---------|
| 200 | Adapter loaded successfully |
| 400 | Invalid path or invalid adapter format |
| 409 | Adapter name already exists with a different path |
| 500 | Unexpected error |
| 501 | LoRA not enabled on server |

**Note:** Remote HuggingFace repositories are not supported for `lora_path`. Download adapters locally first.

### Unload an Adapter

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "my_adapter"}'
```

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_name` | string | Yes | Name of the adapter to unload |

**Responses:**

| Status | Meaning |
|--------|---------|
| 200 | Adapter unloaded successfully |
| 404 | Adapter name not found |
| 500 | Unexpected error |
| 501 | LoRA not enabled on server |

### Use an Adapter for Inference

Specify the adapter name as the `model` field in any OpenAI-compatible request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my_adapter", "messages": [{"role": "user", "content": "Hello"}]}'
```

If the `model` field is empty, `null`, or matches the base model path, the base model is used without any LoRA adapter.

### List Loaded Adapters

Loaded adapters appear in the models list:

```bash
curl http://localhost:8000/v1/models
```

---

## Adapter Directory Format

Each adapter directory must contain:

1. **`adapter_config.json`** — adapter metadata
2. **Weight files in safetensors format** (e.g., `adapter_model.safetensors`)

### adapter_config.json

```json
{
  "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "lora_dropout": 0.0,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `r` | Yes | LoRA rank (must be <= `--max-lora-rank`) |
| `lora_alpha` | Yes | Scaling factor (scale = `lora_alpha / r`) |
| `target_modules` | Yes | Layers to apply LoRA (only `q_proj`, `k_proj`, `v_proj`, `o_proj` supported) |
| `bias` | No | Must be `"none"` (bias training not supported) |
| `lora_dropout` | No | Dropout rate (default 0.0) |
| `task_type` | No | `"CAUSAL_LM"` for text generation |

---

## Common Patterns

### Multi-Tenant Adapter Management

```python
import requests

BASE_URL = "http://localhost:8000"

def ensure_adapter_loaded(name: str, path: str) -> bool:
    """Load adapter if not already loaded."""
    response = requests.post(
        f"{BASE_URL}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": path}
    )
    # 200 = loaded, 409 = name exists (already loaded with same path returns 200)
    return response.status_code == 200

def unload_adapter(name: str) -> bool:
    """Unload adapter to free memory."""
    response = requests.post(
        f"{BASE_URL}/v1/unload_lora_adapter",
        json={"lora_name": name}
    )
    return response.status_code == 200
```

### Per-Request Adapter Selection

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Use specific LoRA adapter
response = client.chat.completions.create(
    model="my_adapter",  # adapter name, not base model
    messages=[{"role": "user", "content": "Hello!"}]
)

# Use base model (no LoRA)
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",  # base model path
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Constraints and Limitations

- **Attention modules only:** `q_proj`, `k_proj`, `v_proj`, `o_proj`. MLP target modules (`gate_proj`, `up_proj`, `down_proj`) are not yet supported.
- **safetensors only:** Other weight formats are not supported for LoRA adapters.
- **bias="none" only:** LoRA adapters trained with bias are not supported.
- **Local paths only:** `lora_path` must be a local filesystem path. Download HuggingFace adapters locally before loading.
- **LRU eviction:** When `max_num_loras` active adapters are in use, the least recently used adapter is evicted to make room for new activations.
- **Memory per adapter:** Each adapter adds approximately `2 * rank * hidden_size * num_layers` parameters.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `LoRA functionality is not enabled` | Server started without `--enable-lora` | Restart with `--enable-lora` |
| `LoRA of rank N exceeds maximum rank` | Adapter rank > `--max-lora-rank` | Increase `--max-lora-rank` or use a lower-rank adapter |
| `LoRA only supports files in safetensors format` | Adapter uses PyTorch `.bin` format | Convert to safetensors |
| `LoRA adapter contains unsupported target modules` | Adapter targets MLP layers | Use adapter targeting only attention modules |
| `LoRA bias training is not currently supported` | Adapter has `bias != "none"` | Use adapter with `bias="none"` |
| `Adapter config file not found` | Missing `adapter_config.json` | Ensure adapter directory has `adapter_config.json` |
| `Invalid LoRA adapter path` | Path doesn't exist locally | Verify path exists; download remote adapters first |
| `LoRA adapter name already exists with different path` | Name collision | Use a different name or unload the existing adapter first |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **LoRA serving via CLI** | Stable | `--enable-lora`, `--lora-paths`, `--max-lora-rank`, `--max-num-loras` |
| **REST adapter management** | Stable | `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` |
| **Attention-only LoRA** | Stable | Only attention module targets supported |
| **Safetensors format** | Required | PyTorch `.bin` format not supported |

---

## See Also

- [MAX Serve API Features](../patterns/serve-api.md) — Full LoRA lifecycle section with additional context
- [Engine Operations Patterns](engine-operations.md) — General engine management
