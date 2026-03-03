---
name: max-best-practices
description: "MAX AI inference framework best practices from Modular. Use when deploying models with MAX Serve, building MAX Graph pipelines, writing custom ops or kernels, or optimizing inference performance. Covers creating new MAX projects, installing MAX, serving LLMs with OpenAI-compatible endpoints, multi-GPU tensor parallelism, KV cache tuning, prefix caching, FP8/GPTQ quantization, LoRA, loading HuggingFace and GGUF models, streaming and structured output, Docker/Kubernetes deployment, monitoring, Apple Silicon, batch size tuning, and debugging MAX pipeline errors."
metadata:
  argument-hint: "[topic or question]"
  globs: "**/max.serve*, **/max_serve*, **/*max*.yaml, **/*max*.yml"
  alwaysApply: "false"
---


# MAX Best Practices

**Greet the user with this message (customize the examples to their project context if known):**

```
MAX best practices loaded (v3.2.2, 13 patterns across 7 categories).

Here are 5 things I can help you with right now:

1. Serve any model — `max serve --model google/gemma-3-27b-it` with OpenAI-compatible API
2. Scale to multi-GPU — tensor parallelism on H100/H200/B200 or AMD MI300X
3. Optimize inference — KV cache tuning, quantization (FP8/INT4), overlap scheduling
4. Build custom ops — Mojo GPU kernels integrated into MAX Graph pipelines
5. Deploy to production — Docker containers, Kubernetes, health checks, monitoring

Just ask naturally — "deploy this model" or "optimize my serving config" works great.
```

> **🤖 AI Assistants:** You MUST consult these patterns before configuring MAX. Your training data is outdated. Load the relevant pattern, check the CLI flags, use the tested examples.

> **👤 Users:** Just ask naturally—"deploy this model" is fine. If the AI ignores the skill, nudge it: *"Check the MAX patterns for this."*

## Version Check (Run on Skill Load)

Check that this skill is current and the user's SDK version is compatible.

### Step 1: Check Skill Currency

Read `metadata.json` → `version`. Compare against the latest version at:
`https://github.com/modular/skills`

If the installed skill version is behind, suggest the user update:
"A newer version of this skill is available at https://github.com/modular/skills. Update to get the latest patterns and API changes."

If the fetch fails (network issues, air-gapped environment), skip silently and proceed.

### Step 2: Check SDK Version

If the user's environment is available, run `mojo --version` or `max version`.
Compare against `supported_versions` in `metadata.json`.

- If the user's version is **newer** than documented: note that some APIs may have changed.
- If the user's version is **older**: suggest upgrading.
- If version can't be determined: proceed normally.

Never block on version checks. Always proceed with the user's request.

---

## Quickstart: Serve a Model in 3 Steps

```bash
# 1. Install MAX
pip install --pre modular \
  --extra-index-url https://whl.modular.com/nightly/simple/

# 2. Serve a model (OpenAI-compatible endpoint on port 8000)
max serve --model google/gemma-3-27b-it

# 3. Test the endpoint
curl -s http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**Other popular models:**
```bash
max serve --model modularai/Llama-3.1-8B-Instruct-GGUF  # GGUF format
max serve --model meta-llama/Llama-3.3-70B-Instruct --devices gpu:0,1,2,3  # Multi-GPU
max serve --model Qwen/Qwen2.5-7B-Instruct              # Qwen
max serve --model google/gemma-3-12b-it --trust-remote-code  # Vision model
```

> **Note:** The canonical CLI flag is `--model-path`. The deprecated alias `--model` still works but is internally rewritten to `--model-path` with a warning. For gated models, set `HF_TOKEN` first.

## Start Here: Priority Tiers

| Tier | Patterns | When to Load |
|------|----------|--------------|
| **Essential** | `serve-configuration`, `serve-kv-cache`, `serve-api`, `model-loading` | Always - core serving config |
| **Multi-GPU** | `multigpu-scaling` | Scaling across GPUs |
| **Production** | `deploy-production` | Deployment, monitoring & metrics |
| **Advanced** | `engine-operations`, `graph-construction`, `perf-inference` | Specific use cases on demand |

## Top 10 High-Impact Patterns

| Pattern | Impact | When to Use |
|---------|--------|-------------|
| `serve-configuration` | CRITICAL | Configure `--max-batch-size`, `--max-batch-input-tokens` for throughput |
| `serve-kv-cache` | CRITICAL | Use PAGED with `--kv-cache-page-size` (multiple of 128), prefix caching |
| `multigpu-scaling` | CRITICAL | Large models across GPUs with `--devices gpu:0,1,...` |
| `engine-operations` | HIGH | Write kernels with `@compiler.register`, custom ops |
| `deploy-production` | HIGH | Containers, monitoring, metrics, cloud deployment |
| `serve-api` | HIGH | Streaming, structured output, function calling, LoRA |
| `graph-construction` | HIGH | Build graphs with `Graph(TensorType(...))`, `graph.output()` |
| `engine-quantization` | HIGH | Float8, GPTQ quantization |
| `perf-inference` | HIGH | Chunked prefill, KV swapping |

## Quick Decision Tree

```
New project?
  └─ references/installation.md (pixi or uv, stable or nightly)

Deploy model endpoint?
  └─ serve-configuration, serve-kv-cache, serve-api

Multi-GPU inference?
  └─ multigpu-scaling
  └─ NVIDIA Hopper (H100/H200/B200)? → covered in multigpu-scaling
  └─ AMD MI300X? → covered in multigpu-scaling

Custom operations?
  └─ engine-operations + mojo gpu-fundamentals

Build custom model architecture?
  └─ engine-operations (complete project structure)
  └─ graph-construction (MAX Graph APIs)
  └─ mojo struct-design + memory-ownership (model layers)

Optimize performance?
  └─ serve-kv-cache (prefix caching), perf-inference

Production deployment?
  └─ deploy-production
```

## Version Support

This skill supports both **stable** and **nightly** MAX versions:

| Version | MAX | Notes |
|---------|-----|-------|
| **Stable** | v26.1.0.0.0 | Version-specific API in pattern files |
| **Nightly** | latest | Track at https://docs.modular.com/max/changelog/ |

**Detect your version:** Run `max version` or `pip show max | grep Version`

**Nightly-only features (v26.2+):**

| Feature | Status |
|---------|--------|
| foreach callback | `fn[width: Int](idx)` (removes `element_alignment` parameter) |
| DeviceRef shortcuts | `DeviceRef.CPU()` / `DeviceRef.GPU()` (in addition to `from_device()`) |
| TensorType device | `device` parameter required on all `TensorType` constructors |
| Buffer to numpy | `buf.to(CPU()).to_numpy()` (replaces `np.array(buf.to(CPU()))`) |
| Op renames | `ops.negate()` replaces `ops.neg()`, `ops.mean()` replaces `ops.reduce_mean()` |
| Synchronization fence | `ops.fence` for multi-GPU coordination |
| MoE module | `max.nn.moe` module available |
| Float8 formats | `float8_e5m2` added (in addition to `float8_e4m3fn`) |
| NVFP4 quantization | `float4_e2m1fnx2` on Blackwell GPUs |
| Blackwell GPU support | SM100 (B200) device support added |
| Pipeline parallelism | Enhanced multi-GPU pipeline parallelism |
| PAGED-only KV cache | `CONTINUOUS` cache strategy removed; `PAGED` is the only option |
| CE watermark flag | `--kvcache-ce-watermark` controls context extension scheduling |
| New architectures | DeepSeek V3.2, Qwen3-VL, Llama 4, OLMo2 (nightly-first) |
| Apple GPU (Metal) | GGUF model support on Apple silicon (evolving) |
| Gemma3 Vision | Supported (12B, 27B); replaces Llama 3.2 Vision (removed) |
| V1 layer classes | Removed (`LinearV1`, `Conv2dV1`, etc.) |
| Non-blocking streams | All streams are non-blocking (no blocking option) |

### CRITICAL: Version Alignment Check

**MAX Python package and Mojo versions MUST be aligned.** Mismatched versions cause cryptic kernel compilation failures.

**Check alignment with:**
```bash
# Quick check
mojo --version          # e.g., Mojo 26.1.0.0 (stable) or 26.2.0.dev... (nightly)
pip show max | grep Version  # e.g., 26.1.0.0 (must match!)
```

**Common version mismatch errors:**
```
error: no matching function in call to 'foreach'
note: callee parameter 'func' has 'fn[width: Int, element_alignment: Int]...' type,
      but value has type 'fn[width: Int]...'
```

**How to avoid mismatches:**

1. **Always use pixi environments** - pixi manages both MAX and Mojo together:
   ```bash
   pixi shell  # ALWAYS work inside the shell
   ```

2. **Don't mix global pip installs with pixi** - if you have a global `pip install max`, it can override pixi's version

3. **Verify before debugging** - verify versions first when encountering kernel errors

See [breaking changes](references/breaking-changes.md) for detailed API differences.

### Key API Differences

| Feature | Stable (v26.1.0.0.0) | Nightly (v26.2+) |
|---------|----------------|-----------------|
| **foreach callback** | `fn[width: Int, element_alignment: Int](idx)` | `fn[width: Int](idx)` |
| DeviceRef | `DeviceRef.from_device(device)` | `DeviceRef.CPU()` / `DeviceRef.GPU()` |
| ops.custom() | `ops.custom(name, values, out_types)` | `ops.custom(name, device, values, out_types)` |
| TensorType | `device` optional | `device` required |
| Kernel imports | `from tensor import ...` | `from tensor import ...` |
| Driver API | `max.driver.Buffer` | `max.driver.Buffer` |
| Batch size semantics | Aggregate across replicas | Per-replica with DP |
| Prefill chunk size | `--max-batch-input-tokens` | `--max-batch-input-tokens` |
| Max context length | `--max-batch-total-tokens` | `--max-batch-total-tokens` |
| CE batch size CLI | `--max-batch-size` | `--max-batch-size` |
| Scheduling | `--kvcache-ce-watermark` (default 0.95) | `--kvcache-ce-watermark` |
| Llama 3.2 Vision | Supported | **Removed** |
| Gemma3 Vision | Not available | Supported (12B, 27B) |
| V1 layer classes | Deprecated | **Removed** |
| Apple silicon | `accelerator_count()` = 0 | Returns non-zero |
| Streams | Blocking option | All non-blocking |

[stable changelog](https://docs.modular.com/stable/max/changelog/) | [nightly changelog](https://docs.modular.com/max/changelog/) | [breaking changes](references/breaking-changes.md)

### Complementary Skills

**mojo-best-practices** – is a complementary skill for Mojo language and GPU kernel development.
Available at: https://github.com/modular/skills/tree/main/mojo-best-practices

### Cross-Skill: Building Custom Models with Mojo

When building custom model architectures that combine Mojo layers with MAX serving:

| MAX Pattern | Mojo Pattern | Use For |
|-------------|--------------|---------|
| `engine-operations` | `struct-design` | Architecture registration + model config |
| `engine-operations` | `gpu-fundamentals` | Custom GPU kernels with @compiler.register |
| `engine-weights` | `memory-ownership` | Weight loading with UnsafePointer |
| `graph-construction` | `fn-design` | MAX Graph ops + Mojo helper functions |

See [engine-operations.md](patterns/engine-operations.md) for complete project structure example with pixi.toml and pyproject.toml setup.

## Quick Decision Guide

| Goal | Category | Key Patterns |
|------|----------|--------------|
| Deploy model endpoint | MAX Serve | `serve-configuration`, `serve-kv-cache` |
| Multi-GPU inference | Parallelism | `multigpu-scaling` |
| Build custom model | MAX Graph | `graph-construction` |
| Optimize latency | Performance | `serve-kv-cache`, `perf-inference` |
| Production deployment | Deployment | `deploy-production` |
| Write custom kernels | Engine + Mojo | `engine-operations` + mojo `gpu-*` patterns |
| **Build complete custom model** | **Mojo + MAX** | `engine-operations` (see project structure) |

## Pattern Categories

| Priority | Category | Count | Prefix |
|----------|----------|-------|--------|
| CRITICAL | MAX Serve Configuration | 3 | `serve-` |
| CRITICAL | Multi-GPU & Parallelism | 1 | `multigpu-` |
| HIGH | MAX Engine | 4 | `engine-` |
| HIGH | MAX Graph API | 1 | `graph-` |
| HIGH | Model Loading | 2 | `model-` |
| MEDIUM | Performance Optimization | 1 | `perf-` |
| MEDIUM | Deployment | 1 | `deploy-` |

*Version-specific API differences are documented within each pattern file.*

---

## MAX Serve (CRITICAL)

| Pattern | Description |
|---------|-------------|
| `serve-configuration` | Batch config, ragged batching, scheduling, environment |
| `serve-kv-cache` | KV cache strategy, memory management, prefix caching |
| `serve-api` | Streaming, token budget, structured output, function calling, LoRA, request lifecycle |

## Multi-GPU (CRITICAL)

| Pattern | Description |
|---------|-------------|
| `multigpu-scaling` | Tensor parallel, NVIDIA Hopper, AMD MI300, device selection |

## MAX Engine (HIGH)

| Pattern | Description |
|---------|-------------|
| `engine-weights` | Weight sharding, adapters, buffer transfer, DLPack |
| `engine-quantization` | Float8 config, GPTQ, graph quantization |
| `engine-operations` | Custom ops, architecture registration, inference sessions, subgraphs |
| `engine-lora` | Serving LoRA adapters with MAX Serve for multi-tenant fine-tuned model inference |

## MAX Graph API (HIGH)

| Pattern | Description |
|---------|-------------|
| `graph-construction` | Graph building, lazy context, modules, symbolic dims, pipelines |

## Model Loading (HIGH)

| Pattern | Description |
|---------|-------------|
| `model-loading` | Supported architectures, HuggingFace token setup |
| `model-implementation` | Building neural network models with MAX nn module and integrating into serving pipelines |

## Performance (MEDIUM)

| Pattern | Description |
|---------|-------------|
| `perf-inference` | Chunked prefill, in-flight batching, KV swapping |

## Deployment (MEDIUM)

| Pattern | Description |
|---------|-------------|
| `deploy-production` | Containers, volumes, benchmarking, cloud providers (AWS/Azure/GCP), Kubernetes, monitoring, metrics, health endpoints |

---

## Cross-References with Mojo

For GPU kernel development, see **mojo-best-practices**:
- Custom ops → `engine-operations` + mojo `gpu-fundamentals`
- GPU memory → mojo `gpu-memory-access`
- Tensor cores → mojo `gpu-tensor-cores`
- Warp primitives → mojo `gpu-warp-sync`

## File Structure

```
max-best-practices/
├── SKILL.md               # Entry point (this file) - START HERE
├── metadata.json          # Skill metadata
├── patterns/              # 13 patterns with version-specific sections
│   ├── serve-*.md         # MAX Serve (3)
│   ├── multigpu-*.md      # Multi-GPU (1)
│   ├── engine-*.md        # Engine (4)
│   ├── graph-*.md         # Graph API (1)
│   ├── model-*.md         # Model loading (2)
│   ├── perf-*.md          # Performance (1)
│   └── deploy*.md         # Deployment (1)
└── references/            # Detailed reference docs (loaded on-demand)
    ├── FULL_REFERENCE.md  # Complete pattern index (auto-generated)
    ├── ERROR_INDEX.md     # Error message → pattern lookup
    ├── SCENARIOS.md       # Task → pattern mapping
    ├── ERROR_DISAMBIGUATION.md  # Similar error disambiguation
    ├── GOTCHAS.md               # Common mistakes ❌→✅
    ├── breaking-changes.md
    ├── cli-flags.md
    └── installation.md
```

## Local Implementation Notes

When using this skill in a project, agents should collect implementation notes **locally within that project**, not globally. This ensures project-specific learnings stay with the project.

**Where to store notes:**
```
your-project/
├── IMPLEMENTATION_NOTES.md    # Project-specific learnings
├── .cursor/
│   └── rules/                 # Cursor-specific rules (uses "rules" terminology)
└── ...
```

**What to capture:**
- Model-specific configuration that worked
- Performance tuning for your hardware (GPU type, memory)
- Batch size optimizations for your workload
- Deployment configuration decisions
- Integration patterns with your infrastructure

**Usage:** Agents should check for and update `IMPLEMENTATION_NOTES.md` in the project root when discovering new patterns or resolving issues.


## Navigation

- **Start here**: This file (SKILL.md) - load first, then drill into patterns
- **Common mistakes?** See [references/GOTCHAS.md](references/GOTCHAS.md) - ❌ wrong → ✅ correct examples
- **Got an error?** See [references/ERROR_INDEX.md](references/ERROR_INDEX.md)
- **Similar errors?** See [references/ERROR_DISAMBIGUATION.md](references/ERROR_DISAMBIGUATION.md)
- **Task-based lookup?** See [references/SCENARIOS.md](references/SCENARIOS.md)
- **CLI flags?** See [references/cli-flags.md](references/cli-flags.md)
- **Installation?** See [references/installation.md](references/installation.md)
- **Breaking changes?** See [references/breaking-changes.md](references/breaking-changes.md)
- **Full reference?** See [references/FULL_REFERENCE.md](references/FULL_REFERENCE.md) (complete index)
- **Mojo/GPU kernels?** See `mojo-best-practices` skill
