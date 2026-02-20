# MAX Best Practices - Full Reference

> **Auto-generated.** Do not edit manually.
>
> **Note:** Start with [SKILL.md](../SKILL.md) for the recommended entry point. This file is a complete index for deep dives.

## Table of Contents

**15 patterns** across **7 categories**

- [Multi-GPU & Parallelism](#multi-gpu--parallelism) (1 patterns)
- [MAX Serve Configuration](#max-serve-configuration) (5 patterns)
- [MAX Engine](#max-engine) (4 patterns)
- [MAX Graph API](#max-graph-api) (1 patterns)
- [Model Loading](#model-loading) (2 patterns)
- [Deployment](#deployment) (1 patterns)
- [Performance Optimization](#performance-optimization) (1 patterns)

---

## Multi-GPU & Parallelism

**Priority:** CRITICAL | **Patterns:** 1

### Multi-GPU Scaling

**Pattern:** `multigpu-scaling` | **Impact:** CRITICAL

Tensor parallelism, NVIDIA Hopper/Blackwell optimizations, AMD MI300 support, and device selection

See: [../patterns/multigpu-scaling.md](../patterns/multigpu-scaling.md)

---

## MAX Serve Configuration

**Priority:** CRITICAL | **Patterns:** 5

### MAX Serve API Features

**Pattern:** `serve-api` | **Impact:** HIGH

Streaming, token budgets, structured output, function calling, LoRA adapters, and health endpoints

See: [../patterns/serve-api.md](../patterns/serve-api.md)

### MAX Serve Configuration

**Pattern:** `serve-configuration` | **Impact:** CRITICAL

Comprehensive configuration for batch processing, ragged batching, scheduling, and environment settings

See: [../patterns/serve-configuration.md](../patterns/serve-configuration.md)

### KV Cache Management

**Pattern:** `serve-kv-cache` | **Impact:** CRITICAL

KV cache strategies, memory management, and prefix caching for efficient inference

See: [../patterns/serve-kv-cache.md](../patterns/serve-kv-cache.md)

### MAX Serve Monitoring and Telemetry

**Pattern:** `serve-monitoring` | **Impact:** HIGH

Metric levels, telemetry configuration, worker lifecycle, and disaggregated inference

See: [../patterns/serve-monitoring.md](../patterns/serve-monitoring.md)

### Request Lifecycle Management

**Pattern:** `serve-request-lifecycle` | **Impact:** CRITICAL

Request cancellation, preemption handling, and error propagation patterns

See: [../patterns/serve-request-lifecycle.md](../patterns/serve-request-lifecycle.md)

---

## MAX Engine

**Priority:** HIGH | **Patterns:** 4

### LoRA Adapter Serving

**Pattern:** `engine-lora` | **Impact:** MEDIUM

Serving LoRA adapters with MAX Serve for multi-tenant fine-tuned model inference

See: [../patterns/engine-lora.md](../patterns/engine-lora.md)

### MAX Engine Operations

**Pattern:** `engine-operations` | **Impact:** HIGH

Custom operations, architecture registration, inference sessions, and graph management

See: [../patterns/engine-operations.md](../patterns/engine-operations.md)

### Quantization Configuration

**Pattern:** `engine-quantization` | **Impact:** HIGH

Float8, GPTQ, and graph-level quantization patterns for memory-efficient inference

See: [../patterns/engine-quantization.md](../patterns/engine-quantization.md)

### Weight Management and Sharding

**Pattern:** `engine-weights` | **Impact:** HIGH

Weight sharding strategies, weight adapters, buffer transfers, and DLPack integration

See: [../patterns/engine-weights.md](../patterns/engine-weights.md)

---

## MAX Graph API

**Priority:** HIGH | **Patterns:** 1

### MAX Graph Construction

**Pattern:** `graph-construction` | **Impact:** HIGH

Graph building patterns including lazy context, modules, symbolic dimensions, and pipeline models

See: [../patterns/graph-construction.md](../patterns/graph-construction.md)

---

## Model Loading

**Priority:** HIGH | **Patterns:** 2

### Model Implementation

**Pattern:** `model-implementation` | **Impact:** HIGH

Building neural network models with MAX nn module and integrating into serving pipelines

See: [../patterns/model-implementation.md](../patterns/model-implementation.md)

### Model Loading and Configuration

**Pattern:** `model-loading` | **Impact:** HIGH

Supported model architectures, quantization formats, and HuggingFace token usage

See: [../patterns/model-loading.md](../patterns/model-loading.md)

---

## Deployment

**Priority:** MEDIUM | **Patterns:** 1

### Production Deployment

**Pattern:** `deploy-deployment` | **Impact:** HIGH

Container deployment, volume configuration, benchmarking, and cloud provider templates

See: [../patterns/deploy-deployment.md](../patterns/deploy-deployment.md)

---

## Performance Optimization

**Priority:** MEDIUM | **Patterns:** 1

### Performance Inference Optimization

**Pattern:** `perf-inference` | **Impact:** HIGH

Chunked prefill, in-flight batching, and KV cache swapping for optimal inference performance

See: [../patterns/perf-inference.md](../patterns/perf-inference.md)

---
