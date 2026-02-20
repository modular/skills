# Error Disambiguation Guide

When an error message matches multiple patterns, use this guide to determine which pattern to consult first.

---

## "latency" errors

**Priority 1: Check [`serve-monitoring.md`](../patterns/serve-monitoring.md) if:**
- You're investigating response time in production
- Monitoring or observability tools report high latency
- You need to identify slow endpoints or request phases

**Priority 2: Check [`perf-inference.md`](../patterns/perf-inference.md) if:**
- Latency is in the model inference step
- You're optimizing model execution time
- The latency occurs during forward pass, not in serving layer

---

## "out of memory" errors

**Priority 1: Check [`serve-kv-cache.md`](../patterns/serve-kv-cache.md) if:**
- OOM occurs during LLM serving with long contexts
- Memory grows with sequence length
- Error mentions KV cache or attention memory

**Priority 2: Check [`serve-configuration.md`](../patterns/serve-configuration.md) if:**
- OOM occurs at model loading or server startup
- You're configuring batch sizes or worker counts
- Error is about model weights not fitting in memory

---

## "throughput" errors

**Priority 1: Check [`serve-monitoring.md`](../patterns/serve-monitoring.md) if:**
- You're measuring requests per second
- Monitoring shows throughput below expectations
- You need to identify bottlenecks in the serving pipeline

**Priority 2: Check [`perf-inference.md`](../patterns/perf-inference.md) if:**
- Throughput is limited by model execution speed
- You want to optimize batching or parallelism
- The bottleneck is GPU utilization during inference

---

## General Disambiguation Strategy

When unsure which pattern to check:

1. **Check where the issue manifests**:
   - At startup or config time → Check configuration patterns
   - During serving → Check serving/monitoring patterns
   - During inference → Check inference patterns

2. **Check the resource being exhausted**:
   - GPU memory → Usually KV cache or model size
   - CPU memory → Usually configuration or worker settings
   - Time (latency/throughput) → Check both serving and inference

3. **Check both patterns** - Start with Priority 1, then move to Priority 2 if unresolved

---

*This file is maintained manually. If you discover new ambiguous error patterns, please update this file.*
