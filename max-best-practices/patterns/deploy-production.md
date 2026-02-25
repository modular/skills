---
title: Production Deployment and Monitoring
description: Container deployment, volume configuration, benchmarking, cloud provider templates, metrics, telemetry, and disaggregated inference
impact: HIGH
category: deploy
tags: [container, docker, kubernetes, aws, azure, gcp, benchmark, metrics, telemetry, monitoring, workers, disaggregated]
error_patterns:
  - "container"
  - "docker"
  - "kubernetes"
  - "deployment failed"
  - "volume"
  - "mount"
  - "pod"
  - "image"
  - "metrics"
  - "telemetry"
  - "monitoring"
  - "worker"
  - "TTFT"
  - "latency"
  - "throughput"
scenarios:
  - "Deploy MAX Serve in production"
  - "Configure Docker container"
  - "Set up Kubernetes deployment"
  - "Deploy on AWS/Azure/GCP"
  - "Benchmark production deployment"
  - "Configure volume mounts"
  - "Monitor MAX Serve deployment"
  - "Configure metrics collection"
  - "Set up telemetry"
  - "Debug performance issues"
  - "Track TTFT and latency"
  - "Monitor worker lifecycle"
consolidates:
  - deploy-container.md
  - deploy-container-volumes.md
  - deploy-benchmark.md
  - deploy-aws-cloudformation.md
  - deploy-azure-arm.md
  - deploy-gcp-deployment-manager.md
  - deploy-kubernetes.md
  - serve-metric-levels.md
  - serve-metrics-telemetry.md
  - serve-model-worker-lifecycle.md
  - serve-disaggregated-inference.md
  - deploy-deployment.md
  - serve-monitoring.md
---
<!-- PATTERN QUICK REF
WHEN: Deploying MAX models to production, monitoring inference services, configuring telemetry
KEY_TYPES: Docker containers, Kubernetes Deployments, CloudFormation/ARM/GCP templates, Prometheus metrics, disaggregated inference workers
SYNTAX:
  - docker run --gpus=1 -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 modular/max-nvidia-full:latest --model-path MODEL
  - export MAX_SERVE_METRIC_LEVEL=BASIC; export MAX_SERVE_METRIC_RECORDING_METHOD=PROCESS
  - max benchmark --model MODEL --collect-gpu-stats
  - max serve --pipeline-role PrefillOnly|DecodeOnly
PITFALLS: Missing volume mounts (30+ min cold start), DETAILED metrics in prod (5-15% overhead), SYNC recording method (blocking), no heartbeat for large models, fork multiprocessing with GPU models
RELATED: serve-api, serve-configuration, perf-inference, multigpu-scaling
-->

# Production Deployment and Monitoring Runbook

**Category:** deploy | **Impact:** HIGH

End-to-end runbook for deploying MAX Serve to production: containers, volumes, Kubernetes, cloud providers, benchmarking, metrics, telemetry, worker lifecycle, and disaggregated inference.

---

## 1. Deployment

### Official MAX Containers

| Container | Target |
|-----------|--------|
| `modular/max-nvidia-full` | NVIDIA GPU |
| `modular/max-amd` | AMD GPU |

**NVIDIA GPUs:**
```bash
docker run --gpus=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    modular/max-nvidia-full:latest \
    --model-path google/gemma-3-27b-it
```

**AMD GPUs:**
```bash
docker run --device=/dev/kfd --device=/dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    modular/max-amd:latest \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

### Volume Configuration for Fast Startup

Mount cache volumes to reduce container startup from 30+ minutes to seconds.

| Volume | Container Path | Purpose |
|--------|---------------|---------|
| HuggingFace cache | `/root/.cache/huggingface` | Downloaded model files |
| MAX cache | `/opt/venv/share/max/.max_cache` | Compiled model artifacts |

**Do:**
```bash
docker run --gpus 1 -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
  --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
  modular/max-nvidia-full:latest \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

**Don't:**
```bash
# Every container start downloads and compiles (30+ minutes)
docker run --gpus 1 -p 8000:8000 \
  modular/max-nvidia-full:latest \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

**Cold Start Comparison:**

| Phase | Without Cache | With Cache |
|-------|--------------|------------|
| Model download | 5-30 min | 0 sec |
| Model compilation | 2-10 min | 0 sec |
| Model loading | 10-30 sec | 10-30 sec |
| **Total** | 7-40 min | 10-30 sec |

### Pre-warming Strategy

Pre-download models before container deployment:

```bash
# Pre-download model to host
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct

# Then start container (skips download)
docker run ...
```

### Complete NVIDIA Container Setup

```bash
docker run \
  --gpus 1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
  --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
  --env "HF_TOKEN=<your-token>" \
  -p 8000:8000 \
  --ipc=host \
  modular/max-nvidia-full:latest \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

Key flags: `--ipc=host` (optimal GPU memory sharing), `HF_HUB_ENABLE_HF_TRANSFER=1` (faster downloads).

### Complete AMD Container Setup

```bash
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
  --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
  --env "HF_TOKEN=$HF_TOKEN" \
  --group-add keep-groups \
  --device /dev/kfd \
  --device /dev/dri \
  -p 8000:8000 \
  modular/max-amd:latest \
  --model-path meta-llama/Llama-3.1-8B-Instruct
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: max-llm
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: max-serve
        image: modular/max-nvidia-full:latest
        args:
          - "--model-path"
          - "meta-llama/Llama-3.1-8B-Instruct"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        volumeMounts:
        - name: hf-cache
          mountPath: /root/.cache/huggingface
        - name: max-cache
          mountPath: /opt/venv/share/max/.max_cache
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
      volumes:
      - name: hf-cache
        persistentVolumeClaim:
          claimName: hf-cache-pvc
      - name: max-cache
        persistentVolumeClaim:
          claimName: max-cache-pvc
```

**PersistentVolumeClaim:**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: max-cache-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi  # Adjust based on model size
```

Use `/health` endpoint for probes, not inference endpoints.

---

## 2. Configuration

### Worker Timeout and Health

Configure timeouts based on model size for production reliability.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SERVE_MW_TIMEOUT` | None | Model worker startup timeout (seconds) |
| `MAX_SERVE_USE_HEARTBEAT` | False | Enable periodic heartbeat checks |
| `MAX_SERVE_MW_HEALTH_FAIL` | 60.0 | Max seconds without heartbeat |

**Do (large models 70B+):**
```bash
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

Timeout rule of thumb: ~30s per 10B parameters.

### Model Worker Lifecycle

MAX Serve uses a factory pattern with async context management. Use `spawn` multiprocessing context (not `fork`) for GPU models.

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

---

## 3. Health Checks

### Endpoints

- **Health check**: `GET /health` on port 8000
- **Metrics**: `GET /metrics` on port 8001 (dedicated)

### Kubernetes Probes

Use `/health` for readiness/liveness probes. For large models, increase `initialDelaySeconds`:

| Model Size | `initialDelaySeconds` |
|------------|----------------------|
| 7-8B | 120s |
| 13B | 180s |
| 70B+ | 600s |

### Heartbeat Monitoring

Enable for production to detect hung workers:

```bash
export MAX_SERVE_USE_HEARTBEAT=true
export MAX_SERVE_MW_HEALTH_FAIL=60  # Restart if no heartbeat for 60s
```

---

## 4. Monitoring

### Metric Levels

| Level | Value | Metrics Collected | Performance Impact | Use Case |
|-------|-------|-------------------|-------------------|----------|
| `NONE` | 0 | No metrics | Zero overhead | Minimal latency requirements |
| `BASIC` | 10 | Request counts, TTFT, input/output tokens | Minimal (<1%) | **Production deployments** |
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
| `PROCESS` | Separate process | Minimal | **Production (isolated)** |

### Production Metrics Configuration

**Do:**
```bash
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
# Result: 5-15% latency increase from ITL measurements on every token
```

### Debug/Optimization Metrics

```bash
export MAX_SERVE_METRIC_LEVEL=DETAILED
export MAX_SERVE_METRIC_RECORDING_METHOD=ASYNCIO
export MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR=20

max serve --model meta-llama/Llama-3.1-8B-Instruct
```

### Prometheus Endpoint

```bash
curl http://localhost:8001/metrics

# Example output
maxserve_time_to_first_token_milliseconds_bucket{le="100.0"} 42
maxserve_request_count_total{code="200",path="/v1/chat/completions"} 1234
maxserve_batch_size_bucket{le="16"} 89
```

### Grafana Dashboard Queries

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

### Benchmarking

Run before production deployment for baseline performance:

```bash
# Basic benchmark
max benchmark --model google/gemma-3-27b-it

# With GPU stats collection
max benchmark --model meta-llama/Llama-3.1-8B-Instruct \
  --collect-gpu-stats

# With tracing for profiling
max benchmark --model meta-llama/Llama-3.1-8B-Instruct \
  --trace \
  --trace-file benchmark.nsys-rep \
  --gpu-profiling detailed

# Specific batch size
max benchmark --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 32 \
  --devices gpu:0

# Multi-GPU benchmark
max benchmark --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3
```

**Key Metrics:**

| Metric | Description |
|--------|-------------|
| TTFT | Time To First Token |
| ITL | Inter-Token Latency |
| Throughput | Tokens per second |
| GPU utilization | Memory and compute usage |

Use `max benchmark` for raw model performance. Use `max serve` + load testing tool (`wrk`, `hey`) for end-to-end latency including API overhead.

---

## 5. Disaggregated Inference

Separates prefill (context encoding) and decode (token generation) into different workers.

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

### When to Use

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

### Trade-offs

| Aspect | Unified | Disaggregated |
|--------|---------|---------------|
| Latency | Lower for small requests | Higher (transfer overhead) |
| Throughput | Good | Better at scale |
| Complexity | Simple | More operational overhead |
| GPU utilization | Variable | More consistent |

### Configuration

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

### Worker Characteristics

| Property | Prefill Workers | Decode Workers |
|----------|----------------|----------------|
| Workload | Context encoding | Token generation |
| Compute pattern | Compute-bound | Memory-bound |
| Batch strategy | Small batches | Large batches |
| GPU memory | Higher KV allocation | Smaller per-seq |
| Optimal GPU | High compute (H100, MI300X) | High bandwidth (H100, MI300X) |

### KV Transfer Mechanisms

| Mechanism | Bandwidth | Latency | Use Case |
|-----------|-----------|---------|----------|
| PCIe 4.0 | 32 GB/s | ~1-10ms | Single-node, different PCIe domains |
| PCIe 5.0 | 64 GB/s | ~0.5-5ms | Single-node, latest hardware |
| NVLink | 450-900 GB/s | ~0.1ms | Same-node, NVIDIA GPUs |
| RDMA (InfiniBand) | 100-400 GB/s | ~1-5us | Multi-node |
| NVSwitch | 1.8 TB/s | ~0.1ms | DGX systems |

### Monitoring Disaggregated Deployments

Monitor using standard metrics with special attention to:
- `maxserve.time_to_first_token` — higher due to KV transfer overhead
- `maxserve.num_requests_queued` — separate prefill/decode queue pressure
- `maxserve.cache.num_used_blocks` — cache utilization across workers

---

## 6. Cloud Provider Deployments

### AWS CloudFormation

| Instance | GPU | Memory | Use Case |
|----------|-----|--------|----------|
| g5.xlarge | 1x A10G (24GB) | 16GB | Small models (7-8B) |
| g5.4xlarge | 1x A10G (24GB) | 64GB | Medium models (8-13B) |
| g5.12xlarge | 4x A10G (96GB) | 192GB | Large models (70B) |
| p4d.24xlarge | 8x A100 (320GB) | 1.1TB | Very large models |

```bash
aws cloudformation create-stack \
  --stack-name max-serve \
  --template-body file://max-nvidia-aws.yaml \
  --parameters ParameterKey=HuggingFaceHubToken,ParameterValue=$HF_TOKEN
```

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: MAX Serve deployment on EC2 with NVIDIA GPU

Parameters:
  InstanceType:
    Type: String
    Default: g5.4xlarge
  HuggingFaceHubToken:
    Type: String
    NoEcho: true

Resources:
  MaxServeInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: ami-02769e6d1f6a88067  # Deep Learning AMI
      BlockDeviceMappings:
        - DeviceName: /dev/xvda
          Ebs:
            VolumeSize: 100
            VolumeType: gp3
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          docker run -d \
            --env "HF_TOKEN=${HuggingFaceHubToken}" \
            -v /home/ec2-user/.cache/huggingface:/root/.cache/huggingface \
            --gpus 1 -p 80:8000 --ipc=host \
            modular/max-nvidia-full:latest \
            --model-path meta-llama/Llama-3.1-8B-Instruct
```

Best practices: Use Deep Learning AMIs, store HF_TOKEN in Secrets Manager, 100GB+ gp3 storage, `--ipc=host`.

### Azure ARM Templates

| VM Size | GPU | Memory | Use Case |
|---------|-----|--------|----------|
| Standard_NC4as_T4_v3 | 1x T4 (16GB) | 28GB | Small models |
| Standard_NV36ads_A10_v5 | 1x A10 (24GB) | 440GB | Medium models (8-13B) |
| Standard_NC24ads_A100_v4 | 1x A100 (80GB) | 220GB | Large models (70B) |
| Standard_ND96isr_MI300X_v5 | 8x MI300X | 1.8TB | AMD, very large models |

```bash
# NVIDIA deployment
az deployment group create \
  --resource-group myResourceGroup \
  --template-file max-nvidia-azure.json \
  --parameters hfToken=$HF_TOKEN

# AMD deployment
az deployment group create \
  --resource-group myResourceGroup \
  --template-file max-amd-azure.json \
  --parameters hfToken=$HF_TOKEN
```

Azure notes: NVIDIA uses NVIDIA AI Enterprise marketplace image. AMD MI300X uses Ubuntu HPC + ROCm with larger disk (256GB).

### GCP Deployment Manager

| Machine Type | GPU | Memory | Use Case |
|--------------|-----|--------|----------|
| n1-standard-8 + T4 | 1x T4 (16GB) | 30GB | Small models |
| n1-standard-16 + L4 | 1x L4 (24GB) | 60GB | Medium models (8B) |
| a2-highgpu-1g | 1x A100 (40GB) | 85GB | Large models (13B) |
| a2-highgpu-4g | 4x A100 (160GB) | 340GB | Very large models (70B) |

```bash
gcloud deployment-manager deployments create max-serve \
  --config max-nvidia-gcp.yaml
```

GCP notes: `onHostMaintenance: TERMINATE` required for GPU instances. Use Deep Learning VM images from `deeplearning-platform-release`. Preemptible instances reduce costs 60-80%.

---

## 7. Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `nvidia driver not found` | Missing NVIDIA container toolkit | Install nvidia-container-toolkit; use `--gpus all` |
| `model download failed` | No HuggingFace token or network issue | Set `HF_TOKEN` env var; check firewall/proxy settings |
| `port already in use` | Port 8000 occupied | Use `-p 8001:8000` to map to different host port |
| `out of memory` | Model too large for GPU | Use quantization or multi-GPU; check `--max-cache-memory` |
| `health check failed` | Server not ready | Increase readiness probe timeout; check startup logs |
| `CUDA version mismatch` | Container CUDA != host driver | Use matching container tag; check `nvidia-smi` compatibility |
| `metrics endpoint not responding` | Metrics disabled or wrong port | Set `MAX_SERVE_METRIC_LEVEL=BASIC`; check port 8001 |
| `Prometheus scrape timeout` | Too many detailed metrics | Use BASIC mode in production; reduce scrape frequency |
| `missing TTFT metric` | Streaming not enabled | TTFT only measured for streaming requests |
| `high preemption count` | KV cache pressure | Increase cache size; reduce max batch size |
| `worker crash not detected` | No health monitoring | Configure readiness/liveness probes; use heartbeat |

---

## Decision Guides

### Deployment Target

| Scenario | Approach |
|----------|----------|
| Development | Docker with local volumes |
| Single server production | Docker Compose |
| Kubernetes cluster | Deployment with PVC |
| AWS | CloudFormation with Deep Learning AMI |
| Azure | ARM template with NVIDIA AI Enterprise image |
| GCP | Deployment Manager with DL VM |
| AMD GPUs | Use AMD-specific containers and device mounts |

### Monitoring Configuration

| Scenario | Metric Level | Recording Method | Notes |
|----------|--------------|------------------|-------|
| Production | BASIC | PROCESS | Minimal overhead, isolated |
| Development | DETAILED | ASYNCIO | Full visibility |
| Debugging | DETAILED | ASYNCIO | Temporarily enable |
| Latency-critical | NONE or BASIC | PROCESS | Minimize overhead |
| Capacity planning | DETAILED | PROCESS | Monitor cache metrics |

### Inference Architecture

| Scenario | Architecture | When to Use |
|----------|--------------|-------------|
| Standard deployment | Unified | <4 GPUs, mixed workloads |
| High-scale production | Disaggregated | 4+ GPUs, long prompts |
| RAG workloads | Disaggregated | Long context, short generation |

---

## Quick Reference

**Containers:**
- NVIDIA: `modular/max-nvidia-full:latest`
- AMD: `modular/max-amd:latest`

**Paths:**
- HuggingFace cache: `/root/.cache/huggingface`
- MAX cache: `/opt/venv/share/max/.max_cache`

**Endpoints:**
- Health: `GET /health` (port 8000)
- Metrics: `GET /metrics` (port 8001)

**Monitoring:**
- BASIC metrics: <1% overhead, recommended for production
- DETAILED metrics: 5-15% overhead, debugging only
- PROCESS recording: Isolates collection from model worker
- Heartbeat: Enable for production to detect hung workers

**Environment:**
- `HF_HUB_ENABLE_HF_TRANSFER=1` — Faster downloads
- `MAX_SERVE_METRIC_LEVEL=BASIC` — Production metrics
- `MAX_SERVE_METRIC_RECORDING_METHOD=PROCESS` — Isolated collection
- `MAX_SERVE_USE_HEARTBEAT=true` — Worker health monitoring

---

## Best Practices

1. **Always mount volumes** — HuggingFace and MAX cache for fast startup
2. **Use BASIC metrics with PROCESS recording** in production
3. **Set MW_TIMEOUT** based on model size (~30s per 10B parameters)
4. **Enable heartbeat** for production reliability
5. **Use `--ipc=host`** for optimal GPU memory sharing (NVIDIA)
6. **Benchmark before deploying** — `max benchmark` for raw perf baseline
7. **Temporarily enable DETAILED** for debugging, then disable
8. **Monitor preemption counts and TTFT P99** as primary alerts
9. **Use `spawn` multiprocessing** (not `fork`) for GPU models
10. **Set `MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR=20`** when using DETAILED metrics

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|------------------|
| Container tags | `modular/max-nvidia-full:26.1` | `modular/max-nvidia-full:latest` |
| AMD container | `modular/max-amd:26.1` | `modular/max-amd:latest` |
| Cache path | `/opt/venv/share/max/.max_cache` | Same |
| Health endpoint | `/health` | Same |
| Metric levels | NONE, BASIC, DETAILED | Same |
| Metrics port | 8001 (default) | Same |
| Disaggregated | `--pipeline-role` flag | Same |
| CE watermark | `--kvcache-ce-watermark` (0.95) | Same |
| Batch size | Per-replica | Same |

**Stable (v26.1):**
```bash
docker run --gpus=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
    -p 8000:8000 \
    modular/max-nvidia-full:26.1 \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

**Nightly (v26.2+):**
```bash
docker run --gpus=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
    -p 8000:8000 \
    modular/max-nvidia-full:latest \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

Volume mount paths, Kubernetes patterns, and cloud provider templates are stable across versions.

---

## Related Patterns

- [`multigpu-scaling.md`](multigpu-scaling.md) — Multi-GPU deployment
- [`perf-inference.md`](perf-inference.md) — Performance optimization
- [`model-loading.md`](model-loading.md) — Supported models
- [`serve-configuration.md`](serve-configuration.md) — Environment configuration
- [`serve-kv-cache.md`](serve-kv-cache.md) — KV cache monitoring
- [`serve-api.md`](serve-api.md) — Error and preemption handling

---

## References

- [MAX Serve Docker](https://docs.modular.com/max/serve)
- [MAX Serve Observability](https://docs.modular.com/max/serve)
- [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/)
- [Azure GPU VMs](https://docs.microsoft.com/azure/virtual-machines/sizes-gpu)
- [GCP GPU VMs](https://cloud.google.com/compute/docs/gpus)
- Source: `max/serve/config.py`, `max/serve/telemetry/metrics.py`
