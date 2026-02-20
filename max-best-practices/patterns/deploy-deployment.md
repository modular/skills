---
title: Production Deployment
description: Container deployment, volume configuration, benchmarking, and cloud provider templates
impact: HIGH
category: deploy
tags: [container, docker, kubernetes, aws, azure, gcp, benchmark]
error_patterns:
  - "container"
  - "docker"
  - "kubernetes"
  - "deployment failed"
  - "volume"
  - "mount"
  - "pod"
  - "image"
scenarios:
  - "Deploy MAX Serve in production"
  - "Configure Docker container"
  - "Set up Kubernetes deployment"
  - "Deploy on AWS/Azure/GCP"
  - "Benchmark production deployment"
  - "Configure volume mounts"
consolidates:
  - deploy-container.md
  - deploy-container-volumes.md
  - deploy-benchmark.md
  - deploy-aws-cloudformation.md
  - deploy-azure-arm.md
  - deploy-gcp-deployment-manager.md
  - deploy-kubernetes.md
---

# Production Deployment

**Category:** deploy | **Impact:** HIGH

Comprehensive patterns for production deployment including official containers, volume configuration for fast startup, benchmarking, Kubernetes deployment, and cloud provider templates for AWS, Azure, and GCP.

---

## Core Concepts

### Official MAX Containers

Use Modular's official containers for production deployments.

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

**Available Containers:**
| Container | Target |
|-----------|--------|
| `modular/max-nvidia-full` | NVIDIA GPU |
| `modular/max-amd` | AMD GPU |

### Volume Configuration for Fast Startup

Mount cache volumes to reduce container startup from 30+ minutes to seconds.

**Required Volume Mounts:**

| Volume | Container Path | Purpose |
|--------|---------------|---------|
| HuggingFace cache | `/root/.cache/huggingface` | Downloaded model files |
| MAX cache | `/opt/venv/share/max/.max_cache` | Compiled model artifacts |

**Pattern:**
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

---

## Common Patterns

### Complete NVIDIA Container Setup

**When:** Production deployment on NVIDIA GPUs.

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

**Key flags:**
- `--ipc=host`: Optimal GPU memory sharing
- `HF_HUB_ENABLE_HF_TRANSFER=1`: Faster downloads

---

### Complete AMD Container Setup

**When:** Production deployment on AMD MI300X GPUs.

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

---

### Kubernetes Deployment

**When:** Production Kubernetes deployments with health checks and auto-scaling.

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

**Note:** Use `/health` endpoint for probes, not inference endpoints.

---

### Benchmarking

**When:** Before production deployment, performance regression testing, hardware comparison.

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

# Benchmark with specific batch size and sequence lengths
max benchmark --model meta-llama/Llama-3.1-8B-Instruct \
  --max-batch-size 32 \
  --devices gpu:0

# Multi-GPU benchmark
max benchmark --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3
```

**Metrics Collected:**
| Metric | Description |
|--------|-------------|
| TTFT | Time To First Token (latency to first generated token) |
| ITL | Inter-Token Latency (average time between tokens) |
| Throughput | Tokens per second (generation throughput) |
| GPU utilization | Hardware usage (memory and compute) |

**Benchmark vs Serve:** Use `max benchmark` to measure raw model performance without HTTP overhead. Use `max serve` + a load testing tool (e.g., `wrk`, `hey`) for end-to-end latency including API overhead.

---

## Cloud Provider Deployments

### AWS CloudFormation

**Instance Selection:**

| Instance | GPU | Memory | Use Case |
|----------|-----|--------|----------|
| g5.xlarge | 1x A10G (24GB) | 16GB | Small models (7-8B) |
| g5.4xlarge | 1x A10G (24GB) | 64GB | Medium models (8-13B) |
| g5.12xlarge | 4x A10G (96GB) | 192GB | Large models (70B) |
| p4d.24xlarge | 8x A100 (320GB) | 1.1TB | Very large models |

**Deploy Command:**
```bash
aws cloudformation create-stack \
  --stack-name max-serve \
  --template-body file://max-nvidia-aws.yaml \
  --parameters ParameterKey=HuggingFaceHubToken,ParameterValue=$HF_TOKEN
```

**Template Pattern:**
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

**Best Practices:**
- Use Deep Learning AMIs (pre-installed NVIDIA drivers)
- Store HF_TOKEN in Secrets Manager
- Configure 100GB+ gp3 storage for model caching
- Use `--ipc=host` for optimal GPU memory sharing

---

### Azure ARM Templates

**Instance Selection:**

| VM Size | GPU | Memory | Use Case |
|---------|-----|--------|----------|
| Standard_NC4as_T4_v3 | 1x T4 (16GB) | 28GB | Small models |
| Standard_NV36ads_A10_v5 | 1x A10 (24GB) | 440GB | Medium models (8-13B) |
| Standard_NC24ads_A100_v4 | 1x A100 (80GB) | 220GB | Large models (70B) |
| Standard_ND96isr_MI300X_v5 | 8x MI300X | 1.8TB | AMD, very large models |

**Deploy Command:**
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

**Azure-Specific Considerations:**

| GPU Type | Image | Notes |
|----------|-------|-------|
| NVIDIA | NVIDIA AI Enterprise | Marketplace image |
| AMD MI300X | Ubuntu HPC + ROCm | Larger disk (256GB) |

---

### GCP Deployment Manager

**Instance Selection:**

| Machine Type | GPU | Memory | Use Case |
|--------------|-----|--------|----------|
| n1-standard-8 + T4 | 1x T4 (16GB) | 30GB | Small models |
| n1-standard-16 + L4 | 1x L4 (24GB) | 60GB | Medium models (8B) |
| a2-highgpu-1g | 1x A100 (40GB) | 85GB | Large models (13B) |
| a2-highgpu-4g | 4x A100 (160GB) | 340GB | Very large models (70B) |

**Deploy Command:**
```bash
gcloud deployment-manager deployments create max-serve \
  --config max-nvidia-gcp.yaml
```

**GCP-Specific Requirements:**
- `onHostMaintenance: TERMINATE` required for GPU instances
- Use Deep Learning VM images from `deeplearning-platform-release`
- Preemptible instances reduce costs 60-80%

---

## Decision Guide

| Scenario | Approach |
|----------|----------|
| Development | Docker with local volumes |
| Single server production | Docker Compose |
| Kubernetes cluster | Deployment with PVC |
| AWS | CloudFormation with Deep Learning AMI |
| Azure | ARM template with NVIDIA AI Enterprise image |
| GCP | Deployment Manager with DL VM |
| AMD GPUs | Use AMD-specific containers and device mounts |

---

## Quick Reference

- **NVIDIA container**: `modular/max-nvidia-full:latest`
- **AMD container**: `modular/max-amd:latest`
- **HuggingFace cache**: `/root/.cache/huggingface`
- **MAX cache**: `/opt/venv/share/max/.max_cache`
- **Health endpoint**: `/health`
- **Fast downloads**: `HF_HUB_ENABLE_HF_TRANSFER=1`

---

## Pre-warming Strategy

Pre-download models before container deployment:

```bash
# Pre-download model to host
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct

# Then start container (skips download)
docker run ...
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `nvidia driver not found` | Missing NVIDIA container toolkit | Install nvidia-container-toolkit; use `--gpus all` |
| `model download failed` | No HuggingFace token or network issue | Set `HF_TOKEN` env var; check firewall/proxy settings |
| `port already in use` | Port 8000 occupied | Use `-p 8001:8000` to map to different host port |
| `out of memory` | Model too large for GPU | Use quantization or multi-GPU; check `--max-cache-memory` |
| `health check failed` | Server not ready | Increase readiness probe timeout; check startup logs |
| `CUDA version mismatch` | Container CUDA != host driver | Use matching container tag; check `nvidia-smi` compatibility |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|--------------------|
| **Container tags** | `modular/max-nvidia-full:26.1` | `modular/max-nvidia-full:latest` or `26.2.0` |
| **AMD container** | `modular/max-amd:26.1` | `modular/max-amd:latest` |
| **Cache path** | `/opt/venv/share/max/.max_cache` | `/opt/venv/share/max/.max_cache` (unchanged) |
| **Health endpoint** | `/health` | `/health` (unchanged) |

**Stable (v26.1):**
```bash
# Use versioned container tags for reproducibility
docker run --gpus=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
    -p 8000:8000 \
    modular/max-nvidia-full:26.1 \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

**Nightly (v26.2+):**
```bash
# Latest container includes newest features
docker run --gpus=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/max_cache:/opt/venv/share/max/.max_cache \
    -p 8000:8000 \
    modular/max-nvidia-full:latest \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

**Notes:**
- Container image paths are stable; version tags change with releases
- Volume mount paths for HuggingFace and MAX cache are stable
- Kubernetes deployment patterns are stable across versions
- Cloud provider templates (AWS CloudFormation, Azure ARM, GCP) are stable
- `--ipc=host` flag for optimal GPU memory sharing is stable

---

## Related Patterns

- [`multigpu-scaling.md`](multigpu-scaling.md) — Multi-GPU deployment
- [`perf-inference.md`](perf-inference.md) — Performance optimization
- [`model-loading.md`](model-loading.md) — Supported models

---

## References

- [MAX Serve Docker](https://docs.modular.com/max/serve)
- [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/)
- [Azure GPU VMs](https://docs.microsoft.com/azure/virtual-machines/sizes-gpu)
- [GCP GPU VMs](https://cloud.google.com/compute/docs/gpus)
