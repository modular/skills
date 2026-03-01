---
title: Model Loading and Configuration
description: Supported model architectures, quantization formats, and HuggingFace token usage
impact: HIGH
category: model
tags: [architectures, huggingface, quantization, gguf, gptq]
error_patterns:
  - "model not found"
  - "architecture"
  - "unsupported model"
  - "HuggingFace"
  - "token"
  - "GGUF"
  - "loading failed"
scenarios:
  - "Load model from HuggingFace"
  - "Configure HF token for gated models"
  - "Check if model is supported"
  - "Load GGUF quantized model"
  - "Fix model loading errors"
consolidates:
  - model-architectures.md
  - model-hf-token.md
---

# Model Loading and Configuration

**Category:** model | **Impact:** HIGH

Comprehensive patterns for model loading including supported architectures, quantization formats, and HuggingFace authentication for gated models.

> **CLI Flag:** Use `--model` (not `--model-path`) to specify the model. The legacy `--model-path` alias still works.

---

## Core Concepts

### Supported Model Architectures

MAX supports 35+ optimized model architectures natively with MAX Graph.

**Text Generation (Causal LM):**

| Architecture | Example Models | Encodings | Multi-GPU |
|--------------|---------------|-----------|-----------|
| `LlamaForCausalLM` | Llama 3.1, Llama 3.2, DeepSeek-R1-Distill, Llama-Guard-3 | q4_k, q4_0, q6_k, float32, float16, bfloat16, float8, float4 | Yes |
| `MistralForCausalLM` | Mistral-Nemo-Instruct-2407 | bfloat16, float16 | Yes |
| `Mistral3ForConditionalGeneration` | Mistral-Small-3.1-24B-Instruct | bfloat16 | Yes |
| `Qwen2ForCausalLM` | Qwen2.5-7B-Instruct, QwQ-32B | float32, bfloat16 | Yes |
| `Qwen3ForCausalLM` | Qwen3-8B, Qwen3-30B-A3B (dense + MoE) | bfloat16, float32 | Yes |
| `Phi3ForCausalLM` | Microsoft Phi-4, Phi-3.5-mini-instruct | float32, bfloat16 | No |
| `GraniteForCausalLM` | IBM Granite 3.1-8B-instruct | float32, bfloat16 | No |
| `Gemma3ForCausalLM` | Gemma 3 1B (text-only, <4B models) | bfloat16 | Yes |
| `OlmoForCausalLM` | OLMo-1B-hf | float32, bfloat16 | No |
| `Olmo2ForCausalLM` | OLMo-2-0425-1B, OLMo-2-1124-7B/13B, OLMo-2-0325-32B | bfloat16, float32 | No |
| `ExaoneForCausalLM` | EXAONE-3.5-2.4B/7.8B/32B-Instruct | q4_k, q6_k, float32, bfloat16 | No |
| `GptOssForCausalLM` | gpt-oss-20b | bfloat16 | Yes |
| `Llama4ForConditionalGeneration` | Llama-4-Scout-17B-16E, Llama-4-Maverick-17B-128E | bfloat16 | Yes |
| `LlamaForCausalLMEagle` | sglang-EAGLE-LLaMA3-Instruct-8B (speculative decoding) | bfloat16, float32 | No |

**Mixture of Experts (MoE):**

| Architecture | Example Models | Encodings | Multi-GPU |
|--------------|---------------|-----------|-----------|
| `DeepseekV2ForCausalLM` | DeepSeek-V2-Lite-Chat | bfloat16 | Yes |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | bfloat16, float8, float4 | Yes |
| `DeepseekV32ForCausalLM` | DeepSeek-V3.2, DeepSeek-V3.2-Exp | float8 | Yes |
| `DeepseekV3ForCausalLMNextN` | DeepSeek-V3-NextN (speculative) | bfloat16, float8 | Yes |

**Vision-Language Models:**

| Architecture | Example Models | Encodings | Multi-GPU |
|--------------|---------------|-----------|-----------|
| `LlavaForConditionalGeneration` | Pixtral-12B | bfloat16 | No |
| `Idefics3ForConditionalGeneration` | Idefics3-8B-Llama3 | bfloat16 | No |
| `Gemma3ForConditionalGeneration` | Gemma 3 4B/12B/27B (multimodal, >=4B) | bfloat16, float8 | Yes |
| `InternVLChatModel` | InternVL3-8B-Instruct | bfloat16 | Yes |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL-3B/7B-Instruct | float32, bfloat16, float8 | Yes |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL-2B/4B-Instruct | float32, bfloat16, float8 | Yes |
| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL-30B-A3B-Instruct (MoE) | float32, bfloat16, float8 | Yes |

**Embedding Models:**

| Architecture | Example Models | Encodings |
|--------------|---------------|-----------|
| `BertModel` | all-MiniLM-L6-v2, all-MiniLM-L12-v2 | float32, bfloat16 |
| `MPNetForMaskedLM` | all-mpnet-base-v2 | float32, bfloat16 |
| `Qwen3Embedding` | Qwen3-Embedding-0.6B/4B/8B | float32, bfloat16 |

**Image Generation:**

| Architecture | Example Models | Encodings |
|--------------|---------------|-----------|
| `FluxPipeline` | FLUX.1-dev, FLUX.1-schnell | bfloat16 |

**Other (Encoder-Decoder / Audio):**

| Architecture | Example Models | Notes |
|--------------|---------------|-------|
| `T5` | T5 encoder-decoder | Encoder-decoder model |
| `Whisper` | Whisper ASR | Speech-to-text |
| `CLIP` | CLIP vision-text encoder | Used as component in other pipelines |
| `AutoencoderKL` | VAE models | Used in image generation pipelines |

### HuggingFace Token Authentication

Set `HF_TOKEN` environment variable for gated HuggingFace models.

**Pattern:**

```bash
# Set token first
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Or inline
HF_TOKEN=hf_xxx max serve --model meta-llama/Llama-3.1-8B-Instruct
```

**Docker:**

```bash
docker run --gpus=1 \
    -e HF_TOKEN=$HF_TOKEN \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    modular/max-nvidia-full:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

**Don't:**

```bash
# Fails for gated models
max serve --model meta-llama/Llama-3.1-8B-Instruct
```

**Gated Models Include:**

- `meta-llama/*` (Llama 3.x)
- `mistralai/*` (some models)
- `google/gemma-*` (some models)

---

## Common Patterns

### Quantization Formats

**GPU Quantization:**

| Encoding | Bits | Device | Use Case |
|----------|------|--------|----------|
| `bfloat16` | 16 | GPU | Default, best quality |
| `float8_e4m3fn` | 8 | GPU | Reduced memory |
| `float4_e2m1fnx2` | 4 | GPU | NVFP4, maximum compression |
| `gptq` | 4 | GPU | Pre-quantized models |

**CPU Quantization (GGUF):**

| Encoding | Bits | Quality | Use Case |
|----------|------|---------|----------|
| `q6_k` | 6 | High | Best CPU quality |
| `q4_k` | 4 | Medium | Balanced |
| `q4_0` | 4 | Lower | Maximum compression |

**Pattern - GPU with bfloat16:**

```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --devices gpu:0 \
  --quantization-encoding bfloat16
```

**Pattern - CPU with GGUF:**

```bash
max serve --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --quantization-encoding q4_k
```

**Pattern - GPTQ quantized model:**

```bash
max serve --model TheBloke/Llama-2-13B-GPTQ \
  --devices gpu:0 \
  --quantization-encoding gptq
```

---

### Model Loading Examples

**Llama 3.1/3.3:**

```bash
# 8B model
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --devices gpu:0

# 70B model (multi-GPU)
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3
```

**Mistral:**

```bash
max serve --model mistralai/Mistral-7B-Instruct-v0.3 \
  --devices gpu:0
```

**Qwen 2.5:**

```bash
max serve --model Qwen/Qwen2.5-7B-Instruct \
  --devices gpu:0
```

**Phi-4:**

```bash
max serve --model microsoft/phi-4 \
  --devices gpu:0
```

**Gemma 3 (recommended quickstart model):**

```bash
max serve --model google/gemma-3-27b-it \
  --devices gpu:0
```

---

### Vision-Language Models

> **Note:** Some vision models require `--trust-remote-code` to load custom model code from HuggingFace.

**Pixtral:**

```bash
max serve --model mistral-community/pixtral-12b \
  --devices gpu:0
```

**Gemma 3 Multimodal (4B+):**

```bash
max serve --model google/gemma-3-12b-it \
  --devices gpu:0 --trust-remote-code
```

**InternVL:**

```bash
max serve --model OpenGVLab/InternVL3-8B-Instruct \
  --devices gpu:0
```

**Qwen2.5-VL:**

```bash
max serve --model Qwen/Qwen2.5-VL-7B-Instruct \
  --devices gpu:0
```

**Qwen3-VL:**

```bash
max serve --model Qwen/Qwen3-VL-4B \
  --devices gpu:0
```

---

### Embedding Models

**BERT-based (sentence-transformers):**

```bash
max serve --model sentence-transformers/all-MiniLM-L6-v2 \
  --task embeddings
```

**MPNet:**

```bash
max serve --model sentence-transformers/all-mpnet-base-v2 \
  --task embeddings
```

**Qwen3 Embedding:**

```bash
max serve --model Qwen/Qwen3-Embedding-0.6B \
  --task embeddings
```

---

### Image Generation Models

**FLUX.1:**

```bash
max serve --model black-forest-labs/FLUX.1-schnell \
  --devices gpu:0
```

---

## Decision Guide

| Model Size | Recommended Quantization | GPUs Needed |
|------------|--------------------------|-------------|
| 7-8B | bfloat16 or q4_k (CPU) | 1 |
| 13B | bfloat16 or GPTQ | 1 (large GPU) |
| 34B | bfloat16 | 2 |
| 70B+ | bfloat16 | 4 |

| Device | Recommended Formats |
|--------|---------------------|
| NVIDIA GPU | bfloat16, float8, float4 (NVFP4 on Blackwell) |
| AMD GPU | bfloat16 |
| Apple GPU (Metal) | q4_k, q4_0, q6_k (GGUF only) |
| CPU | q4_k, q6_k (GGUF) |

---

## Quick Reference

- **Gated models**: Require `HF_TOKEN` environment variable
- **GPU formats**: bfloat16, float8_e4m3fn, float4_e2m1fnx2 (NVFP4)
- **CPU formats**: q4_k, q6_k, q4_0 (GGUF)
- **Vision-language**: Pixtral, Idefics3, InternVL, Gemma 3 multimodal, Qwen2.5-VL, Qwen3-VL
- **MoE models**: DeepSeek V2/V3/V3.2, Qwen3-MoE, Qwen3-VL-MoE, Llama 4
- **Embeddings**: BERT (MiniLM), MPNet, Qwen3-Embedding
- **Image generation**: FLUX.1-dev, FLUX.1-schnell
- **Speculative decoding**: EAGLE-LLaMA3, DeepSeek-V3-NextN

---

## Weights Format Support

| Format | Extension | Use Case |
|--------|-----------|----------|
| SafeTensors | `.safetensors` | Default, secure |
| GGUF | `.gguf` | CPU quantized |
| PyTorch | `.bin`, `.pt` | Legacy |

---

## Model Memory Requirements

Approximate VRAM requirements for bfloat16:

| Model | Parameters | VRAM (bf16) |
|-------|------------|-------------|
| Llama 3.1 8B | 8B | ~16GB |
| Mistral 7B | 7B | ~14GB |
| Gemma 3 27B | 27B | ~54GB |
| DeepSeek-V3 | 685B (37B active) | ~1.3TB (multi-GPU) |
| Llama 3.3 70B | 70B | ~140GB |
| Qwen 2.5 72B | 72B | ~144GB |

**Memory reduction with quantization:**

- float8: ~50% of bfloat16
- GPTQ (4-bit): ~25% of bfloat16
- GGUF q4_k: ~25% of bfloat16

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `model not found` | Wrong HuggingFace model ID | Verify model exists; check for typos; use `huggingface-cli download` to test |
| `authentication required` | Gated model without token | Set `HF_TOKEN` environment variable or `huggingface-cli login` |
| `architecture not supported` | Model type not in MAX | Check supported architectures list; use GGUF for unsupported models |
| `out of memory loading weights` | Model too large for GPU | Use quantization (float8/GPTQ); enable multi-GPU with `--devices` |
| `GGUF format error` | Corrupted or incompatible GGUF | Re-download file; verify quantization type is supported |
| `tokenizer not found` | Missing tokenizer files | Ensure `tokenizer.json` or `tokenizer_config.json` exists |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly

The architecture list above reflects nightly. Stable versions may have fewer architectures. Check the [MAX changelog](https://docs.modular.com/max/changelog/) for details on which models are available in each release.

**Key differences:**

- New architectures (DeepSeek V3.2, Qwen3-VL, Llama 4, OLMo2) are typically nightly-first
- NVFP4 (`float4_e2m1fnx2`) is available on Blackwell GPUs in nightly
- Apple GPU (Metal) support for GGUF models is evolving - use nightly for best compatibility
- GGUF and SafeTensors weight formats are stable across versions
- HuggingFace token authentication is stable (`HF_TOKEN` environment variable)

---

## Related Patterns

- [`deploy-production.md`](deploy-production.md) — Production deployment
- [`multigpu-scaling.md`](multigpu-scaling.md) — Multi-GPU for large models
- [`engine-operations.md`](engine-operations.md) — Custom architecture registration

---

## References

- [MAX Supported Models](https://docs.modular.com/max/model-formats/)
- [HuggingFace Token](https://huggingface.co/settings/tokens)
- [GGUF Format](https://github.com/ggerganov/ggml)
