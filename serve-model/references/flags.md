# `max serve` flags: what to set, and when NOT to

The default command (`max serve --model <repo>`) auto-detects device, dtype, and
sequence length. Every flag below is something you add for a *specific* reason.
The most common failure mode is over-specifying: a flag that fights
auto-detection. When in doubt, leave it off and let MAX decide.

## Contents

- [Choosing from the checkpoint (config.json)](#choosing-from-the-checkpoint)
- [Device and memory](#device-and-memory)
- [Precision / quantization](#precision--quantization)
- [Sequence length and batching](#sequence-length-and-batching)
- [Task selection](#task-selection)
- [Trust remote code](#trust-remote-code)
- [Custom architectures](#custom-architectures)
- [Multi-GPU parallelism](#multi-gpu-parallelism)
- [Sliding window / RoPE](#sliding-window--rope)
- [Compile time](#compile-time)
- [Serving-feature flags](#serving-feature-flags)

## Choosing from the checkpoint

When a flag choice is load-bearing, read the checkpoint's `config.json` (on the
Hub, or `~/.cache/huggingface/...` locally) and map fields to flags:

| `config.json` field | Tells you | Flag it informs |
|---|---|---|
| `architectures[0]` | Which arch to match | must equal `SupportedArchitecture.name` |
| `torch_dtype` (`bfloat16`/`float16`/`float32`) | Native precision | `--quantization-encoding` |
| `quantization_config` | Pre-quantized (gptq/awq/fp8) | `--quantization-encoding` |
| `max_position_embeddings` / `n_positions` | Max context | ceiling for `--max-length` |
| `sliding_window` | Windowed attention | usually auto; `--sliding-window` to override |
| `rope_scaling.rope_type` | RoPE variant | `--rope-type` (GGUF only) |

Also check the arch package's `SupportedArchitecture.supported_encodings`: the
`--quantization-encoding` you pass must be in that set, or serve rejects it.

## Device and memory

- `--devices`: the first-class device selector. `cpu`, `gpu`, `gpu:all`,
  `gpu:0`, or a list `gpu:0,1,2,3`. Omit to use the model/config default.
  **Do not** also set `CUDA_VISIBLE_DEVICES`; the two are translated
  independently and stacking them routes to the wrong device.
- `--device-memory-utilization <0..1>`: fraction of free device memory the
  process may consume; the remainder holds the KV cache
  (`kv_workspace = free_mem * util - weights`). Raise toward `0.9`–`0.95` on a
  busy GPU when you need more KV room; lower it if you're sharing the GPU.

## Precision / quantization

`--quantization-encoding` options:
`float32 | float16 | bfloat16 | q4_k | q4_0 | q6_k | float8_e4m3fn |
float4_e2m1fnx2 | gptq`.

- **Leave unset** when the repo ships one format; MAX auto-detects it.
- **Set it** when the repo ships several (choose one) or auto-detect picks
  wrong. Match `torch_dtype`/`quantization_config` from `config.json`, and make
  sure the value is in the arch's `supported_encodings`.
- CPU serving generally needs `float32` (or a model with a CPU-feasible
  footprint); GPU commonly `bfloat16`.
- `--kv-cache-format {float32|bfloat16|float8_e4m3fn}`: overrides only the KV
  cache dtype (memory vs accuracy), independent of weight encoding.

## Sequence length and batching

- `--max-length`: max sequence length. Defaults to
  `max_position_embeddings`; MAX may clamp it down to fit memory. Set a smaller
  value to save KV memory when you don't need the full context.
- `--max-batch-size`: max concurrent requests in a batch. Auto when unset. For
  a real server set it to your expected concurrency; for a single-stream
  correctness/latency check, `1` keeps things simple.
- `--max-batch-input-tokens`, `--max-batch-total-tokens`: target/ceiling token
  budgets per batch; only needed for throughput tuning or chunked prefill.

## Task selection

- `--task`: `text_generation` (default for LLMs), `embeddings_generation`,
  `pixel_generation`. Only needed to disambiguate an architecture registered
  under one name for multiple tasks, or to serve embeddings/image endpoints.
  The OpenAI routes only function with a compatible task (for example `/v1/embeddings`
  needs `embeddings_generation`).

## Trust remote code

- `--trust-remote-code`: required when the HF repo ships custom Python
  (`modeling_*.py`, `configuration_*.py`, custom tokenizer) that the loader
  executes. If you hit a `trust_remote_code` error at load, add it. Only use it
  with repos you trust; it runs arbitrary code from the repo.

## Custom architectures

- `--custom-architectures`: repeatable. Either a package directory name (when
  you run from its parent) or `IMPORT_PATH:MODULE_NAME` (robust, path-anywhere).
  The package's `__init__.py` must expose `ARCHITECTURES = [<SupportedArch>...]`.
  A custom `name` that matches a built-in takes precedence over the built-in.

## Multi-GPU parallelism

Only for models too big for one GPU, or when you're deliberately scaling. For a
single-stream latency test, **fewer GPUs is usually faster**; tensor-parallel
comms can dominate and hurt batch-1 latency. Reach for these on real
multi-GPU serving:

- `--data-parallel-degree`: replicate the model across devices.
- `--ep-size`: expert-parallelism size for MoE (1, or total GPU count).
- `--pipeline-role {prefill_and_decode|prefill_only|decode_only}`:
  disaggregated prefill/decode across processes.

## Sliding window / RoPE

- `--sliding-window <N>`: force a sliding-window causal mask of N tokens.
  Defaults to the config's `sliding_window`, or full causal attention. Override
  only if the config is missing/wrong.
- `--rope-type {none|normal|neox|longrope|yarn}`: force a RoPE variant. Only
  matters for **GGUF** weights; safetensors models carry this in config.

## Compile time

- `--use-subgraphs / --no-use-subgraphs`: subgraphs (default **on**) cut
  compile time dramatically for large models with many identical blocks. Leave
  it on. Turning it off (a leftover from debugging) can blow up compile time on
  a deep model.
- `--device-graph-capture / --no-device-graph-capture`: capture+replay for
  execution; auto-enabled for some architectures.

## Serving-feature flags

Layer these onto any working serve; none are required to get started:

- `--enable-prefix-caching`: reuse KV across requests sharing a prefix; cuts
  TTFT on repeated prompts.
- `--enable-chunked-prefill`: split long prefills into chunks
  (`--max-batch-input-tokens`).
- `--enable-structured-output`: accept a JSON schema in `response_format`.
- `--enable-lora` + `--lora-paths`: serve LoRA adapters over a base model.
- `--temperature`, `--top-k`: server-level sampling defaults for requests that
  don't set their own.
- `--served-model-name`: override the client-facing model name (what clients
  put in the request `model` field); defaults to `--model`.
- `--port`: HTTP port (default `8000`).

## Two serve-time gotchas that bite everyone

Not model-derived (they come from how MAX serves), but they cause confusing
"server looked fine, then failed" symptoms:

- **`--no-enable-overlap-scheduler --force` for logprobs.** MAX auto-enables the
  overlap scheduler for single-step decode, and that path can't serve `logprobs`
  requests (parity/verification tools send these). Disable it if you need
  logprobs; the `--force` bypasses the auto-enable heuristic. Plain chat serving
  doesn't need this.
- **Never serve on `--port 8001`.** The metrics ASGI app defaults to 8001 and
  binds before the main API server. If the main server also lands on 8001, the
  log prints `Server ready` then immediately `Shutting down workers…` with no
  error and every request fails with a JSON decode error. Use 8000 or another
  port; when running multiple servers, also move the metrics port via the
  `MAX_SERVE_METRICS_ENDPOINT_PORT` environment variable.
