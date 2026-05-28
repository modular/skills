# Config and registration pitfalls

In scope: Phase 1 traps surfaced while reading `config.json`, choosing
dtypes, registering the architecture, and resolving MAX import paths.

Covered:

- `arch.py::name` must match `architectures[0]` exactly
- `default_encoding` should match the Hub checkpoint dtype
- Memory estimator reads HF `torch_dtype`, not the adapter's post-cast dtype
- `gelu_new` ≠ `gelu` ≠ `gelu_tanh`
- Import and config API traps

## `arch.py::name` must match `architectures[0]` exactly

MAX dispatches custom architectures by string-matching the registered
name against `config.json::architectures[0]`. Off by a character and the
registration is silently ignored, leaving MAX to try the default
auto-arch path (which probably won't work).

## `default_encoding` should match the Hub checkpoint dtype

The Hub config's `torch_dtype` field tells you what dtype the released
weights are in. Setting `default_encoding="float32"` for a BF16-only
checkpoint forces a conversion step that may not exist, or wastes memory
unnecessarily. Match what the model ships in.

## Memory estimator reads HF `torch_dtype`, not the adapter's post-cast dtype

MAX's `MemoryEstimator.estimate_memory_footprint` sizes the model from the HF
config's `torch_dtype`, *before* `weight_adapters.convert_safetensor_state_dict`
runs. If the released checkpoint ships FP32 tensors but your adapter casts them
to BF16 at load time, the estimator still thinks the model is FP32-sized — a 25
GiB BF16 model will pre-estimate at ~50 GiB and trip
`--device-memory-utilization 0.5` even though the real on-device footprint fits
comfortably.

**Workaround:** pass `--quantization-encoding bfloat16` explicitly *and*
bump `--device-memory-utilization` (0.7–0.9). The encoding flag tells
the estimator to compute against BF16; the utilization bump absorbs the
remaining slack. Common when the Hub ships FP32 tensors but the adapter
casts to BF16 at load time.

## `gelu_new` ≠ `gelu` ≠ `gelu_tanh`

The exact GELU function matters. Look up `config.hidden_act` in
`transformers/activations.py::ACT2FN` to see which one HF uses, and
match it in MAX. The three common variants:

- `gelu` — erf-based: `0.5 * x * (1 + erf(x / sqrt(2)))`.
- `gelu_new` / `gelu_tanh` — tanh approximation:
  `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`.
- `gelu_fast` — a different approximation; rare.

Visually similar outputs but cos_sim of ~0.95 at every MLP exit is the
signal.

## Import and config API traps

Do not maintain a parallel API cheat sheet. When imports or pydantic fields
fail, **copy from the donor arch you scaffolded** under
``modular/max/python/max/pipelines/architectures/<donor>/`` (``arch.py``,
``model.py``, ``model_config.py``) — that tree is the source of truth.

Common mistakes after older MAX examples or blog posts:

| Wrong                                                   | Right                                                                                                |
|---------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `from max.pipelines.core import PipelineTask`           | `from max.interfaces import PipelineTask` or `from max.pipelines.modeling.types import PipelineTask` |
| `from max.driver import Tensor`                         | `Buffer`, `Device` from `max.driver`                                                                 |
| `pipeline_config.model_config`                          | `pipeline_config.model`                                                                              |
| `pipeline_config.max_length`                            | `pipeline_config.model.max_length`                                                                   |
| `pipeline_config.max_batch_size`                        | `pipeline_config.runtime.max_batch_size`                                                             |
| `KVCacheParams(..., cache_strategy=..., n_devices=...)` | Removed in current MAX — use `kv_cache_config.to_params(...)` only                                   |
| `RMSNorm(..., devices=...)`                             | No `devices` on `RMSNorm` (unlike `LayerNorm`, which takes a device list)                            |
| `RotaryEmbedding(..., device=...)`                      | No `device` on `RotaryEmbedding.__init__`                                                            |

`SupportedArchitecture` uses plain strings for `default_encoding`,
`supported_encodings`, and `rope_type` — not enums.

**Weights on disk:** only `WeightsFormat.safetensors` and `WeightsFormat.gguf`
exist (`max/graph/weights/format.py`). No `.bin` / PyTorch shard loader. See
weights preflight in [serve-and-iterate.md](serve-and-iterate.md).

**Encoding vs device** (`max/pipelines/lib/config/config_enums.py`):

| Encoding                                   | Devices      |
|--------------------------------------------|--------------|
| `float32`, `bfloat16`                      | `cpu`, `gpu` |
| `float8_e4m3fn`, `float4_e2m1fnx2`, `gptq` | `gpu` only   |
| `q4_k`, `q4_0`, `q6_k`                     | `cpu` only   |

`DeviceSpec` is only `"cpu"` or `"gpu"` (Metal uses `gpu` on Apple Silicon).
