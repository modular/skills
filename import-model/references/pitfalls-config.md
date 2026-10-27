# Config and registration pitfalls

This page covers Phase 1 traps that surface while you read `config.json`,
choose dtypes, register the architecture, and resolve MAX import paths. It
describes these pitfalls:

- `arch.py::name` must match `architectures[0]` exactly
- `default_encoding` should match the Hub checkpoint dtype
- Memory estimator sizes weights from the checkpoint files
- `gelu` and `gelu_new` are different GELUs
- Import and config API traps

## `arch.py::name` must match `architectures[0]` exactly

MAX dispatches custom architectures by string-matching the registered name
against `config.json::architectures[0]`. If the names differ by one character,
MAX doesn't select your registration. It uses a built-in architecture registered
under that name if one exists, and otherwise fails with
`ValueError: No architecture found for <name>`.

Donors named `<arch>_modulev3` register with a `_ModuleV3` suffix
(`LlamaForCausalLM_ModuleV3`) because they sit next to an older
implementation of the same model. Your port registers the plain
`architectures[0]` value (`olmo3` registers `Olmo3ForCausalLM`), so don't
copy the donor's `name=`.

## `default_encoding` should match the Hub checkpoint dtype

The Hub config's `torch_dtype` field tells you what dtype the released
weights are in. Setting `default_encoding="float32"` for a BF16-only
checkpoint forces a conversion step that may not exist, or wastes memory.
Match what the model ships in.

## Memory estimator sizes weights from the checkpoint files

MAX's `MemoryEstimator` sizes the weights from the checkpoint's file sizes on
disk (`MAXModelConfig.weights_size()`), *before*
`weight_adapters.convert_safetensor_state_dict` runs. If the released
checkpoint ships FP32 tensors but your adapter casts them to BF16 at load
time, the estimator still counts the FP32 bytes. A 25 GiB BF16 model
pre-estimates at ~50 GiB and trips `--device-memory-utilization 0.5`, even
though the BF16 weights fit on the device.

**Workaround:** raise `--device-memory-utilization` (0.7–0.9) so the
FP32-sized estimate fits. The fraction bounds the weights and the KV cache
together. Raise it when the weights estimate doesn't fit, and lower it when
the server fails to allocate the KV cache at startup.

## `gelu` and `gelu_new` are different GELUs

`config.hidden_act` names the GELU variant HF uses. Look it up in
`transformers/activations.py::ACT2FN`, and call the matching MAX function:

| HF `hidden_act`                              | Formula                                                    | MAX                              |
|----------------------------------------------|------------------------------------------------------------|----------------------------------|
| `gelu`                                       | `0.5 * x * (1 + erf(x / sqrt(2)))`                         | `F.gelu(x)`                      |
| `gelu_new`, `gelu_pytorch_tanh`, `gelu_fast` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))` | `F.gelu(x, approximate="tanh")`  |
| `quick_gelu`                                 | `x * sigmoid(1.702 * x)`                                   | `F.gelu(x, approximate="quick")` |

The erf and tanh variants differ by less than 0.001 per element, so a
single-layer comparison can pass with the wrong one. Match the name from
`config.hidden_act`.

## Import and config API traps

Do not maintain a parallel API cheat sheet. When imports or pydantic fields
fail, copy from the donor arch you scaffolded under
``max/pipelines/architectures/<donor>/`` (``arch.py``,
``model.py``, ``model_config.py``). That tree is the source of truth.

These mistakes are common in code copied from older MAX examples or blog
posts:

| Wrong                                                   | Right                                                                                             |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `from max.pipelines.core import PipelineTask`           | `from max.pipelines.modeling.types import PipelineTask`                                           |
| `from max.driver import Tensor`                         | `Buffer`, `Device` from `max.driver`                                                              |
| `pipeline_config.model_config`                          | `pipeline_config.model`                                                                           |
| `pipeline_config.max_length`                            | `pipeline_config.model.max_length`                                                                |
| `pipeline_config.max_batch_size`                        | `pipeline_config.runtime.max_batch_size`                                                          |
| `KVCacheParams(..., cache_strategy=..., n_devices=...)` | Removed in current MAX. Use `kv_cache_config.to_params(...)` only                                 |
| `from max.nn.legacy import Linear, RMSNorm` in the port        | `from max.nn import Linear`, `from max.nn.norm import RMSNorm`          |
| `from max.nn.legacy import PagedCacheValues`                   | `from max.nn.common_layers.kv_cache import PagedCacheValues` (has `from_upstream()`) |
| `Linear(in_dim, out_dim, dtype=..., device=...)`        | `Linear(in_dim, out_dim, bias=False)`. `model.py` places the module with `.to()`                  |
| `ModuleList(*layers)`                                   | `ModuleList(layers)`. It takes one iterable and raises `TypeError` on several arguments           |

`SupportedArchitecture` takes plain strings for `default_encoding` and
`supported_encodings` (`"bfloat16"`).

**Weights on disk:** only `WeightsFormat.safetensors` and `WeightsFormat.gguf`
exist (`max/graph/weights/format.py`). No `.bin` / PyTorch shard loader. See
weights preflight in [serve-and-iterate.md](serve-and-iterate.md).

**Encoding vs device** (`max/pipelines/modeling/config_enums.py`):

| Encoding                                   | Devices      |
|--------------------------------------------|--------------|
| `float32`                                  | `cpu`, `gpu` |
| `float16`, `bfloat16`                      | `gpu` only   |
| `float8_e4m3fn`, `float4_e2m1fnx2`, `float6_e2m3fn`, `gptq` | `gpu` only |
| `q4_k`, `q4_0`, `q6_k`                     | `cpu` only   |

`DeviceSpec` is only `"cpu"` or `"gpu"` (Metal uses `gpu` on Apple Silicon).
