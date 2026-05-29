# Is it already in MAX?

Before writing any code, read the model's `config.json` and check whether MAX
already registers that architecture class.

## Pre-check HF gated-repo access (for VL ports especially)

Cohere, Meta, some Mistral, and many VL repos ship as **gated** — even with a
valid `$HF_TOKEN`, your account may not be in the per-repo allowlist. A gated
repo blocks the architecture audit and any direct-serve attempt five minutes
into the bring-up, after the rest of the guard check has already spent your
tokens.

```bash
pixi run python -c "
from huggingface_hub import HfApi
info = HfApi().model_info('<HF_ID>')
print('gated:', info.gated)
# Real probe — the metadata 'gated' flag is unreliable; only a download
# call confirms access:
from huggingface_hub import hf_hub_download
hf_hub_download('<HF_ID>', 'config.json')
print('access ok')
"
```

If gated: search for an ungated mirror (often `mlx-community/<base>-bf16` or
similar Apple-MLX repacks); if none exists, request access or mark the
bring-up BLOCKED on auth (don't scaffold a slug you can't validate).
Common on gated vision-language repos when your account lacks allowlist access.

## Identify the architecture

Every Hugging Face checkpoint ships a `config.json`. Two fields identify the
architecture:

| Field              | Example            | Use in this guard                                                                                |
|--------------------|--------------------|--------------------------------------------------------------------------------------------------|
| `architectures[0]` | `Qwen3ForCausalLM` | **This** is what MAX registers — pass it to `--match`.                                           |
| `model_type`       | `qwen3`            | Locates `modeling_<type>.py` in Transformers when comparing to a donor. Not used for this check. |

MAX maintains a registry mapping each Hugging Face class name to an internal
slug (for example `LlamaForCausalLM` → `llama3`). If your model's
`architectures[0]` is already in that registry, you serve Hub weights
directly — no custom graph, no port.

### Read `architectures[0]`

```bash
pixi run python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True)
print('architectures[0]:', cfg.architectures[0])
print('model_type:', cfg.model_type)
"
```

Or open the raw Hub file:
`https://huggingface.co/<HF_MODEL_ID>/raw/main/config.json` and read the
`"architectures"` array.

### Check the MAX registry

```bash
pixi run python list_native_archs.py --match <architectures[0]>
```

- **Exit 0** and prints `ClassName\tslug` → MAX already supports it. Run
  `pixi run max serve --model <HF_MODEL_ID>` and **stop**. No port is needed.
- **Exit 1** (no output) → not registered. Continue with Phase 1 (read the model
  card).
- **Exit 2** (stderr install hint) → MAX is not in the active Python env.
  Fix the pixi environment first; do **not** treat this as “needs a port”.

To browse everything MAX ships:

```bash
pixi run python list_native_archs.py
```

## Scan for port walls

Before scaffolding, scan the Hub config for hard blockers (ALiBi, quant-only
weights, SSM/recurrence signals, extreme scale):

```bash
pixi run python check_walls.py <HF_MODEL_ID>
```

- **Exit 0** — no blockers.
- **Exit 1** — warnings only (review [recognize-walls.md](recognize-walls.md)).
- **Exit 2** — at least one hard blocker; do not scaffold until resolved.

## Worked example

`Qwen/Qwen3-8B` has this in `config.json`:

```json
"architectures": ["Qwen3ForCausalLM"],
"model_type": "qwen3"
```

Check:

```bash
pixi run python list_native_archs.py --match Qwen3ForCausalLM
# Qwen3ForCausalLM    qwen3
```

Registered → done:

```bash
pixi run max serve --model Qwen/Qwen3-8B
```

Contrast with a model whose class is **not** in the registry — say
`NewFamilyForCausalLM`. The same `--match` command exits 1 with no output.
That is your signal to port (continue with Phase 1).

**Do not confuse `model_type` with `architectures[0]`.** A checkpoint can have
`"model_type": "llama"` while `"architectures": ["LlamaForCausalLM"]` — always
match on the class name in `architectures[0]`, not the folder name.
