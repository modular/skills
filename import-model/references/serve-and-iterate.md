# Serve and iterate (detail)

**Path:** `<port_dir>` is the slug folder with `arch.py` and `ARCHITECTURES` in
`__init__.py` (after scaffold, `<port_dir> = <output_dir>/<slug>/`). Pass
**the same** `<port_dir>` to `--custom-architectures` and to
`run_oss_gates.py --port-dir`.

MAX loads custom archs by taking `dirname(<port_dir>)` for `sys.path` and
importing `basename(<port_dir>)` as the module. Do **not** pass the parent of
`<port_dir>` — you will import the wrong package (e.g. `custom-arch` instead
of your slug) and see `AttributeError: module '…' has no attribute
'ARCHITECTURES'`.

Optional colon form: ``<parent_on_sys.path>:<module_name>`` (same effect as
passing `<port_dir>/`).

## Prerequisites: run these before `pixi run max serve`

`max serve` cold-compiles for 5–25 minutes. Before serving, run the four local
checks below. `run_oss_gates.py` covers walls and `arch.py` registration only.

### Import smoke

Manual import test (parent on path, slug as module name — mirrors what MAX
does):

```bash
pixi run python -c "
import importlib, sys
port_dir = '<absolute/path/to>/<slug>'
sys.path.insert(0, str(__import__('pathlib').Path(port_dir).parent))
mod = importlib.import_module('<slug>')
arches = getattr(mod, 'ARCHITECTURES', None)
assert arches and len(arches) >= 1, f'ARCHITECTURES missing: {arches}'
print('arch.name:', arches[0].name)
print('OK')
"
```

### Graph dry-build with a stubbed state_dict

```bash
pixi run python -c "
import sys
from pathlib import Path
port_dir = Path('<port_dir>')
sys.path.insert(0, str(port_dir.parent))
from max.graph import Graph
from transformers import AutoConfig
from <slug>.<slug> import <YourGraphClass>
from <slug>.model_config import <YourConfig>

hf = AutoConfig.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True)
cfg = <YourConfig>(huggingface_config=hf, quantization_encoding=None)
cfg.finalize(state_dict={}, devices=[...])
with Graph('smoke') as g:
    model = <YourGraphClass>(cfg)
print('graph built; n_params =', sum(1 for _ in model.parameters()))
"
```

### Adapter ⇄ graph key cross-check

Use `list_checkpoint_keys.py` for Hub keys and diff against your graph's
expected FQNs after the adapter runs. See
[rename-weights.md](rename-weights.md).

### Weights-format preflight

MAX loads only `.safetensors` or `.gguf` (`WeightsFormat` in
`max/graph/weights/format.py`) — no `.bin`. In `<port_dir>/arch.py`, copy
`default_weights_format` and `weight_adapters` from your scaffold donor under
`modular/max/python/max/pipelines/architectures/<donor>/arch.py`; see
[pitfalls-config.md § Import and config API traps](pitfalls-config.md#import-and-config-api-traps).

Repo file check:

```bash
pixi run python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('<HF_MODEL_ID>')
has_st = any(f.endswith('.safetensors') for f in files)
has_gguf = any(f.endswith('.gguf') for f in files)
has_bin = any(f.endswith('.bin') for f in files)
print(f'safetensors={has_st}  gguf={has_gguf}  bin_only_legacy={has_bin and not has_st}')
if has_bin and not has_st:
    print('STOP: convert to safetensors or pick a GGUF repo — MAX cannot load .bin')
"
```

## Sanity-check the HF reference first

```bash
pixi run python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = '<HF_MODEL_ID>'
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    mid, trust_remote_code=True, device_map='auto', dtype=torch.bfloat16,
).eval()
ids = tok('<MODEL CARD EXAMPLE PROMPT>', return_tensors='pt').input_ids.to(m.device)
with torch.no_grad():
    out = m.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=False))
"
```

## Serve and probe

```bash
pixi run max serve --model-path <HF_MODEL_ID> \
  --custom-architectures <port_dir> \
  --quantization-encoding <default_encoding from arch.py>

curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<slug>", "prompt": "<MODEL CARD EXAMPLE PROMPT>", "max_tokens": 64}' \
  | pixi run python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

Use the model card's template for the prompt.

## Encoder / embedding slugs

Same `--custom-architectures <port_dir>`; endpoint is `/v1/embeddings` with
`input` (not `prompt`).

## Three possible outcomes

- **Crash on load** → config, imports, or weight adapter.
- **Garbage tokens** → graph still implements donor math.
- **Plausible short output** → run `max_tokens=64+` before celebrating.

## Iterating the fix-test loop

Cold compile is 5–25 minutes per iteration: one fix per serve, `pkill -9 max`
between runs, use [layer-by-layer-debugging.md](layer-by-layer-debugging.md)
when logits diverge.
