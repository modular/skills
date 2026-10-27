# Renaming safetensor weights for MAX

Hugging Face and MAX usually have different module names for the same
weights. `weight_adapters.py` is where you bridge them.

## Discover checkpoint keys first (metadata only)

Before writing or editing `weight_adapters.py`, list what the Hub
checkpoint ships (keys, shapes, and dtypes) without
downloading tensor payloads:

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> --summary
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> \
  --prefix model. --limit 40
```

This uses `huggingface_hub.get_safetensors_metadata` (HTTP header / index
parsing). Use it to check:

- whether the repo is safetensors at all (fail fast on bin-only repos)
- per-tensor shapes (derive `head_dim`, MoE expert layout, and
  shared-expert width)
- per-tensor dtypes (catch stray F32 norms in BF16 checkpoints)
- the key prefixes your adapter must strip or drop

Run this **after** reading `config.json` and **before** scaffold or adapter
edits. See also [read-config-json.md](read-config-json.md).

## How MAX's weight loader works

The pipeline runs your `weight_adapters.py` over the checkpoint first. The
adapter is a function that takes a dict of checkpoint key → tensor and
returns a dict of MAX FQN → tensor. `compile(weights=...)` then walks the
root module's parameters and looks up each fully-qualified name (FQN) in
that dict. The FQN is the attribute path from the root module. A
parameter with no entry raises `KeyError`, and a shape or dtype mismatch
raises `ValueError`. An entry that matches no parameter is left out
without an error.

## Discovering MAX's expected FQNs

After scaffolding, before debugging anything, list the FQNs your MAX model
expects:

```python
# In a Python REPL with your scaffold importable:
from max.experimental import functional as F
from your_package.your_slug import YourModel

# Build lazily, so no weights are allocated:
with F.lazy():
    m = YourModel(
        ...
    )  # use the args your pipeline model's _instantiate_module passes
print(sorted(name for name, _ in m.parameters))
```

These are the names your adapter must produce.

Then list the Hub safetensor keys (preferred, because it skips the
download):

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID>
```

Or, if you already have a local cache, open shards with `safe_open`:

```python
from safetensors import safe_open

keys = []
for shard in [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]:
    with safe_open(f"<checkpoint_path>/{shard}", framework="pt") as f:
        keys.extend(f.keys())
print(sorted(keys))
```

Diff the two lists. The diffs are what your adapter must rewrite.

## Common rename patterns

### Rewrite the `model.` prefix

HF wraps the decoder inside `model.layers.<i>...`. The MAX root module
usually holds the text model under an attribute of its own, and its
parameter names start with that attribute (`language_model.` in `olmo3`).
Most adapters start with a prefix rewrite:

```python
PREFIX_MAP = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "model.layers.": "language_model.layers.",
    "lm_head.": "language_model.lm_head.",
}


def adapt(state):
    out = {}
    for key, value in state.items():
        for before, after in PREFIX_MAP.items():
            if key.startswith(before):
                key = after + key.removeprefix(before)
                break
        out[key] = value
    return out
```

Match the prefixes to your root module's attribute names. A root that
holds the decoder layers directly takes a plain `removeprefix("model.")`.

### Embedding layer

| HF                          | MAX (varies by arch)                             |
|-----------------------------|--------------------------------------------------|
| `model.embed_tokens.weight` | `embed_tokens.weight` or `tok_embeddings.weight` |
| `model.norm.weight`         | `norm.weight` or `output_norm.weight`            |
| `lm_head.weight`            | `lm_head.weight` (absent when tied)              |

### Attention projections

If the MAX attention holds **unfused** Q/K/V parameters and concatenates
them in `forward()` (`olmo3`, `gemma3_modulev3`):

| HF                                 | MAX                                |
|------------------------------------|------------------------------------|
| `layers.N.self_attn.q_proj.weight` | `layers.N.self_attn.q_proj.weight` |
| `layers.N.self_attn.k_proj.weight` | `layers.N.self_attn.k_proj.weight` |
| `layers.N.self_attn.v_proj.weight` | `layers.N.self_attn.v_proj.weight` |
| `layers.N.self_attn.o_proj.weight` | `layers.N.self_attn.o_proj.weight` |

Often no rename is needed beyond the prefix.

If the MAX attention holds one **fused** Q/K/V parameter (some custom
modules), concatenate the HF tensors along the output dimension in the
adapter:

| HF                                                | MAX                                       |
|---------------------------------------------------|-------------------------------------------|
| `q_proj.weight`, `k_proj.weight`, `v_proj.weight` | `qkv_proj.weight` (Q, K, V rows in order) |

### MLP

For SwiGLU (Llama-family):

| HF                     | MAX                    |
|------------------------|------------------------|
| `mlp.gate_proj.weight` | `mlp.gate_proj.weight` |
| `mlp.up_proj.weight`   | `mlp.up_proj.weight`   |
| `mlp.down_proj.weight` | `mlp.down_proj.weight` |

For two-layer MLP (GPT-style):

| HF               | MAX              |
|------------------|------------------|
| `mlp.fc1.weight` | `mlp.fc1.weight` |
| `mlp.fc2.weight` | `mlp.fc2.weight` |

Some HF models use `dense_h_to_4h` / `dense_4h_to_h`. Rename those in the
adapter.

### Layer norms

| HF                                         | MAX                                        |
|--------------------------------------------|--------------------------------------------|
| `layers.N.input_layernorm.weight`          | `layers.N.input_layernorm.weight`          |
| `layers.N.post_attention_layernorm.weight` | `layers.N.post_attention_layernorm.weight` |

For post-norm or peri-LN models with extra norms (`post_norm1`,
`post_feedforward_layernorm`, etc.), add explicit renames pointing them
at the corresponding MAX module attribute names.

## Common weight-mapping mistakes

### `Conv2d` weight layout follows `permute`

`max.nn.Conv2d(..., permute=True)` takes its `weight` in PyTorch
order, `(out_channels, in_channels / num_groups, height, width)`, and reads it
in that layout, so the adapter passes the HF tensor through unchanged. That mode
also takes NCHW input, and its FCRS filter layout needs a cuDNN GPU. With the
default `permute=False`, the layer expects MAX order,
`(height, width, in_channels / num_groups, out_channels)`, and the adapter must
permute the HF tensor (`w.permute(2, 3, 1, 0)` in torch). Pick one and keep the
adapter consistent with it.

### Tied embeddings keep one tensor

When `tie_word_embeddings` is set, the module has no `lm_head` parameter
and multiplies by `embed_tokens.weight.T` in `forward()`. The adapter
passes `embed_tokens.weight` through and drops any `lm_head.weight` the
checkpoint ships:

```python
def adapt(state):
    out = {
        k.replace("model.", "language_model.", 1): v for k, v in state.items()
    }
    if config.tie_word_embeddings:
        out.pop("lm_head.weight", None)
    return out
```

A copied `lm_head.weight` for a tied model matches no parameter, so it is
left out. A port that builds an untied `lm_head` for a tied checkpoint
fails `compile()` with a `KeyError` for it.

### Fused QKV checkpoints

GPT-NeoX, BLOOM, and a few others ship `query_key_value.weight` as a
single fused tensor. Split in the adapter:

```python
import torch

hidden = config.hidden_size
qkv = state["transformer.h.0.attention.query_key_value.weight"].data()
# Layout 1: [Q, K, V] concatenated along the output dim
arr = torch.from_dlpack(qkv)  # NumPy can't read bfloat16
q = arr[:hidden]
k = arr[hidden : 2 * hidden]
v = arr[2 * hidden :]
# Layout 2 (head-interleaved): every head is [Q_head, K_head, V_head]
# Check the modeling code for which layout the model uses.
```

## Discovering renames by diffing

To write an adapter from the two key lists:

1. Run `list_checkpoint_keys.py` on the Hub repo (metadata only).
2. Print MAX's expected FQNs (from your port's module tree).
3. Pair them by inspection.
4. Write the adapter to do the renames.

This helper prints both differences:

```python
def diff_keys(hf_keys, max_keys):
    """Print HF keys MAX doesn't expect, and MAX keys HF doesn't supply."""
    hf_set, max_set = set(hf_keys), set(max_keys)
    print("HF only:")
    for k in sorted(hf_set - max_set)[:30]:
        print(f"  {k}")
    print("MAX only:")
    for k in sorted(max_set - hf_set)[:30]:
        print(f"  {k}")
```

If "MAX only" is non-empty after your adapter runs, `compile()` fails
with a `KeyError` for the first one. If "HF only" is non-empty, those
tensors are left out without an error. Each one is either a rename bug or
a component your module doesn't build. The adapter must produce *every*
MAX FQN, and every HF tensor must land on one or be dropped on purpose
(see [state-dict-audit.md](state-dict-audit.md)).

## Verifying the adapter

Check the adapter output before compiling. This sanity check reads one small
layer:

```python
import torch

state = convert_safetensor_state_dict(raw_weights, hf_config, pipeline_config)
w = torch.from_dlpack(state["language_model.layers.0.self_attn.q_proj.weight"])
print(w.abs().max())
# Should be O(0.1) for a trained model. Suspect O(0.01) or O(1.0).
```

For a thorough check, run the divergence-hunt `compare_layers.py` once
weights load cleanly and before declaring logits done. The first layer should
already match. If it doesn't, the adapter is wrong.
