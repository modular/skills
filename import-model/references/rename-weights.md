# Renaming safetensor weights for MAX

Hugging Face and MAX usually have different module names for the same
weights. `weight_adapters.py` is where you bridge them.

## Discover checkpoint keys first (metadata only)

Before writing or editing `weight_adapters.py`, list what the Hub
checkpoint **actually** ships — keys, shapes, and dtypes — without
downloading tensor payloads:

```bash
pixi run python list_checkpoint_keys.py <HF_MODEL_ID> --summary
pixi run python list_checkpoint_keys.py <HF_MODEL_ID> \
  --prefix model. --limit 40
```

This uses `huggingface_hub.get_safetensors_metadata` (HTTP header / index
parsing). It is the ground truth for:

- whether the repo is safetensors at all (fail fast on bin-only repos)
- per-tensor shapes (derive `head_dim`, MoE expert layout, shared-expert width)
- per-tensor dtypes (catch stray F32 norms in BF16 checkpoints)
- the exact key prefixes your adapter must strip or drop

Run this **after** reading `config.json` and **before** scaffold or adapter
edits. See also [read-config-json.md](read-config-json.md).

## How MAX's weight loader works

When MAX loads a checkpoint, it walks the module tree, looks up each
parameter's fully-qualified name (FQN), and tries to find a matching
tensor in the safetensor file. If the FQN matches a checkpoint key
directly, it loads. If not, your `weight_adapters.py` is given a chance to
rewrite the mapping before loading happens.

The adapter is a function that takes a dict of checkpoint key → tensor
and returns a dict of MAX FQN → tensor.

## Discovering MAX's expected FQNs

After scaffolding, before debugging anything, list the FQNs your MAX model
expects:

```python
# In a Python REPL with your scaffold importable:
from your_package import YourModel
# Instantiate without loading weights:
m = YourModel(...)  # use the same args as your pipeline model
print(sorted(m.state_dict().keys()))
```

These are the names your adapter must produce.

Then list the Hub safetensor keys (preferred — no download):

```bash
pixi run python list_checkpoint_keys.py <HF_MODEL_ID>
```

Or, if you already have a local cache, open shards with `safe_open`:

```python
from safetensors import safe_open
keys = []
for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    with safe_open(f"<checkpoint_path>/{shard}", framework="pt") as f:
        keys.extend(f.keys())
print(sorted(keys))
```

Diff the two lists. The diffs are what your adapter must rewrite.

## Common rename patterns

### Strip the `model.` prefix

HF wraps the decoder inside `model.layers.<i>...`. MAX often has the
decoder at the top level. Common first step in any adapter:

```python
def adapt(state):
    return {k.removeprefix("model."): v for k, v in state.items()}
```

### Embedding layer

| HF                          | MAX (varies by arch)                             |
|-----------------------------|--------------------------------------------------|
| `model.embed_tokens.weight` | `embed_tokens.weight` or `tok_embeddings.weight` |
| `model.norm.weight`         | `norm.weight` or `output_norm.weight`            |
| `lm_head.weight`            | `lm_head.weight` (or aliased if tied)            |

### Attention projections

If the MAX attention uses **unfused** Q/K/V (most stock attentions):

| HF                                 | MAX                                |
|------------------------------------|------------------------------------|
| `layers.N.self_attn.q_proj.weight` | `layers.N.self_attn.q_proj.weight` |
| `layers.N.self_attn.k_proj.weight` | `layers.N.self_attn.k_proj.weight` |
| `layers.N.self_attn.v_proj.weight` | `layers.N.self_attn.v_proj.weight` |
| `layers.N.self_attn.o_proj.weight` | `layers.N.self_attn.o_proj.weight` |

Often no rename is needed — just strip `model.`.

If the MAX attention uses **fused** Q/K/V (some custom modules):

| HF                                                | MAX                                                           |
|---------------------------------------------------|---------------------------------------------------------------|
| `q_proj.weight`, `k_proj.weight`, `v_proj.weight` | `qkv_proj.q.weight`, `qkv_proj.k.weight`, `qkv_proj.v.weight` |

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

Some HF models use `dense_h_to_4h` / `dense_4h_to_h` — rename in adapter.

### Layer norms

| HF                                         | MAX                                        |
|--------------------------------------------|--------------------------------------------|
| `layers.N.input_layernorm.weight`          | `layers.N.input_layernorm.weight`          |
| `layers.N.post_attention_layernorm.weight` | `layers.N.post_attention_layernorm.weight` |

For post-norm or peri-LN models with extra norms (`post_norm1`,
`post_feedforward_layernorm`, etc.), add explicit renames pointing them
at the corresponding MAX module attribute names.

## Patterns that always bite

### `Conv2d.weight` is NOT `Conv2d.filter`

In MAX's `Conv2d`, the runtime attribute is `filter` but the `state_dict`
key is `weight`. **Do not** remap `weight` → `filter` in your adapter.
Access in graph code via `self.conv.filter`, but treat the parameter as
`weight` for loading.

### Tied embeddings need explicit aliasing

```python
def adapt(state):
    out = {k.removeprefix("model."): v for k, v in state.items()}
    if config.tie_word_embeddings and "lm_head.weight" not in out:
        out["lm_head.weight"] = out["embed_tokens.weight"]
    return out
```

If you forget this, the LM head loads zeros (or random init) and the model
produces nonsense after the first few tokens.

### Fused QKV checkpoints

GPT-NeoX, BLOOM, and a few others ship `query_key_value.weight` as a
single fused tensor. Split in the adapter:

```python
hidden = config.hidden_size
qkv = state["transformer.h.0.attention.query_key_value.weight"]
# Layout 1: [Q, K, V] concatenated along the output dim
arr = qkv.to_buffer().to_numpy()
q = arr[:hidden]
k = arr[hidden:2 * hidden]
v = arr[2 * hidden:]
# Layout 2 (head-interleaved): every head is [Q_head, K_head, V_head]
# — check the modeling code for which layout the model uses.
```

## Discovering renames by diffing

The fastest way to write an adapter:

1. Run `list_checkpoint_keys.py` on the Hub repo (metadata only).
2. Print MAX's expected FQNs (from your port's module tree).
3. Pair them by inspection (most pairs are obvious).
4. Write the adapter to do the renames.

A short helper:

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

If "MAX only" is non-empty after your adapter runs, those tensors will
load with their initial random values — almost always a silent bug. The
load must produce *every* MAX FQN.

## Verifying the adapter

Load the model and check that no weight stayed at its random init. A
quick sanity check on a small layer:

```python
import numpy as np
# Compare against expected absmax of a real weight
print(np.abs(model.layers[0].self_attn.q_proj.weight.to_numpy()).max())
# Should be O(0.1) for a trained model; if it's O(0.01) or O(1.0), suspect.
```

For a thorough check, run the divergence-hunt `compare_layers.py` once
weights load cleanly and before declaring logits done. The first layer should
already match — if it doesn't, the adapter is wrong.
