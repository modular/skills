# Reading `config.json`

Every Hugging Face model ships a `config.json`. Reading it carefully is the
single highest-leverage thing you do in a port — every architectural delta
shows up here first, before you ever touch the modeling code.

Load it and print every field:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("<HF_MODEL_ID>", trust_remote_code=True)
print(config)
```

## Fields you map 1:1 to MAX

These have the same meaning in every transformer-family model. Your MAX
`model_config.py` will pull them directly from
`pipeline_config.model.huggingface_config`.

| Field                                          | Meaning                                                                 |
|------------------------------------------------|-------------------------------------------------------------------------|
| `hidden_size`                                  | Width of the residual stream                                            |
| `num_hidden_layers`                            | Number of transformer blocks                                            |
| `num_attention_heads`                          | Number of Q heads                                                       |
| `num_key_value_heads`                          | Number of K/V heads (equals `num_attention_heads` when not GQA)         |
| `intermediate_size`                            | MLP hidden dim                                                          |
| `vocab_size`                                   | Tokenizer vocabulary size                                               |
| `max_position_embeddings`                      | Training context length                                                 |
| `rms_norm_eps` / `layer_norm_eps`              | Norm epsilon                                                            |
| `hidden_act`                                   | MLP activation (`silu`, `gelu`, `gelu_new`, etc.)                       |
| `tie_word_embeddings`                          | Whether LM head reuses the embedding matrix                             |
| `rope_theta`                                   | RoPE base frequency (default 10000.0)                                   |
| `rope_scaling`                                 | RoPE extension config (Llama3, YaRN, etc.) — if present, read carefully |
| `attention_bias`                               | Whether Q/K/V/O projections have a bias term                            |
| `bos_token_id`, `eos_token_id`, `pad_token_id` | Special tokens                                                          |

## Fields that signal a delta

If any of these are present and non-default, you have architectural work to
do beyond a simple rename.

| Field                                                                                   | Signals                                                                                                                 |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `sliding_window` (int)                                                                  | Sliding-window attention. Stock attention won't match.                                                                  |
| `attn_logit_softcapping`                                                                | Softcap applied to pre-softmax attention scores (Gemma 2).                                                              |
| `final_logit_softcapping`                                                               | Softcap applied to final LM-head logits (Gemma 2).                                                                      |
| `partial_rotary_factor` (float < 1.0)                                                   | RoPE only applied to a fraction of head dim.                                                                            |
| `head_dim` (when not equal to `hidden_size / num_attention_heads`)                      | Non-standard head shape.                                                                                                |
| `num_experts` / `num_local_experts`                                                     | MoE layer count.                                                                                                        |
| `num_experts_per_tok`                                                                   | Top-k routing.                                                                                                          |
| `router_aux_loss_coef`, `output_router_logits`                                          | Confirms MoE.                                                                                                           |
| `q_lora_rank`, `kv_lora_rank`                                                           | MLA (DeepSeek-style latent attention).                                                                                  |
| `qk_nope_head_dim`, `qk_rope_head_dim`                                                  | MLA head splitting.                                                                                                     |
| `use_qk_norm` (bool)                                                                    | Q and K each normalized before the dot product.                                                                         |
| `quantization_config`                                                                   | Released weights are quantized; you need a dequant path.                                                                |
| `rope_parameters` (dict)                                                                | `rope_theta` nested instead of top-level. Read `rope_parameters['rope_theta']`.                                         |
| `use_post_norm`                                                                         | Peri-LN: post-norm applied to each sublayer output before residual add.                                                 |
| `mlp_bias`                                                                              | MLP linear layers have a bias term.                                                                                     |
| `embedding_multiplier`, `logits_scaling`, `attention_multiplier`, `residual_multiplier` | MuP scalars — must be threaded through layers or output is garbage.                                                     |
| `dim_model_base`                                                                        | Often pairs with a pre-`lm_head` width divisor: `h / (hidden_size / dim_model_base)`. Grep `lm_head(` in modeling code. |
| `output_multiplier_scale`, `pre_attn_norm_scale`, ...                                   | Custom scale fields — read modeling code to see where they apply.                                                       |

## What "non-standard" actually means

Most novel models have **two or three** non-standard fields, not ten. The
trick is recognizing which ones are cosmetic (a new field name for a known
concept) vs. structural (a real new computation).

Quick triage:

- A new *scalar* (eps, theta, scale factor) → almost always cosmetic.
  Pull it from config, multiply or add at the right point. One-line fix.
- A new *boolean* (`use_qk_norm`, `use_post_norm`) → conditionally
  enables a structural change. Read the modeling code to see what it
  guards.
- A new *integer count* (`num_experts`, `q_lora_rank`) → structural.
  Changes the shape of the computation.
- A new *dict* (`rope_scaling`, `quantization_config`) → structural.
  Read it carefully; each subfield matters.

## Verifying you have the right config

The config and the released checkpoint must agree on shape. After loading
the config, sanity-check it against safetensors metadata (no weight download):

```bash
pixi run python list_checkpoint_keys.py <HF_MODEL_ID> --summary
```

Or inspect a few attention keys from the metadata table:

```bash
pixi run python list_checkpoint_keys.py <HF_MODEL_ID> \
  --prefix model.layers.0.self_attn --limit 10
```

Programmatically (same data as the script):

```python
from huggingface_hub import get_safetensors_metadata

meta = get_safetensors_metadata("<HF_MODEL_ID>")
for name, info in meta.files_metadata["model.safetensors"].tensors.items():
    if "self_attn.q_proj" in name:
        print(name, info.shape, info.dtype)
        break
```

If `config.hidden_size = 4096` and `num_attention_heads = 32` but the
checkpoint's `q_proj.weight` is `[4096, 4608]`, the model has a non-default
`head_dim = 4608 / 32 = 144`. The config either has an explicit `head_dim`
field or you need to add one in `model_config.py`.

## Output of reading `config.json`

Write down two lists:

1. **Standard fields** — names + values, for `model_config.py`.
2. **Watch-out signals** — non-default fields and what they imply. This
   feeds the delta list you build by comparing against the donor MAX arch.
