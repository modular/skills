# Pick a starting MAX architecture

You're answering: which ModuleV3 architecture is closest to mine? The
closest one becomes the starting point when you scaffold. You implement the
deltas while implementing the module.

## List what's available

Don't maintain a static slug list here, because it drifts every MAX
release. List the ModuleV3 architectures your installed MAX ships:

```bash
pixi run python scripts/list_native_archs.py --donors
```

That script, `list_native_archs.py` in this skill's `scripts/`, prints each
ModuleV3 architecture's slug and registered name from the arch trees on
disk. Pick a donor slug from the output, then open
`max/pipelines/architectures/<slug>/` in your installed MAX package.

## Decision table

Map the `config.json` findings to a starting arch:

| HF signal                                       | Starting arch                                                         |
|-------------------------------------------------|-----------------------------------------------------------------------|
| `LlamaForCausalLM` (or compatible)              | `llama3_modulev3`                                                     |
| `Qwen2ForCausalLM`, `Qwen3ForCausalLM`          | `llama3_modulev3`, with QK-norm from `gemma3_modulev3`                |
| `layer_types` mixing sliding and full attention | `olmo3` or `gemma3_modulev3`                                          |
| `Gemma3ForCausalLM`                             | `gemma3_modulev3`                                                     |
| `Phi3ForCausalLM` (fused `qkv_proj`)            | `phi3_modulev3`                                                       |
| `GraniteForCausalLM` (MuP scalars)              | `granite_modulev3`                                                    |
| Any `*MoeForCausalLM` (sparse experts, top-k)   | `gpt_oss_modulev3`                                                    |
| `DeepseekV2ForCausalLM`, MLA on one GPU         | `deepseekV2_modulev3`                                                 |
| `DeepseekV3ForCausalLM`, MLA + MoE across GPUs  | `deepseekV3_modulev3`                                                 |
| Hybrid attention and Mamba layers               | `nemotron_h_modulev3`                                                 |
| Vision-language (image + text)                  | `gemma3multimodal_modulev3`, `idefics3_modulev3`, `kimik2_5_modulev3` |
| `*ForMaskedLM` / sentence-embedding encoder     | `mpnet_modulev3`, `qwen3_embedding_modulev3`                          |

`gemma3_modulev3` puts QK-norm per head (Qwen3's layout). `olmo3` puts it
across the full projection width. Check which one HF does.

`phi3_modulev3` rotates the full head dimension. No single-module ModuleV3
donor implements partial RoPE, so a `partial_rotary_factor` below 1 is a delta
you implement. `ProportionalRotaryEmbedding` in
`gemma4_modulev3/layers/rotary_embedding.py` is a ModuleV3 example to copy.

The vision-language donors build several compiled modules (vision tower and
language model). Scaffold them with `scaffold.py --full-copy`, which copies
every donor file and renames it. The default subclass skeleton covers
single-module donors.

## When nothing fits

If your config has multiple uncommon signals (for example, MLA *and* a custom
routing scheme, or recurrence with non-standard memory), no template will
match. Pick one of these paths:

- Pick the closest decoder and write the unique pieces from scratch as
  `Module` subclasses built from `max.nn` and
  `max.functional`. Accept that the scaffold-stage parity check
  will fail until you replace the divergent module.
- See [recognize-walls.md](recognize-walls.md): some architectures aren't
  portable with the public MAX surface today.

## Read the chosen arch

Once you've picked, read its source in your installed MAX package. Start from
`max/pipelines/architectures/<chosen_slug>/arch.py`. A registered donor
names its root module there, in `modules={GraphRole.DECODER: <Module>}`
(`gemma3_modulev3/gemma3.py`). Otherwise, `pipeline_model=` names the
pipeline model in `model.py`, whose `_instantiate_module()` names the root
module (`olmo3/olmo3.py`). Read the file that defines the root module next.

What you're looking for:

- How the donor is served. A registered donor has no pipeline model: the
  framework constructs its root module as `Module(config)` and calls its
  `forward(inputs: DecoderInputs)`. Any other donor has a pipeline model
  class in `model.py` that subclasses `ModuleV3PipelineModelWithKVCache`
  from `max.pipelines.lib`, directly or through another architecture's
  model (`phi3_modulev3` reuses `llama3_modulev3`'s).
- The root module: a `Module` whose `forward()` unflattens the KV cache inputs
  and calls a text model, often stored as `self.language_model`, which sets
  the weight-name prefix.
- The block class: each architecture defines its own block as a `Module`
  (for example, `olmo3/layers/transformer.py`).
- The attention class: a `Module` over `Tensor` and `PagedCacheValues` (for
  example, `olmo3/layers/attention.py`), composed from
  `max.nn.common_layers` and the fused kernels in
  `common_layers.functional_kernels`.
- The MLP class: `MLP` from `max.nn.common_layers.mlp`, or a
  `Module` built from `max.nn.Linear` and the activations in
  `max.functional`.

These classes show what you can subclass when you need to change one
method. A subclass that overrides one method keeps your port small.

## Output of the comparison

A short note for yourself:

- **Starting arch:** `<slug>`
- **What I will reuse unchanged:** (probably the embedding, the final
  RMSNorm, the LM head)
- **What I need to subclass:** (attention? MLP? block?)
- **What I need to add:** (extra norms, MoE routing, multi-step head)

This note becomes the edit list when you implement the module.
