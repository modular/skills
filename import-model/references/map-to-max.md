# Picking a starting MAX architecture

You're answering: which already-ported MAX architecture is closest to mine?
The closest one becomes the starting point when you scaffold; you
implement deltas while implementing the graph.

## Listing what's available

Do not maintain a static slug list here; it drifts every MAX release. List
what your installed MAX registers:

```bash
pixi run python scripts/list_native_archs.py
```

That script, `list_native_archs.py` in this skill's `scripts/`, prints
`HF_architectures[0] → max_slug` from the arch trees on disk. Pick a donor
slug from the output, then open `max/pipelines/architectures/<slug>/` in your
installed MAX package.

## Decision table

Map the `config.json` findings to a starting arch:

| HF signal                                      | Starting arch                                         |
|------------------------------------------------|-------------------------------------------------------|
| `LlamaForCausalLM` (or compatible)             | `llama3`                                              |
| `Qwen2ForCausalLM`                             | `qwen2`                                               |
| `Qwen3ForCausalLM` (with `use_qk_norm`)        | `qwen3`                                               |
| `MistralForCausalLM` with `sliding_window`     | `mistral`                                             |
| `Gemma3ForCausalLM`                            | `gemma3`                                              |
| `Phi3ForCausalLM` with `partial_rotary_factor` | `phi3`                                                |
| `GraniteForCausalLM` (MuP scalars)             | `granite`                                             |
| Any `*MoEForCausalLM` (sparse experts, top-k)  | `qwen3` (registers `Qwen3MoeForCausalLM`)             |
| `DeepseekV3ForCausalLM`, MLA + MoE             | `deepseekV3`                                          |
| Encoder-decoder audio                          | `whisper`                                             |
| `*ForSequenceClassification` text encoder      | none stock; start from any decoder, drop the LM head  |
| Vision-language (image + text)                 | `qwen2_5vl` or `internvl`                             |

## When nothing fits

If your config has multiple uncommon signals (for example, MLA *and* a custom
routing scheme, or recurrence with non-standard memory), no template will
match. Two paths:

- Pick the closest decoder and write the unique pieces from scratch with
  the lane's primitives (`max.nn` on V2, `max.experimental.nn` on
  ModuleV3). Accept that the scaffold-stage parity check will
  fail until you replace the divergent module.
- See [recognize-walls.md](recognize-walls.md): some architectures aren't
  portable with the public MAX surface today.

## Reading the chosen arch

Once you've picked, read its source in your installed MAX package:

`max/pipelines/architectures/<chosen_slug>/<chosen_slug>.py`

What you're looking for:

- The top-level model class: usually inherits from a base in
  `max.pipelines.lib`, on both lanes.
- The block class: on V2 it usually inherits from `TransformerBlock`
  in `max.nn.transformer`. On ModuleV3 there is no shared base; each
  architecture defines its own block as a `Module` (for example,
  `olmo3/layers/transformer.py`).
- The attention class: on V2 it usually inherits from `AttentionWithRope`
  or similar. On ModuleV3 it is a `Module` over `Tensor` and
  `PagedCacheValues` (for example, `olmo3/layers/attention.py`), composed
  from `max.experimental.nn.common_layers` and `max.experimental.nn.rope`.
- The MLP class: on V2 it usually inherits from `MLP` (SwiGLU) in
  `max.nn`. On ModuleV3 it is a `Module` built from
  `max.experimental.nn.linear` and the activations in
  `max.experimental.nn.common_layers`.

These inheritance chains tell you what is available to subclass when you
hit "I need to change one method." If you change just one method on a
subclass, your port stays small.

## Output of the comparison

A short note for yourself:

- **Starting arch:** `<slug>`
- **What I will reuse unchanged:** (probably the embedding, the final
  RMSNorm, the LM head)
- **What I need to subclass:** (attention? MLP? block?)
- **What I need to add:** (extra norms, MoE routing, multi-step head)

This note becomes the edit list when you implement the graph.
