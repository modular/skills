# Picking a starting MAX architecture

You're answering: which already-ported MAX architecture is closest to mine?
The closest one becomes the starting point when you scaffold; you
implement deltas while implementing the graph.

## Listing what's available

Do not maintain a static slug list here — it drifts every MAX release. List
what your installed MAX actually registers:

```bash
pixi run python list_native_archs.py
```

That prints `HF_architectures[0] → max_slug` from the arch trees on disk (same
discovery as `list_native_archs.py` in this skill's `scripts/`). Pick a donor
slug from the output, then open
`modular/max/python/max/pipelines/architectures/<slug>/` in the monorepo.

## Decision table

Map the `config.json` findings to a starting arch:

| HF signal                                      | Starting arch                                         |
|------------------------------------------------|-------------------------------------------------------|
| `LlamaForCausalLM` (or compatible)             | `llama3`                                              |
| `Qwen2ForCausalLM`                             | `qwen2`                                               |
| `Qwen3ForCausalLM` (with `use_qk_norm`)        | `qwen3`                                               |
| `MistralForCausalLM` with `sliding_window`     | `mistral`                                             |
| `Gemma2ForCausalLM` / `Gemma3ForCausalLM`      | `gemma2` / `gemma3`                                   |
| `Phi3ForCausalLM` with `partial_rotary_factor` | `phi3`                                                |
| `GraniteForCausalLM` (MuP scalars)             | `granite`                                             |
| Any `*MoEForCausalLM` with dense expert MLPs   | `mixtral` or `qwen3_moe`                              |
| `DeepseekV3ForCausalLM`, MLA + MoE             | `deepseekV3`                                          |
| Encoder-decoder audio                          | `whisper`                                             |
| `*ForSequenceClassification` text encoder      | none stock — start from any decoder, drop the LM head |
| Vision-language (image + text)                 | `qwen2_5_vl` or `internvl`                            |

## When nothing fits

If your config has multiple uncommon signals (e.g. MLA *and* a custom
routing scheme, or recurrence with non-standard memory), no template will
match. Two paths:

- Pick the closest decoder and write the unique pieces from scratch with
  `max.nn` primitives. Accept that the scaffold-stage parity check will
  fail until you replace the divergent module.
- See [recognize-walls.md](recognize-walls.md) — some architectures aren't
  portable with the public MAX surface today.

## Reading the chosen arch

Once you've picked, read its source in the monorepo:

`modular/max/python/max/pipelines/architectures/<chosen_slug>/<chosen_slug>.py`

What you're looking for:

- The top-level model class — usually inherits from a base in
  `max.pipelines.lib`.
- The block class — usually inherits from a `TransformerBlock` in
  `max.nn.transformer`.
- The attention class — usually inherits from `AttentionWithRope` or
  similar.
- The MLP class — usually inherits from `MLP` (SwiGLU) in `max.nn`.

These inheritance chains tell you what's available to subclass when you
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
