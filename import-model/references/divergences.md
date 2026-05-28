# Common divergence causes (indexed by symptom)

When the layer-by-layer divergence hunt shows divergence, the bug is almost
always one of the patterns below.

## Quick index: symptom → candidates

Match what you're seeing to a row, then read every listed cause — not
just the first one that looks plausible. Several causes produce the same
symptom, and the bug is the one you haven't checked yet.

| What you observe                                                            | Most likely causes (in order)                                                                                                                      |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Gibberish at token 0, low-id token unrelated to prompt                      | #1 Q/K/V naming, #10 fused QKV, #4 tied embeddings, #15 norm variant                                                                               |
| Repeated same character/token (`&&&&`, single punct loop)                   | #17 NoPE identity freqs_cis layout, #3 RoPE packing, #1 weight map                                                                                 |
| Mild divergence at every MLP post-output (cos_sim ~0.95)                    | #14 activation variant, #12 non-gated MLP, #13 MuP scalars                                                                                         |
| Divergence at post-attn block 0, cos_sim 0.1–0.5                            | #2 GQA repeat_kv, #1 Q/K/V naming                                                                                                                  |
| Divergence at post-attn block 0, cos_sim 0.7–0.9                            | #3 RoPE split-half vs interleaved, #15 QK-norm dim, #18 final softcap                                                                              |
| First token plausible, output degrades over generation                      | #4 tied embeddings, #6 partial-rotary padding, #8 nested rope_theta                                                                                |
| Repetition / position-0 loop after first token                              | #7 absolute positional embeddings, #6 partial-rotary padding                                                                                       |
| Divergence grows with sequence length, short prompts fine                   | #6 partial-rotary padding, #8 nested rope_theta, #3 RoPE style                                                                                     |
| Prefill ok, decode collapses to repetition/word-fragment loop by token 8-12 | #3 RoPE style — `RotaryEmbedding(interleaved=True)` default mismatches HF `rotate_half`; pass `interleaved=False`                                  |
| Divergence at post-block-norm but both sublayers ok                         | #5 pre/post/peri-norm layout, #13 residual_multiplier (MuP)                                                                                        |
| `actual_token == last input token` on first decode                          | #5 post-norm degenerating to identity                                                                                                              |
| All-NaN logits from HF reference                                            | [pitfalls-serving.md](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan) (`trust_remote_code` + `.to("cuda")` device mismatch) |
| Layer outputs match but generated text drifts                               | #16 tokenizer/chat-template mismatch, dtype mismatch                                                                                               |

Then walk the causes below in the order listed. They are also numbered
roughly by frequency — when no symptom is obvious, start at #1 and work
down.

## 1. Q/K/V projection naming

MAX's `Attention` modules can use *fused* QKV (`qkv_proj.q.weight`,
`qkv_proj.k.weight`, `qkv_proj.v.weight`) or *unfused* (`q_proj.weight`,
`k_proj.weight`, `v_proj.weight`). HF checkpoints almost always use
unfused. Make sure your `weight_adapters.py` rename matches MAX's actual
module naming, not what you assume it is.

To verify: import the attention class you used, instantiate it, print
`state_dict().keys()`. The keys are the names your adapter must produce.

**Symptom:** garbage output at the very first token, often a low-id token
unrelated to the prompt. Random Q/K/V weights (silently left random
because `strict=False` let the load "succeed") project the embedding into
nonsense.

## 2. Missing GQA `repeat_kv`

For models where `num_key_value_heads < num_attention_heads`, K and V
must be repeated `num_attention_heads / num_key_value_heads` times before
the attention dot product. MAX's stock `AttentionWithRope` handles this
when given the right `num_key_value_heads`; custom attention modules
must do it manually.

**Symptom:** divergence starts at the very first attention block, post-attn
cos_sim around 0.1–0.5.

## 3. RoPE style mismatch (split-half vs. interleaved)

There are two RoPE conventions in the HF zoo:

- **Split-half** (GPT-J style): rotates
  `[x_first_half, x_second_half]` → `[-x_second_half, x_first_half]`.
  Used by Llama, Qwen, Mistral, OLMo, OLMoE, Gemma, Phi, and most
  modern decoders.
- **Adjacent-pair** (NeoX style): rotates
  `[x0, x1, x2, x3, ...]` → `[-x1, x0, -x3, x2, ...]`. Used by GPT-NeoX,
  helium, ChatGLM/GLM-4, ERNIE 4.5, and a smaller set of models.

The shapes are identical; the values are different. The bug is silent in
the scaffold-stage check (no crash, no error) but produces wrong
attention outputs from the first layer.

**Critical: the HF function name `rotate_half` is overloaded.** Models in
both groups call their RoPE rotation function `rotate_half`, but the
*implementations differ*. Always **inspect the implementation** in
`modeling_<type>.py` — don't trust the name. Look for the indexing:

```python
# Split-half (Llama, OLMoE, etc.):
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]

# Adjacent-pair (helium, ChatGLM, etc.):
x1 = x[..., 0::2]
x2 = x[..., 1::2]
```

In MAX, the `interleaved=` argument to `RotaryEmbedding` selects between
them. **`RotaryEmbedding(...)` defaults to `interleaved=True` (NeoX
adjacent-pair).** Pass `interleaved=False` for split-half models. Donor
archs that work (llama3, qwen3) read this from
`config.interleaved_rope_weights`, which is `False` for safetensors and
only `True` for the (rare) GGUF + rope_type="normal" combination.

**Adjacent-pair models** can also be served via `interleaved=False` if
the weight adapter pre-permutes Q/K weights into split-half order — see
a weight adapter that pre-permutes Q/K weights from adjacent-pair into
split-half order before load (search in-tree ports for
`_permute_interleaved_to_split_half`). That's the right move when the model has
*partial* adjacent-pair RoPE (some channels rotated, others not) and you
want to reuse the `Llama3RotaryEmbedding` codepath.

**Symptom A (text generation):** prefill produces a sensible first token
or two; decode collapses into single-word repetition or fragment loops
around token 8-12. The phase error is near-zero at position 0 and grows
with position, so short decodes hide it and 16+-token decodes expose it.
Typical on dense decoders when ``interleaved`` is left at the MAX default.

**Symptom B (parity dump):** divergence at post-attn block 0, cos_sim
around 0.7–0.9 (close but visibly wrong).

## 4. Tied embeddings

If `config.tie_word_embeddings=True`, the LM head must use the *same
tensor object* as the embedding matrix, not just identical values. MAX
weight adapters handle this through aliasing; verify by checking that
`lm_head.weight` and `embed_tokens.weight` resolve to the same storage
after load.

**Symptom:** first-token output is plausible (embedding works), but as
generation proceeds, the model degrades — because untied lm_head sees
random initialization for any token id the embedding rarely sees.

## 5. Pre-norm vs. post-norm vs. peri-LN block layout

MAX's stock `TransformerBlock.__call__` is hard-coded to pre-norm Llama
order: `h = x + attn(norm(x)); out = h + mlp(norm(h))`. Two patterns
break this:

- **Pure post-norm** (EXAONE 4, some older models): norm is applied to the
  sublayer *output* before residual add. `h = x + norm(attn(x))`.
- **Peri-LN / dual norm** (Gemma 2, HyperCLOVAX with
  `use_post_norm=True`): pre-norm *and* an additional norm on the sublayer
  output before residual add.

Both require a custom block subclass with a rewritten `__call__`. See
[read-modeling-code.md](read-modeling-code.md) for the templates.

**Symptom:** divergence first appears at post-block-norm, not post-attn or
post-mlp. The "actual token" at coherence-check time is often the
*last input token* (model degenerates to identity on the first decode step).

## 6. Partial RoPE padding

The base `RotaryEmbedding` produces interleaved `[cos0, sin0, cos1, sin1, ...]`
frequency pairs, not split-half. When `partial_rotary_factor < 1.0`
(applies RoPE to only part of the head dim), the padding for the
unrotated section must be *interleaved identity pairs* `[1, 0, 1, 0, ...]`,
not split `[1, 1, ..., 0, 0, ...]`.

**Symptom:** divergence grows with sequence length. Short prompts may
pass; 200+ token decodes go off the rails.

## 7. Absolute positional embeddings (non-RoPE)

For models without RoPE (GPT-BigCode, OPT, BLOOM), positions are added
to the embedding. During autoregressive decoding, the position index must
come from `cache_lengths` in the KV cache inputs — not reset to 0 each
step.

**Symptom:** first token is fine; every subsequent token attends to
position 0 instead of its actual position, producing repetitive output.

## 8. `rope_theta` nested in `rope_parameters`

Newer Gemma 2, Helium, and some other configs put `rope_theta` inside a
`rope_parameters` dict instead of as a top-level field. Reading
`config.rope_theta` returns `None`, falling back to the default 10000.0,
which is wrong.

```python
theta = getattr(config, "rope_theta", None) or \
    config.rope_parameters.get("rope_theta", 10000.0)
```

**Symptom:** outputs look plausible at short lengths but degrade as
positions grow — RoPE error scales with position.

## 9. `ops.constant` missing `device=`

Every `ops.constant` inside a `Graph` needs `device=`. See
[pitfalls-graph.md](pitfalls-graph.md#opsconstant-requires-device).

**Symptom:** silent wrong values, divergence at the layer that uses the
constant.

## 10. Fused QKV in HF checkpoints

GPT-NeoX, BLOOM, and a few others ship a single fused `query_key_value`
weight. To load into MAX's separate Q/K/V projections, the weight
adapter must split:

```python
arr = checkpoint["...query_key_value.weight"].to_buffer().to_numpy()
q = arr[:hidden_size]
k = arr[hidden_size:2 * hidden_size]
v = arr[2 * hidden_size:]
```

For models where the fused layout interleaves heads
(`[Q0, K0, V0, Q1, K1, V1, ...]`), the split is more involved — read the HF
`forward` carefully.

## 11. Parallel residuals

Standard transformer blocks use sequential residuals (attn output goes
into the residual stream, then MLP reads from that stream). GPT-NeoX uses
*parallel residuals*: attention and MLP both read from the same `x`, and
their outputs are summed.

```text
out = x + attn(ln1(x)) + mlp(ln2(x))   # parallel
```

vs.

```text
h = x + attn(ln1(x)); out = h + mlp(ln2(h))   # sequential
```

Stock `TransformerBlock` only does sequential. Subclass with a custom
`__call__` matching the parallel-residual computation.

## 12. Non-gated MLP

MAX's stock `MLP` is gated (SwiGLU): `down(silu(gate(x)) * up(x))`. For
models with a plain two-layer MLP (`fc2(act(fc1(x)))`, common in
GPT-style, OPT, XGLM, BLOOM, BioGPT), you need a custom MLP class with
two linears and the right activation. Make sure the layer attribute names
match the HF weight keys (`fc1`/`fc2` or `dense_h_to_4h`/`dense_4h_to_h`).

## 13. MuP scalars

Models trained with Maximal Update Parametrization (MuP) — Granite,
Granite-MoE, HyperCLOVAX, and a growing set — depend on four scalar
multipliers:

- `embedding_multiplier` — scales the output of the embedding layer.
- `logits_scaling` — scales logits *after* `lm_head` in HF (divides them in
  some families). Not the same as a pre-head width divisor
  (`h / (hidden_size / dim_model_base)`); grep `lm_head(` in
  [read-modeling-code.md](read-modeling-code.md).
- `residual_multiplier` — scales sublayer outputs before residual add.
- `attention_multiplier` — replaces `1/sqrt(d)` as the attention scale.

Defaults are 1.0 (no-op), so non-MuP models ignore them. MuP models
default these to non-1.0 values, and *every one* must be threaded through
the layers. Miss any of them and outputs are garbage with no error.

In MAX, these are first-class kwargs on the stock `TransformerBlock` and
`Transformer`. In your `model_config.py`, pull them from the HF config
with defaults of `1.0`.

## 14. Activation function variants

`gelu`, `gelu_new`, `gelu_tanh`, `gelu_pytorch_tanh`, `geglu`,
`gelu_fast`, `silu`, `swish`. These are not all the same. Look up the
exact function in `transformers/activations.py` (`ACT2FN` dict) and
match it in MAX. Common gotchas:

- `gelu` is the exact erf-based GELU.
- `gelu_new` and `gelu_tanh` are the tanh approximation (same function,
  different name).
- `geglu` is the gated variant (used in T5 and some MLPs).

**Symptom:** mild divergence at post-mlp, every block. cos_sim around
0.95 — close, but not exact.

## 15. Norm variants

- **RMSNorm with `+1` scale** (Gemma 2): `x * weight + x` instead of
  `x * weight`. Equivalent to `x * (1 + weight)`.
- **Per-head QK-norm** (Olmo2, Qwen3, EXAONE 4): norm dim is `head_dim`,
  not `hidden_size`. Norm is applied after the per-head reshape, not
  before.
- **LayerNorm vs. RMSNorm** — most modern models use RMSNorm, but some
  embedders, encoders, and older decoders use LayerNorm (with both scale
  and bias).

## 16. Tokenizer / chat-template mismatch

If layer-by-layer outputs match but the *generated text* doesn't, the
model itself is fine. The issue is one of:

- Different special tokens (BOS, EOS) being added.
- Different chat-template wrapping for instruct models.
- Different tokenization (BPE merge differences, normalization).

To isolate: tokenize the same prompt with both HF and MAX tokenizers,
diff the token IDs. If they differ, the bug is in tokenizer setup, not
the model.

## 17. NoPE / skip-RoPE layers via identity `freqs_cis`

Some models (certain MoE and EXAONE-style hybrids) apply RoPE **only** on
sliding-window layers and skip rotation on full-attention layers. HF
branches in `Attention.forward`; MAX fused kernels (`rope_split_store_ragged`)
often have **no runtime skip flag**.

Common MAX pattern: build a second **identity** `freqs_cis` table
(cos slots = 1.0, sin slots = 0.0) and select real vs identity per layer
index from `config.layer_types[]`.

**Critical:** identity table packing must match how MAX stores `freqs_cis`
for your rotary class — not how HF names the RoPE style. `interleaved=True`
on `Llama3RotaryEmbedding` may mean the *rotation math* is GPT-J style
while storage is still `[cos_0, cos_1, …, sin_0, sin_1, …]`. If you
build identity as `[cos_0, sin_0, cos_1, sin_1, …]` against split-half
storage, full-attention layers get a bogus rotation → garbage or repeated
characters even when sliding layers look partially right.

**Verify before iterating MoE or router fixes:**

```python
# Dump first row of real freqs_cis from MAX rotary module vs your identity table
print(real_freqs_cis[0, :8])
print(identity_freqs_cis[0, :8])
```

**Symptom:** compiles and serves; sliding-window path partially plausible
then collapses to `&&&&` or punctuation loops after you add NoPE.

## 18. `ops.sum` shape

In MAX, `ops.sum([N, H, D], axis=-1)` returns `[N, H, 1]`, not `[N, H]`
like PyTorch. Squeeze after if you need the rank-2 shape.

## 19. Final logit softcap

Gemma 2 and a few others apply `softcap * tanh(logits / softcap)` to the
LM-head output. This sits between the linear and the sampling — a one-line
addition, but missing it produces logits that are systematically too large
on rare tokens, distorting greedy decoding.
