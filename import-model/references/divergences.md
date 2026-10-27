# Common divergence causes (indexed by symptom)

When the layer-by-layer divergence hunt shows divergence, the bug is almost
always one of the patterns below.

## Quick index: symptom → candidates

Match what you're seeing to a row, then read every listed cause, even
after one looks plausible. Several causes produce the same
symptom, so the first plausible match can be the wrong one.

| What you observe                                                            | Most likely causes (in order)                                                                                                                      |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Gibberish at token 0, low-id token unrelated to prompt                      | #1 Q/K/V naming, #10 fused QKV, #4 tied embeddings, #15 norm variant                                                                               |
| Repeated same character/token (`&&&&`, single punct loop)                   | #17 NoPE identity freqs_cis layout, #3 RoPE packing, #1 weight map                                                                                 |
| Mild divergence at every MLP post-output (cos_sim ~0.95)                    | #14 activation variant, #12 non-gated MLP, #13 MuP scalars                                                                                         |
| Divergence at post-attn block 0, cos_sim 0.1–0.5                            | #2 GQA repeat_kv, #1 Q/K/V naming                                                                                                                  |
| Divergence at post-attn block 0, cos_sim 0.7–0.9                            | #3 RoPE split-half vs interleaved, #15 QK-norm dim, #19 final softcap                                                                              |
| First token plausible, output degrades over generation                      | #4 tied embeddings, #6 partial-rotary padding, #8 nested rope_theta                                                                                |
| Repetition / position-0 loop after first token                              | #7 absolute positional embeddings, #6 partial-rotary padding                                                                                       |
| Divergence grows with sequence length, short prompts fine                   | #6 partial-rotary padding, #8 nested rope_theta, #3 RoPE style                                                                                     |
| Prefill ok, decode collapses to repetition/word-fragment loop by token 8-12 | #3 RoPE style: the `RotaryEmbedding(interleaved=True)` default mismatches HF `rotate_half`, so pass `interleaved=False`                            |
| Divergence at post-block-norm but both sublayers ok                         | #5 pre/post/peri-norm layout, #13 residual_multiplier (MuP)                                                                                        |
| `actual_token == last input token` on first decode                          | #5 post-norm degenerating to identity                                                                                                              |
| All-NaN logits from HF reference                                            | [pitfalls-serving.md](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan) (`trust_remote_code` + `.to("cuda")` device mismatch) |
| Layer outputs match but generated text drifts                               | #16 tokenizer/chat-template mismatch, dtype mismatch                                                                                               |

Then walk the causes below in the order listed. They are also numbered
roughly by frequency. When no symptom is obvious, start at #1 and work
down.

## 1. Q/K/V projection naming

MAX attention modules can hold *fused* QKV (one `qkv_proj.weight`) or
*unfused* projections (`q_proj.weight`, `k_proj.weight`, `v_proj.weight`,
concatenated in `forward()` as `olmo3/layers/attention.py` does). HF
checkpoints almost always ship unfused. Make sure your `weight_adapters.py`
rename matches the module's actual parameter names, not what you assume
they are.

To verify: build the root module under `F.lazy()` and print
`[name for name, _ in model.parameters]`. Those names are what your adapter
must produce.

**Symptom:** `compile()` raises `KeyError` for the unmatched parameter. If
the adapter maps the tensors to a name that exists but holds the wrong
projection, the first token is garbage, often a low-id token unrelated to
the prompt.

## 2. Missing GQA `repeat_kv`

For models where `num_key_value_heads < num_attention_heads`, K and V
must be repeated `num_attention_heads / num_key_value_heads` times before
the attention dot product. The ragged attention kernels in
`max.nn.common_layers.functional_kernels`
(`flash_attention_ragged`) handle this from the KV head count in
`kv_params`. Custom attention math outside those kernels must repeat K and
V itself.

**Symptom:** divergence starts at the very first attention block, post-attn
cos_sim around 0.1–0.5.

## 3. RoPE style mismatch (split-half vs. interleaved)

There are two RoPE conventions in the HF zoo:

- **Split-half** (GPT-NeoX style): rotates
  `[x_first_half, x_second_half]` → `[-x_second_half, x_first_half]`.
  Used by GPT-NeoX, Llama, Qwen, Mistral, OLMo, OLMoE, Gemma, Phi, and
  most modern decoders.
- **Adjacent-pair** (GPT-J style): rotates
  `[x0, x1, x2, x3, ...]` → `[-x1, x0, -x3, x2, ...]`. Used by GPT-J,
  Helium, ChatGLM/GLM-4, ERNIE 4.5, and a smaller set of models.

Both conventions produce the same shapes and different values. The
scaffold-stage check passes with no crash or error, and attention outputs
are wrong from the first layer.

**The HF function name `rotate_half` is overloaded.** Models in both
groups call their RoPE rotation function `rotate_half`, and the
implementations differ. Read the implementation in `modeling_<type>.py`
and look for the indexing:

```python
# Split-half (Llama, OLMoE, etc.):
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]

# Adjacent-pair (helium, ChatGLM, etc.):
x1 = x[..., 0::2]
x2 = x[..., 1::2]
```

In MAX, the `interleaved=` argument to `RotaryEmbedding` selects between
them. **`RotaryEmbedding(...)` defaults to `interleaved=True` (GPT-J
adjacent-pair).** Pass `interleaved=False` for split-half models.
`llama3_modulev3` passes `config.interleaved_rope_weights` as `interleaved`.
Its config sets the field to `False` for safetensors checkpoints and to
`True` only for GGUF with `rope_type="normal"`. For an adjacent-pair
safetensors model on that donor, set `cfg.interleaved_rope_weights = True`
in the port's `from_donor()`.

**Adjacent-pair models** can also be served with `interleaved=False` if
the weight adapter pre-permutes Q/K weights into split-half order before
load. For a related Q/K weight permutation, see `_build_partial_rope_perm`
in `step3p5/weight_adapters.py`. That's the right move when the model
has *partial* adjacent-pair RoPE (some channels rotated, others not) and you
want to reuse the `Llama3RotaryEmbedding` codepath.

**Symptom A (text generation):** the output degrades with position. In
milder cases, prefill produces a sensible first token or two, then decode
collapses into single-word repetition or fragment loops around token 8-12.
In sharper cases the first token is already wrong: ERNIE 4.5 served with
split-half RoPE prints `' 100000000…'` from token 1. Either way,
`compare_layers.py`'s prefill check reports `DIVERGED`, while the
multi-position top-1 probe can still match at short prefixes.

**Symptom B (parity dump):** divergence at post-attn block 0, cos_sim
around 0.7–0.9 (close but visibly wrong).

## 4. Tied embeddings

If `config.tie_word_embeddings=True`, every logit comes from the embedding
weight. The ModuleV3 pattern builds no `lm_head` parameter: `forward()`
computes the logits as `h @ self.embed_tokens.weight.T`, and the weight
adapter drops the checkpoint's `lm_head.weight` (see
[pitfalls-weights.md](pitfalls-weights.md)).

**Symptom:** first-token output is plausible (embedding works), but the
model degrades as generation proceeds. A port with its own `lm_head`
parameter loads random initialization for any token id the embedding
rarely sees.

## 5. Pre-norm vs. post-norm vs. peri-LN block layout

A Llama-style donor block (`LlamaTransformerBlock.forward()` in
`llama3_modulev3`) is hard-coded to pre-norm order:
`h = x + attn(norm(x)); out = h + mlp(norm(h))`. These patterns break it:

- **Pure post-norm** (EXAONE 4, some older models): norm is applied to the
  sublayer *output* before residual add. `h = x + norm(attn(x))`.
- **Peri-LN / dual norm** (Gemma 2, HyperCLOVAX with
  `use_post_norm=True`): pre-norm *and* an additional norm on the sublayer
  output before residual add.

Both require a custom block with a rewritten `forward()`. See
[read-modeling-code.md](read-modeling-code.md#choosing-an-edit-strategy)
for the edit strategy.

**Symptom:** divergence first appears at post-block-norm, not post-attn or
post-mlp. The "actual token" at coherence-check time is often the
*last input token* (model degenerates to identity on the first decode step).

## 6. Partial RoPE padding

The base `RotaryEmbedding` produces interleaved
`[cos0, sin0, cos1, sin1, ...]` frequency pairs, not split-half. When
`partial_rotary_factor < 1.0`, RoPE applies to only part of the head dim.
The padding for the unrotated section must then be *interleaved identity
pairs* `[1, 0, 1, 0, ...]`, not split `[1, 1, ..., 0, 0, ...]`.

**Symptom:** divergence grows with sequence length. Short prompts may
pass, but 200+ token decodes produce wrong output.

## 7. Absolute positional embeddings (non-RoPE)

For models without RoPE (GPT-BigCode, OPT, BLOOM), positions are added
to the embedding. During autoregressive decoding, the position index must
come from `cache_lengths` in the KV cache inputs. Don't reset it to 0 each
step.

**Symptom:** the first token is fine. Every later token attends to
position 0, not its actual position, producing repetitive output.

## 8. `rope_theta` nested in `rope_parameters`

Newer Gemma 2, Helium, and some other configs nest `rope_theta` inside a
`rope_parameters` dict. Reading the top-level `config.rope_theta` returns
`None`, falling back to the default 10000.0,
which is wrong.

```python
theta = getattr(config, "rope_theta", None) or config.rope_parameters.get(
    "rope_theta", 10000.0
)
```

**Symptom:** outputs look plausible at short lengths but degrade as
positions grow, because RoPE error scales with position.

## 9. `F.constant` created with default dtype or device

`F.constant(value)` with no `dtype` and `device` lands on the accelerator as
`bfloat16`. See
[pitfalls-graph.md](pitfalls-graph.md#fconstant-defaults-to-the-accelerator-and-bfloat16).

**Symptom:** small precision errors (an epsilon or scale rounded to
bfloat16), or a device mismatch at the op that consumes the constant.

## 10. Fused QKV in HF checkpoints

GPT-NeoX, BLOOM, and a few others ship a single fused `query_key_value`
weight. To load into MAX's separate Q/K/V projections, the weight
adapter must split:

```python
import torch

qkv = checkpoint["...query_key_value.weight"].data()
arr = torch.from_dlpack(qkv)  # NumPy can't read bfloat16
q = arr[:hidden_size]
k = arr[hidden_size : 2 * hidden_size]
v = arr[2 * hidden_size :]
```

For models where the fused layout interleaves heads
(`[Q0, K0, V0, Q1, K1, V1, ...]`), the split is more involved. Derive the
head layout from the HF `forward`.

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

A Llama-style donor block only does sequential. Write a custom block
whose `forward()` matches the parallel-residual computation.

## 12. Non-gated MLP

MAX's stock `MLP` is gated (SwiGLU): `down(silu(gate(x)) * up(x))`. For
models with a plain two-layer MLP (`fc2(act(fc1(x)))`, common in
GPT-style, OPT, XGLM, BLOOM, BioGPT), you need a custom MLP class with
two linears and the right activation. Make sure the layer attribute names
match the HF weight keys (`fc1`/`fc2` or `dense_h_to_4h`/`dense_4h_to_h`).

## 13. MuP scalars

Models trained with Maximal Update Parametrization (MuP), such as Granite,
Granite-MoE, and HyperCLOVAX, depend on these scalar multipliers:

- `embedding_multiplier`: scales the output of the embedding layer.
- `logits_scaling`: scales logits *after* `lm_head` in HF (divides them in
  some families). A pre-head width divisor
  (`h / (hidden_size / dim_model_base)`) is a different scalar. Grep
  `lm_head(` in [read-modeling-code.md](read-modeling-code.md).
- `residual_multiplier`: scales sublayer outputs before residual add.
- `attention_multiplier`: replaces `1/sqrt(d)` as the attention scale.

Defaults are 1.0 (no-op), so non-MuP models ignore them. MuP models
default these to non-1.0 values, and each one must be threaded through
the layers. A missing multiplier produces wrong outputs without an error.

In MAX, `llama3_modulev3`'s config reads `embedding_multiplier`,
`residual_multiplier`, `attention_multiplier`, and `logits_scaling`, and
its modules apply them. In your `model_config.py`, pull them from the HF
config with defaults of `1.0`.

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
0.95.

## 15. Norm variants

- **RMSNorm with `+1` scale** (Gemma 2): `x * weight + x`, where standard
  RMSNorm uses `x * weight`. Equivalent to `x * (1 + weight)`.
- **Per-head QK-norm** (Olmo2, Qwen3, EXAONE 4): norm dim is `head_dim`,
  not `hidden_size`. Norm is applied after the per-head reshape, not
  before.
- **LayerNorm vs. RMSNorm**: most modern models use RMSNorm, but some
  embedders, encoders, and older decoders use LayerNorm (with both scale
  and bias).

## 16. Tokenizer / chat-template mismatch

If layer-by-layer outputs match but the *generated text* doesn't, the
model itself is fine. The issue is one of:

- Different special tokens (BOS, EOS) being added.
- Different chat-template wrapping for instruct models.
- Different tokenization (BPE merge differences, normalization).

To isolate it, tokenize the same prompt with both HF and MAX tokenizers
and diff the token IDs. If they differ, the bug is in tokenizer setup, not
the model.

## 17. NoPE / skip-RoPE layers via identity `freqs_cis`

Some models skip RoPE on a subset of layers (NoPE layers). The config
says which, and the field differs by family:

- `no_rope_layers` (SmolLM3): a per-layer list, `1` for RoPE and `0` for
  NoPE. `no_rope_layer_interval` generates the same list when the list is
  absent.
- `layer_types` (certain MoE and EXAONE-style hybrids): RoPE on
  `"sliding_attention"` layers only, NoPE on `"full_attention"` layers.

Read HF's `Attention.forward` for the condition it actually checks. A model
can carry both fields: SmolLM3's `layer_types` are all `"full_attention"`,
and its NoPE layers come from `no_rope_layers`.

HF branches in `Attention.forward`, but MAX fused kernels
(`rope_split_store_ragged`) often have **no runtime skip flag**. Common MAX
pattern: build a second **identity** `freqs_cis` table (cos slots = 1.0,
sin slots = 0.0) and give each NoPE layer the identity table. On
`llama3_modulev3`, each layer's attention reads its `rope` module, so a
NoPE layer can get a rotary module whose `freqs_cis` is the identity
table.

The identity table must match how MAX stores `freqs_cis`,
not how HF names the RoPE style. The rotary classes in
`max.nn.common_layers.rotary_embedding` store `freqs_cis` as
interleaved pairs, `[cos_0, sin_0, cos_1, sin_1, …]`, whatever the
`interleaved` flag says. The flag only selects how Q and K are rotated. So
build the identity table as `[1, 0, 1, 0, …]`. A split-half identity,
`[1, 1, …, 0, 0, …]`, gives full-attention layers a wrong rotation. The
output is garbage or repeated characters, even when sliding layers look
partially right.

**Verify before iterating MoE or router fixes:**

```python
# Dump first row of real freqs_cis from MAX rotary module vs your identity table
print(real_freqs_cis[0, :8])
print(identity_freqs_cis[0, :8])
```

**Symptom:** depends on the model. With a wrong identity layout, the
sliding-window path looks partially plausible, then collapses to `&&&&` or
punctuation loops. With NoPE left out entirely, the output can stay fluent
and on topic: SmolLM3 served with RoPE on every layer answers sensibly and
drifts from HF after about 12 tokens. Only the logprob check catches it
(`logits_prefill` reported `rel_diff=0.2404`), so a coherent answer doesn't
rule NoPE out.

## 18. `F.sum` shape

In MAX, `F.sum(x, axis=-1)` on a `[N, H, D]` tensor returns `[N, H, 1]`,
where PyTorch returns `[N, H]`. Squeeze after if you need the rank-2
shape.

## 19. Final logit softcap

Gemma 2 and a few others apply `softcap * tanh(logits / softcap)` to the
LM-head output. This sits between the linear layer and sampling. The
addition is one line. Without it, logits on rare tokens are too large,
which distorts greedy decoding.
