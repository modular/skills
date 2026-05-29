# Reading Hugging Face modeling code

The HF `modeling_<type>.py` file is the ground truth for what your MAX
implementation must compute. You read it while building the delta list,
while implementing the graph, and again during the divergence hunt.

## Locating the file

```python
import importlib
mod = importlib.import_module("transformers.models.<model_type>.modeling_<model_type>")
print(mod.__file__)
```

For models with `trust_remote_code=True`, the modeling code lives in the
Hugging Face cache under `~/.cache/huggingface/hub/<repo>/snapshots/...`.

## What to read, in order

Read the file top-down following the class hierarchy. For a typical causal
LM, the relevant classes are:

1. **`<ModelType>Config`** — already covered when you read `config.json`.
2. **`<ModelType>Attention`** — how Q/K/V are computed and combined.
3. **`<ModelType>MLP`** — the feedforward layer.
4. **`<ModelType>DecoderLayer`** — how attention + MLP + norms are wired.
5. **`<ModelType>Model`** — the embedding, layer stack, final norm.
6. **`<ModelType>ForCausalLM`** — the LM head and the loss.

You're looking at `__init__` (what layers exist) and `forward` (how they
compose). Skip everything else.

## What to look for in `__init__`

This is the easiest delta hunt. Compare the list of `self.xxx = …`
assignments to the equivalent class in the MAX architecture you're starting
from.

Common findings:

- **Extra norms** — `self.q_norm = nn.RMSNorm(...)`, `self.k_norm = ...`
  indicates QK-norm. Two extra norms inside attention; their dim is either
  `head_dim` (Olmo2-style, per-head) or `hidden_size` (full Q/K).
- **Extra projections** — `self.q_a_proj`, `self.q_b_proj`,
  `self.kv_a_proj_with_mqa`, `self.kv_b_proj` is MLA. Two-stage Q
  projection, two-stage KV projection with a latent in the middle.
- **Router** — `self.gate = nn.Linear(hidden, num_experts, bias=False)` is
  MoE. Followed by `self.experts = nn.ModuleList([...])`.
- **Side embeddings** — `self.alibi_bias`, `self.rotary_emb` at the
  attention level (vs. module level) signals positional encoding variants.

## What to look for in attention `forward`

This is where most architectures diverge. Read line by line:

1. **Q/K/V computation** — are they separate (`self.q_proj(x)`, ...) or
   fused (`self.qkv_proj(x).chunk(3)`)? The weight adapter needs to match.
2. **Shape reshaping** — `view(bsz, q_len, num_heads, head_dim)` is
   standard. If you see `view(bsz, q_len, num_heads, 2, head_dim // 2)`,
   that's interleaved RoPE.
3. **QK norm** — `q = self.q_norm(q)` before the dot product. Check the
   *dim* the norm acts on (head_dim is per-head, hidden_size is full).
4. **RoPE application** — `apply_rotary_pos_emb(q, k, cos, sin)`. Look
   at the rotary embedding class: does it use `rotate_half` (split-half)
   or some variant? Is RoPE applied to only part of the head
   (`q[..., :rope_dim]` vs. full `q`)?
5. **GQA repeat** — for `num_key_value_heads < num_attention_heads`,
   K and V get repeated. `repeat_kv(key_states, self.num_key_value_groups)`.
6. **Mask** — sliding window vs. causal vs. sink-token. Check whether the
   layer index conditionally selects between two masks.
7. **Softmax scale** — `attn_weights / math.sqrt(self.head_dim)` is
   default. Some models use a different scaling (`1 / d` for normalized
   attention, MuP `attention_multiplier`).
8. **Softcap** — `attn_weights = soft_cap * torch.tanh(attn_weights / soft_cap)`
   sits between the score and the softmax in Gemma 2.
9. **Output projection** — `self.o_proj(attn_output)`, usually
   unremarkable.

## What to look for in MLP `forward`

Most MLPs are one of three shapes:

- **Gated SwiGLU** (Llama-family): `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **Standard two-layer** (GPT-style): `fc2(act(fc1(x)))`. Activation may
  be `gelu`, `gelu_new`, `gelu_tanh`, `relu`.
- **MoE** (Mixtral/Qwen3-MoE/DeepSeek): `gate(x)` → `top_k` →
  per-expert SwiGLU → weighted sum. May include a "shared expert" applied
  to every token in addition to the routed experts.

Things that bite:

- Activation name. `gelu_new` and `gelu_tanh` and plain `gelu` are *not*
  the same function. Check `ACT2FN` in `transformers/activations.py`.
- Bias. Most modern models have `mlp_bias=False`, but GPT-NeoX, OPT, and
  some embedders have biases.
- Up/gate fusion. Some implementations fuse `gate_proj` and `up_proj` into
  a single `gate_up_proj` with `chunk(2)`; the weight adapter must match.

## What to look for in the block `forward`

The block class composes attention and MLP with norms and residual
connections. Three common patterns:

- **Pre-norm** (Llama, Mistral, Qwen, default):

  ```text
  h = x + attn(input_layernorm(x))
  out = h + mlp(post_attention_layernorm(h))
  ```

- **Post-norm** (some older models, EXAONE 4):

  ```text
  h = x + post_attention_layernorm(attn(x))
  out = h + post_feedforward_layernorm(mlp(h))
  ```

- **Peri-LN / dual norm** (Gemma 2, HyperCLOVAX): both pre-norm and an
  *additional* norm on the sublayer output before residual add:

  ```text
  h = x + post_norm1(attn(input_layernorm(x)))
  out = h + post_norm2(mlp(post_attention_layernorm(h)))
  ```

If your model is anything other than the first pattern, the stock MAX
`TransformerBlock` will not match — see "Choosing an edit strategy" below.

## What to look for in the final head

Read **`<ModelType>ForCausalLM.forward`**, not only `__init__`. The LM head
class is often a plain `nn.Linear`, but the forward pass may scale hidden
states before the matmul.

### Search for `lm_head(` and pre-head scaling

In the causal-LM `forward`, find the line that produces logits. Grep for
`lm_head(` and read the expression passed in:

```bash
pixi run rg -n 'lm_head\(' modeling_<type>.py
```

Flag anything that is not a bare `hidden_states` (or `outputs[0]`):

- **Width divisor** — `self.lm_head(h / (hidden_size / dim_model_base))`,
  `h / self.scale_width`, or `h * (dim_model_base / hidden_size)`. Common on
  MiniCPM-family and some Cohere-style configs. `dim_model_base` is often
  smaller than `hidden_size`; missing the divisor makes logits wrong with no
  load error.
- **MuP `logits_scaling`** — multiply or divide logits *after* `lm_head` (see
  [divergences.md §13](divergences.md#13-mup-scalars)). Do not confuse with a
  pre-head width divisor; check order in HF.
- **Final logit softcap** — Gemma 2: `softcap * tanh(logits / softcap)` after
  the linear.

Record the exact formula in your delta list (scalar name, config keys, and
whether scaling happens before or after `lm_head`). In MAX, mirror that
order in `<slug>.py` — usually `ops.mul` / `ops.div` on the last hidden
state before the output `Linear`, not only a post-hoc logits tweak.

### Other head variants

- **LM head** — `self.lm_head = nn.Linear(hidden, vocab, bias=False)`.
  If `config.tie_word_embeddings=True`, the LM head reuses the embedding
  matrix.
- **Multi-step head** — some models apply additional layers between the
  final block and the LM head (`LayerNorm → Linear → activation →
  Linear(vocab)`). The MAX template likely ends with a single Linear; you
  need to add the extra layers.

## Choosing an edit strategy

Given your delta list, pick one of three approaches:

| What differs                                             | Strategy                                         |
|----------------------------------------------------------|--------------------------------------------------|
| Only weight names; computation identical                 | Edit `weight_adapters.py` only                   |
| One sublayer differs (custom MLP, QK-norm)               | Subclass the layer, override one method          |
| One attention variant (sliding window, MLA, softcap)     | Subclass `Attention`, override `__call__`        |
| Block layout differs (post-norm, peri-LN)                | Subclass `TransformerBlock`, override `__call__` |
| Multi-step head                                          | Subclass the top-level model, override the head  |
| Attention is fundamentally new (recurrence, state-space) | Write from scratch with `max.nn` primitives      |
| MoE routing differs from existing MAX MoE archs          | Write from scratch                               |

Prefer subclassing. Every layer you write from scratch is a layer you can
get wrong; every layer you inherit is a layer that already passed someone
else's parity check.

## Inheritance traps

HF modeling code often inherits across model families in non-obvious ways.
Before you conclude "this is just Llama with renamed fields":

1. Read the *full* MRO of the class. `print(MyClass.__mro__)`.
2. Check whether any parent overrides `forward` or `__init__` in ways your
   class inherits silently.
3. Look for `if config.use_xxx:` branches in parent classes that your model
   activates via config.

The class hierarchy is where the easter eggs live.
