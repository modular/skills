# Recognizing hard-to-port architectures

Some models can't be ported to MAX with the public surface alone, at least
not without significant infrastructure work. The patterns below are signals
that you'll need to either wait for upstream MAX support or contribute a
new primitive.

The list of "blocked" models isn't exhaustive. It describes the signals of a
model that needs infrastructure work before a port can start.

## Custom CUDA kernels in the modeling code

If `modeling_<type>.py` imports from a CUDA extension package (e.g.
`from flash_attn import flash_attn_func`), the upstream model assumes a
specific kernel that MAX may or may not have.

- **MAX has flash attention** for standard causal masks. Sliding-window
  and most masked-attention variants are supported.
- **MAX does not have** custom variants like Lightning Indexer (DeepSeek
  V3-Flash), Native Sparse Attention, and similar research-grade
  attention kernels.

Signal: the paper section is titled something like "Our novel attention
mechanism" and a single CUDA file in the repo implements it.

### MoE quantization compatibility (pre-flight before scoping FP8/NVFP4)

MAX's quantized routed-expert kernels support only specific scaling
schemes. Check the HF checkpoint's ``quantization_config`` before
committing to any MoE quant variant. The ModuleV3 path (``QuantizedMoE``
in ``deepseekV3_modulev3/layers/quant_moe.py``) accepts block-scaled FP8
and NVFP4 only, and raises ``ValueError`` for any other format. The table
covers the graph-API path (``max.nn.legacy.moe.MoEQuantized``), where an
unsupported scheme fails a ``block_size`` assertion at first serve.

| HF format (compressed-tensors / native)  | Weight scaling                              | Activation scaling                       | MAX 26.4 `MoEQuantized`?                                                         |
|------------------------------------------|---------------------------------------------|------------------------------------------|----------------------------------------------------------------------------------|
| **Block-scaled FP8** (DeepSeek-V3 style) | ``block_size=(128, 128)``                   | Block-scaled per input                   | ✅ Supported via the non-FP4 branch (asserts ``(128, 128)``)                     |
| **NVFP4** (``nvfp4-pack-quantized``)     | ``group_size=16`` block                     | Block, with per-projection global scales | ✅ Supported via ``is_fp4`` branch                                               |
| **Compressed-tensors per-channel FP8**   | ``strategy="channel"`` (per-output-channel) | ``strategy="token"``, ``dynamic=true``   | ❌ Not supported. ``input_scale.block_size`` is ``None``, so the assertion fires |
| **FBGEMM FP8**                           | row-wise tensor scale                       | per-tensor static scale                  | Limited. Check the parser logic                                                  |
| **MXFP4** (``quark`` / GPT-OSS style)    | block-scaled                                | block-scaled                             | ✅ Via the FP4 branch                                                            |

How to check in advance:

```bash
pixi run python -c "
from huggingface_hub import hf_hub_download
import json
cfg = json.loads(open(hf_hub_download(repo_id='<HF_ID>', filename='config.json')).read())
qc = cfg.get('quantization_config') or cfg.get('text_config', {}).get('quantization_config') or {}
grp = (qc.get('config_groups') or {}).get('group_0', {})
print('format:', qc.get('format'))
print('weights:', {k: grp.get('weights', {}).get(k) for k in ('strategy','group_size','num_bits','type')})
print('inputs :', {k: grp.get('input_activations', {}).get(k) for k in ('strategy','group_size','num_bits','dynamic')})
"
```

If the output is ``weights.strategy='channel'`` and
``inputs.strategy='token'`` with ``dynamic=True``, the variant is in
the unsupported column. Pick one of these options:

1. **Register the variant in ``arch.py`` but defer serve.** The
   weight adapter, ignore-list normalization, and selective per-layer
   ``quant_config`` plumbing can all land. The wall is in MAX
   serve's quantized MoE kernels. Document the gap, and ship when MAX
   adds support upstream.
2. **Skip this variant for now.** If the BF16 variant fits your
   hardware, serve that. Flag the smaller variant as "register only,
   blocked on upstream."

Either choice fits bring-up scope at the plan-and-veto step. Run the check
at config-read time, before you build the FP8 plumbing, so an unsupported
scheme surfaces before that work.

### Attention biases in NVFP4 checkpoints

Some NVFP4 quantization recipes absorb the per-output-channel
quantization residual into a separate ``bias`` term on each projection.
That includes attention ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``
and the router ``gate``. If the HF ``config.json`` says
``attention_bias: false`` but the checkpoint ships
``self_attn.o_proj.bias`` tensors, the port needs attention linears
with a bias (``Linear(..., bias=True)``) for that variant only. Detect
from the state dict (presence of ``.bias`` keys) and override the
config flag when the module is built. The bias is variant-specific, not
model-specific. The BF16 release of the same model won't have biases.

## ALiBi positional encoding

ALiBi (Attention with Linear Biases, used in BLOOM, MPT, and some older
models) adds a per-head linear bias to attention scores and uses no RoPE.
MAX 26.x removed the `CAUSAL_ALIBI_MASK` variant. The public surface has no
first-class ALiBi path through `flash_attention_ragged`.

Signal: `config.json::position_embedding_type == "alibi"`, or the
attention `forward` reads `self.alibi_bias` and adds it to scores.

If you need an ALiBi model, prefer one of its RoPE-converted descendants
or wait for upstream MAX support.

## Models requiring custom tokenization or chat templates

If the tokenizer isn't a standard BPE/SentencePiece/Unigram tokenizer
loadable via `AutoTokenizer`, you'll spend significant time on tokenizer
wrapping before any model work matters.

Signal: `tokenizer_config.json` references a non-standard tokenizer
class, or the model card explicitly says "use our custom inference
script."

## Diffusion / multi-step samplers

Image diffusion (Stable Diffusion variants, FLUX), video diffusion, and
similar models use multi-step sampling loops outside the standard causal
LM template. MAX supports some of these as dedicated arch slugs (check
the architectures list), but porting a new diffusion model is not in the
same workflow as a causal LM port.

Signal: the model has a UNet, a VAE, and a scheduler, and no decoder
stack.

## Multimodal mixture-of-experts with non-shared routing

Standard MoE (each MoE block has its own router) is well-supported in
MAX (`gpt_oss_modulev3`, `deepseekV3_modulev3`). Variants where the router is
shared across multiple layers, or where experts are themselves
attention layers, are not.

Signal: paper mentions "shared router," "hyper-MoE," or "expert
attention."

## What to do when you hit a wall

1. Check whether a closely-related model (same family, simpler variant)
   is portable, and port that instead. Often the *first* model in a
   family is hard, and later variants reuse infra.
2. If the architecture is novel and you need it ported, the
   path is to upstream a new MAX primitive. That's a different project
   from this skill's workflow.

Don't work around a wall with an approximation, such as a substitute
kernel that computes something close to the original. A port built that way
can return wrong results without an error. Leave the model unported until MAX
supports the missing piece.
