# Propose a plan and accept a veto

Before any code, write a short paragraph stating what you'd do by default,
then wait for the user to confirm or veto. You know more than the user
told you at this point (you read the config), so don't ask blank
questions. State a default and let them push back.

The paragraph should cover each of these, derived from what you've
already read:

- **Missing ModuleV3 pieces.** Name any piece the model needs that
  ModuleV3 doesn't have yet (a layer, a distributed primitive, a quantized
  path), and whether you'll add it to `max.nn` or need the
  user to decide.
- **Distribution shape.** Single-GPU dense, multi-GPU tensor-parallel,
  multi-GPU DP+EP (MoE), or text-only port of a multimodal wrapper. The
  rough rule: estimate `num_params × bytes_per_param` (with the MoE
  factor when applicable) and compare to one GPU's HBM minus cushion
  for KV cache + activations. If it doesn't fit single-GPU, the port
  shards over a `DeviceMesh` and your donor is a mesh-sharded ModuleV3
  architecture. See [distributed-transformer.md](distributed-transformer.md).
- **Donor.** The ModuleV3 architecture you'll scaffold from, and the
  attention, MLP, and distribution shape that make it the closest match
  (see [map-to-max.md](map-to-max.md)).
- **Quantization variants in scope.** BF16 only, or also FP8 / NVFP4 /
  GPTQ. Check the HF org for sibling repos. For any non-BF16, pre-flight
  the MAX kernel walls in [recognize-walls.md](recognize-walls.md).
  Flag a wall up front, before building the plumbing.
- **Validation depth.** Generation smoke only, GSM8K-style generation
  eval, lm-eval-harness loglikelihood (hellaswag/mmlu), or formal HF
  logit parity. See [validation-tiers.md](validation-tiers.md) for the
  tiers.
- **Hardware target.** Which machine(s) you'll serve on. Required for
  multi-GPU. Confirm capacity before downloading large checkpoints.

Write the paragraph as a proposal with defaults the user can accept or
change. One example shape:

> `org/LargeMoE-70B-bf16`, a MoE causal LM, text-only (vision wrapper
> dropped). I'll scaffold from `deepseekV3_modulev3`'s mesh-sharded MoE
> pattern on 4 GPUs (TP attention, EP experts) and register BF16 only for
> now. The FP8 sibling repo exists, but MoE quant has a known MAX gap. I'll
> validate with greedy generation + GSM8K + HellaSwag. Go?

The user can say "go" and they're done. Or they can veto one axis ("just
BF16 for now") in a few words, with no form. Proceed only after they confirm.

If the user already specified any of these in their initial request,
take it as given. Don't re-ask.

After they confirm, state the routing decision in one sentence
("I'll start from `deepseekV3_modulev3` on a 4-GPU mesh because the BF16
weights need ~140 GB HBM"). That gives the user one more chance to catch
wrong routing before files get rewritten.
