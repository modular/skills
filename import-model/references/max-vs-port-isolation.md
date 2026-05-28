# Is the bug in MAX or in my port?

When accuracy regresses, a feature path misbehaves, or a benchmark drops to
random, the first instinct is to debug your port. Sometimes the bug is in
MAX serve or a shared kernel — not your graph. Run the same probe on a
stock MAX-native model with **identical serve flags** to decide which side
to debug.

## When to run this

- Generation works; loglikelihood (echo path) fails
- Accuracy passes at 0-shot, fails at few-shot
- A serve flag path (batched prefill, echo, KV eviction) misbehaves
- Chat-formatted or few-shot prompts break while plain prompts work

If the failure tracks a **serve flag or endpoint format**, suspect MAX first.
If it tracks **attention shape, MoE routing, or RoPE wiring**, debug the port
(see [layer-by-layer-debugging.md](layer-by-layer-debugging.md)).

## The control test

1. **Pick a small native arch** in the same family:
   - Dense decoder: ``Qwen/Qwen3-1.7B``
   - MoE: ``Qwen/Qwen3-30B-A3B`` or ``mistralai/Mixtral-8x7B``
   - MLA: ``deepseek-ai/DeepSeek-V2-Lite``
2. **Serve with the same flags** as your port: ``--devices``,
   ``--max-batch-size``, ``--max-length``, ``--enable-echo``,
   ``--quantization-encoding``, etc.
3. **Run the same probe** (same lm-eval task, ``num_fewshot``, ``curl``
   payload).
4. **Compare:**
   - Native arch **fails the same way** → MAX bug; file upstream with a
     minimal repro. Stop tuning your port for this symptom.
   - Native arch **passes** → port bug; continue locally.

## Worked example: few-shot loglikelihood collapse

A large MoE port: generation and GSM8K looked fine; HellaSwag 0-shot was
reasonable; 5-shot and 10-shot were near random (~0.25–0.31 on 4-choice).

Tempting port-side theories: wrong NoPE ``freqs_cis``, sliding-window bug,
router wiring. Tweaking those moved scores slightly — not a fix.

Control: ``Qwen/Qwen3-1.7B`` with the **same** ``--enable-echo`` and
``--max-batch-size``:

| Task                      | Your port | Qwen3-1.7B (native) |
|---------------------------|-----------|---------------------|
| HellaSwag 0-shot acc_norm | 0.76      | 0.49                |
| HellaSwag 5-shot acc_norm | 0.31      | **0.27**            |

The native model showed the same 0-shot → 5-shot drop. Diagnosis: few-shot
loglikelihood through ``--enable-echo`` / ``compute_log_probabilities`` —
not the custom graph. Five minutes on the control test; hours saved on the
port.

## What this isolates — and what it doesn't

**Isolated:** feature paths any model exercises the same way (echo,
batched prefill, certain quantization formats).

**Less clear:** size-dependent bugs — use a native arch in the same class
(MoE control for MoE ports).

**Not isolated:** attention, MoE routing, block wiring — use
[divergences.md](divergences.md) and layer taps.

## Known-good controls

| Your port's class              | Control                                              |
|--------------------------------|------------------------------------------------------|
| Dense decoder, RoPE, RMSNorm   | ``Qwen/Qwen3-1.7B``                                  |
| Sliding window                 | ``mistralai/Mistral-7B-v0.3``                        |
| MoE, top-k routing             | ``Qwen/Qwen3-30B-A3B`` or ``mistralai/Mixtral-8x7B`` |
| MLA + MoE                      | ``deepseek-ai/DeepSeek-V2-Lite``                     |
| Multimodal wrapper (text-only) | ``google/gemma-3-12b-it``                            |

Default to ``Qwen3-1.7B`` when unsure — fastest cold compile.

## Escalate after the control fails

1. Minimize repro (smallest flags + probe that still fails).
2. Capture ``pixi run max --version``.
3. Report that the failure reproduces on **both** your port and a stock
   native arch — that makes it a MAX issue, not a bug in your port graph.
4. Note the gap in bring-up notes; ship what passes; do not keep debugging
   the port for that symptom.
