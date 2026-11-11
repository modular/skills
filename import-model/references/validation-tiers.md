# Validation tiers

Run the model end-to-end with pretrained weights, then run HF on the same
prompt with greedy sampling. On the MAX side, serve the encoding the model's
released weights use (most ship bfloat16). Load the HF reference in float32
when it fits: a bfloat16 reference adds its own rounding, and a correct
bfloat16 port can drift from it within a few tokens.

Send the model card's prompt. For an instruction-tuned model, send it through
`/v1/chat/completions` on MAX and `apply_chat_template()` on HF, so both apply
the same template. For a base model, send a plain prompt through
`/v1/completions`, as below. The `model` field is the Hugging Face model ID,
or the `--served-model-name` you passed to `max serve`.

```bash
# MAX: served at the released weight encoding
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "<HF_MODEL_ID>", "prompt": "The capital of France is", \
       "max_tokens": 64, "temperature": 0.0}' \
  | pixi run python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"

# HuggingFace reference in float32
pixi run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    '<HF_MODEL_ID>', trust_remote_code=True, device_map='auto', dtype='float32',
)
ids = tok('The capital of France is', return_tensors='pt').input_ids.to(m.device)
out = m.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
"
```

The outputs should be identical or nearly identical, and bfloat16 rounding
can make them drift in long generations. At the first token where they
differ, look at HF's two top logprobs. When they're within about 0.1 nats,
the models tie there, and either token is correct. A divergence with a clear
HF winner in the *first* few tokens after the divergence hunt passed usually
means:

- **Tokenizer or chat-template mismatch.** Try swapping in the HF
  tokenizer/chat template and re-running. If outputs converge, the issue
  was prompt formatting, not the model.
- **Dtype mismatch with the released weights.** Confirm MAX is using the
  encoding the model ships in (most are bfloat16). Compare
  `arch.py::default_encoding` against the Hub config's `torch_dtype`.
- **Sampling drift in MAX.** Confirm `temperature=0.0`, `top_p=1.0` on the
  MAX side. Any nonzero temperature diverges from greedy HF.

Matching text means the port passes the greedy text comparison. Completion
depends on the validation depth picked during plan-and-veto.

## Validation tiers

Pick the highest tier your bring-up scope requires. Higher tiers
include lower ones.

| Tier                      | What it tests                                                                             | Tool                                            | Pass criterion                                                                                                                                                       |
|---------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1: Smoke                  | Greedy text on one prompt looks plausible                                                 | ``curl /v1/completions``                        | Real tokens, not garbage                                                                                                                                             |
| 2: Coherence battery      | 4–6 diverse prompts at ``max_tokens=64`` and ``256`` (factual, reasoning, code, creative) | ``curl`` loop                                   | All outputs on-topic, no decode-loop collapse                                                                                                                        |
| 3: GSM8K                  | 8-shot math generation eval                                                               | ``run_lm_eval.py``                              | ≥ 0.5 for any modern base model, ≥ 0.7 for >70B                                                                                                                      |
| 4: HellaSwag 0-shot       | Loglikelihood path, short context                                                         | ``run_lm_eval.py`` + ``--enable-echo`` on serve | Within HF model card's ±5%                                                                                                                                           |
| 5: Few-shot loglikelihood | hellaswag/mmlu 5-shot                                                                     | ``run_lm_eval.py``                              | Match HF model card. **If random (~0.25–0.30): run the qwen3-as-control check before debugging your port**. See [max-vs-port-isolation.md](max-vs-port-isolation.md) |
| 6: Logit parity           | Per-position logprob diff vs HF reference                                                 | ``compare_layers.py`` + debug-model comparators | rel_diff < 5% top-1 at test prompt(s), per-layer cosine from the comparators                                                                                         |

``run_lm_eval.py`` is the repo's eval harness, at
``max/tests/integration/accuracy/run_lm_eval.py``.

Required serve flags by tier:

- Tiers 1–3 (generation): no flags beyond ``--custom-architectures``. If
  startup fails allocating the KV cache, lower
  ``--device-memory-utilization`` (default ``0.95``).
- Tiers 4–5 (loglikelihood): pass ``--enable-echo``. The
  OpenAI-compat ``/v1/completions`` endpoint with ``echo=true,
  logprobs=N`` requires the model to be compiled with echo support.
- Tier 6 (parity): same as tiers 4–5. Where logprobs diverge, localize the
  layer with the debug-model skill's per-layer comparators.
- Tiers 4–6 read logprobs, which the overlap scheduler rejects. Unless
  ``arch.py`` sets ``supports_overlap_scheduler=False``, also pass
  ``--no-enable-overlap-scheduler --force``.
