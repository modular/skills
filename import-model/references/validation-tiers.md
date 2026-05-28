# Validation tiers

Run the model end-to-end with pretrained weights, then run HF on the same
prompt with greedy sampling. On the MAX side, use the dtype that matches
the weight encoding the model supports — most models ship bfloat16 weights;
match that, not whatever the HF reference happens to be loaded as.

```bash
# MAX — use the dtype that matches the model's released weight encoding
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "<slug>", "prompt": "The capital of France is", \
       "max_tokens": 64, "temperature": 0.0}' \
  | pixi run python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"

# HuggingFace reference — load at the released weight dtype (bfloat16 for most models)
pixi run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    '<HF_MODEL_ID>', trust_remote_code=True, device_map='auto', dtype='bfloat16',
)
ids = tok('The capital of France is', return_tensors='pt').input_ids.to(m.device)
out = m.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
"
```

The outputs should be identical or nearly identical. Small differences from
BF16/FP16 rounding can cause divergence in long generations after a dozen
tokens or so — that's normal. Persistent divergence in the *first* few
tokens after the divergence hunt passed usually means:

- **Tokenizer or chat-template mismatch.** Try swapping in the HF
  tokenizer/chat template and re-running. If outputs converge, the issue
  was prompt formatting, not the model.
- **Dtype mismatch with the released weights.** Confirm MAX is using the
  encoding the model ships in (most are bfloat16). Compare
  `arch.py::default_encoding` against the Hub config's `torch_dtype`.
- **Sampling drift in MAX.** Confirm `temperature=0.0`, `top_p=1.0` on the
  MAX side. Any nonzero temperature will diverge from greedy HF.

When matching text comes out, the port is done **for greedy text**.
Real "done" depends on the validation depth picked during plan-and-veto.

## Validation tiers

Pick the highest tier your bring-up scope requires. Higher tiers
include lower ones.

| Tier                       | What it tests                                                                             | Tool                                                    | Pass criterion                                                                                                                                                        |
|----------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 — Smoke                  | Greedy text on one prompt looks plausible                                                 | ``curl /v1/completions``                                | Real tokens, not garbage                                                                                                                                              |
| 2 — Coherence battery      | 4–6 diverse prompts at ``max_tokens=64`` and ``256`` (factual, reasoning, code, creative) | ``curl`` loop                                           | All outputs on-topic; no decode-loop collapse                                                                                                                         |
| 3 — GSM8K                  | 8-shot math generation eval                                                               | ``run_lm_eval.py``                                      | ≥ 0.5 for any modern base model; ≥ 0.7 for >70B                                                                                                                       |
| 4 — HellaSwag 0-shot       | Loglikelihood path, short context                                                         | ``run_lm_eval.py`` + ``--enable-echo`` on serve         | Within HF model card's ±5%                                                                                                                                            |
| 5 — Few-shot loglikelihood | hellaswag/mmlu 5-shot                                                                     | ``run_lm_eval.py``                                      | Match HF model card. **If random (~0.25–0.30): run the qwen3-as-control check before debugging your port** — see [max-vs-port-isolation.md](max-vs-port-isolation.md) |
| 6 — Logit parity           | Per-position logprob diff vs HF reference                                                 | ``compare_layers.py`` + manual ``ops.output()`` taps    | rel_diff < 5% top-1 at test prompt(s); per-layer cos-sim is manual                                                                                                    |

Required serve flags by tier:

- Tiers 1–3 (generation): default ``pixi run max serve`` flags are fine.
- Tiers 4–5 (loglikelihood): **``--enable-echo`` is mandatory** — the
  OpenAI-compat ``/v1/completions`` endpoint with ``echo=true,
  logprobs=N`` requires the graph to be built with echo support.
- Tier 6 (parity): same as tiers 4–5, plus tap insertion in
  ``<slug>.py`` for intermediate-tensor dumps where logprobs diverge.
