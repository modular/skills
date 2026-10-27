---
name: debug-model
description: >
  Debug a MAX model that loads, compiles, serves, and generates tokens but
  whose output disagrees with a reference implementation. Use whenever parity
  debugging stalls on scalar taps, the model returns gibberish or wrong greedy
  tokens, logit cosine is high but argmax differs, or generation is coherent then
  diverges. Applies during an architecture port, a quantization bring-up, a
  multi-GPU conversion, or after a MAX upgrade. Triggers on "parity failure", "silent
  corruption", "logits match but tokens diverge", "top-1 mismatch", "greedy
  divergence", and "model serves but generates garbage". Not for crashes on load
  or pre-serve scaffolding (use import-model). Mandates reference-vs-MAX
  tensor-dump comparators first, numerical verification of fixes before
  recompiling, and a serve-vs-pipeline bisect when dumps match but text diverges.
compatibility: Requires pixi env with MAX installed, network access to Hugging Face Hub, and a GPU for dumping and serving.
---

# Parity/coherence failure protocol

The model runs without errors but output is wrong. Scalar `F.print` taps hide
directional bugs, and each new tap costs a full recompile. Build a per-layer
tensor-dump comparator first. Every later check becomes a numpy read from disk.

**Use this skill when** MAX output disagrees with a PyTorch reference you can
run and hook. The primary case is a custom-architecture port that serves but
fails parity or coherence checks. The same protocol covers a quantized variant
of a working port, a multi-GPU conversion of a working single-GPU port, and a
regression after a MAX upgrade. It applies anywhere a trusted reference exists.

**Don't use this skill when:**

- The server crashes on load → fix config, weights, module (`import-model`)
- You haven't finished implementing the module → `import-model` Phase 2
- An already-verified model needs logit-comparison tolerances tuned → that's
  threshold calibration, which this skill doesn't cover

## References

| File                                                                      | Read when                                                       |
|---------------------------------------------------------------------------|-----------------------------------------------------------------|
| [comparator-build.md](references/comparator-build.md)                     | Building HF/MAX dumpers and the comparator                      |
| [comparator-output-patterns.md](references/comparator-output-patterns.md) | Interpreting comparator output, false cliffs, token-0 invariant |
| [agent-workflow.md](references/agent-workflow.md)                         | Dispatching parallel investigation agents                       |
| [stacked-failures.md](references/stacked-failures.md)                     | A fix helped but verification still fails                       |

For MAX's built-in runtime debugging options (NaN checks, source tracebacks,
op logging), see
[the MAX debugging tools](https://max.modular.com/develop/debugging/).
`max.nn.legacy.hooks.PrintHook` (covered in comparator-build.md) prints layer
inputs and outputs for quick triage.

## Protocol

### Step 0: Sanity-check HF

Run `model.generate(...)` on the same HF repo, prompt, and checkpoint. If HF is
incoherent, fix the tokenizer or chat template first. The MAX model isn't the
problem.

```bash
pixi run python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('<repo>')
model = AutoModelForCausalLM.from_pretrained('<repo>', dtype=torch.float32, device_map='auto')
text = tok.apply_chat_template([{'role':'user','content':'Hello!'}], tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(text, return_tensors='pt').to(model.device), max_new_tokens=32, do_sample=False)
print(tok.decode(out[0]))
"
```

### Step 1: Build the comparator

Follow [comparator-build.md](references/comparator-build.md). You need three
artifacts: HF dumper, MAX dumper (`forward()` taps + standalone runner),
comparator script. Cast dump tensors to FP32 inside the MAX `forward()`.

> **Guard: validate the dumpers before trusting them.** Run both dumpers on a
> ModuleV3 model that MAX already serves correctly, with the taps added to its
> `forward()`. Use `olmo3`, or `llama3_modulev3` served with
> `--prefer-module-v3`. Expect cos ≈ 0.999 at every layer, identical
> `prompt_tokens.npy` on both sides, and `post_embed` cos = 1.0. Anything less
> means the dumpers are broken. Fix them before reading anything into a
> comparison on your port.

### Step 2: Read comparator output, then branch

Follow
[comparator-output-patterns.md](references/comparator-output-patterns.md). Check
false cliffs (wrong `hidden_states` indexing, missing `attention_mask` on
decode-prefix dumps) before bisecting the model.

Choose the next step from the first comparator run that passes the guard:

- **Some layer diverges** → layer hunt. Continue with Steps 3 to 5.
- **Every layer matches (cos ≥ 0.99) but generation still diverges** → the
  model math is likely correct. Skip to Step 6. Don't bisect layers.
- **Pattern matches a false-cliff signature** → fix the dumper, re-dump,
  re-read. Don't debug the model against a broken comparator.

Compute per-token and per-dim cosine slices when global cos looks ambiguous:

```python
cos_per_token = [cos(h[t], m[t]) for t in range(h.shape[0])]
cos_per_dim = [cos(h[:, d], m[:, d]) for d in range(h.shape[1])]
```

High `max_diff` where HF spikes and MAX is flat usually means HF formed an
attention anchor your port didn't.

### Step 3: Dispatch investigation agents

Follow [agent-workflow.md](references/agent-workflow.md). One lead agent
analyzes dumps and ranks hypotheses with tensor evidence. Helpers run in
parallel (weight stats, code diff, kernel inspection, sub-tap prep). Don't
dispatch fix-attempt agents until the lead localizes.

### Step 4: Verify numerically before recompiling

For each hypothesis: read dump tensors, compute what the fix would produce,
compare to HF. Match → recompile. No match → next hypothesis.

### Step 5: Apply fix, re-dump, re-compare

One compile, one smoke, full comparator pass (cos > 0.99 all layers). If
verification still fails, see
[stacked-failures.md](references/stacked-failures.md).

### Step 6: Serve vs pipeline

When teacher-forced dumps at decode step K match HF but generated text diverges,
the model math is likely correct. Bisect before re-bisecting layers:

| Check                           | Pass                       | Fail →                                                 |
|---------------------------------|----------------------------|--------------------------------------------------------|
| Teacher-forced dump @ K         | cos ≥ 0.99, argmax matches | Steps 1 to 5 (model bug)                               |
| Incremental pipeline decode @ K | token K matches HF         | Decode-state bug (KV, conv cache)                      |
| Serve vs pipeline @ K           | match                      | Harness bug (tokenizer, chat template, token recovery) |

Build if missing: pipeline decode compare, incremental layer dump, serve
compare scripts. If teacher-forced and pipeline both pass but serve fails, don't
edit the model.
