---
name: import-model
description: >
  Use when importing a new model architecture into MAX from a Hugging Face model ID.
  Triggers on: "import a model into MAX", "add model to MAX", "bring up <HF model> in MAX".
  Workflow: inspect Hugging Face config and modeling code, scaffold from a similar
  MAX architecture, implement each graph layer to match HF, serve, then verify against
  the Hugging Face reference. When the server runs but output is wrong (gibberish,
  greedy mismatch, coherent-then-diverges), load debug-model for the
  divergence hunt instead of scalar-tap iteration.
compatibility: Requires pixi env with MAX installed, network access to Hugging Face Hub, and a GPU for serving/verification.
metadata:
  argument-hint: "[Hugging Face model ID, e.g. 'Qwen/Qwen3-8B']"
---

# Import a model into MAX

**Input:** a Hugging Face model ID (`$ARGUMENTS`).

Copy [references/template.md](references/template.md) to track this port as
you work through the phases.

Porting a model to MAX means writing a MAX graph that performs the same
computation as the model's `modeling_<type>.py` in Hugging Face `transformers`,
then loading the released weights into that graph and verifying the outputs
match.

The workflow has three phases: **decide & plan**, **implement**, **verify**.
Phase 1 is reading and planning. Phase 2 is the port: implement every divergent
sublayer in the graph. Phase 3 is verification, only after implementation is
complete. Guards (preconditions that stop the line) gate the transitions
between activities; they are not steps of their own.

**Anti-pattern:** running `scaffold.py`, tweaking `arch.py`, and serving
while `<slug>.py` still implements the donor (`llama3`, `qwen3`, …). That is
not a bring-up — logit verification will fail because the wrong architecture
is running. Do not run verification scripts until
[implement-graph.md](references/implement-graph.md) completion criteria pass.

Each phase links to references with the details. Read the reference for the
activity you're on, not all of them upfront.

**Environment:** run **every** command through the pixi env that has MAX
installed (`pixi run python …`, `pixi run max serve …`), from the skill
root where `pixi.toml` lives (do not use bare `python` or `max` on the
shell PATH):

```bash
cd <path-to-skill>
pixi install
pixi run python scripts/inspect_hf.py <HF_MODEL_ID>
# Or: pixi run test-scripts   # smoke-test all scripts (no GPU)
```

Helper scripts live in this skill's `scripts/` directory (copy or vendor
them into your repo). All helpers are also reachable through a unified
dispatcher with the same argument names and exit codes:

```bash
pixi run python scripts/import_model.py inspect <HF_MODEL_ID>
pixi run python scripts/import_model.py scaffold <HF_MODEL_ID> --start-from llama3 --output-dir ./
pixi run python scripts/import_model.py list-archs --match LlamaForCausalLM
pixi run python scripts/import_model.py check-walls <HF_MODEL_ID>
pixi run python scripts/import_model.py list-keys <HF_MODEL_ID> --summary
pixi run python scripts/import_model.py gates <HF_MODEL_ID> --port-dir <port_dir>/
pixi run python scripts/import_model.py compare <HF_MODEL_ID> --slug <slug> --port 8000
```

Port layout:

- **`<port_dir>`** — slug folder containing `arch.py` and `ARCHITECTURES` in
  `__init__.py` (usually `<output_dir>/<slug>/`). Pass this path to both
  `--custom-architectures` and `run_oss_gates.py --port-dir`.

MAX resolves `--custom-architectures <port_dir>` by adding `dirname(<port_dir>)`
to `sys.path` and importing `basename(<port_dir>)` as the module. Passing the
parent directory imports the wrong module name (e.g. `custom-arch` instead of
your slug).

Import/API errors while editing: copy the donor arch under
`modular/max/python/max/pipelines/architectures/<donor>/`; see
[pitfalls-config.md § Import and config API traps](references/pitfalls-config.md#import-and-config-api-traps).

---

## Phase 1 — Decide & plan

> **Guard: is the architecture already registered in MAX?**
> Before writing any code, check whether MAX already registers the architecture
> class in your model's `config.json::architectures[0]`. If
> `pixi run python list_native_archs.py --match <Class>` returns a slug, run
> `pixi run max serve --model <HF_MODEL_ID>`
> and stop — no port needed. Full procedure:
> [native-arch-check.md](references/native-arch-check.md).

### Read `config.json`

Pull the config and read every field:

```bash
pixi run python -c "from transformers import AutoConfig; \
  print(AutoConfig.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True))"
```

Or use the helper, which fetches raw `config.json` from the Hub, runs the
native-arch check, and prints every key mapped to the MAX API:

```bash
pixi run python inspect_hf.py <HF_MODEL_ID>
```

Then list safetensors metadata (keys, shapes, dtypes — no weight download):

```bash
pixi run python list_checkpoint_keys.py <HF_MODEL_ID> --summary
```

Each row is one `config.json` key → `pipeline_config.model.huggingface_config`
(or `SupportedArchitecture` in `arch.py` for `architectures`, `torch_dtype`).
Keys you cannot wire through `MyConfig.initialize()` are the deltas you
implement in the graph. Field meanings and common deltas:
[read-config-json.md](references/read-config-json.md).

Scan for hard blockers before you commit to a port:

```bash
pixi run python check_walls.py <HF_MODEL_ID>
```

Exit 0 → continue. Exit 1 → review
[recognize-walls.md](references/recognize-walls.md). Exit 2 → stop until the
wall is resolved or scoped out.

### Read the model card

Open `https://huggingface.co/<HF_MODEL_ID>` and read the model card for:

- **The paper or blog post.** Skim its architecture section — authors call out
  the *interesting* modifications (QK-norm, MLA, sliding-window attention,
  MoE routing) because those are what they want credit for.
- **"Tricks" mentioned in the card.** Phrases like "we introduce", "unlike
  prior models", "this is the first model to" mark deltas that will bite you
  during implementation if you miss them now.

If the card says the model is from a known family (Llama, Mistral, Qwen,
Gemma), note that; the donor-comparison activity below will start from the
closest already-ported variant of that family.

If the card mentions custom CUDA kernels, custom attention with no public
reference, FP8/FP4-only released weights, ALiBi, recurrence or state-space
layers; see [recognize-walls.md](references/recognize-walls.md) before going
further. Some models can't be ported with the public MAX surface alone.

### Propose a plan; accept a veto

Before any code, write a short paragraph stating what you'd do by default,
then wait for the user to confirm or veto. Cover four axes (distribution
shape, quantization variants, validation depth, hardware target) — all
derived from what you've already read. Don't ask blank questions; state a
default and let them push back.

Full guidance and an example paragraph:
[plan-and-veto.md](references/plan-and-veto.md).

If estimated weight bytes do not fit one GPU, read
[distributed-transformer.md](references/distributed-transformer.md) before
choosing `--start-from` — distribution shape matters more than attention
family alone.

### Compare with other MAX architectures

You're picking the closest already-ported MAX architecture to copy from.
"Closest" means: same attention shape (dense vs. GQA vs. MLA vs. MoE), same
MLP shape (gated vs. non-gated, dense vs. routed), same head layout (tied vs.
untied, single Linear vs. multi-step).

List what your installed MAX registers (do not hard-code a slug list):

```bash
pixi run python list_native_archs.py
```

Heuristic HF-signal → donor slug hints are in
[map-to-max.md](references/map-to-max.md). Quick version:

| Your model                                          | Start from               |
|-----------------------------------------------------|--------------------------|
| Llama 3-ish (GQA, RoPE, SwiGLU MLP)                 | `llama3`                 |
| Gemma-ish (RMSNorm scale, logit softcap, dual norm) | `gemma2` or `gemma3`     |
| Qwen-ish (GQA, RoPE, may have QK-norm)              | `qwen2` / `qwen3`        |
| Mistral-ish (sliding window)                        | `mistral`                |
| Phi-ish (partial RoPE)                              | `phi3`                   |
| MoE (sparse experts, top-k routing)                 | `mixtral` or `qwen3_moe` |
| MLA (latent KV)                                     | `deepseekV3`             |

Open the chosen MAX arch's directory and read its top-level model file
(usually `<slug>.py`). You're answering: which functions/classes need to
change vs. stay the same when I port my model?

Now read the corresponding Hugging Face modeling file:

```bash
pixi run python -c "from transformers.models.<model_type> import modeling_<model_type>; \
  print(modeling_<model_type>.__file__)"
```

Read the `__init__`, the attention `forward`, the MLP `forward`, the block
class, and the final head. Compare each to the MAX equivalent. The reference
[read-modeling-code.md](references/read-modeling-code.md) covers what to look
for in each.

Output of this activity: a **delta list** — one row per real difference between
HF and the donor MAX arch (attention, MLP/MoE, block wiring, head, RoPE,
masks). You implement every row in Phase 2. Three or fewer structural deltas
→ good donor choice. Many deltas → pick a closer donor or plan to rewrite
whole classes. Do not proceed to verification with an empty or "looks
Llama-ish" delta list.

---

## Phase 2 — Implement

### Scaffold the file layout

`scaffold.py` **only copies files**. It does not implement your model.

```bash
pixi run python scaffold.py <HF_MODEL_ID> --start-from <max_arch_slug> --output-dir <output_dir>
```

This reads ``architectures[0]`` from the Hub ``config.json`` for
``arch.py::name``, then copies the chosen native MAX architecture into
``<output_dir>/<slug>/`` as five files:

- `arch.py` — registration shell (verify `name=` and encoding)
- `model_config.py` — donor config (must be rewired during implementation)
- `model.py` — pipeline model shell
- `weight_adapters.py` — donor renames (must be rewritten for your checkpoint)
- `<slug>.py` — **donor graph** (must be edited to match HF during
  implementation)

After scaffold, you have a directory layout and a **wrong** graph. Stop here
until the graph is implemented — do not serve.

**Scaffold also leaves the donor's docstrings and code comments in place.**
Sed-renaming class names doesn't touch text that records *what the file
claims to do*. After scaffold, ``<slug>.py`` opens with a docstring
describing the donor; the new class claims behaviors (single-GPU support,
QK-norm, post-attention norm, etc.) the new file may not have. Rewriting
those docstrings is a required part of the graph implementation — not
optional polish. See [honest-docstrings.md](references/honest-docstrings.md)
for the three-sentence pattern every module docstring should follow and a
mandatory audit checklist before declaring the implementation done.

### Implement the graph

This is the bring-up. Phase 1 produced the config map and delta list; the
implementation activity executes them in code.

Full checklist, work order, anti-patterns, and completion criteria:
[implement-graph.md](references/implement-graph.md).

In order:

1. **`model_config.py`** — wire every `config.json` key from Phase 1 /
   `inspect_hf.py`. Set `get_kv_params()` head counts and head_dim to match HF.
2. **`weight_adapters.py`** — map your checkpoint's safetensor keys to the MAX
   module names you will use. Run `list_checkpoint_keys.py` first; see
   [rename-weights.md](references/rename-weights.md). After load, wire the
   coverage audit in [state-dict-audit.md](references/state-dict-audit.md)
   (especially MoE and `strict=False` tied embeddings).
3. **`<slug>.py`** — for **each row in the delta list**, edit or replace
   the donor class so MAX `forward()` mirrors HF `forward()`:
   - Attention (Q/K/V, RoPE, mask, GQA, softcap, …)
   - MLP or MoE (activation, routing, shared experts, …)
   - Decoder block (**norm order and residual wiring** — not interchangeable
     with Llama)
   - Final norm and LM head (tie, logit scale, softcap)
4. **`arch.py`** — confirm `name=` matches `architectures[0]`;
   `default_encoding` matches Hub `torch_dtype`.
5. **`model.py`** — only if HF wraps the backbone differently (VL, multi-modal).

Read HF `modeling_<type>.py` **while editing**, not after verification fails.
Subclass the donor only where HF and donor match; rewrite the class where the
delta list said they differ.

**The implementation is done when** every item in
[implement-graph.md](references/implement-graph.md#completion-criteria-required-before-serving)
is checked — especially: every delta has a corresponding code change, weights
load without orphan keys, and the **scaffold-comment audit** in
[honest-docstrings.md](references/honest-docstrings.md#mandatory-audit-before-declaring-the-implementation-done)
has been run with each match classified as OK / Lie / Stale. A passing audit is
**mandatory**; declaring the implementation done without it leaves donor lies in
the codebase that nothing downstream will catch.

Quick grep recipe (full classification rules in
[honest-docstrings.md](references/honest-docstrings.md)):

```bash
pixi run rg -i -n 'qwen|llama|mistral|cohere|gemma|phi|deepseek|exaone|olmo|granite|qwen3|mixtral|single-GPU|single GPU|RMSNorm|QK-norm' <port_dir>/
```

Your implementation-complete message must explicitly attest to the audit
(e.g. ``"docstrings rewritten to the three-sentence pattern; rg returns N
hits, all legitimate lineage references"``). A claim without the
attestation isn't a completion.

Preflight (Hub config + arch registration — run before first serve):

```bash
pixi run python run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>/
```

> **Guard: local smoke gate (mandatory before Phase 3).**
> `pixi run max serve` cold-compiles for 5–25 minutes. Before serving, run the
> four local checks in [serve-and-iterate.md](references/serve-and-iterate.md)
> (import smoke, graph dry-build, adapter⇄graph key diff, weights-format
> preflight). `run_oss_gates.py` covers walls, checkpoint metadata, and
> `arch.py` name/encoding — not a substitute for those four.

---

## Phase 3 — Verify

### Check if it generates coherent text

**Prerequisite:** graph implementation complete. Do not serve to "see what
happens" during implementation — fix config, adapters, and graph first.

**Sanity-check the HF reference FIRST.** Run HF alone on the model card's
intended prompt template, before involving MAX. If HF itself produces
gibberish, your oracle is broken — fixing your port against a broken
oracle wastes days.

Then serve with
`pixi run max serve --model-path <HF_MODEL_ID> --custom-architectures <port_dir>`
and probe with the model card's intended template (not just "The capital of
France is" — that prompt is wrong for PrefixLMs and instruction-tuned models).
Three possible outcomes: server crashes during load → fix config/adapters;
server starts but returns garbage → divergence hunt; server returns plausible
text → run at `max_tokens=64+` before celebrating.

Full HF-reference sanity check, encoder/embedding slug serve flow, and
fix-test loop discipline:
[serve-and-iterate.md](references/serve-and-iterate.md).

### Parity/coherence failure (invoke `debug-model`)

When the server starts but output is wrong (gibberish, wrong greedy token
at index K, high logit cosine with wrong argmax, or coherent for N tokens
then diverges), **stop the scalar-tap loop** and load the
[`debug-model`](../debug-model/SKILL.md) skill.

That skill is the authoritative workflow for silent corruption. It
mandates:

1. HF sanity-check on the same prompt + checkpoint
2. Per-layer HF vs MAX tensor-dump comparators (not `ops.print` eyeballing)
3. Parallel investigation agents with numerical verification before recompile
4. Serve-vs-pipeline bisect when dumps match but generated text diverges

Use `import-model` for bring-up scaffolding and gates. Use
`debug-model` for the divergence hunt itself.

#### Quick logit probe (first 5 minutes only)

Before building dumpers, a fast sanity check:

```bash
pixi run python compare_layers.py <HF_MODEL_ID> \
  --slug <your_slug> --port 8000 \
  --prompt "The capital of France is"
```

Requires ``pixi run max serve`` with ``--custom-architectures <port_dir>`` on
the same port. This script prints HF-only layer stats and compares top-1
logprob at the prompt. See
[layer-by-layer-debugging.md](references/layer-by-layer-debugging.md) for
flag details.

If logprobs diverge or output is garbage, **switch to
`debug-model`**. Do not iterate with manual ``ops.output()``
taps alone. The symptom catalog in
[divergences.md](references/divergences.md) still applies once the
comparator localizes the failing layer.

### Check against Hugging Face

Run the model end-to-end with pretrained weights, then run HF on the same
prompt with greedy sampling. On the MAX side, use the dtype that matches the
weight encoding the model supports (most models ship bfloat16). Outputs
should be identical or nearly identical; small BF16/FP16 rounding can cause
divergence past a dozen tokens. Persistent divergence in the *first* tokens
after the divergence hunt passed usually means tokenizer/chat-template
mismatch, dtype mismatch with the released weights, or nonzero MAX sampling.

When matching text comes out, the port is done **for greedy text**. Real
"done" depends on the validation depth picked during planning — pick a tier
from 1 (smoke) to 6 (logit parity).

Full HF-comparison recipe, divergence triage, and the 6-tier validation
table: [validation-tiers.md](references/validation-tiers.md).

---

## Common pitfalls

Use [pitfalls.md](references/pitfalls.md) as an index — find your symptom,
then load the one category file (config, weights, graph, or serving) —
[honest-docstrings.md](references/honest-docstrings.md) for the docstring
audit specifically. The two big ones:

- **Scaffold ≠ port.** Do not serve or verify until the graph implements
  every delta in `<slug>.py`.
- **Sed-rename leaves donor docstrings intact.** Class names get
  renamed, but docstrings and comments still describe the donor. Rewrite
  them and run the audit grep before declaring the implementation done.

## Tests and CI

When you add `pytest` tests for the ported model, minimize the number of
MAX graph compilations per file. Compile once via a module-scoped fixture
and reuse it across `@pytest.mark.parametrize` cases. For files that must
compile different graphs, parallelize them with Bazel `shard_count`
instead of splitting the file. Full patterns and examples:
[tests-and-ci.md](references/tests-and-ci.md).
