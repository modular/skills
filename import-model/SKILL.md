---
name: import-model
description: >
  Use when importing a new model architecture into MAX from a Hugging Face model ID.
  Triggers on: "import a model into MAX", "add model to MAX", "bring up <HF model> in MAX".
  Workflow: inspect the Hugging Face config and modeling code, then scaffold from
  a similar ModuleV3 architecture. Implement each layer as a max.nn
  Module that matches HF, serve, then verify against the Hugging Face reference.
  When the server runs but output is wrong (gibberish, greedy mismatch,
  coherent-then-diverges), load debug-model for the divergence hunt.
compatibility: Requires pixi env with MAX installed, network access to Hugging Face Hub, and a GPU for serving/verification.
metadata:
  argument-hint: "[Hugging Face model ID, for example 'Qwen/Qwen3-8B']"
---

# Import a model into MAX

**Input:** a Hugging Face model ID (`$ARGUMENTS`).

Copy [references/template.md](references/template.md) to track this port as
you work through the phases. Its file references are paths from this skill's
root, so they still resolve after you copy it elsewhere.

Porting a model to MAX means writing a ModuleV3 model that performs the same
computation as the model's `modeling_<type>.py` in Hugging Face
`transformers`. You then compile it with the released weights and verify that
the outputs match. ModuleV3 is MAX's model API. Layers are `Module` subclasses
from `max.nn` with a `forward()` over `Tensor` values. Ops come
from `max.experimental.functional` (imported as `F`). The pipeline builds the
module under `F.lazy()` and compiles it with `compile(weights=...)`.
Single-GPU, tensor-parallel, and expert-parallel models all use ModuleV3.
Multi-GPU models shard through a `DeviceMesh`.

The workflow phases are **decide and plan**, **implement**, and **verify**.
Phase 1 is reading and planning. Phase 2 is the port: implement every divergent
sublayer in the module. Phase 3 is verification, only after implementation is
complete. Guards (preconditions that halt the workflow) gate the transitions
between activities. They aren't steps of their own.

**Anti-pattern:** running `scaffold.py`, tweaking `arch.py`, and serving
while `<slug>.py` still implements the donor (`llama3_modulev3`, `olmo3`, …).
Logit verification fails on that port because `<slug>.py` still runs the
donor architecture. Don't run verification scripts until
[implement-graph.md](references/implement-graph.md) completion criteria pass.

Each phase links to references with the details. Read the reference for the
activity you're on, not all of them upfront.

**Environment:** run every command through the pixi env that has MAX
installed (`pixi run python …`, `pixi run max serve …`). Run them from the
skill root where `pixi.toml` lives, and don't use bare `python` or `max` on
the shell PATH:

```bash
cd <path-to-skill>
pixi install
pixi run python scripts/inspect_hf.py <HF_MODEL_ID>
# Or: pixi run test-scripts   # smoke-test all scripts (no GPU)
```

The default environment installs on Linux (x86-64 and ARM64) and on macOS
with Apple silicon, and needs no CUDA. Its PyTorch runs the HF reference
model on CPU, or on an Apple silicon GPU. On an NVIDIA host, `pixi run -e
cuda …` runs the same commands with CUDA PyTorch, so the HF reference runs
on the GPU too.

Helper scripts live in this skill's `scripts/` directory (copy or vendor
them into your repo). All helpers are also reachable through a unified
dispatcher with the same argument names and exit codes:

```bash
pixi run python scripts/import_model.py inspect <HF_MODEL_ID>
pixi run python scripts/import_model.py list-archs --donors
pixi run python scripts/import_model.py scaffold <HF_MODEL_ID> --start-from olmo3 --output-dir ./
pixi run python scripts/import_model.py list-archs --match LlamaForCausalLM
pixi run python scripts/import_model.py check-walls <HF_MODEL_ID>
pixi run python scripts/import_model.py list-keys <HF_MODEL_ID> --summary
pixi run python scripts/import_model.py gates <HF_MODEL_ID> --port-dir <port_dir>
pixi run python scripts/import_model.py compare <HF_MODEL_ID> --port 8000
```

Port layout:

- **`<port_dir>`**: slug folder containing `arch.py` and `ARCHITECTURES` in
  `__init__.py` (usually `<output_dir>/<slug>`, with no trailing slash). Pass
  this path to both `--custom-architectures` and `run_oss_gates.py --port-dir`.

MAX resolves `--custom-architectures <port_dir>` by adding `dirname(<port_dir>)`
to `sys.path` and importing `basename(<port_dir>)` as the module. Passing the
parent directory imports the wrong module name (for example, `custom-arch`
where your slug belongs).

If you hit import or API errors while editing, copy the imports from the donor
arch under `max/pipelines/architectures/<donor>/` in your installed MAX package.
See
[pitfalls-config.md § Import and config API traps](references/pitfalls-config.md#import-and-config-api-traps).

---

## Phase 1: Decide and plan

> **Guard: is the architecture already registered in MAX?**
> Before writing any code, check whether MAX already registers the architecture
> class in your model's `config.json::architectures[0]`. If
> `pixi run python scripts/list_native_archs.py --match <Class>` returns a
> slug, run `pixi run max serve --model-path <HF_MODEL_ID>` and stop. No port is
> needed. Full procedure:
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
pixi run python scripts/inspect_hf.py <HF_MODEL_ID>
```

Then list safetensors metadata (keys, shapes, and dtypes, with no weight
download):

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> --summary
```

The summary counts tensors and dtypes. Run it again without `--summary` to
list the key names before you write the weight adapter.

Each row of the `inspect_hf.py` table maps one `config.json` key to
`pipeline_config.model.huggingface_config` (or to `SupportedArchitecture` in
`arch.py` for `architectures` and `torch_dtype`). Once you've picked a donor,
rerun it with `--start-from <donor_slug>`: it marks each key the donor's
config sources never mention. Those keys are the config deltas you
implement, a config field plus the module code that uses it, unless no model
code reads them (`bos_token_id`, `initializer_range`). Field meanings and common
deltas: [read-config-json.md](references/read-config-json.md).

Scan for hard blockers before you commit to a port:

```bash
pixi run python scripts/check_walls.py <HF_MODEL_ID>
```

Exit 0 → continue. Exit 1 → review
[recognize-walls.md](references/recognize-walls.md). Exit 2 → stop until the
wall is resolved or scoped out.

### Read the model card

Open `https://huggingface.co/<HF_MODEL_ID>` and read the model card for:

- **The paper or blog post.** Skim its architecture section. Authors list
  their modifications there (QK-norm, MLA, sliding-window attention, MoE
  routing).
- **"Tricks" mentioned in the card.** Phrases like "we introduce", "unlike
  prior models", and "this is the first model to" mark deltas. If you miss
  them now, they cause bugs during implementation.

If the card says the model is from a known family (Llama, Mistral, Qwen,
Gemma), the donor-comparison activity below will start from the
closest already-ported variant of that family.

Some model card signals need a wall check: custom CUDA kernels, custom
attention with no public reference, FP8/FP4-only released weights, ALiBi, and
recurrence or state-space layers. If the card mentions any of them, read
[recognize-walls.md](references/recognize-walls.md) before going further.
Some models can't be ported with the public MAX surface alone.

### Propose a plan and accept a veto

Before any code, write a short paragraph stating what you'd do by default,
then wait for the user to confirm or veto. Cover distribution shape, the
donor, quantization variants, validation depth, and hardware target, all
derived from what you've already read. If the model needs a piece ModuleV3
doesn't have yet (a layer, a distributed primitive, a quantized path), name
it and say whether you'll add it to `max.nn` or need the user
to decide. Don't ask blank questions. State a default and let the user push
back.

Full guidance and an example paragraph:
[plan-and-veto.md](references/plan-and-veto.md).

If estimated weight bytes don't fit one GPU, read
[distributed-transformer.md](references/distributed-transformer.md) before
choosing `--start-from`: distribution shape matters more than attention
family alone.

### Compare with other MAX architectures

You're picking the closest ModuleV3 architecture to copy from. "Closest"
means a match on:

- Attention shape (dense vs. GQA vs. MLA)
- MLP shape (gated vs. non-gated, dense vs. routed)
- Head layout (tied vs. untied, single Linear vs. multi-step)
- Distribution shape

List the ModuleV3 architectures your installed MAX ships (don't hard-code a
slug list):

```bash
pixi run python scripts/list_native_archs.py --donors
```

Heuristic HF-signal → donor slug hints are in
[map-to-max.md](references/map-to-max.md). Quick version:

| Your model                                          | Start from                                       |
|-----------------------------------------------------|--------------------------------------------------|
| Llama 3-ish (GQA, RoPE, SwiGLU MLP)                 | `llama3_modulev3`                                |
| QK-norm, mixed sliding and full attention           | `olmo3`                                          |
| Gemma-ish (`1 + weight` RMSNorm, dual norm)         | `gemma3_modulev3`                                |
| Phi-ish (fused `qkv_proj` and `gate_up_proj`)       | `phi3_modulev3`                                  |
| Granite-ish (MuP scalars)                           | `granite_modulev3`                               |
| MoE (sparse experts, top-k routing)                 | `gpt_oss_modulev3`                               |
| MLA (latent KV), single GPU                         | `deepseekV2_modulev3`                            |
| MLA + MoE, multi-GPU (TP, DP + EP)                  | `deepseekV3_modulev3`                            |
| Hybrid attention and state-space layers             | `nemotron_h_modulev3`                            |
| Vision-language (scaffold with `--full-copy`)       | `gemma3multimodal_modulev3`, `kimik2_5_modulev3` |

Open the chosen arch's directory and read its model file (the module that
`model.py` instantiates in `_instantiate_module()`, for example
`olmo3/olmo3.py` or `llama3_modulev3/llama3.py`) and its `layers/`. You're
answering one question: which modules change and which stay the same in your
port?

Now read the corresponding Hugging Face modeling file:

```bash
pixi run python -c "from transformers.models.<model_type> import modeling_<model_type>; \
  print(modeling_<model_type>.__file__)"
```

Read the `__init__`, the attention `forward`, the MLP `forward`, the block
class, and the final head. Compare each to the MAX equivalent. The reference
[read-modeling-code.md](references/read-modeling-code.md) covers what to look
for in each. Check the RoPE layout, norm formula, and activation against the
symptom table in [divergences.md](references/divergences.md) now, before you
serve: a function named `rotate_half` can still rotate adjacent pairs
(ERNIE 4.5), and the donor's default reads the wrong pairs.

Output of this activity: a **delta list**, one row per real difference between
HF and the donor MAX arch (attention, MLP/MoE, block wiring, head, RoPE,
masks). You implement every row in Phase 2. Three or fewer structural deltas
→ good donor choice. Many deltas → pick a closer donor or plan to rewrite
whole modules. Don't proceed to verification with an empty or "looks
Llama-ish" delta list.

---

## Phase 2: Implement

### Scaffold the file layout

`scaffold.py` only writes a skeleton. It doesn't implement your model.

```bash
pixi run python scripts/scaffold.py <HF_MODEL_ID> --start-from <donor_slug> --output-dir <output_dir>
```

This reads `architectures[0]` from the Hub `config.json` for
`arch.py::name`, then writes a skeleton that subclasses the chosen ModuleV3
architecture into `<output_dir>/<slug>/`. The slug defaults to the repo name,
size suffix included (`ernie_4_5_0_3b_pt`). Pass `--slug` to name the
package after the architecture family. The generated classes are
`<Short>Model` and `<Short>Config` (`Ernie4_5Model`), which can share names
with Hugging Face's own classes, so qualify them when you read the two side
by side.

- `arch.py`: registration shell carrying every `SupportedArchitecture`
  keyword the donor sets verbatim (`memory_planner=`, `batching=`,
  `reasoning_parser=`, ...), including a configured planner such as
  `PagedMemoryPlanner.with_activation_reservation(...)`. The scaffold writes
  `name=`, the repo IDs, the encodings, the weights format, and the port's
  own classes and adapter (verify `name=`, encoding, and `memory_planner=`).
  When the donor declares no planner, the field is left out with a TODO. A
  value the scaffold can't trace to an import becomes a TODO and a warning.
- `model_config.py`: donor config subclass with a `from_donor()` hook for
  the fields the donor doesn't read (must be rewired during implementation)
- `model.py`: pipeline model shell. `_create_model_config()` re-types the
  donor's config as yours, and `_instantiate_module()` builds your root
  module and places it on the device or mesh with the donor's placement. It
  inherits the donor's batch processor, so input preparation keeps working
  without a `batch_processor.py`.
- `weight_adapters.py`: delegates to the donor's renames (must be rewritten
  for your checkpoint)
- `<slug>.py`: **donor module** (must be edited to match HF during
  implementation)

After scaffold, `<slug>.py` still computes the donor's math. Don't serve
until the module is implemented.

The scaffold formats the port with `ruff format` and rewrites the donor's
`huggingface_config.rope_theta` reads to `get_rope_theta(huggingface_config)`,
which reads the value under both Transformers 4 and Transformers 5. A port
scaffolded with an earlier version of this skill can still read
`huggingface_config.rope_theta` in `model.py` and fail
`check_port.py` with `AttributeError: ... has no attribute 'rope_theta'`.
Re-scaffold it, or make the same edit by hand and import `get_rope_theta`
from `max.pipelines.lib.pipeline_variants.utils`.

**Donor docstrings and code comments survive any donor code you copy.** The
default scaffold writes fresh docstrings for its generated files. `--full-copy`
mode keeps the donor's text, and so does any donor file you copy or
subclass-and-edit by hand. Class renames don't touch text that records
*what the file claims to do*. Those files open with docstrings that describe the
donor, and they claim behaviors (single-GPU support, QK-norm, post-attention
norm, and so on) your port may not have. Rewriting those docstrings is a
required implementation step. See
[rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md) for the
docstring pattern every module docstring should follow. It also has the
audit checklist you must run before you declare the implementation done.

### Implement the module

Phase 1 produced the config map and delta list. The implementation activity
turns them into code.

Full checklist, work order, the ModuleV3 API surface, anti-patterns, and
completion criteria: [implement-graph.md](references/implement-graph.md).

In order:

1. **`model_config.py`**: the donor's config reads the keys it knows. For
   every other `config.json` key from Phase 1 / `inspect_hf.py`, declare a
   field on the port's config and set it in `from_donor()`, which
   `model.py::_create_model_config()` calls with the donor's finished
   config. Confirm `construct_kv_params()` head counts and head_dim match HF.
   The pipeline sizes the KV cache from your config's `construct_kv_params()`,
   but `from_donor()` copies `kv_params` from the donor's config, and the
   modules read that copy. If you override `construct_kv_params()`,
   `get_head_dim()`, or `get_num_layers()`, also set `kv_params` in
   `from_donor()` from your override, so the cache and the modules agree.
2. **`weight_adapters.py`**: map your checkpoint's safetensor keys to the
   parameter names your root module yields from `parameters`, including the
   root's wrapper prefix (`language_model.` in `olmo3`). Run
   `list_checkpoint_keys.py` first. See
   [rename-weights.md](references/rename-weights.md). Wire the coverage audit
   in [state-dict-audit.md](references/state-dict-audit.md): `compile()`
   rejects a missing or mis-shaped parameter, and the audit catches checkpoint
   tensors that no parameter consumes.
3. **`<slug>.py`**: for **each row in the delta list**, edit or replace
   the donor module so MAX `forward()` mirrors HF `forward()`:
   - Attention (Q/K/V, RoPE, mask, GQA, softcap, …)
   - MLP or MoE (activation, routing, shared experts, …)
   - Decoder block (**norm order and residual wiring**, not interchangeable
     with Llama)
   - Final norm and LM head (tie, logit scale, softcap)
4. **`arch.py`**: confirm that `name=` matches `architectures[0]` and
   `default_encoding` matches Hub `torch_dtype`. Keep the scaffolded
   `memory_planner=` (copied from the donor). Without it, MAX budgets no
   activation memory for the KV cache and skips `max_batch_size` inference.
   If scaffold left a `memory_planner` TODO, the donor had none. Resolve it
   before serving: KV-cache ports need `memory_planner=PagedMemoryPlanner`,
   and only architectures doing their own memory estimation (diffusion,
   embedding) should leave it unset. The scaffold also carries the donor's
   `batching=`. Change it only if the port's batching differs from the
   donor's.
5. **`model.py`**: `_instantiate_module()` constructs your root module and calls
   `.to(self.devices[0])`, or builds the module inside `default_device(mesh)`
   for a multi-GPU port. Edit further only if HF wraps the backbone differently
   (VL, multi-modal).

Read HF `modeling_<type>.py` **while editing**, not after verification fails.
Subclass the donor only where HF and donor match. Rewrite the module where
the delta list said they differ.

**The implementation is done when** every item in
[implement-graph.md](references/implement-graph.md#completion-criteria-required-before-serving)
is checked, especially these:

- Every delta has a corresponding code change.
- Weights compile without orphan keys.
- You've run the **scaffold-comment audit** in
  [rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md#mandatory-audit-before-declaring-the-implementation-done)
  and classified each match as OK, Wrong, or Stale.

Run the audit before you declare the implementation done. `max serve` and
`compare_layers.py` don't read docstrings, so no later step finds the donor
claims the audit misses.

Quick grep recipe (full classification rules in
[rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md)):

```bash
grep -rniE 'qwen|llama|mistral|cohere|gemma|phi|deepseek|olmo|granite|qwen3|mixtral|single-GPU|single GPU|RMSNorm|QK-norm' <port_dir>/
```

In a subclass-mode port, most hits are the donor imports and the lineage
lines the scaffold writes. Those are accurate. Read the docstrings and
comments among the hits.

Your implementation-complete message must explicitly attest to the audit
(for example, `"docstrings rewritten to the pattern in
rewrite-donor-docstrings.md, and rg returns N hits, all legitimate lineage
references"`). Don't report the implementation complete without that
statement.

Preflight (Hub config + arch registration, run before first serve):

```bash
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>
```

> **Guard: local smoke gate (mandatory before Phase 3).**
> `max serve` compiles the whole model before it answers a request, so a wrong
> parameter name or shape costs a full compile to discover. Run the port check
> first. It imports the port the way `max serve` does, builds the root module
> under `F.lazy()` without compiling, and lists every parameter the adapted
> checkpoint doesn't fill, every shape or dtype mismatch, and every tensor no
> parameter reads:
>
> ```bash
> pixi run python scripts/check_port.py <HF_MODEL_ID> --port-dir <port_dir>
> ```
>
> It exits 1 when `compile()` would fail. It reads the real checkpoint, so the
> first run downloads the weights. Format and lint the port before each
> check, since your edits add code the scaffold didn't format:
>
> ```bash
> pixi run ruff format <port_dir>
> pixi run ruff check <port_dir>
> ```
>
> `ruff check` must report no errors. `run_oss_gates.py` covers walls,
> checkpoint metadata, and `arch.py` name/encoding. It doesn't replace
> this check. See [serve-and-iterate.md](references/serve-and-iterate.md) for
> the weights-format preflight.

---

## Phase 3: Verify

### Check if it generates coherent text

**Prerequisite:** module implementation complete. Don't serve to "see what
happens" during implementation. Fix config, adapters, and the module first.

**Sanity-check the HF reference first.** Run HF alone on the model card's
intended prompt template, before involving MAX. If HF itself produces
gibberish, the reference is broken, and comparisons against it report
mismatches that no port change fixes.

Then serve:

```bash
pixi run max serve --model-path <HF_MODEL_ID> --custom-architectures <port_dir>
```

`max serve` reserves `--device-memory-utilization` (default `0.9`) of the
device's memory for the KV cache. If startup fails allocating it (out of
memory, or a buffer larger than the device allows), lower the fraction, for
example `--device-memory-utilization 0.6`. Lower it too when the HF
reference runs on the same machine as the server, where the two compete for
memory: a float32 reference can otherwise spill to disk and make each
comparison take minutes.

Probe an instruction-tuned model through `/v1/chat/completions`, which
applies its chat template, and a base model with a plain prompt through
`/v1/completions`. A PrefixLM needs the prompt format its model card shows.
For a model that reasons before it answers (a `<think>` block by default),
turn reasoning off on both sides, with `chat_template_kwargs` on MAX and
the same keyword (`enable_thinking=False`) in HF's `apply_chat_template()`,
or the comparison only covers the reasoning text. When you tokenize
chat-templated text on the HF side, pass `add_special_tokens=False`, since
the template already adds them. Each outcome has a next step:

- Server crashes during load → fix config and adapters.
- Server starts but returns garbage → start the divergence hunt.
- Server returns plausible text → run at `max_tokens=64+`, then the verify
  gate. A port missing a delta can still answer fluently.

Full HF-reference sanity check, encoder/embedding slug serve flow, and
fix-test loop discipline:
[serve-and-iterate.md](references/serve-and-iterate.md).

### Parity/coherence failure (invoke `debug-model`)

The server can start but produce wrong output: gibberish, a wrong greedy
token at index K, high logit cosine with the wrong argmax, or coherent text
for N tokens that then diverges. When that happens, stop adding scalar
output taps and load the [`debug-model`](../debug-model/SKILL.md) skill.

That skill handles wrong output from a model that loads and runs. Its
workflow requires these steps:

1. HF sanity-check on the same prompt + checkpoint
2. Per-layer HF vs MAX tensor-dump comparators (not `F.print` eyeballing)
3. Parallel investigation agents with numerical verification before recompile
4. Serve-vs-pipeline bisect when dumps match but generated text diverges

Use `import-model` for bring-up scaffolding and gates. Use
`debug-model` for the divergence hunt itself.

#### Quick logit probe (first 5 minutes only)

Before you build dumpers, run a fast sanity check:

```bash
pixi run python scripts/compare_layers.py <HF_MODEL_ID> \
  --port 8000 \
  --prompt "The capital of France is"
```

Requires `pixi run max serve` with `--custom-architectures <port_dir>` on
the same port. The probe reads logprobs, which the overlap scheduler
rejects. MAX enables that scheduler unless `arch.py` sets
`supports_overlap_scheduler=False`, so when `arch.py` leaves it on, add
`--no-enable-overlap-scheduler --force` to the serve command. This script
prints HF-only layer stats and compares top-1 logprob at the prompt. See
[layer-by-layer-debugging.md](references/layer-by-layer-debugging.md) for
flag details.

If logprobs diverge or output is garbage, switch to
`debug-model`. Don't iterate with manual output taps alone. The symptom
catalog in [divergences.md](references/divergences.md) still applies once the
comparator localizes the failing layer.

### Check against Hugging Face

Run the model end-to-end with pretrained weights, then run HF on the same
prompt with greedy sampling. On the MAX side, use the dtype that matches the
weight encoding the model supports (most models ship bfloat16). Run the HF
side in float32 when it fits: a bfloat16 reference adds rounding of its own,
and a correct bfloat16 port can drift from it within a few tokens.

At the first token where the outputs differ, look at HF's two top logprobs.
When they're within about 0.1 nats, the models tie there, and either token is
correct. A divergence with a clear HF winner in the first tokens usually
means a tokenizer or chat-template mismatch, a dtype mismatch with the
released weights, or nonzero MAX sampling.

When the outputs match, or differ first at a tie, the port is done **for
greedy text**. The other completion criteria depend on the validation depth
picked during planning. Pick a tier from smoke to logit parity.

Full HF-comparison recipe, divergence triage, and the validation-tier
table: [validation-tiers.md](references/validation-tiers.md).

---

## Common pitfalls

Use [pitfalls.md](references/pitfalls.md) as an index: find your symptom, then
load the one category file (config, weights, module, or serving), and
[rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md) for the
docstring audit specifically. The most common pitfalls:

- **Serving the scaffold.** Don't serve or verify until the module
  implements every delta in `<slug>.py`, because the scaffolded module runs
  the donor's math.
- **Renames leave donor docstrings intact.** In donor files you copy
  (`--full-copy` or by hand), class names get renamed, but docstrings and
  comments still describe the donor. Rewrite them and run the audit grep
  before declaring the implementation done.

## Tests and CI

When you add `pytest` tests for the ported model, minimize the number of
compilations per file. Compile once via a module-scoped fixture and reuse it
across `@pytest.mark.parametrize` cases. For files that must compile
different modules, parallelize them with Bazel `shard_count` so the file
stays whole. Full patterns and examples:
[tests-and-ci.md](references/tests-and-ci.md).
