# Port: `<HF_MODEL_ID>`

Copy this file when you start a bring-up. Fill it in as you work through the
phases in [SKILL.md](../SKILL.md). Prefer data from Hub `config.json` and
`inspect_hf.py` over guesses.

**HF model ID:** `<org/model-name>`
**Slug (your port directory name):** `<slug>`
**Started:** `<YYYY-MM-DD>`
**Operator:**

---

## Phase 1 — Decide & plan

### Guard: native in MAX?

**`config.json` → `architectures[0]`:** `` `<ArchitecturesClassFromConfig>` ``
**`config.json` → `model_type`:** `` `<model_type>` ``

```bash
pixi run python inspect_hf.py <HF_MODEL_ID>
pixi run python list_checkpoint_keys.py <HF_MODEL_ID> --summary
pixi run python list_native_archs.py --match <ArchitecturesClassFromConfig>
pixi run python check_walls.py <HF_MODEL_ID>
```

| Check                       | Result                                |
|-----------------------------|---------------------------------------|
| Registered in MAX?          | yes / no                              |
| MAX directory slug (if yes) | `` `<native_slug>` ``                 |
| `check_walls.py` exit code  | 0 / 1 / 2                             |
| Action                      | serve native / continue with the port |

If registered, stop here:

```bash
pixi run max serve --model <HF_MODEL_ID>
```

### Model card

**Card URL:** https://huggingface.co/`<HF_MODEL_ID>`

**Architecture family (from card):**

**Advertised modifications** (phrases like "we introduce", "unlike prior
models"):

**Paper / blog:**

**Blockers?** (custom CUDA only, FP8-only weights, ALiBi, SSM, etc.)
→ see [recognize-walls.md](recognize-walls.md) if yes.

### `config.json` → MAX API

Paste the `config.json → MAX API` table from `inspect_hf.py`, or fill one
row per key in the Hub file:

| Key                     | Value                         | MAX API                                   | Wired in `MyConfig`? |
|-------------------------|-------------------------------|-------------------------------------------|----------------------|
| `architectures`         |                               | `SupportedArchitecture.name` in `arch.py` |                      |
| `torch_dtype` / `dtype` |                               | `SupportedArchitecture.default_encoding`  |                      |
| (safetensors metadata)  | see `list_checkpoint_keys.py` | confirms shapes / dtypes on disk          |                      |

**Checkpoint metadata** (`list_checkpoint_keys.py --summary`):

| dominant dtype | tensor count | notes |
|----------------|--------------|-------|
|                |              |       |

**Deltas** (keys that need custom nn-layer code, not just config):

1.
2.

### Delta list (HF vs donor MAX arch)

**Chosen `--start-from` slug:** `` `<llama3|qwen3|…>` ``
**Why this donor** (attention shape, MLP, MoE, norm order, RoPE):

| Component          | HF class / behavior | MAX donor | Change needed |
|--------------------|---------------------|-----------|---------------|
| Attention          |                     |           |               |
| MLP / MoE          |                     |           |               |
| Block / norm order |                     |           |               |
| LM head            |                     |           |               |

**HF modeling file:**

```python
import importlib
mod = importlib.import_module("transformers.models.<model_type>.modeling_<model_type>")
print(mod.__file__)
```

**Difference count:** ___ (target ≤ 3 structural deltas; otherwise pick a
different donor)

---

## Phase 2 — Implement

### Scaffold (files only — not the port)

```bash
pixi run python scaffold.py <HF_MODEL_ID> \
  --start-from <max_arch_slug> \
  --output-dir <output_dir>
```

**Port directory (`<port_dir>`):** `` `<output_dir>/<slug>/` `` — pass to
`--custom-architectures` and `run_oss_gates.py --port-dir` (not `<output_dir>/`)

| File                 | Copied | Notes                                     |
|----------------------|--------|-------------------------------------------|
| `arch.py`            | yes    | Donor shell — verify `name=`              |
| `model_config.py`    | yes    | Donor config — rewire                     |
| `<slug>.py`          | yes    | **Donor graph — wrong until implemented** |
| `weight_adapters.py` | yes    | Donor renames — rewrite                   |
| `model.py`           | yes    | Edit only if HF wrapper differs           |

**Do not serve yet.**

### Implement the graph

See [implement-graph.md](implement-graph.md). Every delta row must have
a code change before serving.

| Component                               | Implemented | HF `forward` line ref |
|-----------------------------------------|-------------|-----------------------|
| `model_config.py` (all config keys)     |             |                       |
| `weight_adapters.py` (checkpoint loads) |             |                       |
| Attention                               |             |                       |
| MLP / MoE                               |             |                       |
| Decoder block wiring                    |             |                       |
| Final norm + LM head                    |             |                       |
| `arch.py` (`name=`, encoding)           |             |                       |

**Implementation complete?** all boxes checked per
[implement-graph.md](implement-graph.md#completion-criteria-required-before-serving)

```bash
pixi run python run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>/
```

### Guard: smoke gate

All four checks from [serve-and-iterate.md](serve-and-iterate.md) PASS.

---

## Phase 3 — Verify

### Coherence smoke

**Prerequisite:** implementation complete.

```bash
pixi run max serve --model-path <HF_MODEL_ID> \
  --custom-architectures <port_dir> \
  --quantization-encoding <encoding from arch.py>
```

**Prompt:** `The capital of France is`

| Outcome                             | yes / no | Notes |
|-------------------------------------|----------|-------|
| Server loads without crash          |          |       |
| Output is non-garbage at 16+ tokens |          |       |

If garbage → divergence hunt (re-check deltas — block wiring often missed).

### Logit / layer debug

**Symptom** (from [divergences.md](divergences.md)):

```bash
pixi run python compare_layers.py <HF_MODEL_ID> \
  --slug <slug> --port 8000 \
  --prompt "The capital of France is"

# Or full gate after serve:
pixi run python run_oss_gates.py <HF_MODEL_ID> \
  --port-dir <port_dir>/ --phase verify --slug <slug> --port 8000
```

| Run | top-1 logprob rel_diff | Verdict | Fix applied |
|-----|------------------------|---------|-------------|
| 1   |                        |         |             |
| 2   |                        |         |             |

**Root cause (when found):**

**Manual `ops.output()` taps** (if used):

### Greedy text match

**Prompt:** `The capital of France is`
**max_tokens:** 64
**dtype:** `` `<encoding the model supports — usually bfloat16>` ``

| Side | Output |
|------|--------|
| MAX  |        |
| HF   |        |

**Match?** yes / no (first-token mismatch after divergence-hunt pass → tokenizer
/ dtype / sampling)

---

## Done

- [ ] Phase 2: graph implements HF (not donor shim)
- [ ] Phase 3: logits aligned at test prompt(s)
- [ ] Phase 3: greedy generation matches HF on short prompt
- [ ] Lessons captured (pitfall, divergences entry, or skill patch if reusable)
