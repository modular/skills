---
name: serve-model
description: >
  Serve a model with MAX's `max serve` command: set up the environment (pixi or
  uv with the max-nightly conda channel / nightly wheel index), point the server
  at a Hugging Face repo or local checkpoint, target a custom architecture with
  `--custom-architectures`, and pick the right serve flags for the model.
  Use this whenever the user wants to run, launch, start, or host a model on MAX,
  bring up an OpenAI-compatible endpoint, serve a custom/ported architecture,
  debug a `max serve` startup failure, or figure out which serve flags
  (devices, quantization-encoding, max-length, task, trust-remote-code) a given
  model needs, even if they don't say "max serve" by name.
compatibility: Requires a pip or pixi MAX (nightly) install and a Hugging Face repo or local checkpoint; serves on GPU by default, CPU for float32 checkpoints.
---

# Serve a model with MAX

`max serve` launches an OpenAI-compatible HTTP server for a model. It handles
tokenization, batching, KV cache, and the whole serving stack. You point it at
a checkpoint and, if the model isn't built into MAX, at a custom architecture
package. This skill takes you from "no environment" to "server answering
requests," and helps you choose flags that fit the specific model instead of
guessing.

The guiding principle: **start from the smallest command that could work, then
add flags only when the model or the hardware forces you to**. MAX auto-detects
most things (dtype, sequence length, device defaults). Over-specifying flags is
the most common way people turn a working serve into a broken one.

**Use this skill when** you want to run, launch, or host a model on MAX: bring
up an OpenAI-compatible endpoint, serve a built-in or a custom/ported
architecture, or debug a `max serve` startup failure.

**Do not use this skill when** the model isn't implemented in MAX yet (no
working `arch.py`, graph, and weights). That's a bring-up task: use
`import-model` to port the architecture, and `debug-model` if it serves but the
output is wrong. This skill runs an existing model; it doesn't author one.

## References

| File                                                           | Read when                                                                                                  |
|----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| [references/custom-arch.md](references/custom-arch.md)         | Serving a custom architecture: the `arch.py`-to-flags mapping, encoding and device, and serve-time gotchas |
| [references/flags.md](references/flags.md)                     | Choosing any serve flag beyond `--model`, `--devices`, `--quantization-encoding`, and `--max-length`       |
| [references/troubleshooting.md](references/troubleshooting.md) | A `max serve` startup failure or a cryptic error                                                           |

Read the reference for what you're doing, not all of them upfront.

## Fast path (custom architecture): do this first

If MAX is already installed and you have a working custom-arch package, this is
the whole job in four calls. Don't hand-read `arch.py` and `config.json` and
reason about flags yourself. The bundled inspector does exactly that and prints
a ready-to-run command plus the reasoning:

```bash
# 1. Get the recommended command + notes (reads arch.py + config.json for you).
python <skill>/scripts/suggest_serve_command.py \
  --custom-architectures /abs/path/to/my_arch --model <hf-repo-or-path>

# 2. Launch it (add `pixi run` if in a pixi project). On a REMOTE box, wrap with
#    `setsid ... </dev/null` so it survives the SSH session:
setsid <the suggested command> </dev/null > /tmp/max-serve.log 2>&1 &

# 3. Wait for readiness in ONE call (fails fast on a crash; ~10 min budget for
#    a cold compile, advancing `Still compiling` heartbeats are normal):
timeout 600 bash -c 'until grep -qE "Server ready|Uvicorn running" /tmp/max-serve.log; do
  grep -qiE "Traceback|CRASHED|Error building|cannot be found|not found in registry" /tmp/max-serve.log && { echo SERVE_FAILED; tail -30 /tmp/max-serve.log; exit 1; }
  sleep 3; done' && echo SERVE_READY

# 4. Confirm with one request (model field must equal --served-model-name):
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"my_arch","messages":[{"role":"user","content":"The capital of France is"}],"max_completion_tokens":32}'
```

**Read the inspector's `# notes:`**. That's where the traps surface (a
`default_encoding` that disagrees with the checkpoint, a `name` colliding with a
built-in, GPU-only encodings, MoE). If the command works and the output is
coherent, you're done. Only drop into the detailed steps below when a note or a
failure tells you to. The rest of this doc is the "why" behind what the
inspector does and what to do when it isn't enough.

## 1. Make sure MAX is installed

The user needs a `max` binary from the **nightly** build. Check first, and don't
reinstall if it's already there (the `max serve` command works only if the
project includes the `max[serve]` or `max-serve` extra dependencies):

```bash
max serve --help           # already in a MAX env?
pixi run max serve --help   # or inside a pixi project
```

If the `max serve` command isn't available, set up an environment. **pixi** is
the default; the key detail is the conda channel
`https://conda.modular.com/max-nightly/` plus `conda-forge`:

```bash
# pixi (conda channels)
curl -fsSL https://pixi.sh/install.sh | sh
pixi init my-max-project \
  -c https://conda.modular.com/max-nightly/ -c conda-forge && cd my-max-project
pixi add max-serve
```

If a `pixi.toml` already exists, the channels line must read exactly:

```toml
[workspace]                                # or [project] on older pixi
channels = ["https://conda.modular.com/max-nightly/", "conda-forge"]
```

Some users prefer **uv**, which pulls MAX from Modular's wheel index instead of
conda:

```bash
# uv (pip wheels)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init my-max-project && cd my-max-project
uv venv && source .venv/bin/activate
uv add "max[serve]" --index https://whl.modular.com/nightly/simple/ --prerelease allow
```

After a pixi setup, prefix commands with `pixi run` (or enter `pixi shell`).
After uv, activate the venv (`source .venv/bin/activate`) and call `max`
directly. The rest of this skill writes bare `max serve ...`; add `pixi run` in
front when you're in a pixi project and haven't entered the shell.

## 2. Is this a built-in model or a custom architecture?

MAX ships with many architectures. If the model's architecture is already
supported, you don't need `--custom-architectures` at all, just `--model`.

```bash
max list                 # every registered architecture + example repo IDs
```

Match the checkpoint's `config.json::architectures[0]` (for example
`LlamaForCausalLM`, `Qwen2ForCausalLM`) against that list.

- **Listed**: built-in. Skip to step 4 and omit `--custom-architectures`.
- **Not listed**: you need a custom architecture package (step 3). If the user
  doesn't have one yet, this skill can't manufacture it; that's a *bring-up*
  task (implementing `arch.py`, `model.py`, the graph, weight adapters). Point
  them at the model bring-up workflow and stop here.

## 3. Target a custom architecture

A custom architecture is a Python **package** (a directory with `__init__.py`)
that exposes a top-level `ARCHITECTURES` list of `SupportedArchitecture`
instances. You pass the package with `--custom-architectures`; MAX imports it,
registers each arch by `name`, and on each request matches the checkpoint's
`config.json::architectures[0]` against a registered `name`.

**Read the package before you write the command**. The architecture package is
the source of truth for most of the flags, so don't guess them. Open two files:

- **`arch.py`**: the `SupportedArchitecture(...)` call. Its `default_encoding`
  is the encoding to serve with; `supported_encodings` is the set your
  `--quantization-encoding` must belong to; `name` must match the checkpoint;
  `multi_gpu_supported` says whether you can shard; a `chat_template.jinja` in
  the package means you'll need `--chat-template`.
- **`config.json`** (the checkpoint): `architectures[0]` must equal `arch.py`'s
  `name`; `max_position_embeddings` caps `--max-length`; `torch_dtype` /
  `quantization_config` should agree with `default_encoding`; MoE fields
  (`num_experts` etc.) hint that you want `--device-graph-capture`.

**`references/custom-arch.md` is the detailed guide**: a field-by-field
`arch.py`-to-flags mapping, the encoding-to-device table, and the non-obvious
serve-time gotchas (overlap-scheduler vs logprobs, the port-8001 metrics
collision, trust-remote-code). Read it whenever you're serving a custom arch.

Point `--custom-architectures` at the package **directory** (an absolute path is
safest for scripts and remote hosts). The `IMPORT_PATH:MODULE_NAME` colon form
also works, but the directory path is what most tooling uses:

```bash
max serve --model <hf-repo-or-path> --custom-architectures /abs/path/to/my_arch
```

Run these three checks up front rather than reading a stack trace; they head off
almost every custom-arch serve failure:

1. `name=` in the `SupportedArchitecture` **exactly** equals
   `config.json::architectures[0]`.
2. `supported_encodings` includes the encoding the checkpoint actually ships
   (a bf16 checkpoint needs `bfloat16`, a GPTQ checkpoint needs `gptq`, etc.).
3. `weight_adapters` has an entry for the checkpoint's weight format
   (`WeightsFormat.safetensors` for `.safetensors`, `WeightsFormat.gguf` for
   `.gguf`).

If the model *isn't* already a working custom-arch package (the graph, weight
adapters, and config aren't implemented yet), that's a **bring-up** task, not a
serving one. This skill serves an existing package; it doesn't author one.

## 4. Build the serve command (default-first)

Start with the minimal command and run it. Let MAX auto-detect the rest.

```bash
max serve --model <hf-repo-or-path> [--custom-architectures ...]
```

That already binds `0.0.0.0:8000`, serves on the GPU when one is present (CPU
otherwise), infers dtype from the checkpoint, and sets `max_length` from
`max_position_embeddings`. GPU is the default, so you don't pass `--devices` for
the common single-GPU case. For a lot of models on a single GPU, that minimal
command is the whole job.

Add flags only for a concrete reason. The three you'll reach for most:

| Flag                      | Add it when                                                                                 | Example                                                                             |
|---------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `--devices`               | You must pin specific GPUs, shard across GPUs, or force CPU (GPU is already the default).   | `--devices gpu:0` · `--devices gpu:0,1,2,3` · `--devices gpu:all` · `--devices cpu` |
| `--quantization-encoding` | The repo has multiple formats, or auto-detect picks the wrong one.                          | `--quantization-encoding bfloat16`                                                  |
| `--max-length`            | You want a shorter context than the model's max (saves KV memory) or need to cap it to fit. | `--max-length 4096`                                                                 |

For everything else (device memory, batch size, task selection, sliding window,
trust-remote-code, multi-GPU parallelism, speculative decoding) see
**`references/flags.md`**. Read it before adding any flag you're unsure about;
it explains what each one does and when *not* to set it.

**How to decide, in order**:

1. **Default first**. Try the minimal command. Auto-detection is usually right
   for built-ins, and GPU is the default device.
2. **For a custom arch, read the package**. `arch.py`'s `default_encoding` is
   your `--quantization-encoding`. That encoding constrains the device: fp8,
   fp4, and gptq are GPU-only, and GPU is already the default, so you don't add
   `--devices` for them (see `references/custom-arch.md`). Cap `--max-length` at
   `config.json::max_position_embeddings`. Add `--trust-remote-code` if the
   checkpoint ships custom modeling files, and `--chat-template` if the package
   bundles one. These aren't guesses; you read them off the package and config.
3. **Inspect further when a choice is load-bearing**. For unusual properties
   (sliding window, partial RoPE, MoE routing), check `config.json` and map
   findings to flags with `references/flags.md`.
4. **Ask when unsure**. If a decision depends on something you can't see (which
   GPUs are free, how much context they need, CPU vs GPU), ask the user rather
   than guessing. A wrong `--devices` or dtype fails slowly and confusingly; a
   quick question is cheaper.

## 5. Launch and confirm it works

Launch (backgrounded, with a log you can tail):

```bash
max serve --model <hf-repo-or-path> [flags] > /tmp/max-serve.log 2>&1 &
```

If you're launching on a **remote box over SSH**, a bare `&` dies when the SSH
session closes, and the compile can outlast your connection. Fully detach it:

```bash
setsid max serve --model <hf-repo-or-path> [flags] </dev/null > /tmp/max-serve.log 2>&1 &
```

Wait for readiness with a single call that watches the log's heartbeat and fails
fast on a crash instead of blocking for the full timeout:

```bash
timeout 600 bash -c 'until grep -qE "Server ready|Uvicorn running" /tmp/max-serve.log; do
  grep -qiE "Traceback|CRASHED|Error building|cannot be found|not found in registry" /tmp/max-serve.log && { echo SERVE_FAILED; tail -30 /tmp/max-serve.log; exit 1; }
  sleep 3; done' && echo SERVE_READY
```

The server prints this line when it's ready:

```output
Server ready on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Large models compile on first launch. A quiet gap with
`Still compiling model (Ns elapsed)` heartbeats is normal, not a hang: the
elapsed counter is the liveness signal. Wait while it advances, and only treat
the serve as stuck if the counter freezes (or the log's mtime stops moving while
the process is alive).

Confirm health, then send a real request:

```bash
curl -s http://localhost:8000/v1/health          # 200 when ready
curl -s http://localhost:8000/v1/models           # served model name

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<hf-repo-or-path>",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_completion_tokens": 32
  }'
```

The `model` field in the request must match what you passed to `--model` (or
`--served-model-name` if you overrode it). Read the response: a bring-up isn't
done just because the server is green. Check that the text is actually coherent,
not repetition or gibberish. If it serves but the output is wrong, that's a
parity/correctness problem, not a serving problem.

## Cache weights and compilation for faster re-serves

The first serve of a model downloads its weights and compiles the graph, which
is the slow part. Both results are cached, so later serves of the same model are
much faster.

- **Weights** download through `huggingface_hub` into the shared Hugging Face
  cache (`~/.cache/huggingface` by default; set `HF_HOME` to relocate it).
  Re-serving the same repo reuses the cached weights with no re-download.
- **Compilation** is cached too. To warm both caches ahead of time (before a
  demo or deployment) so the first `max serve` skips the download and the
  `Still compiling` wait, run `max warm-cache` first:

  ```bash
  max warm-cache --model <hf-repo-or-path> [--custom-architectures /abs/path/to/my_arch]
  ```

  The compiled artifact (MEF) is platform-specific, so warm the cache on the
  same hardware (or pass `--target`, for example `cuda:sm_90`, to compile for a
  deployment target from a different host).

## Troubleshooting

Match the symptom against **`references/troubleshooting.md`**. It covers the
startup failures that look cryptic but have one-line fixes (encoding mismatch,
`architectures[0]` name mismatch, `trust_remote_code`, OOM at load, port in
use, wrong device routing). Read the serve log first; the real error is usually
a few lines above the final traceback.
