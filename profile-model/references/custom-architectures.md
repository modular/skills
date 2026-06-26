# Profiling a model loaded with `--custom-architectures`

Use this when the model isn't a built-in MAX architecture but a custom one
loaded with `--custom-architectures <dir>`. The profiling steps in
[SKILL.md](../SKILL.md) still apply — this file covers the extra load-time setup
and the failure modes that don't exist for built-in models.

The path that works on current MAX is
`max generate --profile --no-device-graph-capture --custom-architectures <dir>`
for a *decoder* (text-generation) architecture. It prints the same ranked kernel
table as for built-in models — dense decoders, sparse-MoE models, and
low-bit-quantized models alike. The kernel *mix* differs by model type, but the
command and the way you read the table don't (see
[SKILL.md's "Reading the result"](../SKILL.md#reading-the-result-what-the-kernel-mix-tells-you)).

For example, an NVFP4 sparse-MoE decoder profiles cleanly as a single-GPU
decoder. Its top kernel families are:

- NVFP4 grouped matmul (`*_block_scaled_*_mxf4nvf4`)
- `gemv_split_k` attention projections and `sm100_mha`
- FP4 quant-prep (`*quantize*`, `*block_scales*`)
- MoE routing (`moe_create_indices`, `topk_*`)

For comparable GPU-time / tok-s numbers across runs, re-run `max generate
--profile` on the same model with the same settings rather than reusing a
differently-configured run.

## Prerequisite: the architecture must import on your MAX

A custom architecture is written against whatever MAX version its source pins.
If your installed MAX is newer, the architecture's `from max... import` lines
may point at moved modules and the load fails before profiling. Confirm it
imports first:

```bash
pixi run python -c "import sys; sys.path.insert(0,'<dir_parent>'); \
  import <module>; print([a.name for a in <module>.ARCHITECTURES])"
```

If that raises `ModuleNotFoundError` / `ImportError`, the architecture has
import drift against your MAX — update its imports to your version before
profiling. A `TypeError: ... got an unexpected keyword argument` at *load* time
is deeper *signature* drift — fix the call site. "Imports clean" is the gate for
"can be profiled."

## Run the profiling command

```bash
pixi run max generate \
  --profile --no-device-graph-capture \
  --custom-architectures /abs/path/to/<dir> \
  --model-path <HF_ID> \
  --prompt "Explain what a GPU kernel is." \
  --max-length 512 --num-warmups 1
```

Filled in with a concrete architecture and model, that looks like:

```bash
pixi run max generate \
  --profile --no-device-graph-capture \
  --custom-architectures /home/me/ports/my_arch \
  --model-path my-org/My-Model-7B \
  --prompt "Explain what a GPU kernel is." \
  --max-length 512 --num-warmups 1
```

Per-flag, and why each matters here:

- `--no-device-graph-capture`: without it, MAX compiles a full
  device-graph at `max_batch_size=512`, which can take minutes for a one-shot
  profile. Disable it so you reach the profiled run quickly.
- `--custom-architectures /abs/path/to/<dir>`: point at the architecture
  directory. MAX adds its parent to `sys.path` and imports the dir name as a
  package.
- Do not set `PYTHONPATH=<dir>`. It puts the architecture dir itself on the
  path, so `import <module>` resolves to a graph `.py` file inside it instead of
  the package, which yields
  `module '<module>' has no attribute 'ARCHITECTURES'`. The architecture's own
  relative-import fallbacks resolve correctly without it.
- `--max-length` ≥ the chat-templated prompt length: instruct models expand
  a short prompt to hundreds of tokens via their chat template; if
  `--max-length` is below that you get `PromptTooLongError` (not a profiling
  failure — bump it to 512).

## Model-type gotchas

- Base model (no chat template): raises
  `ValueError: tokenizer.chat_template is not set`. Pass a passthrough template:

  ```bash
  printf '%s' '{% for m in messages %}{{ m["content"] }}{% endfor %}' > /tmp/raw.jinja
  # add: --chat-template /tmp/raw.jinja
  ```

- Repo with custom modeling code: add `--trust-remote-code`.
- Name collision with a built-in architecture: raises
  `ValueError: Refusing to override existing architecture for 'X'`. This happens
  when the custom architecture name shadows a built-in (for example a
  llama-derived architecture registering `LlamaForCausalLM`). Genuinely-custom
  names don't collide; for a shadowing architecture, profile the built-in
  directly instead.

## `max serve --custom-architectures` for embeddings is broken (current MAX)

Serving a *custom embeddings* architecture via `max serve
--custom-architectures` fails with `Architecture 'X' not found in registry`:
`serve` auto-detects the pipeline task **before** it imports the custom
architectures, and the `--task` flag doesn't set `pipeline_config.task` for
serve. The model worker is also a re-exec'd `bin/max` subprocess, so an
in-process pre-registration doesn't reach it.

Consequence for profiling: **decoder architectures profile fine via
`max generate`** (it imports custom architectures in time); **you can't serve
custom encoder/embeddings architectures on current MAX**, so the
serving flows in `nsys.md` don't apply to them until MAX fixes the ordering.
Profile decoders with `max generate --profile`; for embeddings, fall back to the
`gpu_snapshot.py` utilization check around whatever load you can drive, or wait
for the serve fix.
