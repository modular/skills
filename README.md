[![skills.sh](https://skills.sh/b/modular/skills)](https://skills.sh/modular/skills)

# Modular skills

These are the official AI agent skills from [Modular](https://www.modular.com/)
for working with the Modular Platform, including MAX and Mojo. They follow the
[Agent Skills Standard](https://agentskills.io/specification). Any AI coding
agent can use them to write Mojo, or to import, serve, and measure models on
MAX.

## Install

Pick the instructions for your agent:

### Claude Code

```text
/plugin marketplace add modular/skills
/plugin install skills@modular
```

For a subset, install `max-skills@modular` to work with models on MAX, or
`mojo-skills@modular` to write Mojo.

### Codex and other agents

With [Node.js installed](https://nodejs.org/en/download):

```bash
npx skills add modular/skills
```

This installs the skills into the right location for your agent.

Update them later:

```bash
npx skills update
```

Install one skill at a time:

```bash
npx skills add modular/skills --skill mojo-syntax
```

To install by hand, clone
[the repository](https://github.com/modular/skills) and copy or symlink the
individual skill directories into your agent's skills directory (for Claude
Code, `~/.claude/skills/`).

## Mojo and project skills

Skills that cover writing Mojo and setting up a project:

- [`/new-modular-project`](new-modular-project/SKILL.md): Creates a new Mojo or
  MAX project, setting up the `pixi` or `uv` environment for you.
- [`/mojo-syntax`](mojo-syntax/SKILL.md): Corrects pretrained assumptions so
  your agent writes modern Mojo. Use it whenever an agent writes Mojo.
- [`/mojo-gpu-fundamentals`](mojo-gpu-fundamentals/SKILL.md): Adds the patterns
  for programming NVIDIA, AMD, and Apple silicon GPUs in Mojo. Pairs with
  `/mojo-syntax`.
- [`/mojo-python-interop`](mojo-python-interop/SKILL.md): Handles Mojo calling
  Python and Python calling Mojo, including building Python extension modules.
  Pairs with `/mojo-syntax`.

## Model lifecycle

Skills that take a model from a Hugging Face checkpoint to a deployment you've
verified and measured. They hand off to each other, and each one names the state
it expects the model to be in:

- [`/import-model`](import-model/SKILL.md): Imports a new model architecture
  into MAX from a Hugging Face model ID, scaffolding from a similar registered
  architecture and verifying outputs match. Hands off to `/debug-model` when
  the server runs but the text is wrong.
- [`/serve-model`](serve-model/SKILL.md): Takes you from no environment to a
  running OpenAI-compatible endpoint with `max serve`, choosing the flags a
  model needs rather than guessing: `--devices`, `--quantization-encoding`,
  `--max-length`, `--task`, and `--trust-remote-code`. Use
  `--custom-architectures` for an architecture you ported with
  `/import-model`.
- [`/debug-model`](debug-model/SKILL.md): Takes over once a model loads and
  generates tokens but the output is wrong. Builds tensor-dump comparators and
  bisects serve versus pipeline. For crashes on load, use `/import-model`.
- [`/benchmark-model`](benchmark-model/SKILL.md): Drives load against an
  endpoint you started with `/serve-model` and reports throughput and latency
  (TTFT, TPOT, inter-token latency), plus GPU utilization when it runs on the
  same NVIDIA host as the server. Hand off to `/profile-model` when the numbers
  show a bottleneck.
- [`/profile-model`](profile-model/SKILL.md): Finds where inference time goes
  and whether the GPU is saturated, working cheapest-first: a utilization
  snapshot, a kernel breakdown with `nsys` or `rocprofv3`, then an `ncu` deep
  dive on a single kernel when one dominates.
- [`/eval-model`](eval-model/SKILL.md): Measures task accuracy on standard
  benchmarks: GSM8K, MMLU, HellaSwag, ARC, AIME, GPQA, TruthfulQA, WinoGrande,
  and BABILong. Distinguishes serving failures from wrong model answers. Ask
  for it by name; it won't trigger on its own.

## Examples

Once you install these skills, you can use them for many common tasks.
Examples include:

### Starting a new Mojo project

```text
I'd like to create a new Mojo project named "nvfp4-for-metal".
```

### Translating CUDA C++ code to Mojo

```text
A CUDA kernel is present in `../example`, please create a new Mojo project that implements that same kernel.
```

For several of these skills, your AI agent may prompt you for more information
to clarify your objectives and to make sure it uses the right tools and
patterns.

## License

Apache 2.0. See the [LICENSE](./LICENSE) file for details.
