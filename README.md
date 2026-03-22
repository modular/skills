# Modular Skills

These are the official AI agent skills from [Modular](https://www.modular.com/)
that encapsulate modern best-practices for working with the Modular Platform,
including MAX🧑‍🚀 and Mojo🔥. They allow any AI coding agent to become fluent in
developing new Mojo projects and more.

The skills are structured to follow the
[Agent Skills Standard](https://agentskills.io/specification).

## Installation

### Installing via `npx`

With [node.js installed](https://nodejs.org/en/download), you can install all
skills using a single command:

```bash
npx skills add modular/skills
```

Individual skills can also be installed in isolation:

```bash
npx skills add modular/skills --skill new-modular-project
```

This will install the latest version of the Modular skills into the appropriate
location for your AI coding agent. These skills will be updated to the latest
version on a global skill update:

```bash
npx skills update
```

### Manual installation

Clone this repository:

```bash
git clone https://github.com/modular/skills.git
```

and then copy or symbolically link the individual skills into the relevant
location in your AI coding agent's configuration directory. For Claude Code,
that would be `~/.claude/skills/`. Refer to your specific agent's documentation
for where this lives.

## Skills

### `new-modular-project`

[This skill](new-modular-project/SKILL.md) provides a wizard-like experience
for creating a new MAX or Mojo project, letting your AI agent install the right
tools and modules you need to get started. This is triggered on asking your
agent to help you begin a new Mojo or MAX project. You can either provide all
the needed details upfront for such a project (working against nightly or
stable, what package manager to use, etc.) or your agent will prompt you for
any missing information.

### `mojo-syntax`

[This skill](mojo-syntax/SKILL.md) adjusts the pretrained behavior of many
coding models around generating Mojo code to overcome incorrect assumptions and
allow them to generate correct modern Mojo syntax. This skill should be hooked
in whenever an agent is writing Mojo code.

### `mojo-gpu-fundamentals`

[This skill](mojo-gpu-fundamentals/SKILL.md) builds upon the fundamentals in
`mojo-syntax` and makes sure that the correct modern patterns for programming
GPUs using Mojo are followed. It is activated when Mojo code targeting an
accelerator is being generated. This skill does not go into
architecture-specific optimizations, but covers general patterns of how GPUs
are programmed using Mojo.

### `mojo-python-interop`

[This skill](mojo-python-interop/SKILL.md) pairs with `mojo-syntax` to handle
cases where either Mojo works with Python or Python calls into Mojo. It is
triggered when Python types are used Mojo or a Python module needs to interact
with Mojo code. Many capabilities of Mojo - Python interoperability are fairly
new, and existing coding agents don't handle them correctly without guidance.

### `mojo-stdlib-contributing`

[This skill](mojo-stdlib-contributing/SKILL.md) captures patterns and pitfalls
for contributing to the Mojo standard library, distilled from 30+ reviewed PRs.
It covers process (GitHub issues before new APIs, draft PRs), assertion
semantics (`assert_mode="safe"` is intentional), benchmark patterns, testing
requirements, SIMD/memory safety, and common reviewer feedback. Use when
modifying code under `mojo/stdlib/`.

## Examples

Once these skills are installed, you can use them for many common tasks.
Examples include:

### Starting a new Mojo project

```text
I'd like to create a new Mojo project named "my-cool-library".
```

### Translating CUDA C++ code to Mojo

```text
A CUDA kernel is present in `../example`, please create a new Mojo project that implements that same kernel.
```

For several of these skills, your AI agent may prompt you for more information
to clarify your objectives and to make sure the right tools and patterns are
used.

## License

Apache 2.0 — See [LICENSE](./LICENSE) file for details.
