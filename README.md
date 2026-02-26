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
npx skills add modular/skills --skill mojo-best-practices
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

and then copy or symbolically link this into the relevant location in your AI
coding agent's configuration directory. For Claude Code, that would be
`~/.claude/plugins/skills/`. Refer to your specific agent's documentation for
where this lives.

## Skills

### `mojo-best-practices`

[This skill](mojo-best-practices/SKILL.md) gives your AI coding agent the right
context for working with modern Mojo code, from current syntax to the best ways
to maximize performance on accelerator hardware. Mojo is a rapidly-evolving
language, and this skill helps agents stay up-to-date and develop functional
and fast Mojo code.

Dedicated instructions are also present for starting new Mojo projects from
scratch, installing both stable and nightly versions of the language, and
helping you to translate from other languages.

### `max-best-practices`

[This skill](max-best-practices/SKILL.md) provides the fundamentals for working
with MAX, ranging from AI model construction to serving. This skill is in the
early stages of development and testing.

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
