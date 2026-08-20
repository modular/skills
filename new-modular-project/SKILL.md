---
name: new-modular-project
description: Creates a new Mojo or MAX project. Use when wanting to start a new Mojo or MAX project, initializing the `pixi` or `uv` environment to use Mojo or MAX, or when the user wants to begin a new Mojo or MAX project from scratch.
---

When the user wants to create a new project, first infer as many options as
possible from the user's request (for example, "new Mojo project" means
type=Mojo, "called foo" means name=foo). Then use a structured multiple-choice
prompt (not plain text) to gather only the **remaining unspecified** options in
a single interaction. Do NOT ask about options the user has already provided or
implied. The options to determine are:

- **Project name**: ask if the user hasn't specified one.
- **Type of project**: Mojo or MAX (infer from context if the user said "Mojo
  project" or "MAX project").
- **Environment manager**: `pixi` (recommended) or `uv`.
- **uv project type** (only when the environment manager is `uv`): full uv
  project (`uv init` + `uv add`, recommended) or quick uv environment
  (`uv venv` + `uv pip install`, lighter weight).
- **Channel**: nightly or stable. Default to nightly for MAX projects and
  stable for Mojo projects, and ask only if the user hasn't implied one.

Then follow the appropriate section below (`pixi` or `uv`) to initialize the
project and choose `max` or `mojo` as appropriate. Don't pin a version: each
channel already resolves to the right one.

MAX and Mojo ship together but number their releases differently, so their
version strings don't look alike. On the stable channel, `max` is `26.5`
while `mojo` is `1.0.0`; on nightly they're `26.6.0.dev*` and `1.1.0.dev*`.
That's expected, not a mismatch.

> [!NOTE]
> Don't look for or use `magic` for Mojo or MAX projects; it's no longer
> supported. Pixi has fully replaced its capabilities.

---

## System prerequisites

Mojo requires a C linker for compilation. Install one if not already present:

| OS            | Command                                                    |
|---------------|------------------------------------------------------------|
| Ubuntu/Debian | `sudo apt install gcc`                                     |
| Fedora/RHEL   | `sudo dnf install gcc`                                     |
| macOS         | `xcode-select --install`                                   |
| Windows       | Install WSL2 first (see Windows users), then install `gcc` |

**Windows users**: Mojo doesn't run natively on Windows.
Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
(`wsl --install` in PowerShell), then follow the Linux instructions
inside your WSL environment.

---

## Pixi (recommended)

Pixi manages Python, Mojo, and other dependencies in a reproducible
manner inside a controlled environment.

First, determine whether `pixi` is installed. If it isn't available at the
command line, install it using the latest instructions at
<https://pixi.prefix.dev/latest/#installation>.

After installing `pixi`, you may need to add it to the local shell environment.

### Nightly

```bash
# New project
pixi init [PROJECT] \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd [PROJECT]
pixi add [max / mojo]
pixi shell

# Existing project - add to pixi.toml channels first:
# [workspace]
# channels = ["https://conda.modular.com/max-nightly/", "conda-forge"]
pixi add [max / mojo]
```

### Stable

```bash
# New project
pixi init [PROJECT] \
  -c https://conda.modular.com/max/ -c conda-forge \
  && cd [PROJECT]
pixi add [max / mojo]
pixi shell

# Existing project
pixi add [max / mojo]
```

### Python-using projects

If your project uses Python libraries with Mojo:

```bash
pixi add python
pixi add requests           # conda-forge packages
pixi add --pypi some-pkg    # PyPI-only packages
```

---

## uv

`uv` is a fast and very popular package manager, familiar to developers coming
from a Python background. It also works well with Mojo projects.

### Nightly (project)

```bash
uv init [PROJECT] && cd [PROJECT]
uv add [max / mojo] \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

### Stable (project)

```bash
uv init [PROJECT] && cd [PROJECT]
uv add "max[all]"
```

This resolves from PyPI. The `all` extra pulls in `mojo` and the rest of the
MAX stack, so the MAX and Mojo versions always match.

### Nightly (quick environment)

```bash
mkdir [PROJECT] && cd [PROJECT]
uv venv
uv pip install [max / mojo] \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

### Stable (quick environment)

```bash
mkdir [PROJECT] && cd [PROJECT]
uv venv
uv pip install "max[all]"
```

When using `uv`, you can use `max` or `mojo` directly by working within the
project environment:

```bash
 source .venv/bin/activate
```

---

## pip

Standard Python package manager.

### Nightly

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --pre [max / mojo] \
  --extra-index-url https://whl.modular.com/nightly/simple/
```

Use `--extra-index-url`, not `--index-url`. The latter replaces PyPI, and the
nightly index doesn't carry third-party dependencies like `numpy`, so `pip`
backtracks through every `max` version instead of reporting a clear error.

### Stable

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "max[all]"
```

As with `uv`, the `all` extra installs `mojo` as well, so the MAX and Mojo
versions always match.

---

## Conda

For `conda` and `mamba` users.

### Nightly

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max-nightly/ [max / mojo]
```

### Stable

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max/ [max / mojo]
```

---

## Version alignment with MAX

If using MAX with custom Mojo kernels, both must come from the same channel.
Don't compare their version numbers: MAX and Mojo number releases
differently, so a matching pair looks mismatched (stable is `max` `26.5`
with `mojo` `1.0.0`).

```bash
# Check that both came from the same channel
pixi list | grep -E "^(max|mojo)\b"
```

Or, instead install `max[all]` (with pip/uv) or `max-all` (with conda/pixi):

```bash
uv add "max[all]"
```

```bash
pixi add max-all
```

Installing `max` with "all" optional dependencies instead of installing `max`
and `mojo` separately will ensure that the `max` and `mojo` versions always
match. Mixing versions between the two causes kernel compilation failures.

---

## References

- [Mojo Installation Guide](https://mojolang.org/install)
- [Mojo Stable Docs](https://mojolang.org/docs/)
- [Mojo Nightly Docs](https://mojolang.org/nightly/docs/)
