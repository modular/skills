# Mojo Installation Guide

Install Mojo using pixi (recommended), uv, pip, or conda. Choose **stable** for production or **nightly** for latest features.

> **Note:** Version numbers in this file reflect `metadata.json` as the source of truth.

## Create a New Mojo Project

When the user wants to create a new Mojo project, interactively determine:
1. **Project name** — ask if not specified
2. **Environment manager** — Pixi (recommended) or uv
3. **If uv**: **Project type** — full uv project (`uv init` + `uv add`, recommended) or quick uv environment (`uv venv` + `uv pip install`, lighter weight)
4. **Channel** — stable (production) or nightly (latest features)

Then follow the appropriate section below (Pixi or uv) to initialize the project.

---

## Pixi (Recommended)

Pixi manages Python, Mojo, and other dependencies in a reproducible
manner inside a controlled environment.

First, determine if `pixi` is installed. If it is not available for use at the
command line, install it with

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

You may need to place the `pixi` tool in the local shell environment after
installation if it had not already been installed.

### Nightly

```bash
# New project
pixi init my-project \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd my-project
pixi add mojo
pixi shell

# Existing project - add to pixi.toml channels first:
# [workspace]
# channels = ["https://conda.modular.com/max-nightly/", "conda-forge"]
pixi add mojo
```

### Stable (v26.1.0.0.0)

```bash
# New project
pixi init my-project \
  -c https://conda.modular.com/max/ -c conda-forge \
  && cd my-project
pixi add "mojo==0.26.1.0.0.0"
pixi shell

# Existing project
pixi add "mojo==0.26.1.0.0.0"
```

---

## uv

uv is a fast and very popular package manager, familiar to developers coming
from a Python background. It also works well with Mojo projects.

### Nightly (project)

```bash
uv init my-project && cd my-project
uv add mojo \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

### Stable (project)

```bash
uv init my-project && cd my-project
uv add mojo \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

### Nightly (quick environment)

```bash
mkdir my-project && cd my-project
uv venv
uv pip install mojo \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

### Stable (quick environment)

```bash
mkdir my-project && cd my-project
uv venv
uv pip install mojo \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

When using `uv`, you can use `mojo` directly by working within the project environment:

```bash
 source .venv/bin/activate
```

---

## pip

Standard Python package manager.

### Nightly

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --pre mojo \
  --index https://whl.modular.com/nightly/simple/
```

### Stable

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mojo \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

---

## Conda

For conda/mamba users.

### Nightly

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max-nightly/ mojo
```

### Stable (v26.1.0.0.0)

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max/ "mojo==0.26.1.0.0.0"
```

---

## Version Alignment with MAX

If using MAX with custom Mojo kernels, versions must match:

```bash
# Check alignment
uv pip show mojo | grep Version   # e.g., 0.26.2
pixi run mojo --version           # Must match major.minor (e.g., 0.26.2)
```

Mismatched versions cause kernel compilation failures. Always use the same channel (stable or nightly) for both.

---

## Quick Reference

| Method | Nightly | Stable |
|--------|---------|--------|
| pixi | `pixi add mojo` | `pixi add "mojo==0.26.1.0.0.0"` |
| uv | `uv pip install mojo --index https://whl.modular.com/nightly/simple/ --prerelease allow` | `uv pip install mojo --extra-index-url https://modular.gateway.scarf.sh/simple/` |
| pip | `pip install --pre mojo --index https://whl.modular.com/nightly/simple/` | `pip install mojo --extra-index-url https://modular.gateway.scarf.sh/simple/` |
| conda | `conda install -c conda-forge -c https://conda.modular.com/max-nightly/ mojo` | `conda install -c conda-forge -c https://conda.modular.com/max/ "mojo==0.26.1.0.0.0"` |

---

## References

- [Mojo Installation Guide](https://docs.modular.com/mojo/manual/install)
- [Mojo Stable Docs](https://docs.modular.com/stable/mojo/)
- [Mojo Nightly Docs](https://docs.modular.com/mojo/)
