# MAX Installation Guide

Install MAX using pixi (recommended), uv, pip, or conda. Choose **stable** for production or **nightly** for latest features.

> **Note:** Version numbers in this file reflect `metadata.json` as the source of truth.

## Create a New MAX Project

When the user wants to create a new MAX project, interactively determine:
1. **Project name** — ask if not specified
2. **Environment manager** — Pixi (recommended) or uv
3. **Channel** — stable (production) or nightly (latest features)

Then follow the appropriate section below (Pixi or uv) to initialize the project.

## Version Check

```bash
max version                    # Direct check
pip show modular | grep Version  # In pip environment
pixi list | grep modular       # In pixi environment
```

**Current versions:** Stable v26.1.0.0.0 | Nightly v26.2

---

## Pixi (Recommended)

Pixi manages both Python and native dependencies in a reproducible environment.

### Nightly

```bash
# New project
pixi init my-project \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd my-project
pixi add modular
pixi shell

# Existing project - add to pixi.toml channels first:
# [workspace]
# channels = ["https://conda.modular.com/max-nightly/", "conda-forge"]
pixi add modular
```

### Stable (v26.1.0.0.0)

```bash
# New project
pixi init my-project \
  -c https://conda.modular.com/max/ -c conda-forge \
  && cd my-project
pixi add "modular==26.1.0.0"
pixi shell

# Existing project
pixi add "modular==26.1.0.0"
```

---

## uv

uv is a fast Python package manager. Good for Python-only workflows.

### Nightly

```bash
uv init my-project && cd my-project
uv venv && source .venv/bin/activate
uv pip install modular \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

### Stable

```bash
uv init my-project && cd my-project
uv venv && source .venv/bin/activate
uv pip install modular \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

---

## pip

Standard Python package manager.

### Nightly

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --pre modular \
  --extra-index-url https://whl.modular.com/nightly/simple/
```

### Stable

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install modular \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

---

## Conda

For conda/mamba users.

### Nightly

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max-nightly/ modular
```

### Stable

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max/ modular
```

---

## Version Alignment

**Critical:** MAX Python package and Mojo versions must match. Mismatched versions cause kernel compilation failures.

```bash
# Check alignment
pip show modular | grep Version   # e.g., 26.2.0
mojo --version                    # Must match major.minor (e.g., 0.26.2)
```

If versions don't match, upgrade both to the same channel (stable or nightly).

---

## Quick Reference

| Method | Nightly | Stable |
|--------|---------|--------|
| pixi | `pixi add modular` | `pixi add "modular==26.1.0.0"` |
| uv | `uv pip install modular --index https://whl.modular.com/nightly/simple/ --prerelease allow` | `uv pip install modular --extra-index-url https://modular.gateway.scarf.sh/simple/` |
| pip | `pip install --pre modular --extra-index-url https://whl.modular.com/nightly/simple/` | `pip install modular --extra-index-url https://modular.gateway.scarf.sh/simple/` |
| conda | `conda install -c conda-forge -c https://conda.modular.com/max-nightly/ modular` | `conda install -c conda-forge -c https://conda.modular.com/max/ modular` |

---

## References

- [MAX Get Started](https://docs.modular.com/max/get-started)
- [MAX Stable Docs](https://docs.modular.com/stable/max/)
- [MAX Nightly Docs](https://docs.modular.com/max/)
