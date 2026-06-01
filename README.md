# b1like

`b1like` is a Cobaya likelihood package for BK component-separation style
bandpower likelihoods.

## Installation

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

The Wigner-d routines are built as a compiled extension during installation.
The extension is compiled from the Cython-generated C wrapper and the Wigner C
implementation. This routine is only used for moment-expansion foreground models.


After installation, Cobaya can import the public components as:

```yaml
likelihood:
  b1like.likelihood.BKCompLike:

theory:
  b1like.theory.FixedLCDM:
```


## Layout

```text
b1like/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── setup.py
├── b1like/
│   ├── __init__.py
│   ├── likelihood.py
│   ├── theory.py
│   ├── wignerd.py
│   ├── BKCompLike.yaml
│   ├── fixlcdm_spec.npz
│   ├── c/
│   │   ├── cwignerd.c
│   │   ├── wignerd.c
│   │   └── wignerd.h
│   └── dev/
│       ├── __init__.py
│       └── bpcm.py
└── scripts/
    └── ...
```

## Library Code

Installable library code lives under `b1like`.

The public Cobaya-facing components are:

- `b1like.likelihood`: the likelihood classes.
- `b1like.theory`: the fixed-spectrum theory provider.
- `b1like.wignerd`: high-performance Wigner-d quadrature.

`BKCompLike.yaml` is a Cobaya component-default file. It defines defaults of the
`BKCompLike` likelihood. `FixedLCDM` keeps its small set of defaults in Python.


Development and data-preparation utilities live under `b1like/dev`.
These modules (`bpcm.py`) supports BPCM construction and Cobaya dataset writing. The public likelihood and
theory classes should not depend on `dev`.

## Development

This repository is intended to be a library of likelihood and theory
components, not a collection of user-specific runs. Keep committed changes
focused on reusable package code, public defaults, tests, documentation, and
small reference data that are required for the installed package to work.

Do not commit generated data products, Cobaya sampler outputs, user-specific
configuration files, machine-local paths, private run YAMLs, or scratch scripts.
Local analysis state should live outside the repository or under `local/`, which
is ignored by git.

Development and data-preparation utilities may live under `b1like/dev` when
they provide reusable support code for building likelihood inputs. Public
likelihood and theory components should not depend on `b1like.dev`.

The top-level `scripts/` directory, if present, is for maintained examples,
templates, and lightweight run orchestration. Scripts committed there should be
portable, documented enough to reuse, and free of machine-specific paths or
generated outputs.

### Code Style and Pre-commit

Python code should follow the Ruff lint and format settings in `pyproject.toml`.
Keep changes small, readable, and consistent with the surrounding module style.

Before committing, install and run the configured pre-commit hooks:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The hooks check whitespace, YAML files, large added files, end-of-file newlines,
and Ruff lint/format rules. If a hook rewrites files, review the diff and rerun
the hooks before committing.
