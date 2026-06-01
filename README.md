# b1like

`b1like` is a Cobaya likelihood package for BK component-separation style
bandpower likelihoods.

## Installation

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

The Wigner-d routines are built as a compiled extension during installation.
The extension is compiled from the generated C wrapper and the Wigner C
implementation, so Cython is not required for installation.

For local data-preparation utilities that need optional dependencies:

```bash
python -m pip install -e ".[dev]"
```

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
│   │   ├── wignerd.h
│   │   └── wignerd.pyx
│   └── dev/
│       ├── __init__.py
│       └── bpcm.py
└── scripts/
    └── ...
```

## Library Code

Installable library code lives under `b1like`.

The public Cobaya-facing components are:

- `b1like.likelihood.BKCompLike`: the likelihood class.
- `b1like.theory.FixedLCDM`: the fixed-spectrum theory provider.
- `b1like.wignerd.GaussLegendreQuadrature`: high-performance Wigner-d quadrature.
- `b1like.wignerd.get_product_spectra`: product-spectrum helper using the compiled Wigner-d backend.

`BKCompLike.yaml` is a Cobaya component-default file. It may define component
defaults, parameter metadata, labels, priors, and proposal widths. It must not
contain local paths, run names, selected map sets, generated dataset filenames,
or machine-specific configuration. `FixedLCDM` keeps its small set of defaults
in Python.

Private implementation helpers should use a leading underscore if they are added
later. Public users should not import from underscore-prefixed modules directly.

Development and data-preparation utilities live under `src/b1like/dev`.
These modules may support data IO, covariance construction, and Cobaya dataset writing. The public likelihood and
theory classes should not depend on `dev`.

## Scripts

The top-level `scripts/` directory is for local run orchestration and debugging.
It may contain script templates and non-secret configuration templates, but it
should not contain generated data products or Cobaya outputs.

Examples of script responsibilities:

- generate Cobaya `.dataset` inputs from external bandpower products
- generate Cobaya run YAMLs
- inspect covariance matrices and bandpower windows
- translate local run configs into calls to `b1like.dev`

Run-specific configuration belongs in `scripts/configs/`. Machine-local config
files should use a `.local.yaml` suffix and should not be committed.

## Data and Output Locations

Generated data and sampler outputs should not be written inside this repository.
Use environment variables to point scripts to system-specific locations:

```bash
export B1LIKE_DATA_ROOT=/path/to/b1like/data
export B1LIKE_OUTPUT_ROOT=/path/to/b1like/outputs
```

`B1LIKE_DATA_ROOT` is the root for external or generated data products, such as
raw bandpower products, processed Cobaya data files, covariance matrices,
bandpower windows, noise spectra, and fiducial spectra.

`B1LIKE_OUTPUT_ROOT` is the root for Cobaya run products, such as generated run
YAMLs, minimizer outputs, chains, logs, and diagnostic plots.

This keeps the repository portable while allowing each system or user account to
choose its own storage layout.
