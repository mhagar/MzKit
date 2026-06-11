# MzKit

A toolkit for **inspecting LC/MS data**, built for natural-products chemists.

MzKit is *not* a metabolomics pipeline. It's a fast, fluid GUI for going into
your data and inspecting it at fine grain — extracting coeluting-ion groups
("ensembles"), finding molecular formulae for manual annotation, and zipping
around large datasets seamlessly. It can also staple bioassay ("fingerprint")
data alongside your LC/MS samples.

## Requirements

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) for environment + dependency management

## Setup

```bash
uv sync
```

> **Note:** MzKit currently depends on an unreleased build of `find-mfs` via a
> local editable path (see `[tool.uv.sources]` in `pyproject.toml`). Until those
> features ship to PyPI, you need a checkout of `find-mfs` at the path that entry
> points to. Once they ship, that section goes away and `uv sync` works from a
> clean clone.

## Running

```bash
uv run python main.py        # launch the GUI
uv run mzkit <subcommand>    # headless CLI (import-features, filter, export-table, ...)
uv run pytest                # run the test suite
```

Always use `uv run` rather than a bare `python`.

> Tests that need real MS data are **skipped** automatically when the data isn't
> present (the `.mzML`/`.mzk` fixtures are gitignored). The non-data tests
> (feature-extraction parity, formula finder) run from a clean clone.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a map of the system: the domain
model, the `core` (Qt-free) vs `gui` (PyQt5) split, and how background
processing flows through the `ProcessController`.
