# MzKit

A toolkit for inspecting LC/MS data, built for natural-products chemists.

This is not intended to be a metaboomics pipeline; rather, it's designed
for fast, fluid data inspection at fine grain.

Key ideas:

- Extracting coeluting features ("ensembles")
- Finding molecular formulae
- Manual spectrum annotation
- Zipping around large datasets seamlessly
- Viewing bioassay data (i.e. "fingerprints") data alongside your LC/MS runs.

## Requirements

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) for environment + dependency management

## Setup

Pull repo into a directory and then:
```bash
uv sync
```
## Running
```bash
uv run python main.py        # launch the GUI
uv run mzkit <subcommand>    # headless CLI (import-features, filter, export-table, ...)
uv run pytest                # run the test suite - requires .mzML files
```

Always use `uv run`.

> Tests that need real MS data are **skipped** automatically when the data isn't
> present (the `.mzML`/`.mzk` fixtures are gitignored). 
