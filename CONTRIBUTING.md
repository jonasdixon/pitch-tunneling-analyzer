# Contributing

- Install: `pip install -e '.[dev]'`, then `pre-commit install`.
- Lint: `ruff check src tests scripts`. Test: `pytest`.
- Branches: `feature/<short-name>` or `fix/<short-name>`; open a pull request and keep CI green.
- Raw Statcast data is never committed. `data/` is git-ignored; the package ships a small sample and
  `scripts/make_sample.py` shows how it was made.
- A change to a definition (zone, whiff, chase, commit point) needs a test that fails on the old
  behaviour and an entry in `docs/definitions.md`.

Layout: `src/tunneling/` (package), `tests/`, `docs/`, `notebooks/`, `reports/` (tables and figures
from the runs described in the README), `scripts/`.
