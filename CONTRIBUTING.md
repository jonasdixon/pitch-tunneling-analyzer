# Contributing

- Use Conda: `conda env create -f environment.yml && conda activate tunneling`.
- Install hooks: `pre-commit install` and `nbstripout --install`.
- Branch naming: `feature/<short-name>` or `fix/<short-name>`.
- Lint locally: `black . && isort . && flake8 .`.
- Open a PR; ensure CI is green.

Project structure
- notebooks/: analysis and outputs
- src/: code (data, features, metrics, viz)
- data/: raw/ and processed/ (gitignored)
- reports/figures/: exported charts
- docs/: project plan and notes
