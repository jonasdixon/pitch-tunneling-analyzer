# Changelog

## [Unreleased]

### Fixed
- README chase finding had the sign backwards. The corrected direction matches the tunneling story:
  pitches that looked like the previous pitch at the commit point, but finished far from it, are chased
  more. (The placebo test, not the sign, is what rules out reading this as tunneling.)
- README overstated the plate-location replication check as proof the rebuilt commit-point positions
  are correct; reworded to describe it as a check on the plate end of the rebuild only.

### Added
- README states the model: penalised-spline logistic regression with a ridge-shrunk pitcher effect.
- README model paragraph now names previous pitch type, not just previous pitch result, among the
  held-fixed covariates (the model already included it; the README did not say so).

## [0.2.0]

Rewrite. The measurements and the analysis changed, so old scores are not comparable.

### Changed
- One package, `tunneling`, under `src/`. The overlapping `src/` and top-level `tunneling/` code is gone
  (`tunneling/init.py` was never a package init).
- Tunnel features are rebuilt from trajectories: separation at a commit point (default 23.8 ft) and at the
  plate, in inches. The z-scored "TunnelScore" and release-point distance are removed.
- Pairs use `game_pk` and require consecutive `pitch_number` (448 skipped-pitch pairs in 2023; 4
  doubleheader keys in 2024).
- Chase% is per out-of-zone pitch (it was per swing). Bunts are swings; whiff, swing and chase use explicit
  description sets. Pitchouts and automatic balls and strikes are not scored.
- Outcomes are modelled with a penalised spline logistic regression with controls and a pitcher-cluster
  bootstrap, plus a lag-2 placebo and a season-ahead held-out check. Binned quintiles are descriptive only.
- CI runs ruff and pytest on Python 3.10, 3.11 and 3.12. Added pre-commit.
- Notebooks replaced by one that runs offline on a packaged 1,300-pitch sample.

### Removed from version control
- `data/raw/statcast_2023_04.csv` and `data/processed/pitch_pairs_2023_04.csv` (about 90 MB), compiled
  `.pyc` files and `.DS_Store`. They remain in git history.

## [0.1.0] - 2025-08-11
- Initial scaffold: environment file, fetch, pairs, tunneling features, outcomes, basic plots.
