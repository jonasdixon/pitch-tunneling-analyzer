# Pitch Tunneling Analyzer

Quantify how well pitchers tunnel back-to-back pitch pairs (release similarity → late movement/location separation) and how tunneling correlates with Swing%, Whiff%, Chase%, and CSW.

Quick start
1) Create Conda environment:
   conda env create -f environment.yml
   conda activate tunneling

2) Fetch a sample (April 2023):
   python -m src.data.fetch_statcast --start 2023-04-01 --end 2023-04-30 --out data/raw/statcast_2023_04.csv

3) Build prior→current pitch pairs:
   python -m src.data.pair_builder --infile data/raw/statcast_2023_04.csv --out data/processed/pitch_pairs_2023_04.csv

4) Score Tunnel features and outcomes (in a Python session or notebook):
   import pandas as pd
   from src.features.tunneling import add_tunnel_features
   from src.metrics.outcomes import add_outcome_flags, summarize_outcomes

   pairs = pd.read_csv("data/processed/pitch_pairs_2023_04.csv", low_memory=False)
   pairs = add_tunnel_features(pairs)
   pairs = add_outcome_flags(pairs)
   pairs.to_csv("data/processed/pitch_pairs_scored_2023_04.csv", index=False)

   # Example summary by pitcher-season
   summary = summarize_outcomes(pairs, by=["pitcher", "season"], min_n=100)
   print(summary.head())

5) Plot quintile effects (inside a notebook/Python):
   from src.viz.pair_maps import tunnel_quintile_effect
   tunnel_quintile_effect(pairs, value_col="csw")

Data
- Statcast via `pybaseball` (pitch-level).

Metrics
- RP (Release Proximity): smaller is better (release similarity).
- SD (Shape Divergence): larger is better (movement/location separation).
- TunnelScore: z(SD) − z(RP), computed within pitcher-season.
- Outcomes (second pitch): Swing, Whiff, Chase, CSW.

License
- MIT
