# Pitch Tunneling Analyzer — Project Plan

Objectives
- Build reproducible pipeline from Statcast → pitch pairs → TunnelScore → outcome lift.
- Produce league- and pitcher-level insights with visuals and a short write-up.

Milestones
- v0.1 Data & Pairs: Fetch 2023 month sample; build prior→current pairs; save processed CSV.
- v0.2 Tunnel Features: Implement RP/SD/TunnelScore; sanity checks and basic plots.
- v0.3 Outcomes & Effects: Add outcome flags; quantify Whiff/Chase/CSW lift by TunnelScore quintile and pitch-pair type.
- v1.0 Profiles & Report: Leaderboards, figures, README/blog-style summary.

Tasks (Issues)
- Set up environment and pre-commit (type:infra)
- Implement fetch_statcast with chunking (type:feature)
- Implement pair_builder (type:feature)
- Implement tunneling features (type:feature)
- Implement outcomes flags and summaries (type:feature)
- Visuals: quintile effect plot and sample scatters (type:feature)
- README + example outputs (type:docs)
- Sensitivity: weights (alpha, beta), per-handedness splits (type:research)

Labels
- type:{feature,bug,docs,infra,research}
- status:{todo,doing,blocked,review}
- priority:{high,normal,low}
