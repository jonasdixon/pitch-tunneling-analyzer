# Requirements

## Technical

- T1. Statcast pitch data, downloaded one game day at a time (Savant caps an export at 25,000 rows)
  and cached; an interrupted download resumes. Raw data is not committed.
- T2. Trajectories are rebuilt from the fitted kinematics, and the rebuild is checked against the
  reported plate location on every run (`reconstruction_error`).
- T3. Pairs are formed on `game_pk`, never on date, and only from consecutive `pitch_number`.
- T4. Every row excluded by an integrity rule is counted by the rule that caught it.
- T5. The commit distance is a parameter, and a sweep from 20 to 30 feet is reported.
- T6. The package is a single importable module, `tunneling`, with a command-line entry point.

## User

- U1. An analyst can score the pairs for any pitcher-season and see commit separation, plate
  separation and velocity drop for each, in inches and miles per hour.
- U2. An analyst can ask whether those features are associated with whiffs, chases and called or
  swinging strikes *after* holding pitch type, location, count and pitcher fixed.
- U3. The analyst is told how much of an association belongs to adjacent-pitch sequencing and
  how much does not, using a placebo with the pitch two earlier.

## Functional

- F1. `tunneling fetch | pairs | eda | effects | holdout` cover download, pairing and scoring,
  feature checks, the controlled models, and held-out evaluation.
- F2. `integrity.clean` returns the cleaned frame and the counts per rule.
- F3. `pairs.build_pairs` and `pairs.score_pairs` are separate, so scoring can be redone with
  another commit distance without re-pairing.
- F4. `effects.fit_effects` returns point curves, bootstrap curves and contrasts for any of the
  three outcomes.
