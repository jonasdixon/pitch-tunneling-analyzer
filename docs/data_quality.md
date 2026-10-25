# Data quality

Counts are from `integrity.integrity_report` and `integrity.clean` on the cached 2023 and 2024
regular-season exports (`tunneling pairs` prints the same table).

## Where the rows went

| | 2023 | 2024 |
|---|---|---|
| Pitches in the export | 720,684 | 711,899 |
| Duplicate (game, at-bat, pitch) key | 0 | 0 |
| Missing trajectory or plate location | 3,012 | 2,672 |
| Release speed outside 50-106 mph | 349 | 294 |
| `plate_x` beyond 5 ft | 4 | 3 |
| `plate_z` outside -3 to 7 ft | 23 | 11 |
| Zone top at or below zone bottom | 0 | 0 |
| Rebuilt plate location more than 2 inches from Statcast's | 2 | 2 |
| **Kept** | **717,294** | **708,917** |
| Consecutive-pitch pairs | 532,822 | 526,242 |
| Pairs that also have a second lag (pitch two back) | 369,368 | 365,297 |

## Findings

**The trajectory rebuild matches Statcast's own plate location.** Median error 0.05 inches over 2023,
99th percentile 0.08 inches, which is the reason the tunnel features can be trusted as geometry. Two
pitches per season were more than 2 inches off (the largest 7 inches) and are dropped.

**Pairing on the date was wrong.** The original code grouped on `(game_date, pitcher, at_bat_number)`.
`at_bat_number` restarts every game, so on a doubleheader the same pitcher-and-number can belong to
two games. In 2024 there are 4 such keys, none in 2023. The effect on a fit is tiny, but a pitch
could be paired with one from the other game. Pairs are now formed on `game_pk`.

**Skipped pitches were being bridged.** Shifting within a plate appearance links a pitch to the
previous row even if tracking dropped a pitch in between. In 2023 that is 448 of 533,270 pairs
(0.08%). Pairs now require `pitch_number` exactly one lower. The second lag has the same rule.

**Plate appearances shared by two pitchers.** 62 of 183,962 plate appearances in 2023 had a pitching
change mid-appearance. Grouping includes the pitcher, so a pair never crosses the change.

**The first plausibility range for `plate_z` was too tight.** Balls in the dirt extrapolate below
-1 ft at the front of the plate. The range is -3 to 7 ft.

**Release speeds below 50 mph.** These are position players pitching and some eephus pitches. They
are real pitches but far outside the range of the rest, so the 350 or so are excluded from the models.

**`Chase%` was dividing by swings, not by out-of-zone pitches.** This is a code error in version 0.1
rather than a data problem. It would have reported chase as a share of swings, which is a different
statistic (what fraction of swings are at balls).

## Coverage

All 2023 and 2024 regular-season games are included. 859 pitchers appear in the 2023 pairs and 850 in 2024. League
rates in the scored pairs, 2023: swing 53.2%, whiff per swing 23.4%, chase 32.6%, CSW 24.0%.

## What is not checked

- Pitch type labels. A new pitch mislabelled for part of a season would change the pair-type controls.
- The commit distance. 23.8 feet is a convention, not a measured quantity for every hitter; see the
  sensitivity table.
- Hitters' swing decisions are not observed before the ball reaches them, so "the point where a
  hitter commits" is a modelling assumption, not a data field.
