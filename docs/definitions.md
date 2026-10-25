# Definitions

## Pitch pair

A pitch and the pitch the same pitcher threw immediately before it in the same plate appearance
(`pitch_number` exactly one lower, same `game_pk`, same `at_bat_number`, same pitcher). Pairs are
never formed across plate appearances, games or pitchers. A pitch whose predecessor was not tracked
has no pair. Pitchouts and automatic balls and strikes are not scored.

## Trajectory

Statcast fits each pitch with constant acceleration (`ax`, `ay`, `az`) from a velocity (`vx0`,
`vy0`, `vz0`) referenced to the plane 50 feet from the back of the plate, and reports the release
position. The release point lies on the same fitted curve, so the position at 50 feet is recovered by
stepping back from release, and the position anywhere else follows by solving the quadratic in
distance. The rebuilt location at the front of the plate matches Statcast's `plate_x`/`plate_z`
(median error 0.05 inches over 2023).

## Commit point

The distance from the front of the plate where a hitter must decide whether to swing. 23.8 feet is
used by default (about 0.17 seconds before the ball arrives, a figure commonly quoted for decision
time). It is a parameter, and the sensitivity to it is reported rather than assumed.

## Tunnel features, in inches

- **Commit separation.** Distance between the two pitches' positions at the commit point. A good
  tunnel has this small.
- **Plate separation.** Distance between the two pitches' positions at the front of the plate. A good
  tunnel has this large.
- **Tunnel ratio.** Plate separation divided by commit separation (with commit separation floored at
  half an inch so near-identical pairs do not give huge ratios). Reported for description; the
  models use the two separations themselves, because a ratio hides which one matters.
- **Velocity drop.** Previous pitch's release speed minus the current pitch's.

There is no combined "TunnelScore" in version 0.2. The earlier score (a difference of z-scores of
release similarity and plate separation) depended on arbitrary weights and was computed with
different scalings for the two pitches in a pair.

## Outcomes

Swing, whiff, called strike, chase and CSW as in `tunneling/outcomes.py`: whiff rate is whiffs per
swing; chase rate is swings at out-of-zone pitches per out-of-zone pitch; CSW is called strikes plus
whiffs per pitch. `foul_tip` is a swing and not a whiff. The zone is Statcast's zone 1-9 where
recorded, otherwise the geometric zone.

## Effect

The change in an outcome rate when one tunnel feature moves from its 10th to its 90th percentile for
every pitch in the sample, with everything else left as observed (g-computation), from a model that
also holds fixed the pitch types, where the pitch ended up and how close it was to the zone, where the
current and previous pitches were, the previous pitch's result, the count, the matchup and the pitcher.
Intervals resample pitchers.
