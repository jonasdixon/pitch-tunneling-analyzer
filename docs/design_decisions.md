# Design decisions

## Rebuild trajectories instead of using release point and plate location

**Chosen.** Rebuild each pitch's path from release position, velocity and acceleration, and measure
separation at the commit point and at the plate.

**Why.** Tunneling is a claim about where the two pitches are *before* the hitter commits. Release-point
similarity and plate separation, which version 0.1 used, do not describe the middle of the flight. Two
pitches can leave the same arm slot and still be 8 inches apart at 24 feet.

**Rejected.** A single combined "TunnelScore". The weights (0.4/0.6 or an `alpha` and `beta`) were not
estimated from anything, and a score hides which component drives an association. The components are
reported on their own, in inches.

**Weakness.** The constant-acceleration fit is Statcast's, and trajectory error in the middle of the flight
is not observed. The only check is the plate location, where the median error is 0.05 inches.

## Commit distance of 23.8 feet

A convention (about 0.17 seconds before arrival). The model is re-fit at 20, 23.8, 27 and 30 feet
(`reports/commit_distance_sensitivity.csv`) so that the conclusion does not hinge on it. A time-based commit point
(a fixed number of milliseconds before the plate) would give different distances for fast and slow pitches
and is the obvious next variant.

## Controlled model, not quintiles

**Chosen.** A penalised spline logistic regression per outcome, with the tunnel features as smooth terms.

**Why.** A raw comparison of outcomes across quintiles of a tunnel score conflates tunneling with pitch
type, location and count. A fastball and a changeup differ in plate separation and in whiff rate for reasons
that have nothing to do with how they were sequenced. Quintiles also throw away information.

**What is held fixed.** Current and previous pitch type, speed, plate location (a tensor spline), distance to
the zone edge, where the current pitch was at the commit point, the previous pitch's distance to the edge
and its result, count, same-handed matchup, and a ridge-shrunk pitcher effect.

**Rejected.** A mixed-effects model with pitcher random slopes. The data are large and the pitcher
effect is in the model; random slopes for tunnel features would be the way to ask whether tunneling
matters more for some pitchers.

**Weakness.** The controls are flexible but not complete: no hitter effects, no catcher or park, no
information on the pitcher's intended target. Selection into sequences remains (pitchers choose their
sequences).

## Contrasts and intervals

Effects are the change in the average predicted rate when a feature goes from its 10th to its 90th
percentile for every pitch in the sample and everything else is left as observed. This is a population-
average contrast and avoids choosing a "typical" pitch. Intervals come from 30 pitcher-cluster bootstrap refits
(the pitcher is the independent unit, since the same pitcher's pitches share a repertoire and a style).
Thirty is few for the tails of a percentile interval; the intervals above are indicative.

## The lag-2 placebo

**Why.** An association between a feature and an outcome may be caused by the adjacent-pitch relationship
(tunneling) or by something about where earlier pitches were, which would also show up with a pitch
further back. Re-computing the same features against the pitch two earlier, on the same rows, with the
same controls (including the previous pitch's location and result), asks whether the effect is
specific to the previous pitch.

**Result.** For chases and for velocity drop the lag-2 effect is as large as the lag-1 effect or
larger. The exception is plate separation on whiffs and CSW, where lag 1 is 1 to 2 points *lower*
and lag 2 is near zero: a pitch that finishes farther from the previous one goes with slightly fewer
whiffs, the opposite of what tunneling predicts. That is the only adjacent-pitch-specific signal, and
it points the wrong way. (README table.)

**Weakness.** Features at lags 1 and 2 are correlated (0.23 for commit separation), so a lag-2
effect can partly be a lag-1 effect in disguise. It would be stronger to put both in one model; the
placebo answers a narrower question.

## Zone and outcomes

Whiff, chase and CSW use the definitions in `docs/definitions.md`. `foul_tip` is counted as a swing, not a
whiff. Zone is Statcast's zone 1-9 where recorded. Chase uses out-of-zone pitches as the denominator;
version 0.1 used swings, which answers a different question.

## Things that would change the conclusion

- A time-based commit point, or one that depends on pitch speed, giving a lag-1 effect clearly larger than
  lag-2.
- A hitter-level model (some hitters may be much more susceptible to tunneling).
- Video-based or ball-flight-based confirmation of how late hitters actually commit.
