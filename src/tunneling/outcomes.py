"""Pitch outcome flags and their rates, with the denominators spelled out.

Definitions (Statcast `description` values):
  swing        swinging_strike, swinging_strike_blocked, foul, foul_tip, foul_bunt, missed_bunt,
               bunt_foul_tip, hit_into_play
  whiff        swinging_strike, swinging_strike_blocked, missed_bunt
  called_strike  called_strike
  csw          called_strike or whiff
Rates:
  Swing%  = swings / pitches        Whiff% = whiffs / swings
  Zone%   = in-zone / pitches       Chase% = swings at out-of-zone pitches / out-of-zone pitches
  CSW%    = csw / pitches

`foul_tip` is a swing but not a whiff here. Counting it as a whiff moves league Whiff% by well
under one point, and the choice is a one-line change in WHIFF.

Pitchouts and automatic balls/strikes are not decisions about the pitch's quality and are dropped
by `drop_non_competitive`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SWING = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "foul_bunt",
    "missed_bunt", "bunt_foul_tip", "hit_into_play",
})
WHIFF = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
CALLED_STRIKE = frozenset({"called_strike"})
NON_COMPETITIVE = frozenset({
    "automatic_ball", "automatic_strike", "pitchout", "foul_pitchout", "swinging_pitchout",
})

PLATE_HALF_WIDTH_FT = 0.83  # 17 in plate half-width plus the ball radius, as Statcast zones 1-9


def drop_non_competitive(df: pd.DataFrame, col: str = "description") -> pd.DataFrame:
    return df.loc[~df[col].isin(NON_COMPETITIVE)]


def in_zone(df: pd.DataFrame) -> pd.Series:
    """Statcast zone 1-9 where it is recorded, otherwise the geometric zone from sz_bot/sz_top."""
    geometric = (
        (df["plate_x"].abs() <= PLATE_HALF_WIDTH_FT)
        & (df["plate_z"] >= df["sz_bot"])
        & (df["plate_z"] <= df["sz_top"])
    )
    if "zone" not in df.columns:
        return geometric
    zone = df["zone"]
    return pd.Series(np.where(zone.notna(), zone.between(1, 9), geometric), index=df.index)


def add_outcome_flags(df: pd.DataFrame, col: str = "description") -> pd.DataFrame:
    out = df.copy()
    desc = out[col]
    out["swing"] = desc.isin(SWING)
    out["whiff"] = desc.isin(WHIFF)
    out["called_strike"] = desc.isin(CALLED_STRIKE)
    out["in_zone"] = in_zone(out).astype(bool)
    out["chase"] = out["swing"] & ~out["in_zone"]
    out["csw"] = out["called_strike"] | out["whiff"]
    return out


def summarize_outcomes(df: pd.DataFrame, by: list[str], min_n: int = 50) -> pd.DataFrame:
    """Rates per group from a frame that already has the flags from `add_outcome_flags`."""
    g = df.groupby(by, observed=True)
    out = g.agg(
        n=("swing", "size"),
        swings=("swing", "sum"),
        whiffs=("whiff", "sum"),
        out_of_zone=("in_zone", lambda s: int((~s).sum())),
        chases=("chase", "sum"),
        csw=("csw", "sum"),
    ).reset_index()
    out = out[out["n"] >= min_n].copy()
    out["swing_pct"] = out["swings"] / out["n"]
    out["whiff_pct"] = out["whiffs"] / out["swings"].where(out["swings"] > 0)
    out["chase_pct"] = out["chases"] / out["out_of_zone"].where(out["out_of_zone"] > 0)
    out["csw_pct"] = out["csw"] / out["n"]
    return out
