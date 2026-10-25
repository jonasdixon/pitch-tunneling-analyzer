"""Data-integrity checks. Each rule reports how many rows it flags; `clean` drops them in order."""

from __future__ import annotations

import pandas as pd

from tunneling.kinematics import KINEMATIC_COLS, reconstruction_error

KEY = ["game_pk", "at_bat_number", "pitch_number"]
SPEED_RANGE_MPH = (50.0, 106.0)  # a 125 mph reading is a tracking error, not a pitch
PLATE_X_MAX_FT = 5.0
PLATE_Z_RANGE_FT = (-3.0, 7.0)  # balls in the dirt extrapolate below 0; they are real pitches
RECON_TOLERANCE_IN = 2.0


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks (True = bad row), one per rule."""
    lo, hi = SPEED_RANGE_MPH
    masks = {
        "duplicate_key": df.duplicated(KEY, keep="first"),
        "missing_kinematics": df[KINEMATIC_COLS + ["plate_x", "plate_z"]].isna().any(axis=1),
        "impossible_speed": ~df["release_speed"].between(lo, hi) & df["release_speed"].notna(),
        "impossible_plate_x": df["plate_x"].abs() > PLATE_X_MAX_FT,
        "impossible_plate_z": ~df["plate_z"].between(*PLATE_Z_RANGE_FT) & df["plate_z"].notna(),
        "bad_zone_bounds": df["sz_top"] <= df["sz_bot"],
    }
    complete = ~masks["missing_kinematics"]
    err = pd.Series(0.0, index=df.index)
    err.loc[complete] = reconstruction_error(df.loc[complete])
    masks["trajectory_mismatch"] = err > RECON_TOLERANCE_IN
    return masks


def integrity_report(df: pd.DataFrame) -> dict:
    """Counts per rule plus structural facts that motivate how pairs are built."""
    report = {"rows": len(df)}
    report.update({k: int(m.sum()) for k, m in rule_masks(df).items()})
    if {"game_date", "pitcher", "at_bat_number", "game_pk"} <= set(df.columns):
        by_date = df.groupby(["game_date", "pitcher", "at_bat_number"])["game_pk"].nunique()
        report["date_pitcher_ab_keys_spanning_two_games"] = int((by_date > 1).sum())
    return report


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop flagged rows (a row is counted under the first rule that catches it)."""
    masks = rule_masks(df)
    keep = pd.Series(True, index=df.index)
    dropped = {}
    for name, m in masks.items():
        hit = m & keep
        dropped[name] = int(hit.sum())
        keep &= ~m
    return df.loc[keep].copy(), {"rows_in": len(df), "rows_out": int(keep.sum()), **dropped}
