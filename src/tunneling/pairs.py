"""Consecutive-pitch pairs and their tunnel features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tunneling.kinematics import (
    DEFAULT_COMMIT_FT,
    KINEMATIC_COLS,
    PLATE_FRONT_Y,
    pair_separation,
    position_at_y,
    tunnel_ratio,
)
from tunneling.outcomes import add_outcome_flags, drop_non_competitive, edge_distance

REQUIRED = [
    "game_pk", "at_bat_number", "pitch_number", "pitcher", "pitch_type", "release_speed",
    "description", "plate_x", "plate_z", "sz_top", "sz_bot", *KINEMATIC_COLS,
]
LAG_COLS = ["pitch_type", "release_speed", "plate_x", "plate_z", "description", *KINEMATIC_COLS]
PA_KEY = ["game_pk", "at_bat_number", "pitcher"]
PREV_RESULT = {
    "called_strike": "called_strike", "ball": "ball", "blocked_ball": "ball",
    "swinging_strike": "whiff", "swinging_strike_blocked": "whiff", "missed_bunt": "whiff",
    "foul": "foul", "foul_tip": "foul", "foul_bunt": "foul", "bunt_foul_tip": "foul",
}


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the previous pitch (prev_*) and the one before it (prev2_*) to each pitch.

    A pitch is paired only with pitches thrown by the same pitcher in the same plate appearance.
    The pairing is by `game_pk`, because (date, pitcher, at_bat_number) is not unique on
    doubleheaders, and a lag counts only if `pitch_number` is exactly `lag` lower, so a pitch the
    tracking system dropped never silently links two pitches that were not consecutive.
    Rows without a previous pitch are dropped; prev2_* is NaN where no valid second lag exists.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df.sort_values([*PA_KEY, "pitch_number"]).reset_index(drop=True)
    g = df.groupby(PA_KEY, sort=False)
    out = df
    for lag, prefix in ((1, "prev_"), (2, "prev2_")):
        shifted = g[LAG_COLS + ["pitch_number"]].shift(lag)
        valid = (df["pitch_number"] - shifted["pitch_number"]) == lag
        shifted = shifted.drop(columns="pitch_number").where(valid)
        shifted.columns = [prefix + c for c in shifted.columns]
        out = pd.concat([out, shifted], axis=1)
    return out.loc[out["prev_pitch_type"].notna()].reset_index(drop=True)


def _as_pitch(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df[[prefix + c for c in KINEMATIC_COLS]].rename(columns=lambda c: c[len(prefix):])


def score_pairs(pairs: pd.DataFrame, commit_ft: float = DEFAULT_COMMIT_FT) -> pd.DataFrame:
    """Add outcome flags and the tunnel features for lag 1 (and lag 2 where available).

    Features (inches unless noted): commit_sep_in, plate_sep_in, tunnel_ratio, velo_diff_mph
    (previous minus current, so positive means the second pitch is slower), same_type.
    """
    out = add_outcome_flags(drop_non_competitive(pairs)).reset_index(drop=True)
    cur = out[KINEMATIC_COLS]
    sep = pair_separation(_as_pitch(out, "prev_"), cur, commit_ft)
    out["commit_sep_in"] = sep["commit_sep_in"].to_numpy()
    out["plate_sep_in"] = sep["plate_sep_in"].to_numpy()
    out["tunnel_ratio"] = tunnel_ratio(out["commit_sep_in"], out["plate_sep_in"])
    out["velo_diff_mph"] = out["prev_release_speed"] - out["release_speed"]
    out["same_type"] = out["prev_pitch_type"] == out["pitch_type"]
    out["edge_dist"] = edge_distance(out)
    out["prev_edge_dist"] = edge_distance(
        out.assign(plate_x=out["prev_plate_x"], plate_z=out["prev_plate_z"])
    )
    out["prev_result"] = out["prev_description"].map(PREV_RESULT).fillna("other")
    cx, cz = position_at_y(cur, PLATE_FRONT_Y + commit_ft)
    out["cur_commit_x_ft"], out["cur_commit_z_ft"] = cx, cz

    has2 = out["prev2_vx0"].notna()
    out["l2_commit_sep_in"] = np.nan
    out["l2_plate_sep_in"] = np.nan
    if has2.any():
        sep2 = pair_separation(_as_pitch(out.loc[has2], "prev2_"), cur.loc[has2], commit_ft)
        out.loc[has2, "l2_commit_sep_in"] = sep2["commit_sep_in"].to_numpy()
        out.loc[has2, "l2_plate_sep_in"] = sep2["plate_sep_in"].to_numpy()
    return out
