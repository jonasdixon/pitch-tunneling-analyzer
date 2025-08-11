import numpy as np
import pandas as pd

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
}
CALLED_STRIKE = {"called_strike"}

def is_swing(desc: str) -> bool:
    return desc in SWING_DESCRIPTIONS

def is_whiff(desc: str) -> bool:
    return desc in WHIFF_DESCRIPTIONS

def is_called_strike(desc: str) -> bool:
    return desc in CALLED_STRIKE

def in_zone(row: pd.Series) -> bool:
    # Prefer Statcast 'zone' if present (1-9 in-zone), else geometric fallback
    if "zone" in row and not pd.isna(row["zone"]):
        try:
            return 1 <= int(row["zone"]) <= 9
        except Exception:
            pass
    # Fallback: horizontal within ~0.83 ft, vertical within sz_bot/sz_top
    x = row["plate_x"]
    z = row["plate_z"]
    return (abs(x) <= 0.83) and (row["sz_bot"] <= z <= row["sz_top"])

def add_outcome_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["swing"] = df["description"].map(is_swing).fillna(False)
    df["whiff"] = df["description"].map(is_whiff).fillna(False)
    df["called_strike"] = df["description"].map(is_called_strike).fillna(False)
    df["in_zone"] = df.apply(in_zone, axis=1)
    df["chase"] = df["swing"] & (~df["in_zone"])
    df["csw"] = df["called_strike"] | df["whiff"]
    return df

def summarize_outcomes(df: pd.DataFrame, by=["pitcher", "season"], min_n: int = 50) -> pd.DataFrame:
    g = df.groupby(by)
    out = g.agg(
        n=("pitch_number", "size"),
        swing=("swing", "sum"),
        whiff=("whiff", "sum"),
        chase=("chase", "sum"),
        csw=("csw", "sum"),
    ).reset_index()
    out = out[out["n"] >= min_n]
    out["Swing%"] = out["swing"] / out["n"].clip(lower=1)
    out["Whiff%"] = out["whiff"] / out["swing"].clip(lower=1)
    out["Chase%"] = out["chase"] / out["swing"].clip(lower=1)
    out["CSW%"] = out["csw"] / out["n"].clip(lower=1)
    return out
