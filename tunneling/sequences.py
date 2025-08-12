from __future__ import annotations
import pandas as pd

REQ_COLS = [
    "game_date",
    "pitcher",
    "batter",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "plate_x",
    "plate_z",
    "description",  # for CSW flag later
]


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build consecutive-pitch pairs (previous -> current) for a pitcher within an at-bat.
    Returns a wide DataFrame with prev_* columns aligned to each current pitch row.
    """
    _ensure_columns(df)
    # Sort to guarantee order
    df = df.sort_values(["game_date", "pitcher", "at_bat_number", "pitch_number"]).copy()

    # Partition by pitcher and at-bat to avoid cross-PA pairing
    g = df.groupby(["game_date", "pitcher", "at_bat_number"], sort=False)

    # Shift previous pitch columns
    prev = g[[
        "pitch_type",
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "plate_x",
        "plate_z",
        "description",
    ]].shift(1)

    prev.columns = [f"prev_{c}" for c in prev.columns]
    out = pd.concat([df, prev], axis=1)

    # Only keep rows that actually have a previous pitch in the same at-bat
    out = out.loc[out["prev_pitch_type"].notna()].reset_index(drop=True)
    return out