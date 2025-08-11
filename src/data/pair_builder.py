import argparse
import os
import pandas as pd

REQUIRED_COLS = [
    "game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "game_year",
    "pitch_type", "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "stand", "p_throws", "sz_top", "sz_bot",
    "description", "events", "zone"
]

def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows with necessary fields
    have = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[have].copy()

    # Sort within PA
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    # Lag previous pitch within the same PA by the same pitcher
    grp = df.groupby(["game_pk", "at_bat_number", "pitcher"], group_keys=False)
    prev = grp.shift(1).add_prefix("prev_")
    pairs = pd.concat([df, prev], axis=1)

    # Ensure previous pitch exists
    pairs = pairs.dropna(subset=["prev_pitch_type", "prev_pitch_number"])

    # Keep only adjacent pitches within the PA
    pairs = pairs[(pairs["pitch_number"] - pairs["prev_pitch_number"]) == 1]

    # Add season column if missing
    if "season" not in pairs.columns:
        if "game_year" in pairs.columns:
            pairs["season"] = pairs["game_year"]
        else:
            pairs["season"] = pd.NaT

    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.infile, low_memory=False)
    pairs = build_pairs(df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pairs.to_csv(args.out, index=False)
    print(f"Built {len(pairs):,} pitch pairs -> {args.out}")

if __name__ == "__main__":
    main()
