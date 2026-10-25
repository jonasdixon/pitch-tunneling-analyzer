"""Write the small offline sample shipped with the package (a few whole games from the cache).

Usage: python scripts/make_sample.py --cache /path/to/cache
"""

import argparse

import numpy as np

from tunneling import sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--games", type=int, default=4)
    a = ap.parse_args()
    df = sources.load_cached(a.cache, "2023-06-12", "2023-06-18")
    rng = np.random.default_rng(0)
    games = rng.choice(df["game_pk"].unique(), size=a.games, replace=False)
    sample = df.loc[df["game_pk"].isin(games)].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    keep = [c for c in sources.KEEP_COLUMNS if c not in {"game_year", "pfx_x", "pfx_z"}]
    sample = sample[keep].round(4)
    sample.to_csv("src/tunneling/data/sample_pitches.csv", index=False)
    print(len(sample), "pitches from games", sorted(int(g) for g in games), sample["player_name"].nunique(), "pitchers")


if __name__ == "__main__":
    main()
