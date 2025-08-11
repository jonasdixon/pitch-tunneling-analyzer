import argparse
import datetime as dt
import os
from typing import List, Optional

import pandas as pd
from pybaseball import cache, statcast

cache.enable()

def daterange_chunks(start: str, end: str, days: int = 7):
    s = dt.datetime.fromisoformat(start).date()
    e = dt.datetime.fromisoformat(end).date()
    cur = s
    while cur <= e:
        nxt = min(cur + dt.timedelta(days=days - 1), e)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt + dt.timedelta(days=1)

def fetch_range(start: str, end: str) -> pd.DataFrame:
    frames = []
    for s, e in daterange_chunks(start, end, days=7):
        df = statcast(start_dt=s, end_dt=e)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Ensure consistent dtypes minimal set
    if "game_date" in out.columns:
        out["game_year"] = pd.to_datetime(out["game_date"]).dt.year
    return out

def optional_filter(df: pd.DataFrame, pitchers: Optional[List[int]] = None, team: Optional[str] = None):
    if pitchers:
        df = df[df["pitcher"].isin(pitchers)]
    if team:
        df = df[(df.get("home_team") == team) | (df.get("away_team") == team)]
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", required=True, help="CSV path")
    p.add_argument("--pitchers", nargs="*", type=int, help="Optional MLBAM pitcher IDs")
    p.add_argument("--team", type=str, help="Optional team abbr (e.g., NYY)")
    args = p.parse_args()

    df = fetch_range(args.start, args.end)
    df = optional_filter(df, args.pitchers, args.team)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
