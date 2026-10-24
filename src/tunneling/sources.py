"""Statcast loading: a resumable per-day download cache, local CSV/parquet reads, packaged sample."""

from __future__ import annotations

import datetime as dt
import io
import time
from importlib import resources
from pathlib import Path

import pandas as pd
import requests

try:  # use the OS trust store; some networks re-sign TLS with a private root CA
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

KEEP_COLUMNS = [
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "pitcher", "player_name", "batter", "stand", "p_throws", "balls", "strikes",
    "pitch_type", "release_speed", "description", "zone",
    "plate_x", "plate_z", "sz_top", "sz_bot", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
]


def _get(url: str, params: dict, retries: int = 5, pause: float = 2.0) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=90)
            if resp.status_code == 200:
                return resp
            last = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch_day(day: dt.date) -> pd.DataFrame:
    """One day of regular-season Statcast. Savant caps exports at 25,000 rows; a day is ~5,000."""
    params = {
        "all": "true", "type": "details", "hfGT": "R|", "player_type": "pitcher",
        "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat(),
        "min_pitches": "0", "min_results": "0",
    }
    text = _get(SAVANT_URL, params).text
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if len(df) >= 25_000:
        raise RuntimeError(f"{day}: hit the 25,000-row export cap")
    return df[[c for c in KEEP_COLUMNS if c in df.columns]]


def fetch_range(start: str, end: str, cache_dir: str | Path) -> None:
    """Download each missing day in [start, end] into `cache_dir/daily/`."""
    daily = Path(cache_dir) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    day, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while day <= last:
        path = daily / f"{day.isoformat()}.parquet"
        if not path.exists():
            df = fetch_day(day)
            df.to_parquet(path, index=False)
            print(f"{day} {len(df):>6} pitches", flush=True)
            time.sleep(0.5)
        day += dt.timedelta(days=1)


def load_cached(cache_dir: str | Path, start: str, end: str) -> pd.DataFrame:
    """Concatenate cached daily files with start <= date <= end."""
    daily = Path(cache_dir) / "daily"
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    files = sorted(p for p in daily.glob("*.parquet") if lo <= dt.date.fromisoformat(p.stem) <= hi)
    frames = [f for f in (pd.read_parquet(p) for p in files) if len(f)]
    if not frames:
        raise FileNotFoundError(f"no cached days in {daily} for {start}..{end}")
    df = pd.concat(frames, ignore_index=True)
    return df[[c for c in KEEP_COLUMNS if c in df.columns]]


def load_file(path: str | Path) -> pd.DataFrame:
    """Read a Statcast CSV or parquet export and keep the columns this package uses."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    return df[[c for c in KEEP_COLUMNS if c in df.columns]]


def load_sample() -> pd.DataFrame:
    """A few thousand real pitches (several pitchers, 2023) packaged for offline use."""
    with resources.files("tunneling").joinpath("data/sample_pitches.csv").open("rb") as fh:
        return pd.read_csv(fh)
