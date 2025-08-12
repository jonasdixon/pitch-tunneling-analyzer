from __future__ import annotations
import io
import pkgutil
from pathlib import Path

import pandas as pd


def load_sample() -> pd.DataFrame:
    """
    Load a tiny, packaged CSV sample so users can run examples without any API calls.

    Uses pkgutil.get_data for maximum compatibility with editable installs.
    Falls back to a filesystem path if the package isn't installed.
    """
    data = pkgutil.get_data("tunneling", "data/sample_pitches.csv")
    if data is None:
        # Fallback when running from source without installation
        p = Path(__file__).parent / "data" / "sample_pitches.csv"
        data = p.read_bytes()
    return pd.read_csv(io.BytesIO(data), parse_dates=["game_date"])