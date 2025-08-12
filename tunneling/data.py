from __future__ import annotations
from importlib import resources
import pandas as pd


def load_sample() -> pd.DataFrame:
    """
    Load a tiny, packaged CSV sample so users can run examples without any API calls.
    Note: There is a reference to the top-level package ('tunneling') and then join 
    the subpath, so there's no conflict with the module name 'tunneling.data'.
    """
    with resources.files("tunneling").joinpath("data/sample_pitches.csv").open("rb") as f:
        return pd.read_csv(f, parse_dates=["game_date"])