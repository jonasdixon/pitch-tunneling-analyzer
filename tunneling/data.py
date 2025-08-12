from __future__ import annotations
from importlib import resources
import pandas as pd

# Optional: uncomment if you enabled pybaseball in pyproject
# from pybaseball import statcast


def load_sample() -> pd.DataFrame:
    """
    Load a tiny, packaged CSV sample so users can run examples without any API calls.
    """
    with resources.files("tunneling.data").joinpath("sample_pitches.csv").open("rb") as f:
        return pd.read_csv(f, parse_dates=["game_date"])


# Example fetch helper (requires pybaseball dependency and statcast credentials if needed)
# def fetch_statcast_sample(start="2023-04-01", end="2023-04-03") -> pd.DataFrame:
#     df = statcast(start_dt=start, end_dt=end)
#     # select and rename to expected columns if needed
#     cols = {
#         "game_date": "game_date",
#         "pitcher": "pitcher",
#         "batter": "batter",
#         "at_bat_number": "at_bat_number",
#         "pitch_number": "pitch_number",
#         "pitch_type": "pitch_type",
#         "release_pos_x": "release_pos_x",
#         "release_pos_y": "release_pos_y",
#         "release_pos_z": "release_pos_z",
#         "plate_x": "plate_x",
#         "plate_z": "plate_z",
#         "description": "description",
#     }
#     return df[list(cols.keys())].rename(columns=cols)