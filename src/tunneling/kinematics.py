"""Rebuild pitch trajectories from Statcast's constant-acceleration fit.

Statcast stores, per pitch, a velocity (vx0, vy0, vz0) and an acceleration (ax, ay, az) that are
referenced to the plane y = 50 ft, plus the position at release. The release point is on the same
fitted curve, so the 50 ft position can be recovered by stepping back from release. The
reconstruction is checked against the reported plate location in `reconstruction_error`.

Coordinates are feet. y runs from the plate (the front edge is at 17/12 ft) out to the mound;
x is the catcher's-view horizontal axis and z is height.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLATE_FRONT_Y = 17.0 / 12.0
REF_Y = 50.0
DEFAULT_COMMIT_FT = 23.8  # distance from the front of the plate where a hitter is usually committed

KINEMATIC_COLS = [
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
]


def time_at_y(y, vy0, ay) -> np.ndarray:
    """Seconds after the y = 50 ft reference at which the ball reaches plane `y` (negative if earlier)."""
    vy0, ay = np.asarray(vy0, dtype=float), np.asarray(ay, dtype=float)
    disc = vy0**2 - 2.0 * ay * (REF_Y - np.asarray(y, dtype=float))
    with np.errstate(invalid="ignore"):
        return (-vy0 - np.sqrt(disc)) / ay


def reference_position(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """x and z at y = 50 ft, recovered from the release point and the fitted velocity/acceleration."""
    t_rel = time_at_y(df["release_pos_y"].to_numpy(float), df["vy0"], df["ay"])
    x0 = df["release_pos_x"].to_numpy(float) - df["vx0"].to_numpy(float) * t_rel \
        - 0.5 * df["ax"].to_numpy(float) * t_rel**2
    z0 = df["release_pos_z"].to_numpy(float) - df["vz0"].to_numpy(float) * t_rel \
        - 0.5 * df["az"].to_numpy(float) * t_rel**2
    return x0, z0


def position_at_y(df: pd.DataFrame, y: float) -> tuple[np.ndarray, np.ndarray]:
    """(x, z) in feet where each pitch crosses plane `y`."""
    x0, z0 = reference_position(df)
    t = time_at_y(y, df["vy0"], df["ay"])
    x = x0 + df["vx0"].to_numpy(float) * t + 0.5 * df["ax"].to_numpy(float) * t**2
    z = z0 + df["vz0"].to_numpy(float) * t + 0.5 * df["az"].to_numpy(float) * t**2
    return x, z


def reconstruction_error(df: pd.DataFrame) -> pd.Series:
    """Distance in inches between the reconstructed and reported plate location."""
    x, z = position_at_y(df, PLATE_FRONT_Y)
    return pd.Series(
        12.0 * np.hypot(x - df["plate_x"].to_numpy(float), z - df["plate_z"].to_numpy(float)),
        index=df.index,
        name="recon_err_in",
    )


def pair_separation(
    a: pd.DataFrame, b: pd.DataFrame, commit_ft: float = DEFAULT_COMMIT_FT
) -> pd.DataFrame:
    """Separation in inches between two aligned sets of pitches at the commit point and at the plate.

    Row i of `a` is paired with row i of `b`; the two frames need only share length.
    """
    y_commit = PLATE_FRONT_Y + commit_ft
    ax_, az_ = position_at_y(a, y_commit)
    bx_, bz_ = position_at_y(b, y_commit)
    pax, paz = position_at_y(a, PLATE_FRONT_Y)
    pbx, pbz = position_at_y(b, PLATE_FRONT_Y)
    return pd.DataFrame(
        {
            "commit_sep_in": 12.0 * np.hypot(ax_ - bx_, az_ - bz_),
            "plate_sep_in": 12.0 * np.hypot(pax - pbx, paz - pbz),
        },
        index=a.index,
    )


def tunnel_ratio(commit_sep_in, plate_sep_in, floor_in: float = 0.5) -> np.ndarray:
    """Plate separation per inch of commit-point separation. The floor stops near-identical pairs
    (commit separation near zero) from producing arbitrarily large ratios."""
    return np.asarray(plate_sep_in, float) / np.maximum(np.asarray(commit_sep_in, float), floor_in)
