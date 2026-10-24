"""Synthetic pitches built from known physics, so tests do not depend on downloaded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

REF_Y = 50.0


def make_pitch(
    x0=-1.5, z0=6.0, vx0=5.0, vy0=-130.0, vz0=-5.0, ax=-8.0, ay=25.0, az=-20.0, release_y=54.0
) -> dict:
    """A pitch defined at y = 50 ft; release position is the same curve evaluated at release_y."""
    disc = vy0**2 - 2 * ay * (REF_Y - release_y)
    t = (-vy0 - np.sqrt(disc)) / ay
    return {
        "release_pos_x": x0 + vx0 * t + 0.5 * ax * t**2,
        "release_pos_y": release_y,
        "release_pos_z": z0 + vz0 * t + 0.5 * az * t**2,
        "vx0": vx0, "vy0": vy0, "vz0": vz0, "ax": ax, "ay": ay, "az": az,
        "_x0": x0, "_z0": z0,
    }


def analytic_xz(p: dict, y: float) -> tuple[float, float]:
    t = (-p["vy0"] - np.sqrt(p["vy0"] ** 2 - 2 * p["ay"] * (REF_Y - y))) / p["ay"]
    x = p["_x0"] + p["vx0"] * t + 0.5 * p["ax"] * t**2
    z = p["_z0"] + p["vz0"] * t + 0.5 * p["az"] * t**2
    return x, z


def pitch_row(game_pk=1, ab=1, n=1, pitcher=10, desc="ball", ptype="FF", speed=95.0, **kin) -> dict:
    from tunneling.kinematics import PLATE_FRONT_Y

    p = make_pitch(**kin)
    px, pz = analytic_xz(p, PLATE_FRONT_Y)
    row = {k: v for k, v in p.items() if not k.startswith("_")}
    row.update(
        game_pk=game_pk, game_date="2023-04-01", at_bat_number=ab, pitch_number=n, pitcher=pitcher,
        batter=99, pitch_type=ptype, release_speed=speed, description=desc, zone=5,
        plate_x=px, plate_z=pz, sz_top=3.4, sz_bot=1.6, balls=0, strikes=0, stand="R", p_throws="R",
    )
    return row


@pytest.fixture
def frame():
    def build(rows):
        return pd.DataFrame(rows)

    return build
