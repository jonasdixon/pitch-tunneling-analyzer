import numpy as np
import pandas as pd

from conftest import analytic_xz, make_pitch, pitch_row
from tunneling.kinematics import (
    PLATE_FRONT_Y,
    pair_separation,
    position_at_y,
    reconstruction_error,
    time_at_y,
    tunnel_ratio,
)


def test_plate_position_matches_analytic_trajectory():
    p = make_pitch()
    df = pd.DataFrame([{k: v for k, v in p.items() if not k.startswith("_")}])
    x, z = position_at_y(df, PLATE_FRONT_Y)
    ex, ez = analytic_xz(p, PLATE_FRONT_Y)
    assert np.isclose(x[0], ex, atol=1e-9) and np.isclose(z[0], ez, atol=1e-9)


def test_position_at_release_plane_recovers_release_point():
    p = make_pitch(release_y=54.5)
    df = pd.DataFrame([{k: v for k, v in p.items() if not k.startswith("_")}])
    x, z = position_at_y(df, 54.5)
    assert np.isclose(x[0], p["release_pos_x"]) and np.isclose(z[0], p["release_pos_z"])


def test_time_is_negative_before_reference_plane():
    assert time_at_y(54.0, -130.0, 25.0) < 0 < time_at_y(30.0, -130.0, 25.0)


def test_identical_pitches_have_zero_separation():
    row = pitch_row()
    a = pd.DataFrame([row])
    sep = pair_separation(a, a.copy())
    assert sep["commit_sep_in"].iloc[0] == 0 and sep["plate_sep_in"].iloc[0] == 0


def test_separation_matches_hand_computation_and_grows_toward_plate():
    a = pitch_row(ax=-8.0, az=-20.0)
    b = pitch_row(ax=-14.0, az=-26.0)  # same release and velocity, different acceleration
    pa, pb = make_pitch(ax=-8.0, az=-20.0), make_pitch(ax=-14.0, az=-26.0)
    sep = pair_separation(pd.DataFrame([a]), pd.DataFrame([b]), commit_ft=23.8)
    y_c = PLATE_FRONT_Y + 23.8
    (xa, za), (xb, zb) = analytic_xz(pa, y_c), analytic_xz(pb, y_c)
    assert np.isclose(sep["commit_sep_in"].iloc[0], 12 * np.hypot(xa - xb, za - zb), atol=1e-6)
    assert sep["plate_sep_in"].iloc[0] > sep["commit_sep_in"].iloc[0] > 0


def test_reconstruction_error_flags_a_flipped_plate_x():
    good = pitch_row()
    bad = pitch_row()
    bad["plate_x"] = -bad["plate_x"]
    err = reconstruction_error(pd.DataFrame([good, bad]))
    assert err.iloc[0] < 1e-6 and err.iloc[1] > 2.0


def test_tunnel_ratio_floor_blocks_blowup():
    assert tunnel_ratio(0.0, 10.0) == 20.0
    assert np.isclose(tunnel_ratio(5.0, 10.0), 2.0)
