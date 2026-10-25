import numpy as np
import pandas as pd

from tunneling.effects import SplineLogit, fit_effects, outcome_frame, use_lag


def simulate(n=12_000, n_pitchers=60, seed=3):
    """Whiff depends on plate separation (+) and location, not on commit separation."""
    rng = np.random.default_rng(seed)
    pitcher = rng.integers(0, n_pitchers, n)
    p_eff = rng.normal(0, 0.3, n_pitchers)[pitcher]
    commit = rng.gamma(3, 2, n)
    plate = commit * rng.uniform(1, 4, n) + rng.gamma(2, 2, n)
    velo = rng.normal(8, 4, n)
    px, pz = rng.normal(0, 0.8, n), rng.normal(2.5, 0.8, n)
    eta = -1.2 + 0.06 * (plate - 15) - 0.5 * px**2 + p_eff
    whiff = rng.random(n) < 1 / (1 + np.exp(-eta))
    return pd.DataFrame({
        "pitcher": pitcher, "commit_sep_in": commit, "plate_sep_in": plate, "velo_diff_mph": velo,
        "plate_x": px, "plate_z": pz, "release_speed": 90 - velo / 2,
        "edge_dist": rng.normal(0, 0.5, n), "cur_commit_x_ft": rng.normal(0, 0.6, n),
        "cur_commit_z_ft": rng.normal(3.5, 0.6, n), "prev_edge_dist": rng.normal(0, 0.5, n),
        "prev_result": rng.choice(["ball", "called_strike", "foul", "whiff", "other"], n),
        "pitch_type": rng.choice(["FF", "SL", "CH"], n), "prev_pitch_type": rng.choice(["FF", "SL"], n),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
        "stand": rng.choice(["L", "R"], n), "p_throws": "R", "same_type": rng.random(n) < 0.3,
        "swing": True, "whiff": whiff, "in_zone": rng.random(n) < 0.5, "csw": whiff,
        "chase": False,
    })


def test_model_recovers_planted_plate_separation_effect():
    df = simulate()
    d, y = outcome_frame(df, "whiff")
    res = fit_effects(d, y, "whiff", n_boot=6, n_jobs=1)
    c = res.contrasts.set_index("feature")
    assert c.loc["plate_sep_in", "effect_pp"] > 3
    assert c.loc["plate_sep_in", "ci_lo_pp"] > 0
    assert abs(c.loc["commit_sep_in", "effect_pp"]) < 4


def test_curve_is_monotone_for_a_monotone_truth():
    df = simulate()
    d, y = outcome_frame(df, "whiff")
    m = SplineLogit().fit(d, y)
    grid = np.linspace(8, 30, 12)
    curve = m.curve(d, "plate_sep_in", grid)
    assert np.all(np.diff(curve) > -0.005) and curve[-1] > curve[0]


def test_unseen_pitchers_get_zero_pitcher_effect():
    df = simulate()
    d, y = outcome_frame(df, "whiff")
    m = SplineLogit().fit(d.iloc[:6000], y[:6000])
    new = d.iloc[6000:6010].copy()
    new["pitcher"] = 99_999
    assert np.isfinite(m.predict(new)).all()


def test_use_lag_swaps_tunnel_columns():
    df = simulate(n=50)
    df["l2_commit_sep_in"] = df["commit_sep_in"] + 1
    df["l2_plate_sep_in"] = df["plate_sep_in"] + 1
    df["prev2_release_speed"] = df["release_speed"] + 3
    df["prev2_pitch_type"] = "FF"
    out = use_lag(df, 2)
    assert np.allclose(out["commit_sep_in"], df["commit_sep_in"] + 1)
    assert np.allclose(out["velo_diff_mph"], 3)
