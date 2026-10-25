import numpy as np
import pandas as pd

from conftest import pitch_row
from tunneling.integrity import clean, integrity_report
from tunneling.outcomes import (
    add_outcome_flags,
    drop_non_competitive,
    in_zone,
    summarize_outcomes,
)
from tunneling.pairs import build_pairs, score_pairs


def test_pairs_require_same_game_pitcher_and_consecutive_numbers():
    rows = [
        pitch_row(game_pk=1, ab=3, n=1), pitch_row(game_pk=1, ab=3, n=2),
        pitch_row(game_pk=1, ab=3, n=4),  # pitch 3 was never tracked: no pair with pitch 2
        pitch_row(game_pk=1, ab=3, n=3, pitcher=11),  # relief pitcher, same plate appearance
    ]
    pairs = build_pairs(pd.DataFrame(rows))
    assert sorted(zip(pairs["pitcher"], pairs["pitch_number"], strict=True)) == [(10, 2)]


def test_doubleheader_at_bat_numbers_do_not_collide():
    rows = [
        pitch_row(game_pk=1, ab=5, n=1), pitch_row(game_pk=2, ab=5, n=2, pitcher=10),
    ]
    assert len(build_pairs(pd.DataFrame(rows))) == 0


def test_second_lag_only_when_pitch_numbers_are_consecutive():
    rows = [pitch_row(n=i) for i in (1, 2, 3)]
    pairs = build_pairs(pd.DataFrame(rows))
    third = pairs.loc[(pairs["at_bat_number"] == 1) & (pairs["pitch_number"] == 3)].iloc[0]
    second = pairs.loc[(pairs["at_bat_number"] == 1) & (pairs["pitch_number"] == 2)].iloc[0]
    assert pd.notna(third["prev2_vx0"]) and pd.isna(second["prev2_vx0"])


def test_score_pairs_adds_features_and_drops_pitchouts():
    rows = [
        pitch_row(n=1, ptype="FF", speed=96), pitch_row(n=2, ptype="SL", speed=86, ax=-14, az=-26),
        pitch_row(n=3, desc="pitchout"),
    ]
    scored = score_pairs(build_pairs(pd.DataFrame(rows)))
    assert len(scored) == 1
    r = scored.iloc[0]
    assert r["velo_diff_mph"] == 10 and not r["same_type"] and r["plate_sep_in"] > 0


def test_chase_rate_uses_out_of_zone_pitches_as_denominator():
    df = pd.DataFrame({
        "description": ["swinging_strike", "ball", "ball", "foul", "called_strike"],
        "plate_x": [0.0, 1.5, 1.6, 1.4, 0.0], "plate_z": [2.5, 2.5, 2.5, 2.5, 2.5],
        "sz_top": 3.4, "sz_bot": 1.6, "zone": [5, 12, 12, 12, 5], "pitcher": 1,
    })
    s = summarize_outcomes(add_outcome_flags(df), ["pitcher"], min_n=1).iloc[0]
    assert s["out_of_zone"] == 3 and s["chases"] == 1 and np.isclose(s["chase_pct"], 1 / 3)
    assert np.isclose(s["whiff_pct"], 1 / 2)  # 1 whiff on 2 swings (swinging_strike, foul)


def test_bunt_swings_and_whiffs_are_counted():
    df = pd.DataFrame({"description": ["missed_bunt", "foul_bunt", "bunt_foul_tip"]})
    df = df.assign(plate_x=0.0, plate_z=2.5, sz_top=3.4, sz_bot=1.6)
    f = add_outcome_flags(df)
    assert f["swing"].all() and list(f["whiff"]) == [True, False, False]


def test_zone_falls_back_to_geometry_when_zone_missing():
    df = pd.DataFrame({
        "plate_x": [0.0, 0.9, 0.0], "plate_z": [2.5, 2.5, 4.0],
        "sz_top": 3.4, "sz_bot": 1.6, "zone": [np.nan, np.nan, np.nan],
    })
    assert list(in_zone(df)) == [True, False, False]


def test_non_competitive_pitches_are_dropped():
    df = pd.DataFrame({"description": ["ball", "automatic_ball", "pitchout"]})
    assert list(drop_non_competitive(df)["description"]) == ["ball"]


def test_integrity_rules_catch_planted_bugs():
    good = pitch_row(n=1)
    dup = dict(good)
    fast = pitch_row(n=2, speed=125.0)
    flipped = pitch_row(n=3)
    flipped["plate_x"] = -flipped["plate_x"]
    missing = pitch_row(n=4)
    missing["vx0"] = np.nan
    below = pitch_row(n=5)
    below["plate_z"] = -9.0
    df = pd.DataFrame([good, dup, fast, flipped, missing, below])
    report = integrity_report(df)
    assert report["duplicate_key"] == 1
    assert report["impossible_speed"] == 1
    assert report["trajectory_mismatch"] >= 1
    assert report["missing_kinematics"] == 1
    assert report["impossible_plate_z"] == 1
    cleaned, counts = clean(df)
    assert len(cleaned) == 1 and counts["rows_in"] == 6
