from tunneling import integrity, pairs, sources
from tunneling.kinematics import reconstruction_error


def test_packaged_sample_runs_end_to_end():
    raw = sources.load_sample()
    clean, report = integrity.clean(raw)
    assert report["duplicate_key"] == 0 and report["rows_out"] > 1000
    assert reconstruction_error(clean).median() < 0.2  # rebuilt plate location matches Statcast's
    scored = pairs.score_pairs(pairs.build_pairs(clean))
    assert len(scored) > 500
    assert (scored["commit_sep_in"] >= 0).all() and (scored["plate_sep_in"] >= 0).all()
    assert scored["commit_sep_in"].median() < scored["plate_sep_in"].median()  # pitches diverge toward the plate
    assert scored["chase"].sum() > 0 and scored["whiff"].sum() > 0


def test_pairs_never_cross_pitchers_or_plate_appearances():
    scored = pairs.score_pairs(pairs.build_pairs(integrity.clean(sources.load_sample())[0]))
    raw = sources.load_sample().set_index(["game_pk", "at_bat_number", "pitch_number"])
    for r in scored.sample(50, random_state=0).itertuples():
        prev = raw.loc[(r.game_pk, r.at_bat_number, r.pitch_number - 1)]
        assert prev["pitcher"] == r.pitcher
