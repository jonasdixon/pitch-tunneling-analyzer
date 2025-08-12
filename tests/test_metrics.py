from __future__ import annotations
import pandas as pd
from tunneling.data import load_sample
from tunneling.sequences import build_sequences
from tunneling.metrics import add_metrics


def test_sequences_and_metrics_basic():
    df = load_sample()
    seq = build_sequences(df)

    # We should have at least some consecutive pairs
    assert len(seq) >= 2

    out = add_metrics(seq)
    for col in ["RP", "SD", "RP_norm", "SD_norm", "TunnelScore", "CSW"]:
        assert col in out.columns, f"Missing {col}"

    # Numerical sanity checks
    assert (out["RP"] >= 0).all()
    assert (out["SD"] >= 0).all()
    assert out["TunnelScore"].between(0, 1).all()

    # CSW from sample data should have at least one
    assert out["CSW"].sum() >= 1