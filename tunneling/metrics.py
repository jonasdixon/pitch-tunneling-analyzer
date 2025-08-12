from __future__ import annotations
import numpy as np
import pandas as pd


def _euclid(a, b):
    return float(np.sqrt(np.sum((np.array(a, dtype=float) - np.array(b, dtype=float)) ** 2)))


def add_metrics(seq: pd.DataFrame) -> pd.DataFrame:
    """
    Given a sequence DataFrame from build_sequences, add:
      - RP: release proximity (lower is better deception)
      - SD: separation at plate (higher is better divergence)
      - RP_norm, SD_norm: 0..1 scaled by robust percentiles
      - TunnelScore: 0..1 where higher is better (combines low RP and high SD)
      - CSW: flag whether current pitch is a CSW (called strike or whiff types)
    """
    seq = seq.copy()

    # Compute RP (distance between release points)
    seq["RP"] = [
        _euclid(
            (r.prev_release_pos_x, r.prev_release_pos_y, r.prev_release_pos_z),
            (r.release_pos_x, r.release_pos_y, r.release_pos_z),
        )
        for r in seq.itertuples(index=False)
    ]

    # Compute SD (distance at the plate)
    seq["SD"] = [
        _euclid((r.prev_plate_x, r.prev_plate_z), (r.plate_x, r.plate_z))
        for r in seq.itertuples(index=False)
    ]

    # Robust scaling using percentiles to avoid outliers dominating
    def robust_scale(s: pd.Series, invert: bool = False) -> pd.Series:
        lo, hi = np.nanpercentile(s, [5, 95])
        # prevent zero division
        rng = hi - lo if hi > lo else (np.nanmax(s) - np.nanmin(s) + 1e-9)
        scaled = (s - lo) / (rng if rng != 0 else 1.0)
        scaled = np.clip(scaled, 0.0, 1.0)
        if invert:
            scaled = 1.0 - scaled
        return scaled

    # Lower RP is better -> invert for scoring
    seq["RP_norm"] = robust_scale(seq["RP"], invert=True)
    seq["SD_norm"] = robust_scale(seq["SD"], invert=False)

    # Weighted combination (tweak weights if you like)
    w_rp, w_sd = 0.4, 0.6
    seq["TunnelScore"] = w_rp * seq["RP_norm"] + w_sd * seq["SD_norm"]

    # Outcome flag: CSW (called strikes + swinging strikes + foul tips)
    # You may refine this mapping for your dataset.
    csw_labels = {
        "called_strike",
        "swinging_strike",
        "swinging_strike_blocked",
        "foul_tip",
    }
    seq["CSW"] = seq["description"].astype(str).str.lower().isin(csw_labels).astype(int)

    return seq