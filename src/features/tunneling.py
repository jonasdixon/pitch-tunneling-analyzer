import numpy as np
import pandas as pd

def _z_by_group(df: pd.DataFrame, cols, by):
    df = df.copy()
    for c in cols:
        df[c + "_z"] = df.groupby(by)[c].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    return df

def add_tunnel_features(pairs: pd.DataFrame, alpha_speed: float = 0.5, beta_plate: float = 1.0):
    cols_release = ["release_pos_x", "release_pos_y", "release_pos_z", "release_speed"]
    cols_prev_release = ["prev_" + c for c in cols_release]
    cols_move_plate = ["pfx_x", "pfx_z", "plate_x", "plate_z"]
    cols_prev_move_plate = ["prev_" + c for c in cols_move_plate]

    needed = set(["pitcher", "season"] + cols_release + cols_prev_release + cols_move_plate + cols_prev_move_plate)
    missing = needed - set(pairs.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = pairs.copy()

    # Standardize features by pitcher-season
    by = ["pitcher", "season"]

    tmp = df.copy()
    # Create 'prevcopy' so we can z-score prev/current with same group stats
    for c in cols_prev_release + cols_prev_move_plate:
        base = c.replace("prev_", "")
        tmp[base + "_prevcopy"] = tmp[c]

    to_z = (
        cols_release
        + cols_move_plate
        + [c + "_prevcopy" for c in (cols_release + cols_move_plate)]
    )
    tmp = _z_by_group(tmp, to_z, by)

    # Extract z-scored arrays
    cur_rel = tmp[[c + "_z" for c in cols_release]].to_numpy()
    prev_rel = tmp[[c + "_prevcopy_z" for c in cols_release]].to_numpy()
    cur_mvpl = tmp[[c + "_z" for c in cols_move_plate]].to_numpy()
    prev_mvpl = tmp[[c + "_prevcopy_z" for c in cols_move_plate]].to_numpy()

    # Weight speed dimension in release vector
    w_cur_rel = cur_rel.copy()
    w_prev_rel = prev_rel.copy()
    w_cur_rel[:, 3] *= np.sqrt(alpha_speed)
    w_prev_rel[:, 3] *= np.sqrt(alpha_speed)

    # RP: smaller is better
    rp = np.linalg.norm(w_cur_rel - w_prev_rel, axis=1)

    # Emphasize plate separation at finish (beta) on plate_x, plate_z
    w_cur_mvpl = cur_mvpl.copy()
    w_prev_mvpl = prev_mvpl.copy()
    w_cur_mvpl[:, 2:] *= np.sqrt(beta_plate)
    w_prev_mvpl[:, 2:] *= np.sqrt(beta_plate)

    # SD: larger is better
    sd = np.linalg.norm(w_cur_mvpl - w_prev_mvpl, axis=1)

    df["RP"] = rp
    df["SD"] = sd

    def _z(x):
        return (x - x.mean()) / (x.std(ddof=0) + 1e-9)

    df["RP_z"] = df.groupby(by)["RP"].transform(_z)
    df["SD_z"] = df.groupby(by)["SD"].transform(_z)
    df["TunnelScore"] = df["SD_z"] - df["RP_z"]

    return df
