"""Controlled model of outcomes given tunnel features.

The question is not "do high-TunnelScore pairs have more whiffs" (they do, partly because they are
different pitch types in different places) but "holding where the pitch ended up, what it was,
the count, the matchup and who threw it fixed, how does the outcome move with commit-point
separation, plate separation and velocity difference?".

Model: logistic regression with
  - cubic B-splines on each tunnel feature (continuous, not binned),
  - a tensor-product spline surface on plate location and a spline on signed distance to the zone
    edge (so the model knows how close to the zone the pitch finished),
  - a tensor-product spline on where the *current* pitch was at the commit point. Without it,
    commit-point separation partly measures "the current pitch is at an unusual place" rather than
    "the previous pitch was somewhere different", and the hitter reacts to the former directly,
  - a spline on release speed,
  - the previous pitch's context: signed distance of its location to the zone edge and its result
    (called strike, ball, whiff, foul, other). A hitter who just took a strike behaves differently
    whatever the trajectories were, and without this the separation features partly stand in for
    "where and how the last pitch ended",
  - one-hot current / previous pitch type, count, same-handed matchup, same-type flag,
  - a ridge-shrunk pitcher effect (columns scaled by `pitcher_scale`, which is a prior standard
    deviation of about that size on the logit scale).

Effects are reported as g-computation contrasts: set a feature to a value for every pitch in the
sample, leave everything else as observed, and average the predicted probability. Uncertainty is
a pitcher-cluster bootstrap, because pitches from one pitcher are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import SplineTransformer

TUNNEL_FEATURES = ["commit_sep_in", "plate_sep_in", "velo_diff_mph"]
OUTCOMES = {
    "whiff": ("whiff", lambda d: d["swing"], "Whiff per swing"),
    "chase": ("chase", lambda d: ~d["in_zone"], "Chase per out-of-zone pitch"),
    "csw": ("csw", lambda d: np.ones(len(d), dtype=bool), "Called strike or whiff per pitch"),
}


def outcome_frame(scored: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows at risk for `outcome` and the 0/1 response."""
    col, subset, _ = OUTCOMES[outcome]
    mask = np.asarray(subset(scored), dtype=bool)
    df = scored.loc[mask].reset_index(drop=True)
    return df, df[col].astype(int).to_numpy()


def use_lag(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Copy of `df` whose tunnel features describe the pitch `lag` pitches back (placebo uses lag 2)."""
    if lag == 1:
        return df
    out = df.loc[df["l2_commit_sep_in"].notna()].copy()
    out["commit_sep_in"] = out["l2_commit_sep_in"]
    out["plate_sep_in"] = out["l2_plate_sep_in"]
    out["velo_diff_mph"] = out["prev2_release_speed"] - out["release_speed"]
    out["prev_pitch_type"] = out["prev2_pitch_type"]
    out["same_type"] = out["prev_pitch_type"] == out["pitch_type"]
    return out.reset_index(drop=True)


def _spline(n_knots: int, bias: bool) -> SplineTransformer:
    return SplineTransformer(
        n_knots=n_knots, degree=3, knots="quantile", extrapolation="constant", include_bias=bias
    )


class SplineLogit:
    def __init__(self, tunnel_cols=None, pitcher_scale: float = 0.3, C: float = 1.0):
        self.tunnel_cols = list(tunnel_cols or TUNNEL_FEATURES)
        self.pitcher_scale = pitcher_scale
        self.C = C

    # ----- encoders -----
    def _fit_encoders(self, df: pd.DataFrame) -> None:
        self.bounds = {c: tuple(np.nanpercentile(df[c], [1, 99])) for c in self.tunnel_cols}
        self.tun_tf = {
            c: _spline(5, False).fit(self._clip(df, c).reshape(-1, 1)) for c in self.tunnel_cols
        }
        self.sx = _spline(6, True).fit(np.clip(df["plate_x"], -2, 2).to_numpy().reshape(-1, 1))
        self.sz = _spline(6, True).fit(np.clip(df["plate_z"], 0, 5).to_numpy().reshape(-1, 1))
        self.speed_bounds = tuple(np.nanpercentile(df["release_speed"], [1, 99]))
        self.speed_tf = _spline(4, False).fit(self._clip_speed(df).reshape(-1, 1))
        self.edge_tf = _spline(8, False).fit(np.clip(df["edge_dist"], -0.8, 1.5).to_numpy().reshape(-1, 1))
        self.pedge_tf = _spline(6, False).fit(np.clip(df["prev_edge_dist"], -0.8, 1.5).to_numpy().reshape(-1, 1))
        self.results = ["called_strike", "ball", "whiff", "foul", "other"]
        self.cx_tf = _spline(5, True).fit(np.clip(df["cur_commit_x_ft"], -3, 3).to_numpy().reshape(-1, 1))
        self.cz_tf = _spline(5, True).fit(np.clip(df["cur_commit_z_ft"], 0, 8).to_numpy().reshape(-1, 1))
        top = df["pitch_type"].value_counts().index[:8]
        self.types = list(top)
        self.counts = sorted({f"{b}-{s}" for b, s in zip(df["balls"], df["strikes"], strict=True)})
        self.pitchers = {p: i for i, p in enumerate(pd.unique(df["pitcher"]))}

    def _clip(self, df, c):
        lo, hi = self.bounds[c]
        return np.clip(df[c].to_numpy(float), lo, hi)

    def _clip_speed(self, df):
        return np.clip(df["release_speed"].to_numpy(float), *self.speed_bounds)

    def _onehot(self, values: pd.Series, levels: list) -> np.ndarray:
        return np.column_stack([(values == lv).to_numpy(float) for lv in levels])

    def design(self, df: pd.DataFrame) -> sparse.csr_matrix:
        blocks, self.slices, start = [], {}, 0

        def add(name, arr):
            nonlocal start
            blocks.append(arr if sparse.issparse(arr) else sparse.csr_matrix(arr))
            self.slices[name] = slice(start, start + blocks[-1].shape[1])
            start += blocks[-1].shape[1]

        for c in self.tunnel_cols:
            add(c, self.tun_tf[c].transform(self._clip(df, c).reshape(-1, 1)))
        bx = self.sx.transform(np.clip(df["plate_x"], -2, 2).to_numpy().reshape(-1, 1))
        bz = self.sz.transform(np.clip(df["plate_z"], 0, 5).to_numpy().reshape(-1, 1))
        add("location", (bx[:, :, None] * bz[:, None, :]).reshape(len(df), -1))
        add("speed", self.speed_tf.transform(self._clip_speed(df).reshape(-1, 1)))
        add("edge", self.edge_tf.transform(np.clip(df["edge_dist"], -0.8, 1.5).to_numpy().reshape(-1, 1)))
        cx = self.cx_tf.transform(np.clip(df["cur_commit_x_ft"], -3, 3).to_numpy().reshape(-1, 1))
        cz = self.cz_tf.transform(np.clip(df["cur_commit_z_ft"], 0, 8).to_numpy().reshape(-1, 1))
        add("commit_position", (cx[:, :, None] * cz[:, None, :]).reshape(len(df), -1))
        add("prev_edge", self.pedge_tf.transform(np.clip(df["prev_edge_dist"], -0.8, 1.5).to_numpy().reshape(-1, 1)))
        add("prev_result", self._onehot(df["prev_result"], self.results))
        add("cur_type", self._onehot(df["pitch_type"], self.types))
        add("prev_type", self._onehot(df["prev_pitch_type"], self.types))
        count = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
        add("count", self._onehot(count, self.counts))
        add("flags", np.column_stack([
            (df["stand"] == df["p_throws"]).to_numpy(float), df["same_type"].to_numpy(float),
        ]))
        idx = df["pitcher"].map(self.pitchers)
        ok = idx.notna().to_numpy()
        pit = sparse.csr_matrix(
            (np.full(ok.sum(), self.pitcher_scale), (np.flatnonzero(ok), idx[ok].astype(int))),
            shape=(len(df), len(self.pitchers)),
        )
        add("pitcher", pit)
        return sparse.hstack(blocks, format="csr")

    # ----- fit / predict -----
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> SplineLogit:
        self._fit_encoders(df)
        X = self.design(df)
        self.clf = LogisticRegression(C=self.C, max_iter=500, solver="lbfgs").fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.design(df))[:, 1]

    # ----- g-computation -----
    def curve(self, df: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
        """Mean predicted probability when `col` is set to each value in `grid` for every row."""
        X = self.design(df)
        coef, sl = self.clf.coef_[0], self.slices[col]
        eta_rest = X @ coef + self.clf.intercept_[0] - X[:, sl] @ coef[sl]
        basis = self.tun_tf[col].transform(np.clip(grid, *self.bounds[col]).reshape(-1, 1))
        contrib = basis @ coef[sl]
        return np.array([_sigmoid(eta_rest + c).mean() for c in contrib])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class EffectResult:
    outcome: str
    grids: dict
    point_curves: dict
    boot_curves: dict
    contrasts: pd.DataFrame
    n_rows: int
    n_pitchers: int


def _contrast_points(df, cols=TUNNEL_FEATURES, lo_q=0.10, hi_q=0.90):
    pts = {}
    for c in cols:
        lo, hi = np.nanpercentile(df[c], [lo_q * 100, hi_q * 100])
        pts[c] = (lo, hi)
    return pts


def _grid(df, col, n=25):
    lo, hi = np.nanpercentile(df[col], [2, 98])
    return np.linspace(lo, hi, n)


def fit_effects(
    df: pd.DataFrame,
    y: np.ndarray,
    outcome: str,
    n_boot: int = 30,
    seed: int = 7,
    max_rows: int = 300_000,
    n_jobs: int = 4,
    model_kwargs: dict | None = None,
) -> EffectResult:
    """Point fit plus pitcher-cluster bootstrap for curves and 10th-to-90th percentile contrasts."""
    model_kwargs = model_kwargs or {}
    rng = np.random.default_rng(seed)
    if len(df) > max_rows:
        keep = rng.choice(len(df), size=max_rows, replace=False)
        df, y = df.iloc[keep].reset_index(drop=True), y[keep]
    grids = {c: _grid(df, c) for c in TUNNEL_FEATURES}
    points = _contrast_points(df)

    def summarize(model, d):
        curves = {c: model.curve(d, c, grids[c]) for c in TUNNEL_FEATURES}
        lohi = {
            c: model.curve(d, c, np.array(points[c])) for c in TUNNEL_FEATURES
        }
        return curves, lohi

    model = SplineLogit(**model_kwargs).fit(df, y)
    point_curves, point_lohi = summarize(model, df)

    by_pitcher = df.groupby("pitcher").indices
    pitchers = np.array(list(by_pitcher.keys()))

    def boot(seed_b):
        r = np.random.default_rng(seed_b)
        draw = r.choice(pitchers, size=len(pitchers), replace=True)
        idx = np.concatenate([by_pitcher[p] for p in draw])
        d, yb = df.iloc[idx].reset_index(drop=True), y[idx]
        m = SplineLogit(**model_kwargs).fit(d, yb)
        return summarize(m, d)

    results = Parallel(n_jobs=n_jobs)(delayed(boot)(seed + 1 + b) for b in range(n_boot)) if n_boot else []
    boot_curves = {c: np.vstack([r[0][c] for r in results]) for c in TUNNEL_FEATURES} if results else {}
    rows = []
    for c in TUNNEL_FEATURES:
        est = point_lohi[c][1] - point_lohi[c][0]
        b = np.array([r[1][c][1] - r[1][c][0] for r in results]) if results else np.array([np.nan])
        rows.append({
            "outcome": outcome, "feature": c,
            "p10": points[c][0], "p90": points[c][1],
            "effect_pp": 100 * est,
            "ci_lo_pp": 100 * np.percentile(b, 2.5),
            "ci_hi_pp": 100 * np.percentile(b, 97.5),
        })
    return EffectResult(
        outcome=outcome, grids=grids, point_curves=point_curves, boot_curves=boot_curves,
        contrasts=pd.DataFrame(rows), n_rows=len(df), n_pitchers=len(pitchers),
    )


def holdout_log_loss(train: pd.DataFrame, y_train, test: pd.DataFrame, y_test) -> dict:
    """Log loss on `test` for the full model, a model without tunnel features, and the base rate."""
    full = SplineLogit().fit(train, y_train)
    no_tunnel = SplineLogit(tunnel_cols=["velo_diff_mph"]).fit(train, y_train)
    base = np.full(len(test), float(np.mean(y_train)))
    return {
        "full": log_loss(y_test, full.predict(test)),
        "without_commit_and_plate_sep": log_loss(y_test, no_tunnel.predict(test)),
        "base_rate": log_loss(y_test, base),
    }
