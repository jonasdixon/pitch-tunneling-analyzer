"""Command line: tunneling fetch | pairs | effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tunneling import effects, integrity, pairs, sources, viz
from tunneling.kinematics import DEFAULT_COMMIT_FT


def _cmd_fetch(a):
    sources.fetch_range(a.start, a.end, a.cache)


def _load(a) -> pd.DataFrame:
    if a.infile:
        return sources.load_file(a.infile)
    return sources.load_cached(a.cache, a.start, a.end)


def _cmd_pairs(a):
    raw = _load(a)
    clean, report = integrity.clean(raw)
    print(json.dumps(report, indent=1))
    scored = pairs.score_pairs(pairs.build_pairs(clean), a.commit_ft)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(a.out, index=False)
    print(f"{len(scored):,} scored pairs -> {a.out}")


def _cmd_effects(a):
    scored = pd.read_parquet(a.pairs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, tables = {}, []
    for name in a.outcomes:
        df, y = effects.outcome_frame(scored, name)
        res = effects.fit_effects(df, y, name, n_boot=a.n_boot, n_jobs=a.jobs)
        results[name] = res
        tables.append(res.contrasts.assign(lag=1, sample="all pairs"))
        if a.placebo:
            col = effects.OUTCOMES[name][0]
            both = df.loc[df["l2_commit_sep_in"].notna()].reset_index(drop=True)
            for lag in (1, 2):
                d = effects.use_lag(both, lag)
                fit = effects.fit_effects(d, d[col].astype(int).to_numpy(), name, n_boot=a.n_boot, n_jobs=a.jobs)
                tables.append(fit.contrasts.assign(lag=lag, sample="rows with a lag-2 pitch"))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "effect_contrasts.csv", index=False)
    viz.effect_curves(results, str(out / "effect_curves.png"))
    if a.placebo:
        viz.placebo_plot(table.loc[table["sample"] == "rows with a lag-2 pitch"], str(out / "lag_placebo.png"))
    print(table.round(3).to_string(index=False))


def _cmd_sensitivity(a):
    base = pd.read_parquet(a.pairs)
    rows = []
    for ft in a.commit_ft:
        scored = pairs.score_pairs(base, ft)
        for name in a.outcomes:
            df, y = effects.outcome_frame(scored, name)
            res = effects.fit_effects(df, y, name, n_boot=0, max_rows=250_000)
            rows.append(res.contrasts.assign(commit_ft=ft))
    table = pd.concat(rows, ignore_index=True).drop(columns=["ci_lo_pp", "ci_hi_pp"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(a.out) / "commit_distance_sensitivity.csv", index=False)
    print(table.round(2).to_string(index=False))


def _cmd_holdout(a):
    train, test = pd.read_parquet(a.train), pd.read_parquet(a.test)
    rows = []
    for name in a.outcomes:
        d_tr, y_tr = effects.outcome_frame(train, name)
        d_te, y_te = effects.outcome_frame(test, name)
        if len(d_tr) > 300_000:
            keep = pd.Series(range(len(d_tr))).sample(300_000, random_state=0).to_numpy()
            d_tr, y_tr = d_tr.iloc[keep].reset_index(drop=True), y_tr[keep]
        rows.append({"outcome": name, **effects.holdout_log_loss(d_tr, y_tr, d_te, y_te)})
    table = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(a.out) / "holdout_log_loss.csv", index=False)
    print(table.round(5).to_string(index=False))


def _cmd_eda(a):
    from tunneling.kinematics import reconstruction_error

    raw = _load(a)
    complete = raw.dropna(subset=["release_pos_x", "vx0", "ay", "plate_x"])
    err = reconstruction_error(complete)
    scored = pd.read_parquet(a.pairs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    viz.eda_features(scored, err, str(out / "feature_distributions.png"))
    viz.binned_descriptive(scored, str(out / "whiff_by_decile_descriptive.png"))
    summary = {
        "recon_err_in": {"median": float(err.median()), "p99": float(err.quantile(0.99)), "max": float(err.max())},
        "feature_quantiles": scored[["commit_sep_in", "plate_sep_in", "velo_diff_mph", "tunnel_ratio"]]
        .quantile([0.1, 0.5, 0.9]).round(2).to_dict(),
        "corr_commit_plate": float(scored["commit_sep_in"].corr(scored["plate_sep_in"])),
        "corr_commit_l1_l2": float(scored["commit_sep_in"].corr(scored["l2_commit_sep_in"])),
        "pairs": len(scored),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tunneling", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download Statcast days into a cache directory")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--cache", default="data/cache")
    f.set_defaults(func=_cmd_fetch)

    q = sub.add_parser("pairs", help="clean, pair and score pitches")
    q.add_argument("--infile")
    q.add_argument("--cache", default="data/cache")
    q.add_argument("--start", default="2023-03-01")
    q.add_argument("--end", default="2023-11-30")
    q.add_argument("--commit-ft", type=float, default=DEFAULT_COMMIT_FT)
    q.add_argument("--out", default="data/processed/pairs.parquet")
    q.set_defaults(func=_cmd_pairs)

    s = sub.add_parser("sensitivity", help="re-fit point estimates at several commit distances")
    s.add_argument("--pairs", default="data/processed/pairs_2023.parquet")
    s.add_argument("--out", default="reports")
    s.add_argument("--commit-ft", nargs="+", type=float, default=[20.0, 23.8, 27.0, 30.0])
    s.add_argument("--outcomes", nargs="+", default=["whiff", "chase", "csw"])
    s.set_defaults(func=_cmd_sensitivity)

    h = sub.add_parser("holdout", help="train on one season, score the next, with and without tunnel features")
    h.add_argument("--train", default="data/processed/pairs_2023.parquet")
    h.add_argument("--test", default="data/processed/pairs_2024.parquet")
    h.add_argument("--out", default="reports")
    h.add_argument("--outcomes", nargs="+", default=["whiff", "chase", "csw"])
    h.set_defaults(func=_cmd_holdout)

    e = sub.add_parser("eda", help="feature distributions and reconstruction check")
    e.add_argument("--infile")
    e.add_argument("--cache", default="data/cache")
    e.add_argument("--start", default="2023-03-01")
    e.add_argument("--end", default="2023-11-30")
    e.add_argument("--pairs", default="data/processed/pairs.parquet")
    e.add_argument("--out", default="reports/figures")
    e.set_defaults(func=_cmd_eda)

    e = sub.add_parser("effects", help="fit the controlled outcome models")
    e.add_argument("--pairs", default="data/processed/pairs.parquet")
    e.add_argument("--out", default="reports")
    e.add_argument("--outcomes", nargs="+", default=["whiff", "chase", "csw"])
    e.add_argument("--n-boot", type=int, default=30)
    e.add_argument("--jobs", type=int, default=4)
    e.add_argument("--placebo", action="store_true", help="also fit with lag-2 features")
    e.set_defaults(func=_cmd_effects)

    a = p.parse_args(argv)
    a.func(a)
    return 0
