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
        tables.append(res.contrasts.assign(lag=1))
        if a.placebo:
            d2 = effects.use_lag(df, 2)
            y2 = d2[effects.OUTCOMES[name][0]].astype(int).to_numpy()
            tables.append(
                effects.fit_effects(d2, y2, name, n_boot=a.n_boot, n_jobs=a.jobs).contrasts.assign(lag=2)
            )
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "effect_contrasts.csv", index=False)
    viz.effect_curves(results, str(out / "effect_curves.png"))
    print(table.round(3).to_string(index=False))


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
