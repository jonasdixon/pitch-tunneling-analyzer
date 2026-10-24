"""Figures. Binned plots are descriptive only and say so in the title."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {
    "commit_sep_in": "Separation at commit point (in)",
    "plate_sep_in": "Separation at plate (in)",
    "velo_diff_mph": "Velocity drop, previous minus current (mph)",
}


def eda_features(scored: pd.DataFrame, recon_err: pd.Series, path: str) -> None:
    """Distributions of the derived features and the trajectory reconstruction error."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    spec = [
        ("commit_sep_in", 0, 40), ("plate_sep_in", 0, 60), ("velo_diff_mph", -15, 30),
    ]
    for ax, (col, lo, hi) in zip(axes, spec, strict=False):
        ax.hist(scored[col].clip(lo, hi), bins=60, color="#2f5d8a")
        ax.set_xlabel(LABELS[col])
        ax.set_yticks([])
    axes[3].hist(recon_err.clip(0, 2), bins=60, color="#8a5a2f")
    axes[3].set_xlabel("Rebuilt vs reported plate location (in)")
    axes[3].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def binned_descriptive(scored: pd.DataFrame, path: str) -> None:
    """Whiff rate per swing by decile of each feature. Descriptive only; the model does not use bins."""
    swings = scored.loc[scored["swing"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, col in zip(axes, ["commit_sep_in", "plate_sep_in", "velo_diff_mph"], strict=True):
        q = pd.qcut(swings[col], 10, duplicates="drop")
        m = swings.groupby(q, observed=True)["whiff"].mean()
        x = swings.groupby(q, observed=True)[col].median()
        ax.plot(x, m, "o-", color="#2f5d8a")
        ax.set_xlabel(LABELS[col])
    axes[0].set_ylabel("Whiff per swing")
    fig.suptitle("Descriptive only: raw deciles, no controls", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def effect_curves(results: dict, path: str) -> None:
    """Adjusted outcome probability as each feature is set to a value for every pitch."""
    feats = ["commit_sep_in", "plate_sep_in", "velo_diff_mph"]
    fig, axes = plt.subplots(
        len(results), 3, figsize=(12, 2.9 * len(results)), squeeze=False, sharey="row"
    )
    for i, (name, res) in enumerate(results.items()):
        for j, f in enumerate(feats):
            ax, g = axes[i][j], res.grids[f]
            lo, hi = np.percentile(res.boot_curves[f], [2.5, 97.5], axis=0)
            ax.fill_between(g, 100 * lo, 100 * hi, color="#9dbbd8", alpha=0.7)
            ax.plot(g, 100 * res.point_curves[f], color="#1d3f63")
            ax.set_xlabel(LABELS[f] if i == len(results) - 1 else "")
            if j == 0:
                ax.set_ylabel(f"{name} (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def placebo_plot(table: pd.DataFrame, path: str) -> None:
    """Lag-1 versus lag-2 contrast (10th to 90th percentile of a feature), with 95% intervals."""
    t = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.errorbar(
        t["effect_pp"], y,
        xerr=[t["effect_pp"] - t["ci_lo_pp"], t["ci_hi_pp"] - t["effect_pp"]],
        fmt="o", color="#1d3f63", capsize=3,
    )
    ax.set_yticks(y, [f"{a} / {b} / lag {c}" for a, b, c in zip(
        t["outcome"], t["feature"], t["lag"], strict=True)], fontsize=8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Change in outcome rate, 10th to 90th percentile (points)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
