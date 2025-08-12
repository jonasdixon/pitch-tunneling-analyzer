from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

from tunneling.data import load_sample
from tunneling.sequences import build_sequences
from tunneling.metrics import add_metrics


def main():
    df = load_sample()
    seq = build_sequences(df)
    out = add_metrics(seq)

    print("First 5 rows with metrics:")
    print(out[["prev_pitch_type", "pitch_type", "RP", "SD", "TunnelScore", "CSW"]].head())

    # Simple visualization: TunnelScore distribution, and average CSW by TunnelScore decile
    out["decile"] = pd.qcut(out["TunnelScore"], 10, labels=False, duplicates="drop")
    csw_by_decile = out.groupby("decile", dropna=True)["CSW"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(out["TunnelScore"], bins=10, color="#2e7d32", alpha=0.8)
    axes[0].set_title("TunnelScore distribution")
    axes[0].set_xlabel("TunnelScore"); axes[0].set_ylabel("Count")

    csw_by_decile.plot(kind="bar", ax=axes[1], color="#c2410c", alpha=0.9)
    axes[1].set_title("Avg CSW by TunnelScore decile")
    axes[1].set_xlabel("Decile (low→high)"); axes[1].set_ylabel("CSW rate")

    fig.tight_layout()
    os.makedirs("examples/out", exist_ok=True)
    out_path = "examples/out/tunnel_demo.png"
    plt.savefig(out_path, dpi=140)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()