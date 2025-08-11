import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def tunnel_quintile_effect(df: pd.DataFrame, value_col: str = "csw"):
    d = df.copy()
    d = d.dropna(subset=["TunnelScore"])
    d["q"] = pd.qcut(d["TunnelScore"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    res = d.groupby("q").agg(
        n=("pitch_number", "size"),
        csw=("csw", "mean"),
        swing=("swing", "mean"),
        whiff=("whiff", "mean"),
        chase=("chase", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.lineplot(data=res, x="q", y=value_col, marker="o", ax=ax)
    ax.set_ylabel(f"{value_col} rate")
    ax.set_xlabel("TunnelScore quintile (low → high)")
    ax.set_title(f"{value_col} vs TunnelScore quintile")
    plt.tight_layout()
    return fig

def movement_release_scatter(df: pd.DataFrame, sample=20000):
    if len(df) == 0:
        raise ValueError("Empty DataFrame provided")
    d = df.sample(min(sample, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(data=d, x="prev_release_pos_x", y="prev_release_pos_z", s=5, alpha=0.3, ax=axes[0])
    axes[0].set_title("Prev release X vs Z (sample)")
    sns.scatterplot(data=d, x="release_pos_x", y="release_pos_z", s=5, alpha=0.3, ax=axes[1])
    axes[1].set_title("Current release X vs Z (sample)")
    plt.tight_layout()
    return fig
