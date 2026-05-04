"""Render IF V1 — anomalies by month (same logic as FE.ipynb) for PPT asset."""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
from matplotlib import pyplot as plt
import pandas as pd

COLORS = {
    "imports": "#E94F37",
    "text": "#2D2D2D",
    "grid": "#E8E8E8",
}

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/final/anomaly_results.csv")
df["date"] = pd.to_datetime(df["date"])
monthly = df.groupby("date")["is_anomaly"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(monthly["date"], monthly["is_anomaly"], width=25, color=COLORS["imports"], alpha=0.8)
ax.set_ylabel("Number of Anomalies")
ax.set_title("Trade Anomalies by Month (2017–2024)")
ax.set_xlabel("Month")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

out = Path(__file__).resolve().parent / "ppt_if_anomalies_by_month.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Wrote", out)
