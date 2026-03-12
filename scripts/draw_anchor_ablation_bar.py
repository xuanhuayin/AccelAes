#!/usr/bin/env python3
"""
Anchor ablation bar chart: Uniform vs Non-aesthetic Token vs AesMask (Ours).
3 metrics: ImageReward, Aesthetic, CLIP Score.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

_fm.fontManager.addfont("/home/runkai/.local/share/fonts/Carlito/Carlito-Regular.ttf")
_fm.fontManager.addfont("/home/runkai/.local/share/fonts/Carlito/Carlito-Bold.ttf")

plt.rcParams.update({
    "font.family":      "Carlito",
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "pdf.fonttype":     42,
})

OUT_DIR = "/home/runkai/xuanhua/SASD/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["Uniform", "Non-aesthetic\nToken", "AesMask\n(Ours)"]
COLORS  = ["#7D1A1A", "#C95030", "#E0C060"]

DATA = {
    "ImageReward ↑":  [0.822, 0.821, 0.841],
    "Aesthetic ↑":    [5.939, 5.925, 6.041],
    "CLIP Score ↑":   [0.2556, 0.2539, 0.2640],
}

# y-axis limits: (ymin, ymax) — tight headroom above max bar
YLIMS = {
    "ImageReward ↑":  (0.775, 0.854),
    "Aesthetic ↑":    (5.75,  6.09),
    "CLIP Score ↑":   (0.248,  0.268),
}

fig, axes = plt.subplots(1, 3, figsize=(26, 6.0))
fig.subplots_adjust(wspace=0.42)

x   = np.arange(len(METHODS))
bw  = 0.50

for ax, (metric, vals) in zip(axes, DATA.items()):
    ymin, ymax = YLIMS[metric]
    bars = ax.bar(x, vals, width=bw, color=COLORS,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # value labels on bars — 4 decimal places for CLIP Score
    fmt = ".4f" if "CLIP" in metric else ".3f"
    for xi, v in enumerate(vals):
        is_ours = (xi == 2)
        ax.text(xi, v + (ymax - ymin) * 0.012, f"{v:{fmt}}",
                ha="center", va="bottom",
                fontsize=20, fontweight="bold" if is_ours else "normal",
                color="#222222", zorder=5)

    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=22, multialignment="center")
    ax.tick_params(axis="y", labelsize=20)
    ax.set_title(metric, fontsize=26, fontweight="bold", pad=10)
    ax.grid(axis="y", ls="--", lw=0.8, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")

for ext, dpi in [("pdf", 300), ("png", 200)]:
    path = f"{OUT_DIR}/anchor_ablation_bar.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
