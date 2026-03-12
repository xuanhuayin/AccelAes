#!/usr/bin/env python3
"""
Radar chart: all methods on a single hexagon.
Style: outer circle, concentric grey rings, legend top-right.
Metrics (6 axes): Speedup, CLIP, IR, HPS, Aesthetic, Edge
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/home/runkai/.local/share/fonts/Carlito/Carlito-Regular.ttf")
_fm.fontManager.addfont("/home/runkai/.local/share/fonts/Carlito/Carlito-Bold.ttf")

plt.rcParams.update({
    "font.family":      "Carlito",
    "figure.facecolor": "white",
    "pdf.fonttype":     42,
})

OUT_DIR = "/home/runkai/xuanhua/SASD/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Table 1 data ──────────────────────────────────────────────────────────────
METHODS = [
    ("Baseline",         "#9E9E9E", 1.00, 0.2531, 0.752, 0.2710, 5.941, 0.583),
    ("Δ-DiT",            "#E53935", 1.52, 0.2523, 0.485, 0.2540, 5.873, 0.522),
    ("FORA",             "#FB8C00", 1.00, 0.2463, 0.517, 0.2496, 5.766, 0.564),
    ("RAS",              "#1976D2", 1.47, 0.2550, 0.788, 0.2713, 5.927, 0.581),
    ("TeaCache",         "#7B1FA2", 1.49, 0.2512, 0.670, 0.2640, 5.978, 0.599),
    ("TaylorSeer",       "#00838F", 2.05, 0.2491, 0.604, 0.2650, 5.940, 0.668),
    ("AccelAes (Ours)",  "#2E7D32", 2.11, 0.2640, 0.841, 0.2740, 5.991, 0.629),
]

METRIC_LABELS = ["Speedup", "CLIP", "IR", "HPS", "Aesthetic", "Edge"]
METRIC_COLS   = [2, 3, 4, 5, 6, 7]
N = len(METRIC_LABELS)

# ── Normalise ─────────────────────────────────────────────────────────────────
vals  = np.array([[m[ci] for ci in METRIC_COLS] for m in METHODS])
vmin  = vals.min(axis=0)
vmax  = vals.max(axis=0)
norms = np.clip((vals - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)

angles   = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_c = np.append(angles, angles[0])

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10))
ax  = fig.add_subplot(111, polar=True)
ax.set_facecolor("white")

ax.spines["polar"].set_visible(True)
ax.spines["polar"].set_color("#AAAAAA")
ax.spines["polar"].set_linewidth(2.0)

RMAX = 1.18

# Concentric grey rings
for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.plot(angles_c, np.full(N + 1, r), color="#CCCCCC", lw=1.2, zorder=1, clip_on=False)

# Spokes
for a in angles:
    ax.plot([0, a], [0, RMAX], color="#CCCCCC", lw=1.2, zorder=1)

# Axis labels
label_r_extra = {
    "Aesthetic": RMAX + 0.38,
    "CLIP":      RMAX + 0.28,
}
label_va_extra = {"CLIP": "top"}
for i, lbl in enumerate(METRIC_LABELS):
    a  = angles[i]
    r  = label_r_extra.get(lbl, RMAX + 0.28)
    va = label_va_extra.get(lbl, "center")
    ax.text(a, r, lbl, ha="center", va=va,
            fontsize=30, color="#222222", fontweight="bold", zorder=7)

# Data polygons
for i, m in enumerate(METHODS):
    name, color = m[0], m[1]
    is_ours = "Ours" in name
    v     = np.append(norms[i], norms[i][0])
    lw    = 5.0 if is_ours else 3.0
    alpha = 0.18 if is_ours else 0.09
    ax.plot(angles_c, v, color=color, lw=lw, zorder=4 + int(is_ours), clip_on=False)
    ax.fill(angles_c, v, color=color, alpha=alpha, zorder=3)

ax.set_ylim(0, RMAX)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_yticks([])
ax.set_xticks([])

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color=m[1],
                  linewidth=5.0 if "Ours" in m[0] else 3.0,
                  label=m[0], linestyle="-")
    for m in METHODS
]
ax.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(1.72, 1.22),
    fontsize=28,
    frameon=True,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    handlelength=2.2,
    handletextpad=0.8,
    labelspacing=0.55,
)

fig.tight_layout(pad=2.0)

for ext, dpi in [("png", 200), ("pdf", 300)]:
    path = f"{OUT_DIR}/radar_all_in_one.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  {path}")
plt.close(fig)
print("Done.")
