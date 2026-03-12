#!/usr/bin/env python3
"""
Ablation component bar chart — AccelAes (Lumina-Next-T2I).

Design: single axis with component checkmark table embedded below x-axis.
Bars show Speedup (left y-axis, blue) and ImageReward (right y-axis, green).

Output: outputs/figures/ablation_components.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "pdf.fonttype":      42,
})

C_OURS  = "#2E7D32"
C_SPEED = "#1565C0"
C_IR    = "#388E3C"
C_LIGHT = "#E8F5E9"
C_GREY  = "#9E9E9E"

BASE_DIR = "/home/runkai/xuanhua/SASD"
OUT_DIR  = f"{BASE_DIR}/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
#  (label, Spatial CFG, Sparse Attn, Sparse FFN, Step Skip, speedup, IR)
CONFIGS = [
    ("Baseline",         False, False, False, False, 1.00, 0.752),
    ("Step Skip",        False, False, False, True,  1.57, 0.743),
    ("Sem. Mask",        True,  True,  False, False, 1.42, 0.818),
    ("+ Sparse FFN",     True,  True,  True,  False, 1.43, 0.818),
    ("AccelAes\n(Ours)", True,  True,  True,  True,  2.11, 0.841),
]
COMPONENTS = ["Spatial CFG", "Sparse Attn", "Sparse FFN", "Step Skip"]

n         = len(CONFIGS)
labels    = [c[0] for c in CONFIGS]
comp_grid = [c[1:5] for c in CONFIGS]
speedups  = np.array([c[5] for c in CONFIGS])
irs       = np.array([c[6] for c in CONFIGS])

bar_colors = ["#BDBDBD", "#90A4AE", "#42A5F5", "#42A5F5", C_OURS]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5.4))
fig.subplots_adjust(left=0.09, right=0.91, top=0.90, bottom=0.38)

x     = np.arange(n)
bw    = 0.30
gap   = 0.04

# ── Bars ──────────────────────────────────────────────────────────────────────
ax2 = ax.twinx()

bars_sp = ax.bar(x - bw/2 - gap/2, speedups, width=bw,
                 color=[c + "DD" for c in bar_colors],
                 edgecolor="white", linewidth=0.8, zorder=3)

bars_ir = ax2.bar(x + bw/2 + gap/2, irs, width=bw,
                  color=bar_colors, edgecolor="white",
                  linewidth=0.8, alpha=0.88, zorder=3)

# ── Value labels ──────────────────────────────────────────────────────────────
for i, (sp, ir) in enumerate(zip(speedups, irs)):
    is_ours = (i == n - 1)
    fw, fs = ("bold", 9.0) if is_ours else ("normal", 8.5)
    ax.text(i - bw/2 - gap/2, sp + 0.03,
            f"{sp:.2f}×", ha="center", va="bottom",
            fontsize=fs, fontweight=fw, color="#333333", zorder=5)
    ax2.text(i + bw/2 + gap/2, ir + 0.003,
             f"{ir:.3f}", ha="center", va="bottom",
             fontsize=fs, fontweight=fw, color="#333333", zorder=5)

# baseline IR dashed reference
ax2.axhline(irs[0], color=C_GREY, ls="--", lw=1.0, zorder=1, alpha=0.8)
ax2.text(n - 0.38, irs[0] + 0.003, "baseline", fontsize=7.5,
         color=C_GREY, va="bottom", ha="right", style="italic")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(-0.55, n - 0.45)
ax.set_ylim(0, 2.85)
ax.set_xticks(x)
ax.set_xticklabels([])                  # hidden; shown in component table below
ax.set_ylabel("Speedup  (×)", fontsize=10.5, color=C_SPEED, labelpad=5)
ax.tick_params(axis="y", colors=C_SPEED, labelsize=9)
ax.spines["left"].set_color(C_SPEED);  ax.spines["left"].set_alpha(0.7)
ax.spines["bottom"].set_color("#CCCCCC")
ax.spines["top"].set_visible(False)
ax.grid(axis="y", ls="--", lw=0.4, color="#EBEBEB", zorder=0)
ax.set_axisbelow(True)

ax2.set_ylim(0.60, 0.96)
ax2.set_ylabel("ImageReward  ↑", fontsize=10.5, color=C_IR, labelpad=5)
ax2.tick_params(axis="y", colors=C_IR, labelsize=9)
ax2.spines["right"].set_color(C_IR);   ax2.spines["right"].set_alpha(0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_color("#CCCCCC")

# highlight "ours" column
ax.axvspan(n - 1.5, n - 0.5, color=C_LIGHT, alpha=0.45, zorder=0, lw=0)

# ── Legend ────────────────────────────────────────────────────────────────────
p1 = mpatches.Patch(facecolor=C_SPEED + "DD", edgecolor="white",
                    label="Speedup (×)")
p2 = mpatches.Patch(facecolor=C_IR,           edgecolor="white",
                    label="ImageReward ↑")
ax.legend(handles=[p1, p2], fontsize=9, loc="upper left",
          framealpha=0.9, edgecolor="#DDDDDD", borderpad=0.6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("AccelAes — Component Ablation  (Lumina-Next-T2I)",
             fontsize=11.5, fontweight="bold", pad=10)

# ── Component checkmark table (drawn in figure coordinates) ──────────────────
# We draw below the axes using figure-level text and patches.
n_comp = len(COMPONENTS)

# axes bounding box in figure fraction
ax_bb   = ax.get_position()         # Bbox in figure fraction [0,1]
ax_l    = ax_bb.x0
ax_r    = ax_bb.x1
ax_b    = ax_bb.y0
ax_w    = ax_bb.width

col_w   = ax_w / n                  # width of one bar-group in fig fraction
row_h   = 0.055                     # height of one component row in fig frac
table_t = ax_b - 0.015             # top of table (just below axis)
table_b = table_t - n_comp * row_h

# row label x position
label_x = ax_l - 0.01

# draw horizontal separator lines
for ri in range(n_comp + 1):
    y = table_t - ri * row_h
    line = plt.Line2D([ax_l, ax_r], [y, y],
                      transform=fig.transFigure,
                      color="#EEEEEE", lw=0.7, zorder=0)
    fig.add_artist(line)

# row labels
for ri, comp in enumerate(COMPONENTS):
    cy = table_t - (ri + 0.5) * row_h
    fig.text(label_x, cy, comp,
             ha="right", va="center", fontsize=8.8, color="#444444",
             transform=fig.transFigure)

# column config labels (x-tick replacement)
col_label_y = table_t + 0.005
for ci, lbl in enumerate(labels):
    cx = ax_l + (ci + 0.5) * col_w
    is_ours = (ci == n - 1)
    fig.text(cx, col_label_y, lbl,
             ha="center", va="bottom",
             fontsize=8.8 if not is_ours else 9.0,
             fontweight="bold" if is_ours else "normal",
             color=C_OURS if is_ours else "#333333",
             transform=fig.transFigure,
             multialignment="center")

# checkmarks
for ci, flags in enumerate(comp_grid):
    is_ours = (ci == n - 1)
    cx = ax_l + (ci + 0.5) * col_w
    for ri, flag in enumerate(flags):
        cy = table_t - (ri + 0.5) * row_h
        sym   = "●" if flag else "○"
        color = (C_OURS if is_ours else C_SPEED) if flag else "#CCCCCC"
        fig.text(cx, cy, sym,
                 ha="center", va="center",
                 fontsize=12, color=color,
                 transform=fig.transFigure)

# ours column highlight box in table
ours_x  = ax_l + (n - 1) * col_w
rect = mpatches.FancyBboxPatch(
    (ours_x + 0.005, table_b + 0.005),
    col_w - 0.01, table_t - table_b - 0.005,
    boxstyle="round,pad=0.003",
    linewidth=1.2, edgecolor=C_OURS,
    facecolor=C_LIGHT, alpha=0.4,
    transform=fig.transFigure, zorder=-1)
fig.add_artist(rect)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext, dpi in (("png", 200), ("pdf", 300)):
    path = f"{OUT_DIR}/ablation_components.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
