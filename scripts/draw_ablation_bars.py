#!/usr/bin/env python3
"""
AccelAes ablation bar chart with component checkmark table.
5 configs clearly showing contributions of AesMask, SkipSparse, StepCache.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

C_OURS   = "#2E7D32"
C_SPEED  = "#1565C0"
C_IR     = "#388E3C"
C_LIGHT  = "#E8F5E9"
C_GREY   = "#9E9E9E"

# ── Data ──────────────────────────────────────────────────────────────────────
# (label, AesMask, SkipSparse, StepCache, speedup, IR)
CONFIGS = [
    ("Baseline",             False, False, False, 1.00, 0.752),
    ("StepCache\nOnly",      False, False, True,  1.57, 0.743),
    ("Spatial\nOnly",        True,  True,  False, 1.43, 0.818),
    ("w/o\nAesMask",         False, True,  True,  2.11, 0.822),
    ("AccelAes\n(Ours)",     True,  True,  True,  2.11, 0.841),
]
COMPONENTS = ["AesMask", "SkipSparse", "StepCache"]

n          = len(CONFIGS)
labels     = [c[0] for c in CONFIGS]
comp_grid  = [c[1:4] for c in CONFIGS]
speedups   = np.array([c[4] for c in CONFIGS])
irs        = np.array([c[5] for c in CONFIGS])

bar_colors = ["#BDBDBD", "#90A4AE", "#42A5F5", "#42A5F5", C_OURS]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.8))
fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.40)

x   = np.arange(n)
bw  = 0.30
gap = 0.04

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
    fw = "bold" if is_ours else "normal"
    fs_sp = 10.5 if is_ours else 9.5
    fs_ir = 10.5 if is_ours else 9.5
    ax.text(i - bw/2 - gap/2,  sp + 0.04,
            f"{sp:.2f}×", ha="center", va="bottom",
            fontsize=fs_sp, fontweight=fw, color="#333333", zorder=5)
    ax2.text(i + bw/2 + gap/2, ir + 0.003,
             f"{ir:.3f}", ha="center", va="bottom",
             fontsize=fs_ir, fontweight=fw, color="#333333", zorder=5)

# baseline IR dashed reference
ax2.axhline(irs[0], color=C_GREY, ls="--", lw=1.0, zorder=1, alpha=0.7)
ax2.text(n - 0.38, irs[0] + 0.003, "baseline IR",
         fontsize=8, color=C_GREY, va="bottom", ha="right", style="italic")

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xlim(-0.55, n - 0.45)
ax.set_ylim(0, 2.75)
ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_ylabel("Speedup  (×)", fontsize=12, color=C_SPEED, labelpad=6)
ax.tick_params(axis="y", colors=C_SPEED, labelsize=10)
ax.spines["left"].set_color(C_SPEED);   ax.spines["left"].set_alpha(0.7)
ax.spines["bottom"].set_color("#CCCCCC")
ax.spines["top"].set_visible(False)
ax.grid(axis="y", ls="--", lw=0.4, color="#EBEBEB", zorder=0)
ax.set_axisbelow(True)

ax2.set_ylim(0.60, 0.96)
ax2.set_ylabel("ImageReward  ↑", fontsize=12, color=C_IR, labelpad=6)
ax2.tick_params(axis="y", colors=C_IR, labelsize=10)
ax2.spines["right"].set_color(C_IR);    ax2.spines["right"].set_alpha(0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_color("#CCCCCC")

# "Ours" column highlight
ax.axvspan(n - 1.5, n - 0.5, color=C_LIGHT, alpha=0.45, zorder=0, lw=0)

# ── Legend ────────────────────────────────────────────────────────────────────
p1 = mpatches.Patch(facecolor=C_SPEED + "DD", edgecolor="white", label="Speedup (×)")
p2 = mpatches.Patch(facecolor=C_IR,           edgecolor="white", label="ImageReward ↑")
ax.legend(handles=[p1, p2], fontsize=10.5, loc="upper left",
          framealpha=0.9, edgecolor="#DDDDDD", borderpad=0.6)

# ── Component checkmark table ─────────────────────────────────────────────────
n_comp  = len(COMPONENTS)
ax_bb   = ax.get_position()
ax_l, ax_r = ax_bb.x0, ax_bb.x1
ax_b    = ax_bb.y0
ax_w    = ax_bb.width

col_w   = ax_w / n
row_h   = 0.065
table_t = ax_b - 0.012
table_b = table_t - n_comp * row_h

# horizontal separator lines
for ri in range(n_comp + 1):
    y = table_t - ri * row_h
    fig.add_artist(plt.Line2D([ax_l, ax_r], [y, y],
                              transform=fig.transFigure,
                              color="#EEEEEE", lw=0.8, zorder=0))

# row labels (component names)
for ri, comp in enumerate(COMPONENTS):
    cy = table_t - (ri + 0.5) * row_h
    fig.text(ax_l - 0.01, cy, comp,
             ha="right", va="center", fontsize=10.5, color="#444444",
             transform=fig.transFigure)

# column config labels (x-tick replacement)
for ci, lbl in enumerate(labels):
    cx = ax_l + (ci + 0.5) * col_w
    is_ours = (ci == n - 1)
    fig.text(cx, table_t + 0.006, lbl,
             ha="center", va="bottom",
             fontsize=10.5 if not is_ours else 11.0,
             fontweight="bold" if is_ours else "normal",
             color=C_OURS if is_ours else "#333333",
             transform=fig.transFigure,
             multialignment="center")

# checkmarks / crosses
for ci, flags in enumerate(comp_grid):
    is_ours = (ci == n - 1)
    cx = ax_l + (ci + 0.5) * col_w
    for ri, flag in enumerate(flags):
        cy = table_t - (ri + 0.5) * row_h
        sym   = "●" if flag else "○"
        color = (C_OURS if is_ours else C_SPEED) if flag else "#CCCCCC"
        fig.text(cx, cy, sym, ha="center", va="center",
                 fontsize=13, color=color,
                 transform=fig.transFigure)

# "Ours" column highlight box in table
ours_x = ax_l + (n - 1) * col_w
rect = mpatches.FancyBboxPatch(
    (ours_x + 0.005, table_b + 0.005),
    col_w - 0.01, table_t - table_b - 0.005,
    boxstyle="round,pad=0.003",
    linewidth=1.2, edgecolor=C_OURS,
    facecolor=C_LIGHT, alpha=0.4,
    transform=fig.transFigure, zorder=-1)
fig.add_artist(rect)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext, dpi in [("pdf", 300), ("png", 200)]:
    path = f"{OUT_DIR}/ablation_bars.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Done.")
