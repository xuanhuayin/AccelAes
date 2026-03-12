#!/usr/bin/env python3
"""
Ablation study combined figure: table (left) + dual-axis bar chart (right).
Updated component names: AesMask, SkipSparse, StepCache.
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

C_OURS  = "#2E7D32"
C_SPEED = "#1565C0"
C_IR    = "#388E3C"
C_LIGHT = "#E8F5E9"

# ── Data ──────────────────────────────────────────────────────────────────────
# (label, AesMask, SkipSparse, StepCache, speedup, IR)
CONFIGS = [
    ("Baseline",           False, False, False, 1.00, 0.752),
    ("StepCache\nOnly",    False, False, True,  1.57, 0.743),
    ("Spatial\nOnly",      True,  True,  False, 1.43, 0.818),
    ("w/o\nAesMask",       False, True,  True,  2.11, 0.822),
    ("AccelAes\n(Ours)",   True,  True,  True,  2.11, 0.841),
]
COMPONENTS = ["AesMask", "SkipSparse", "StepCache"]

n          = len(CONFIGS)
labels     = [c[0] for c in CONFIGS]
comp_grid  = [c[1:4] for c in CONFIGS]
speedups   = [c[4] for c in CONFIGS]
irs        = [c[5] for c in CONFIGS]

BAR_COLORS = ["#BDBDBD", "#90A4AE", "#42A5F5", "#42A5F5", C_OURS]

# ── Figure layout ─────────────────────────────────────────────────────────────
# width_ratios: table narrower, bars wider
fig, (ax_tab, ax_bar) = plt.subplots(
    1, 2, figsize=(17.0, 6.2),
    gridspec_kw={"width_ratios": [1.0, 2.6]}
)
fig.subplots_adjust(left=0.03, right=0.96, top=0.92, bottom=0.14, wspace=0.10)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT: Component table (configs as rows, components as columns)
# ══════════════════════════════════════════════════════════════════════════════
n_comp = len(COMPONENTS)   # 3 columns

def ry(ri):  return n - 1 - ri   # row index → y coordinate (top-to-bottom)

ax_tab.set_xlim(-1.05, n_comp - 0.5)
ax_tab.set_ylim(-0.65, n - 0.22)
ax_tab.axis("off")

# Alternating row backgrounds
for ri in range(n):
    if ri % 2 == 0:
        y = ry(ri)
        ax_tab.axhspan(y - 0.5, y + 0.5, color="#F9F9F9",
                       xmin=0.0, xmax=1.0, zorder=0)

# Horizontal separators
for ri in np.arange(-0.5, n, 1.0):
    ax_tab.axhline(ri, color="#E8E8E8", lw=0.7)

# Column headers
col_labels = ["AesMask", "SkipSparse", "StepCache"]
for ci, comp in enumerate(col_labels):
    ax_tab.text(ci, n - 0.35, comp,
                ha="center", va="bottom",
                fontsize=13, color="#333333", fontweight="bold",
                linespacing=1.15)

# Row labels (config names)
for ri, lbl in enumerate(labels):
    is_ours = (ri == n - 1)
    ax_tab.text(-0.55, ry(ri), lbl,
                ha="right", va="center", multialignment="right",
                fontsize=13 if is_ours else 12,
                fontweight="bold" if is_ours else "normal",
                color=C_OURS if is_ours else "#333333",
                linespacing=1.1)

# Checkmarks / dashes
for ri, flags in enumerate(comp_grid):
    is_ours = (ri == n - 1)
    for ci, flag in enumerate(flags):
        sym   = "✓" if flag else "–"
        color = (C_OURS if is_ours else C_SPEED) if flag else "#CCCCCC"
        ax_tab.text(ci, ry(ri), sym, ha="center", va="center",
                    fontsize=18 if flag else 15, color=color,
                    fontweight="bold" if flag else "normal",
                    fontfamily="DejaVu Sans")

# Highlight AccelAes (Ours) row
ax_tab.add_patch(mpatches.FancyBboxPatch(
    (-0.5, ry(n - 1) - 0.48), n_comp, 0.96,
    boxstyle="round,pad=0.02",
    linewidth=1.3, edgecolor=C_OURS,
    facecolor=C_LIGHT, alpha=0.50, zorder=-1))

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: Dual-axis bar chart (Speedup left, IR right)
# ══════════════════════════════════════════════════════════════════════════════
x   = np.arange(n)
bw  = 0.30
gap = 0.05

ax2 = ax_bar.twinx()

# Speedup bars (left axis)
ax_bar.bar(x - bw/2 - gap/2, speedups, width=bw,
           color=[c + "DD" for c in BAR_COLORS],
           edgecolor="white", linewidth=0.8, zorder=3)
ax_bar.axhline(1.0, color="#BBBBBB", ls="--", lw=0.9, zorder=1)

for i, sp in enumerate(speedups):
    is_ours = (i == n - 1)
    ax_bar.text(i - bw/2 - gap/2, sp + 0.06, f"{sp:.2f}×",
                ha="center", va="bottom",
                fontsize=14.5 if is_ours else 13.5,
                fontweight="bold" if is_ours else "normal",
                color="#333333", zorder=5)

# IR bars (right axis)
ax2.bar(x + bw/2 + gap/2, irs, width=bw,
        color=BAR_COLORS, edgecolor="white",
        linewidth=0.8, alpha=0.88, zorder=3)
ax2.axhline(irs[0], color="#BBBBBB", ls="--", lw=0.9, zorder=1)

for i, ir in enumerate(irs):
    is_ours = (i == n - 1)
    ax2.text(i + bw/2 + gap/2, ir + 0.004, f"{ir:.3f}",
             ha="center", va="bottom",
             fontsize=14.5 if is_ours else 13.5,
             fontweight="bold" if is_ours else "normal",
             color="#333333", zorder=5)

# "Ours" highlight column
ax_bar.axvspan(n - 1.48, n - 0.52, color=C_LIGHT, alpha=0.45, zorder=0, lw=0)

# Axes formatting
ax_bar.set_xlim(-0.58, n - 0.42)
ax_bar.set_ylim(0, 2.80)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=15, multialignment="center")
ax_bar.set_ylabel("Speedup  (×)", fontsize=18, color=C_SPEED, labelpad=8)
ax_bar.tick_params(axis="y", colors=C_SPEED, labelsize=14, pad=4)
ax_bar.spines["left"].set_color(C_SPEED);  ax_bar.spines["left"].set_alpha(0.7)
ax_bar.spines["bottom"].set_color("#CCCCCC")
ax_bar.spines["top"].set_visible(False)
ax_bar.grid(axis="y", ls="--", lw=0.4, color="#EBEBEB", zorder=0)
ax_bar.set_axisbelow(True)

ax2.set_ylim(0.60, 0.98)
ax2.set_ylabel("ImageReward  ↑", fontsize=18, color=C_IR, labelpad=8)
ax2.tick_params(axis="y", colors=C_IR, labelsize=14, pad=4)
ax2.spines["right"].set_color(C_IR);      ax2.spines["right"].set_alpha(0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_color("#CCCCCC")

# Legend
p1 = mpatches.Patch(facecolor=C_SPEED + "DD", edgecolor="white", label="Speedup (×)")
p2 = mpatches.Patch(facecolor=C_IR,           edgecolor="white", label="ImageReward ↑")
ax_bar.legend(handles=[p1, p2], fontsize=14, loc="upper left",
              framealpha=0.92, edgecolor="#DDDDDD", borderpad=0.7)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext, dpi in [("pdf", 300), ("png", 200)]:
    path = f"{OUT_DIR}/ablation_combined.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
plt.close(fig)
print("Done.")
