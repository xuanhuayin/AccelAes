#!/usr/bin/env python3
"""
Generate ablation study as TWO separate subfigures for manual assembly.

  ablation_table.png  — component checkmark table only
  ablation_bars.png   — dual-axis bar chart (Speedup + ImageReward)
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
    "font.family": "Carlito",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
})

C_OURS  = "#2E7D32"
C_SPEED = "#1565C0"
C_IR    = "#388E3C"
C_LIGHT = "#E8F5E9"

OUT_DIR = "/home/runkai/xuanhua/SASD/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Semantic Mask subsumes both Spatial CFG and Sparse Attn
# (they are always enabled/disabled together)
CONFIGS = [
    # label,               Sem.Mask, Sparse FFN, Step Skip, speedup,  IR
    ("Baseline",           False,    False,      False,     1.00,     0.752),
    ("Step Skip",          False,    False,      True,      1.57,     0.743),
    ("Sem. Mask",          True,     False,      False,     1.42,     0.818),
    ("Sparse FFN",         True,     True,       False,     1.43,     0.818),
    ("AccelAes\n(Ours)",   True,     True,       True,      2.11,     0.841),
]
# Row labels for the table
COMPONENTS = ["Semantic Mask", "Sparse FFN", "Step Skip"]

n         = len(CONFIGS)
labels    = [c[0] for c in CONFIGS]
comp_grid = [c[1:4] for c in CONFIGS]
speedups  = [c[4] for c in CONFIGS]
irs       = [c[5] for c in CONFIGS]

BAR_COLORS = ["#BDBDBD", "#90A4AE", "#42A5F5", "#42A5F5", C_OURS]


# ── helpers ───────────────────────────────────────────────────────────────────
def save(fig, name, dpi=200):
    for ext in ("png", "pdf"):
        p = f"{OUT_DIR}/{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# (1) COMPONENT TABLE  — transposed: configs as ROWS, components as COLS
# ═══════════════════════════════════════════════════════════════════════
def make_table():
    n_comp = len(COMPONENTS)   # 3 columns
    # Draw top-to-bottom: ri=0 (Baseline) → top row (y=n-1),
    #                     ri=n-1 (AccelAes) → bottom row (y=0)
    def ry(ri):  return n - 1 - ri

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    # Slightly wider left margin for row labels; ylim gives header room above y=n-1
    ax.set_xlim(-1.15, n_comp - 0.5)
    ax.set_ylim(-0.65, n - 0.22)
    ax.axis("off")

    # alternating row background
    for ri in range(n):
        if ri % 2 == 0:
            y = ry(ri)
            ax.axhspan(y - 0.5, y + 0.5, color="#F9F9F9",
                       xmin=0.0, xmax=1.0, zorder=0)

    # horizontal separators
    for ri in np.arange(-0.5, n, 1.0):
        ax.axhline(ri, color="#E8E8E8", lw=0.7)

    # column headers — wrapped to two lines so they don't overlap horizontally
    col_labels = ["Semantic\nMask", "Sparse\nFFN", "Step\nSkip"]
    for ci, comp in enumerate(col_labels):
        ax.text(ci, n - 0.38, comp,
                ha="center", va="bottom",
                fontsize=13, color="#333333", fontweight="bold",
                linespacing=1.15)

    # row labels (config names on the left)
    for ri, lbl in enumerate(labels):
        is_ours = (ri == n - 1)
        ax.text(-0.58, ry(ri), lbl,
                ha="right", va="center", multialignment="right",
                fontsize=13 if is_ours else 12,
                fontweight="bold" if is_ours else "normal",
                color=C_OURS if is_ours else "#333333",
                linespacing=1.1)

    # checkmarks / dashes
    for ri, flags in enumerate(comp_grid):
        is_ours = (ri == n - 1)
        for ci, flag in enumerate(flags):
            sym   = "✓" if flag else "–"
            color = (C_OURS if is_ours else C_SPEED) if flag else "#CCCCCC"
            ax.text(ci, ry(ri), sym, ha="center", va="center",
                    fontsize=18 if flag else 15, color=color,
                    fontweight="bold" if flag else "normal",
                    fontfamily="DejaVu Sans")

    # highlight AccelAes (Ours) row — now at the bottom (y=0)
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.5, ry(n - 1) - 0.48), n_comp, 0.96,
        boxstyle="round,pad=0.02",
        linewidth=1.3, edgecolor=C_OURS,
        facecolor=C_LIGHT, alpha=0.50, zorder=-1))

    fig.tight_layout(pad=0.4)
    save(fig, "ablation_table")


# ═══════════════════════════════════════════════════════════════════════
# (2) COMBINED DUAL-AXIS BAR CHART  (Speedup left, IR right)
# ═══════════════════════════════════════════════════════════════════════
def make_combined_bars():
    fig, ax = plt.subplots(figsize=(14.4, 8.5))
    fig.subplots_adjust(left=0.11, right=0.87, top=0.92, bottom=0.18)

    x   = np.arange(n)
    bw  = 0.30
    gap = 0.05

    ax2 = ax.twinx()

    # ── Speedup bars (left axis, blue) ────────────────────────────────
    ax.bar(x - bw/2 - gap/2, speedups, width=bw,
           color=[c + "DD" for c in BAR_COLORS],
           edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(1.0, color="#BBBBBB", ls="--", lw=0.9, zorder=1)

    for i, sp in enumerate(speedups):
        is_ours = (i == n - 1)
        ax.text(i - bw/2 - gap/2, sp + 0.06, f"{sp:.2f}×",
                ha="center", va="bottom",
                fontsize=22.0 if is_ours else 21.0,
                fontweight="bold" if is_ours else "normal",
                color="#333333", zorder=5)

    # ── IR bars (right axis, green) ───────────────────────────────────
    ax2.bar(x + bw/2 + gap/2, irs, width=bw,
            color=BAR_COLORS, edgecolor="white",
            linewidth=0.8, alpha=0.88, zorder=3)

    ax2.axhline(irs[0], color="#BBBBBB", ls="--", lw=0.9, zorder=1)

    for i, ir in enumerate(irs):
        is_ours = (i == n - 1)
        ax2.text(i + bw/2 + gap/2, ir + 0.004, f"{ir:.3f}",
                 ha="center", va="bottom",
                 fontsize=22.0 if is_ours else 21.0,
                 fontweight="bold" if is_ours else "normal",
                 color="#333333", zorder=5)

    # ── "Ours" highlight column ────────────────────────────────────────
    ax.axvspan(n - 1.48, n - 0.52, color=C_LIGHT, alpha=0.45, zorder=0, lw=0)

    # ── Axes formatting ───────────────────────────────────────────────
    ax.set_xlim(-0.58, n - 0.42)
    ax.set_ylim(0, 2.80)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=23, multialignment="center")
    ax.set_ylabel("Speedup  (×)", fontsize=27, color=C_SPEED, labelpad=10)
    ax.tick_params(axis="y", colors=C_SPEED, labelsize=22, pad=6)
    ax.spines["left"].set_color(C_SPEED)
    ax.spines["left"].set_alpha(0.7)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.4, color="#EBEBEB", zorder=0)
    ax.set_axisbelow(True)

    ax2.set_ylim(0.60, 0.98)
    ax2.set_ylabel("ImageReward  ↑", fontsize=27, color=C_IR, labelpad=10)
    ax2.tick_params(axis="y", colors=C_IR, labelsize=22, pad=6)
    ax2.spines["right"].set_color(C_IR)
    ax2.spines["right"].set_alpha(0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_color("#CCCCCC")

    # ── Legend ────────────────────────────────────────────────────────
    p1 = mpatches.Patch(facecolor=C_SPEED + "DD", edgecolor="white",
                        label="Speedup (×)")
    p2 = mpatches.Patch(facecolor=C_IR,           edgecolor="white",
                        label="ImageReward ↑")
    ax.legend(handles=[p1, p2], fontsize=22, loc="upper left",
              framealpha=0.92, edgecolor="#DDDDDD", borderpad=0.8)

    save(fig, "ablation_bars")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating ablation subfigures...")
    make_table()
    make_combined_bars()
    print("Done.")
