#!/usr/bin/env python3
"""
Generate ECCV teaser figure + individual sub-figures.

Outputs (outputs/figures/):
  teaser.png / teaser.pdf
  subfig_a_pareto.png / .pdf
  subfig_b_comparison.png / .pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── global style (must come before any use of color constants) ──────────────
import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/home/runkai/.local/share/fonts/Carlito/Carlito-Regular.ttf")
_fm.fontManager.addfont("/home/runkai/.local/share/fonts/Carlito/Carlito-Bold.ttf")

plt.rcParams.update({
    "font.family":       "Carlito",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
})

C_OURS  = "#2E7D32"   # dark green  — AccelAes
C_GREY  = "#757575"   # mid grey    — secondary text
C_LIGHT = "#E8F5E9"   # light green — callout bg

# ── paths ────────────────────────────────────────────────────────────────────
BASE    = "/home/runkai/xuanhua/SASD"
OUT_DIR = f"{BASE}/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

P0_DIR = f"{BASE}/outputs/p0_ablation_direct"
TS_DIR = f"{BASE}/outputs/teaser_samples"

# ── pareto data (Lumina-Next-T2I, Table 1) ───────────────────────────────────
PARETO = [
    dict(name="Baseline",         speedup=1.00, ir=0.752, c="#9E9E9E", m="o",  sz=100),
    dict(name="Δ-DiT",            speedup=1.52, ir=0.485, c="#E53935", m="s",  sz=80),
    dict(name="FORA",             speedup=1.00, ir=0.517, c="#FB8C00", m="D",  sz=75),
    dict(name="RAS",              speedup=1.47, ir=0.788, c="#1976D2", m="^",  sz=85),
    dict(name="TeaCache",         speedup=1.49, ir=0.670, c="#7B1FA2", m="v",  sz=80),
    dict(name="TaylorSeer",       speedup=2.05, ir=0.604, c="#00838F", m="P",  sz=80),
    dict(name="SDiT",             speedup=1.69, ir=0.609, c="#AD1457", m="h",  sz=80),
    dict(name="AccelAes (Ours)",  speedup=2.11, ir=0.841, c=C_OURS,    m="*",  sz=300),
]

LABEL_OFF = {
    "Baseline":         (-0.04, -0.042),
    "Δ-DiT":            (+0.05, -0.018),
    "FORA":             (-0.33, -0.038),
    "RAS":              (-0.04, +0.022),
    "TeaCache":         (+0.05, -0.020),
    "TaylorSeer":       (+0.05, -0.020),
    "SDiT":             (+0.05, +0.010),
    "AccelAes (Ours)":  (+0.06, +0.009),
}

# ── all 5 methods for visual comparison ──────────────────────────────────────
# (key, img_dir, col_label, speedup_str, ir_str, color)
METHODS_VIS = [
    ("baseline",  f"{P0_DIR}/baseline/images",  "Baseline",         "1.00×", "IR 0.752", "#9E9E9E"),
    ("delta_dit", f"{TS_DIR}/delta_dit/images",  "Δ-DiT",            "1.52×", "IR 0.485", "#E53935"),
    ("fora",      f"{TS_DIR}/fora/images",        "FORA",             "1.00×", "IR 0.517", "#FB8C00"),
    ("ras",       f"{TS_DIR}/ras/images",          "RAS",              "1.47×", "IR 0.788", "#1976D2"),
    ("sasd",      f"{P0_DIR}/sasd_full/images",   "AccelAes (Ours)", "2.11×", "IR 0.841", C_OURS),
]

# 3 representative prompt samples
SAMPLES = [
    ("p0004_s0000", "Traditional library"),
    ("p0007_s0001", "Fantasy city\n(drawing)"),
    ("p0015_s0002", "Mascot\nhelicopter"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sq(path: str, size: int = 200) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2,
                    (w + m) // 2, (h + m) // 2))
    return np.array(img.resize((size, size), Image.LANCZOS))


def save_fig(fig, stem: str, dpi_png: int = 200):
    for ext, dpi in (("png", dpi_png), ("pdf", 300)):
        path = f"{OUT_DIR}/{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  {path}")


# ── sub-figure A: pareto scatter ─────────────────────────────────────────────

def draw_pareto(ax, fs: float = 9.5, ms: float = 1.0):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, which="both", ls=":", lw=0.5, color="#E8E8E8", zorder=0)

    # faint "better zone" — top-right corner
    ax.fill_betweenx([0.790, 0.910], 1.88, 2.36,
                     color=C_OURS, alpha=0.05, zorder=0, lw=0)

    # ── scatter points ─────────────────────────────────────────────
    for d in sorted(PARETO, key=lambda x: "Ours" in x["name"]):
        is_ours = "Ours" in d["name"]
        if is_ours:
            ax.scatter(d["speedup"], d["ir"],
                       c=d["c"], marker="*", s=460 * ms,
                       zorder=8, linewidths=1.0, edgecolors="white")
            ax.scatter(d["speedup"], d["ir"],
                       c="none", marker="o", s=640 * ms,
                       zorder=7, linewidths=1.5,
                       edgecolors=C_OURS + "55")
        else:
            ax.scatter(d["speedup"], d["ir"],
                       c=d["c"], marker=d["m"], s=d["sz"] * 1.1 * ms,
                       zorder=5, linewidths=0.7,
                       edgecolors=d["c"] + "88")

    # ── labels ─────────────────────────────────────────────────────
    for d in PARETO:
        dx, dy = LABEL_OFF[d["name"]]
        is_ours = "Ours" in d["name"]
        ax.text(d["speedup"] + dx, d["ir"] + dy, d["name"],
                fontsize=fs + (1.0 if is_ours else 0),
                color=C_OURS if is_ours else "#444444",
                fontweight="bold" if is_ours else "normal",
                va="center", zorder=9)

    # ── callout box ────────────────────────────────────────────────
    ax.annotate(
        "  +11.9% IR   \n  2.11× faster  ",
        xy=(2.11, 0.841), xytext=(0.82, 0.893),
        fontsize=fs - 0.5, color=C_OURS, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=1.3 * ms ** 0.4,
                        mutation_scale=10 * ms ** 0.4,
                        connectionstyle="arc3,rad=-0.22"),
        bbox=dict(boxstyle="round,pad=0.55",
                  fc="white", ec=C_OURS, lw=1.3 * ms ** 0.4, alpha=1.0),
        ha="left", va="center", zorder=10,
    )

    # ── "better" arrow ─────────────────────────────────────────────
    ax.annotate("", xy=(2.30, 0.900), xytext=(2.10, 0.778),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=1.6 * ms ** 0.4, mutation_scale=14 * ms ** 0.4),
                zorder=4)
    ax.text(2.315, 0.903, "better", fontsize=fs - 1.5,
            color="#888888", va="bottom", style="italic", fontweight="bold")

    # ── axes ───────────────────────────────────────────────────────
    ax.set_xlabel("Speedup  (×)", fontsize=fs + 1.5, labelpad=10,
                  color="#333333")
    ax.set_ylabel("ImageReward  ↑", fontsize=fs + 1.5, labelpad=10,
                  color="#333333")
    ax.set_title("Speed–Quality Trade-off",
                 fontsize=fs + 1.2, pad=14, fontweight="bold", color="#222222")

    ax.set_xlim(0.62, 2.46)
    ax.set_ylim(0.42, 0.920)
    ax.tick_params(labelsize=fs, length=4, color="#CCCCCC", pad=6)
    for sp in ax.spines.values():
        sp.set_color("#DDDDDD")
        sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── sub-figure B: 5-method × 3-prompt image grid ────────────────────────────

def draw_comparison(fig, outer_gs, fs: float = 7.5):
    n_rows = len(SAMPLES)
    n_cols = len(METHODS_VIS)

    # narrow label column + 5 image columns
    gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols + 1, subplot_spec=outer_gs,
        hspace=0.04, wspace=0.025,
        width_ratios=[0.22] + [1] * n_cols,
    )

    for col, (key, img_dir, label, spd, ir_str, color) in enumerate(METHODS_VIS):
        is_ours = (key == "sasd")
        for row, (fname, caption) in enumerate(SAMPLES):
            path = f"{img_dir}/{fname}.png"

            # row label (leftmost column, row 0 only renders caption once)
            if col == 0:
                ax_c = fig.add_subplot(gs[row, 0])
                ax_c.axis("off")
                ax_c.text(0.95, 0.50, caption,
                          transform=ax_c.transAxes,
                          fontsize=fs - 0.5, ha="right", va="center",
                          style="italic", color="#444444", linespacing=1.3)

            ax = fig.add_subplot(gs[row, col + 1])

            if os.path.exists(path):
                ax.imshow(load_sq(path), interpolation="lanczos")
            else:
                ax.set_facecolor("#EEEEEE")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=fs, color="#999999",
                        transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

            # border: thick green for ours, thin grey otherwise
            lw = 2.2 if is_ours else 0.5
            ec = C_OURS if is_ours else "#CCCCCC"
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(ec)
                sp.set_linewidth(lw)

            # column header (row 0 only)
            if row == 0:
                weight = "bold" if is_ours else "normal"
                ax.set_title(f"{label}\n{spd}  {ir_str}",
                             fontsize=fs, color=color,
                             fontweight=weight, pad=3)


# ── combined teaser ───────────────────────────────────────────────────────────

def make_teaser():
    fig = plt.figure(figsize=(16.0, 4.8))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.72, 1.28],
        wspace=0.10,
        left=0.04, right=0.99,
        top=0.90, bottom=0.09,
    )

    ax_p = fig.add_subplot(gs[0])
    draw_pareto(ax_p, fs=9.5)
    fig.text(0.042, 0.965, "(a)", fontsize=12, fontweight="bold", va="top")

    draw_comparison(fig, gs[1], fs=7.8)
    fig.text(0.445, 0.965, "(b)  Visual Comparison", fontsize=10.5,
             fontweight="bold", va="top")

    print("Saving teaser...")
    save_fig(fig, "teaser", dpi_png=200)
    plt.close(fig)


# ── individual sub-figures ────────────────────────────────────────────────────

def make_subfig_a():
    fig, ax = plt.subplots(figsize=(11, 9))
    draw_pareto(ax, fs=23, ms=3.0)
    fig.tight_layout(pad=2.0)
    print("Saving subfig_a_pareto...")
    save_fig(fig, "subfig_a_pareto", dpi_png=200)
    plt.close(fig)


def make_subfig_b():
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = gridspec.GridSpec(1, 1, figure=fig,
                            left=0.08, right=0.99,
                            top=0.88, bottom=0.02)
    draw_comparison(fig, gs[0, 0], fs=8.5)
    fig.text(0.50, 0.97, "Baseline  ·  Δ-DiT  ·  FORA  ·  RAS  ·  AccelAes (Ours)",
             fontsize=11, ha="center", fontweight="bold", va="top")
    print("Saving subfig_b_comparison...")
    save_fig(fig, "subfig_b_comparison", dpi_png=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}\n")
    make_teaser()
    make_subfig_a()
    make_subfig_b()
    print("\nAll done.")
