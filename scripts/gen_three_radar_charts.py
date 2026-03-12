#!/usr/bin/env python3
"""
Generate 3 separate radar charts for Lumina, SD3, and FLUX.
Data taken directly from the paper tables (table1_2.png and table_3.png).
Style matches radar_all_in_one.pdf.
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

METRIC_LABELS = ["Speedup", "CLIP", "IR", "HPS", "Aesthetic", "Edge"]
N = len(METRIC_LABELS)

# ── Table 2: Lumina-Next-T2I ──────────────────────────────────────────────────
# Exact values from table1_2.png Table 2
# Columns: (name, color, Speedup, CLIP, IR, HPS, Aesth, Edge)
METHODS_LUMINA = [
    ("Baseline",         "#9E9E9E", 1.00, 0.2531, 0.752,   0.2710, 5.941, 0.583),
    ("Δ-DiT",            "#E53935", 1.52, 0.2523, 0.485,   0.2540, 5.873, 0.522),
    ("FORA",             "#FB8C00", 1.00, 0.2463, 0.517,   0.2496, 5.766, 0.564),
    ("RAS",              "#1976D2", 1.47, 0.2550, 0.788,   0.2713, 5.927, 0.581),
    ("SDiT",             "#FF7043", 1.69, 0.2502, 0.6093,  0.2538, 5.822, 0.4289),
    ("TeaCache",         "#7B1FA2", 1.49, 0.2512, 0.670,   0.2640, 5.978, 0.599),
    ("TaylorSeer",       "#00838F", 2.05, 0.2491, 0.604,   0.2650, 5.940, 0.668),
    ("AccelAes (Ours)",  "#2E7D32", 2.11, 0.2640, 0.841,   0.2740, 6.041, 0.629),
]

# ── Table 3: SD3-Medium ───────────────────────────────────────────────────────
# Exact values from table1_2.png Table 3
# AccelAes best on IR, HPS, Aesth; TaylorSeer best Speedup+Edge; TeaCache best CLIP
METHODS_SD3 = [
    ("SD3-Medium",       "#9E9E9E", 1.00, 0.2662, 0.879, 0.2895, 5.733, 0.916),
    ("TeaCache",         "#7B1FA2", 1.41, 0.2667, 0.847, 0.2863, 5.717, 0.881),
    ("TaylorSeer",       "#00838F", 2.02, 0.2624, 0.729, 0.2723, 5.561, 0.998),
    ("Δ-DiT",            "#E53935", 1.51, 0.2634, 0.828, 0.2804, 5.656, 0.850),
    ("FORA",             "#FB8C00", 0.99, 0.2663, 0.703, 0.2643, 5.522, 0.830),
    ("AccelAes (Ours)",  "#2E7D32", 1.50, 0.2644, 0.904, 0.3014, 5.990, 0.944),
]

# ── Table 4: FLUX.1-dev ───────────────────────────────────────────────────────
# Source: table_3.png (clearly readable)
# AccelAes: best on IR, Aesth, Edge; TeaCache best CLIP+Speedup; TaylorSeer best HPS
METHODS_FLUX = [
    ("FLUX.1-dev",       "#9E9E9E", 1.00, 0.2753, 1.233, 0.3200, 6.243, 0.560),
    ("TeaCache",         "#7B1FA2", 2.14, 0.2777, 1.225, 0.3154, 6.301, 0.529),
    ("TaylorSeer",       "#00838F", 2.12, 0.2765, 1.304, 0.3217, 6.309, 0.572),
    ("AccelAes (Ours)",  "#2E7D32", 1.73, 0.2772, 1.317, 0.3214, 6.372, 0.590),
]

METRIC_COLS = [2, 3, 4, 5, 6, 7]


def draw_radar(methods, title, out_name):
    vals  = np.array([[m[ci] for ci in METRIC_COLS] for m in methods])
    vmin  = vals.min(axis=0)
    vmax  = vals.max(axis=0)
    norms = np.clip((vals - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)

    angles   = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_c = np.append(angles, angles[0])

    fig = plt.figure(figsize=(24, 15))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor("white")

    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color("#AAAAAA")
    ax.spines["polar"].set_linewidth(2.5)

    RMAX = 1.18

    # Concentric grey rings
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles_c, np.full(N + 1, r), color="#CCCCCC", lw=1.5, zorder=1, clip_on=False)

    # Spokes
    for a in angles:
        ax.plot([0, a], [0, RMAX], color="#CCCCCC", lw=1.5, zorder=1)

    # Axis labels
    label_r_extra = {
        "Aesthetic": RMAX + 0.42,
        "CLIP":      RMAX + 0.32,
    }
    label_va_extra = {"CLIP": "top"}
    for i, lbl in enumerate(METRIC_LABELS):
        a  = angles[i]
        r  = label_r_extra.get(lbl, RMAX + 0.32)
        va = label_va_extra.get(lbl, "center")
        ax.text(a, r, lbl, ha="center", va=va,
                fontsize=60, color="#222222", fontweight="bold", zorder=7)

    # Data polygons
    for i, m in enumerate(methods):
        name, color = m[0], m[1]
        is_ours = "Ours" in name
        v     = np.append(norms[i], norms[i][0])
        lw    = 7.0 if is_ours else 4.0
        alpha = 0.18 if is_ours else 0.09
        ax.plot(angles_c, v, color=color, lw=lw, zorder=4 + int(is_ours), clip_on=False)
        ax.fill(angles_c, v, color=color, alpha=alpha, zorder=3)

    ax.set_ylim(0, RMAX)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    ax.set_xticks([])

    # Legend
    handles = [
        mlines.Line2D([], [], color=m[1],
                      linewidth=7.0 if "Ours" in m[0] else 4.0,
                      label=m[0], linestyle="-")
        for m in methods
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(2.05, 1.22),
        fontsize=56,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        handlelength=2.2,
        handletextpad=0.8,
        labelspacing=0.55,
    )

    fig.tight_layout(pad=2.0)

    for ext, dpi in [("png", 200), ("pdf", 300)]:
        path = f"{OUT_DIR}/{out_name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


print("Generating Lumina radar chart...")
draw_radar(METHODS_LUMINA, "Lumina-Next-T2I", "radar_lumina")

print("Generating SD3 radar chart...")
draw_radar(METHODS_SD3, "SD3-Medium", "radar_sd3")

print("Generating FLUX radar chart...")
draw_radar(METHODS_FLUX, "FLUX.1-dev", "radar_flux")

print("Done.")
