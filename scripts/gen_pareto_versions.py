#!/usr/bin/env python3
"""
Three redesigned versions of the Pareto scatter plot.

  pareto_v1.png  — Pareto frontier + leader lines
  pareto_v2.png  — Clean academic / legend style
  pareto_v3.png  — Zoned background + prominent star
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "pdf.fonttype":     42,
})

C_OURS  = "#2E7D32"
C_LIGHT = "#E8F5E9"
OUT_DIR = "/home/runkai/xuanhua/SASD/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = [
    dict(name="Baseline",        x=1.00, y=0.752, c="#9E9E9E", m="o",  s=110),
    dict(name="Δ-DiT",           x=1.52, y=0.485, c="#E53935", m="s",  s=90),
    dict(name="FORA",            x=1.00, y=0.517, c="#FB8C00", m="D",  s=85),
    dict(name="RAS",             x=1.47, y=0.788, c="#1976D2", m="^",  s=100),
    dict(name="AccelAes (Ours)", x=2.11, y=0.841, c=C_OURS,   m="*",  s=420),
]

# Pareto frontier (non-dominated in speedup-IR space, by IR descending)
# Non-dominated: AccelAes, RAS, Baseline — but Baseline dominated by AccelAes & RAS on IR
# Strict Pareto: AccelAes (2.11, 0.841) dominates all others on both axes
# Frontier points (max IR for each speedup level):
FRONTIER = [(1.00, 0.752), (1.47, 0.788), (2.11, 0.841)]


def save(fig, stem):
    for ext, dpi in (("png", 200), ("pdf", 300)):
        p = f"{OUT_DIR}/{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# VERSION 1: Pareto frontier line + leader-line labels
# ─────────────────────────────────────────────────────────────────────────────
def make_v1():
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.11)

    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, ls=":", lw=0.5, color="#E8E8E8", zorder=0)

    # ── Pareto frontier shading ────────────────────────────────────────
    fx = [p[0] for p in FRONTIER]
    fy = [p[1] for p in FRONTIER]
    # extend left to axis boundary
    ax.fill_between([0.62] + fx + [2.44], [fy[0]] * 1 + fy + [fy[-1]],
                    [0.39] * (len(fx) + 2),
                    color=C_OURS, alpha=0.04, zorder=0, lw=0)
    # frontier line
    ax.plot(fx, fy, color=C_OURS, lw=1.4, ls="--", alpha=0.5,
            zorder=3, label="Pareto frontier")

    # ── Scatter points ─────────────────────────────────────────────────
    LABEL_OFFSETS = {
        "Baseline":        (-0.22,  0.022),
        "Δ-DiT":           (+0.05, -0.020),
        "FORA":            (-0.22, -0.030),
        "RAS":             (+0.05,  0.012),
        "AccelAes (Ours)": (+0.06,  0.012),
    }
    for d in sorted(METHODS, key=lambda x: "Ours" in x["name"]):
        is_ours = "Ours" in d["name"]
        kw = dict(c=d["c"], linewidths=0.8, edgecolors="white")
        if is_ours:
            ax.scatter(d["x"], d["y"], marker="*", s=480, zorder=8, **kw)
        else:
            ax.scatter(d["x"], d["y"], marker=d["m"], s=d["s"], zorder=5, **kw)

        # label with leader line for crowded points
        ox, oy = LABEL_OFFSETS[d["name"]]
        lx, ly = d["x"] + ox, d["y"] + oy
        # leader line only when offset is large
        if abs(ox) > 0.06 or abs(oy) > 0.02:
            ax.plot([d["x"], lx + (0.02 if ox < 0 else -0.02)],
                    [d["y"], ly], color=d["c"], lw=0.6, alpha=0.6, zorder=4)
        ax.text(lx, ly, d["name"],
                fontsize=9.2 if is_ours else 8.8,
                color=C_OURS if is_ours else "#444444",
                fontweight="bold" if is_ours else "normal",
                va="center", ha="left" if ox > 0 else "right", zorder=9)

    # ── Compact metric badge near AccelAes ────────────────────────────
    ax.annotate("  +11.9% IR  \n  2.11× faster  ",
                xy=(2.11, 0.841), xytext=(1.58, 0.875),
                fontsize=8.5, color=C_OURS, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=1.1,
                                mutation_scale=9,
                                connectionstyle="arc3,rad=0.07"),
                bbox=dict(boxstyle="round,pad=0.38", fc="white",
                          ec=C_OURS, lw=1.1, alpha=1.0),
                ha="left", va="center", zorder=10)

    # ── "better" indicator ────────────────────────────────────────────
    ax.annotate("", xy=(2.35, 0.892), xytext=(2.18, 0.775),
                arrowprops=dict(arrowstyle="-|>", color="#BBBBBB",
                                lw=0.8, mutation_scale=7), zorder=4)
    ax.text(2.37, 0.893, "better", fontsize=6.8, color="#AAAAAA",
            va="bottom", style="italic")

    ax.set_xlim(0.62, 2.44)
    ax.set_ylim(0.39, 0.935)
    ax.set_xlabel("Speedup  (×)", fontsize=11, labelpad=6, color="#333333")
    ax.set_ylabel("ImageReward  ↑", fontsize=11, labelpad=6, color="#333333")
    ax.set_title("Speed–Quality Trade-off  (Lumina-Next-T2I)",
                 fontsize=11, pad=9, fontweight="bold", color="#222222")
    ax.tick_params(labelsize=9, length=3, color="#CCCCCC")
    for sp in ax.spines.values():
        sp.set_color("#DDDDDD"); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "pareto_v1")


# ─────────────────────────────────────────────────────────────────────────────
# VERSION 2: Clean academic — uniform circles, legend box, no annotation clutter
# ─────────────────────────────────────────────────────────────────────────────
def make_v2():
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.11)

    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, ls=":", lw=0.5, color="#EBEBEB", zorder=0)

    # uniform circle size ∝ sqrt(IR) for readability
    BASE_S = 160
    for d in sorted(METHODS, key=lambda x: "Ours" in x["name"]):
        is_ours = "Ours" in d["name"]
        sz = BASE_S * 2.2 if is_ours else BASE_S
        ax.scatter(d["x"], d["y"], s=sz,
                   c=d["c"], marker="o",
                   zorder=8 if is_ours else 5,
                   edgecolors="white", linewidths=1.2 if is_ours else 0.8,
                   label=d["name"])

    # accentuate ours with outer ring
    ours = METHODS[-1]
    ax.scatter(ours["x"], ours["y"], s=BASE_S * 2.2 * 2.2,
               c="none", marker="o", zorder=7,
               edgecolors=C_OURS + "44", linewidths=2.0)

    # Pareto frontier step line
    fx, fy = zip(*FRONTIER)
    ax.step(list(fx) + [2.44], list(fy) + [fy[-1]],
            color=C_OURS, lw=1.2, ls="--", alpha=0.35,
            where="post", zorder=3)

    # metric annotation — small text above the Ours dot
    ax.text(ours["x"] + 0.04, ours["y"] + 0.025,
            "+11.9% IR\n2.11× faster",
            fontsize=8.5, color=C_OURS, fontweight="bold",
            va="bottom", ha="left", zorder=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=C_OURS + "88", lw=0.9, alpha=0.95))

    # legend — upper left
    ax.legend(fontsize=8.5, loc="upper left",
              framealpha=0.95, edgecolor="#DDDDDD",
              borderpad=0.7, handletextpad=0.5,
              markerscale=0.9)

    # "better" shading zone
    ax.fill_betweenx([0.81, 0.935], 1.92, 2.44,
                     color=C_OURS, alpha=0.05, zorder=0, lw=0)
    ax.text(2.35, 0.930, "better ↗", fontsize=7, color=C_OURS + "99",
            ha="right", va="top", style="italic")

    ax.set_xlim(0.62, 2.44)
    ax.set_ylim(0.39, 0.935)
    ax.set_xlabel("Speedup  (×)", fontsize=11, labelpad=6, color="#333333")
    ax.set_ylabel("ImageReward  ↑", fontsize=11, labelpad=6, color="#333333")
    ax.set_title("Speed–Quality Trade-off  (Lumina-Next-T2I)",
                 fontsize=11, pad=9, fontweight="bold", color="#222222")
    ax.tick_params(labelsize=9, length=3, color="#CCCCCC")
    for sp in ax.spines.values():
        sp.set_color("#DDDDDD"); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "pareto_v2")


# ─────────────────────────────────────────────────────────────────────────────
# VERSION 3: Zone-based background + large star focal point
# ─────────────────────────────────────────────────────────────────────────────
def make_v3():
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.11)

    # Zone backgrounds
    # "good" zone: top-right quadrant from (1.4, 0.77)
    ax.fill_betweenx([0.77, 0.935], 1.38, 2.44,
                     color=C_OURS, alpha=0.06, zorder=0, lw=0)
    # "trade-off" zone: middle speedup, middle IR (RAS territory)
    ax.fill_betweenx([0.60, 0.77], 1.38, 2.44,
                     color="#1976D2", alpha=0.03, zorder=0, lw=0)

    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, ls=":", lw=0.4, color="#EBEBEB", zorder=1)

    # Zone labels (faint)
    ax.text(2.40, 0.930, "Ideal", fontsize=7.5, color=C_OURS + "88",
            ha="right", va="top", style="italic", fontweight="bold")

    # Pareto frontier
    fx, fy = zip(*FRONTIER)
    ax.plot(list(fx) + [2.44], list(fy) + [fy[-1]],
            color=C_OURS, lw=1.6, ls=(0, (4, 3)), alpha=0.4,
            zorder=3)

    # ── Scatter points ─────────────────────────────────────────────────
    LABEL_POS = {
        "Baseline":        dict(dx=-0.04, dy=-0.045, ha="right"),
        "Δ-DiT":           dict(dx=+0.05, dy=-0.022, ha="left"),
        "FORA":            dict(dx=-0.28, dy=-0.032, ha="right"),
        "RAS":             dict(dx=+0.05, dy=+0.010, ha="left"),
        "AccelAes (Ours)": dict(dx=+0.06, dy=+0.008, ha="left"),
    }
    for d in sorted(METHODS, key=lambda x: "Ours" in x["name"]):
        is_ours = "Ours" in d["name"]
        if is_ours:
            # large star with subtle shadow
            ax.scatter(d["x"], d["y"], c=d["c"], marker="*", s=520,
                       zorder=9, linewidths=1.0, edgecolors="white")
            # pulse ring
            for r, a in [(700, 0.18), (950, 0.08)]:
                ax.scatter(d["x"], d["y"], c="none", marker="o", s=r,
                           zorder=7, linewidths=1.2,
                           edgecolors=C_OURS, alpha=a)
        else:
            ax.scatter(d["x"], d["y"], c=d["c"], marker=d["m"],
                       s=d["s"], zorder=5,
                       linewidths=0.6, edgecolors=d["c"] + "99")

        p = LABEL_POS[d["name"]]
        ax.text(d["x"] + p["dx"], d["y"] + p["dy"], d["name"],
                fontsize=9.2 if is_ours else 8.8,
                color=C_OURS if is_ours else "#444444",
                fontweight="bold" if is_ours else "normal",
                ha=p["ha"], va="center", zorder=10)

    # ── Metric callout — placed bottom-right to avoid overlap ─────────
    ax.text(2.42, 0.430,
            "+11.9% IR\n2.11× faster",
            fontsize=9, color=C_OURS, fontweight="bold",
            ha="right", va="bottom", zorder=10,
            bbox=dict(boxstyle="round,pad=0.42", fc=C_LIGHT,
                      ec=C_OURS, lw=1.2, alpha=0.95))
    # arrow from badge to point
    ax.annotate("", xy=(2.11, 0.841), xytext=(2.28, 0.494),
                arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=1.0,
                                mutation_scale=8,
                                connectionstyle="arc3,rad=-0.15"),
                zorder=8)

    ax.set_xlim(0.62, 2.44)
    ax.set_ylim(0.39, 0.935)
    ax.set_xlabel("Speedup  (×)", fontsize=11, labelpad=6, color="#333333")
    ax.set_ylabel("ImageReward  ↑", fontsize=11, labelpad=6, color="#333333")
    ax.set_title("Speed–Quality Trade-off  (Lumina-Next-T2I)",
                 fontsize=11, pad=9, fontweight="bold", color="#222222")
    ax.tick_params(labelsize=9, length=3, color="#CCCCCC")
    for sp in ax.spines.values():
        sp.set_color("#DDDDDD"); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "pareto_v3")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating pareto versions...")
    make_v1()
    make_v2()
    make_v3()
    print("Done.")
