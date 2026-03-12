#!/usr/bin/env python3
"""
Phase 3 交付物: 质量曲线 vs skip_ratio
绘制 CLIP Score 和 Edge Density vs skip_ratio，对比 copy vs linear, with/without refresh。
数据来源: Quick Sweep (12 组) + baseline
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Quick Sweep 数据 (10 prompts × 3 seeds = 30 images per condition)
ratios = [0.125, 0.25, 0.5]

data = {
    "copy_norefresh":   {"clip": [0.2572, 0.2564, 0.2562], "edge": [0.7321, 0.7150, 0.6791]},
    "copy_refresh5":    {"clip": [0.2574, 0.2571, 0.2570], "edge": [0.7346, 0.7203, 0.6905]},
    "linear_norefresh": {"clip": [0.2582, 0.2580, 0.2572], "edge": [0.7481, 0.7469, 0.7455]},
    "linear_refresh5":  {"clip": [0.2579, 0.2582, 0.2574], "edge": [0.7483, 0.7473, 0.7458]},
}

baseline_clip = 0.2579
baseline_edge = 0.7488

# Style settings
styles = {
    "copy_norefresh":   {"color": "#e74c3c", "marker": "o", "ls": "--", "label": "copy, no refresh"},
    "copy_refresh5":    {"color": "#e74c3c", "marker": "s", "ls": "-",  "label": "copy, refresh=5"},
    "linear_norefresh": {"color": "#2980b9", "marker": "o", "ls": "--", "label": "linear, no refresh"},
    "linear_refresh5":  {"color": "#2980b9", "marker": "s", "ls": "-",  "label": "linear, refresh=5"},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- CLIP Score ---
ax = axes[0]
ax.axhline(y=baseline_clip, color="gray", ls=":", lw=1.5, label=f"baseline ({baseline_clip:.4f})")
for key, d in data.items():
    s = styles[key]
    ax.plot(ratios, d["clip"], marker=s["marker"], ls=s["ls"], color=s["color"],
            lw=2, markersize=8, label=s["label"])
ax.set_xlabel("skip_ratio (fraction of background skipped)", fontsize=12)
ax.set_ylabel("CLIP Score", fontsize=12)
ax.set_title("CLIP Score vs skip_ratio", fontsize=13, fontweight="bold")
ax.set_xticks(ratios)
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.55)

# --- Edge Density ---
ax = axes[1]
ax.axhline(y=baseline_edge, color="gray", ls=":", lw=1.5, label=f"baseline ({baseline_edge:.4f})")
for key, d in data.items():
    s = styles[key]
    ax.plot(ratios, d["edge"], marker=s["marker"], ls=s["ls"], color=s["color"],
            lw=2, markersize=8, label=s["label"])
ax.set_xlabel("skip_ratio (fraction of background skipped)", fontsize=12)
ax.set_ylabel("Edge Density", fontsize=12)
ax.set_title("Edge Density vs skip_ratio", fontsize=13, fontweight="bold")
ax.set_xticks(ratios)
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.55)

fig.suptitle("Phase 3: Skip-Update Quality vs Ratio (Quick Sweep, N=30/condition)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out_path = "outputs/phase3_quality_curves.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")

# Also generate a delta% version
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes2[0]
ax.axhline(y=0, color="gray", ls=":", lw=1.5, label="baseline")
for key, d in data.items():
    s = styles[key]
    deltas = [(v - baseline_clip) / baseline_clip * 100 for v in d["clip"]]
    ax.plot(ratios, deltas, marker=s["marker"], ls=s["ls"], color=s["color"],
            lw=2, markersize=8, label=s["label"])
ax.set_xlabel("skip_ratio", fontsize=12)
ax.set_ylabel("CLIP Score Δ%", fontsize=12)
ax.set_title("CLIP Score Change vs skip_ratio", fontsize=13, fontweight="bold")
ax.set_xticks(ratios)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.55)

ax = axes2[1]
ax.axhline(y=0, color="gray", ls=":", lw=1.5, label="baseline")
for key, d in data.items():
    s = styles[key]
    deltas = [(v - baseline_edge) / baseline_edge * 100 for v in d["edge"]]
    ax.plot(ratios, deltas, marker=s["marker"], ls=s["ls"], color=s["color"],
            lw=2, markersize=8, label=s["label"])
ax.set_xlabel("skip_ratio", fontsize=12)
ax.set_ylabel("Edge Density Δ%", fontsize=12)
ax.set_title("Edge Density Change vs skip_ratio", fontsize=13, fontweight="bold")
ax.set_xticks(ratios)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.55)

fig2.suptitle("Phase 3: Quality Degradation (%) vs Ratio",
              fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
out_path2 = "outputs/phase3_quality_delta.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path2}")
