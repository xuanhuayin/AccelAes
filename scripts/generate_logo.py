import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.collections import LineCollection

OUT_PNG = "/home/runkai/xuanhua/SASD/outputs/figures/logo_accelAes.png"
OUT_PDF = "/home/runkai/xuanhua/SASD/outputs/figures/logo_accelAes.pdf"

DARK_GREY  = "#333333"
DARK_GREEN = "#2E7D32"
MID_GREY   = "#888888"
LIGHT_GREY = "#BBBBBB"

W_IN, H_IN = 8.0, 2.8
fig, ax = plt.subplots(figsize=(W_IN, H_IN))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)   # fill figure — NO tight_layout
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── top-left star ─────────────────────────────────────────────────────────────
ax.text(0.042, 0.90, "★", transform=ax.transAxes,
        fontsize=20, color=DARK_GREEN, va="top", ha="left")

# ── main title ────────────────────────────────────────────────────────────────
# Strategy: render "Accel" first, measure its INK bbox BEFORE any layout change,
# then place "Aes" exactly at the ink right-edge (no tight_layout is called).

title_y  = 0.70
title_x0 = 0.09

t_accel = ax.text(title_x0, title_y, "Accel",
                  transform=ax.transAxes,
                  fontsize=60, fontweight="bold", color=DARK_GREY,
                  va="center", ha="left", fontfamily="DejaVu Sans")

# Measure now — axes position is fixed (no tight_layout called yet)
fig.canvas.draw()
renderer  = fig.canvas.get_renderer()
bbox      = t_accel.get_window_extent(renderer=renderer)

# Convert display px → axes fraction
inv = ax.transAxes.inverted()
x_end = inv.transform((bbox.x1, 0))[0]

ax.text(x_end, title_y, "Aes",
        transform=ax.transAxes,
        fontsize=60, fontweight="bold", color=DARK_GREEN,
        va="center", ha="left", fontfamily="DejaVu Sans")

# ── gradient divider ──────────────────────────────────────────────────────────
div_y = 0.32
xs = np.linspace(0.042, 0.958, 512)
pts = np.array([xs, np.full_like(xs, div_y)]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
t = np.linspace(0, 1, len(segs))
grey  = np.array([0x33, 0x33, 0x33]) / 255.
green = np.array([0x2E, 0x7D, 0x32]) / 255.
cols  = np.hstack([np.outer(1-t, grey) + np.outer(t, green),
                   np.ones((len(segs), 1))])
ax.add_collection(LineCollection(segs, colors=cols, linewidth=1.6,
                                  transform=ax.transAxes, clip_on=False))

# ── tagline ───────────────────────────────────────────────────────────────────
ax.text(0.09, 0.17,
        "Aesthetic-Aware Acceleration for Diffusion Transformers",
        transform=ax.transAxes, fontsize=13, color=MID_GREY,
        style="italic", va="center", ha="left", fontfamily="DejaVu Sans")

# ── ECCV 2026 ─────────────────────────────────────────────────────────────────
ax.text(0.958, 0.05, "ECCV 2026", transform=ax.transAxes,
        fontsize=9, color=LIGHT_GREY, va="bottom", ha="right")

# ── save (bbox_inches='tight' crops whitespace; no tight_layout called) ───────
for path, kw in [(OUT_PNG, dict(dpi=150)), (OUT_PDF, dict())]:
    fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
