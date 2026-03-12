#!/usr/bin/env python3
"""
Tune heatmap + mask_overlay from saved npy arrays — no model re-run needed.
Edit the TUNABLE block and re-run: python scripts/tune_astronaut_viz.py
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter

OUT_DIR  = "outputs/paper_figures/flux/astronaut"
imp_raw  = np.load(f"{OUT_DIR}/imp.npy")   # (64, 64) float32
mask_raw = np.load(f"{OUT_DIR}/mask.npy")  # (64, 64) float32 binary {0,1}
base_img = Image.open(f"{OUT_DIR}/baseline.png").convert("RGB")
W, H     = base_img.size

print(f"imp  range: [{imp_raw.min():.4f}, {imp_raw.max():.4f}]")
print(f"mask fg%  : {mask_raw.mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — edit freely, then re-run
# ═══════════════════════════════════════════════════════════════════════

# ── Heatmap ──────────────────────────────────────────────────────────
HEAT_PRE_SMOOTH  = 1.2   # gaussian sigma on 64×64 before coloring (0=off)
HEAT_LO_PCT      = 5     # percentile clip (low → blue saturation point)
HEAT_HI_PCT      = 95    # percentile clip (high → red saturation point)
HEAT_GAMMA       = 0.65  # <1 softens contrast (closer to reference style)
HEAT_POST_BLUR   = 18    # gaussian blur AFTER upscaling to 1024px — key for smooth gradients

# ── Overlay (smooth continuous tint, no harsh binary border) ─────────
OV_FG_COLOR      = (255, 100,  10)  # warm orange — foreground (face/suit)
OV_BG_COLOR      = (20,   80, 220)  # cool blue   — background
OV_ALPHA_FG      = 0.50             # foreground tint strength
OV_ALPHA_BG      = 0.35             # background tint (subtle)
OV_MASK_SMOOTH   = 2.5              # sigma for smooth mask boundary (on 64×64)
OV_UPSCALE_BLUR  = 20               # post-upscale blur for smooth overlay

# ═══════════════════════════════════════════════════════════════════════

def make_heatmap(imp, size):
    # 1. Light pre-smooth at patch resolution
    imp_s = gaussian_filter(imp, sigma=HEAT_PRE_SMOOTH) if HEAT_PRE_SMOOTH > 0 else imp.copy()
    # 2. Percentile clip & normalize
    lo = np.percentile(imp_s, HEAT_LO_PCT)
    hi = np.percentile(imp_s, HEAT_HI_PCT)
    n  = np.clip((imp_s - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    n  = n ** HEAT_GAMMA
    # 3. Jet colormap
    r  = np.clip(1.5 - abs(n * 4 - 3), 0, 1)
    g  = np.clip(1.5 - abs(n * 4 - 2), 0, 1)
    b  = np.clip(1.5 - abs(n * 4 - 1), 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    # 4. Upscale with BICUBIC (smooth) then gaussian blur → SD3-like gradients
    hm = Image.fromarray(rgb).resize(size, Image.BICUBIC)
    if HEAT_POST_BLUR > 0:
        hm = hm.filter(ImageFilter.GaussianBlur(HEAT_POST_BLUR))
    return hm


def make_soft_mask(imp, size):
    """Smooth importance → soft probability map for overlay."""
    imp_s  = gaussian_filter(imp, sigma=OV_MASK_SMOOTH)
    # Normalize to [0,1] (not hard threshold — keeps gradient feel)
    lo = np.percentile(imp_s, 10)
    hi = np.percentile(imp_s, 90)
    soft = np.clip((imp_s - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    img = Image.fromarray((soft * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
    if OV_UPSCALE_BLUR > 0:
        img = img.filter(ImageFilter.GaussianBlur(OV_UPSCALE_BLUR))
    return img


def make_overlay(base_img, soft_mask_img):
    img = np.array(base_img).astype(float)
    m   = np.array(soft_mask_img).astype(float) / 255.0   # (H, W) soft [0,1]

    fg_c = np.array(OV_FG_COLOR, float)
    bg_c = np.array(OV_BG_COLOR, float)

    # Continuous blend: high-m → warm tint, low-m → cool tint
    result = img.copy()
    result = result * (1 - m[:, :, None] * OV_ALPHA_FG) + fg_c * m[:, :, None] * OV_ALPHA_FG
    inv    = 1.0 - m
    result = result * (1 - inv[:, :, None] * OV_ALPHA_BG) + bg_c * inv[:, :, None] * OV_ALPHA_BG

    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ── Generate ──────────────────────────────────────────────────────────────────
hm = make_heatmap(imp_raw, (W, H))
hm.save(f"{OUT_DIR}/heatmap.png")
print(f"saved heatmap.png")

soft_mask = make_soft_mask(imp_raw, (W, H))
soft_mask.save(f"{OUT_DIR}/binary_mask.png")
print(f"saved binary_mask.png")

ov = make_overlay(base_img, soft_mask)
ov.save(f"{OUT_DIR}/mask_overlay.png")
print(f"saved mask_overlay.png")

print("Done. Edit parameters at top of script and re-run to iterate.")
