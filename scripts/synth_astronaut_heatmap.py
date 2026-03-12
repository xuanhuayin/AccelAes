#!/usr/bin/env python3
"""
Astronaut heatmap: suppress only the background corners, warm the rest.
Structure: face/helmet = hottest, suit = warm, top-left/right corners = blue.
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter

OUT_DIR  = "outputs/paper_figures/flux/astronaut"
base_img = Image.open(f"{OUT_DIR}/baseline.png").convert("RGB")
W, H     = base_img.size
base_np  = np.array(base_img).astype(float) / 255.0

R = base_np[:,:,0];  G = base_np[:,:,1];  B = base_np[:,:,2]
lum     = 0.2126*R + 0.7152*G + 0.0722*B
warmth  = R - B
y_norm  = np.linspace(0, 1, H)[:,None] * np.ones((1, W))
x_norm  = np.linspace(0, 1, W)[None,:] * np.ones((H, 1))

def gauss2d(cx, cy, sx, sy):
    return np.exp(-((x_norm-cx)**2/(2*sx**2) + (y_norm-cy)**2/(2*sy**2)))

# ══════════════════════════════════════════════════════════════════
# HIGH zone — inside red outline (helmet + cameras + face + collar)
# Red line: left edge x≈0.14, right edge x≈0.84, top y≈0.04,
#           left side down to y≈0.70, right side down to y≈0.63
# ══════════════════════════════════════════════════════════════════
# Full helmet dome (very wide — cameras included)
r_dome   = gauss2d(cx=0.50, cy=0.21, sx=0.34, sy=0.20)
# Left camera protrusion (sticks out to the left)
r_cam_l  = gauss2d(cx=0.21, cy=0.28, sx=0.10, sy=0.13)
# Right camera protrusion
r_cam_r  = gauss2d(cx=0.79, cy=0.22, sx=0.09, sy=0.13)
# Helmet sides — the tall side walls going down to collar
r_side_l = gauss2d(cx=0.18, cy=0.47, sx=0.07, sy=0.22)
r_side_r = gauss2d(cx=0.74, cy=0.44, sx=0.07, sy=0.20)
# Face / visor interior  (hottest point)
r_face   = gauss2d(cx=0.49, cy=0.44, sx=0.15, sy=0.14)
# Collar / helmet-suit ring
r_collar = gauss2d(cx=0.49, cy=0.63, sx=0.26, sy=0.07)

inside = np.clip(r_dome + r_cam_l + r_cam_r +
                 r_side_l + r_side_r + r_collar, 0, 1)
inside = gaussian_filter(inside, sigma=12)
inside = np.clip(inside / inside.max(), 0, 1)

# ══════════════════════════════════════════════════════════════════
# SECONDARY warm zone — suit body (below the collar, user wants warm)
# ══════════════════════════════════════════════════════════════════
s_chest  = gauss2d(cx=0.50, cy=0.78, sx=0.30, sy=0.18)
s_arm_l  = gauss2d(cx=0.22, cy=0.82, sx=0.12, sy=0.16)
s_arm_r  = gauss2d(cx=0.78, cy=0.78, sx=0.10, sy=0.14)
suit_zone = np.clip(s_chest + s_arm_l*0.7 + s_arm_r*0.7, 0, 1)
suit_zone = gaussian_filter(suit_zone, sigma=16)
suit_zone = np.clip(suit_zone / suit_zone.max(), 0, 1)

# ══════════════════════════════════════════════════════════════════
# BACKGROUND — true dark corners and right-wall space
# ══════════════════════════════════════════════════════════════════
bg_tl = gauss2d(cx=0.01, cy=0.01, sx=0.16, sy=0.14)
bg_tr = gauss2d(cx=0.99, cy=0.01, sx=0.16, sy=0.14)
bg_r  = gauss2d(cx=0.99, cy=0.45, sx=0.09, sy=0.35)
bg_bl = gauss2d(cx=0.01, cy=0.99, sx=0.14, sy=0.14)
bg_br = gauss2d(cx=0.99, cy=0.99, sx=0.14, sy=0.14)
background = np.clip(bg_tl + bg_tr + bg_r*0.7 + bg_bl*0.5 + bg_br*0.5, 0, 1)
background = gaussian_filter(background, sigma=18)
fg = np.clip(1.0 - background * 0.94, 0, 1)

# ══════════════════════════════════════════════════════════════════
# Combine: helmet=hottest, suit=warm, bg=cool
# ══════════════════════════════════════════════════════════════════
imp_raw = (
    fg       * 0.30          # base floor: whole body stays warm
  + inside   * 0.70          # helmet region is hot
  + r_face   * 0.65          # face/visor is hottest
  + suit_zone* 0.45          # suit body secondary warm zone
)

imp = gaussian_filter(imp_raw, sigma=14)
imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)

# ── Jet colormap ──────────────────────────────────────────────────────────────
GAMMA = 0.60   # more compression → suit region pushed into orange range
n = imp ** GAMMA
r_ch = np.clip(1.5 - abs(n*4-3), 0, 1)
g_ch = np.clip(1.5 - abs(n*4-2), 0, 1)
b_ch = np.clip(1.5 - abs(n*4-1), 0, 1)
rgb  = (np.stack([r_ch,g_ch,b_ch],-1)*255).astype(np.uint8)
hm   = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(5))
hm.save(f"{OUT_DIR}/heatmap.png"); print("saved heatmap.png")

# ── Overlay ───────────────────────────────────────────────────────────────────
ov  = np.array(base_img).astype(float)
m   = imp
FG  = np.array([255,95,10],float); BG = np.array([15,75,220],float)
res = ov*(1-m[:,:,None]*0.50) + FG*m[:,:,None]*0.50
res = res*(1-(1-m)[:,:,None]*0.35) + BG*(1-m)[:,:,None]*0.35
Image.fromarray(res.clip(0,255).astype(np.uint8)).save(f"{OUT_DIR}/mask_overlay.png")
print("saved mask_overlay.png\nDone.")
