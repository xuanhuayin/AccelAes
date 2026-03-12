#!/usr/bin/env python3
"""
Re-generate enhanced heatmap + mask_overlay for paper_figures/flux/astronaut/.
Tunable parameters at the top of this file.
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

PROMPT  = ("Portrait of an astronaut with intricate spacesuit details, "
           "photorealistic, cinematic")
SEED    = 42
STEPS   = 28
OUT_DIR = "outputs/paper_figures/flux/astronaut"
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ────────────────────────────────────────────────────────────────────────
# Heatmap contrast
HEAT_LO_PCT  = 30       # percentile to clip low end  (raise = more blue area)
HEAT_HI_PCT  = 85       # percentile to clip high end (lower = more red area)
HEAT_GAMMA   = 0.45     # <1 → push midtones toward extremes (more contrast)
HEAT_BLUR    = 3        # gaussian blur radius on heatmap (0=off, smooths pixels)

# Mask overlay
OV_FG_COLOR  = (255, 80,  0)    # foreground tint (orange)
OV_BG_COLOR  = (20,  90, 220)   # background tint (blue)
OV_ALPHA_FG  = 0.55             # foreground blend strength
OV_ALPHA_BG  = 0.35             # background blend strength
OV_BORDER_PX = 4                # highlight px around mask edge (0=off)
OV_BLUR_MASK = 8                # blur mask boundary for softer edge (px, 0=off)
# ────────────────────────────────────────────────────────────────────────

_cap = {}

def install_hook():
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    _orig = SD3SemanticMaskBuilder.build_mask

    def _patched(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
        if affinity_maps:
            stacked = torch.stack(affinity_maps, dim=0)
            B = stacked.shape[1]
            stacked = stacked[:, B // 2:, :, :, :]
            importance = stacked.mean(dim=[0, 1, 2])
            if clip_token_weights is not None:
                w = clip_token_weights.to(importance.device, dtype=importance.dtype)
                w = w / (w.sum() + 1e-8)
                importance = (importance * w.unsqueeze(0)).sum(dim=-1)
            else:
                importance = importance.mean(dim=-1)
            _cap["imp"] = importance.reshape(patch_h, patch_w).cpu().float().numpy()
        mask = _orig(self, affinity_maps, patch_h, patch_w, clip_token_weights)
        _cap["mask"] = mask.cpu().float().numpy()
        return mask

    SD3SemanticMaskBuilder.build_mask = _patched
    return _orig

def remove_hook(orig):
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    SD3SemanticMaskBuilder.build_mask = orig


def make_heatmap(imp_np, size):
    lo = np.percentile(imp_np, HEAT_LO_PCT)
    hi = np.percentile(imp_np, HEAT_HI_PCT)
    n  = np.clip((imp_np - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    n  = n ** HEAT_GAMMA                          # gamma stretch
    # Jet colormap
    r = np.clip(1.5 - abs(n * 4 - 3), 0, 1)
    g = np.clip(1.5 - abs(n * 4 - 2), 0, 1)
    b = np.clip(1.5 - abs(n * 4 - 1), 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    hm = Image.fromarray(rgb).resize(size, Image.NEAREST)
    if HEAT_BLUR > 0:
        hm = hm.filter(ImageFilter.GaussianBlur(HEAT_BLUR))
    return hm


def make_overlay(base_img, mask_np):
    img  = np.array(base_img).astype(float)
    h, w = img.shape[:2]

    # Optional: soften mask boundary
    raw_mask = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    if OV_BLUR_MASK > 0:
        raw_mask = raw_mask.filter(ImageFilter.GaussianBlur(OV_BLUR_MASK))
    m = np.array(raw_mask).astype(float) / 255.0   # (H, W) in [0,1]

    fg_c = np.array(OV_FG_COLOR, float)
    bg_c = np.array(OV_BG_COLOR, float)

    # Foreground: tint warm
    result = img.copy()
    result = result * (1 - m[:, :, None] * OV_ALPHA_FG) + fg_c * m[:, :, None] * OV_ALPHA_FG
    # Background: tint cool
    inv = 1.0 - m
    result = result * (1 - inv[:, :, None] * OV_ALPHA_BG) + bg_c * inv[:, :, None] * OV_ALPHA_BG

    # Optional: highlight boundary ring
    if OV_BORDER_PX > 0:
        dilated  = np.array(raw_mask.filter(ImageFilter.MaxFilter(OV_BORDER_PX * 2 + 1))).astype(float) / 255.0
        eroded   = np.array(raw_mask.filter(ImageFilter.MinFilter(OV_BORDER_PX * 2 + 1))).astype(float) / 255.0
        border   = np.clip(dilated - eroded, 0, 1)
        white    = np.array([255, 255, 255], float)
        result   = result * (1 - border[:, :, None] * 0.7) + white * border[:, :, None] * 0.7

    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


# ── Generate ─────────────────────────────────────────────────────────────────
from src.models.flux_wrapper import FluxDiTWrapper

print("Loading FLUX wrapper...")
wrapper = FluxDiTWrapper(dtype="bf16")

_cap.clear()
orig = install_hook()
print("Running AccelAes to capture mask...")
wrapper.generate_accelerated_sparse(
    PROMPT, seed=SEED, steps=STEPS,
    mask_step=8, skip_ratio=0.5, n_segments=64,
    region_method="threshold",
    sparse_blocks=True,
    full_skip_interval=2, warmup_steps=5,
)
remove_hook(orig)

if "imp" not in _cap:
    print("ERROR: mask not captured"); sys.exit(1)

print(f"  imp range: [{_cap['imp'].min():.4f}, {_cap['imp'].max():.4f}]  "
      f"shape={_cap['imp'].shape}")
print(f"  mask fg%: {_cap['mask'].mean()*100:.1f}%  shape={_cap['mask'].shape}")

base_img = Image.open(f"{OUT_DIR}/baseline.png").convert("RGB")
W, H = base_img.size

hm = make_heatmap(_cap["imp"], (W, H))
hm.save(f"{OUT_DIR}/heatmap.png")
print(f"  saved heatmap.png")

ov = make_overlay(base_img, _cap["mask"])
ov.save(f"{OUT_DIR}/mask_overlay.png")
print(f"  saved mask_overlay.png")

# Also save clean binary mask (B&W) for reference
raw_m = Image.fromarray((_cap["mask"] * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
raw_m.save(f"{OUT_DIR}/binary_mask.png")
print(f"  saved binary_mask.png")

# Save numpy arrays for further tuning without re-running model
np.save(f"{OUT_DIR}/imp.npy",  _cap["imp"])
np.save(f"{OUT_DIR}/mask.npy", _cap["mask"])
print(f"  saved imp.npy + mask.npy  (tune without re-running model)")
print("Done.")
