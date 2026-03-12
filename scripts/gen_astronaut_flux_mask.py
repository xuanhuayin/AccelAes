#!/usr/bin/env python3
"""
Re-generate paper_figures/flux/astronaut/ with heatmap + mask_overlay.
Uses the SHORT prompt matching gen_paper_figures.py.
"""
import os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

PROMPT  = ("Portrait of an astronaut with intricate spacesuit details, "
           "photorealistic, cinematic")
SEED    = 42
STEPS   = 28
OUT_DIR = "outputs/paper_figures/flux/astronaut"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Mask capture hook ────────────────────────────────────────────────────────
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

# ── Visualization ────────────────────────────────────────────────────────────
def jet_heatmap(imp_np, size):
    lo, hi = imp_np.min(), imp_np.max()
    n = (imp_np - lo) / (hi - lo + 1e-8)
    r = np.clip(1.5 - abs(n * 4 - 3), 0, 1)
    g = np.clip(1.5 - abs(n * 4 - 2), 0, 1)
    b = np.clip(1.5 - abs(n * 4 - 1), 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.NEAREST)

def mask_overlay(base_img, mask_np,
                 fg=(255, 80, 0), bg=(0, 100, 220), alpha=0.45):
    img = np.array(base_img).astype(float)
    h, w = img.shape[:2]
    m = np.array(
        Image.fromarray((mask_np * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    ).astype(float) / 255.0
    fg_c, bg_c = np.array(fg, float), np.array(bg, float)
    result = img * (1 - m[:, :, None] * alpha) + fg_c * m[:, :, None] * alpha
    result = result * (1 - (1 - m[:, :, None]) * alpha * 0.5) \
             + bg_c * (1 - m[:, :, None]) * alpha * 0.5
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))

# ── Main ─────────────────────────────────────────────────────────────────────
from src.models.flux_wrapper import FluxDiTWrapper

print("Loading FLUX wrapper...")
wrapper = FluxDiTWrapper(dtype="bf16")

# 1) Baseline
print("Generating baseline...")
img_base = wrapper.generate(PROMPT, seed=SEED, steps=STEPS)
img_base.save(f"{OUT_DIR}/baseline.png")
print(f"  saved baseline.png  {img_base.size}")

# 2) AccelAes with semantic mask
_cap.clear()
orig = install_hook()
print("Generating AccelAes (sparse)...")
img_sasd = wrapper.generate_accelerated_sparse(
    PROMPT, seed=SEED, steps=STEPS,
    mask_step=8, skip_ratio=0.5, n_segments=64,
    region_method="threshold",
    sparse_blocks=True,
    full_skip_interval=2, warmup_steps=5,
)
remove_hook(orig)
img_sasd.save(f"{OUT_DIR}/sasd.png")
print(f"  saved sasd.png  {img_sasd.size}")

# 3) Save heatmap + overlay
if "imp" in _cap:
    W, H = img_base.size
    hm = jet_heatmap(_cap["imp"], (W, H))
    hm.save(f"{OUT_DIR}/heatmap.png")
    print(f"  saved heatmap.png")

    ov = mask_overlay(img_base, _cap["mask"])
    ov.save(f"{OUT_DIR}/mask_overlay.png")
    print(f"  saved mask_overlay.png")

    # binary mask
    msk = Image.fromarray((_cap["mask"] * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    msk.save(f"{OUT_DIR}/binary_mask.png")
    print(f"  saved binary_mask.png")
else:
    print("WARNING: mask not captured")

print("Done.")
