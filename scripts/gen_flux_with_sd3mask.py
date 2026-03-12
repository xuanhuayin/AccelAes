#!/usr/bin/env python3
"""
Use the SD3 astronaut heatmap as a pre-computed foreground mask
and feed it into FLUX AccelAes sparse generation.
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

PROMPT  = ("Portrait of an astronaut with intricate spacesuit details, "
           "photorealistic, cinematic")
SEED    = 42
STEPS   = 28
OUT_DIR = "outputs/paper_figures/flux/astronaut"
PATCH_H = PATCH_W = 64   # AccelAes patch resolution for FLUX 1024px

# ── Build 64×64 binary mask from SD3 heatmap ──────────────────────────────
hm = Image.open("outputs/paper_figures/sd3/astronaut/heatmap.png").convert("RGB")
hm_np = np.array(hm).astype(float) / 255.0

# Jet warmth: red - blue
warmth = hm_np[:,:,0] - hm_np[:,:,2]

# Resize to patch resolution
w64 = np.array(
    Image.fromarray(((warmth + 1)/2 * 255).astype(np.uint8))
    .resize((PATCH_W, PATCH_H), Image.BICUBIC)
).astype(float) / 255.0 * 2 - 1

# Normalize + threshold: keep top warm regions as foreground
w64 = (w64 - w64.min()) / (w64.max() - w64.min() + 1e-8)
binary_mask = (w64 > 0.45).astype(np.float32)

print(f"Pre-computed mask fg%: {binary_mask.mean()*100:.1f}%  shape={binary_mask.shape}")
# Save preview
Image.fromarray((binary_mask * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)\
    .save(f"{OUT_DIR}/injected_mask_preview.png")
print("saved injected_mask_preview.png")

# ── Hook build_mask to return our pre-computed mask ───────────────────────
_mask_tensor = torch.from_numpy(binary_mask)  # (64, 64) float32

def install_hook():
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    _orig = SD3SemanticMaskBuilder.build_mask

    def _patched(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
        print(f"  [hook] returning pre-computed mask  fg={_mask_tensor.mean()*100:.1f}%")
        return _mask_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

    SD3SemanticMaskBuilder.build_mask = _patched
    return _orig

def remove_hook(orig):
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    SD3SemanticMaskBuilder.build_mask = orig

# ── Run FLUX with injected mask ───────────────────────────────────────────
from src.models.flux_wrapper import FluxDiTWrapper

print("Loading FLUX wrapper...")
wrapper = FluxDiTWrapper(dtype="bf16")

orig = install_hook()
print("Running FLUX with SD3 mask injected...")
img = wrapper.generate_accelerated_sparse(
    PROMPT, seed=SEED, steps=STEPS,
    mask_step=8, skip_ratio=0.5, n_segments=64,
    region_method="threshold",
    sparse_blocks=True,
    full_skip_interval=2, warmup_steps=5,
)
remove_hook(orig)

out_path = f"{OUT_DIR}/flux_with_sd3mask.png"
img.save(out_path)
print(f"saved {out_path}  size={img.size}")
print("Done.")
