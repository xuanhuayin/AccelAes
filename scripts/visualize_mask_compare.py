#!/usr/bin/env python3
"""
对比 SLIC mask vs 去SLIC直接patch阈值 mask，并可视化。

输出：outputs/mask_compare/{tag}/
  - baseline.png
  - heatmap.png           (attention importance, jet colormap)
  - mask_slic.png         (当前SLIC方案)
  - mask_direct.png       (直接percentile阈值，无SLIC)
  - overlay_slic.png      (baseline + SLIC mask overlay)
  - overlay_direct.png    (baseline + direct mask overlay)
  - compare_5panel.png    (5-panel grid: base|heatmap|SLIC|direct|sasd)
  - importance.npy        (raw importance float32 array)
"""

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Config ────────────────────────────────────────────────────────────────────

PROMPTS = [
    ("lion",       "A majestic lion at golden hour in the savanna, photorealistic, "
                   "cinematic lighting, intricate fur detail"),
    ("astronaut",  "Portrait of an astronaut with intricate spacesuit details, "
                   "photorealistic, cinematic"),
    ("city",       "A futuristic Tokyo street at night with neon lights and intricate "
                   "reflections on wet pavement, cinematic"),
]

SEED       = 42
SD3_STEPS  = 28
RATIO      = 0.5
OUT_ROOT   = "outputs/mask_compare"


# ── Mask capture hook ─────────────────────────────────────────────────────────

_cap = {}

def install_hook():
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    _orig = SD3SemanticMaskBuilder.build_mask

    def _patched(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
        if affinity_maps:
            stacked = torch.stack(affinity_maps, dim=0)
            B = stacked.shape[1]
            stacked = stacked[:, B // 2:, :, :, :]
            importance = stacked.mean(dim=[0, 1, 2])   # (N_img, N_txt)
            if clip_token_weights is not None:
                w = clip_token_weights.to(importance.device, dtype=importance.dtype)
                w = w / (w.sum() + 1e-8)
                importance = (importance * w.unsqueeze(0)).sum(dim=-1)
            else:
                importance = importance.mean(dim=-1)
            _cap["imp"] = importance.reshape(patch_h, patch_w).cpu().float().numpy()
            _cap["ph"] = patch_h
            _cap["pw"] = patch_w

        mask_slic = _orig(self, affinity_maps, patch_h, patch_w, clip_token_weights)
        _cap["mask_slic"] = mask_slic.cpu().float().numpy()
        return mask_slic

    SD3SemanticMaskBuilder.build_mask = _patched
    return _orig

def remove_hook(orig):
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    SD3SemanticMaskBuilder.build_mask = orig


# ── Direct threshold mask (no SLIC) ──────────────────────────────────────────

def direct_threshold_mask(imp_np: np.ndarray, ratio: float = 0.5,
                           blur_sigma: float = 1.5) -> np.ndarray:
    """Percentile threshold directly on patch importance map."""
    thresh = np.percentile(imp_np, (1 - ratio) * 100)
    mask = (imp_np >= thresh).astype(np.float32)

    # Gaussian blur for soft edges (same as SLIC path)
    if blur_sigma > 0:
        t = torch.from_numpy(mask)
        from src.sparse.boundary_ops import gaussian_blur
        t = gaussian_blur(t, sigma=blur_sigma)
        mask = t.numpy()

    return mask


# ── Visualization helpers ─────────────────────────────────────────────────────

def jet(imp_np, size):
    lo, hi = imp_np.min(), imp_np.max()
    norm = (imp_np - lo) / (hi - lo + 1e-8)
    r = np.clip(1.5 - abs(norm * 4 - 3), 0, 1)
    g = np.clip(1.5 - abs(norm * 4 - 2), 0, 1)
    b = np.clip(1.5 - abs(norm * 4 - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.NEAREST)


def overlay(base_img: Image.Image, mask_np: np.ndarray,
            fg=(255, 80, 0), bg=(0, 100, 220), alpha=0.45) -> Image.Image:
    img = np.array(base_img).astype(float)
    h, w = img.shape[:2]
    m = np.array(Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
        (w, h), Image.NEAREST)).astype(float) / 255.0
    fg_c, bg_c = np.array(fg, float), np.array(bg, float)
    out = img.copy()
    out = out * (1 - m[:, :, None] * alpha) + fg_c * m[:, :, None] * alpha
    out = out * (1 - (1 - m[:, :, None]) * alpha * 0.5) + bg_c * (1 - m[:, :, None]) * alpha * 0.5 * 0
    # Only tint FG; leave BG untouched for cleaner look
    result = img.copy()
    result = result * (1 - m[:, :, None] * alpha) + fg_c * m[:, :, None] * alpha
    # BG: slight blue tint
    bm = (1 - m)[:, :, None]
    result = result * (1 - bm * alpha * 0.4) + bg_c * bm * alpha * 0.4
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def add_label(img, text, fsize=18):
    bar = fsize + 8
    out = Image.new("RGB", (img.width, img.height + bar), (30, 30, 30))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 4), text, fill=(240, 240, 240), font=font)
    return out


def hstack(imgs, pad=4):
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), (240, 240, 240))
    x = 0
    for img in imgs:
        canvas.paste(img, (x, (h - img.height) // 2))
        x += img.width + pad
    return canvas


def vstack(imgs, pad=4):
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), (240, 240, 240))
    y = 0
    for img in imgs:
        canvas.paste(img, ((w - img.width) // 2, y))
        y += img.height + pad
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import gc
    from src.models.sd3_wrapper import SD3DiTWrapper

    os.makedirs(OUT_ROOT, exist_ok=True)

    wrapper = SD3DiTWrapper(dtype="bf16")
    orig = install_hook()

    rows = []  # per-prompt row for big comparison grid

    for tag, prompt in PROMPTS:
        d = os.path.join(OUT_ROOT, tag)
        os.makedirs(d, exist_ok=True)
        print(f"\n[{tag}]")
        _cap.clear()

        # ── Generate ──────────────────────────────────────────────────────
        print("  baseline...")
        img_base = wrapper.generate(prompt, seed=SEED, steps=SD3_STEPS)
        img_base.save(os.path.join(d, "baseline.png"))

        print("  sasd (capturing mask)...")
        img_sasd = wrapper.generate_accelerated(
            prompt, seed=SEED, steps=SD3_STEPS,
            mask_type="semantic", s_fg=9.0, s_bg=2.0,
            full_skip_interval=2, n_segments=64,
            mask_step=5, skip_ratio=0.5,
        )
        img_sasd.save(os.path.join(d, "sasd.png"))

        if "imp" not in _cap:
            print("  WARNING: mask not captured, skipping")
            continue

        imp     = _cap["imp"]          # (patch_h, patch_w) float32
        m_slic  = _cap["mask_slic"]    # (patch_h, patch_w) float32, already blurred

        # ── Direct threshold mask ──────────────────────────────────────────
        m_direct = direct_threshold_mask(imp, ratio=RATIO, blur_sigma=1.5)

        # ── Save arrays ────────────────────────────────────────────────────
        np.save(os.path.join(d, "importance.npy"), imp)

        # ── Visualize ──────────────────────────────────────────────────────
        W, H = img_base.size
        sz = (W, H)

        heatmap_img    = jet(imp, sz)
        overlay_slic   = overlay(img_base, m_slic)
        overlay_direct = overlay(img_base, m_direct)

        mask_slic_img   = Image.fromarray((m_slic   * 255).astype(np.uint8)).resize(sz, Image.NEAREST).convert("RGB")
        mask_direct_img = Image.fromarray((m_direct * 255).astype(np.uint8)).resize(sz, Image.NEAREST).convert("RGB")

        heatmap_img.save(   os.path.join(d, "heatmap.png"))
        overlay_slic.save(  os.path.join(d, "overlay_slic.png"))
        overlay_direct.save(os.path.join(d, "overlay_direct.png"))
        mask_slic_img.save( os.path.join(d, "mask_slic.png"))
        mask_direct_img.save(os.path.join(d, "mask_direct.png"))

        # ── 5-panel row: base | heatmap | SLIC overlay | direct overlay | sasd ──
        cell = 384
        def rs(img): return img.resize((cell, cell), Image.LANCZOS)

        row = hstack([
            add_label(rs(img_base),       "Original"),
            add_label(rs(heatmap_img),    "Attn Heatmap"),
            add_label(rs(overlay_slic),   "SLIC mask (current)"),
            add_label(rs(overlay_direct), "Direct threshold (no SLIC)"),
            add_label(rs(img_sasd),       "SASD result"),
        ])
        row.save(os.path.join(d, "compare_5panel.png"))
        rows.append(row)
        print(f"  Saved all to {d}/")

        gc.collect()
        torch.cuda.empty_cache()

    remove_hook(orig)
    del wrapper

    # ── Combine all prompt rows ────────────────────────────────────────────
    if rows:
        combined = vstack(rows, pad=6)
        out = os.path.join(OUT_ROOT, "compare_all.png")
        combined.save(out)
        size_kb = os.path.getsize(out) // 1024
        print(f"\nSaved combined grid → {out}  ({combined.width}×{combined.height}, {size_kb} KB)")


if __name__ == "__main__":
    main()
