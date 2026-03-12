#!/usr/bin/env python3
"""
Supplement mask_compare visualizations with additional prompts.

For each prompt generates:
  baseline.png        — dense baseline
  sasd.png            — our accelerated method
  heatmap.png         — token importance heatmap (jet colormap)
  mask_direct.png     — binary foreground mask (direct percentile threshold)
  overlay_direct.png  — baseline image + mask overlay
  compare_4panel.png  — 4-panel: baseline | heatmap | overlay | sasd

Output: outputs/mask_compare/{tag}/

New prompts cover:
  portrait_woman  — portrait, beautiful, intricate, elegant, vivid  (5 anchors)
  deer_forest     — volumetric lighting, intricate, sharp focus      (3 anchors)
  galaxy_jar      — detailed, volumetric lighting                    (2 anchors, from gallery p0078)
  anime_fullbody  — detailed, full body                              (2 anchors, from gallery p0067)
  robot_plant     — (no anchors) → uniform heatmap baseline
"""

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Config ────────────────────────────────────────────────────────────────────

NEW_PROMPTS = [
    ("portrait_woman",
     "Digital painting portrait of a beautiful woman with intricate floral crown, "
     "depth of field, elegant, vivid colors, photorealistic, sharp focus"),

    ("deer_forest",
     "A majestic deer in an enchanted forest, volumetric lighting, intricate details, "
     "photorealistic, sharp focus, cinematic"),

    ("galaxy_jar",
     "Supernova in a glass jar, Insanely detailed, photorealistic, 8k, "
     "ultra high resolution, volumetric lighting, taken with canon eos 5d"),

    ("anime_fullbody",
     "anime, highly detailed, colored pencil and pastel drawing 16k wallpaper, "
     "Cute girl, jumping, carmine hair, wavy hairstyle, turquoise eyes, "
     "wearing frilled black dress, black knee high socks, full body, symmetrical face"),

    ("robot_plant",
     "kawaii illustration of a robot watering a plant"),  # no anchors — uniform heatmap
]

SEED      = 42
STEPS     = 28
RATIO     = 0.5
OUT_ROOT  = "outputs/mask_compare"

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
            importance = stacked.mean(dim=[0, 1, 2])
            if clip_token_weights is not None:
                w = clip_token_weights.to(importance.device, dtype=importance.dtype)
                w = w / (w.sum() + 1e-8)
                importance = (importance * w.unsqueeze(0)).sum(dim=-1)
            else:
                importance = importance.mean(dim=-1)
            _cap["imp"] = importance.reshape(patch_h, patch_w).cpu().float().numpy()

        mask = _orig(self, affinity_maps, patch_h, patch_w, clip_token_weights)
        return mask

    SD3SemanticMaskBuilder.build_mask = _patched
    return _orig

def remove_hook(orig):
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    SD3SemanticMaskBuilder.build_mask = orig


# ── Mask / vis helpers ────────────────────────────────────────────────────────

def direct_threshold_mask(imp_np, ratio=0.5, blur_sigma=1.5):
    thresh = np.percentile(imp_np, (1 - ratio) * 100)
    mask = (imp_np >= thresh).astype(np.float32)
    if blur_sigma > 0:
        t = torch.from_numpy(mask)
        from src.sparse.boundary_ops import gaussian_blur
        t = gaussian_blur(t, sigma=blur_sigma)
        mask = t.numpy()
    return mask


def jet_colormap(imp_np, size):
    lo, hi = imp_np.min(), imp_np.max()
    norm = (imp_np - lo) / (hi - lo + 1e-8)
    r = np.clip(1.5 - abs(norm * 4 - 3), 0, 1)
    g = np.clip(1.5 - abs(norm * 4 - 2), 0, 1)
    b = np.clip(1.5 - abs(norm * 4 - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.NEAREST)


def overlay_mask(base_img, mask_np, fg=(255, 80, 0), bg=(0, 100, 220), alpha=0.45):
    img = np.array(base_img).astype(float)
    h, w = img.shape[:2]
    m = np.array(Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
        (w, h), Image.NEAREST)).astype(float) / 255.0
    fg_c = np.array(fg, float)
    bg_c = np.array(bg, float)
    result = img * (1 - m[:, :, None] * alpha) + fg_c * m[:, :, None] * alpha
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import gc
    from src.models.sd3_wrapper import SD3Wrapper

    os.makedirs(OUT_ROOT, exist_ok=True)
    wrapper = SD3Wrapper()
    orig = install_hook()

    for tag, prompt in NEW_PROMPTS:
        d = os.path.join(OUT_ROOT, tag)
        os.makedirs(d, exist_ok=True)

        # Skip if all outputs already exist
        needed = ["baseline.png", "sasd.png", "heatmap.png",
                  "mask_direct.png", "overlay_direct.png"]
        if all(os.path.exists(os.path.join(d, f)) for f in needed):
            print(f"[{tag}] already done, skip")
            continue

        print(f"\n[{tag}]")
        print(f"  prompt: {prompt[:80]}")
        _cap.clear()

        # ── Baseline ──────────────────────────────────────────────────────
        base_path = os.path.join(d, "baseline.png")
        if not os.path.exists(base_path):
            print("  baseline...", end=" ", flush=True)
            img_base = wrapper.generate(prompt, seed=SEED, steps=STEPS, cfg_scale=7.0)
            img_base.save(base_path)
            print("done")
        else:
            img_base = Image.open(base_path)
            print("  baseline: loaded from cache")

        # ── SASD (captures heatmap) ────────────────────────────────────────
        sasd_path = os.path.join(d, "sasd.png")
        if not os.path.exists(sasd_path):
            print("  sasd (capturing mask)...", end=" ", flush=True)
            img_sasd = wrapper.generate_accelerated(
                prompt, seed=SEED, steps=STEPS,
                cfg_scale=7.0,
                mask_type="semantic", s_fg=9.0, s_bg=2.0,
                full_skip_interval=2, n_segments=64,
                mask_step=5, skip_ratio=RATIO,
            )
            img_sasd.save(sasd_path)
            print("done")
        else:
            img_sasd = Image.open(sasd_path)
            print("  sasd: loaded from cache")

        if "imp" not in _cap:
            print("  WARNING: importance map not captured, skipping visualization")
            continue

        imp = _cap["imp"]

        # ── Compute direct mask ────────────────────────────────────────────
        m_direct = direct_threshold_mask(imp, ratio=RATIO, blur_sigma=1.5)

        # ── Save raw importance ────────────────────────────────────────────
        np.save(os.path.join(d, "importance.npy"), imp)

        # ── Visualizations ────────────────────────────────────────────────
        W, H = img_base.size
        sz = (W, H)

        heatmap_img    = jet_colormap(imp, sz)
        mask_direct_img = Image.fromarray(
            (m_direct * 255).astype(np.uint8)).resize(sz, Image.NEAREST).convert("RGB")
        overlay_img    = overlay_mask(img_base, m_direct)

        heatmap_img.save(    os.path.join(d, "heatmap.png"))
        mask_direct_img.save(os.path.join(d, "mask_direct.png"))
        overlay_img.save(    os.path.join(d, "overlay_direct.png"))

        # ── 4-panel: baseline | heatmap | overlay | sasd ──────────────────
        cell = 384
        def rs(img): return img.resize((cell, cell), Image.LANCZOS)

        panel = hstack([
            add_label(rs(img_base),     "Baseline"),
            add_label(rs(heatmap_img),  "Importance Heatmap"),
            add_label(rs(overlay_img),  "Foreground Mask"),
            add_label(rs(img_sasd),     "AccelAes (Ours)"),
        ])
        panel.save(os.path.join(d, "compare_4panel.png"))
        print(f"  Saved all to {d}/")

        gc.collect()
        torch.cuda.empty_cache()

    remove_hook(orig)
    del wrapper
    print("\nDone.")


if __name__ == "__main__":
    main()
