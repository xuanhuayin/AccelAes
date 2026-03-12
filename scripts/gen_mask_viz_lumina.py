#!/usr/bin/env python3
"""
Mask compare visualizations using Lumina-Next-T2I (main model).

For each prompt generates:
  baseline.png        — dense baseline (no acceleration)
  sasd.png            — AccelAes (semantic mask + fskip2 + sparse)
  heatmap.png         — token importance heatmap (jet colormap)
  mask_direct.png     — binary foreground mask (direct percentile threshold)
  overlay_direct.png  — baseline + foreground mask overlay
  compare_4panel.png  — baseline | heatmap | overlay | sasd

Output: outputs/mask_compare/{tag}_lumina/

Prompts cover:
  lion_lumina         — intricate, realistic          (natural/animal)
  astronaut_lumina    — intricate, portrait            (human portrait)
  city_lumina         — intricate                     (urban scene)
  portrait_woman_lumina — portrait, beautiful, intricate, elegant, vivid, sharp focus (6 anchors)
  deer_forest_lumina  — volumetric lighting, intricate, sharp focus
  galaxy_jar_lumina   — detailed, volumetric lighting  (object close-up, from gallery p0078)
  anime_fullbody_lumina — detailed, full body          (from gallery p0067)
  robot_plant_lumina  — (no anchors)                   → uniform heatmap control
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
    ("lion_lumina",
     "A majestic lion at golden hour in the savanna, photorealistic, "
     "cinematic lighting, intricate fur detail"),

    ("astronaut_lumina",
     "Portrait of an astronaut with intricate spacesuit details, "
     "photorealistic, cinematic"),

    ("city_lumina",
     "A futuristic Tokyo street at night with neon lights and intricate "
     "reflections on wet pavement, cinematic"),

    ("portrait_woman_lumina",
     "Digital painting portrait of a beautiful woman with intricate floral crown, "
     "depth of field, elegant, vivid colors, photorealistic, sharp focus"),

    ("deer_forest_lumina",
     "A majestic deer in an enchanted forest, volumetric lighting, intricate details, "
     "photorealistic, sharp focus, cinematic"),

    ("galaxy_jar_lumina",
     "Supernova in a glass jar, Insanely detailed, photorealistic, 8k, "
     "ultra high resolution, volumetric lighting, taken with canon eos 5d"),

    ("anime_fullbody_lumina",
     "anime, highly detailed, colored pencil and pastel drawing 16k wallpaper, "
     "Cute girl, jumping, carmine hair, wavy hairstyle, turquoise eyes, "
     "wearing frilled black dress, black knee high socks, full body, symmetrical face"),

    ("robot_plant_lumina",
     "kawaii illustration of a robot watering a plant"),  # no anchors
]

# AccelAes full config (matches SASD_FULL_KWARGS from main experiments)
SASD_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    skip_ratio=0.50, s_fg=7.0, s_bg=1.0,
    mask_step=5, full_skip_interval=2,
    sparse_ffn=True, sparse_blocks=True,
)

SEED     = 42
STEPS    = 30
RATIO    = 0.50
OUT_ROOT = "outputs/mask_compare"


# ── Heatmap capture hook ──────────────────────────────────────────────────────

_cap = {}

def install_hook():
    from src.sparse.mask_builders import SemanticMaskBuilder
    _orig = SemanticMaskBuilder.build_mask_from_cross_attention

    def _patched(self, cross_attn_maps, token_importance, latent_h, latent_w):
        # Replicate heatmap computation to capture before thresholding
        if cross_attn_maps:
            all_attn = torch.stack([m.mean(dim=0) for m in cross_attn_maps])
            avg_attn = all_attn.mean(dim=0)
            weighted = avg_attn * token_importance.unsqueeze(0)
            heatmap = weighted.sum(dim=-1)
            num_patches = latent_h * latent_w
            heatmap = heatmap[:num_patches].reshape(latent_h, latent_w)
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            _cap["heatmap"] = heatmap.cpu().float().numpy()
            _cap["token_importance"] = token_importance.cpu().float().numpy()

        return _orig(self, cross_attn_maps, token_importance, latent_h, latent_w)

    SemanticMaskBuilder.build_mask_from_cross_attention = _patched
    return _orig

def remove_hook(orig):
    from src.sparse.mask_builders import SemanticMaskBuilder
    SemanticMaskBuilder.build_mask_from_cross_attention = orig


# ── Visualization helpers ─────────────────────────────────────────────────────

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


def vstack(imgs, pad=6):
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
    from src.models.dit_wrapper import LuminaDiTWrapper

    os.makedirs(OUT_ROOT, exist_ok=True)
    print("Loading Lumina-Next-T2I...")
    wrapper = LuminaDiTWrapper()
    orig = install_hook()

    rows = []

    for tag, prompt in PROMPTS:
        d = os.path.join(OUT_ROOT, tag)
        os.makedirs(d, exist_ok=True)

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
            img_base = wrapper.generate(prompt=prompt, seed=SEED, steps=STEPS,
                                        cfg_scale=4.0)
            img_base.save(base_path)
            print("done")
        else:
            img_base = Image.open(base_path)
            print("  baseline: loaded")

        # ── SASD (captures heatmap) ────────────────────────────────────────
        sasd_path = os.path.join(d, "sasd.png")
        if not os.path.exists(sasd_path):
            print("  sasd (capturing mask)...", end=" ", flush=True)
            img_sasd = wrapper.generate_accelerated_dual(
                prompt=prompt, seed=SEED, steps=STEPS,
                cfg_scale=4.0,
                **SASD_KWARGS,
            )
            img_sasd.save(sasd_path)
            print("done")
        else:
            img_sasd = Image.open(sasd_path)
            print("  sasd: loaded")
            # Re-run sasd to capture heatmap if only image was cached
            if "heatmap" not in _cap:
                print("  re-running sasd to capture heatmap...", end=" ", flush=True)
                _cap.clear()
                wrapper.generate_accelerated_dual(
                    prompt=prompt, seed=SEED, steps=STEPS,
                    cfg_scale=4.0,
                    **SASD_KWARGS,
                )
                print("done")

        if "heatmap" not in _cap:
            print("  WARNING: heatmap not captured (no anchor match?), using fallback")
            # Fallback: uniform heatmap
            W, H = img_base.size
            ph, pw = H // 16, W // 16
            _cap["heatmap"] = np.ones((ph, pw), dtype=np.float32) * 0.5

        imp = _cap["heatmap"]

        # ── Compute direct mask ────────────────────────────────────────────
        m_direct = direct_threshold_mask(imp, ratio=RATIO, blur_sigma=1.5)

        # ── Save raw importance ────────────────────────────────────────────
        np.save(os.path.join(d, "importance.npy"), imp)

        # ── Visualizations ────────────────────────────────────────────────
        W, H = img_base.size
        sz = (W, H)

        heatmap_img     = jet_colormap(imp, sz)
        mask_direct_img = Image.fromarray(
            (m_direct * 255).astype(np.uint8)).resize(sz, Image.NEAREST).convert("RGB")
        overlay_img     = overlay_mask(img_base, m_direct)

        heatmap_img.save(     os.path.join(d, "heatmap.png"))
        mask_direct_img.save( os.path.join(d, "mask_direct.png"))
        overlay_img.save(     os.path.join(d, "overlay_direct.png"))

        # ── 4-panel: baseline | heatmap | overlay | sasd ──────────────────
        cell = 384
        def rs(img): return img.resize((cell, cell), Image.LANCZOS)

        panel = hstack([
            add_label(rs(img_base),    "Baseline"),
            add_label(rs(heatmap_img), "Importance Heatmap"),
            add_label(rs(overlay_img), "Foreground Mask"),
            add_label(rs(img_sasd),    "AccelAes (Ours)"),
        ])
        panel.save(os.path.join(d, "compare_4panel.png"))
        rows.append(panel)
        print(f"  Saved all to {d}/")

        gc.collect()
        torch.cuda.empty_cache()

    remove_hook(orig)
    del wrapper

    # ── Combined overview grid ─────────────────────────────────────────────
    if rows:
        combined = vstack(rows, pad=6)
        out = os.path.join(OUT_ROOT, "compare_all_lumina.png")
        combined.save(out)
        print(f"\nSaved combined grid → {out}  ({combined.width}×{combined.height})")

    print("\nDone.")


if __name__ == "__main__":
    main()
