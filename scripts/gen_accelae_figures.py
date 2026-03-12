#!/usr/bin/env python3
"""
AccelAes 论文可视化图生成脚本。

每个 prompt 生成 5 列横向拼图：
  [Baseline | Attn Heatmap | Semantic Mask | Mask Overlay | AccelAes Result]

mask 使用 direct threshold（无 SLIC），与正式评估一致。

输出：
  outputs/accelae_figures/
    sd3/{tag}/  — baseline.png, heatmap.png, mask.png, overlay.png, result.png
    rows/       — {tag}_row.png (5-panel horizontal)
    full_grid.png  — 所有 prompt 竖排大图

用法：
  python scripts/gen_accelae_figures.py
  python scripts/gen_accelae_figures.py --skip_gen   # 图已存在，只重拼
  python scripts/gen_accelae_figures.py --prompts_n 4
"""

import argparse
import gc
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = [
    ("lion",
     "A majestic lion at golden hour in the savanna, photorealistic, "
     "cinematic lighting, intricate fur detail"),
    ("astronaut",
     "Portrait of an astronaut with intricate spacesuit details, "
     "photorealistic, cinematic"),
    ("city",
     "A futuristic Tokyo street at night with neon lights and intricate "
     "reflections on wet pavement, cinematic"),
    ("dragon",
     "An ancient dragon with intricate scales perched on a mountain peak, "
     "dramatic lighting, cinematic fantasy art"),
    ("scientist",
     "A Victorian-era scientist in her detailed laboratory, photorealistic, "
     "intricate steampunk equipment"),
    ("flowers",
     "Macro photograph of orchids with morning dew drops, "
     "photorealistic, intricate petals, soft bokeh"),
    ("warrior",
     "A samurai warrior in intricate armor standing in a bamboo forest, "
     "cinematic lighting, photorealistic, sharp focus"),
    ("palace",
     "Grand baroque palace interior with intricate golden decorations, "
     "dramatic lighting, photorealistic, detailed architecture"),
    ("waterfall",
     "A serene mountain waterfall with intricate rock textures and "
     "lush ferns, photorealistic, cinematic, vivid colors"),
    ("portrait",
     "Close up portrait of an elegant woman with intricate floral headpiece, "
     "photorealistic, studio lighting, detailed skin texture"),
]

SEED      = 42
STEPS     = 28
OUT_ROOT  = "outputs/accelae_figures"
CELL      = 384   # pixel size per panel

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

        mask_slic = _orig(self, affinity_maps, patch_h, patch_w, clip_token_weights)
        return mask_slic

    SD3SemanticMaskBuilder.build_mask = _patched
    return _orig

def remove_hook(orig):
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    SD3SemanticMaskBuilder.build_mask = orig


# ── Direct threshold mask ─────────────────────────────────────────────────────

def direct_mask(imp_np, ratio=0.5, blur_sigma=1.5):
    thresh = np.percentile(imp_np, (1 - ratio) * 100)
    mask = (imp_np >= thresh).astype(np.float32)
    if blur_sigma > 0:
        t = torch.from_numpy(mask)
        from src.sparse.boundary_ops import gaussian_blur
        mask = gaussian_blur(t, sigma=blur_sigma).numpy()
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


def mask_binary_img(mask_np, size):
    """White = foreground, black = background."""
    m = (mask_np * 255).astype(np.uint8)
    return Image.fromarray(m).resize(size, Image.NEAREST).convert("RGB")


def overlay_img(base, mask_np, fg=(255, 80, 0), bg=(0, 100, 220), alpha=0.45):
    img = np.array(base).astype(float)
    h, w = img.shape[:2]
    m = np.array(Image.fromarray(
        (mask_np * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    ).astype(float) / 255.0
    fg_c, bg_c = np.array(fg, float), np.array(bg, float)
    result = img * (1 - m[:, :, None] * alpha) + fg_c * m[:, :, None] * alpha
    result = result * (1 - (1 - m[:, :, None]) * alpha * 0.4) + bg_c * (1 - m[:, :, None]) * alpha * 0.4
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def add_label(img, text, fsize=20, bg=(20, 20, 20), fg=(240, 240, 240)):
    bar = fsize + 10
    out = Image.new("RGB", (img.width, img.height + bar), bg)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 5), text, fill=fg, font=font)
    return out


def hstack(imgs, pad=5, bg=(230, 230, 230)):
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for img in imgs:
        canvas.paste(img, (x, (h - img.height) // 2))
        x += img.width + pad
    return canvas


def vstack(imgs, pad=6, bg=(230, 230, 230)):
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    y = 0
    for img in imgs:
        canvas.paste(img, ((w - img.width) // 2, y))
        y += img.height + pad
    return canvas


def rs(img, cell=CELL):
    return img.resize((cell, cell), Image.LANCZOS)


# ── Generation ────────────────────────────────────────────────────────────────

def gen_sd3(prompts, skip_gen=False):
    from src.models.sd3_wrapper import SD3DiTWrapper
    wrapper = SD3DiTWrapper(dtype="bf16")
    orig = install_hook()

    for tag, prompt in prompts:
        d = os.path.join(OUT_ROOT, "sd3", tag)
        os.makedirs(d, exist_ok=True)
        print(f"\n[{tag}]")
        _cap.clear()

        base_path   = os.path.join(d, "baseline.png")
        result_path = os.path.join(d, "result.png")

        # baseline
        if not skip_gen or not os.path.exists(base_path):
            print("  baseline...")
            img_base = wrapper.generate(prompt, seed=SEED, steps=STEPS)
            img_base.save(base_path)
        else:
            img_base = Image.open(base_path).convert("RGB")

        # AccelAes result (with mask capture)
        if not skip_gen or not os.path.exists(result_path):
            print("  AccelAes...")
            img_result = wrapper.generate_accelerated(
                prompt, seed=SEED, steps=STEPS,
                mask_type="semantic", s_fg=9.0, s_bg=2.0,
                full_skip_interval=2, n_segments=64,
                mask_step=5, skip_ratio=0.5,
            )
            img_result.save(result_path)
        else:
            # Re-run to capture mask
            print("  re-running for mask capture...")
            img_result = wrapper.generate_accelerated(
                prompt, seed=SEED, steps=STEPS,
                mask_type="semantic", s_fg=9.0, s_bg=2.0,
                full_skip_interval=2, n_segments=64,
                mask_step=5, skip_ratio=0.5,
            )
            img_result.save(result_path)

        # Save mask visualizations
        if "imp" in _cap:
            W, H = img_base.size
            imp = _cap["imp"]
            m   = direct_mask(imp, ratio=0.5, blur_sigma=1.5)

            jet(imp, (W, H)).save(os.path.join(d, "heatmap.png"))
            mask_binary_img(m, (W, H)).save(os.path.join(d, "mask.png"))
            overlay_img(img_base, m).save(os.path.join(d, "overlay.png"))
            print(f"  Saved mask artifacts → {d}/")

        gc.collect()
        torch.cuda.empty_cache()

    remove_hook(orig)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()


# ── Grid assembly ─────────────────────────────────────────────────────────────

COL_LABELS = [
    "Baseline",
    "Attn Heatmap",
    "Semantic Mask\n(Direct Threshold)",
    "Mask Overlay",
    "AccelAes (ours)",
]


def build_rows(prompts, cell=384):
    rows_dir = os.path.join(OUT_ROOT, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    all_rows = []

    for tag, prompt in prompts:
        d = os.path.join(OUT_ROOT, "sd3", tag)
        panels = []
        for key, label in zip(
            ["baseline", "heatmap", "mask", "overlay", "result"], COL_LABELS
        ):
            p = os.path.join(d, f"{key}.png")
            img = Image.open(p).convert("RGB") if os.path.exists(p) \
                else Image.new("RGB", (cell, cell), (180, 180, 180))
            panels.append(add_label(img.resize((cell, cell), Image.LANCZOS), label))

        row = hstack(panels)
        row_path = os.path.join(rows_dir, f"{tag}_row.png")
        row.save(row_path)
        all_rows.append(row)
        print(f"  Row saved → {row_path}  ({row.width}×{row.height})")

    return all_rows


def build_full_grid(rows, cell=384):
    col_w = cell + 5
    header_h = 36
    total_w = rows[0].width if rows else cell * 5 + 5 * 4

    header = Image.new("RGB", (total_w, header_h), (255, 255, 255))
    draw = ImageDraw.Draw(header)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for ci, label in enumerate(COL_LABELS):
        x = 5 + ci * (col_w + 5) + col_w // 2
        text = label.replace("\n", " ")
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 10), text, fill=(30, 30, 30), font=font)

    grid = vstack([header] + rows, pad=4, bg=(230, 230, 230))
    out = os.path.join(OUT_ROOT, "full_grid.png")
    grid.save(out)
    size_kb = os.path.getsize(out) // 1024
    print(f"\nFull grid → {out}  ({grid.width}×{grid.height}, {size_kb} KB)")
    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_gen",   action="store_true")
    parser.add_argument("--prompts_n",  type=int, default=None)
    parser.add_argument("--cell",       type=int, default=384)
    args = parser.parse_args()
    cell = args.cell

    os.makedirs(OUT_ROOT, exist_ok=True)
    prompts = PROMPTS[:args.prompts_n] if args.prompts_n else PROMPTS
    print(f"AccelAes figures: {len(prompts)} prompts, cell={cell}")
    print(f"Output: {os.path.abspath(OUT_ROOT)}")

    print("\n=== Generating SD3 images ===")
    gen_sd3(prompts, skip_gen=args.skip_gen)

    print("\n=== Building grids ===")
    rows = build_rows(prompts, cell=cell)
    build_full_grid(rows, cell=cell)

    print("\nDone.")


if __name__ == "__main__":
    main()
