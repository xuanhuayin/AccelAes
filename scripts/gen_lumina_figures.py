#!/usr/bin/env python3
"""
AccelAes 论文可视化图生成脚本 — Lumina 版本。

选取"前后提升最大"的 prompt（含 intricate/detailed 等美学关键词，前后景分明）。

5 列横向拼图：
  [Baseline | Attn Heatmap | Semantic Mask | Mask Overlay | AccelAes Result]

输出：
  outputs/accelae_figures_lumina/
    lumina/{tag}/  — baseline.png, heatmap.png, mask.png, overlay.png, result.png
    rows/          — {tag}_row.png (5-panel horizontal)
    full_grid.png  — 所有 prompt 竖排大图

用法：
  python scripts/gen_lumina_figures.py
  python scripts/gen_lumina_figures.py --skip_gen   # 图已存在，只重拼
  python scripts/gen_lumina_figures.py --prompts_n 4
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

# ── Prompts（专为"前后提升大"挑选：含美学关键词 + 明确前后景）─────────────────

PROMPTS = [
    ("lion",
     "A majestic lion at golden hour in the savanna, photorealistic, "
     "cinematic lighting, intricate fur detail"),
    ("warrior",
     "A samurai warrior in intricate armor standing in a bamboo forest, "
     "cinematic lighting, photorealistic, sharp focus"),
    ("dragon",
     "An ancient dragon with intricate scales perched on a mountain peak, "
     "dramatic lighting, cinematic fantasy art"),
    ("portrait",
     "Close up portrait of an elegant woman with intricate floral headpiece, "
     "photorealistic, studio lighting, detailed skin texture"),
    ("flowers",
     "Macro photograph of orchids with morning dew drops, "
     "photorealistic, intricate petals, soft bokeh"),
    ("knight",
     "A medieval knight in intricate engraved armor, dramatic side lighting, "
     "photorealistic, sharp focus, dark stone background"),
    ("tiger",
     "Close up portrait of a Bengal tiger with intricate fur patterns, "
     "golden hour light, photorealistic, cinematic, vivid colors"),
    ("mech",
     "A steampunk mechanical engineer with intricate brass gears and clockwork "
     "prosthetic arm, photorealistic, cinematic, detailed"),
]

SEED      = 42
STEPS     = 30
OUT_ROOT  = "outputs/accelae_figures_lumina"
CELL      = 384

SASD_KWARGS = dict(
    mask_type="semantic",
    region_method="threshold",
    skip_ratio=0.50,
    s_fg=7.0, s_bg=1.0,
    mask_step=5,
    full_skip_interval=2,
    sparse_ffn=True,
    sparse_blocks=True,
)

# ── Mask capture hook ─────────────────────────────────────────────────────────

_cap = {}


def install_hook():
    from src.sparse.mask_builders import SemanticMaskBuilder
    _orig = SemanticMaskBuilder.build_mask_from_cross_attention

    def _patched(self, cross_attn_maps, token_importance, latent_h, latent_w):
        # Recompute heatmap for visualization (same logic as original method)
        if cross_attn_maps:
            num_patches = latent_h * latent_w
            all_attn = torch.stack([m.mean(dim=0) for m in cross_attn_maps])  # (L, P, T)
            avg_attn = all_attn.mean(dim=0)                                    # (P, T)
            weighted = avg_attn * token_importance.unsqueeze(0)                # (P, T)
            heatmap = weighted.sum(dim=-1)[:num_patches].reshape(latent_h, latent_w)
            lo, hi = heatmap.min(), heatmap.max()
            if hi > lo:
                heatmap = (heatmap - lo) / (hi - lo + 1e-8)
            _cap["imp"] = heatmap.cpu().float().numpy()

        mask = _orig(self, cross_attn_maps, token_importance, latent_h, latent_w)
        if mask is not None:
            _cap["mask"] = mask.cpu().float().numpy()
        return mask

    SemanticMaskBuilder.build_mask_from_cross_attention = _patched
    return _orig


def remove_hook(orig):
    from src.sparse.mask_builders import SemanticMaskBuilder
    SemanticMaskBuilder.build_mask_from_cross_attention = orig


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


# ── Generation ────────────────────────────────────────────────────────────────

def gen_lumina(prompts, skip_gen=False):
    from src.models.dit_wrapper import LuminaDiTWrapper
    wrapper = LuminaDiTWrapper()
    orig = install_hook()

    for tag, prompt in prompts:
        d = os.path.join(OUT_ROOT, "lumina", tag)
        os.makedirs(d, exist_ok=True)
        print(f"\n[{tag}]")
        _cap.clear()

        base_path   = os.path.join(d, "baseline.png")
        result_path = os.path.join(d, "result.png")

        # baseline
        if not skip_gen or not os.path.exists(base_path):
            print("  baseline...")
            img_base = wrapper.generate(prompt, seed=SEED, cfg_scale=4.0)
            img_base.save(base_path)
        else:
            img_base = Image.open(base_path).convert("RGB")

        # AccelAes result (with mask capture via hook)
        print("  AccelAes...")
        img_result = wrapper.generate_accelerated_dual(
            prompt, seed=SEED, steps=STEPS, **SASD_KWARGS
        )
        img_result.save(result_path)

        # Save mask visualizations
        if "imp" in _cap:
            W, H = img_base.size
            imp = _cap["imp"]
            # Use captured mask if available, else recompute
            if "mask" in _cap:
                m = _cap["mask"]
            else:
                thresh = np.percentile(imp, 50)
                m = (imp >= thresh).astype(np.float32)

            jet(imp, (W, H)).save(os.path.join(d, "heatmap.png"))
            mask_binary_img(m, (W, H)).save(os.path.join(d, "mask.png"))
            overlay_img(img_base, m).save(os.path.join(d, "overlay.png"))
            print(f"  Saved mask artifacts → {d}/")
        else:
            print("  WARNING: mask not captured")

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
        d = os.path.join(OUT_ROOT, "lumina", tag)
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
    total_w = rows[0].width if rows else cell * 5 + 5 * 4
    header_h = 36

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
    parser.add_argument("--skip_gen",  action="store_true",
                        help="Skip image generation, only rebuild grids")
    parser.add_argument("--prompts_n", type=int, default=None,
                        help="Use only first N prompts")
    parser.add_argument("--cell",      type=int, default=384,
                        help="Panel size in pixels (default 384)")
    args = parser.parse_args()
    cell = args.cell

    os.makedirs(OUT_ROOT, exist_ok=True)
    prompts = PROMPTS[:args.prompts_n] if args.prompts_n else PROMPTS
    print(f"AccelAes Lumina figures: {len(prompts)} prompts, cell={cell}")
    print(f"Output: {os.path.abspath(OUT_ROOT)}")

    if not args.skip_gen:
        print("\n=== Generating Lumina images ===")
        gen_lumina(prompts, skip_gen=False)
    else:
        print("\n=== Skipping generation (--skip_gen) ===")

    print("\n=== Building grids ===")
    rows = build_rows(prompts, cell=cell)
    build_full_grid(rows, cell=cell)

    print("\nDone.")


if __name__ == "__main__":
    main()
