#!/usr/bin/env python3
"""
论文图生成脚本：生成用于 paper 的高质量对比图 + 语义 mask 可视化。

输出结构：
  outputs/paper_figures/
    sd3/{tag}/baseline.png, sasd.png, heatmap.png, mask_overlay.png
    flux/{tag}/baseline.png, sasd.png
    grids/motivation_sd3.png     -- 3 prompts × [img|heatmap|mask|sasd]
    grids/compare_sd3.png        -- N prompts × [baseline|sasd]
    grids/compare_flux.png       -- N prompts × [baseline|sasd]

用法：
  python scripts/gen_paper_figures.py                    # 全部
  python scripts/gen_paper_figures.py --model sd3        # 只跑 SD3
  python scripts/gen_paper_figures.py --model flux       # 只跑 FLUX
  python scripts/gen_paper_figures.py --skip_gen         # 只做拼图，图片已存在
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

# ── Prompts (精选：有明确美学锚定词，视觉多样) ─────────────────────────────────

PROMPTS = [
    # tag, prompt
    ("lion",       "A majestic lion at golden hour in the savanna, photorealistic, "
                   "cinematic lighting, intricate fur detail"),
    ("astronaut",  "Portrait of an astronaut with intricate spacesuit details, "
                   "photorealistic, cinematic"),
    ("city",       "A futuristic Tokyo street at night with neon lights and intricate "
                   "reflections on wet pavement, cinematic"),
    ("dragon",     "An ancient dragon with intricate scales perched on a mountain peak, "
                   "dramatic lighting, cinematic fantasy art"),
    ("scientist",  "A Victorian-era scientist in her detailed laboratory, photorealistic, "
                   "intricate steampunk equipment"),
    ("flowers",    "Macro photograph of orchids with morning dew drops, "
                   "photorealistic, intricate petals, soft bokeh"),
]

# First 3 prompts used for motivation figure (with mask visualization)
MOTIVATION_TAGS = ["lion", "astronaut", "city"]

SEED = 42
SD3_STEPS  = 28
FLUX_STEPS = 28
OUT_ROOT   = "outputs/paper_figures"


# ── Semantic mask capture hook ────────────────────────────────────────────────

_mask_capture = {}   # {"raw_importance": np, "binary_mask": np}


def _install_mask_capture_hook():
    """Monkeypatch SD3SemanticMaskBuilder.build_mask to export intermediate maps."""
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    _orig = SD3SemanticMaskBuilder.build_mask

    def _patched(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
        # Recompute raw importance (same logic as original, before SLIC)
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
                importance = importance.mean(dim=-1)   # (N_img,)
            imp_np = importance.reshape(patch_h, patch_w).cpu().float().numpy()
            _mask_capture["raw_importance"] = imp_np

        mask = _orig(self, affinity_maps, patch_h, patch_w, clip_token_weights)
        _mask_capture["binary_mask"] = mask.cpu().float().numpy()
        return mask

    SD3SemanticMaskBuilder.build_mask = _patched
    return _orig


def _remove_mask_capture_hook(orig_fn):
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    SD3SemanticMaskBuilder.build_mask = orig_fn


# ── Visualization helpers ─────────────────────────────────────────────────────

def importance_to_heatmap(imp_np: np.ndarray, size: tuple) -> Image.Image:
    """Convert importance map to a jet-colored heatmap PIL image."""
    # Normalize
    lo, hi = imp_np.min(), imp_np.max()
    if hi - lo > 1e-8:
        norm = (imp_np - lo) / (hi - lo)
    else:
        norm = np.ones_like(imp_np) * 0.5

    # Jet colormap (R,G,B) approximation
    r = np.clip(1.5 - abs(norm * 4 - 3), 0, 1)
    g = np.clip(1.5 - abs(norm * 4 - 2), 0, 1)
    b = np.clip(1.5 - abs(norm * 4 - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB").resize(size, Image.NEAREST)
    return img


def mask_to_overlay(image: Image.Image, mask_np: np.ndarray,
                    fg_color=(255, 80, 0), bg_color=(0, 100, 220),
                    alpha=0.45) -> Image.Image:
    """Draw fg/bg overlay on image: foreground=warm, background=cool."""
    img_np = np.array(image).astype(float)
    h, w = img_np.shape[:2]
    mask_resized = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
        (w, h), Image.NEAREST)
    m = np.array(mask_resized).astype(float) / 255.0

    fg_c = np.array(fg_color, dtype=float)
    bg_c = np.array(bg_color, dtype=float)

    result = img_np.copy()
    # Foreground blend
    fg_mask = m[:, :, None]
    result = result * (1 - fg_mask * alpha) + fg_c * fg_mask * alpha
    # Background blend (subtle)
    bg_mask = (1 - m)[:, :, None]
    result = result * (1 - bg_mask * alpha * 0.5) + bg_c * bg_mask * alpha * 0.5

    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def add_label(img: Image.Image, text: str, font_size: int = 20,
              bg=(0, 0, 0), fg=(255, 255, 255)) -> Image.Image:
    """Add a text label bar at the bottom of an image."""
    bar_h = font_size + 8
    new_img = Image.new("RGB", (img.width, img.height + bar_h), bg)
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    tx = (img.width - tw) // 2
    draw.text((tx, img.height + 4), text, fill=fg, font=font)
    return new_img


def make_grid(images: list, cols: int, pad: int = 4,
              bg=(240, 240, 240)) -> Image.Image:
    """Arrange images in a grid."""
    if not images:
        return Image.new("RGB", (100, 100))
    rows = (len(images) + cols - 1) // cols
    W = images[0].width
    H = images[0].height
    grid_w = cols * W + (cols + 1) * pad
    grid_h = rows * H + (rows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), bg)
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        x = pad + col * (W + pad)
        y = pad + row * (H + pad)
        canvas.paste(img, (x, y))
    return canvas


def resize_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


# ── Generation ────────────────────────────────────────────────────────────────

def gen_sd3_figures(prompts, out_root, skip_gen=False):
    from src.models.sd3_wrapper import SD3DiTWrapper
    wrapper = SD3DiTWrapper(dtype="bf16")

    orig_build_mask = _install_mask_capture_hook()

    for tag, prompt in prompts:
        d = os.path.join(out_root, "sd3", tag)
        os.makedirs(d, exist_ok=True)

        baseline_path = os.path.join(d, "baseline.png")
        sasd_path     = os.path.join(d, "sasd.png")
        heatmap_path  = os.path.join(d, "heatmap.png")
        overlay_path  = os.path.join(d, "mask_overlay.png")
        mask_path     = os.path.join(d, "binary_mask.png")

        if not skip_gen or not os.path.exists(baseline_path):
            print(f"  [SD3/{tag}] baseline...")
            img_base = wrapper.generate(prompt, seed=SEED, steps=SD3_STEPS)
            img_base.save(baseline_path)
        else:
            img_base = Image.open(baseline_path)

        _mask_capture.clear()
        if not skip_gen or not os.path.exists(sasd_path):
            print(f"  [SD3/{tag}] sasd...")
            img_sasd = wrapper.generate_accelerated(
                prompt, seed=SEED, steps=SD3_STEPS,
                mask_type="semantic", s_fg=9.0, s_bg=2.0,
                full_skip_interval=2, n_segments=64,
                mask_step=5, skip_ratio=0.5,
            )
            img_sasd.save(sasd_path)
        else:
            # Still need to run to capture mask
            print(f"  [SD3/{tag}] re-running for mask capture...")
            img_sasd = wrapper.generate_accelerated(
                prompt, seed=SEED, steps=SD3_STEPS,
                mask_type="semantic", s_fg=9.0, s_bg=2.0,
                full_skip_interval=2, n_segments=64,
                mask_step=5, skip_ratio=0.5,
            )
            img_sasd.save(sasd_path)

        # Save mask visualizations
        if "raw_importance" in _mask_capture:
            W, H = img_base.size
            imp = _mask_capture["raw_importance"]
            mask = _mask_capture["binary_mask"]

            heatmap = importance_to_heatmap(imp, (W, H))
            heatmap.save(heatmap_path)

            overlay = mask_to_overlay(img_base, mask)
            overlay.save(overlay_path)

            mask_img = Image.fromarray(
                (mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
            mask_img.save(mask_path)
            print(f"  [SD3/{tag}] saved heatmap + mask overlay.")

        gc.collect()
        torch.cuda.empty_cache()

    _remove_mask_capture_hook(orig_build_mask)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()


def gen_flux_figures(prompts, out_root, skip_gen=False):
    from src.models.flux_wrapper import FluxDiTWrapper
    wrapper = FluxDiTWrapper(dtype="bf16")

    for tag, prompt in prompts:
        d = os.path.join(out_root, "flux", tag)
        os.makedirs(d, exist_ok=True)

        baseline_path = os.path.join(d, "baseline.png")
        sasd_path     = os.path.join(d, "sasd.png")

        if not skip_gen or not os.path.exists(baseline_path):
            print(f"  [FLUX/{tag}] baseline...")
            img_base = wrapper.generate(prompt, seed=SEED, steps=FLUX_STEPS)
            img_base.save(baseline_path)

        if not skip_gen or not os.path.exists(sasd_path):
            print(f"  [FLUX/{tag}] sasd...")
            img_sasd = wrapper.generate_accelerated(
                prompt, seed=SEED, steps=FLUX_STEPS,
                full_skip_interval=2, warmup_steps=5,
            )
            img_sasd.save(sasd_path)

        gc.collect()
        torch.cuda.empty_cache()

    del wrapper
    gc.collect()
    torch.cuda.empty_cache()


# ── Grid assembly ─────────────────────────────────────────────────────────────

def build_motivation_grid(prompts, out_root, cell=384):
    """
    Motivation figure: 3 prompts × 4 panels
      [Original | Attention Heatmap | Semantic Mask | SASD Result]
    """
    print("\nBuilding motivation grid...")
    panels = []
    col_labels = ["Original", "Attn Heatmap", "Semantic Mask", "SASD (2.09×)"]

    for tag, prompt in prompts:
        d = os.path.join(out_root, "sd3", tag)
        paths = {
            "baseline":     os.path.join(d, "baseline.png"),
            "heatmap":      os.path.join(d, "heatmap.png"),
            "mask_overlay": os.path.join(d, "mask_overlay.png"),
            "sasd":         os.path.join(d, "sasd.png"),
        }
        for key in paths:
            if os.path.exists(paths[key]):
                img = resize_square(Image.open(paths[key]).convert("RGB"), cell)
            else:
                img = Image.new("RGB", (cell, cell), (200, 200, 200))
            panels.append(img)

    grid = make_grid(panels, cols=4, pad=3)

    # Add column headers
    header_h = 30
    final = Image.new("RGB", (grid.width, grid.height + header_h), (255, 255, 255))
    final.paste(grid, (0, header_h))
    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    col_w = (grid.width - 3) // 4
    for ci, label in enumerate(col_labels):
        x = 3 + ci * (col_w + 3) + col_w // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 5), label, fill=(30, 30, 30), font=font)

    out = os.path.join(out_root, "grids", "motivation_sd3.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    final.save(out)
    print(f"  Saved → {out}  ({final.width}×{final.height})")
    return final


def build_compare_grid(prompts, model: str, out_root, cell=384):
    """
    Qualitative comparison: N prompts × [Baseline | SASD]
    """
    print(f"\nBuilding compare grid ({model})...")
    panels = []
    for tag, prompt in prompts:
        d = os.path.join(out_root, model, tag)
        for name in ["baseline", "sasd"]:
            p = os.path.join(d, f"{name}.png")
            if os.path.exists(p):
                img = resize_square(Image.open(p).convert("RGB"), cell)
            else:
                img = Image.new("RGB", (cell, cell), (200, 200, 200))
            label = "Baseline" if name == "baseline" else "SASD (ours)"
            panels.append(add_label(img, label, font_size=16))

    grid = make_grid(panels, cols=2, pad=3)
    out = os.path.join(out_root, "grids", f"compare_{model}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    grid.save(out)
    print(f"  Saved → {out}  ({grid.width}×{grid.height})")
    return grid


def build_full_compare_grid(prompts, out_root, cell=256):
    """
    Full multi-model grid: N prompts × [Baseline SD3 | SASD SD3 | Baseline FLUX | SASD FLUX]
    """
    print("\nBuilding full multi-model compare grid...")
    panels = []
    col_labels = ["SD3 Baseline", "SD3 SASD (ours)", "FLUX Baseline", "FLUX SASD (ours)"]

    for tag, prompt in prompts:
        for model in ["sd3", "flux"]:
            d = os.path.join(out_root, model, tag)
            for name in ["baseline", "sasd"]:
                p = os.path.join(d, f"{name}.png")
                if os.path.exists(p):
                    img = resize_square(Image.open(p).convert("RGB"), cell)
                else:
                    img = Image.new("RGB", (cell, cell), (200, 200, 200))
                panels.append(img)

    grid = make_grid(panels, cols=4, pad=3)

    # Column headers
    header_h = 28
    final = Image.new("RGB", (grid.width, grid.height + header_h), (255, 255, 255))
    final.paste(grid, (0, header_h))
    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    col_w = (grid.width - 3) // 4
    for ci, label in enumerate(col_labels):
        x = 3 + ci * (col_w + 3) + col_w // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 5), label, fill=(30, 30, 30), font=font)

    out = os.path.join(out_root, "grids", "compare_all_models.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    final.save(out)
    print(f"  Saved → {out}  ({final.width}×{final.height})")
    return final


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["sd3", "flux", "all"], default="all")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip generation (images already exist), just rebuild grids")
    parser.add_argument("--cell",     type=int, default=384,
                        help="Cell size in pixels for grid images")
    parser.add_argument("--prompts_n", type=int, default=None,
                        help="Only use first N prompts (default: all)")
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    prompts = PROMPTS[:args.prompts_n] if args.prompts_n else PROMPTS
    motivation_prompts = [(t, p) for t, p in prompts if t in MOTIVATION_TAGS]

    print(f"Paper figures: {len(prompts)} prompts, model={args.model}")
    print(f"Output: {os.path.abspath(OUT_ROOT)}")

    # ── Generation phase ──────────────────────────────────────────────────────
    if args.model in ("sd3", "all"):
        print("\n=== SD3 generation ===")
        gen_sd3_figures(prompts, OUT_ROOT, skip_gen=args.skip_gen)

    if args.model in ("flux", "all"):
        print("\n=== FLUX generation ===")
        gen_flux_figures(prompts, OUT_ROOT, skip_gen=args.skip_gen)

    # ── Grid assembly ─────────────────────────────────────────────────────────
    print("\n=== Building paper grids ===")

    if args.model in ("sd3", "all") and motivation_prompts:
        build_motivation_grid(motivation_prompts, OUT_ROOT, cell=args.cell)

    if args.model in ("sd3", "all"):
        build_compare_grid(prompts, "sd3", OUT_ROOT, cell=args.cell)

    if args.model in ("flux", "all"):
        build_compare_grid(prompts, "flux", OUT_ROOT, cell=args.cell)

    if args.model == "all":
        build_full_compare_grid(prompts, OUT_ROOT, cell=min(args.cell, 256))

    print(f"\n=== All done! Files in {os.path.abspath(OUT_ROOT)} ===")
    print("\nFiles:")
    for root, dirs, files in os.walk(os.path.join(OUT_ROOT, "grids")):
        for f in sorted(files):
            fp = os.path.join(root, f)
            size_kb = os.path.getsize(fp) // 1024
            print(f"  {fp}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
