#!/usr/bin/env python3
"""
AccelAes FLUX.1-dev 可视化：批量评分，选 ΔIR Top-K 做带注释可视化。

FLUX 加速组件：
  - Semantic sparse attention (single-stream blocks)
  - Step-level caching (full_skip_interval=2)
  - 无 spatial CFG（FLUX 单 pass guidance distillation，不走双路 CFG）

流程：
  Phase 1 — 生成所有 prompt 的 baseline + AccelAes，保存图片和 mask artifacts
  Phase 2 — ImageReward 评分，保存 scores.json
  Phase 3 — 按 ΔIR 选 Top-K，生成带 IR 标注的 5-panel 可视化

输出：
  outputs/accelae_topk_flux/
    flux/{tag}/  — baseline.png, heatmap.png, mask.png, overlay.png, result.png
    scores.json
    topk_grid.png
    rows/{tag}_row.png

用法：
  python scripts/gen_flux_topk_figures.py
  python scripts/gen_flux_topk_figures.py --skip_gen
  python scripts/gen_flux_topk_figures.py --skip_score
  python scripts/gen_flux_topk_figures.py --top_k 6
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── 与 Lumina 版本相同的 24 个 prompt ──────────────────────────────────────

PROMPTS = [
    ("lion",        "A majestic lion at golden hour in the savanna, photorealistic, "
                    "cinematic lighting, intricate fur detail"),
    ("tiger",       "Close up portrait of a Bengal tiger with intricate fur patterns, "
                    "golden hour light, photorealistic, cinematic, vivid colors"),
    ("wolf",        "A silver wolf howling under moonlight, intricate fur and sharp claws, "
                    "photorealistic, cinematic, dramatic contrast"),
    ("eagle",       "A bald eagle in flight with intricate feather details, "
                    "dramatic sky backdrop, photorealistic, cinematic"),
    ("peacock",     "A peacock displaying its intricate iridescent plumage, "
                    "garden setting, photorealistic, vivid colors, sharp focus"),
    ("warrior",     "A samurai warrior in intricate armor standing in a bamboo forest, "
                    "cinematic lighting, photorealistic, sharp focus"),
    ("knight",      "A medieval knight in intricate engraved full plate armor, "
                    "dramatic side lighting, photorealistic, dark stone background"),
    ("portrait",    "Close up portrait of an elegant woman with intricate floral headpiece, "
                    "photorealistic, studio lighting, detailed skin texture"),
    ("bride",       "Portrait of a bride in intricate lace wedding gown, "
                    "soft bokeh garden background, photorealistic, cinematic, beautiful"),
    ("geisha",      "A geisha in intricate traditional kimono with detailed embroidery, "
                    "cherry blossom background, photorealistic, cinematic"),
    ("dragon",      "An ancient dragon with intricate scales perched on a mountain peak, "
                    "dramatic lighting, cinematic fantasy art"),
    ("phoenix",     "A phoenix rising from flames with intricate golden feather details, "
                    "dramatic lighting, photorealistic, vivid colors"),
    ("mermaid",     "A mermaid with intricate iridescent scales resting on sea rocks, "
                    "golden hour light, photorealistic, cinematic, beautiful"),
    ("mech",        "A steampunk engineer with intricate brass gears and clockwork prosthetic arm, "
                    "photorealistic, cinematic, detailed"),
    ("cyborg",      "Portrait of a female cyborg with intricate circuit patterns on face, "
                    "neon lighting, photorealistic, cinematic, sharp focus"),
    ("flowers",     "Macro photograph of orchids with morning dew drops, "
                    "photorealistic, intricate petals, soft bokeh"),
    ("butterfly",   "Macro photograph of a morpho butterfly with intricate wing patterns, "
                    "vivid blue iridescence, photorealistic, soft bokeh"),
    ("mushroom",    "Macro photograph of intricate glowing mushrooms in a dark forest, "
                    "photorealistic, cinematic, vivid colors, beautiful"),
    ("cathedral",   "Interior of a gothic cathedral with intricate stained glass windows, "
                    "dramatic light beams, photorealistic, cinematic"),
    ("palace",      "Grand baroque palace interior with intricate golden decorations, "
                    "dramatic lighting, photorealistic, detailed architecture"),
    ("cake",        "An intricate three-tier wedding cake with detailed sugar flowers, "
                    "studio lighting, photorealistic, sharp focus, beautiful"),
    ("jewelry",     "Macro photograph of intricate diamond and sapphire jewelry, "
                    "studio lighting, photorealistic, vivid colors, sharp focus"),
    ("dancer",      "A ballet dancer in intricate tutu performing a leap, "
                    "stage lighting, photorealistic, cinematic, sharp focus, elegant"),
    ("astronaut",   "Portrait of an astronaut with intricate spacesuit details, "
                    "dramatic space backdrop, photorealistic, cinematic"),
]

SEED      = 42
STEPS     = 28
OUT_ROOT  = "outputs/accelae_topk_flux"
CELL      = 384
TOP_K     = 6

FLUX_ACCEL_KWARGS = dict(
    skip_ratio=0.5,
    sparse_blocks=True,
    full_skip_interval=2,
    warmup_steps=5,
    mask_step=8,          # 28.6% of 28 steps; must be a compute step (not skip)
    n_segments=64,
    region_method="threshold",  # direct percentile, consistent with Lumina
)

COL_LABELS = [
    "Baseline",
    "Attn Heatmap",
    "Semantic Mask",
    "Mask Overlay",
    "AccelAes (ours)",
]

# ── Mask capture hook (SD3SemanticMaskBuilder.build_mask) ─────────────────────

_cap = {}


def install_hook():
    from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
    _orig = SD3SemanticMaskBuilder.build_mask

    def _patched(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
        if affinity_maps:
            stacked = torch.stack(affinity_maps, dim=0)
            B = stacked.shape[1]
            stacked = stacked[:, B // 2:, :, :, :]
            importance = stacked.mean(dim=[0, 1, 2])  # (N_img, N_txt)
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


# ── ImageReward scorer ────────────────────────────────────────────────────────

_ir_model = None


def get_ir_model():
    global _ir_model
    if _ir_model is None:
        import ImageReward as IR
        _ir_model = IR.load("ImageReward-v1.0")
    return _ir_model


def score_ir(prompt: str, image: Image.Image) -> float:
    with torch.no_grad():
        return float(get_ir_model().score(prompt, image))


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


def _load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_label(img, text, fsize=20, bg=(20, 20, 20), fg_color=(240, 240, 240)):
    bar = fsize + 10
    out = Image.new("RGB", (img.width, img.height + bar), bg)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font = _load_font(fsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 5), text, fill=fg_color, font=font)
    return out


def add_label_with_score(img, label, ir_score, delta=None, fsize=20,
                          bg=(20, 20, 20), score_color=(255, 220, 60)):
    bar = fsize + 12
    out = Image.new("RGB", (img.width, img.height + bar), bg)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font    = _load_font(fsize)
    font_sm = _load_font(max(fsize - 4, 12))

    if delta is not None:
        sign = "+" if delta >= 0 else ""
        score_text = f"IR: {ir_score:.3f} ({sign}{delta:.1%})"
    else:
        score_text = f"IR: {ir_score:.3f}"

    label_bbox = draw.textbbox((0, 0), label, font=font)
    lw = label_bbox[2] - label_bbox[0]
    draw.text(((img.width - lw) // 2, img.height + 5), label,
              fill=(240, 240, 240), font=font)

    score_bbox = draw.textbbox((0, 0), score_text, font=font_sm)
    sw = score_bbox[2] - score_bbox[0]
    draw.text((img.width - sw - 6, img.height + 7), score_text,
              fill=score_color, font=font_sm)
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


# ── Phase 1: Generate ─────────────────────────────────────────────────────────

def gen_all(prompts):
    from src.models.flux_wrapper import FluxDiTWrapper
    wrapper = FluxDiTWrapper()
    orig = install_hook()

    for tag, prompt in prompts:
        d = os.path.join(OUT_ROOT, "flux", tag)
        os.makedirs(d, exist_ok=True)
        print(f"\n[{tag}]")
        _cap.clear()

        base_path   = os.path.join(d, "baseline.png")
        result_path = os.path.join(d, "result.png")

        if not os.path.exists(base_path):
            print("  baseline...")
            img_base = wrapper.generate(prompt, seed=SEED, steps=STEPS)
            img_base.save(base_path)
        else:
            img_base = Image.open(base_path).convert("RGB")
            print("  baseline (cached)")

        print("  AccelAes (mask_step=7, threshold)...")
        img_result = wrapper.generate_accelerated_sparse(
            prompt, seed=SEED, steps=STEPS, **FLUX_ACCEL_KWARGS
        )
        img_result.save(result_path)

        # Mask artifacts
        if "imp" in _cap:
            W, H = img_base.size
            imp = _cap["imp"]
            m = _cap["mask"]
            jet(imp, (W, H)).save(os.path.join(d, "heatmap.png"))
            mask_binary_img(m, (W, H)).save(os.path.join(d, "mask.png"))
            overlay_img(img_base, m).save(os.path.join(d, "overlay.png"))
            print(f"  mask saved → {d}/")
        else:
            print("  WARNING: mask not captured")

        gc.collect()
        torch.cuda.empty_cache()

    remove_hook(orig)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()


# ── Phase 2: Score ────────────────────────────────────────────────────────────

def score_all(prompts, scores_path):
    scores = {}
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)

    changed = False
    for tag, prompt in prompts:
        d = os.path.join(OUT_ROOT, "flux", tag)
        base_path   = os.path.join(d, "baseline.png")
        result_path = os.path.join(d, "result.png")

        if not os.path.exists(base_path) or not os.path.exists(result_path):
            print(f"  [{tag}] images missing, skip")
            continue

        if tag in scores:
            print(f"  [{tag}] cached  IR_base={scores[tag]['ir_base']:.4f}  "
                  f"IR_result={scores[tag]['ir_result']:.4f}  "
                  f"delta={scores[tag]['delta_ir']:+.4f}")
            continue

        print(f"  [{tag}] scoring...")
        ir_base   = score_ir(prompt, Image.open(base_path).convert("RGB"))
        ir_result = score_ir(prompt, Image.open(result_path).convert("RGB"))
        delta = ir_result - ir_base
        scores[tag] = {"ir_base": ir_base, "ir_result": ir_result,
                       "delta_ir": delta, "prompt": prompt}
        print(f"    IR_base={ir_base:.4f}  IR_result={ir_result:.4f}  Δ={delta:+.4f}")
        changed = True

    if changed:
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"  Scores saved → {scores_path}")

    return scores


# ── Phase 3: Build annotated Top-K grid ──────────────────────────────────────

def build_topk_grid(prompts, scores, top_k=6, cell=384):
    ranked = sorted(scores.items(), key=lambda x: x[1]["delta_ir"], reverse=True)
    top = ranked[:top_k]

    print(f"\nTop-{top_k} by ΔIR:")
    for i, (tag, s) in enumerate(top):
        pct = s["delta_ir"] / (abs(s["ir_base"]) + 1e-8)
        print(f"  #{i+1} {tag:12s}  IR {s['ir_base']:.3f} → {s['ir_result']:.3f}  "
              f"Δ={s['delta_ir']:+.4f} ({pct:+.1%})")

    rows_dir = os.path.join(OUT_ROOT, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    all_rows = []

    for tag, s in top:
        d = os.path.join(OUT_ROOT, "flux", tag)
        ir_base   = s["ir_base"]
        ir_result = s["ir_result"]
        delta_rel = (ir_result - ir_base) / (abs(ir_base) + 1e-8)

        panels = []
        for key, label in zip(
            ["baseline", "heatmap", "mask", "overlay", "result"], COL_LABELS
        ):
            p = os.path.join(d, f"{key}.png")
            img = Image.open(p).convert("RGB") if os.path.exists(p) \
                else Image.new("RGB", (cell, cell), (180, 180, 180))
            img_rs = img.resize((cell, cell), Image.LANCZOS)

            if key == "baseline":
                panel = add_label_with_score(img_rs, label, ir_base, delta=None)
            elif key == "result":
                panel = add_label_with_score(img_rs, label, ir_result, delta=delta_rel)
            else:
                panel = add_label(img_rs, label)

            panels.append(panel)

        row = hstack(panels)
        row_path = os.path.join(rows_dir, f"{tag}_row.png")
        row.save(row_path)
        all_rows.append(row)
        print(f"  Row → {row_path}  ({row.width}×{row.height})")

    grid = vstack(all_rows, pad=6, bg=(210, 210, 210))
    out = os.path.join(OUT_ROOT, "topk_grid.png")
    grid.save(out)
    size_kb = os.path.getsize(out) // 1024
    print(f"\nTop-{top_k} grid → {out}  ({grid.width}×{grid.height}, {size_kb} KB)")
    return grid


def print_ranking(scores):
    ranked = sorted(scores.items(), key=lambda x: x[1]["delta_ir"], reverse=True)
    print("\n" + "=" * 60)
    print(f"{'Rank':<5} {'Tag':<12} {'IR_base':>8} {'IR_accel':>9} {'ΔIR':>8} {'Δ%':>7}")
    print("-" * 60)
    for i, (tag, s) in enumerate(ranked):
        pct = s["delta_ir"] / (abs(s["ir_base"]) + 1e-8)
        print(f"  {i+1:<4} {tag:<12} {s['ir_base']:8.4f} {s['ir_result']:9.4f} "
              f"{s['delta_ir']:+8.4f} {pct:+7.1%}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_gen",   action="store_true")
    parser.add_argument("--skip_score", action="store_true")
    parser.add_argument("--prompts_n",  type=int, default=None)
    parser.add_argument("--top_k",      type=int, default=TOP_K)
    parser.add_argument("--cell",       type=int, default=CELL)
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    prompts = PROMPTS[:args.prompts_n] if args.prompts_n else PROMPTS
    scores_path = os.path.join(OUT_ROOT, "scores.json")

    print(f"AccelAes FLUX Top-K figures: {len(prompts)} prompts, top_k={args.top_k}")
    print(f"Note: FLUX has NO spatial CFG — acceleration via sparse attn + step cache only")
    print(f"Output: {os.path.abspath(OUT_ROOT)}")

    if not args.skip_gen:
        print(f"\n=== Phase 1: Generating {len(prompts)} × 2 images (FLUX.1-dev) ===")
        gen_all(prompts)
    else:
        print("\n=== Phase 1: Skipped (--skip_gen) ===")

    if not args.skip_score:
        print(f"\n=== Phase 2: ImageReward scoring ===")
        scores = score_all(prompts, scores_path)
    else:
        print(f"\n=== Phase 2: Loading cached scores ===")
        with open(scores_path) as f:
            scores = json.load(f)

    if not scores:
        print("No scores available, exiting.")
        return

    print_ranking(scores)

    print(f"\n=== Phase 3: Building Top-{args.top_k} annotated grid ===")
    build_topk_grid(prompts, scores, top_k=args.top_k, cell=args.cell)

    print("\nDone.")


if __name__ == "__main__":
    main()
