#!/usr/bin/env python3
"""
AccelAes Lumina 可视化：先批量评分，再选 ΔIR Top-K 做带注释的可视化。

流程：
  Phase 1 — 生成所有 prompt 的 baseline + AccelAes 图，计算 ImageReward，保存 scores.json
  Phase 2 — 按 ΔIR 降序选 TOP_K，生成 5-panel 可视化，面板上标注 IR 分数

输出：
  outputs/accelae_topk/
    lumina/{tag}/  — baseline.png, heatmap.png, mask.png, overlay.png, result.png
    scores.json    — {tag: {ir_base, ir_result, delta_ir}}
    topk_grid.png  — Top-K 合并大图（带 IR 标注）
    rows/{tag}_row.png

用法：
  python scripts/gen_lumina_topk_figures.py
  python scripts/gen_lumina_topk_figures.py --skip_gen      # 重用已有图，只重评分/重拼
  python scripts/gen_lumina_topk_figures.py --skip_score    # 重用 scores.json，只重拼
  python scripts/gen_lumina_topk_figures.py --top_k 8
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

# ── Prompts （24 个，覆盖不同主体，均含美学关键词） ──────────────────────────

PROMPTS = [
    # 动物特写
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

    # 人物特写
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

    # 奇幻生物
    ("dragon",      "An ancient dragon with intricate scales perched on a mountain peak, "
                    "dramatic lighting, cinematic fantasy art"),
    ("phoenix",     "A phoenix rising from flames with intricate golden feather details, "
                    "dramatic lighting, photorealistic, vivid colors"),
    ("mermaid",     "A mermaid with intricate iridescent scales resting on sea rocks, "
                    "golden hour light, photorealistic, cinematic, beautiful"),

    # 机械/科幻
    ("mech",        "A steampunk engineer with intricate brass gears and clockwork prosthetic arm, "
                    "photorealistic, cinematic, detailed"),
    ("cyborg",      "Portrait of a female cyborg with intricate circuit patterns on face, "
                    "neon lighting, photorealistic, cinematic, sharp focus"),

    # 自然微距
    ("flowers",     "Macro photograph of orchids with morning dew drops, "
                    "photorealistic, intricate petals, soft bokeh"),
    ("butterfly",   "Macro photograph of a morpho butterfly with intricate wing patterns, "
                    "vivid blue iridescence, photorealistic, soft bokeh"),
    ("mushroom",    "Macro photograph of intricate glowing mushrooms in a dark forest, "
                    "photorealistic, cinematic, vivid colors, beautiful"),

    # 建筑/室内
    ("cathedral",   "Interior of a gothic cathedral with intricate stained glass windows, "
                    "dramatic light beams, photorealistic, cinematic"),
    ("palace",      "Grand baroque palace interior with intricate golden decorations, "
                    "dramatic lighting, photorealistic, detailed architecture"),

    # 食物/产品（美学高对比度）
    ("cake",        "An intricate three-tier wedding cake with detailed sugar flowers, "
                    "studio lighting, photorealistic, sharp focus, beautiful"),
    ("jewelry",     "Macro photograph of intricate diamond and sapphire jewelry, "
                    "studio lighting, photorealistic, vivid colors, sharp focus"),

    # 运动/动态
    ("dancer",      "A ballet dancer in intricate tutu performing a leap, "
                    "stage lighting, photorealistic, cinematic, sharp focus, elegant"),
    ("astronaut",   "Portrait of an astronaut with intricate spacesuit details, "
                    "dramatic space backdrop, photorealistic, cinematic"),
]

SEED        = 42
STEPS       = 30
OUT_ROOT    = "outputs/accelae_topk"
CELL        = 384
TOP_K       = 6   # 最终可视化选多少个

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

COL_LABELS = [
    "Baseline",
    "Attn Heatmap",
    "Semantic Mask",
    "Mask Overlay",
    "AccelAes (ours)",
]

# ── Mask capture hook ─────────────────────────────────────────────────────────

_cap = {}


def install_hook():
    from src.sparse.mask_builders import SemanticMaskBuilder
    _orig = SemanticMaskBuilder.build_mask_from_cross_attention

    def _patched(self, cross_attn_maps, token_importance, latent_h, latent_w):
        if cross_attn_maps:
            num_patches = latent_h * latent_w
            all_attn = torch.stack([m.mean(dim=0) for m in cross_attn_maps])
            avg_attn = all_attn.mean(dim=0)
            weighted = avg_attn * token_importance.unsqueeze(0)
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


# ── ImageReward scorer ────────────────────────────────────────────────────────

_ir_model = None


def get_ir_model():
    global _ir_model
    if _ir_model is None:
        import ImageReward as IR
        _ir_model = IR.load("ImageReward-v1.0")
    return _ir_model


def score_ir(prompt: str, image: Image.Image) -> float:
    model = get_ir_model()
    with torch.no_grad():
        score = model.score(prompt, image)
    return float(score)


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
    """Bottom label bar with text."""
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
    """Label bar with IR score annotation.

    For baseline: 'Baseline  IR: 0.72'
    For result:   'AccelAes (ours)  IR: 0.94 (+30.6%)'
    """
    bar = fsize + 12
    out = Image.new("RGB", (img.width, img.height + bar), bg)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font = _load_font(fsize)
    font_sm = _load_font(max(fsize - 4, 12))

    # Build text
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        score_text = f"IR: {ir_score:.3f} ({sign}{delta:.1%})"
    else:
        score_text = f"IR: {ir_score:.3f}"

    # Draw label (centered, white)
    label_bbox = draw.textbbox((0, 0), label, font=font)
    lw = label_bbox[2] - label_bbox[0]
    draw.text(((img.width - lw) // 2, img.height + 5), label,
              fill=(240, 240, 240), font=font)

    # Draw score (right-aligned, yellow)
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


# ── Phase 1: Generate images ─────────────────────────────────────────────────

def gen_all(prompts, skip_gen=False):
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

        # AccelAes
        print("  AccelAes...")
        img_result = wrapper.generate_accelerated_dual(
            prompt, seed=SEED, steps=STEPS, **SASD_KWARGS
        )
        img_result.save(result_path)

        # Mask artifacts
        if "imp" in _cap:
            W, H = img_base.size
            imp = _cap["imp"]
            m = _cap["mask"] if "mask" in _cap else (imp >= np.percentile(imp, 50)).astype(np.float32)
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


# ── Phase 2: Score with ImageReward ──────────────────────────────────────────

def score_all(prompts, scores_path):
    scores = {}
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)

    changed = False
    for tag, prompt in prompts:
        d = os.path.join(OUT_ROOT, "lumina", tag)
        base_path   = os.path.join(d, "baseline.png")
        result_path = os.path.join(d, "result.png")

        if not os.path.exists(base_path) or not os.path.exists(result_path):
            print(f"  [{tag}] images missing, skip")
            continue

        if tag in scores:
            print(f"  [{tag}] cached: IR_base={scores[tag]['ir_base']:.4f}  "
                  f"IR_result={scores[tag]['ir_result']:.4f}  "
                  f"delta={scores[tag]['delta_ir']:.4f}")
            continue

        print(f"  [{tag}] scoring...")
        img_base   = Image.open(base_path).convert("RGB")
        img_result = Image.open(result_path).convert("RGB")
        ir_base   = score_ir(prompt, img_base)
        ir_result = score_ir(prompt, img_result)
        delta = ir_result - ir_base

        scores[tag] = {"ir_base": ir_base, "ir_result": ir_result, "delta_ir": delta,
                       "prompt": prompt}
        print(f"    IR_base={ir_base:.4f}  IR_result={ir_result:.4f}  Δ={delta:+.4f}")
        changed = True

    if changed:
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"  Scores saved → {scores_path}")

    return scores


# ── Phase 3: Select Top-K and build annotated grid ───────────────────────────

def build_topk_grid(prompts, scores, top_k=6, cell=384):
    prompt_map = {t: p for t, p in prompts}

    # Sort by delta_ir descending
    ranked = sorted(scores.items(), key=lambda x: x[1]["delta_ir"], reverse=True)
    top = ranked[:top_k]

    print(f"\nTop-{top_k} by ΔIR:")
    for i, (tag, s) in enumerate(top):
        print(f"  #{i+1} {tag:12s}  IR {s['ir_base']:.3f} → {s['ir_result']:.3f}  "
              f"Δ={s['delta_ir']:+.4f} ({s['delta_ir']/abs(s['ir_base']):.1%})")

    rows_dir = os.path.join(OUT_ROOT, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    all_rows = []

    for tag, s in top:
        d = os.path.join(OUT_ROOT, "lumina", tag)
        ir_base   = s["ir_base"]
        ir_result = s["ir_result"]
        delta_rel = (ir_result - ir_base) / (abs(ir_base) + 1e-8)

        panels = []
        for idx, (key, label) in enumerate(zip(
            ["baseline", "heatmap", "mask", "overlay", "result"], COL_LABELS
        )):
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
        print(f"  Row saved → {row_path}  ({row.width}×{row.height})")

    # Full grid
    grid = vstack(all_rows, pad=6, bg=(210, 210, 210))
    out = os.path.join(OUT_ROOT, "topk_grid.png")
    grid.save(out)
    size_kb = os.path.getsize(out) // 1024
    print(f"\nTop-{top_k} grid → {out}  ({grid.width}×{grid.height}, {size_kb} KB)")
    return grid


# ── Also dump full ranking table ─────────────────────────────────────────────

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
    parser.add_argument("--skip_gen",   action="store_true",
                        help="Skip generation, reuse existing images")
    parser.add_argument("--skip_score", action="store_true",
                        help="Skip scoring, reuse existing scores.json")
    parser.add_argument("--prompts_n",  type=int, default=None)
    parser.add_argument("--top_k",      type=int, default=TOP_K)
    parser.add_argument("--cell",       type=int, default=CELL)
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    prompts = PROMPTS[:args.prompts_n] if args.prompts_n else PROMPTS
    scores_path = os.path.join(OUT_ROOT, "scores.json")

    print(f"AccelAes Top-K figures: {len(prompts)} prompts, top_k={args.top_k}, cell={args.cell}")
    print(f"Output: {os.path.abspath(OUT_ROOT)}")

    # Phase 1: Generate
    if not args.skip_gen:
        print(f"\n=== Phase 1: Generating {len(prompts)} × 2 images (Lumina) ===")
        gen_all(prompts, skip_gen=False)
    else:
        print("\n=== Phase 1: Skipped (--skip_gen) ===")

    # Phase 2: Score
    if not args.skip_score:
        print(f"\n=== Phase 2: ImageReward scoring ===")
        scores = score_all(prompts, scores_path)
    else:
        print(f"\n=== Phase 2: Loading cached scores from {scores_path} ===")
        with open(scores_path) as f:
            scores = json.load(f)

    if not scores:
        print("No scores available, exiting.")
        return

    print_ranking(scores)

    # Phase 3: Build Top-K grid
    print(f"\n=== Phase 3: Building Top-{args.top_k} annotated grid ===")
    build_topk_grid(prompts, scores, top_k=args.top_k, cell=args.cell)

    print("\nDone.")


if __name__ == "__main__":
    main()
