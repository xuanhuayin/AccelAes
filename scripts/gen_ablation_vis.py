#!/usr/bin/env python3
"""
Generate vis_ablation_c1.pdf and vis_ablation_c2.pdf
in vis_compare.pdf style (columns=configs, rows=prompts, red border on AccelAes).

C.1: SkipSparse ablation — outputs/p0_ablation_direct/vis_ablation_c1.pdf
C.2: AesMask ablation   — outputs/anchor_ablation/vis_ablation_c2.pdf
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Layout constants (match gen_sensitivity_vis.py) ─────────────────────────
IMG_SIZE   = 256
GAP        = 2
HEADER_H   = 52
LEFT_PAD   = 4
TOP_PAD    = 4
RIGHT_PAD  = 4
BOT_PAD    = 4

BG_COLOR   = (255, 255, 255)
GRID_COLOR = (220, 220, 220)
HL_COLOR   = (220, 30, 30)
HL_WIDTH   = 3

TEXT_COLOR      = (20, 20, 20)
TEXT_COLOR_SUB  = (80, 80, 80)
TEXT_COLOR_BOLD = (0, 0, 0)

PROMPT_IDS = [0, 1, 2, 3, 4]   # 5 aesthetic vis prompts


def get_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.otf",
        "/usr/share/fonts/truetype/freefont/FreeSans.otf",
    ]
    if bold:
        candidates = [c for c in candidates if "Bold" in c or "bold" in c] + candidates
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_grid(configs, prompt_ids, default_key, title_line1_fn, title_line2_fn, out_path):
    n_cols = len(configs)
    n_rows = len(prompt_ids)

    total_w = LEFT_PAD + n_cols * IMG_SIZE + (n_cols - 1) * GAP + RIGHT_PAD
    total_h = TOP_PAD + HEADER_H + n_rows * IMG_SIZE + (n_rows - 1) * GAP + BOT_PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font_bold  = get_font(14, bold=True)
    font_small = get_font(11, bold=False)
    font_tiny  = get_font(10, bold=False)

    for ci, (key, img_dir) in enumerate(configs):
        x0 = LEFT_PAD + ci * (IMG_SIZE + GAP)
        is_default = (key == default_key)

        line1 = title_line1_fn(key)
        line2 = title_line2_fn(key)

        hdr_color = (255, 240, 240) if is_default else (245, 245, 245)
        draw.rectangle([x0, TOP_PAD, x0 + IMG_SIZE - 1, TOP_PAD + HEADER_H - 1],
                       fill=hdr_color)

        try:
            bbox1 = font_bold.getbbox(line1)
            tw1 = bbox1[2] - bbox1[0]
        except Exception:
            tw1 = len(line1) * 8
        tx1 = x0 + (IMG_SIZE - tw1) // 2
        draw.text((tx1, TOP_PAD + 5), line1, font=font_bold,
                  fill=HL_COLOR if is_default else TEXT_COLOR_BOLD)

        try:
            bbox2 = font_small.getbbox(line2)
            tw2 = bbox2[2] - bbox2[0]
        except Exception:
            tw2 = len(line2) * 7
        tx2 = x0 + (IMG_SIZE - tw2) // 2
        draw.text((tx2, TOP_PAD + 24), line2, font=font_small,
                  fill=TEXT_COLOR_SUB)

        draw.line([(x0, TOP_PAD + HEADER_H - 1),
                   (x0 + IMG_SIZE - 1, TOP_PAD + HEADER_H - 1)],
                  fill=GRID_COLOR, width=1)

        for ri, pid in enumerate(prompt_ids):
            y0 = TOP_PAD + HEADER_H + ri * (IMG_SIZE + GAP)

            img_path = os.path.join(img_dir, f"p{pid:04d}_s0000.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            else:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (180, 180, 180))
                d = ImageDraw.Draw(img)
                d.text((10, IMG_SIZE // 2 - 8), f"MISSING\n{os.path.basename(img_path)}",
                       fill=(80, 80, 80), font=font_tiny)

            canvas.paste(img, (x0, y0))

            if is_default:
                for w in range(HL_WIDTH):
                    draw.rectangle(
                        [x0 + w, y0 + w,
                         x0 + IMG_SIZE - 1 - w, y0 + IMG_SIZE - 1 - w],
                        outline=HL_COLOR
                    )

    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(180, 180, 180), width=1)

    canvas.save(out_path, "PDF", resolution=150)
    print(f"  Saved → {out_path}  ({total_w}×{total_h}px)")
    return canvas


# ── C.1 SkipSparse Ablation ───────────────────────────────────────────────

def make_c1():
    print("\n[1/2] C.1 SkipSparse ablation figure")
    with open("outputs/p0_ablation_direct/summary.json") as f:
        stats = json.load(f)

    BASE = "outputs/p0_ablation_direct"
    configs = [
        ("baseline",  os.path.join(BASE, "baseline",  "vis_images")),
        ("fskip_only", os.path.join(BASE, "fskip_only", "vis_images")),
        ("attn_only",  os.path.join(BASE, "attn_only",  "vis_images")),
        ("attn_ffn",   os.path.join(BASE, "attn_ffn",   "vis_images")),
        ("sasd_full",  os.path.join(BASE, "sasd_full",  "vis_images")),
    ]

    labels = {
        "baseline":  "Baseline",
        "fskip_only": "StepSkip only",
        "attn_only":  "Sparse Attn",
        "attn_ffn":   "Sparse Attn+FFN",
        "sasd_full":  "AccelAes ★",
    }

    def line1(key):
        return labels[key]

    def line2(key):
        s = stats[key]
        sp = s.get("speedup") or 1.0
        ir = s.get("mean_ir") or 0.0
        return f"IR={ir:.3f}  {sp:.2f}×"

    make_grid(
        configs, PROMPT_IDS, "sasd_full",
        line1, line2,
        out_path=os.path.join(BASE, "vis_ablation_c1.pdf"),
    )


# ── C.2 AesMask Ablation ─────────────────────────────────────────────────

def make_c2():
    print("\n[2/2] C.2 AesMask ablation figure")
    with open("outputs/anchor_ablation/summary.json") as f:
        stats = json.load(f)

    BASE = "outputs/anchor_ablation"
    configs = [
        ("sasd_nonaesthetic", os.path.join(BASE, "sasd_nonaesthetic", "vis_images")),
        ("sasd_no_anchor",    os.path.join(BASE, "sasd_no_anchor",    "vis_images")),
        ("sasd_with_anchor",  os.path.join(BASE, "sasd_with_anchor",  "vis_images")),
    ]

    labels = {
        "sasd_nonaesthetic": "Non-aesthetic weights",
        "sasd_no_anchor":    "Uniform weights",
        "sasd_with_anchor":  "AesMask ★",
    }

    # sasd_nonaesthetic not in summary (no separate eval); reuse sasd_no_anchor IR for display
    def line2(key):
        if key in stats:
            s = stats[key]
            ir = s.get("mean_ir") or 0.0
            lp = s.get("mean_lpips") or 0.0
            return f"IR={ir:.3f}  LPIPS={lp:.3f}"
        else:
            return ""  # sasd_nonaesthetic has no numeric entry

    def line1(key):
        return labels[key]

    make_grid(
        configs, PROMPT_IDS, "sasd_with_anchor",
        line1, line2,
        out_path=os.path.join(BASE, "vis_ablation_c2.pdf"),
    )


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_c1()
    make_c2()
    print("\nDone.")
