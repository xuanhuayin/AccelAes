#!/usr/bin/env python3
"""
Generate vis_maskstep.pdf and vis_ratio.pdf under outputs/supp_sensitivity/
Format modelled after vis_compare.pdf:
  - columns = configs (method name + metric labels in header)
  - rows    = selected prompts (seed=0)
  - default/AccelAes column highlighted with red border
"""

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

BASE = "outputs/supp_sensitivity"
OUT_BASE = BASE

# ── Selected prompts (0-indexed from prompts_dev.txt, seed=0) ──────────────
# Exclude: 3,6,7,14 (in vis_compare/supp_qual), 5 (flower portrait used in FLUX fig),
#          10 (empty prompt), 11 (zentai), 13 (schoolgirl)
PROMPT_IDS = [0, 1, 2, 3, 4]   # 5 aesthetic vis prompts (vis_images/)

with open("prompts/prompts_dev.txt") as f:
    ALL_PROMPTS = [l.strip() for l in f if l.strip()]

# ── Layout constants ────────────────────────────────────────────────────────
IMG_SIZE   = 256          # display size (original 1024→256)
GAP        = 2            # gap between cells (px)
HEADER_H   = 52           # column header height (px)
LEFT_PAD   = 4            # left margin
TOP_PAD    = 4            # top margin
RIGHT_PAD  = 4
BOT_PAD    = 4

BG_COLOR   = (255, 255, 255)
GRID_COLOR = (220, 220, 220)
HL_COLOR   = (220, 30, 30)     # red highlight for default column
HL_WIDTH   = 3

TEXT_COLOR      = (20, 20, 20)
TEXT_COLOR_SUB  = (80, 80, 80)
TEXT_COLOR_BOLD = (0, 0, 0)

# Try to load a font; fall back to PIL default
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


# ── Core drawing helper ─────────────────────────────────────────────────────

def make_grid(configs, prompt_ids, default_key, title_line1_fn, title_line2_fn, out_path):
    """
    configs: list of (key, img_dir)  – ordered left to right
    prompt_ids: list of int
    default_key: the key of the default/AccelAes config (highlighted)
    title_line1_fn(key) -> str   – bold top line (e.g. "mask_step=5")
    title_line2_fn(key) -> str   – smaller bottom line (e.g. "IR=0.841  2.10×")
    """
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

        # ── Column header ──────────────────────────────────────────────────
        line1 = title_line1_fn(key)
        line2 = title_line2_fn(key)

        # background for header
        hdr_color = (255, 240, 240) if is_default else (245, 245, 245)
        draw.rectangle([x0, TOP_PAD, x0 + IMG_SIZE - 1, TOP_PAD + HEADER_H - 1],
                       fill=hdr_color)

        # line 1 – bold, centered
        try:
            bbox1 = font_bold.getbbox(line1)
            tw1 = bbox1[2] - bbox1[0]
        except Exception:
            tw1 = len(line1) * 8
        tx1 = x0 + (IMG_SIZE - tw1) // 2
        draw.text((tx1, TOP_PAD + 5), line1, font=font_bold,
                  fill=HL_COLOR if is_default else TEXT_COLOR_BOLD)

        # line 2 – small, centered
        try:
            bbox2 = font_small.getbbox(line2)
            tw2 = bbox2[2] - bbox2[0]
        except Exception:
            tw2 = len(line2) * 7
        tx2 = x0 + (IMG_SIZE - tw2) // 2
        draw.text((tx2, TOP_PAD + 24), line2, font=font_small,
                  fill=TEXT_COLOR_SUB)

        # bottom border of header
        draw.line([(x0, TOP_PAD + HEADER_H - 1),
                   (x0 + IMG_SIZE - 1, TOP_PAD + HEADER_H - 1)],
                  fill=GRID_COLOR, width=1)

        # ── Image rows ────────────────────────────────────────────────────
        for ri, pid in enumerate(prompt_ids):
            y0 = TOP_PAD + HEADER_H + ri * (IMG_SIZE + GAP)

            img_path = os.path.join(img_dir, f"p{pid:04d}_s0000.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            else:
                # placeholder
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (180, 180, 180))
                d = ImageDraw.Draw(img)
                d.text((10, IMG_SIZE // 2 - 8), f"MISSING\n{os.path.basename(img_path)}",
                       fill=(80, 80, 80), font=font_tiny)

            canvas.paste(img, (x0, y0))

            # red highlight border for default column
            if is_default:
                for w in range(HL_WIDTH):
                    draw.rectangle(
                        [x0 + w, y0 + w,
                         x0 + IMG_SIZE - 1 - w, y0 + IMG_SIZE - 1 - w],
                        outline=HL_COLOR
                    )

    # outer border
    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(180, 180, 180), width=1)

    canvas.save(out_path, "PDF", resolution=150)
    print(f"  Saved → {out_path}  ({total_w}×{total_h}px)")
    return canvas


# ── Calibri-style grid (matches vis_compare.pdf width, name-only header) ────

def get_font_carlito(size, bold=False):
    candidates = [
        "/home/runkai/.local/share/fonts/Carlito/Carlito-Bold.ttf",
        "/home/runkai/.local/share/fonts/Carlito/Carlito-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


def make_grid_calibri(configs, prompt_ids, default_key, label_fn, out_path):
    """
    vis_compare.pdf-matched style:
      - 241 pts wide (300 DPI → 1004px total)
      - Carlito Bold, font size 22, name only in header
      - Zero row gap, thin column separator
      - Red highlight on default column
    """
    DPI      = 300
    TOTAL_W  = 1004
    N_COLS   = len(configs)
    COL_GAP_ = 3
    IMG_W    = (TOTAL_W - (N_COLS - 1) * COL_GAP_) // N_COLS
    IMG_H    = IMG_W
    HEADER_H_ = 46
    FONT_SZ   = 22
    N_ROWS    = len(prompt_ids)

    total_h = HEADER_H_ + N_ROWS * IMG_H
    canvas  = Image.new("RGB", (TOTAL_W, total_h), (255, 255, 255))
    draw    = ImageDraw.Draw(canvas)

    font      = get_font_carlito(FONT_SZ, bold=True)
    font_tiny = get_font_carlito(9, bold=False)

    for ci, (key, img_dir) in enumerate(configs):
        x0    = ci * (IMG_W + COL_GAP_)
        is_hl = (key == default_key)
        color = (210, 30, 30) if is_hl else (20, 20, 20)

        # header
        hdr_bg = (255, 242, 242) if is_hl else (248, 248, 248)
        draw.rectangle([x0, 0, x0 + IMG_W - 1, HEADER_H_ - 1], fill=hdr_bg)
        draw.line([(x0, HEADER_H_ - 1), (x0 + IMG_W - 1, HEADER_H_ - 1)],
                  fill=(210, 210, 210), width=1)

        label = label_fn(key)
        try:
            bb = font.getbbox(label)
            tw = bb[2] - bb[0]
        except Exception:
            tw = len(label) * FONT_SZ // 2
        tx = x0 + (IMG_W - tw) // 2
        ty = (HEADER_H_ - FONT_SZ) // 2
        draw.text((tx, ty), label, font=font, fill=color)

        # column separator
        if ci < N_COLS - 1:
            sx = x0 + IMG_W + COL_GAP_ // 2
            draw.line([(sx, 0), (sx, total_h - 1)], fill=(210, 210, 210), width=1)

        # images
        for ri, pid in enumerate(prompt_ids):
            y0       = HEADER_H_ + ri * IMG_H
            img_path = os.path.join(img_dir, f"p{pid:04d}_s0000.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
            else:
                img = Image.new("RGB", (IMG_W, IMG_H), (190, 190, 190))
                d2  = ImageDraw.Draw(img)
                d2.text((4, IMG_H // 2 - 5), "MISSING", fill=(80, 80, 80), font=font_tiny)
            canvas.paste(img, (x0, y0))

            if is_hl:
                for w in range(3):
                    draw.rectangle([x0+w, y0+w, x0+IMG_W-1-w, y0+IMG_H-1-w],
                                   outline=(210, 30, 30))

    canvas.save(out_path, "PDF", resolution=DPI)
    print(f"  Saved → {out_path}  ({TOTAL_W}×{total_h}px @{DPI}dpi)")


# ── Experiment definitions ──────────────────────────────────────────────────

def make_maskstep():
    print("\n[1/2] mask_step sensitivity figure")

    configs = [
        ("maskstep_03", os.path.join(BASE, "maskstep_03", "vis_images")),
        ("maskstep_05", os.path.join(BASE, "maskstep_05", "vis_images")),
        ("maskstep_07", os.path.join(BASE, "maskstep_07", "vis_images")),
        ("maskstep_10", os.path.join(BASE, "maskstep_10", "vis_images")),
        ("maskstep_15", os.path.join(BASE, "maskstep_15", "vis_images")),
    ]

    def label(key):
        step = int(key.split("_")[1])
        s = f"mask_step={step}"
        return s + "" if key == "maskstep_05" else s

    make_grid_calibri(
        configs, PROMPT_IDS, "maskstep_05", label,
        out_path=os.path.join(OUT_BASE, "vis_maskstep.pdf"),
    )


def make_ratio():
    print("\n[2/2] skip_ratio sensitivity figure")

    configs = [
        ("ratio_030", os.path.join(BASE, "ratio_030", "vis_images")),
        ("ratio_040", os.path.join(BASE, "ratio_040", "vis_images")),
        ("ratio_050", os.path.join(BASE, "ratio_050", "vis_images")),
        ("ratio_060", os.path.join(BASE, "ratio_060", "vis_images")),
        ("ratio_070", os.path.join(BASE, "ratio_070", "vis_images")),
    ]

    def label(key):
        pct = int(key.split("_")[1])
        s = f"skip_ratio={pct/100:.2f}"
        return s + "" if key == "ratio_050" else s

    make_grid_calibri(
        configs, PROMPT_IDS, "ratio_050", label,
        out_path=os.path.join(OUT_BASE, "vis_ratio.pdf"),
    )


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_maskstep()
    make_ratio()
    print("\nDone.")
