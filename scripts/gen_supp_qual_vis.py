#!/usr/bin/env python3
"""
Generate supplementary qualitative comparison figures in vis_compare.pdf style.

Produces:
  outputs/supp_qual_assets/lumina/vis_lumina.pdf   (8 methods × 5 prompts)
  outputs/supp_qual_assets/flux/vis_flux.pdf       (5 methods × 4 prompts)
  outputs/supp_qual_assets/sd3/vis_sd3.pdf         (5 methods × 5 prompts)

Style matches vis_compare.pdf:
  - Large bold method name in column header
  - Speedup below (smaller)
  - Zero row gap (images flush)
  - Thin column separator
  - Red highlight (text + border) for AccelAes column
"""

import os, sys, glob
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Layout constants (vis_compare.pdf style) ─────────────────────────────────
IMG_SIZE   = 256        # display size per cell
COL_GAP    = 3          # gap between columns
ROW_GAP    = 0          # no gap between rows (flush, like vis_compare)
HEADER_H   = 58         # column header height
LEFT_PAD   = 0
TOP_PAD    = 0
RIGHT_PAD  = 0
BOT_PAD    = 0

BG_COLOR   = (255, 255, 255)
SEP_COLOR  = (210, 210, 210)   # column separator
HL_COLOR   = (210, 30, 30)     # AccelAes red
HL_WIDTH   = 3

FONT_NAME_SIZE  = 17   # method name
FONT_SP_SIZE    = 14   # speedup


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


def text_width(font, text):
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0]
    except Exception:
        return len(text) * (FONT_NAME_SIZE // 2)


def make_vis(configs, n_prompts, default_key, out_path, filenames=None):
    """
    configs: list of (method_label, speedup_str, img_dir, file_pattern)
             file_pattern: e.g. "p{:02d}.png" or "p{:04d}_s0000.png"
             If filenames is given, file_pattern is ignored.
    n_prompts: number of prompt rows to include
    default_key: method_label for AccelAes highlight
    out_path: output PDF path
    filenames: optional explicit list of filenames (overrides file_pattern)
    """
    n_cols = len(configs)
    n_rows = n_prompts

    col_w = IMG_SIZE
    total_w = LEFT_PAD + n_cols * col_w + (n_cols - 1) * COL_GAP + RIGHT_PAD
    total_h = TOP_PAD + HEADER_H + n_rows * IMG_SIZE + max(0, n_rows - 1) * ROW_GAP + BOT_PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font_name = get_font(FONT_NAME_SIZE, bold=True)
    font_sp   = get_font(FONT_SP_SIZE,   bold=False)
    font_tiny = get_font(10, bold=False)

    for ci, (label, speedup, img_dir, pat) in enumerate(configs):
        x0 = LEFT_PAD + ci * (col_w + COL_GAP)
        is_hl = (label == default_key)
        color = HL_COLOR if is_hl else (30, 30, 30)
        sp_color = HL_COLOR if is_hl else (80, 80, 80)

        # ── Column header ──────────────────────────────────────────────────
        hdr_bg = (255, 242, 242) if is_hl else (248, 248, 248)
        draw.rectangle([x0, TOP_PAD, x0 + col_w - 1, TOP_PAD + HEADER_H - 1],
                       fill=hdr_bg)
        # bottom border of header
        draw.line([(x0, TOP_PAD + HEADER_H - 1),
                   (x0 + col_w - 1, TOP_PAD + HEADER_H - 1)],
                  fill=SEP_COLOR, width=1)

        # method name (bold, centered)
        tw = text_width(font_name, label)
        tx = x0 + (col_w - tw) // 2
        draw.text((tx, TOP_PAD + 8), label, font=font_name, fill=color)

        # speedup (smaller, centered)
        tw2 = text_width(font_sp, speedup)
        tx2 = x0 + (col_w - tw2) // 2
        draw.text((tx2, TOP_PAD + 8 + FONT_NAME_SIZE + 5), speedup,
                  font=font_sp, fill=sp_color)

        # column separator (right side)
        if ci < n_cols - 1:
            sx = x0 + col_w + COL_GAP // 2
            draw.line([(sx, TOP_PAD), (sx, total_h - 1)], fill=SEP_COLOR, width=1)

        # ── Image rows ────────────────────────────────────────────────────
        for ri in range(n_rows):
            y0 = TOP_PAD + HEADER_H + ri * (IMG_SIZE + ROW_GAP)
            fname = filenames[ri] if filenames else pat.format(ri)
            img_path = os.path.join(img_dir, fname)

            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            else:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (190, 190, 190))
                d2 = ImageDraw.Draw(img)
                d2.text((5, IMG_SIZE // 2 - 6),
                        f"MISSING\n{fname}",
                        fill=(80, 80, 80), font=font_tiny)

            canvas.paste(img, (x0, y0))

            # red border for AccelAes column
            if is_hl:
                for w in range(HL_WIDTH):
                    draw.rectangle(
                        [x0 + w, y0 + w,
                         x0 + IMG_SIZE - 1 - w, y0 + IMG_SIZE - 1 - w],
                        outline=HL_COLOR
                    )

    canvas.save(out_path, "PDF", resolution=150)
    print(f"  Saved → {out_path}  ({total_w}×{total_h}px)")


# ── SD3 special: vis_compare.pdf width, Carlito, name-only header ────────────

def get_font_carlito(size, bold=False):
    """Carlito = metrically compatible open-source Calibri substitute."""
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


def make_vis_calibri(configs, n_prompts, default_key, out_path, filenames=None):
    """
    vis_compare.pdf-matched style for all backbones:
      - Total PDF width = 241 pts (saves at 300 DPI → 1004px)
      - Column header: Carlito Bold size 22, name only (no speedup/metrics)
      - Zero row gap, thin column separator
      - Red AccelAes highlight
    configs: list of (label, _speedup, img_dir, file_pattern)
    filenames: optional explicit list of filenames per row (overrides file_pattern)
    """
    DPI      = 300
    TOTAL_W  = 1004
    N_COLS   = len(configs)
    COL_GAP_ = 3
    IMG_W    = (TOTAL_W - (N_COLS - 1) * COL_GAP_) // N_COLS
    IMG_H    = IMG_W
    HEADER_H_ = 46
    FONT_SIZE = 22

    total_h = HEADER_H_ + n_prompts * IMG_H
    canvas  = Image.new("RGB", (TOTAL_W, total_h), (255, 255, 255))
    draw    = ImageDraw.Draw(canvas)

    font_name = get_font_carlito(FONT_SIZE, bold=True)
    font_tiny = get_font_carlito(9, bold=False)

    for ci, (label, _speedup, img_dir, pat) in enumerate(configs):
        x0    = ci * (IMG_W + COL_GAP_)
        is_hl = (label == default_key)
        color = HL_COLOR if is_hl else (20, 20, 20)

        hdr_bg = (255, 242, 242) if is_hl else (248, 248, 248)
        draw.rectangle([x0, 0, x0 + IMG_W - 1, HEADER_H_ - 1], fill=hdr_bg)
        draw.line([(x0, HEADER_H_ - 1), (x0 + IMG_W - 1, HEADER_H_ - 1)],
                  fill=(210, 210, 210), width=1)

        try:
            bb = font_name.getbbox(label)
            tw = bb[2] - bb[0]
        except Exception:
            tw = len(label) * FONT_SIZE // 2
        tx = x0 + (IMG_W - tw) // 2
        ty = (HEADER_H_ - FONT_SIZE) // 2
        draw.text((tx, ty), label, font=font_name, fill=color)

        if ci < N_COLS - 1:
            sx = x0 + IMG_W + COL_GAP_ // 2
            draw.line([(sx, 0), (sx, total_h - 1)], fill=(210, 210, 210), width=1)

        for ri in range(n_prompts):
            y0    = HEADER_H_ + ri * IMG_H
            fname = filenames[ri] if filenames else pat.format(ri)
            img_path = os.path.join(img_dir, fname)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
            else:
                img = Image.new("RGB", (IMG_W, IMG_H), (190, 190, 190))
                d2  = ImageDraw.Draw(img)
                d2.text((4, IMG_H // 2 - 5), "MISSING", fill=(80, 80, 80), font=font_tiny)
            canvas.paste(img, (x0, y0))

            if is_hl:
                for w in range(HL_WIDTH):
                    draw.rectangle([x0+w, y0+w, x0+IMG_W-1-w, y0+IMG_H-1-w],
                                   outline=HL_COLOR)

    canvas.save(out_path, "PDF", resolution=DPI)
    print(f"  Saved → {out_path}  ({TOTAL_W}×{total_h}px @{DPI}dpi)")


# ── Lumina ───────────────────────────────────────────────────────────────────

def make_lumina():
    print("\n[1/3] Lumina (8 methods × 5 prompts)")
    BASE = "outputs/supp_qual_assets/lumina"
    # order matches vis_compare.pdf
    configs = [
        ("Baseline",    "1.00×",  f"{BASE}/baseline",   "p{:02d}.png"),
        ("FORA",        "1.00×",  f"{BASE}/FORA",        "p{:02d}.png"),
        ("Δ-DiT",       "1.52×",  f"{BASE}/DeltaDiT",    "p{:02d}.png"),
        ("TeaCache",    "1.49×",  f"{BASE}/TeaCache",    "p{:02d}.png"),
        ("TaylorSeer",  "2.05×",  f"{BASE}/TaylorSeer",  "p{:02d}.png"),
        ("SDiT",        "1.69×",  f"{BASE}/SDiT",        "p{:02d}.png"),
        ("RAS",         "1.47×",  f"{BASE}/RAS",         "p{:02d}.png"),
        ("AccelAes",    "2.11×",  f"{BASE}/AccelAes",    "p{:02d}.png"),
    ]
    make_vis_calibri(configs, n_prompts=5, default_key="AccelAes",
                     out_path=f"{BASE}/vis_lumina.pdf")


# ── FLUX ─────────────────────────────────────────────────────────────────────

def make_flux():
    print("\n[2/3] FLUX (5 methods × 4 prompts)")
    BASE = "outputs/supp_qual_assets/flux"
    flux_files = ["p00.png", "p05.png", "p10.png", "p18.png"]
    configs = [
        ("Baseline",    "1.00×",  f"{BASE}/baseline",       None),
        ("TeaCache",    "1.62×",  f"{BASE}/TeaCache_t015",  None),
        ("TaylorSeer",  "2.12×",  f"{BASE}/TaylorSeer_r2",  None),
        ("AccelAes",    "1.73×",  f"{BASE}/AccelAes",       None),
    ]
    make_vis_calibri(configs, n_prompts=4, default_key="AccelAes",
                     out_path=f"{BASE}/vis_flux.pdf", filenames=flux_files)


# ── SD3 ──────────────────────────────────────────────────────────────────────

def make_sd3():
    print("\n[3/3] SD3 (5 methods × 5 prompts)")
    BASE = "outputs/supp_qual_assets/sd3"
    configs = [
        ("Baseline",   "1.00×",  f"{BASE}/baseline",          "p{:02d}.png"),
        ("TeaCache",   "1.41×",  f"{BASE}/TeaCache_t030",     "p{:02d}.png"),
        ("TaylorSeer", "1.59×",  f"{BASE}/TaylorSeer_r1",     "p{:02d}.png"),
        ("Δ-DiT",      "1.51×",  f"{BASE}/DeltaDiT",          "p{:02d}.png"),
        ("AccelAes",   "1.50×",  f"{BASE}/AccelAes_fskip2",   "p{:02d}.png"),
    ]
    # SD3: match vis_compare.pdf width (241 pts), Carlito font, name-only header
    make_vis_calibri(configs, n_prompts=5, default_key="AccelAes",
                     out_path=f"{BASE}/vis_sd3.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["lumina", "flux", "sd3", "all"], default="all")
    args = parser.parse_args()

    if args.part in ("lumina", "all"):
        make_lumina()
    if args.part in ("flux", "all"):
        make_flux()
    if args.part in ("sd3", "all"):
        make_sd3()

    print("\nDone.")
