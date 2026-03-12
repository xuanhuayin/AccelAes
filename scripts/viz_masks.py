#!/usr/bin/env python3
"""Visualize mask quality: side-by-side comparison of image, masks, and overlay."""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_img(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img

def load_mask(path, size=None):
    img = Image.open(path).convert("L")
    if size:
        img = img.resize(size, Image.NEAREST)
    return img

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.4):
    """Overlay mask on image: foreground regions tinted with color."""
    img_np = np.array(image).astype(float)
    mask_np = np.array(mask).astype(float) / 255.0
    # Expand mask to 3 channels
    mask_3d = mask_np[:, :, None]
    color_np = np.array(color, dtype=float)
    # Blend: where mask=1, mix image with color
    overlay = img_np * (1 - mask_3d * alpha) + color_np * mask_3d * alpha
    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))

def mask_stats(mask_path):
    """Compute mask coverage statistics."""
    mask = np.array(Image.open(mask_path).convert("L")).astype(float) / 255.0
    fg_ratio = (mask > 0.5).mean()
    mean_val = mask.mean()
    return fg_ratio, mean_val

def main():
    base = "/home/runkai/xuanhua/SASD/outputs"
    prompts = [
        "A panda bear as a mad scientist",
        "Gothic cathedral in a stormy night",
        "pentacle made of syringes",
        "pavement in the forest, ghibli anime style",
        "Traditional library with floor-to-ceiling bookcases",
        "Alice Liddell brane whimsy style",
        "porcelain doll, doll, ooak bjd...",
        "Drawing of a towering city, fantasy style...",
        "hot toys of walter white",
        "photo of 60s BMW pickup",
    ]

    configs = [
        ("debug_sem_r25", "Semantic r=0.25"),
        ("debug_sem_r40", "Semantic r=0.40"),
        ("debug_cplx_r25", "Complexity r=0.25"),
    ]

    cell_size = (256, 256)
    n_prompts = 10
    n_cols = 1 + len(configs) * 2  # image + (mask + overlay) per config
    margin = 2
    header_h = 30
    label_w = 200

    total_w = label_w + n_cols * (cell_size[0] + margin)
    total_h = header_h + n_prompts * (cell_size[1] + margin)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Header
    headers = ["Original"]
    for _, label in configs:
        headers.append(f"{label} mask")
        headers.append(f"{label} overlay")

    for col, h in enumerate(headers):
        x = label_w + col * (cell_size[0] + margin) + 5
        draw.text((x, 5), h, fill=(0, 0, 0), font=font_small)

    # Rows
    colors = [(255, 0, 0), (0, 200, 0), (0, 100, 255)]

    for pi in range(n_prompts):
        y = header_h + pi * (cell_size[1] + margin)
        fname = f"p{pi:04d}_s0000"

        # Prompt label
        prompt_short = prompts[pi][:28]
        draw.text((5, y + cell_size[1] // 2 - 6), prompt_short, fill=(0, 0, 0), font=font_small)

        # Original image (from baseline or debug_sem_r25)
        img_path = os.path.join(base, "debug_sem_r25", "images", f"{fname}.png")
        if os.path.exists(img_path):
            img = load_img(img_path, cell_size)
            canvas.paste(img, (label_w, y))

        col_idx = 1
        for ci, (exp_name, label) in enumerate(configs):
            mask_path = os.path.join(base, exp_name, "mask_visualizations", f"{fname}_mask.png")
            img_path = os.path.join(base, exp_name, "images", f"{fname}.png")

            if os.path.exists(mask_path):
                # Mask
                mask = load_mask(mask_path, cell_size)
                mask_rgb = Image.merge("RGB", [mask, mask, mask])
                x = label_w + col_idx * (cell_size[0] + margin)
                canvas.paste(mask_rgb, (x, y))

                # Stats
                fg_r, mean_v = mask_stats(mask_path)
                draw.text((x + 3, y + 3), f"fg={fg_r:.0%} mean={mean_v:.2f}",
                         fill=(255, 255, 0), font=font_small)

                # Overlay
                col_idx += 1
                if os.path.exists(img_path):
                    img = load_img(img_path, cell_size)
                    ov = overlay_mask(img, mask, color=colors[ci], alpha=0.4)
                    x = label_w + col_idx * (cell_size[0] + margin)
                    canvas.paste(ov, (x, y))

            col_idx += 1

    out_path = os.path.join(base, "mask_comparison.png")
    canvas.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Size: {total_w}x{total_h}")

    # Print stats table
    print("\n=== Mask Coverage Statistics ===")
    print(f"{'Prompt':40s} | {'Sem r=0.25':15s} | {'Sem r=0.40':15s} | {'Cplx r=0.25':15s}")
    print("-" * 95)
    for pi in range(n_prompts):
        fname = f"p{pi:04d}_s0000"
        stats = []
        for exp_name, _ in configs:
            mask_path = os.path.join(base, exp_name, "mask_visualizations", f"{fname}_mask.png")
            if os.path.exists(mask_path):
                fg_r, mean_v = mask_stats(mask_path)
                stats.append(f"fg={fg_r:.0%} m={mean_v:.2f}")
            else:
                stats.append("N/A")
        print(f"{prompts[pi][:40]:40s} | {stats[0]:15s} | {stats[1]:15s} | {stats[2]:15s}")

if __name__ == "__main__":
    main()
