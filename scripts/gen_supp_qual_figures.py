#!/usr/bin/env python3
"""
Generate supplementary qualitative comparison figures.

Creates three panel figures (one per backbone):
  Figure S1 — Lumina-Next-T2I
  Figure S2 — FLUX.1-dev
  Figure S3 — SD3-Medium  (requires baseline/fskip2 images)

Layout: rows=selected prompts, cols=methods
Output: outputs/figures/supp_qual_{lumina,flux,sd3}.pdf
"""

import os, sys, textwrap
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR    = "outputs/figures"
PROMPT_FILE = "prompts/prompts_dev.txt"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Prompt selection (0-indexed) ─────────────────────────────────────────────
PROMPT_IDS = [3, 6, 7, 14]   # all within first 20 → common across all backbones

with open(PROMPT_FILE) as f:
    ALL_PROMPTS = [l.strip() for l in f if l.strip()]

SELECTED_PROMPTS = [ALL_PROMPTS[i] for i in PROMPT_IDS]

def short_prompt(p, max_len=42):
    """Truncate and clean prompt for row label."""
    p = p.strip().rstrip(",.")
    if len(p) > max_len:
        p = p[:max_len-1].rsplit(" ", 1)[0] + "…"
    return p


# ── Backbone configurations ──────────────────────────────────────────────────

LUMINA_METHODS = [
    {
        "label": "Baseline",
        "speedup": "1.00×",
        "img_dir": "outputs/p5eval-baseline/images",
        "seed": 0,
        "color": "#555555",
    },
    {
        "label": "AccelAes\n(Ours)",
        "speedup": "2.11×",
        "img_dir": "outputs/p0_ablation_direct/sasd_full/images",
        "seed": 0,
        "color": "#c0392b",
    },
    {
        "label": "TeaCache\n(t=0.15)",
        "speedup": "1.49×",
        "img_dir": "outputs/stepcache_lumina/teacache_t015/images",
        "seed": 42,
        "color": "#2980b9",
    },
    {
        "label": "TaylorSeer\n(r=2)",
        "speedup": "2.05×",
        "img_dir": "outputs/stepcache_lumina/taylor_r2/images",
        "seed": 42,
        "color": "#27ae60",
    },
]

FLUX_METHODS = [
    {
        "label": "Baseline",
        "speedup": "1.00×",
        "img_dir": "outputs/stepcache_flux/baseline/images",
        "seed": 42,
        "color": "#555555",
    },
    {
        "label": "AccelAes\n(Ours)",
        "speedup": "1.73×",
        "img_dir": "outputs/stepcache_flux/accelae_fskip2/images",
        "seed": 42,
        "color": "#c0392b",
    },
    {
        "label": "TeaCache\n(t=0.15)",
        "speedup": "1.62×",
        "img_dir": "outputs/stepcache_flux/teacache_t015/images",
        "seed": 42,
        "color": "#2980b9",
    },
    {
        "label": "TaylorSeer\n(r=1)",
        "speedup": "1.63×",
        "img_dir": "outputs/stepcache_flux/taylor_r1_o2/images",
        "seed": 42,
        "color": "#27ae60",
    },
    {
        "label": "TaylorSeer\n(r=2)",
        "speedup": "2.12×",
        "img_dir": "outputs/stepcache_flux/taylor_r2_o2/images",
        "seed": 42,
        "color": "#8e44ad",
    },
]

SD3_METHODS = [
    {
        "label": "Baseline",
        "speedup": "1.00×",
        "img_dir": "outputs/sd3_compare/baseline_gen",
        "seed": 0,
        "color": "#555555",
    },
    {
        "label": "AccelAes\n(fskip2, Ours)",
        "speedup": "1.50×",
        "img_dir": "outputs/sd3_compare/fskip2_gen",
        "seed": 0,
        "color": "#c0392b",
    },
    {
        "label": "TeaCache\n(t=0.30)",
        "speedup": "1.41×",
        "img_dir": "outputs/sd3_compare/teacache_t030",
        "seed": 0,
        "color": "#2980b9",
    },
    {
        "label": "TaylorSeer\n(r=1)",
        "speedup": "1.59×",
        "img_dir": "outputs/sd3_compare/taylor_r1",
        "seed": 0,
        "color": "#27ae60",
    },
    {
        "label": "Δ-DiT",
        "speedup": "1.51×",
        "img_dir": "outputs/sd3_compare/delta_dit",
        "seed": 0,
        "color": "#8e44ad",
    },
]


# ── Image loading ─────────────────────────────────────────────────────────────

def load_img(img_dir, prompt_idx, seed):
    path = os.path.join(img_dir, f"p{prompt_idx:04d}_s{seed:04d}.png")
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")


def check_availability(methods, prompt_ids):
    """Return True if all required images exist."""
    all_ok = True
    for m in methods:
        for pid in prompt_ids:
            img = load_img(m["img_dir"], pid, m["seed"])
            if img is None:
                print(f"  MISSING: {m['img_dir']}/p{pid:04d}_s{m['seed']:04d}.png")
                all_ok = False
    return all_ok


# ── Figure generation ─────────────────────────────────────────────────────────

def make_figure(backbone_name, methods, prompt_ids, prompts, out_path,
                img_size_px=256, dpi=150):
    n_rows = len(prompt_ids)
    n_cols = len(methods)

    # Dimensions (inches)
    label_w  = 1.5   # left column: prompt text
    cell_w   = 1.55  # each image cell
    header_h = 0.65  # top row: method labels
    cell_h   = 1.55  # each image row
    title_h  = 0.35  # backbone name

    fig_w = label_w + n_cols * cell_w + 0.15
    fig_h = title_h + header_h + n_rows * cell_h + 0.15

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    # GridSpec: rows = header + n_rows, cols = label + n_cols
    gs = GridSpec(
        n_rows + 1, n_cols + 1,
        figure=fig,
        left   = label_w / fig_w,
        right  = 1.0 - 0.05 / fig_w,
        top    = 1.0 - (title_h + header_h) / fig_h,
        bottom = 0.05 / fig_h,
        hspace = 0.04,
        wspace = 0.04,
    )

    # ── Backbone title ──────────────────────────────────────────────────────
    fig.text(
        0.5, 1.0 - title_h / (2 * fig_h),
        backbone_name,
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="#1a1a2e",
        transform=fig.transFigure,
    )

    # ── Column headers ──────────────────────────────────────────────────────
    header_top = 1.0 - title_h / fig_h
    header_bot = 1.0 - (title_h + header_h) / fig_h
    header_mid = (header_top + header_bot) / 2

    for ci, m in enumerate(methods):
        x_left  = label_w / fig_w + ci * cell_w / fig_w
        x_right = x_left + cell_w / fig_w
        x_mid   = (x_left + x_right) / 2

        # Background rectangle
        rect = mpatches.FancyBboxPatch(
            (x_left + 0.005, header_bot + 0.01),
            cell_w / fig_w - 0.010,
            (header_top - header_bot) - 0.015,
            boxstyle="round,pad=0.01",
            linewidth=0, facecolor=m["color"], alpha=0.12,
            transform=fig.transFigure, clip_on=False, zorder=2,
        )
        fig.add_artist(rect)

        # Method label (bold, with line break support)
        label_lines = m["label"].split("\n")
        speedup_line = m["speedup"]

        n_lines = len(label_lines) + 1  # +1 for speedup
        line_h = (header_top - header_bot) / (n_lines + 0.5)

        for li, line in enumerate(label_lines):
            y = header_top - (li + 0.7) * line_h
            fig.text(x_mid, y, line,
                     ha="center", va="center",
                     fontsize=7.5, fontweight="bold", color=m["color"],
                     transform=fig.transFigure, zorder=3)

        fig.text(x_mid, header_top - (len(label_lines) + 0.7) * line_h,
                 speedup_line,
                 ha="center", va="center",
                 fontsize=6.5, color="#666666",
                 transform=fig.transFigure, zorder=3)

    # ── Row labels (prompt text) ─────────────────────────────────────────────
    row_tops = []
    for ri in range(n_rows):
        ax_dummy = fig.add_subplot(gs[ri, 0])
        ax_dummy.set_visible(False)
        bbox = ax_dummy.get_position()
        row_mid = (bbox.y0 + bbox.y1) / 2
        row_tops.append(row_mid)

        label = short_prompt(prompts[ri])
        wrapped = "\n".join(textwrap.wrap(label, width=20))
        fig.text(
            label_w / fig_w - 0.015, row_mid,
            wrapped,
            ha="right", va="center",
            fontsize=6.0, style="italic", color="#333333",
            transform=fig.transFigure,
        )

    # ── Images ──────────────────────────────────────────────────────────────
    for ri, pid in enumerate(prompt_ids):
        for ci, m in enumerate(methods):
            ax = fig.add_subplot(gs[ri, ci])

            img = load_img(m["img_dir"], pid, m["seed"])
            if img is not None:
                ax.imshow(np.array(img.resize((img_size_px, img_size_px),
                                              Image.LANCZOS)))
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=8, color="#999999", transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

            # Border: highlight AccelAes column
            edge_color = m["color"] if "Ours" in m["label"] else "#cccccc"
            edge_lw    = 1.5 if "Ours" in m["label"] else 0.5
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(edge_lw)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["lumina", "flux", "sd3", "all"],
                        default="all")
    parser.add_argument("--check", action="store_true",
                        help="Only check image availability, don't generate")
    args = parser.parse_args()

    configs = {
        "lumina": (LUMINA_METHODS, "Lumina-Next-T2I"),
        "flux":   (FLUX_METHODS,   "FLUX.1-dev"),
        "sd3":    (SD3_METHODS,    "SD3-Medium"),
    }

    targets = list(configs.keys()) if args.backbone == "all" else [args.backbone]

    for name in targets:
        methods, title = configs[name]
        print(f"\n── {title} ──")
        ok = check_availability(methods, PROMPT_IDS)
        if args.check:
            status = "✅" if ok else "❌ (missing images)"
            print(f"  {title}: {status}")
            continue

        out_path = os.path.join(OUT_DIR, f"supp_qual_{name}.pdf")
        make_figure(title, methods, PROMPT_IDS, SELECTED_PROMPTS, out_path)

        # Also save PNG for quick preview
        make_figure(title, methods, PROMPT_IDS, SELECTED_PROMPTS,
                    out_path.replace(".pdf", ".png"), dpi=150)


if __name__ == "__main__":
    main()
