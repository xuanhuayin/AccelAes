#!/usr/bin/env python3
"""
Generate high-quality images for sensitivity visualization figures.
Uses 5 aesthetically rich prompts × 10 configs (5 mask_step + 5 skip_ratio).
Output: outputs/supp_sensitivity/{config}/vis_images/p{i:04d}_s0000.png
"""
import os, sys, gc
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper

# ── 5 aesthetic prompts (from accelae_topk set, not used elsewhere) ──────────
PROMPTS = [
    "A silver wolf howling under moonlight, intricate fur and sharp claws, "
    "dramatic sky background, photorealistic, cinematic, masterpiece",

    "A peacock displaying its intricate iridescent plumage, garden setting, "
    "soft bokeh, photorealistic, stunning, dramatic lighting",

    "A geisha in intricate traditional kimono with detailed embroidery, "
    "cherry blossom background, photorealistic, cinematic, portrait",

    "A phoenix rising from flames with intricate golden feather details, "
    "dramatic lighting, fantasy art, highly detailed, vivid",

    "Grand baroque palace interior with intricate golden decorations, "
    "volumetric lighting, photorealistic, masterpiece, highly detailed",
]

SEED = 0

BASE_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    s_fg=7.0, s_bg=1.0,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
    cfg_scale=4.0, steps=30,
)

CONFIGS = [
    # mask_step sweep (skip_ratio fixed at 0.50)
    ("maskstep_03", {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step":  3}),
    ("maskstep_05", {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step":  5}),
    ("maskstep_07", {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step":  7}),
    ("maskstep_10", {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 10}),
    ("maskstep_15", {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 15}),
    # skip_ratio sweep (mask_step fixed at 5)
    ("ratio_030",   {**BASE_KWARGS, "skip_ratio": 0.30, "mask_step":  5}),
    ("ratio_040",   {**BASE_KWARGS, "skip_ratio": 0.40, "mask_step":  5}),
    ("ratio_050",   {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step":  5}),
    ("ratio_060",   {**BASE_KWARGS, "skip_ratio": 0.60, "mask_step":  5}),
    ("ratio_070",   {**BASE_KWARGS, "skip_ratio": 0.70, "mask_step":  5}),
]

BASE_DIR = "outputs/supp_sensitivity"
TOTAL    = len(PROMPTS) * len(CONFIGS)


def main():
    print(f"Loading Lumina wrapper...")
    wrapper = LuminaDiTWrapper(dtype="bf16")

    done = 0
    for cfg_name, kwargs in CONFIGS:
        vis_dir = os.path.join(BASE_DIR, cfg_name, "vis_images")
        os.makedirs(vis_dir, exist_ok=True)

        for pi, prompt in enumerate(PROMPTS):
            out_path = os.path.join(vis_dir, f"p{pi:04d}_s0000.png")
            if os.path.exists(out_path):
                done += 1
                print(f"  [{done}/{TOTAL}] skip {cfg_name}/p{pi:04d} (exists)")
                continue

            print(f"  [{done+1}/{TOTAL}] {cfg_name} | p{pi:04d}: {prompt[:55]}...")
            img = wrapper.generate_accelerated_dual(
                prompt=prompt, seed=SEED, **kwargs
            )
            img.save(out_path)
            del img; gc.collect(); torch.cuda.empty_cache()
            done += 1
            print(f"    saved → {out_path}")

    del wrapper; gc.collect(); torch.cuda.empty_cache()
    print(f"\nAll done. {done}/{TOTAL} images in vis_images/ subdirs.")


if __name__ == "__main__":
    main()
