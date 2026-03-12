#!/usr/bin/env python3
"""
Phase 4 Skip Ratio Ablation — grid vs SLIC at aggressive ratios.
Tests r=0.25, 0.5, 0.75 for both grid and SLIC masks.
10 prompts × 3 seeds = 30 images per config.
"""

import os
import sys
import time
import gc

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.io_utils import load_config, load_prompts, setup_output_dir
from src.utils.log import JsonLogger
from src.utils.seed import set_seed
from src.eval.speed import setup_torch_determinism


def save_mask_visualization(mask, heatmap, save_dir, prompt_idx, seed):
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"p{prompt_idx:04d}_s{seed:04d}"
    mask_np = mask.cpu().float().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img.save(os.path.join(save_dir, f"{prefix}_mask.png"))
    if heatmap is not None:
        hm_np = heatmap.cpu().float().numpy()
        hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-8)
        hm_img = Image.fromarray((hm_np * 255).astype(np.uint8))
        hm_img.save(os.path.join(save_dir, f"{prefix}_heatmap.png"))


def main():
    config = load_config("configs/base.yaml")
    setup_torch_determinism(
        cudnn_benchmark=config.get("speed", {}).get("cudnn_benchmark", False),
        matmul_precision=config.get("speed", {}).get("matmul_precision", "high"),
    )

    wrapper = LuminaDiTWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bf16"),
    )

    prompts = load_prompts("prompts/prompts_dev.txt")[:10]
    seeds = [0, 1, 2]
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)
    cfg_scale = config["generation"].get("cfg_scale", 4.0)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["output_dir"],
    )

    experiments = []
    for ratio in [0.25, 0.50, 0.75]:
        r_tag = f"r{int(ratio*100)}"
        # Grid + velocity
        experiments.append({
            "name": f"ablation_grid-vel_{r_tag}",
            "desc": f"grid + velocity, skip={ratio}",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=ratio, mask_type="complexity",
                extrapolation="velocity", grid_size=8,
            ),
        })
        # SLIC + velocity + soft
        experiments.append({
            "name": f"ablation_slic-vel_{r_tag}",
            "desc": f"SLIC + velocity + soft, skip={ratio}",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=ratio, mask_type="complexity",
                extrapolation="velocity",
                region_mask=True, n_segments=64,
                dilation_radius=2, mask_blur_sigma=1.5,
            ),
        })

    total_images = len(prompts) * len(seeds)
    print(f"=== Phase 4 Ratio Ablation ({len(experiments)} configs × {total_images} imgs) ===")
    print(f"Prompts: {len(prompts)}, Seeds: {seeds}")
    print()

    for exp_idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp["desc"]
        kwargs = exp["kwargs"]

        exp_dir = setup_output_dir(output_base, name)
        img_dir = os.path.join(exp_dir, "images")
        mask_viz_dir = os.path.join(exp_dir, "mask_visualizations")
        logger = JsonLogger(exp_dir)

        print(f"[{exp_idx+1}/{len(experiments)}] {desc}")
        exp_t0 = time.perf_counter()

        count = 0
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                count += 1
                set_seed(seed)
                t0 = time.perf_counter()

                callback = lambda m, h, _pi=pi, _s=seed: save_mask_visualization(
                    m, h, mask_viz_dir, _pi, _s
                )

                image = wrapper.generate_skip_update_v2(
                    prompt=prompt, seed=seed,
                    cfg_scale=cfg_scale, steps=steps,
                    height=resolution, width=resolution,
                    mask_step=5,
                    save_mask_callback=callback,
                    **kwargs,
                )

                elapsed = time.perf_counter() - t0
                fname = f"p{pi:04d}_s{seed:04d}.png"
                image.save(os.path.join(img_dir, fname))

                logger.log({
                    "prompt_idx": pi, "prompt": prompt, "seed": seed,
                    "time_s": elapsed, "filename": fname,
                    "mode": name, **kwargs,
                })

                del image
                gc.collect()
                torch.cuda.empty_cache()

        exp_elapsed = time.perf_counter() - exp_t0
        print(f"    Done in {exp_elapsed:.0f}s ({exp_elapsed/total_images:.1f}s/img)")

    print()
    print("=== Ratio Ablation complete! ===")


if __name__ == "__main__":
    main()
