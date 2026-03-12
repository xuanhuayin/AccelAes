#!/usr/bin/env python3
"""
Background quality tuning smoke test.

Tests different s_bg and skip_ratio combinations to improve background quality.
Uses CFG magnitude mask (best from mask comparison eval).

Grid:
  s_bg:       1.0 (current), 2.0, 3.0
  skip_ratio: 0.50 (current), 0.40, 0.35

Also includes baseline for reference.
3 prompts x 1 seed x (1 baseline + 9 configs) = 30 images.
"""

import os
import sys
import time
import gc
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.io_utils import load_config, load_prompts, setup_output_dir
from src.utils.seed import set_seed
from src.eval.speed import setup_torch_determinism
from src.eval.metrics import compute_clip_score, compute_edge_density
from PIL import Image


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

    prompts = load_prompts("prompts/prompts_dev.txt")
    test_prompts = prompts[:3]
    seed = 0
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["output_dir"],
    )

    # Parameter grid
    s_bg_values = [1.0, 2.0, 3.0]
    skip_ratio_values = [0.50, 0.40, 0.35]

    # Fixed params
    fixed_kwargs = dict(
        s_fg=7.0,
        mask_step=5,
        sparse_ffn=True,
        skip_step_interval=2,
        mask_type="cfg_magnitude",
    )

    # Build experiment list
    experiments = [
        {
            "name": "bg_tune-baseline",
            "desc": "Baseline (dense, CFG=4.0)",
            "method": "baseline",
        },
    ]

    for s_bg in s_bg_values:
        for skip_ratio in skip_ratio_values:
            experiments.append({
                "name": f"bg_tune-sbg{s_bg:.1f}-sr{skip_ratio:.2f}",
                "desc": f"s_bg={s_bg:.1f}, skip_ratio={skip_ratio:.2f}",
                "method": "accel_dual",
                "s_bg": s_bg,
                "skip_ratio": skip_ratio,
            })

    print(f"=== Background Tuning Smoke Test ({len(experiments)} configs) ===")
    print(f"Prompts: {len(test_prompts)}, Seed: {seed}, Steps: {steps}")
    print(f"Fixed: mask_type=cfg_magnitude, s_fg=7.0, sparse_ffn=True, skip_interval=2")
    print(f"Grid: s_bg={s_bg_values}, skip_ratio={skip_ratio_values}")
    print()

    all_results = {}

    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp["desc"]
        method = exp["method"]

        print(f"[{idx+1}/{len(experiments)}] {desc}")

        exp_dir = setup_output_dir(output_base, name)
        img_dir = os.path.join(exp_dir, "images")

        exp_times = []
        exp_images = []
        exp_prompts = []

        for pi, prompt in enumerate(test_prompts):
            set_seed(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if method == "baseline":
                image = wrapper.generate(
                    prompt=prompt, seed=seed,
                    cfg_scale=4.0, steps=steps,
                    height=resolution, width=resolution,
                )
            elif method == "accel_dual":
                image = wrapper.generate_accelerated_dual(
                    prompt=prompt, seed=seed,
                    steps=steps, height=resolution, width=resolution,
                    skip_ratio=exp["skip_ratio"],
                    s_bg=exp["s_bg"],
                    **fixed_kwargs,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            exp_times.append(elapsed)

            # Check for NaN
            img_arr = np.array(image)
            assert not np.isnan(img_arr).any(), f"NaN in {name}!"

            fname = f"p{pi:04d}_s{seed:04d}.png"
            image.save(os.path.join(img_dir, fname))
            exp_images.append(image)
            exp_prompts.append(prompt)
            print(f"    p{pi}: OK ({elapsed:.2f}s)")

            del image
            gc.collect()
            torch.cuda.empty_cache()

        mean_time = sum(exp_times) / len(exp_times)

        # Quick CLIP score
        loaded_images = [
            Image.open(os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")).convert("RGB")
            for pi in range(len(test_prompts))
        ]
        clip_scores = compute_clip_score(loaded_images, list(test_prompts), device="cuda")
        edge_densities = [compute_edge_density(img) for img in loaded_images]

        all_results[name] = {
            "desc": desc,
            "mean_time": mean_time,
            "mean_clip": float(np.mean(clip_scores)),
            "mean_edge": float(np.mean(edge_densities)),
        }
        print(f"    Mean: {mean_time:.2f}s, CLIP: {np.mean(clip_scores):.4f}, Edge: {np.mean(edge_densities):.4f}")

    # Summary
    print()
    print("=" * 95)
    print(f"{'Config':<40} {'Time':>6} {'Speed':>6} {'CLIP':>8} {'dCLIP':>7} {'Edge':>8}")
    print("=" * 95)

    ref_time = all_results["bg_tune-baseline"]["mean_time"]
    ref_clip = all_results["bg_tune-baseline"]["mean_clip"]

    for name, r in all_results.items():
        t = r["mean_time"]
        speedup = ref_time / t
        mc = r["mean_clip"]
        dc = mc - ref_clip
        edge = r["mean_edge"]
        print(f"  {r['desc']:<38} {t:5.2f}s {speedup:5.2f}x {mc:7.4f} {dc:+6.4f} {edge:7.4f}")

    print("=" * 95)
    print("  dCLIP = delta vs Baseline")
    print()
    print("=== Background tuning smoke test done! ===")


if __name__ == "__main__":
    main()
