#!/usr/bin/env python3
"""
Mask type comparison smoke test.

Compares 4 mask strategies (all semantic/region-based, no grid):
  1. Baseline (dense, uniform CFG=4.0)
  2. Semantic (cross-attention + CLIP aesthetic anchors)
  3. SDiT (gradient complexity + SLIC regions)
  4. CFG magnitude (|cond - uncond| + SLIC regions)

All use: sparse attn1 + sparse FFN + skip_interval=2 for max speed.
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
    # Use first 3 prompts for smoke test diversity
    test_prompts = prompts[:3]
    seed = 0
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["output_dir"],
    )

    # Common accel params (best speedup config)
    accel_kwargs = dict(
        skip_ratio=0.50,
        s_fg=7.0, s_bg=1.0,
        mask_step=5,
        sparse_ffn=True,
        skip_step_interval=2,
    )

    experiments = [
        {
            "name": "mask_cmp-1-baseline",
            "desc": "1. Baseline (dense, uniform CFG=4.0)",
            "method": "baseline",
        },
        {
            "name": "mask_cmp-2-semantic",
            "desc": "2. Semantic (cross-attn + CLIP anchors)",
            "method": "accel_dual",
            "mask_type": "semantic",
        },
        {
            "name": "mask_cmp-3-sdit",
            "desc": "3. SDiT (gradient complexity + SLIC)",
            "method": "accel_dual",
            "mask_type": "sdit",
        },
        {
            "name": "mask_cmp-4-cfg-magnitude",
            "desc": "4. CFG magnitude (|cond-uncond| + SLIC)",
            "method": "accel_dual",
            "mask_type": "cfg_magnitude",
        },
    ]

    print(f"=== Mask Type Comparison Smoke Test ({len(experiments)} configs) ===")
    print(f"Prompts: {len(test_prompts)}, Seed: {seed}, Steps: {steps}")
    print()

    timings = {}

    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp["desc"]
        method = exp["method"]

        print(f"[{idx+1}/{len(experiments)}] {desc}")

        exp_dir = setup_output_dir(output_base, name)
        img_dir = os.path.join(exp_dir, "images")

        exp_times = []
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
                    mask_type=exp["mask_type"],
                    **accel_kwargs,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            exp_times.append(elapsed)

            # Check for NaN
            img_arr = np.array(image)
            assert not np.isnan(img_arr).any(), f"NaN in {name}!"
            assert img_arr.shape[0] == resolution and img_arr.shape[1] == resolution

            fname = f"p{pi:04d}_s{seed:04d}.png"
            image.save(os.path.join(img_dir, fname))
            print(f"    p{pi}: OK ({elapsed:.2f}s)")

            del image
            gc.collect()
            torch.cuda.empty_cache()

        mean_time = sum(exp_times) / len(exp_times)
        timings[name] = mean_time
        print(f"    Mean: {mean_time:.2f}s")

    # Summary
    print()
    print("=" * 70)
    print("Speed Summary (mean over {} prompts)".format(len(test_prompts)))
    print("=" * 70)
    baseline_time = timings.get("mask_cmp-1-baseline", None)
    for name, t in timings.items():
        speedup = ""
        if baseline_time and "baseline" not in name:
            speedup = f"  ({baseline_time / t:.2f}x)"
        print(f"  {name:40s} {t:6.2f}s{speedup}")
    print("=" * 70)
    print()
    print("=== All mask comparison smoke tests passed! ===")


if __name__ == "__main__":
    main()
