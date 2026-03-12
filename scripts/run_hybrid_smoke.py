#!/usr/bin/env python3
"""
Hybrid approach: dense early + sparse late.

Compares:
  1. Baseline (dense, CFG=4.0)
  2. Current (sparse from step 6, skip=2)           → ~2.4x, bad bg
  3. Step-only (no sparse blocks, skip=2)            → ~1.6x, good bg
  4. Hybrid: sparse from step 15 + skip=2            → middle ground
  5. Hybrid: sparse from step 20 + skip=2            → more quality
  6. Hybrid: sparse from step 10 + skip=2            → more speed

30 steps total, mask_step=5:
  - Config 4: 5 dense + 9 dense-skip + 16 sparse-skip
  - Config 5: 5 dense + 14 dense-skip + 11 sparse-skip

All use CFG magnitude mask, s_fg=7.0, s_bg=2.0.
3 prompts x 1 seed.
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

    common = dict(
        skip_ratio=0.50, s_fg=7.0, s_bg=2.0,
        mask_step=5, mask_type="cfg_magnitude",
        skip_step_interval=2,
    )

    experiments = [
        {
            "name": "hybrid-1-baseline",
            "desc": "Baseline (dense, CFG=4.0)",
            "method": "baseline",
        },
        {
            "name": "hybrid-2-current",
            "desc": "Sparse from step 6 (current)",
            "method": "accel_dual",
            "kwargs": dict(**common, sparse_blocks=True, sparse_ffn=True,
                           sparse_start_step=-1),
        },
        {
            "name": "hybrid-3-step-only",
            "desc": "Step-only, no sparse blocks",
            "method": "accel_dual",
            "kwargs": dict(**common, sparse_blocks=False, sparse_ffn=False),
        },
        {
            "name": "hybrid-4-sparse-from-10",
            "desc": "Hybrid: sparse from step 10",
            "method": "accel_dual",
            "kwargs": dict(**common, sparse_blocks=True, sparse_ffn=True,
                           sparse_start_step=10),
        },
        {
            "name": "hybrid-5-sparse-from-15",
            "desc": "Hybrid: sparse from step 15",
            "method": "accel_dual",
            "kwargs": dict(**common, sparse_blocks=True, sparse_ffn=True,
                           sparse_start_step=15),
        },
        {
            "name": "hybrid-6-sparse-from-20",
            "desc": "Hybrid: sparse from step 20",
            "method": "accel_dual",
            "kwargs": dict(**common, sparse_blocks=True, sparse_ffn=True,
                           sparse_start_step=20),
        },
    ]

    print(f"=== Hybrid Approach Smoke Test ({len(experiments)} configs) ===")
    print(f"Prompts: {len(test_prompts)}, Seed: {seed}, Steps: {steps}")
    print()

    all_results = {}

    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp["desc"]

        print(f"[{idx+1}/{len(experiments)}] {desc}")

        exp_dir = setup_output_dir(output_base, name)
        img_dir = os.path.join(exp_dir, "images")

        exp_times = []

        for pi, prompt in enumerate(test_prompts):
            set_seed(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if exp["method"] == "baseline":
                image = wrapper.generate(
                    prompt=prompt, seed=seed,
                    cfg_scale=4.0, steps=steps,
                    height=resolution, width=resolution,
                )
            else:
                image = wrapper.generate_accelerated_dual(
                    prompt=prompt, seed=seed,
                    steps=steps, height=resolution, width=resolution,
                    **exp["kwargs"],
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            exp_times.append(elapsed)

            img_arr = np.array(image)
            assert not np.isnan(img_arr).any(), f"NaN in {name}!"

            fname = f"p{pi:04d}_s{seed:04d}.png"
            image.save(os.path.join(img_dir, fname))
            print(f"    p{pi}: OK ({elapsed:.2f}s)")

            del image
            gc.collect()
            torch.cuda.empty_cache()

        mean_time = sum(exp_times) / len(exp_times)

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
    print(f"{'Config':<45} {'Time':>6} {'Speed':>6} {'CLIP':>8} {'dCLIP':>7} {'Edge':>8}")
    print("=" * 95)

    ref_time = all_results["hybrid-1-baseline"]["mean_time"]
    ref_clip = all_results["hybrid-1-baseline"]["mean_clip"]

    for name, r in all_results.items():
        t = r["mean_time"]
        speedup = ref_time / t
        mc = r["mean_clip"]
        dc = mc - ref_clip
        edge = r["mean_edge"]
        print(f"  {r['desc']:<43} {t:5.2f}s {speedup:5.2f}x {mc:7.4f} {dc:+6.4f} {edge:7.4f}")

    print("=" * 95)
    print()
    print("=== Compare images in outputs/hybrid-*/images/ ===")


if __name__ == "__main__":
    main()
