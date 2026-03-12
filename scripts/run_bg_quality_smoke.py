#!/usr/bin/env python3
"""
Background quality comparison: component-level sparsity vs step-level-only.

Compares:
  1. Baseline (dense, CFG=4.0)
  2. Current best (sparse_blocks=True, skip_interval=2)  → ~2.4x
  3. Step-only (sparse_blocks=False, skip_interval=2)     → ~1.7x expected
  4. Step-only aggressive (sparse_blocks=False, skip_interval=3) → ~2.0x expected

All use CFG magnitude mask, s_fg=7.0, s_bg=2.0.
3 prompts x 1 seed = 3 images per config.
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

    experiments = [
        {
            "name": "bgq-1-baseline",
            "desc": "Baseline (dense, CFG=4.0)",
            "method": "baseline",
        },
        {
            "name": "bgq-2-sparse-blocks",
            "desc": "Sparse blocks + skip=2 (current)",
            "method": "accel_dual",
            "kwargs": dict(
                skip_ratio=0.50, s_fg=7.0, s_bg=2.0,
                mask_step=5, mask_type="cfg_magnitude",
                sparse_ffn=True, sparse_blocks=True,
                skip_step_interval=2,
            ),
        },
        {
            "name": "bgq-3-step-only-skip2",
            "desc": "Step-only skip=2 (no sparse blocks)",
            "method": "accel_dual",
            "kwargs": dict(
                skip_ratio=0.50, s_fg=7.0, s_bg=2.0,
                mask_step=5, mask_type="cfg_magnitude",
                sparse_ffn=False, sparse_blocks=False,
                skip_step_interval=2,
            ),
        },
        {
            "name": "bgq-4-step-only-skip3",
            "desc": "Step-only skip=3 (no sparse blocks)",
            "method": "accel_dual",
            "kwargs": dict(
                skip_ratio=0.50, s_fg=7.0, s_bg=2.0,
                mask_step=5, mask_type="cfg_magnitude",
                sparse_ffn=False, sparse_blocks=False,
                skip_step_interval=3,
            ),
        },
    ]

    print(f"=== Background Quality Smoke Test ({len(experiments)} configs) ===")
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

        # Quick metrics
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

    ref_time = all_results["bgq-1-baseline"]["mean_time"]
    ref_clip = all_results["bgq-1-baseline"]["mean_clip"]

    for name, r in all_results.items():
        t = r["mean_time"]
        speedup = ref_time / t
        mc = r["mean_clip"]
        dc = mc - ref_clip
        edge = r["mean_edge"]
        print(f"  {r['desc']:<43} {t:5.2f}s {speedup:5.2f}x {mc:7.4f} {dc:+6.4f} {edge:7.4f}")

    print("=" * 95)
    print()
    print("=== Compare images in outputs/bgq-*/images/ ===")


if __name__ == "__main__":
    main()
