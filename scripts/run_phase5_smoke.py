#!/usr/bin/env python3
"""
Phase 5 Smoke Test + Speed Benchmark.

Runs 1 prompt x 1 seed for:
  1. Baseline (dense, no skip)
  2. Sparse forward with complexity mask, skip_ratio=0.5
  3. Sparse forward with complexity mask, skip_ratio=0.75
  4. Sparse forward with semantic mask, skip_ratio=0.5
  5. Sparse forward with uniform mask, skip_ratio=0.5

Reports wall-clock time for each config and computes speedup vs baseline.
"""

import os
import sys
import time
import gc
import torch

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
    prompt = prompts[0]
    seed = 0
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)
    cfg_scale = config["generation"].get("cfg_scale", 4.0)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["output_dir"],
    )

    experiments = [
        {
            "name": "smoke_p5-baseline",
            "desc": "Baseline (dense forward)",
            "method": "baseline",
            "kwargs": {},
        },
        {
            "name": "smoke_p5-complexity-skip50",
            "desc": "Sparse complexity skip=0.50",
            "method": "sparse",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="complexity", grid_size=8,
            ),
        },
        {
            "name": "smoke_p5-complexity-skip75",
            "desc": "Sparse complexity skip=0.75",
            "method": "sparse",
            "kwargs": dict(
                skip_ratio=0.75, mask_type="complexity", grid_size=8,
            ),
        },
        {
            "name": "smoke_p5-semantic-skip50",
            "desc": "Sparse semantic skip=0.50",
            "method": "sparse",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="semantic", blur_sigma=1.0,
            ),
        },
        {
            "name": "smoke_p5-uniform-skip50",
            "desc": "Sparse uniform skip=0.50",
            "method": "sparse",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="uniform",
            ),
        },
        {
            "name": "smoke_p5-complexity-skip50-refresh5",
            "desc": "Sparse complexity skip=0.50 refresh=5",
            "method": "sparse",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="complexity", grid_size=8,
                refresh_interval=5,
            ),
        },
    ]

    print(f"=== Phase 5 Smoke Test ({len(experiments)} configs) ===")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Seed: {seed}, Steps: {steps}, CFG: {cfg_scale}")
    print()

    timings = {}

    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp["desc"]
        method = exp["method"]
        kwargs = exp["kwargs"]

        print(f"[{idx+1}/{len(experiments)}] {desc}")

        exp_dir = setup_output_dir(output_base, name)
        img_dir = os.path.join(exp_dir, "images")

        set_seed(seed)

        # Warm-up CUDA (only for first experiment)
        if idx == 0:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        torch.cuda.synchronize()

        if method == "baseline":
            image = wrapper.generate(
                prompt=prompt, seed=seed,
                cfg_scale=cfg_scale, steps=steps,
                height=resolution, width=resolution,
            )
        else:
            image = wrapper.generate_sparse_forward(
                prompt=prompt, seed=seed,
                cfg_scale=cfg_scale, steps=steps,
                height=resolution, width=resolution,
                mask_step=5,
                **kwargs,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        fname = "p0000_s0000.png"
        image.save(os.path.join(img_dir, fname))
        timings[name] = elapsed
        print(f"    OK ({elapsed:.2f}s) -> {os.path.join(img_dir, fname)}")

        del image
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print()
    print("=" * 60)
    print("Speed Summary")
    print("=" * 60)
    baseline_time = timings.get("smoke_p5-baseline", None)
    for name, t in timings.items():
        speedup = ""
        if baseline_time and name != "smoke_p5-baseline":
            speedup = f"  ({baseline_time / t:.2f}x)"
        print(f"  {name:40s}  {t:6.2f}s{speedup}")
    print("=" * 60)
    print()
    print("=== All Phase 5 smoke tests passed! ===")


if __name__ == "__main__":
    main()
