#!/usr/bin/env python3
"""
Phase 6: Accelerated Dual-Degradation Smoke Test.

Compares 5 configurations:
  1. Baseline (dense, uniform CFG)
  2. Current dual (sparse attn1 only, ~1.48x) — Phase 5
  3. Sparse attn1 + sparse FFN (no step skip)
  4. Sparse attn1 + sparse FFN + skip_interval=3
  5. Sparse attn1 + sparse FFN + skip_interval=2

Validates: no NaN/crash, speed measurements, visual output.
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

    # Common dual params
    dual_kwargs = dict(
        skip_ratio=0.50, mask_type="complexity", grid_size=8,
        s_fg=7.0, s_bg=1.0, mask_step=5,
    )

    experiments = [
        {
            "name": "accel_smoke-1-baseline",
            "desc": "1. Baseline (dense, uniform CFG=4.0)",
            "method": "baseline",
            "kwargs": {},
        },
        {
            "name": "accel_smoke-2-dual-attn1-only",
            "desc": "2. Dual (sparse attn1 only, no FFN sparse)",
            "method": "accel_dual",
            "kwargs": dict(**dual_kwargs, sparse_ffn=False, skip_step_interval=0),
        },
        {
            "name": "accel_smoke-3-dual-attn1-ffn",
            "desc": "3. Dual (sparse attn1 + sparse FFN, no step skip)",
            "method": "accel_dual",
            "kwargs": dict(**dual_kwargs, sparse_ffn=True, skip_step_interval=0),
        },
        {
            "name": "accel_smoke-4-dual-ffn-skip3",
            "desc": "4. Dual (attn1 + FFN + skip_interval=3)",
            "method": "accel_dual",
            "kwargs": dict(**dual_kwargs, sparse_ffn=True, skip_step_interval=3),
        },
        {
            "name": "accel_smoke-5-dual-ffn-skip2",
            "desc": "5. Dual (attn1 + FFN + skip_interval=2)",
            "method": "accel_dual",
            "kwargs": dict(**dual_kwargs, sparse_ffn=True, skip_step_interval=2),
        },
    ]

    print(f"=== Phase 6: Accelerated Dual Smoke Test ({len(experiments)} configs) ===")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Seed: {seed}, Steps: {steps}")
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
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if method == "baseline":
            image = wrapper.generate(
                prompt=prompt, seed=seed,
                cfg_scale=cfg_scale, steps=steps,
                height=resolution, width=resolution,
            )
        elif method == "accel_dual":
            image = wrapper.generate_accelerated_dual(
                prompt=prompt, seed=seed,
                steps=steps, height=resolution, width=resolution,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Check for NaN in output
        import numpy as np
        img_arr = np.array(image)
        assert not np.isnan(img_arr).any(), f"NaN detected in output for {name}!"
        assert img_arr.shape[0] == resolution and img_arr.shape[1] == resolution, \
            f"Wrong output shape: {img_arr.shape}"

        fname = "p0000_s0000.png"
        image.save(os.path.join(img_dir, fname))
        timings[name] = elapsed
        print(f"    OK ({elapsed:.2f}s) -> {os.path.join(img_dir, fname)}")

        del image
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print()
    print("=" * 70)
    print("Speed Summary")
    print("=" * 70)
    baseline_time = timings.get("accel_smoke-1-baseline", None)
    for name, t in timings.items():
        speedup = ""
        if baseline_time and "baseline" not in name:
            speedup = f"  ({baseline_time / t:.2f}x)"
        print(f"  {name:50s} {t:6.2f}s{speedup}")
    print("=" * 70)
    print()
    print("=== All Phase 6 smoke tests passed! ===")


if __name__ == "__main__":
    main()
