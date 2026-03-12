#!/usr/bin/env python3
"""
SASD Dual-Degradation Smoke Test + Comparison.

Compares:
  1. Baseline (dense, uniform CFG)
  2. Sparse-only (skip attn1, uniform CFG) — Phase 5
  3. Spatial-CFG-only (full compute, s_fg/s_bg) — Phase 2
  4. Dual degradation (sparse attn1 + spatial CFG) — SASD core
  5. Dual with refresh
  6. Dual with semantic mask

Tests the hypothesis: CFG suppression masks sparse artifacts.
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
            "name": "dual_smoke-baseline",
            "desc": "Baseline (dense, uniform CFG=4.0)",
            "method": "baseline",
            "kwargs": {},
        },
        {
            "name": "dual_smoke-sparse-only",
            "desc": "Sparse-only skip=0.50 (uniform CFG=4.0)",
            "method": "sparse",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="complexity", grid_size=8,
                refresh_interval=5,
            ),
        },
        {
            "name": "dual_smoke-spatial-cfg-only",
            "desc": "Spatial-CFG-only (s_fg=7, s_bg=1, no sparse)",
            "method": "spatial_cfg",
            "kwargs": dict(
                mask_type="complexity", s_fg=7.0, s_bg=1.0,
            ),
        },
        {
            "name": "dual_smoke-dual-s50-cfg71",
            "desc": "DUAL: skip=0.50, s_fg=7, s_bg=1",
            "method": "dual",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="complexity", grid_size=8,
                s_fg=7.0, s_bg=1.0,
            ),
        },
        {
            "name": "dual_smoke-dual-s50-cfg71-r5",
            "desc": "DUAL: skip=0.50, s_fg=7, s_bg=1, refresh=5",
            "method": "dual",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="complexity", grid_size=8,
                s_fg=7.0, s_bg=1.0, refresh_interval=5,
            ),
        },
        {
            "name": "dual_smoke-dual-s75-cfg71",
            "desc": "DUAL: skip=0.75, s_fg=7, s_bg=1",
            "method": "dual",
            "kwargs": dict(
                skip_ratio=0.75, mask_type="complexity", grid_size=8,
                s_fg=7.0, s_bg=1.0,
            ),
        },
        {
            "name": "dual_smoke-dual-s50-cfg52",
            "desc": "DUAL: skip=0.50, s_fg=5, s_bg=2",
            "method": "dual",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="complexity", grid_size=8,
                s_fg=5.0, s_bg=2.0,
            ),
        },
        {
            "name": "dual_smoke-dual-semantic-s50",
            "desc": "DUAL: semantic skip=0.50, s_fg=7, s_bg=1",
            "method": "dual",
            "kwargs": dict(
                skip_ratio=0.50, mask_type="semantic", blur_sigma=1.0,
                s_fg=7.0, s_bg=1.0,
            ),
        },
    ]

    print(f"=== SASD Dual-Degradation Smoke Test ({len(experiments)} configs) ===")
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
        elif method == "sparse":
            image = wrapper.generate_sparse_forward(
                prompt=prompt, seed=seed,
                cfg_scale=cfg_scale, steps=steps,
                height=resolution, width=resolution,
                mask_step=5,
                **kwargs,
            )
        elif method == "spatial_cfg":
            # Phase 2 auto mask + spatial CFG (no sparse)
            from src.sparse.mask_builders import ComplexityMaskBuilder
            mb = ComplexityMaskBuilder(grid_size=8, ratio=0.5)
            image = wrapper.generate_with_auto_mask(
                prompt=prompt, seed=seed,
                mask_builder=mb,
                s_fg=kwargs["s_fg"], s_bg=kwargs["s_bg"],
                steps=steps, height=resolution, width=resolution,
                mask_step=5,
            )
        elif method == "dual":
            image = wrapper.generate_dual_degradation(
                prompt=prompt, seed=seed,
                steps=steps, height=resolution, width=resolution,
                mask_step=5,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

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
    baseline_time = timings.get("dual_smoke-baseline", None)
    for name, t in timings.items():
        speedup = ""
        if baseline_time and name != "dual_smoke-baseline":
            speedup = f"  ({baseline_time / t:.2f}x)"
        print(f"  {name:45s} {t:6.2f}s{speedup}")
    print("=" * 70)
    print()
    print("=== All smoke tests passed! ===")


if __name__ == "__main__":
    main()
