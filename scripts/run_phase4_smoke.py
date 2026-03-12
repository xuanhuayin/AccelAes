#!/usr/bin/env python3
"""
Phase 4 Smoke Test — single-process, loads model once.
Runs 1 prompt × 1 seed for each of the 7 experiment configs.
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
            "name": "smoke_p3-baseline-ref",
            "desc": "P3-baseline-ref (v1 grid-complexity + linear)",
            "method": "v1",
            "kwargs": dict(
                skip_ratio=0.25, mask_type="complexity",
                extrapolation="linear", grid_size=8,
            ),
        },
        {
            "name": "smoke_p4.1-region-only",
            "desc": "P4.1-region-only (SLIC + linear)",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=0.25, mask_type="complexity",
                extrapolation="linear",
                region_mask=True, n_segments=64,
            ),
        },
        {
            "name": "smoke_p4.2-region-soft",
            "desc": "P4.2-region-soft (SLIC + linear + soft boundary)",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=0.25, mask_type="complexity",
                extrapolation="linear",
                region_mask=True, n_segments=64,
                dilation_radius=2, mask_blur_sigma=1.5,
            ),
        },
        {
            "name": "smoke_p4.2-grid-soft",
            "desc": "P4.2-grid-soft (grid + linear + soft boundary)",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=0.25, mask_type="complexity",
                extrapolation="linear", grid_size=8,
                dilation_radius=2, mask_blur_sigma=1.5,
            ),
        },
        {
            "name": "smoke_p4.3-velocity",
            "desc": "P4.3-velocity (SLIC + velocity + soft)",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=0.25, mask_type="complexity",
                extrapolation="velocity",
                region_mask=True, n_segments=64,
                dilation_radius=2, mask_blur_sigma=1.5,
            ),
        },
        {
            "name": "smoke_p4.3-x0",
            "desc": "P4.3-x0 (SLIC + x0 + soft)",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=0.25, mask_type="complexity",
                extrapolation="x0",
                region_mask=True, n_segments=64,
                dilation_radius=2, mask_blur_sigma=1.5,
            ),
        },
        {
            "name": "smoke_p4.3-vel-grid",
            "desc": "P4.3-vel-grid (grid + velocity)",
            "method": "v2",
            "kwargs": dict(
                skip_ratio=0.25, mask_type="complexity",
                extrapolation="velocity", grid_size=8,
            ),
        },
    ]

    print(f"=== Phase 4 Smoke Test ({len(experiments)} configs) ===")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Seed: {seed}, Steps: {steps}, CFG: {cfg_scale}")
    print()

    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp["desc"]
        method = exp["method"]
        kwargs = exp["kwargs"]

        print(f"[{idx+1}/{len(experiments)}] {desc}")

        exp_dir = setup_output_dir(output_base, name)
        img_dir = os.path.join(exp_dir, "images")

        set_seed(seed)
        t0 = time.perf_counter()

        if method == "v1":
            image = wrapper.generate_skip_update(
                prompt=prompt, seed=seed,
                cfg_scale=cfg_scale, steps=steps,
                height=resolution, width=resolution,
                mask_step=5,
                **kwargs,
            )
        else:
            image = wrapper.generate_skip_update_v2(
                prompt=prompt, seed=seed,
                cfg_scale=cfg_scale, steps=steps,
                height=resolution, width=resolution,
                mask_step=5,
                **kwargs,
            )

        elapsed = time.perf_counter() - t0
        fname = "p0000_s0000.png"
        image.save(os.path.join(img_dir, fname))
        print(f"    OK ({elapsed:.2f}s) -> {os.path.join(img_dir, fname)}")

        # Free intermediate tensors
        del image
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=== All smoke tests passed! ===")


if __name__ == "__main__":
    main()
