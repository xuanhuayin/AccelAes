#!/usr/bin/env python3
"""Generate 60 images for sparse_from_20 config only."""

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

    prompts = load_prompts("prompts/prompts_dev.txt")[:20]
    seeds = [0, 1, 2]
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["output_dir"],
    )

    kwargs = dict(
        skip_ratio=0.50, s_fg=7.0, s_bg=2.0,
        mask_step=5, mask_type="cfg_magnitude",
        skip_step_interval=2,
        sparse_blocks=True, sparse_ffn=True,
        sparse_start_step=20,
    )

    name = "full_eval-4-hybrid20-cfgfix"
    exp_dir = setup_output_dir(output_base, name)
    img_dir = os.path.join(exp_dir, "images")

    n_images = len(prompts) * len(seeds)
    print(f"=== Generating {n_images} images: sparse_from_20 ===")

    total_time = 0.0
    count = 0
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            set_seed(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            image = wrapper.generate_accelerated_dual(
                prompt=prompt, seed=seed,
                steps=steps, height=resolution, width=resolution,
                **kwargs,
            )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            count += 1

            fname = f"p{pi:04d}_s{seed:04d}.png"
            image.save(os.path.join(img_dir, fname))
            print(f"  [{count}/{n_images}] p{pi}_s{seed}: {elapsed:.2f}s")

            del image
            gc.collect()
            torch.cuda.empty_cache()

    mean_time = total_time / count
    print(f"\nDone. Mean: {mean_time:.2f}s/img, Total: {total_time:.1f}s")
    print(f"Speedup vs 12.54s baseline: {12.54/mean_time:.2f}x")


if __name__ == "__main__":
    main()
