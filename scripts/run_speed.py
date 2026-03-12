#!/usr/bin/env python3
"""
Standardized speed benchmarking script.

Usage:
  python scripts/run_speed.py --config configs/base.yaml --exp_name baseline_speed
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.eval.speed import SpeedBenchmark, setup_torch_determinism
from src.utils.io_utils import load_config, setup_output_dir, dump_config, dump_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="speed_baseline")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and a lake, 8k resolution")
    # Spatial CFG args
    parser.add_argument("--spatial_cfg", action="store_true")
    parser.add_argument("--s_fg", type=float, default=7.0)
    parser.add_argument("--s_bg", type=float, default=1.0)
    # Phase 3: token merging args
    parser.add_argument("--tome", action="store_true",
                        help="Enable token merging acceleration")
    parser.add_argument("--merge_ratio", type=float, default=0.5,
                        help="Fraction of background tokens to merge (0-1)")
    parser.add_argument("--tome_mask_type", type=str, default="complexity",
                        choices=["complexity", "semantic", "uniform"],
                        help="Mask type for guiding token merging")
    parser.add_argument("--mask_step", type=int, default=5,
                        help="Denoising step at which to compute the mask")
    parser.add_argument("--sparsity_schedule", type=str, default="constant",
                        choices=["constant", "cosine"],
                        help="How merge ratio varies across denoising steps")
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup torch
    speed_cfg = config.get("speed", {})
    setup_torch_determinism(
        cudnn_benchmark=speed_cfg.get("cudnn_benchmark", False),
        matmul_precision=speed_cfg.get("matmul_precision", "high"),
    )

    # Setup output
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(project_root, config["paths"]["output_dir"])
    exp_dir = setup_output_dir(output_base, args.exp_name)
    dump_config(config, exp_dir)
    dump_env(exp_dir)

    # Load model
    model_cfg = config["model"]
    wrapper = LuminaDiTWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bf16"),
    )

    gen_cfg = config["generation"]
    resolution = gen_cfg.get("resolution", 1024)
    cfg_scale = gen_cfg.get("cfg_scale", 4.0)
    steps = config["sampler"].get("steps", 30)

    # Prepare benchmark function
    if args.tome:
        def gen_fn():
            wrapper.generate_tome(
                prompt=args.prompt, seed=0,
                merge_ratio=args.merge_ratio,
                mask_type=args.tome_mask_type,
                cfg_scale=cfg_scale,
                steps=steps, height=resolution, width=resolution,
                mask_step=args.mask_step,
                sparsity_schedule=args.sparsity_schedule,
            )
    elif args.spatial_cfg:
        latent_h = resolution // wrapper.vae_scale_factor
        latent_w = resolution // wrapper.vae_scale_factor
        mask = wrapper.make_center_mask(latent_h, latent_w)

        def gen_fn():
            wrapper.generate_spatial_cfg(
                prompt=args.prompt, seed=0, mask=mask,
                s_fg=args.s_fg, s_bg=args.s_bg,
                steps=steps, height=resolution, width=resolution,
            )
    else:
        def gen_fn():
            wrapper.generate(
                prompt=args.prompt, seed=0,
                cfg_scale=cfg_scale, steps=steps,
                height=resolution, width=resolution,
            )

    # Run benchmark
    bench = SpeedBenchmark(
        warmup_runs=speed_cfg.get("warmup_runs", 5),
        measure_runs=speed_cfg.get("measure_runs", 30),
    )
    print(f"Benchmarking: warmup={bench.warmup_runs}, measure={bench.measure_runs}")
    results = bench.benchmark(gen_fn)

    # Save
    csv_path = os.path.join(exp_dir, "speed.csv")
    SpeedBenchmark.save_csv(results, csv_path)

    # Report
    print(f"\n--- Speed Results ---")
    print(f"Wall time:  {results['wall_mean']:.3f} +/- {results['wall_std']:.3f}s")
    print(f"GPU time:   {results['gpu_mean']:.3f} +/- {results['gpu_std']:.3f}s")
    print(f"Wall std%:  {results['wall_std'] / results['wall_mean'] * 100:.1f}%")
    print(f"Saved to: {csv_path}")

    if args.tome:
        mode = f"tome(mask={args.tome_mask_type}, r={args.merge_ratio})"
    elif args.spatial_cfg:
        mode = "spatial_cfg"
    else:
        mode = "baseline"
    print(f"Mode: {mode}")


if __name__ == "__main__":
    main()
