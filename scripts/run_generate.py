#!/usr/bin/env python3
"""
Main generation script for SASD experiments.

Usage:
  # Baseline
  python scripts/run_generate.py --config configs/base.yaml \
      --prompts prompts/prompts_dev.txt --seeds 0,1,2,3,4 --exp_name baseline

  # Phase 1: spatial CFG with center mask
  python scripts/run_generate.py --config configs/base.yaml \
      --prompts prompts/prompts_dev.txt --seeds 0,1,2,3,4 \
      --spatial_cfg --s_fg 7.0 --s_bg 1.0 --exp_name p1_spatial

  # Phase 2: auto mask (complexity or semantic)
  python scripts/run_generate.py --config configs/base.yaml \
      --prompts prompts/prompts_dev.txt --seeds 0,1,2,3,4 \
      --mask_type complexity --mask_ratio 0.25 --s_fg 7.0 --s_bg 1.0 \
      --exp_name p2_complexity

  python scripts/run_generate.py --config configs/base.yaml \
      --prompts prompts/prompts_dev.txt --seeds 0,1,2,3,4 \
      --mask_type semantic --mask_ratio 0.25 --s_fg 7.0 --s_bg 1.0 \
      --exp_name p2_semantic

  # Phase 3: token merging acceleration
  python scripts/run_generate.py --config configs/base.yaml \
      --prompts prompts/prompts_dev.txt --seeds 0,1,2 \
      --tome --merge_ratio 0.5 --tome_mask_type complexity \
      --exp_name p3_tome_complexity_r50
"""

import argparse
import os
import sys
import time

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.io_utils import load_config, load_prompts, setup_output_dir, dump_config, dump_env
from src.utils.log import JsonLogger
from src.utils.seed import set_seed
from src.eval.speed import setup_torch_determinism


def save_mask_visualization(mask, heatmap, save_dir, prompt_idx, seed):
    """Save mask and optional heatmap as images for visualization."""
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"p{prompt_idx:04d}_s{seed:04d}"

    # Save mask
    mask_np = mask.cpu().float().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img.save(os.path.join(save_dir, f"{prefix}_mask.png"))

    # Save heatmap if available
    if heatmap is not None:
        hm_np = heatmap.cpu().float().numpy()
        hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-8)
        hm_img = Image.fromarray((hm_np * 255).astype(np.uint8))
        hm_img.save(os.path.join(save_dir, f"{prefix}_heatmap.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of prompts (for debugging)")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="Override cfg_scale from config")

    # Phase 1: spatial CFG with center mask
    parser.add_argument("--spatial_cfg", action="store_true",
                        help="Enable spatial CFG with center mask (Phase 1)")
    parser.add_argument("--s_fg", type=float, default=7.0)
    parser.add_argument("--s_bg", type=float, default=1.0)
    parser.add_argument("--fg_ratio", type=float, default=0.5,
                        help="Center mask foreground ratio (Phase 1)")

    # Phase 2: auto mask
    parser.add_argument("--mask_type", type=str, default=None,
                        choices=["complexity", "semantic"],
                        help="Auto mask type (Phase 2)")
    parser.add_argument("--mask_ratio", type=float, default=0.25,
                        help="Fraction of patches marked as foreground")
    parser.add_argument("--mask_step", type=int, default=5,
                        help="Denoising step at which to compute the mask")
    parser.add_argument("--grid_size", type=int, default=8,
                        help="Grid size for complexity mask")
    parser.add_argument("--blur_sigma", type=float, default=1.0,
                        help="Gaussian blur sigma for semantic mask")
    parser.add_argument("--save_masks", action="store_true",
                        help="Save mask visualizations")

    # Phase 3 (ToMe): token merging
    parser.add_argument("--tome", action="store_true",
                        help="Enable token merging acceleration")
    parser.add_argument("--merge_ratio", type=float, default=0.5,
                        help="Fraction of background tokens to merge (0-1)")
    parser.add_argument("--tome_mask_type", type=str, default="complexity",
                        choices=["complexity", "semantic", "uniform"],
                        help="Mask type for guiding token merging")
    parser.add_argument("--sparsity_schedule", type=str, default="constant",
                        choices=["constant", "cosine"],
                        help="How merge ratio varies across denoising steps")

    # Phase 3 (skip-update): background noise_pred reuse
    parser.add_argument("--skip_update", action="store_true",
                        help="Enable skip-update for background regions")
    parser.add_argument("--skip_ratio", type=float, default=0.25,
                        help="Fraction of patches treated as background (skipped)")
    parser.add_argument("--extrapolation", type=str, default="copy",
                        choices=["copy", "linear", "velocity", "x0"],
                        help="Cache extrapolation method")
    parser.add_argument("--refresh_interval", type=int, default=0,
                        help="Full refresh every N steps after mask_step (0=never)")
    parser.add_argument("--skip_mask_type", type=str, default="complexity",
                        choices=["complexity", "semantic", "uniform"],
                        help="Mask type for skip-update")

    # Phase 4: SDiT baseline upgrades
    parser.add_argument("--region_mask", action="store_true",
                        help="Use SLIC superpixel region mask instead of grid blocks")
    parser.add_argument("--n_segments", type=int, default=64,
                        help="SLIC target number of superpixels")
    parser.add_argument("--slic_compactness", type=float, default=10.0,
                        help="SLIC compactness parameter")
    parser.add_argument("--dilation_radius", type=int, default=0,
                        help="Boundary dilation radius in pixels (0=off)")
    parser.add_argument("--mask_blur_sigma", type=float, default=0.0,
                        help="Boundary Gaussian blur sigma (0=off)")

    # Phase 5: sparse forward (token-level skip)
    parser.add_argument("--sparse_forward", action="store_true",
                        help="Enable sparse forward: skip background attn1 computation")
    parser.add_argument("--sparse_skip_ratio", type=float, default=0.5,
                        help="Fraction of tokens treated as background (skipped in attn1)")
    parser.add_argument("--sparse_mask_type", type=str, default="complexity",
                        choices=["complexity", "semantic", "uniform"],
                        help="Mask type for sparse forward")
    parser.add_argument("--sparse_refresh", type=int, default=0,
                        help="Dense refresh every N steps after mask_step (0=never)")

    # SASD: Dual Degradation (sparse forward + spatial CFG)
    parser.add_argument("--dual_degrade", action="store_true",
                        help="Enable dual degradation: sparse attn1 + spatial CFG suppression")
    parser.add_argument("--dual_skip_ratio", type=float, default=0.5,
                        help="Fraction of tokens treated as background")
    parser.add_argument("--dual_mask_type", type=str, default="complexity",
                        choices=["complexity", "semantic", "uniform"],
                        help="Mask type for dual degradation")
    parser.add_argument("--dual_s_fg", type=float, default=7.0,
                        help="CFG scale for foreground regions")
    parser.add_argument("--dual_s_bg", type=float, default=1.0,
                        help="CFG scale for background (suppression) regions")
    parser.add_argument("--dual_refresh", type=int, default=0,
                        help="Dense refresh every N steps (0=never)")

    # Phase 6: Accelerated Dual (sparse attn1 + sparse FFN + step skip)
    parser.add_argument("--accelerated_dual", action="store_true",
                        help="Enable accelerated dual: sparse attn1 + sparse FFN + step skip")
    parser.add_argument("--skip_step_interval", type=int, default=0,
                        help="Skip transformer forward every N steps after warm-up (0=disabled)")
    parser.add_argument("--sparse_ffn_flag", action="store_true", default=True,
                        help="Enable sparse FFN in accelerated dual (default: True)")
    parser.add_argument("--no_sparse_ffn", action="store_true",
                        help="Disable sparse FFN (attn1 sparse only)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]

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
    logger = JsonLogger(exp_dir)

    # Save config and env
    dump_config(config, exp_dir)
    dump_env(exp_dir)

    # Load model
    model_cfg = config["model"]
    wrapper = LuminaDiTWrapper(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bf16"),
    )

    # Load prompts
    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Generating with {len(prompts)} prompts x {len(seeds)} seeds = {len(prompts) * len(seeds)} images")

    gen_cfg = config["generation"]
    resolution = gen_cfg.get("resolution", 1024)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else gen_cfg.get("cfg_scale", 4.0)
    steps = config["sampler"].get("steps", 30)

    # ---- Determine generation mode ----
    mode = "baseline"
    mask_builder = None

    # Determine if Phase 4 v2 features are requested
    use_v2 = (args.region_mask or args.dilation_radius > 0
              or args.mask_blur_sigma > 0
              or args.extrapolation in ("velocity", "x0"))

    if args.accelerated_dual:
        accel_sparse_ffn = not args.no_sparse_ffn
        mode = (f"accel_dual(mask={args.dual_mask_type}, skip={args.dual_skip_ratio}, "
                f"s_fg={args.dual_s_fg}, s_bg={args.dual_s_bg}, "
                f"skip_step={args.skip_step_interval}, sparse_ffn={accel_sparse_ffn}, "
                f"refresh={args.dual_refresh})")
        print(f"Phase 6 accelerated dual: {mode}")

    elif args.dual_degrade:
        mode = (f"dual_degrade(mask={args.dual_mask_type}, skip={args.dual_skip_ratio}, "
                f"s_fg={args.dual_s_fg}, s_bg={args.dual_s_bg}, refresh={args.dual_refresh})")
        print(f"SASD dual degradation: {mode}")

    elif args.sparse_forward:
        mode = (f"sparse_forward(mask={args.sparse_mask_type}, skip={args.sparse_skip_ratio}, "
                f"refresh={args.sparse_refresh})")
        print(f"Phase 5 sparse forward: {mode}, cfg={cfg_scale}")

    elif args.skip_update:
        mask_label = "SLIC-region" if args.region_mask else f"grid-{args.skip_mask_type}"
        mode = (f"skip_update(mask={mask_label}, skip={args.skip_ratio}, "
                f"extrap={args.extrapolation}, refresh={args.refresh_interval}"
                f", dilate={args.dilation_radius}, blur={args.mask_blur_sigma})")
        version = "v2" if use_v2 else "v1"
        print(f"Phase {'4' if use_v2 else '3'} skip-update ({version}): {mode}, cfg={cfg_scale}")

    elif args.tome:
        mode = f"tome(mask={args.tome_mask_type}, r={args.merge_ratio}, sched={args.sparsity_schedule})"
        print(f"Phase 3: {mode}, cfg={cfg_scale}")

    elif args.mask_type == "complexity":
        from src.sparse.mask_builders import ComplexityMaskBuilder
        mask_builder = ComplexityMaskBuilder(
            grid_size=args.grid_size,
            ratio=args.mask_ratio,
        )
        mode = f"complexity_mask(ratio={args.mask_ratio}, grid={args.grid_size})"
        print(f"Phase 2: {mode}, s_fg={args.s_fg}, s_bg={args.s_bg}")

    elif args.mask_type == "semantic":
        from src.sparse.mask_builders import SemanticMaskBuilder
        mask_builder = SemanticMaskBuilder(
            ratio=args.mask_ratio,
            blur_sigma=args.blur_sigma,
        )
        mode = f"semantic_mask(ratio={args.mask_ratio}, blur={args.blur_sigma})"
        print(f"Phase 2: {mode}, s_fg={args.s_fg}, s_bg={args.s_bg}")

    elif args.spatial_cfg:
        latent_h = resolution // wrapper.vae_scale_factor
        latent_w = resolution // wrapper.vae_scale_factor
        center_mask = wrapper.make_center_mask(latent_h, latent_w, fg_ratio=args.fg_ratio)
        mode = f"spatial_cfg(s_fg={args.s_fg}, s_bg={args.s_bg}, fg={args.fg_ratio})"
        print(f"Phase 1: {mode}")

    # Mask visualization dir
    mask_viz_dir = os.path.join(exp_dir, "mask_visualizations") if args.save_masks else None

    # ---- Generate ----
    img_dir = os.path.join(exp_dir, "images")
    total = len(prompts) * len(seeds)
    count = 0
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            count += 1
            set_seed(seed)
            t0 = time.perf_counter()

            if args.accelerated_dual:
                # Phase 6: accelerated dual (sparse attn1 + FFN + step skip)
                callback = None
                if mask_viz_dir:
                    callback = lambda m, h, _pi=pi, _s=seed: save_mask_visualization(
                        m, h, mask_viz_dir, _pi, _s
                    )
                image = wrapper.generate_accelerated_dual(
                    prompt=prompt, seed=seed,
                    skip_ratio=args.dual_skip_ratio,
                    mask_type=args.dual_mask_type,
                    s_fg=args.dual_s_fg, s_bg=args.dual_s_bg,
                    steps=steps, height=resolution, width=resolution,
                    mask_step=args.mask_step,
                    grid_size=args.grid_size,
                    blur_sigma=args.blur_sigma,
                    refresh_interval=args.dual_refresh,
                    skip_step_interval=args.skip_step_interval,
                    sparse_ffn=not args.no_sparse_ffn,
                    save_mask_callback=callback,
                )
            elif args.dual_degrade:
                # SASD: dual degradation (sparse + spatial CFG)
                callback = None
                if mask_viz_dir:
                    callback = lambda m, h, _pi=pi, _s=seed: save_mask_visualization(
                        m, h, mask_viz_dir, _pi, _s
                    )
                image = wrapper.generate_dual_degradation(
                    prompt=prompt, seed=seed,
                    skip_ratio=args.dual_skip_ratio,
                    mask_type=args.dual_mask_type,
                    s_fg=args.dual_s_fg, s_bg=args.dual_s_bg,
                    steps=steps, height=resolution, width=resolution,
                    mask_step=args.mask_step,
                    grid_size=args.grid_size,
                    blur_sigma=args.blur_sigma,
                    refresh_interval=args.dual_refresh,
                    save_mask_callback=callback,
                )
            elif args.sparse_forward:
                # Phase 5: sparse forward
                callback = None
                if mask_viz_dir:
                    callback = lambda m, h, _pi=pi, _s=seed: save_mask_visualization(
                        m, h, mask_viz_dir, _pi, _s
                    )
                image = wrapper.generate_sparse_forward(
                    prompt=prompt, seed=seed,
                    skip_ratio=args.sparse_skip_ratio,
                    mask_type=args.sparse_mask_type,
                    cfg_scale=cfg_scale, steps=steps,
                    height=resolution, width=resolution,
                    mask_step=args.mask_step,
                    grid_size=args.grid_size,
                    blur_sigma=args.blur_sigma,
                    refresh_interval=args.sparse_refresh,
                    save_mask_callback=callback,
                )
            elif args.skip_update:
                # Phase 3/4: skip-update
                callback = None
                if mask_viz_dir:
                    callback = lambda m, h, _pi=pi, _s=seed: save_mask_visualization(
                        m, h, mask_viz_dir, _pi, _s
                    )
                if use_v2:
                    image = wrapper.generate_skip_update_v2(
                        prompt=prompt, seed=seed,
                        skip_ratio=args.skip_ratio,
                        mask_type=args.skip_mask_type,
                        extrapolation=args.extrapolation,
                        refresh_interval=args.refresh_interval,
                        cfg_scale=cfg_scale, steps=steps,
                        height=resolution, width=resolution,
                        mask_step=args.mask_step,
                        grid_size=args.grid_size,
                        blur_sigma=args.blur_sigma,
                        region_mask=args.region_mask,
                        n_segments=args.n_segments,
                        slic_compactness=args.slic_compactness,
                        dilation_radius=args.dilation_radius,
                        mask_blur_sigma=args.mask_blur_sigma,
                        save_mask_callback=callback,
                    )
                else:
                    image = wrapper.generate_skip_update(
                        prompt=prompt, seed=seed,
                        skip_ratio=args.skip_ratio,
                        mask_type=args.skip_mask_type,
                        extrapolation=args.extrapolation,
                        refresh_interval=args.refresh_interval,
                        cfg_scale=cfg_scale, steps=steps,
                        height=resolution, width=resolution,
                        mask_step=args.mask_step,
                        grid_size=args.grid_size,
                        blur_sigma=args.blur_sigma,
                        save_mask_callback=callback,
                    )
            elif args.tome:
                # Phase 3: token merging
                callback = None
                if mask_viz_dir:
                    callback = lambda m, h, _pi=pi, _s=seed: save_mask_visualization(
                        m, h, mask_viz_dir, _pi, _s
                    )
                image = wrapper.generate_tome(
                    prompt=prompt, seed=seed,
                    merge_ratio=args.merge_ratio,
                    mask_type=args.tome_mask_type,
                    cfg_scale=cfg_scale,
                    steps=steps, height=resolution, width=resolution,
                    mask_step=args.mask_step,
                    mask_ratio=args.mask_ratio,
                    sparsity_schedule=args.sparsity_schedule,
                    grid_size=args.grid_size,
                    blur_sigma=args.blur_sigma,
                    save_mask_callback=callback,
                )
            elif mask_builder is not None:
                # Phase 2: auto mask
                callback = None
                if mask_viz_dir:
                    callback = lambda m, h, _pi=pi, _s=seed: save_mask_visualization(
                        m, h, mask_viz_dir, _pi, _s
                    )
                image = wrapper.generate_with_auto_mask(
                    prompt=prompt, seed=seed,
                    mask_builder=mask_builder,
                    s_fg=args.s_fg, s_bg=args.s_bg,
                    steps=steps, height=resolution, width=resolution,
                    mask_step=args.mask_step,
                    save_mask_callback=callback,
                )
            elif args.spatial_cfg:
                # Phase 1: center mask
                image = wrapper.generate_spatial_cfg(
                    prompt=prompt, seed=seed, mask=center_mask,
                    s_fg=args.s_fg, s_bg=args.s_bg,
                    steps=steps, height=resolution, width=resolution,
                )
            else:
                # Baseline
                image = wrapper.generate(
                    prompt=prompt, seed=seed,
                    cfg_scale=cfg_scale, steps=steps,
                    height=resolution, width=resolution,
                )

            elapsed = time.perf_counter() - t0

            # Save image
            fname = f"p{pi:04d}_s{seed:04d}.png"
            image.save(os.path.join(img_dir, fname))

            # Log
            log_entry = {
                "prompt_idx": pi,
                "prompt": prompt,
                "seed": seed,
                "time_s": elapsed,
                "filename": fname,
                "mode": mode,
            }
            if args.accelerated_dual:
                log_entry.update({
                    "dual_skip_ratio": args.dual_skip_ratio,
                    "dual_mask_type": args.dual_mask_type,
                    "dual_s_fg": args.dual_s_fg,
                    "dual_s_bg": args.dual_s_bg,
                    "dual_refresh": args.dual_refresh,
                    "skip_step_interval": args.skip_step_interval,
                    "sparse_ffn": not args.no_sparse_ffn,
                })
            elif args.dual_degrade:
                log_entry.update({
                    "dual_skip_ratio": args.dual_skip_ratio,
                    "dual_mask_type": args.dual_mask_type,
                    "dual_s_fg": args.dual_s_fg,
                    "dual_s_bg": args.dual_s_bg,
                    "dual_refresh": args.dual_refresh,
                })
            elif args.sparse_forward:
                log_entry.update({
                    "sparse_skip_ratio": args.sparse_skip_ratio,
                    "sparse_mask_type": args.sparse_mask_type,
                    "sparse_refresh": args.sparse_refresh,
                    "cfg_scale": cfg_scale,
                })
            elif args.skip_update:
                log_entry.update({
                    "skip_ratio": args.skip_ratio,
                    "skip_mask_type": args.skip_mask_type,
                    "extrapolation": args.extrapolation,
                    "refresh_interval": args.refresh_interval,
                    "cfg_scale": cfg_scale,
                    "region_mask": args.region_mask,
                    "n_segments": args.n_segments,
                    "slic_compactness": args.slic_compactness,
                    "dilation_radius": args.dilation_radius,
                    "mask_blur_sigma": args.mask_blur_sigma,
                    "version": "v2" if use_v2 else "v1",
                })
            elif args.tome:
                log_entry.update({
                    "merge_ratio": args.merge_ratio,
                    "tome_mask_type": args.tome_mask_type,
                    "sparsity_schedule": args.sparsity_schedule,
                    "cfg_scale": cfg_scale,
                })
            else:
                log_entry.update({
                    "mask_type": args.mask_type,
                    "s_fg": args.s_fg if (args.mask_type or args.spatial_cfg) else cfg_scale,
                    "s_bg": args.s_bg if (args.mask_type or args.spatial_cfg) else cfg_scale,
                })
            logger.log(log_entry)
            print(f"[{count}/{total}] seed={seed} time={elapsed:.2f}s -> {fname}")

    print(f"Done. Output: {exp_dir}")


if __name__ == "__main__":
    main()
