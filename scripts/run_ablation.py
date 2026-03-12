#!/usr/bin/env python3
"""
Systematic Ablation Experiments for SASD.

Runs ablation across multiple dimensions on the Lumina DiT model:
  - Mask type: cfg_magnitude, complexity, semantic, uniform
  - Skip ratio: 0.25, 0.50, 0.75
  - Sparse FFN: on / off
  - Full skip interval: 0, 3, 4
  - Sparse start step: 6, 10, 15, 20

Each config: 200 prompts × 1 seed = 200 images.
Metrics: CLIP score + edge density + wall-clock time.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --num_prompts 20  # quick test
    python scripts/run_ablation.py --ablation mask_type  # single dimension
"""

import os
import sys
import time
import gc
import json
import argparse
import itertools

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.io_utils import load_config, load_prompts, setup_output_dir
from src.utils.seed import set_seed
from src.eval.metrics import compute_clip_score, compute_edge_density, save_metrics_csv
from src.eval.speed import setup_torch_determinism


# Default config (best known settings)
DEFAULT_CONFIG = dict(
    skip_ratio=0.50,
    mask_type="cfg_magnitude",
    s_fg=7.0,
    s_bg=1.0,
    mask_step=5,
    sparse_ffn=True,
    sparse_blocks=True,
    sparse_start_step=-1,
    full_skip_interval=0,
    blur_sigma=1.0,
)

# Ablation dimensions
ABLATION_CONFIGS = {
    "mask_type": [
        dict(mask_type="cfg_magnitude"),
        dict(mask_type="complexity"),
        dict(mask_type="semantic"),
        dict(mask_type="uniform"),
    ],
    "skip_ratio": [
        dict(skip_ratio=0.25),
        dict(skip_ratio=0.50),
        dict(skip_ratio=0.75),
    ],
    "sparse_ffn": [
        dict(sparse_ffn=True),
        dict(sparse_ffn=False),
    ],
    "full_skip_interval": [
        dict(full_skip_interval=0),
        dict(full_skip_interval=3),
        dict(full_skip_interval=4),
    ],
    "sparse_start_step": [
        dict(sparse_start_step=6),
        dict(sparse_start_step=10),
        dict(sparse_start_step=15),
        dict(sparse_start_step=20),
    ],
}


def make_experiment_name(ablation_dim, override):
    """Create a descriptive experiment name from ablation override."""
    parts = [ablation_dim]
    for k, v in sorted(override.items()):
        parts.append(f"{k}={v}")
    return "_".join(parts)


def run_single_config(wrapper, prompts, seed, config, exp_dir, steps, resolution, cfg_scale):
    """Run a single ablation config and return timing info."""
    img_dir = os.path.join(exp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    logs = []
    all_times = []

    for pidx, prompt in enumerate(prompts):
        set_seed(seed)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        image = wrapper.generate_accelerated_dual(
            prompt=prompt,
            seed=seed,
            steps=steps,
            height=resolution,
            width=resolution,
            cfg_scale=cfg_scale,
            **config,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        all_times.append(elapsed)

        fname = f"p{pidx:04d}_s{seed:04d}.png"
        image.save(os.path.join(img_dir, fname))
        logs.append({
            "filename": fname,
            "prompt_idx": pidx,
            "prompt": prompt,
            "seed": seed,
            "time_s": elapsed,
        })

        del image
        gc.collect()
        torch.cuda.empty_cache()

    # Save logs
    logs_path = os.path.join(exp_dir, "logs.jsonl")
    with open(logs_path, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

    return logs, all_times


def compute_metrics_for_config(exp_dir, logs):
    """Compute CLIP + edge density for a config's outputs."""
    from PIL import Image
    img_dir = os.path.join(exp_dir, "images")

    images = []
    prompt_list = []
    records = []

    for entry in logs:
        img = Image.open(os.path.join(img_dir, entry["filename"])).convert("RGB")
        images.append(img)
        prompt_list.append(entry["prompt"])
        records.append({
            "filename": entry["filename"],
            "prompt_idx": entry["prompt_idx"],
            "seed": entry["seed"],
            "time_s": entry["time_s"],
        })

    # CLIP
    clip_scores = compute_clip_score(images, prompt_list)
    for rec, score in zip(records, clip_scores):
        rec["clip_score"] = score

    # Edge density
    for rec, img in zip(records, images):
        rec["edge_density"] = compute_edge_density(img)

    csv_path = os.path.join(exp_dir, "metrics.csv")
    save_metrics_csv(records, csv_path)

    mean_clip = sum(clip_scores) / len(clip_scores)
    mean_edge = sum(r["edge_density"] for r in records) / len(records)

    del images
    gc.collect()

    return mean_clip, mean_edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", type=str, default=None,
                        choices=list(ABLATION_CONFIGS.keys()),
                        help="Run single ablation dimension (default: all)")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    args = parser.parse_args()

    config = load_config("configs/base.yaml")
    setup_torch_determinism(
        cudnn_benchmark=config.get("speed", {}).get("cudnn_benchmark", False),
        matmul_precision=config.get("speed", {}).get("matmul_precision", "high"),
    )

    wrapper = LuminaDiTWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bf16"),
    )

    prompts = load_prompts(config["paths"]["prompts_dev"])[:args.num_prompts]
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)
    cfg_scale = config["generation"].get("cfg_scale", 4.0)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output_dir,
    )

    # Determine which ablations to run
    if args.ablation:
        ablation_dims = {args.ablation: ABLATION_CONFIGS[args.ablation]}
    else:
        ablation_dims = ABLATION_CONFIGS

    # Also run baseline
    print(f"=== Ablation Study ===")
    print(f"Prompts: {len(prompts)}, Seed: {args.seed}, Steps: {steps}")
    print(f"Ablation dimensions: {list(ablation_dims.keys())}")
    print()

    # Run baseline first
    print("--- Baseline (dense) ---")
    baseline_dir = os.path.join(output_base, "baseline")
    baseline_img_dir = os.path.join(baseline_dir, "images")
    os.makedirs(baseline_img_dir, exist_ok=True)

    baseline_logs = []
    baseline_times = []
    for pidx, prompt in enumerate(prompts):
        set_seed(args.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = wrapper.generate(
            prompt=prompt, seed=args.seed,
            cfg_scale=cfg_scale, steps=steps,
            height=resolution, width=resolution,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        baseline_times.append(elapsed)

        fname = f"p{pidx:04d}_s{args.seed:04d}.png"
        image.save(os.path.join(baseline_img_dir, fname))
        baseline_logs.append({
            "filename": fname, "prompt_idx": pidx,
            "prompt": prompt, "seed": args.seed, "time_s": elapsed,
        })
        del image; gc.collect(); torch.cuda.empty_cache()

        if (pidx + 1) % 50 == 0:
            print(f"  [{pidx+1}/{len(prompts)}]")

    with open(os.path.join(baseline_dir, "logs.jsonl"), "w") as f:
        for entry in baseline_logs:
            f.write(json.dumps(entry) + "\n")

    baseline_avg = sum(baseline_times) / len(baseline_times)
    print(f"  Baseline avg: {baseline_avg:.2f}s/img")

    # Compute baseline metrics
    baseline_clip, baseline_edge = None, None
    if not args.skip_metrics:
        baseline_clip, baseline_edge = compute_metrics_for_config(baseline_dir, baseline_logs)
        print(f"  Baseline CLIP: {baseline_clip:.4f}, Edge: {baseline_edge:.4f}")
    print()

    # Summary table
    summary = [{
        "config": "baseline",
        "avg_time": baseline_avg,
        "speedup": 1.0,
        "clip": baseline_clip,
        "edge": baseline_edge,
    }]

    # Run each ablation
    for dim_name, overrides in ablation_dims.items():
        print(f"=== Ablation: {dim_name} ===")

        for override in overrides:
            exp_name = make_experiment_name(dim_name, override)
            exp_config = {**DEFAULT_CONFIG, **override}

            print(f"  Config: {exp_name}")
            exp_dir = os.path.join(output_base, exp_name)

            logs, times = run_single_config(
                wrapper, prompts, args.seed, exp_config,
                exp_dir, steps, resolution, cfg_scale,
            )

            avg_time = sum(times) / len(times)
            speedup = baseline_avg / avg_time if avg_time > 0 else 0

            clip_score, edge_score = None, None
            if not args.skip_metrics:
                clip_score, edge_score = compute_metrics_for_config(exp_dir, logs)

            summary.append({
                "config": exp_name,
                "avg_time": avg_time,
                "speedup": speedup,
                "clip": clip_score,
                "edge": edge_score,
            })

            print(f"    Time: {avg_time:.2f}s ({speedup:.2f}x)", end="")
            if clip_score is not None:
                print(f"  CLIP: {clip_score:.4f}  Edge: {edge_score:.4f}", end="")
            print()

        print()

    # Save summary
    summary_path = os.path.join(output_base, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Print summary table
    print()
    print("=" * 80)
    print(f"{'Config':<45s} {'Time':>6s} {'Speed':>6s} {'CLIP':>7s} {'Edge':>7s}")
    print("-" * 80)
    for row in summary:
        clip_str = f"{row['clip']:.4f}" if row['clip'] is not None else "  N/A"
        edge_str = f"{row['edge']:.4f}" if row['edge'] is not None else "  N/A"
        print(f"{row['config']:<45s} {row['avg_time']:6.2f}s {row['speedup']:5.2f}x {clip_str:>7s} {edge_str:>7s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
