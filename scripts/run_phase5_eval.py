#!/usr/bin/env python3
"""
Phase 5 Full Evaluation.

Runs baseline + sparse forward configs on N prompts × M seeds.
Computes CLIP score + edge density for each config.
Outputs a summary table comparing speed and quality.
"""

import os
import sys
import time
import gc
import json
import csv

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.io_utils import load_config, load_prompts, setup_output_dir
from src.utils.seed import set_seed
from src.eval.speed import setup_torch_determinism
from src.eval.metrics import compute_clip_score, compute_edge_density


def run_experiment(wrapper, exp_cfg, prompts, seeds, resolution, steps, cfg_scale, output_base):
    """Run one experiment config across all prompts × seeds. Returns timing + image paths."""
    name = exp_cfg["name"]
    method = exp_cfg["method"]
    kwargs = exp_cfg.get("kwargs", {})

    exp_dir = setup_output_dir(output_base, name)
    img_dir = os.path.join(exp_dir, "images")

    results = []
    total_time = 0.0

    for pi, prompt in enumerate(prompts):
        for seed in seeds:
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
            elif method == "skip_update":
                image = wrapper.generate_skip_update(
                    prompt=prompt, seed=seed,
                    cfg_scale=cfg_scale, steps=steps,
                    height=resolution, width=resolution,
                    mask_step=5,
                    **kwargs,
                )
            elif method == "skip_update_v2":
                image = wrapper.generate_skip_update_v2(
                    prompt=prompt, seed=seed,
                    cfg_scale=cfg_scale, steps=steps,
                    height=resolution, width=resolution,
                    mask_step=5,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            fname = f"p{pi:04d}_s{seed:04d}.png"
            image.save(os.path.join(img_dir, fname))

            results.append({
                "prompt_idx": pi,
                "prompt": prompt,
                "seed": seed,
                "filename": fname,
                "time_s": elapsed,
            })

            del image

    # Save logs
    logs_path = os.path.join(exp_dir, "logs.jsonl")
    with open(logs_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "exp_dir": exp_dir,
        "img_dir": img_dir,
        "results": results,
        "total_time": total_time,
        "mean_time": total_time / len(results) if results else 0,
    }


def evaluate_experiment(exp_result, device="cuda"):
    """Compute CLIP score and edge density for an experiment."""
    img_dir = exp_result["img_dir"]
    results = exp_result["results"]

    images = []
    prompts_list = []
    for r in results:
        img_path = os.path.join(img_dir, r["filename"])
        images.append(Image.open(img_path).convert("RGB"))
        prompts_list.append(r["prompt"])

    # CLIP scores
    clip_scores = compute_clip_score(images, prompts_list, device=device)

    # Edge density
    edge_densities = [compute_edge_density(img) for img in images]

    for r, cs, ed in zip(results, clip_scores, edge_densities):
        r["clip_score"] = cs
        r["edge_density"] = ed

    # Save per-image metrics
    csv_path = os.path.join(exp_result["exp_dir"], "metrics.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    mean_clip = np.mean(clip_scores)
    std_clip = np.std(clip_scores)
    mean_edge = np.mean(edge_densities)

    return {
        "mean_clip": mean_clip,
        "std_clip": std_clip,
        "mean_edge": mean_edge,
        "clip_scores": clip_scores,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prompts", type=int, default=20,
                        help="Number of prompts to evaluate")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seeds")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip CLIP/edge computation (speed-only)")
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

    prompts = load_prompts("prompts/prompts_dev.txt")[:args.n_prompts]
    seeds = [int(s) for s in args.seeds.split(",")]
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)
    cfg_scale = config["generation"].get("cfg_scale", 4.0)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["output_dir"],
    )

    experiments = [
        {
            "name": "p5eval-baseline",
            "desc": "Baseline (dense)",
            "method": "baseline",
        },
        # Phase 5: sparse forward configs
        {
            "name": "p5eval-sparse-complexity-s50",
            "desc": "Sparse complexity skip=0.50",
            "method": "sparse",
            "kwargs": dict(skip_ratio=0.50, mask_type="complexity", grid_size=8),
        },
        {
            "name": "p5eval-sparse-complexity-s50-r5",
            "desc": "Sparse complexity skip=0.50 refresh=5",
            "method": "sparse",
            "kwargs": dict(skip_ratio=0.50, mask_type="complexity", grid_size=8,
                           refresh_interval=5),
        },
        {
            "name": "p5eval-sparse-complexity-s75",
            "desc": "Sparse complexity skip=0.75",
            "method": "sparse",
            "kwargs": dict(skip_ratio=0.75, mask_type="complexity", grid_size=8),
        },
        {
            "name": "p5eval-sparse-semantic-s50",
            "desc": "Sparse semantic skip=0.50",
            "method": "sparse",
            "kwargs": dict(skip_ratio=0.50, mask_type="semantic", blur_sigma=1.0),
        },
        # Phase 3/4 baselines for comparison
        {
            "name": "p5eval-skipupdate-complexity-s25",
            "desc": "Skip-update complexity skip=0.25 (P3 baseline)",
            "method": "skip_update",
            "kwargs": dict(skip_ratio=0.25, mask_type="complexity",
                           extrapolation="copy", grid_size=8),
        },
        {
            "name": "p5eval-skipupdate-v2-velocity",
            "desc": "Skip-update v2 velocity (P4 best)",
            "method": "skip_update_v2",
            "kwargs": dict(skip_ratio=0.25, mask_type="complexity",
                           extrapolation="velocity",
                           region_mask=True, n_segments=64,
                           dilation_radius=2, mask_blur_sigma=1.5),
        },
    ]

    n_images = len(prompts) * len(seeds)
    print(f"=== Phase 5 Full Evaluation ===")
    print(f"{len(prompts)} prompts × {len(seeds)} seeds = {n_images} images per config")
    print(f"{len(experiments)} configs → {n_images * len(experiments)} total images")
    print()

    all_results = {}

    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp.get("desc", name)
        print(f"[{idx+1}/{len(experiments)}] {desc}")
        print(f"    Generating {n_images} images...")

        result = run_experiment(
            wrapper, exp, prompts, seeds,
            resolution, steps, cfg_scale, output_base,
        )
        all_results[name] = result
        print(f"    Done. Mean time: {result['mean_time']:.2f}s/image, Total: {result['total_time']:.1f}s")

    # Compute metrics
    if not args.skip_metrics:
        print()
        print("Computing quality metrics...")
        all_metrics = {}
        for name, result in all_results.items():
            print(f"  Evaluating {name}...")
            metrics = evaluate_experiment(result)
            all_metrics[name] = metrics

        # Summary table
        baseline_time = all_results["p5eval-baseline"]["mean_time"]
        baseline_clip = all_metrics["p5eval-baseline"]["mean_clip"]

        print()
        print("=" * 90)
        print(f"{'Config':<45} {'Time':>6} {'Speed':>6} {'CLIP':>8} {'ΔCLIP':>8} {'Edge':>8}")
        print("=" * 90)
        for exp in experiments:
            name = exp["name"]
            desc = exp.get("desc", name)
            t = all_results[name]["mean_time"]
            speedup = baseline_time / t if t > 0 else 0
            mc = all_metrics[name]["mean_clip"]
            sc = all_metrics[name]["std_clip"]
            delta_clip = mc - baseline_clip
            edge = all_metrics[name]["mean_edge"]
            print(f"  {desc:<43} {t:5.2f}s {speedup:5.2f}x {mc:7.4f} {delta_clip:+7.4f} {edge:7.4f}")
        print("=" * 90)

        # Save summary
        summary_path = os.path.join(output_base, "phase5_eval_summary.json")
        summary = {}
        for exp in experiments:
            name = exp["name"]
            summary[name] = {
                "desc": exp.get("desc", name),
                "mean_time": all_results[name]["mean_time"],
                "total_time": all_results[name]["total_time"],
                "speedup": baseline_time / all_results[name]["mean_time"],
                "mean_clip": float(all_metrics[name]["mean_clip"]),
                "std_clip": float(all_metrics[name]["std_clip"]),
                "mean_edge": float(all_metrics[name]["mean_edge"]),
            }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

    else:
        # Speed-only summary
        baseline_time = all_results["p5eval-baseline"]["mean_time"]
        print()
        print("=" * 60)
        print(f"{'Config':<45} {'Time':>6} {'Speed':>6}")
        print("=" * 60)
        for exp in experiments:
            name = exp["name"]
            desc = exp.get("desc", name)
            t = all_results[name]["mean_time"]
            speedup = baseline_time / t if t > 0 else 0
            print(f"  {desc:<43} {t:5.2f}s {speedup:5.2f}x")
        print("=" * 60)

    print("\n=== Evaluation complete! ===")


if __name__ == "__main__":
    main()
