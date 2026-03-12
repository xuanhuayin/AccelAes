#!/usr/bin/env python3
"""
SASD Dual-Degradation Full Evaluation.

Fair comparison: use spatial-CFG-only as quality reference (same s_fg/s_bg),
then compare dual degradation (sparse + CFG) against it.

Also includes original dense baseline for absolute reference.
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


def run_experiment(wrapper, exp_cfg, prompts, seeds, resolution, steps, output_base):
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
                    steps=steps, height=resolution, width=resolution,
                    **kwargs,
                )
            elif method == "spatial_cfg":
                from src.sparse.mask_builders import ComplexityMaskBuilder
                mb = ComplexityMaskBuilder(
                    grid_size=kwargs.pop("grid_size", 8),
                    ratio=kwargs.pop("mask_ratio", 0.5),
                )
                image = wrapper.generate_with_auto_mask(
                    prompt=prompt, seed=seed,
                    mask_builder=mb,
                    steps=steps, height=resolution, width=resolution,
                    mask_step=5,
                    **kwargs,
                )
            elif method == "sparse":
                image = wrapper.generate_sparse_forward(
                    prompt=prompt, seed=seed,
                    steps=steps, height=resolution, width=resolution,
                    mask_step=5,
                    **kwargs,
                )
            elif method == "dual":
                image = wrapper.generate_dual_degradation(
                    prompt=prompt, seed=seed,
                    steps=steps, height=resolution, width=resolution,
                    mask_step=5,
                    **kwargs,
                )

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
    img_dir = exp_result["img_dir"]
    results = exp_result["results"]

    images = []
    prompts_list = []
    for r in results:
        img_path = os.path.join(img_dir, r["filename"])
        images.append(Image.open(img_path).convert("RGB"))
        prompts_list.append(r["prompt"])

    clip_scores = compute_clip_score(images, prompts_list, device=device)
    edge_densities = [compute_edge_density(img) for img in images]

    for r, cs, ed in zip(results, clip_scores, edge_densities):
        r["clip_score"] = cs
        r["edge_density"] = ed

    csv_path = os.path.join(exp_result["exp_dir"], "metrics.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    return {
        "mean_clip": np.mean(clip_scores),
        "std_clip": np.std(clip_scores),
        "mean_edge": np.mean(edge_densities),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="0,1,2")
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
        # Absolute baseline
        {
            "name": "dualeval-baseline-cfg4",
            "desc": "Baseline dense (CFG=4.0)",
            "method": "baseline",
            "kwargs": dict(cfg_scale=4.0),
        },
        # Quality reference: spatial-CFG-only (same s_fg/s_bg as dual)
        {
            "name": "dualeval-spatialcfg-71",
            "desc": "Spatial-CFG-only (s_fg=7, s_bg=1)",
            "method": "spatial_cfg",
            "kwargs": dict(s_fg=7.0, s_bg=1.0, grid_size=8, mask_ratio=0.5),
        },
        # Sparse-only (Phase 5 best: refresh=5)
        {
            "name": "dualeval-sparse-r5",
            "desc": "Sparse-only skip=0.50, refresh=5 (CFG=4)",
            "method": "sparse",
            "kwargs": dict(skip_ratio=0.50, mask_type="complexity", grid_size=8,
                           cfg_scale=4.0, refresh_interval=5),
        },
        # SASD Dual: complexity + refresh=5
        {
            "name": "dualeval-dual-s50-r5",
            "desc": "DUAL skip=0.50, s_fg=7 s_bg=1, refresh=5",
            "method": "dual",
            "kwargs": dict(skip_ratio=0.50, mask_type="complexity", grid_size=8,
                           s_fg=7.0, s_bg=1.0, refresh_interval=5),
        },
        # SASD Dual: no refresh
        {
            "name": "dualeval-dual-s50-norefresh",
            "desc": "DUAL skip=0.50, s_fg=7 s_bg=1, no refresh",
            "method": "dual",
            "kwargs": dict(skip_ratio=0.50, mask_type="complexity", grid_size=8,
                           s_fg=7.0, s_bg=1.0),
        },
        # SASD Dual: more aggressive skip
        {
            "name": "dualeval-dual-s75-r5",
            "desc": "DUAL skip=0.75, s_fg=7 s_bg=1, refresh=5",
            "method": "dual",
            "kwargs": dict(skip_ratio=0.75, mask_type="complexity", grid_size=8,
                           s_fg=7.0, s_bg=1.0, refresh_interval=5),
        },
        # SASD Dual: moderate CFG
        {
            "name": "dualeval-dual-s50-cfg52-r5",
            "desc": "DUAL skip=0.50, s_fg=5 s_bg=2, refresh=5",
            "method": "dual",
            "kwargs": dict(skip_ratio=0.50, mask_type="complexity", grid_size=8,
                           s_fg=5.0, s_bg=2.0, refresh_interval=5),
        },
    ]

    n_images = len(prompts) * len(seeds)
    print(f"=== SASD Dual-Degradation Evaluation ===")
    print(f"{len(prompts)} prompts × {len(seeds)} seeds = {n_images} images/config")
    print(f"{len(experiments)} configs → {n_images * len(experiments)} total images")
    print()

    all_results = {}
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp.get("desc", name)
        print(f"[{idx+1}/{len(experiments)}] {desc}")
        print(f"    Generating {n_images} images...")
        result = run_experiment(wrapper, exp, prompts, seeds, resolution, steps, output_base)
        all_results[name] = result
        print(f"    Done. Mean: {result['mean_time']:.2f}s/img, Total: {result['total_time']:.1f}s")

    # Compute metrics
    print("\nComputing quality metrics...")
    all_metrics = {}
    for name, result in all_results.items():
        print(f"  {name}...")
        all_metrics[name] = evaluate_experiment(result)

    # Summary
    ref_time = all_results["dualeval-baseline-cfg4"]["mean_time"]
    ref_clip = all_metrics["dualeval-spatialcfg-71"]["mean_clip"]  # fair ref for dual

    print()
    print("=" * 95)
    print(f"{'Config':<50} {'Time':>6} {'Speed':>6} {'CLIP':>8} {'vs-ref':>8} {'Edge':>8}")
    print("=" * 95)
    for exp in experiments:
        name = exp["name"]
        desc = exp.get("desc", name)
        t = all_results[name]["mean_time"]
        speedup = ref_time / t
        mc = all_metrics[name]["mean_clip"]
        delta = mc - ref_clip
        edge = all_metrics[name]["mean_edge"]
        print(f"  {desc:<48} {t:5.2f}s {speedup:5.2f}x {mc:7.4f} {delta:+7.4f} {edge:7.4f}")
    print("=" * 95)
    print(f"  (vs-ref = CLIP delta vs Spatial-CFG-only reference)")

    # Save
    summary_path = os.path.join(output_base, "dual_eval_summary.json")
    summary = {}
    for exp in experiments:
        name = exp["name"]
        summary[name] = {
            "desc": exp.get("desc"),
            "mean_time": all_results[name]["mean_time"],
            "speedup_vs_baseline": ref_time / all_results[name]["mean_time"],
            "mean_clip": float(all_metrics[name]["mean_clip"]),
            "std_clip": float(all_metrics[name]["std_clip"]),
            "mean_edge": float(all_metrics[name]["mean_edge"]),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")
    print("=== Done! ===")


if __name__ == "__main__":
    main()
