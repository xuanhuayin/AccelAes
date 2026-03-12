#!/usr/bin/env python3
"""
Compare semantic mask vs cfg_magnitude vs complexity vs uniform masks.

Runs a subset of prompts with each mask type using the full SASD pipeline
(sparse attn + sparse FFN + spatial CFG + step skip).

Usage:
    python scripts/run_semantic_mask_compare.py
    python scripts/run_semantic_mask_compare.py --num_prompts 50
"""

import os
import sys
import time
import gc
import json
import argparse

import torch
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.seed import set_seed
from src.eval.metrics import compute_clip_score, compute_edge_density, save_metrics_csv
from src.eval.speed import setup_torch_determinism


def load_prompts(path, max_prompts=None):
    with open(path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if max_prompts:
        prompts = prompts[:max_prompts]
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--prompts_file", type=str, default="prompts/prompts_dev.txt")
    parser.add_argument("--output_dir", type=str, default="outputs/mask_compare")
    parser.add_argument("--skip_ratio", type=float, default=0.50)
    parser.add_argument("--full_skip_interval", type=int, default=4)
    args = parser.parse_args()

    setup_torch_determinism()

    wrapper = LuminaDiTWrapper()

    prompts = load_prompts(args.prompts_file, args.num_prompts)
    seeds = list(range(args.num_seeds))

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output_dir,
    )

    # Shared parameters for all mask types
    shared_kwargs = dict(
        skip_ratio=args.skip_ratio,
        s_fg=7.0, s_bg=1.0,
        steps=args.steps,
        mask_step=5,
        sparse_ffn=True,
        sparse_blocks=True,
        full_skip_interval=args.full_skip_interval,
        blur_sigma=1.0,
    )

    experiments = [
        {
            "name": "baseline",
            "method": "baseline",
            "kwargs": {},
        },
        {
            "name": "sasd_semantic",
            "method": "accelerated_dual",
            "kwargs": {**shared_kwargs, "mask_type": "semantic"},
        },
        {
            "name": "sasd_cfg_magnitude",
            "method": "accelerated_dual",
            "kwargs": {**shared_kwargs, "mask_type": "cfg_magnitude"},
        },
        {
            "name": "sasd_complexity",
            "method": "accelerated_dual",
            "kwargs": {**shared_kwargs, "mask_type": "complexity"},
        },
        {
            "name": "sasd_uniform",
            "method": "accelerated_dual",
            "kwargs": {**shared_kwargs, "mask_type": "uniform"},
        },
    ]

    total_images = len(prompts) * len(seeds)
    print(f"=== Mask Type Comparison ===")
    print(f"Prompts: {len(prompts)}, Seeds: {len(seeds)}, Total: {total_images} per config")
    print(f"Configs: {len(experiments)}")
    print(f"Shared params: skip_ratio={args.skip_ratio}, full_skip_interval={args.full_skip_interval}")
    print()

    all_results = {}

    for exp in experiments:
        name = exp["name"]
        method = exp["method"]
        kwargs = exp["kwargs"]

        exp_dir = os.path.join(output_base, name)
        img_dir = os.path.join(exp_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        logs = []
        all_times = []

        print(f"--- Generating: {name} ({total_images} images) ---")

        for pidx, prompt in enumerate(prompts):
            for seed in seeds:
                set_seed(seed)

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                if method == "baseline":
                    image = wrapper.generate(
                        prompt=prompt, seed=seed,
                        steps=args.steps,
                    )
                else:
                    image = wrapper.generate_accelerated_dual(
                        prompt=prompt, seed=seed,
                        **kwargs,
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

            if (pidx + 1) % 10 == 0:
                avg = sum(all_times) / len(all_times)
                print(f"  [{pidx+1}/{len(prompts)}] avg {avg:.2f}s/img")

        # Save logs
        logs_path = os.path.join(exp_dir, "logs.jsonl")
        with open(logs_path, "w") as f:
            for entry in logs:
                f.write(json.dumps(entry) + "\n")

        avg_time = sum(all_times) / len(all_times)

        # Compute metrics
        print(f"  Computing metrics...")
        images = []
        prompt_list = []
        records = []
        for entry in logs:
            img = PILImage.open(os.path.join(img_dir, entry["filename"])).convert("RGB")
            images.append(img)
            prompt_list.append(entry["prompt"])
            records.append({
                "filename": entry["filename"],
                "prompt_idx": entry["prompt_idx"],
                "seed": entry["seed"],
            })

        clip_scores = compute_clip_score(images, prompt_list)
        for rec, score in zip(records, clip_scores):
            rec["clip_score"] = score
        mean_clip = sum(clip_scores) / len(clip_scores)

        for rec, img in zip(records, images):
            rec["edge_density"] = compute_edge_density(img)
        mean_edge = sum(r["edge_density"] for r in records) / len(records)

        csv_path = os.path.join(exp_dir, "metrics.csv")
        save_metrics_csv(records, csv_path)

        all_results[name] = {
            "avg_time": avg_time,
            "mean_clip": mean_clip,
            "mean_edge": mean_edge,
        }

        print(f"  Done: {name}, avg {avg_time:.2f}s/img, CLIP={mean_clip:.4f}, Edge={mean_edge:.4f}")
        print()

        del images
        gc.collect()

    # ---- Summary Table ----
    print("=" * 80)
    print(f"{'Config':<30} {'Time/img':>10} {'CLIP↑':>10} {'Edge↑':>10} {'Speedup':>10}")
    print("=" * 80)
    baseline_time = all_results.get("baseline", {}).get("avg_time", None)
    for name, res in all_results.items():
        speedup = ""
        if baseline_time and name != "baseline":
            speedup = f"{baseline_time / res['avg_time']:.2f}x"
        print(f"  {name:<28} {res['avg_time']:>8.2f}s {res['mean_clip']:>10.4f} {res['mean_edge']:>10.4f} {speedup:>10}")
    print("=" * 80)

    # Save summary
    summary_path = os.path.join(output_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
