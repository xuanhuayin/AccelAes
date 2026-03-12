#!/usr/bin/env python3
"""
SDXL Full Evaluation.

Generates images with baseline and SASD configs, then computes metrics.
200 prompts × 3 seeds = 600 images per config.

Usage:
    python scripts/run_sdxl_eval.py
    python scripts/run_sdxl_eval.py --num_prompts 50 --num_seeds 1  # quick test
"""

import os
import sys
import time
import gc
import json
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.sdxl_wrapper import SDXLWrapper
from src.utils.seed import set_seed
from src.eval.metrics import compute_clip_score, compute_edge_density, save_metrics_csv
from src.eval.speed import SpeedBenchmark, setup_torch_determinism


def load_prompts_from_file(path, max_prompts=None):
    """Load prompts from text file."""
    with open(path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if max_prompts:
        prompts = prompts[:max_prompts]
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--prompts_file", type=str, default="prompts/prompts_dev.txt")
    parser.add_argument("--output_dir", type=str, default="outputs/sdxl_eval")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--speed_benchmark", action="store_true",
                        help="Run speed benchmark (5 warmup + 30 measure)")
    args = parser.parse_args()

    setup_torch_determinism()

    wrapper = SDXLWrapper(
        model_name=args.model_name,
        dtype=args.dtype,
    )

    prompts = load_prompts_from_file(args.prompts_file, args.num_prompts)
    seeds = list(range(args.num_seeds))

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output_dir,
    )

    experiments = []

    if not args.skip_baseline:
        experiments.append({
            "name": "baseline",
            "method": "baseline",
            "kwargs": {},
        })

    # SASD configs
    experiments.extend([
        {
            "name": "sasd_cfg_mag_skip50",
            "method": "accelerated",
            "kwargs": dict(
                skip_ratio=0.50,
                mask_type="cfg_magnitude",
                s_fg=9.0, s_bg=2.0,
                mask_step=5,
                sparse_ffn=True,
                sparse_blocks=True,
            ),
        },
        {
            "name": "sasd_cfg_mag_skip50_fullskip4",
            "method": "accelerated",
            "kwargs": dict(
                skip_ratio=0.50,
                mask_type="cfg_magnitude",
                s_fg=9.0, s_bg=2.0,
                mask_step=5,
                sparse_ffn=True,
                sparse_blocks=True,
                full_skip_interval=4,
            ),
        },
        {
            "name": "sasd_uniform_skip50",
            "method": "accelerated",
            "kwargs": dict(
                skip_ratio=0.50,
                mask_type="uniform",
                s_fg=9.0, s_bg=2.0,
                mask_step=5,
                sparse_ffn=True,
                sparse_blocks=True,
            ),
        },
    ])

    total_images = len(prompts) * len(seeds)
    print(f"=== SDXL Evaluation ===")
    print(f"Prompts: {len(prompts)}, Seeds: {len(seeds)}, Total: {total_images} per config")
    print(f"Configs: {len(experiments)}")
    print()

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
                    image = wrapper.generate_accelerated(
                        prompt=prompt, seed=seed,
                        steps=args.steps,
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

            if (pidx + 1) % 20 == 0:
                avg = sum(all_times) / len(all_times)
                print(f"  [{pidx+1}/{len(prompts)}] avg {avg:.2f}s/img")

        # Save logs
        logs_path = os.path.join(exp_dir, "logs.jsonl")
        with open(logs_path, "w") as f:
            for entry in logs:
                f.write(json.dumps(entry) + "\n")

        avg_time = sum(all_times) / len(all_times)
        print(f"  Done: {name}, avg {avg_time:.2f}s/img, total {sum(all_times):.0f}s")

        # Compute metrics
        if not args.skip_metrics:
            print(f"  Computing metrics...")
            from PIL import Image as PILImage

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

            # CLIP score
            clip_scores = compute_clip_score(images, prompt_list)
            for rec, score in zip(records, clip_scores):
                rec["clip_score"] = score
            mean_clip = sum(clip_scores) / len(clip_scores)
            print(f"  Mean CLIP: {mean_clip:.4f}")

            # Edge density
            for rec, img in zip(records, images):
                rec["edge_density"] = compute_edge_density(img)
            mean_edge = sum(r["edge_density"] for r in records) / len(records)
            print(f"  Mean edge density: {mean_edge:.4f}")

            csv_path = os.path.join(exp_dir, "metrics.csv")
            save_metrics_csv(records, csv_path)
            print(f"  Metrics saved to {csv_path}")

            # Cleanup images from memory
            del images
            gc.collect()

        print()

    # Speed benchmark
    if args.speed_benchmark:
        print("=== Speed Benchmark ===")
        bench = SpeedBenchmark(warmup_runs=5, measure_runs=30)
        test_prompt = prompts[0]
        test_seed = 0

        for exp in experiments:
            name = exp["name"]
            method = exp["method"]
            kwargs = exp["kwargs"]

            if method == "baseline":
                fn = lambda: wrapper.generate(
                    prompt=test_prompt, seed=test_seed, steps=args.steps)
            else:
                fn = lambda kw=kwargs: wrapper.generate_accelerated(
                    prompt=test_prompt, seed=test_seed, steps=args.steps, **kw)

            results = bench.benchmark(fn)
            print(f"  {name}: wall={results['wall_mean']:.2f}s ± {results['wall_std']:.2f}s")

            csv_path = os.path.join(output_base, f"speed_{name}.csv")
            SpeedBenchmark.save_csv(results, csv_path)

    print("=== SDXL Evaluation Complete ===")


if __name__ == "__main__":
    main()
