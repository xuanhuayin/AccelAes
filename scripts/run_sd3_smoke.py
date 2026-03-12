#!/usr/bin/env python3
"""
SD3 Smoke Test.

Runs 3 prompts × 1 seed for baseline and SASD accelerated,
verifying no NaN/crash and saving images for visual inspection.

Usage:
    python scripts/run_sd3_smoke.py
    python scripts/run_sd3_smoke.py --model_name path/to/sd3
"""

import os
import sys
import time
import gc
import argparse
import json

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.sd3_wrapper import SD3Wrapper
from src.utils.seed import set_seed
from src.eval.speed import setup_torch_determinism


SMOKE_PROMPTS = [
    "A corgi sitting on a red couch in a cozy living room, photorealistic",
    "An oil painting of a mountain landscape at sunset, impressionist style",
    "A futuristic city skyline with flying cars, cyberpunk, neon lights, 8k",
]


def check_no_nan(image, name):
    """Verify image has no NaN pixels."""
    arr = np.array(image)
    if np.isnan(arr).any():
        raise RuntimeError(f"NaN detected in {name}!")
    if arr.max() == 0 and arr.min() == 0:
        raise RuntimeError(f"All-black image in {name}!")
    print(f"    [OK] No NaN, pixel range: [{arr.min()}, {arr.max()}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/sd3_smoke")
    args = parser.parse_args()

    setup_torch_determinism()

    wrapper = SD3Wrapper(
        model_name=args.model_name,
        dtype=args.dtype,
    )

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output_dir,
    )

    experiments = [
        {
            "name": "baseline",
            "method": "baseline",
            "kwargs": {},
        },
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
    ]

    print(f"=== SD3 Smoke Test ({len(SMOKE_PROMPTS)} prompts × {len(experiments)} configs) ===")
    print(f"Seed: {args.seed}, Steps: {args.steps}")
    print()

    timings = {}
    logs = []

    for exp in experiments:
        name = exp["name"]
        method = exp["method"]
        kwargs = exp["kwargs"]

        exp_dir = os.path.join(output_base, name)
        img_dir = os.path.join(exp_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        print(f"--- {name} ---")
        exp_times = []

        for pidx, prompt in enumerate(SMOKE_PROMPTS):
            set_seed(args.seed)

            print(f"  [{pidx+1}/{len(SMOKE_PROMPTS)}] {prompt[:50]}...")

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if method == "baseline":
                image = wrapper.generate(
                    prompt=prompt, seed=args.seed,
                    steps=args.steps,
                )
            else:
                image = wrapper.generate_accelerated(
                    prompt=prompt, seed=args.seed,
                    steps=args.steps,
                    **kwargs,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            exp_times.append(elapsed)

            check_no_nan(image, f"{name}/p{pidx}")

            fname = f"p{pidx:04d}_s{args.seed:04d}.png"
            image.save(os.path.join(img_dir, fname))
            logs.append({
                "experiment": name,
                "prompt_idx": pidx,
                "prompt": prompt,
                "seed": args.seed,
                "filename": fname,
                "time_s": elapsed,
            })
            print(f"    Time: {elapsed:.2f}s -> {os.path.join(img_dir, fname)}")

            del image
            gc.collect()
            torch.cuda.empty_cache()

        avg_time = sum(exp_times) / len(exp_times)
        timings[name] = avg_time
        print(f"  Average: {avg_time:.2f}s")
        print()

    # Save logs
    logs_path = os.path.join(output_base, "smoke_logs.jsonl")
    with open(logs_path, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

    # ---- Summary ----
    print("=" * 60)
    print("Speed Summary (avg per image)")
    print("=" * 60)
    baseline_time = timings.get("baseline", None)
    for name, t in timings.items():
        speedup = ""
        if baseline_time and name != "baseline":
            speedup = f"  ({baseline_time / t:.2f}x)"
        print(f"  {name:40s}  {t:6.2f}s{speedup}")
    print("=" * 60)
    print()
    print("=== All SD3 smoke tests passed! ===")
    print(f"Logs saved to {logs_path}")


if __name__ == "__main__":
    main()
