#!/usr/bin/env python3
"""
SD3 Full Evaluation (current SASD API).

30 prompts × 2 seeds = 60 images per config. Skip-if-exists safe.

Usage:
    python scripts/run_sd3_eval.py
    python scripts/run_sd3_eval.py --num_prompts 10 --num_seeds 1
"""

import os
import sys
import time
import gc
import json
import argparse

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.sd3_wrapper import SD3DiTWrapper
from src.eval.metrics import compute_clip_score, compute_edge_density


def load_prompts(path, n):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()][:n]


def run_experiment(wrapper, exp_name, fn, prompts, seeds, out_root):
    img_dir = os.path.join(out_root, exp_name)
    os.makedirs(img_dir, exist_ok=True)

    times, imgs, plist = [], [], []
    total = len(prompts) * len(seeds)
    skipped = 0

    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            out_path = os.path.join(img_dir, f'p{pi:04d}_s{seed:04d}.png')
            if os.path.exists(out_path):
                imgs.append(Image.open(out_path).convert('RGB'))
                plist.append(prompt)
                skipped += 1
                continue

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = fn(prompt, seed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            img.save(out_path)
            imgs.append(img)
            plist.append(prompt)
            times.append(elapsed)

            del img; gc.collect(); torch.cuda.empty_cache()

            done = pi * len(seeds) + seeds.index(seed) + 1
            print(f'  [{exp_name}] {done}/{total} skip={skipped} last={elapsed:.2f}s')

    mean_time = float(np.mean(times)) if times else None
    print(f'  [{exp_name}] done. skipped={skipped}/{total}, mean_time={mean_time}')
    return imgs, plist, mean_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_prompts', type=int, default=30)
    parser.add_argument('--num_seeds', type=int, default=2)
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--prompts_file', type=str, default='prompts/prompts_dev.txt')
    parser.add_argument('--output_dir', type=str, default='outputs/sd3_eval')
    parser.add_argument('--dtype', type=str, default='bf16')
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file, args.num_prompts)
    seeds = list(range(args.num_seeds))
    steps = args.steps

    print(f'SD3 eval: {len(prompts)} prompts × {len(seeds)} seeds = {len(prompts)*len(seeds)} imgs/exp')
    wrapper = SD3DiTWrapper(dtype=args.dtype)

    EXPERIMENTS = [
        ('baseline',
         lambda p, s: wrapper.generate(p, s, steps=steps)),
        ('cfg_mag_fskip2',
         lambda p, s: wrapper.generate_accelerated(
             p, s, steps=steps, mask_type='cfg_magnitude',
             s_fg=9.0, s_bg=2.0, full_skip_interval=2,
             sparse_ffn=False, n_segments=64)),
        ('semantic_fskip2',
         lambda p, s: wrapper.generate_accelerated(
             p, s, steps=steps, mask_type='semantic',
             s_fg=9.0, s_bg=2.0, full_skip_interval=2,
             sparse_ffn=False, n_segments=64)),
    ]

    results = {}
    for name, fn in EXPERIMENTS:
        print(f'\n=== {name} ===')
        imgs, plist, mean_time = run_experiment(
            wrapper, name, fn, prompts, seeds, args.output_dir)

        clip_scores = compute_clip_score(imgs, plist)
        edge_scores = [compute_edge_density(img) for img in imgs]

        results[name] = {
            'mean_time': mean_time,
            'clip': float(np.mean(clip_scores)),
            'edge': float(np.mean(edge_scores)),
        }
        print(f'  CLIP={results[name]["clip"]:.4f}  Edge={results[name]["edge"]:.4f}')

    # Summary
    ref_time = results['baseline']['mean_time']
    print('\n' + '=' * 70)
    print(f'{"config":<30} {"time":>7}  {"speedup":>7}  {"CLIP":>7}  {"Edge":>7}')
    print('=' * 70)
    for name, r in results.items():
        t = r['mean_time']
        spd = f'{ref_time/t:.2f}x' if (t and ref_time) else 'N/A'
        ts = f'{t:.2f}s' if t else 'N/A  '
        print(f'{name:<30} {ts:>7}  {spd:>7}  {r["clip"]:7.4f}  {r["edge"]:7.4f}')
    print('=' * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Summary saved to {args.output_dir}/summary.json')


if __name__ == '__main__':
    main()
