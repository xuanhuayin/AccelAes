#!/usr/bin/env python3
"""
FLUX sparse attention smoke test.

Generates 4 images (2 prompts × 2 modes) to verify:
1. baseline works
2. generate_accelerated_sparse works (semantic mask + sparse attn + fskip2)

Usage:
  python scripts/smoke_flux_sparse.py
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.flux_wrapper import FluxDiTWrapper
from src.eval.metrics import compute_clip_score, compute_edge_density

PROMPTS = [
    "a majestic lion resting on a golden savanna at sunset, photorealistic",
    "an astronaut floating in space with Earth visible in the background",
]

OUT_DIR = "outputs/smoke_flux_sparse"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("Loading FLUX wrapper...")
    wrapper = FluxDiTWrapper(dtype="bf16")

    results = []

    for name, fn in [
        ("baseline",      lambda p, s: wrapper.generate(p, s, guidance_scale=3.5, steps=28)),
        ("fskip2_w5",     lambda p, s: wrapper.generate_accelerated(p, s, guidance_scale=3.5, steps=28, full_skip_interval=2, warmup_steps=5)),
        ("sparse_fskip2", lambda p, s: wrapper.generate_accelerated_sparse(p, s, guidance_scale=3.5, steps=28, mask_step=4, skip_ratio=0.5, sparse_blocks=True, full_skip_interval=2, warmup_steps=5)),
    ]:
        times = []
        imgs = []
        print(f"\n=== {name} ===")
        for pi, prompt in enumerate(PROMPTS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = fn(prompt, 42)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            imgs.append(img)
            img.save(os.path.join(OUT_DIR, f"{name}_p{pi}.png"))
            print(f"  p{pi}: {elapsed:.2f}s")

        clip = compute_clip_score(imgs, PROMPTS)
        edge = [compute_edge_density(im) for im in imgs]
        import numpy as np
        print(f"  avg time: {sum(times)/len(times):.2f}s  CLIP: {np.mean(clip):.4f}  Edge: {np.mean(edge):.4f}")
        results.append({"name": name, "time": sum(times)/len(times), "clip": float(np.mean(clip)), "edge": float(np.mean(edge))})

    print("\n" + "=" * 60)
    print(f"{'Config':<20} {'Time':>8} {'Speedup':>8} {'CLIP':>8} {'Edge':>8}")
    print("=" * 60)
    ref_t = results[0]["time"]
    for r in results:
        spd = f"{ref_t / r['time']:.2f}x"
        print(f"  {r['name']:<18} {r['time']:7.2f}s {spd:>8}  {r['clip']:7.4f}  {r['edge']:7.4f}")
    print("=" * 60)
    print(f"\nImages saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
