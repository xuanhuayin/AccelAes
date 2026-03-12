#!/usr/bin/env python3
"""
Generate exactly the 3 teaser samples for each baseline method.
Only runs 9 images (3 prompts × 3 methods: delta_dit / fora / ras).
baseline + sasd_full images already exist in p0_ablation_direct.

Output: outputs/teaser_samples/{method}/images/{fname}.png
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.baselines.delta_dit import generate_delta_dit
from src.baselines.fora      import generate_fora
from src.baselines.ras       import generate_ras

# ── prompts (0-indexed from prompts_dev.txt) ───────────────────────────────
PROMPT_FILE = "prompts/prompts_dev.txt"
with open(PROMPT_FILE) as f:
    ALL_PROMPTS = [l.strip() for l in f if l.strip()]

# exactly the 3 samples used in the teaser
SAMPLES = [
    (4,  0, "p0004_s0000"),   # "Traditional library..."
    (7,  1, "p0007_s0001"),   # "Drawing of a towering city..."
    (15, 2, "p0015_s0002"),   # "A cartoon mascot helicopter..."
]

METHODS = [
    ("delta_dit", generate_delta_dit),
    ("fora",      generate_fora),
    ("ras",       generate_ras),
]

OUT_BASE = "outputs/teaser_samples"

def main():
    print("Loading Lumina wrapper...")
    wrapper = LuminaDiTWrapper()

    for method_name, gen_fn in METHODS:
        img_dir = os.path.join(OUT_BASE, method_name, "images")
        os.makedirs(img_dir, exist_ok=True)
        print(f"\n=== {method_name} ===")

        for pi, seed, fname in SAMPLES:
            out_path = os.path.join(img_dir, f"{fname}.png")
            if os.path.exists(out_path):
                print(f"  skip (exists): {fname}")
                continue

            prompt = ALL_PROMPTS[pi]
            print(f"  [{fname}] seed={seed}  \"{prompt[:60]}...\"")

            img = gen_fn(wrapper, prompt=prompt, seed=seed)
            img.save(out_path)
            print(f"  -> saved {out_path}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
