#!/usr/bin/env python3
"""
Generate TeaCache / TaylorSeer images on Lumina for the same 3 teaser prompts.

Outputs (outputs/teaser_samples/):
  teacache_t015_lumina/images/  — TeaCache threshold=0.15  (~1.49×)
  taylor_r1_lumina/images/      — TaylorSeer run=1         (~1.62×)
  taylor_r2_lumina/images/      — TaylorSeer run=2         (~2.05×)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.baselines.teacache import generate_teacache_lumina, generate_taylor_lumina

TS_DIR = "outputs/teaser_samples"

SAMPLES = [
    ("p0004_s0000", 0,
     "Traditional library with floor-to-ceiling bookcases"),
    ("p0007_s0001", 1,
     "Drawing of a towering city, fantasy style, tropical, flying buttresses, "
     "arches, vines, Hanging Gardens of Babylon, Renaissance architecture, "
     "detailed, award winning"),
    ("p0015_s0002", 2,
     "A cartoon mascot helicopter with arms and legs"),
]

METHODS = [
    ("teacache_t015_lumina",
     lambda w, p, s: generate_teacache_lumina(w, prompt=p, seed=s,
                                               cfg_scale=4.0, threshold=0.15)),
    ("taylor_r1_lumina",
     lambda w, p, s: generate_taylor_lumina(w, prompt=p, seed=s,
                                             cfg_scale=4.0, taylor_run=1)),
    ("taylor_r2_lumina",
     lambda w, p, s: generate_taylor_lumina(w, prompt=p, seed=s,
                                             cfg_scale=4.0, taylor_run=2)),
]

print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()

for method_name, fn in METHODS:
    img_dir = os.path.join(TS_DIR, method_name, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"\n=== {method_name} ===")
    for fname, seed, prompt in SAMPLES:
        out = os.path.join(img_dir, f"{fname}.png")
        if os.path.exists(out):
            print(f"  {fname}: already exists, skip")
            continue
        print(f"  {fname} (seed={seed})...", end=" ", flush=True)
        img = fn(wrapper, prompt, seed)
        img.save(out)
        print("done")

print("\nAll done.")
