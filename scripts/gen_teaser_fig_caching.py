#!/usr/bin/env python3
"""
Generate TeaCache t=0.15 and TaylorSeer r=2 images for teaser_fig
(same 20 prompts × seeds [0,1,2] = 60 images per method).

Outputs:
  outputs/teaser_fig/teacache_t015/images/
  outputs/teaser_fig/taylor_r2/images/
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.baselines.teacache import generate_teacache_lumina, generate_taylor_lumina

OUT_BASE = "outputs/teaser_fig"
PROMPT_FILE = "prompts/prompts_dev.txt"
N_PROMPTS = 20
SEEDS = [0, 1, 2]

prompts = open(PROMPT_FILE).read().splitlines()[:N_PROMPTS]

METHODS = [
    ("teacache_t015",
     lambda w, p, s: generate_teacache_lumina(w, prompt=p, seed=s,
                                               cfg_scale=4.0, threshold=0.15)),
    ("taylor_r2",
     lambda w, p, s: generate_taylor_lumina(w, prompt=p, seed=s,
                                             cfg_scale=4.0, taylor_run=2)),
]

print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()

for method_name, fn in METHODS:
    img_dir = os.path.join(OUT_BASE, method_name, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"\n=== {method_name} ===")
    done = skipped = 0
    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(SEEDS):
            fname = f"p{pi:04d}_s{si:04d}.png"
            out = os.path.join(img_dir, fname)
            if os.path.exists(out):
                skipped += 1
                continue
            img = fn(wrapper, prompt, seed)
            img.save(out)
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{N_PROMPTS*len(SEEDS)} done", flush=True)
    print(f"  Done: {done} generated, {skipped} skipped")

print("\nAll done.")
