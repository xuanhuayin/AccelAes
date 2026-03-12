#!/usr/bin/env python3
"""
Generate S-DiT images for teaser_fig using the same prompts/seeds.
  p0000-p0019: prompts_dev.txt first 20 prompts, seeds [0,1,2]
  p0020-p0039: prompts_aesthetic.txt 20 prompts, seeds [0,1,2]
Output: outputs/teaser_fig/sdit/images/
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.baselines.sdit import generate_sdit_lumina

OUT_DIR = "outputs/teaser_fig/sdit/images"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]

prompts_dev  = open("prompts/prompts_dev.txt").read().splitlines()[:20]
prompts_aes  = open("prompts/prompts_aesthetic.txt").read().splitlines()

# (filename_index, prompt)
all_prompts = [(i,      p) for i, p in enumerate(prompts_dev)] + \
              [(20 + i, p) for i, p in enumerate(prompts_aes)]

print(f"Total prompts: {len(all_prompts)}  × {len(SEEDS)} seeds = {len(all_prompts)*len(SEEDS)} images")

print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()

times, done, skipped = [], 0, 0
total = len(all_prompts) * len(SEEDS)

for pi, prompt in all_prompts:
    for si, seed in enumerate(SEEDS):
        fname = f"p{pi:04d}_s{si:04d}.png"
        out   = os.path.join(OUT_DIR, fname)
        if os.path.exists(out):
            skipped += 1
            continue
        t0  = time.time()
        img = generate_sdit_lumina(wrapper, prompt=prompt, seed=seed,
                                   cfg_scale=4.0, skip_run=1, ratio=0.5)
        elapsed = time.time() - t0
        img.save(out)
        times.append(elapsed)
        done += 1
        if done % 10 == 0:
            avg = sum(times) / len(times)
            print(f"  [{done}/{total}]  avg={avg:.2f}s  last={fname}", flush=True)

print(f"\nDone: {done} generated, {skipped} skipped")
if times:
    avg = sum(times) / len(times)
    print(f"Avg time: {avg:.2f}s  speedup vs baseline(12.37s): {12.37/avg:.2f}×")
