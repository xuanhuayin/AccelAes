#!/usr/bin/env python3
"""Generate SD3 baseline and fskip2_only images for the 4 selected prompts."""
import os, sys, gc
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.sd3_wrapper import SD3DiTWrapper

PROMPT_FILE = "prompts/prompts_dev.txt"
PROMPT_IDS  = [3, 6, 7, 14]
SEED        = 0
STEPS       = 28
CFG         = 7.0

with open(PROMPT_FILE) as f:
    prompts = [l.strip() for l in f if l.strip()]

selected = [(i, prompts[i]) for i in PROMPT_IDS]

wrapper = SD3DiTWrapper(dtype="bf16")

for out_dir, gen_kwargs in [
    ("outputs/sd3_compare/baseline_gen",   {}),
    ("outputs/sd3_compare/fskip2_gen",     {"full_skip_interval": 2}),
]:
    os.makedirs(out_dir, exist_ok=True)
    for pi, prompt in selected:
        out_path = os.path.join(out_dir, f"p{pi:04d}_s{SEED:04d}.png")
        if os.path.exists(out_path):
            print(f"  skip {out_path}")
            continue
        print(f"  [{out_dir}] p{pi}: {prompt[:50]}")
        if gen_kwargs:
            img = wrapper.generate_accelerated(
                prompt=prompt, seed=SEED, steps=STEPS,
                cfg_scale=CFG, **gen_kwargs)
        else:
            img = wrapper.generate(
                prompt=prompt, seed=SEED, steps=STEPS,
                cfg_scale=CFG)
        img.save(out_path)
        del img; gc.collect(); torch.cuda.empty_cache()
        print(f"    saved → {out_path}")

del wrapper
gc.collect(); torch.cuda.empty_cache()
print("Done.")
