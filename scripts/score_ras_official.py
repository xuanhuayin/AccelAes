#!/usr/bin/env python3
"""
Score ras_official images against baseline using all 8 metrics.
Reuses the evaluator classes from run_p0_eval.py.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from PIL import Image

PROMPTS_FILE = "prompts/prompts_dev.txt"
N_PROMPTS    = 20
SEEDS        = [0, 1, 2]
BASELINE_DIR = "outputs/p0_compare/baseline/images"
RAS_OFF_DIR  = "outputs/p0_compare/ras_official_static/images"
OUT_JSON     = "outputs/p0_compare/ras_official_static_metrics.json"

# ---- re-use evaluators from run_p0_eval.py ---- #
from scripts.run_p0_eval import (
    PickScoreEvaluator, ImageRewardEvaluator,
    HPSv2Evaluator, AestheticScoreEvaluator as AestheticEvaluator, LPIPSEvaluator,
)
from src.eval.metrics import compute_clip_score, compute_edge_density
import cleanfid.fid as cleanfid_fid

def load_images(folder):
    imgs, prompts_for = [], []
    with open(PROMPTS_FILE) as f:
        prompts = [l.strip() for l in f if l.strip()][:N_PROMPTS]
    for pi, prompt in enumerate(prompts):
        for seed in SEEDS:
            path = os.path.join(folder, f"p{pi:04d}_s{seed:04d}.png")
            img = Image.open(path).convert("RGB")
            imgs.append(img)
            prompts_for.append(prompt)
    return imgs, prompts_for

def main():
    device = "cuda"
    print("Loading images...")
    base_imgs, prompts = load_images(BASELINE_DIR)
    ras_imgs, _        = load_images(RAS_OFF_DIR)
    print(f"  baseline: {len(base_imgs)}, ras_official: {len(ras_imgs)}")

    results = {}

    # CLIP
    print("CLIP...", end="", flush=True)
    from src.eval.metrics import compute_clip_score
    clips = compute_clip_score(ras_imgs, prompts, device=device)
    results["clip"] = float(np.mean(clips))
    torch.cuda.empty_cache()
    print(f" {results['clip']:.4f}")

    # Edge
    print("Edge...", end="", flush=True)
    edges = [compute_edge_density(img) for img in ras_imgs]
    results["edge"] = float(np.mean(edges))
    print(f" {results['edge']:.4f}")

    # PickScore
    print("PickScore...", end="", flush=True)
    ev = PickScoreEvaluator(device)
    results["pick"] = float(np.mean(ev.score(ras_imgs, prompts)))
    del ev; torch.cuda.empty_cache()
    print(f" {results['pick']:.4f}")

    # ImageReward
    print("ImageReward...", end="", flush=True)
    ev = ImageRewardEvaluator(device)
    results["ir"] = float(np.mean(ev.score(ras_imgs, prompts)))
    del ev; torch.cuda.empty_cache()
    print(f" {results['ir']:.4f}")

    # Aesthetic
    print("Aesthetic...", end="", flush=True)
    ev = AestheticEvaluator(device)
    results["aesthetic"] = float(np.mean(ev.score(ras_imgs)))
    del ev; torch.cuda.empty_cache()
    print(f" {results['aesthetic']:.4f}")

    # HPSv2
    print("HPSv2...", end="", flush=True)
    ev = HPSv2Evaluator(device)
    results["hps"] = float(np.mean(ev.score(ras_imgs, prompts, RAS_OFF_DIR)))
    del ev; torch.cuda.empty_cache()
    print(f" {results['hps']:.4f}")

    # LPIPS
    print("LPIPS...", end="", flush=True)
    ev = LPIPSEvaluator(device)
    results["lpips"] = float(np.mean(ev.score(ras_imgs, base_imgs)))
    del ev; torch.cuda.empty_cache()
    print(f" {results['lpips']:.4f}")

    # FID
    print("FID...", end="", flush=True)
    fid = cleanfid_fid.compute_fid(BASELINE_DIR, RAS_OFF_DIR, mode="clean", device=device)
    results["fid"] = float(fid)
    print(f" {results['fid']:.2f}")

    # Timing
    results["time_per_img"] = 7.64
    results["speedup"] = 12.34 / 7.64

    print("\n=== RAS Official Static (no flash_attn, static 50% drop) ===")
    print(f"  Time: {results['time_per_img']:.2f}s/img  Speedup: {results['speedup']:.2f}×")
    print(f"  CLIP={results['clip']:.4f}  Edge={results['edge']:.4f}  Pick={results['pick']:.4f}")
    print(f"  IR={results['ir']:.4f}  HPS={results['hps']:.4f}  Aesth={results['aesthetic']:.4f}")
    print(f"  LPIPS={results['lpips']:.4f}  FID={results['fid']:.2f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")

if __name__ == "__main__":
    main()
