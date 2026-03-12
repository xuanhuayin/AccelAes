#!/usr/bin/env python3
"""
Unified evaluation: SD3 + FLUX.1-dev, all metrics.

Configs:
  SD3   → baseline | semantic_fskip2
  FLUX  → baseline | fskip2_w5

Metrics:
  CLIP (open_clip ViT-L-14) | ImageReward | edge density

Images saved to outputs/eval/{model}/{config}/p{pi:04d}_s{seed:04d}.png
Summary → outputs/eval/summary.json + summary.csv

Usage:
  python scripts/run_eval.py
  python scripts/run_eval.py --skip_sd3
  python scripts/run_eval.py --skip_flux
  python scripts/run_eval.py --skip_score
  python scripts/run_eval.py --num_prompts 5 --num_seeds 1
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.eval.metrics import compute_clip_score, compute_edge_density


# ------------------------------------------------------------------ #
#  Prompts                                                             #
# ------------------------------------------------------------------ #

def load_prompts(path: str, n: int) -> list:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()][:n]


# ------------------------------------------------------------------ #
#  Image generation helpers                                            #
# ------------------------------------------------------------------ #

def run_config(name: str, fn, prompts: list, seeds: list, out_dir: str):
    """
    Generate images for one config. Skip-if-exists.
    Returns: (imgs, prompts_list, per_image_times)
    """
    os.makedirs(out_dir, exist_ok=True)
    imgs, plist, times = [], [], []
    total = len(prompts) * len(seeds)
    skipped = 0

    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            path = os.path.join(out_dir, f"p{pi:04d}_s{seed:04d}.png")
            if os.path.exists(path):
                imgs.append(Image.open(path).convert("RGB"))
                plist.append(prompt)
                skipped += 1
                continue

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = fn(prompt, seed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            img.save(path)
            imgs.append(img)
            plist.append(prompt)
            times.append(elapsed)

            del img
            gc.collect()
            torch.cuda.empty_cache()

            done = pi * len(seeds) + seeds.index(seed) + 1
            print(f"  [{name}] {done}/{total}  skip={skipped}  last={elapsed:.2f}s")

    mean_t = float(np.mean(times)) if times else None
    print(f"  [{name}] done. skipped={skipped}/{total}  mean_time={mean_t}")
    return imgs, plist, mean_t


# ------------------------------------------------------------------ #
#  Scoring                                                             #
# ------------------------------------------------------------------ #

def load_clip_model(device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, preprocess, tokenizer


def score_clip_batch(model, preprocess, tokenizer, imgs, prompts, device="cuda"):
    scores = []
    for img, prompt in zip(imgs, prompts):
        img_t = preprocess(img).unsqueeze(0).to(device)
        txt_t = tokenizer([prompt]).to(device)
        with torch.no_grad():
            img_f = model.encode_image(img_t)
            txt_f = model.encode_text(txt_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            scores.append((img_f @ txt_f.T).item())
    return scores


def load_ir_model(device="cuda"):
    import ImageReward as RM
    return RM.load("ImageReward-v1.0", device=device)


def score_ir_batch(ir_model, imgs, prompts):
    scores = []
    for img, prompt in zip(imgs, prompts):
        with torch.no_grad():
            s = ir_model.score(prompt, img)
        scores.append(float(s))
    return scores


def score_edge_batch(imgs):
    return [compute_edge_density(img) for img in imgs]


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=30)
    parser.add_argument("--num_seeds",   type=int, default=2)
    parser.add_argument("--steps",       type=int, default=28)
    parser.add_argument("--prompts_file", type=str, default="prompts/prompts_dev.txt")
    parser.add_argument("--output_dir",  type=str, default="outputs/eval")
    parser.add_argument("--skip_sd3",    action="store_true")
    parser.add_argument("--skip_flux",   action="store_true")
    parser.add_argument("--skip_score",  action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file, args.num_prompts)
    seeds   = list(range(args.num_seeds))
    steps   = args.steps
    out     = args.output_dir

    print(f"Eval: {len(prompts)} prompts × {len(seeds)} seeds = "
          f"{len(prompts)*len(seeds)} imgs/config")
    print(f"Output → {out}\n")

    all_results = {}   # {model/config: {mean_time, clip, ir, edge}}

    # ============================================================== #
    #  SD3                                                             #
    # ============================================================== #
    if not args.skip_sd3:
        print("=" * 60)
        print("SD3 medium (stabilityai/stable-diffusion-3-medium-diffusers)")
        print("=" * 60)

        from src.models.sd3_wrapper import SD3DiTWrapper
        sd3 = SD3DiTWrapper(dtype="bf16", device="cuda")

        SD3_CONFIGS = [
            ("baseline",
             lambda p, s: sd3.generate(p, s, steps=steps)),
            ("semantic_fskip2",
             lambda p, s: sd3.generate_accelerated(
                 p, s, steps=steps, mask_type="semantic",
                 s_fg=9.0, s_bg=2.0, full_skip_interval=2,
                 n_segments=64)),
        ]

        sd3_imgs = {}
        sd3_plist = {}
        for name, fn in SD3_CONFIGS:
            print(f"\n--- SD3 / {name} ---")
            img_dir = os.path.join(out, "sd3", name)
            imgs, plist, mean_t = run_config(name, fn, prompts, seeds, img_dir)
            sd3_imgs[name]  = imgs
            sd3_plist[name] = plist
            all_results[f"sd3/{name}"] = {"mean_time": mean_t}

        del sd3
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nSD3 done. GPU freed.\n")

        if not args.skip_score:
            print("Scoring SD3 images...")
            clip_m, clip_pre, clip_tok = load_clip_model()
            ir_m = load_ir_model()

            for name in sd3_imgs:
                imgs, plist = sd3_imgs[name], sd3_plist[name]
                clip_s = score_clip_batch(clip_m, clip_pre, clip_tok, imgs, plist)
                ir_s   = score_ir_batch(ir_m, imgs, plist)
                edge_s = score_edge_batch(imgs)
                all_results[f"sd3/{name}"].update({
                    "clip":  float(np.mean(clip_s)),
                    "ir":    float(np.mean(ir_s)),
                    "edge":  float(np.mean(edge_s)),
                })
                print(f"  sd3/{name}: CLIP={all_results[f'sd3/{name}']['clip']:.4f}  "
                      f"IR={all_results[f'sd3/{name}']['ir']:.4f}  "
                      f"Edge={all_results[f'sd3/{name}']['edge']:.4f}")

            del clip_m, ir_m
            gc.collect()
            torch.cuda.empty_cache()

    # ============================================================== #
    #  FLUX                                                            #
    # ============================================================== #
    if not args.skip_flux:
        print("=" * 60)
        print("FLUX.1-dev (black-forest-labs/FLUX.1-dev)")
        print("=" * 60)

        from src.models.flux_wrapper import FluxDiTWrapper
        flux = FluxDiTWrapper(dtype="bf16", device="cuda")

        # Pre-encode all prompts once (avoids CPU<->GPU swap overhead in timing)
        print("Pre-encoding prompts for FLUX...")
        for p in prompts:
            flux._encode_prompt(p)
        flux._ensure_transformer_on_gpu()
        print(f"  Prompts cached. GPU mem: "
              f"{torch.cuda.memory_allocated()/1024**3:.1f} GB\n")

        FLUX_CONFIGS = [
            ("baseline",
             lambda p, s: flux.generate(p, s, steps=steps, guidance_scale=3.5)),
            ("fskip2_w5",
             lambda p, s: flux.generate_accelerated(
                 p, s, steps=steps, guidance_scale=3.5,
                 full_skip_interval=2, warmup_steps=5)),
            ("sasd_sparse_fskip2",
             lambda p, s: flux.generate_accelerated_sparse(
                 p, s, steps=steps, guidance_scale=3.5,
                 mask_step=4, skip_ratio=0.5, n_segments=64,
                 sparse_blocks=True,
                 full_skip_interval=2, warmup_steps=5)),
        ]

        flux_imgs  = {}
        flux_plist = {}
        for name, fn in FLUX_CONFIGS:
            print(f"\n--- FLUX / {name} ---")
            img_dir = os.path.join(out, "flux", name)
            imgs, plist, mean_t = run_config(name, fn, prompts, seeds, img_dir)
            flux_imgs[name]  = imgs
            flux_plist[name] = plist
            all_results[f"flux/{name}"] = {"mean_time": mean_t}

        del flux
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nFLUX done. GPU freed.\n")

        if not args.skip_score:
            print("Scoring FLUX images...")
            clip_m, clip_pre, clip_tok = load_clip_model()
            ir_m = load_ir_model()

            for name in flux_imgs:
                imgs, plist = flux_imgs[name], flux_plist[name]
                clip_s = score_clip_batch(clip_m, clip_pre, clip_tok, imgs, plist)
                ir_s   = score_ir_batch(ir_m, imgs, plist)
                edge_s = score_edge_batch(imgs)
                all_results[f"flux/{name}"].update({
                    "clip":  float(np.mean(clip_s)),
                    "ir":    float(np.mean(ir_s)),
                    "edge":  float(np.mean(edge_s)),
                })
                print(f"  flux/{name}: CLIP={all_results[f'flux/{name}']['clip']:.4f}  "
                      f"IR={all_results[f'flux/{name}']['ir']:.4f}  "
                      f"Edge={all_results[f'flux/{name}']['edge']:.4f}")

            del clip_m, ir_m
            gc.collect()
            torch.cuda.empty_cache()

    # ============================================================== #
    #  Summary table                                                   #
    # ============================================================== #
    os.makedirs(out, exist_ok=True)

    # Compute speedups within each model
    ref_times = {}
    for key in all_results:
        model, cfg = key.split("/", 1)
        if cfg == "baseline":
            ref_times[model] = all_results[key].get("mean_time")

    print("\n" + "=" * 80)
    print(f"{'config':<30} {'time':>8}  {'speedup':>8}  {'CLIP':>7}  {'IR':>7}  {'Edge':>7}")
    print("=" * 80)

    rows = []
    for key, r in all_results.items():
        model, cfg = key.split("/", 1)
        t   = r.get("mean_time")
        ref = ref_times.get(model)
        spd = f"{ref/t:.2f}x" if (t and ref and cfg != "baseline") else ("1.00x" if cfg == "baseline" else "N/A")
        ts  = f"{t:.2f}s" if t else "N/A   "
        clip_s = f"{r['clip']:.4f}" if "clip"  in r else "N/A   "
        ir_s   = f"{r['ir']:.4f}"   if "ir"    in r else "N/A   "
        edge_s = f"{r['edge']:.4f}" if "edge"  in r else "N/A   "
        print(f"{key:<30} {ts:>8}  {spd:>8}  {clip_s:>7}  {ir_s:>7}  {edge_s:>7}")
        rows.append({
            "config": key, "model": model, "variant": cfg,
            "mean_time": t, "speedup": spd,
            **{k: r.get(k, "") for k in ("clip", "ir", "edge")},
        })
    print("=" * 80)

    # Save JSON
    json_path = os.path.join(out, "summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {json_path}")

    # Save CSV
    csv_path = os.path.join(out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
