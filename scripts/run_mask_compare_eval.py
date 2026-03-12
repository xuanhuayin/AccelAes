#!/usr/bin/env python3
"""
Mask type comparison full evaluation.

Compares 4 mask strategies (all semantic/region-based, no grid):
  1. Baseline (dense, uniform CFG=4.0)
  2. Semantic (cross-attention + CLIP aesthetic anchors)
  3. SDiT (gradient complexity + SLIC regions)
  4. CFG magnitude (|cond - uncond| + SLIC regions)

All accelerated configs use: sparse attn1 + sparse FFN + skip_interval=2.

Metrics: CLIPScore, PickScore, ImageReward, EdgeDensity, Speed.
20 prompts x 3 seeds = 60 images per config, 4 configs = 240 images total.
"""

import os
import sys
import time
import gc
import json
import csv

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.io_utils import load_config, load_prompts, setup_output_dir
from src.utils.seed import set_seed
from src.eval.speed import setup_torch_determinism
from src.eval.metrics import compute_clip_score, compute_edge_density


# ---- Metric loaders ----

class PickScoreEvaluator:
    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, AutoModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    @torch.no_grad()
    def score(self, images, prompts):
        scores = []
        for img, prompt in zip(images, prompts):
            inputs = self.processor(
                text=[prompt], images=[img],
                return_tensors="pt", padding=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            s = (img_emb @ txt_emb.T).item()
            scores.append(s)
        return scores


class ImageRewardEvaluator:
    def __init__(self, device="cuda"):
        import ImageReward as ir_module
        self.model = ir_module.load("ImageReward-v1.0", device=device)
        self.device = device

    @torch.no_grad()
    def score(self, images, prompts):
        scores = []
        for img, prompt in zip(images, prompts):
            s = self.model.score(prompt, img)
            scores.append(float(s))
        return scores


# ---- Generation runner ----

def run_experiment(wrapper, exp_cfg, prompts, seeds, resolution, steps, output_base):
    name = exp_cfg["name"]
    method = exp_cfg["method"]
    accel_kwargs = exp_cfg.get("accel_kwargs", {})

    exp_dir = setup_output_dir(output_base, name)
    img_dir = os.path.join(exp_dir, "images")

    results = []
    total_time = 0.0

    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            set_seed(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if method == "baseline":
                image = wrapper.generate(
                    prompt=prompt, seed=seed,
                    cfg_scale=4.0, steps=steps,
                    height=resolution, width=resolution,
                )
            elif method == "accel_dual":
                image = wrapper.generate_accelerated_dual(
                    prompt=prompt, seed=seed,
                    steps=steps, height=resolution, width=resolution,
                    **accel_kwargs,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            fname = f"p{pi:04d}_s{seed:04d}.png"
            image.save(os.path.join(img_dir, fname))

            results.append({
                "prompt_idx": pi,
                "prompt": prompt,
                "seed": seed,
                "filename": fname,
                "time_s": elapsed,
            })
            del image

    logs_path = os.path.join(exp_dir, "logs.jsonl")
    with open(logs_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "exp_dir": exp_dir,
        "img_dir": img_dir,
        "results": results,
        "total_time": total_time,
        "mean_time": total_time / len(results) if results else 0,
    }


def evaluate_experiment(exp_result, pick_evaluator=None, ir_evaluator=None, device="cuda"):
    img_dir = exp_result["img_dir"]
    results = exp_result["results"]

    images = []
    prompts_list = []
    for r in results:
        img_path = os.path.join(img_dir, r["filename"])
        images.append(Image.open(img_path).convert("RGB"))
        prompts_list.append(r["prompt"])

    print("    CLIPScore...", end="", flush=True)
    clip_scores = compute_clip_score(images, prompts_list, device=device)
    print(" done")

    pick_scores = None
    if pick_evaluator is not None:
        print("    PickScore...", end="", flush=True)
        pick_scores = pick_evaluator.score(images, prompts_list)
        print(" done")

    ir_scores = None
    if ir_evaluator is not None:
        print("    ImageReward...", end="", flush=True)
        ir_scores = ir_evaluator.score(images, prompts_list)
        print(" done")

    print("    EdgeDensity...", end="", flush=True)
    edge_densities = [compute_edge_density(img) for img in images]
    print(" done")

    for idx, r in enumerate(results):
        r["clip_score"] = clip_scores[idx]
        r["edge_density"] = edge_densities[idx]
        if pick_scores is not None:
            r["pick_score"] = pick_scores[idx]
        if ir_scores is not None:
            r["image_reward"] = ir_scores[idx]

    csv_path = os.path.join(exp_result["exp_dir"], "metrics.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "mean_clip": float(np.mean(clip_scores)),
        "std_clip": float(np.std(clip_scores)),
        "mean_edge": float(np.mean(edge_densities)),
    }
    if pick_scores is not None:
        summary["mean_pick"] = float(np.mean(pick_scores))
        summary["std_pick"] = float(np.std(pick_scores))
    if ir_scores is not None:
        summary["mean_ir"] = float(np.mean(ir_scores))
        summary["std_ir"] = float(np.std(ir_scores))

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--no_pickscore", action="store_true")
    parser.add_argument("--no_imagereward", action="store_true")
    args = parser.parse_args()

    config = load_config("configs/base.yaml")
    setup_torch_determinism(
        cudnn_benchmark=config.get("speed", {}).get("cudnn_benchmark", False),
        matmul_precision=config.get("speed", {}).get("matmul_precision", "high"),
    )

    wrapper = LuminaDiTWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bf16"),
    )

    prompts = load_prompts("prompts/prompts_dev.txt")[:args.n_prompts]
    seeds = [int(s) for s in args.seeds.split(",")]
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["paths"]["output_dir"],
    )

    # Common accel params (best speedup config)
    accel_base = dict(
        skip_ratio=0.50,
        s_fg=7.0, s_bg=1.0,
        mask_step=5,
        sparse_ffn=True,
        skip_step_interval=2,
    )

    experiments = [
        {
            "name": "mask_eval-1-baseline",
            "desc": "Baseline (dense, CFG=4.0)",
            "method": "baseline",
        },
        {
            "name": "mask_eval-2-semantic",
            "desc": "Semantic (cross-attn + CLIP)",
            "method": "accel_dual",
            "accel_kwargs": dict(**accel_base, mask_type="semantic"),
        },
        {
            "name": "mask_eval-3-sdit",
            "desc": "SDiT (complexity + SLIC)",
            "method": "accel_dual",
            "accel_kwargs": dict(**accel_base, mask_type="sdit"),
        },
        {
            "name": "mask_eval-4-cfg-magnitude",
            "desc": "CFG magnitude + SLIC",
            "method": "accel_dual",
            "accel_kwargs": dict(**accel_base, mask_type="cfg_magnitude"),
        },
    ]

    n_images = len(prompts) * len(seeds)
    total_images = n_images * len(experiments)
    print(f"=== Mask Type Comparison Full Evaluation ===")
    print(f"{len(prompts)} prompts x {len(seeds)} seeds = {n_images} images/config")
    print(f"{len(experiments)} configs -> {total_images} total images")
    print()

    # ---- Phase A: Generate ----
    print("=" * 70)
    print("PHASE A: Image Generation")
    print("=" * 70)
    all_results = {}
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp.get("desc", name)
        print(f"\n[{idx+1}/{len(experiments)}] {desc}")
        print(f"    Generating {n_images} images...")
        result = run_experiment(wrapper, exp, prompts, seeds, resolution, steps, output_base)
        all_results[name] = result
        print(f"    Done. Mean: {result['mean_time']:.2f}s/img, Total: {result['total_time']:.1f}s")

    del wrapper
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Phase B: Metrics ----
    print()
    print("=" * 70)
    print("PHASE B: Quality Metrics")
    print("=" * 70)

    pick_eval = None
    ir_eval = None

    if not args.no_pickscore:
        print("Loading PickScore model...")
        try:
            pick_eval = PickScoreEvaluator(device="cuda")
            print("  PickScore loaded.")
        except Exception as e:
            print(f"  PickScore failed: {e}, skipping.")

    if not args.no_imagereward:
        print("Loading ImageReward model...")
        try:
            ir_eval = ImageRewardEvaluator(device="cuda")
            print("  ImageReward loaded.")
        except Exception as e:
            print(f"  ImageReward failed: {e}, skipping.")

    all_metrics = {}
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp.get("desc", name)
        print(f"\n[{idx+1}/{len(experiments)}] {desc}")
        all_metrics[name] = evaluate_experiment(
            all_results[name],
            pick_evaluator=pick_eval,
            ir_evaluator=ir_eval,
        )

    del pick_eval, ir_eval
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Phase C: Summary ----
    print()
    print("=" * 100)
    ref_time = all_results["mask_eval-1-baseline"]["mean_time"]
    ref_clip = all_metrics["mask_eval-1-baseline"]["mean_clip"]

    has_pick = "mean_pick" in all_metrics["mask_eval-1-baseline"]
    has_ir = "mean_ir" in all_metrics["mask_eval-1-baseline"]

    header = f"{'Config':<35} {'Time':>6} {'Speed':>6} {'CLIP':>8} {'dCLIP':>7}"
    if has_pick:
        header += f" {'Pick':>8} {'dPick':>7}"
    if has_ir:
        header += f" {'IR':>8} {'dIR':>7}"
    header += f" {'Edge':>8}"
    print(header)
    print("=" * 100)

    ref_pick = all_metrics["mask_eval-1-baseline"].get("mean_pick", 0)
    ref_ir = all_metrics["mask_eval-1-baseline"].get("mean_ir", 0)

    for exp in experiments:
        name = exp["name"]
        desc = exp.get("desc", name)
        t = all_results[name]["mean_time"]
        speedup = ref_time / t
        mc = all_metrics[name]["mean_clip"]
        dc = mc - ref_clip
        edge = all_metrics[name]["mean_edge"]

        line = f"  {desc:<33} {t:5.2f}s {speedup:5.2f}x {mc:7.4f} {dc:+6.4f}"
        if has_pick:
            mp = all_metrics[name]["mean_pick"]
            dp = mp - ref_pick
            line += f" {mp:7.4f} {dp:+6.4f}"
        if has_ir:
            mi = all_metrics[name]["mean_ir"]
            di = mi - ref_ir
            line += f" {mi:7.4f} {di:+6.4f}"
        line += f" {edge:7.4f}"
        print(line)

    print("=" * 100)
    print(f"  dCLIP/dPick/dIR = delta vs Baseline")
    print()

    # ---- Save summary JSON ----
    summary = {}
    for exp in experiments:
        name = exp["name"]
        entry = {
            "desc": exp.get("desc"),
            "mean_time": all_results[name]["mean_time"],
            "speedup_vs_baseline": ref_time / all_results[name]["mean_time"],
            "mean_clip": float(all_metrics[name]["mean_clip"]),
            "std_clip": float(all_metrics[name]["std_clip"]),
            "mean_edge": float(all_metrics[name]["mean_edge"]),
        }
        if has_pick:
            entry["mean_pick"] = float(all_metrics[name]["mean_pick"])
            entry["std_pick"] = float(all_metrics[name]["std_pick"])
        if has_ir:
            entry["mean_ir"] = float(all_metrics[name]["mean_ir"])
            entry["std_ir"] = float(all_metrics[name]["std_ir"])
        summary[name] = entry

    summary_path = os.path.join(output_base, "mask_compare_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    print("=== Done! ===")


if __name__ == "__main__":
    main()
