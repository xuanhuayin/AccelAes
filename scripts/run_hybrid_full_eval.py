#!/usr/bin/env python3
"""
Full evaluation of hybrid approach with comprehensive metrics.

Configs:
  1. Baseline (dense, CFG=4.0)
  2. Current best (sparse from step 6, skip=2)
  3. Hybrid: sparse from step 10 + skip=2

Metrics:
  - CLIPScore (text-image alignment)
  - PickScore v1 (human preference)
  - ImageReward v1.0 (aesthetic + alignment)
  - HPSv2 (Human Preference Score v2)
  - Aesthetic Score (LAION aesthetic predictor via CLIP+MLP)
  - LPIPS (perceptual similarity vs baseline)
  - Edge Density (structural detail)
  - FID (distribution quality vs baseline)
  - Speed (wall-clock)

20 prompts x 3 seeds = 60 images per config, 3 configs = 180 images.
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


# ---- Metric evaluators ----

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


class HPSv2Evaluator:
    def __init__(self, device="cuda"):
        import hpsv2
        self.hpsv2 = hpsv2
        self.device = device

    @torch.no_grad()
    def score(self, images, prompts, img_dir):
        """HPSv2 works on file paths."""
        scores = []
        for img, prompt in zip(images, prompts):
            # Save temp file for HPSv2
            tmp_path = os.path.join(img_dir, "_tmp_hps.png")
            img.save(tmp_path)
            result = self.hpsv2.score(tmp_path, prompt, hps_version="v2.1")
            scores.append(float(result[0]) if isinstance(result, (list, np.ndarray)) else float(result))
            os.remove(tmp_path)
        return scores


class AestheticScoreEvaluator:
    """LAION Aesthetic Predictor v2 (CLIP ViT-L/14 + MLP head)."""
    def __init__(self, device="cuda"):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device,
        )
        self.model.eval()
        # Load aesthetic MLP head
        self.mlp = self._load_aesthetic_mlp(device)

    def _load_aesthetic_mlp(self, device):
        """Load the LAION aesthetic predictor MLP weights."""
        import torch.nn as nn
        # Simple MLP: 768 -> 1024 -> 128 -> 64 -> 16 -> 1
        mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ).to(device)

        # Download weights
        weights_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "aesthetic_mlp_l14.pth")

        if not os.path.exists(weights_path):
            print(f"    Downloading aesthetic MLP weights...")
            torch.hub.download_url_to_file(weights_url, weights_path)

        state = torch.load(weights_path, map_location=device, weights_only=True)
        # Remap keys: "layers.X.weight" -> "X.weight"
        remapped = {}
        for k, v in state.items():
            new_k = k.replace("layers.", "")
            remapped[new_k] = v
        mlp.load_state_dict(remapped)
        mlp.eval()
        return mlp

    @torch.no_grad()
    def score(self, images):
        scores = []
        for img in images:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            s = self.mlp(img_feat.float()).item()
            scores.append(s)
        return scores


class LPIPSEvaluator:
    """Perceptual similarity vs baseline images."""
    def __init__(self, device="cuda"):
        import lpips
        self.loss_fn = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, images, baseline_images):
        """Lower LPIPS = more similar to baseline (better)."""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        scores = []
        for img, ref in zip(images, baseline_images):
            img_t = transform(img).unsqueeze(0).to(self.device)
            ref_t = transform(ref).unsqueeze(0).to(self.device)
            d = self.loss_fn(img_t, ref_t).item()
            scores.append(d)
        return scores


# ---- Generation ----

def run_experiment(wrapper, exp_cfg, prompts, seeds, resolution, steps, output_base):
    name = exp_cfg["name"]
    method = exp_cfg["method"]

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
                    **exp_cfg["kwargs"],
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="0,1,2")
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

    common = dict(
        skip_ratio=0.50, s_fg=7.0, s_bg=2.0,
        mask_step=5, mask_type="cfg_magnitude",
        skip_step_interval=2,
    )

    experiments = [
        {
            "name": "full_eval-1-baseline",
            "desc": "Baseline (dense, CFG=4.0)",
            "method": "baseline",
        },
        {
            "name": "full_eval-2-current",
            "desc": "Current (sparse from step 6)",
            "method": "accel_dual",
            "kwargs": dict(**common, sparse_blocks=True, sparse_ffn=True,
                           sparse_start_step=-1),
        },
        {
            "name": "full_eval-3-hybrid",
            "desc": "Hybrid (sparse from step 10)",
            "method": "accel_dual",
            "kwargs": dict(**common, sparse_blocks=True, sparse_ffn=True,
                           sparse_start_step=10),
        },
    ]

    n_images = len(prompts) * len(seeds)
    print(f"=== Full Hybrid Evaluation ===")
    print(f"{len(prompts)} prompts x {len(seeds)} seeds = {n_images} images/config")
    print(f"{len(experiments)} configs -> {n_images * len(experiments)} total images")
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

    # Load all images
    all_images = {}
    all_prompts_list = {}
    for exp in experiments:
        name = exp["name"]
        img_dir = all_results[name]["img_dir"]
        results = all_results[name]["results"]
        images = [Image.open(os.path.join(img_dir, r["filename"])).convert("RGB") for r in results]
        prompts_list = [r["prompt"] for r in results]
        all_images[name] = images
        all_prompts_list[name] = prompts_list

    baseline_name = "full_eval-1-baseline"
    baseline_images = all_images[baseline_name]

    # Initialize evaluators
    evaluators = {}

    print("Loading PickScore...", end="", flush=True)
    try:
        evaluators["pick"] = PickScoreEvaluator(device="cuda")
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")

    print("Loading ImageReward...", end="", flush=True)
    try:
        evaluators["ir"] = ImageRewardEvaluator(device="cuda")
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")

    print("Loading HPSv2...", end="", flush=True)
    try:
        evaluators["hps"] = HPSv2Evaluator(device="cuda")
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")

    print("Loading Aesthetic Score...", end="", flush=True)
    try:
        evaluators["aesthetic"] = AestheticScoreEvaluator(device="cuda")
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")

    print("Loading LPIPS...", end="", flush=True)
    try:
        evaluators["lpips"] = LPIPSEvaluator(device="cuda")
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")

    # Compute metrics per experiment
    all_metrics = {}
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp.get("desc", name)
        print(f"\n[{idx+1}/{len(experiments)}] {desc}")

        images = all_images[name]
        prompts_list = all_prompts_list[name]
        img_dir = all_results[name]["img_dir"]
        metrics = {}

        print("    CLIPScore...", end="", flush=True)
        clip_scores = compute_clip_score(images, prompts_list, device="cuda")
        metrics["clip"] = clip_scores
        print(f" {np.mean(clip_scores):.4f}")

        print("    EdgeDensity...", end="", flush=True)
        edge_scores = [compute_edge_density(img) for img in images]
        metrics["edge"] = edge_scores
        print(f" {np.mean(edge_scores):.4f}")

        if "pick" in evaluators:
            print("    PickScore...", end="", flush=True)
            pick_scores = evaluators["pick"].score(images, prompts_list)
            metrics["pick"] = pick_scores
            print(f" {np.mean(pick_scores):.4f}")

        if "ir" in evaluators:
            print("    ImageReward...", end="", flush=True)
            ir_scores = evaluators["ir"].score(images, prompts_list)
            metrics["ir"] = ir_scores
            print(f" {np.mean(ir_scores):.4f}")

        if "hps" in evaluators:
            print("    HPSv2...", end="", flush=True)
            hps_scores = evaluators["hps"].score(images, prompts_list, img_dir)
            metrics["hps"] = hps_scores
            print(f" {np.mean(hps_scores):.4f}")

        if "aesthetic" in evaluators:
            print("    AestheticScore...", end="", flush=True)
            aes_scores = evaluators["aesthetic"].score(images)
            metrics["aesthetic"] = aes_scores
            print(f" {np.mean(aes_scores):.4f}")

        if "lpips" in evaluators and name != baseline_name:
            print("    LPIPS vs baseline...", end="", flush=True)
            lpips_scores = evaluators["lpips"].score(images, baseline_images)
            metrics["lpips"] = lpips_scores
            print(f" {np.mean(lpips_scores):.4f}")

        all_metrics[name] = metrics

    # Clean up evaluators
    del evaluators
    gc.collect()
    torch.cuda.empty_cache()

    # ---- FID ----
    print("\n    Computing FID...", end="", flush=True)
    fid_scores = {}
    try:
        from cleanfid import fid
        baseline_dir = all_results[baseline_name]["img_dir"]
        for exp in experiments:
            name = exp["name"]
            if name == baseline_name:
                continue
            exp_dir = all_results[name]["img_dir"]
            fid_val = fid.compute_fid(baseline_dir, exp_dir, mode="clean", num_workers=0)
            fid_scores[name] = fid_val
            print(f" {name}: {fid_val:.2f}", end="", flush=True)
        print(" done")
    except Exception as e:
        print(f" FAILED: {e}")

    # ---- Phase C: Summary ----
    print()
    print("=" * 130)

    ref_time = all_results[baseline_name]["mean_time"]
    ref = {k: float(np.mean(v)) for k, v in all_metrics[baseline_name].items()}

    # Header
    metric_names = ["CLIP", "Pick", "IR", "HPS", "Aesth", "LPIPS", "Edge", "FID"]
    header = f"{'Config':<35} {'Time':>6} {'Speed':>6}"
    for m in metric_names:
        header += f" {m:>8}"
    print(header)
    print("=" * 130)

    for exp in experiments:
        name = exp["name"]
        desc = exp.get("desc", name)
        t = all_results[name]["mean_time"]
        speedup = ref_time / t

        line = f"  {desc:<33} {t:5.2f}s {speedup:5.2f}x"

        for key, label in [("clip","CLIP"), ("pick","Pick"), ("ir","IR"),
                           ("hps","HPS"), ("aesthetic","Aesth")]:
            if key in all_metrics[name]:
                val = float(np.mean(all_metrics[name][key]))
                line += f" {val:8.4f}"
            else:
                line += f" {'N/A':>8}"

        # LPIPS
        if "lpips" in all_metrics.get(name, {}):
            line += f" {float(np.mean(all_metrics[name]['lpips'])):8.4f}"
        elif name == baseline_name:
            line += f" {'0.0000':>8}"
        else:
            line += f" {'N/A':>8}"

        # Edge
        if "edge" in all_metrics[name]:
            line += f" {float(np.mean(all_metrics[name]['edge'])):8.4f}"
        else:
            line += f" {'N/A':>8}"

        # FID
        if name in fid_scores:
            line += f" {fid_scores[name]:8.2f}"
        elif name == baseline_name:
            line += f" {'0.00':>8}"
        else:
            line += f" {'N/A':>8}"

        print(line)

    print("=" * 130)
    print("  LPIPS: lower=closer to baseline. FID: lower=better distribution match.")
    print()

    # ---- Save JSON summary ----
    summary = {}
    for exp in experiments:
        name = exp["name"]
        entry = {
            "desc": exp.get("desc"),
            "mean_time": all_results[name]["mean_time"],
            "speedup": ref_time / all_results[name]["mean_time"],
        }
        for key in ["clip", "pick", "ir", "hps", "aesthetic", "lpips", "edge"]:
            if key in all_metrics[name]:
                entry[f"mean_{key}"] = float(np.mean(all_metrics[name][key]))
                entry[f"std_{key}"] = float(np.std(all_metrics[name][key]))
        if name in fid_scores:
            entry["fid"] = fid_scores[name]
        summary[name] = entry

    summary_path = os.path.join(output_base, "hybrid_full_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # ---- Save per-image CSV ----
    for exp in experiments:
        name = exp["name"]
        results = all_results[name]["results"]
        metrics = all_metrics[name]
        for i, r in enumerate(results):
            for key in metrics:
                r[key] = metrics[key][i]
        csv_path = os.path.join(all_results[name]["exp_dir"], "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print("=== Done! ===")


if __name__ == "__main__":
    main()
