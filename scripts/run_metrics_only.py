#!/usr/bin/env python3
"""
Metrics-only evaluation on pre-generated images.
Reuses images from run_hybrid_full_eval.py (outputs/full_eval-*/images/).
"""

import os
import sys
import gc
import json
import csv
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
        scores = []
        for img, prompt in zip(images, prompts):
            tmp_path = os.path.join(img_dir, "_tmp_hps.png")
            img.save(tmp_path)
            result = self.hpsv2.score(tmp_path, prompt, hps_version="v2.1")
            scores.append(float(result[0]) if isinstance(result, (list, np.ndarray)) else float(result))
            os.remove(tmp_path)
        return scores


class AestheticScoreEvaluator:
    def __init__(self, device="cuda"):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device,
        )
        self.model.eval()
        self.mlp = self._load_aesthetic_mlp(device)

    def _load_aesthetic_mlp(self, device):
        import torch.nn as nn
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

        weights_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "aesthetic_mlp_l14.pth")

        if not os.path.exists(weights_path):
            print(f"    Downloading aesthetic MLP weights...")
            torch.hub.download_url_to_file(weights_url, weights_path)

        state = torch.load(weights_path, map_location=device, weights_only=True)
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
    def __init__(self, device="cuda"):
        import lpips
        self.loss_fn = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, images, baseline_images):
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


def main():
    from src.utils.io_utils import load_prompts

    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
    )

    prompts = load_prompts("prompts/prompts_dev.txt")[:20]
    seeds = [0, 1, 2]

    # Previously measured times from the generation run
    measured_times = {
        "full_eval-1-baseline": 12.54,
        "full_eval-4-hybrid20-cfgfix": 8.43,
    }

    experiments = [
        {"name": "full_eval-1-baseline", "desc": "Baseline (dense, CFG=4.0)"},
        {"name": "full_eval-4-hybrid20-cfgfix", "desc": "Hybrid20 (cfg_scale=4.0 pre-mask)"},
    ]

    baseline_name = "full_eval-1-baseline"

    # Load all images
    print("Loading images...")
    all_images = {}
    all_prompts_list = {}
    for exp in experiments:
        name = exp["name"]
        img_dir = os.path.join(output_base, name, "images")
        images = []
        prompts_list = []
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                fname = f"p{pi:04d}_s{seed:04d}.png"
                img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
                images.append(img)
                prompts_list.append(prompt)
        all_images[name] = images
        all_prompts_list[name] = prompts_list
        print(f"  {name}: {len(images)} images")

    baseline_images = all_images[baseline_name]

    # Initialize evaluators
    evaluators = {}

    for label, cls, args in [
        ("PickScore", PickScoreEvaluator, {}),
        ("ImageReward", ImageRewardEvaluator, {}),
        ("HPSv2", HPSv2Evaluator, {}),
        ("Aesthetic Score", AestheticScoreEvaluator, {}),
        ("LPIPS", LPIPSEvaluator, {}),
    ]:
        print(f"Loading {label}...", end="", flush=True)
        try:
            evaluators[label] = cls(device="cuda")
            print(" OK")
        except Exception as e:
            print(f" FAILED: {e}")

    # Compute metrics
    all_metrics = {}
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp["desc"]
        print(f"\n[{idx+1}/{len(experiments)}] {desc}")

        images = all_images[name]
        prompts_list = all_prompts_list[name]
        img_dir = os.path.join(output_base, name, "images")
        metrics = {}

        print("    CLIPScore...", end="", flush=True)
        clip_scores = compute_clip_score(images, prompts_list, device="cuda")
        metrics["clip"] = clip_scores
        print(f" {np.mean(clip_scores):.4f}")

        print("    EdgeDensity...", end="", flush=True)
        edge_scores = [compute_edge_density(img) for img in images]
        metrics["edge"] = edge_scores
        print(f" {np.mean(edge_scores):.4f}")

        if "PickScore" in evaluators:
            print("    PickScore...", end="", flush=True)
            pick_scores = evaluators["PickScore"].score(images, prompts_list)
            metrics["pick"] = pick_scores
            print(f" {np.mean(pick_scores):.4f}")

        if "ImageReward" in evaluators:
            print("    ImageReward...", end="", flush=True)
            ir_scores = evaluators["ImageReward"].score(images, prompts_list)
            metrics["ir"] = ir_scores
            print(f" {np.mean(ir_scores):.4f}")

        if "HPSv2" in evaluators:
            print("    HPSv2...", end="", flush=True)
            hps_scores = evaluators["HPSv2"].score(images, prompts_list, img_dir)
            metrics["hps"] = hps_scores
            print(f" {np.mean(hps_scores):.4f}")

        if "Aesthetic Score" in evaluators:
            print("    AestheticScore...", end="", flush=True)
            aes_scores = evaluators["Aesthetic Score"].score(images)
            metrics["aesthetic"] = aes_scores
            print(f" {np.mean(aes_scores):.4f}")

        if "LPIPS" in evaluators and name != baseline_name:
            print("    LPIPS vs baseline...", end="", flush=True)
            lpips_scores = evaluators["LPIPS"].score(images, baseline_images)
            metrics["lpips"] = lpips_scores
            print(f" {np.mean(lpips_scores):.4f}")

        all_metrics[name] = metrics

    del evaluators
    gc.collect()
    torch.cuda.empty_cache()

    # FID
    print("\n    Computing FID...", end="", flush=True)
    fid_scores = {}
    try:
        from cleanfid import fid
        baseline_dir = os.path.join(output_base, baseline_name, "images")
        for exp in experiments:
            name = exp["name"]
            if name == baseline_name:
                continue
            exp_dir = os.path.join(output_base, name, "images")
            fid_val = fid.compute_fid(baseline_dir, exp_dir, mode="clean", num_workers=0)
            fid_scores[name] = fid_val
            print(f" {name}: {fid_val:.2f}", end="", flush=True)
        print(" done")
    except Exception as e:
        print(f" FAILED: {e}")

    # Summary table
    print()
    print("=" * 130)
    ref_time = measured_times[baseline_name]

    metric_names = ["CLIP", "Pick", "IR", "HPS", "Aesth", "LPIPS", "Edge", "FID"]
    header = f"{'Config':<35} {'Time':>6} {'Speed':>6}"
    for m in metric_names:
        header += f" {m:>8}"
    print(header)
    print("=" * 130)

    for exp in experiments:
        name = exp["name"]
        desc = exp["desc"]
        t = measured_times[name]
        speedup = ref_time / t

        line = f"  {desc:<33} {t:5.2f}s {speedup:5.2f}x"

        for key in ["clip", "pick", "ir", "hps", "aesthetic"]:
            if key in all_metrics[name]:
                val = float(np.mean(all_metrics[name][key]))
                line += f" {val:8.4f}"
            else:
                line += f" {'N/A':>8}"

        if "lpips" in all_metrics.get(name, {}):
            line += f" {float(np.mean(all_metrics[name]['lpips'])):8.4f}"
        elif name == baseline_name:
            line += f" {'0.0000':>8}"
        else:
            line += f" {'N/A':>8}"

        if "edge" in all_metrics[name]:
            line += f" {float(np.mean(all_metrics[name]['edge'])):8.4f}"
        else:
            line += f" {'N/A':>8}"

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

    # Save JSON
    summary = {}
    for exp in experiments:
        name = exp["name"]
        entry = {
            "desc": exp["desc"],
            "mean_time": measured_times[name],
            "speedup": ref_time / measured_times[name],
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
    print("=== Done! ===")


if __name__ == "__main__":
    main()
