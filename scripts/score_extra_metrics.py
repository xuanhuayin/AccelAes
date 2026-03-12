#!/usr/bin/env python3
"""
补充打分：在已生成的 240 张图上计算剩余 5 个指标。
PickScore | HPSv2 | Aesthetic Score | LPIPS (vs baseline) | FID (vs baseline)

已有结果从 summary.json 读取，更新后写回。

Usage:
  python scripts/score_extra_metrics.py
  python scripts/score_extra_metrics.py --eval_dir outputs/eval
"""

import argparse
import csv
import gc
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))


def load_images_and_prompts(img_dir: str, prompts: list, seeds: list):
    imgs, plist = [], []
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            p = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
            imgs.append(Image.open(p).convert("RGB"))
            plist.append(prompt)
    return imgs, plist


def load_prompts(path, n):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()][:n]


# ---- PickScore ----
class PickScoreEvaluator:
    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, AutoModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained(
            "yuvalkirstain/PickScore_v1").eval().to(device)

    @torch.no_grad()
    def score(self, images, prompts):
        scores = []
        for img, prompt in zip(images, prompts):
            inputs = self.processor(
                text=[prompt], images=[img],
                return_tensors="pt", padding=True,
                truncation=True, max_length=77,
            ).to(self.device)
            out = self.model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
            scores.append((ie @ te.T).item())
        return scores


# ---- HPSv2 ----
class HPSv2Evaluator:
    def __init__(self, device="cuda"):
        import hpsv2
        self.hpsv2 = hpsv2

    @torch.no_grad()
    def score(self, images, prompts):
        scores = []
        with tempfile.TemporaryDirectory() as td:
            for i, (img, prompt) in enumerate(zip(images, prompts)):
                tmp = os.path.join(td, f"{i}.png")
                img.save(tmp)
                result = self.hpsv2.score(tmp, prompt, hps_version="v2.1")
                scores.append(float(result[0]) if isinstance(
                    result, (list, np.ndarray)) else float(result))
        return scores


# ---- Aesthetic Score ----
class AestheticScoreEvaluator:
    def __init__(self, device="cuda"):
        import open_clip
        import torch.nn as nn
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device)
        self.model.eval()

        mlp = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ).to(device)
        weights_url = ("https://github.com/christophschuhmann/"
                       "improved-aesthetic-predictor/raw/main/"
                       "sac+logos+ava1-l14-linearMSE.pth")
        cache = os.path.join(os.path.dirname(__file__), "..", ".cache")
        os.makedirs(cache, exist_ok=True)
        wp = os.path.join(cache, "aesthetic_mlp_l14.pth")
        if not os.path.exists(wp):
            print("  Downloading aesthetic MLP weights...")
            torch.hub.download_url_to_file(weights_url, wp)
        state = torch.load(wp, map_location=device, weights_only=True)
        remapped = {k.replace("layers.", ""): v for k, v in state.items()}
        mlp.load_state_dict(remapped)
        mlp.eval()
        self.mlp = mlp

    @torch.no_grad()
    def score(self, images):
        scores = []
        for img in images:
            t = self.preprocess(img).unsqueeze(0).to(self.device)
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            scores.append(self.mlp(f.float()).item())
        return scores


# ---- LPIPS ----
class LPIPSEvaluator:
    def __init__(self, device="cuda"):
        import lpips
        self.fn = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, images, refs):
        import torchvision.transforms as T
        tf = T.Compose([
            T.Resize((256, 256)), T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        scores = []
        for img, ref in zip(images, refs):
            it = tf(img).unsqueeze(0).to(self.device)
            rt = tf(ref).unsqueeze(0).to(self.device)
            scores.append(self.fn(it, rt).item())
        return scores


# ---- FID ----
def compute_fid(dir_a, dir_b):
    from cleanfid import fid
    return fid.compute_fid(dir_a, dir_b, mode="clean", num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",     default="outputs/eval")
    parser.add_argument("--prompts_file", default="prompts/prompts_dev.txt")
    parser.add_argument("--num_prompts",  type=int, default=30)
    parser.add_argument("--num_seeds",    type=int, default=2)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file, args.num_prompts)
    seeds   = list(range(args.num_seeds))
    ed      = args.eval_dir

    # Load existing summary
    summary_path = os.path.join(ed, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    # Configs: (key, model_dir, is_ref_for_lpips, ref_key)
    CONFIGS = [
        ("sd3/baseline",             "sd3/baseline",             True,  None),
        ("sd3/semantic_fskip2",      "sd3/semantic_fskip2",      False, "sd3/baseline"),
        ("flux/baseline",            "flux/baseline",            True,  None),
        ("flux/fskip2_w5",           "flux/fskip2_w5",           False, "flux/baseline"),
        ("flux/sasd_sparse_fskip2",  "flux/sasd_sparse_fskip2",  False, "flux/baseline"),
    ]
    # Only score configs whose images actually exist
    CONFIGS = [(k, d, r, rk) for k, d, r, rk in CONFIGS
               if os.path.isdir(os.path.join(ed, d))]

    # Configs that need extra metrics (skip those that already have all 5)
    EXTRA_METRICS = ["pick", "hps", "aesthetic", "lpips", "fid"]
    NEEDS_SCORING = {k for k, _, _, _ in CONFIGS
                     if not all(mk in summary.get(k, {}) for mk in EXTRA_METRICS)}

    # Load all images
    all_imgs  = {}
    all_plist = {}
    for key, rel_dir, _, _ in CONFIGS:
        img_dir = os.path.join(ed, rel_dir)
        imgs, plist = load_images_and_prompts(img_dir, prompts, seeds)
        all_imgs[key]  = imgs
        all_plist[key] = plist
        print(f"Loaded {len(imgs)} images: {key}")

    device = "cuda"

    # ---- PickScore ----
    print("\n[1/5] PickScore...")
    if any(k in NEEDS_SCORING for k, *_ in CONFIGS):
        ev = PickScoreEvaluator(device)
        for key, _, _, _ in CONFIGS:
            if key in NEEDS_SCORING:
                s = ev.score(all_imgs[key], all_plist[key])
                summary[key]["pick"] = float(np.mean(s))
                print(f"  {key}: {summary[key]['pick']:.4f}")
            else:
                print(f"  {key}: {summary[key]['pick']:.4f} (cached)")
        del ev; gc.collect(); torch.cuda.empty_cache()
    else:
        print("  All cached.")

    # ---- HPSv2 ----
    print("\n[2/5] HPSv2...")
    if any(k in NEEDS_SCORING for k, *_ in CONFIGS):
        ev = HPSv2Evaluator(device)
        for key, _, _, _ in CONFIGS:
            if key in NEEDS_SCORING:
                s = ev.score(all_imgs[key], all_plist[key])
                summary[key]["hps"] = float(np.mean(s))
                print(f"  {key}: {summary[key]['hps']:.4f}")
            else:
                print(f"  {key}: {summary[key]['hps']:.4f} (cached)")
        del ev; gc.collect(); torch.cuda.empty_cache()
    else:
        print("  All cached.")

    # ---- Aesthetic Score ----
    print("\n[3/5] Aesthetic Score...")
    if any(k in NEEDS_SCORING for k, *_ in CONFIGS):
        ev = AestheticScoreEvaluator(device)
        for key, _, _, _ in CONFIGS:
            if key in NEEDS_SCORING:
                s = ev.score(all_imgs[key])
                summary[key]["aesthetic"] = float(np.mean(s))
                print(f"  {key}: {summary[key]['aesthetic']:.4f}")
            else:
                print(f"  {key}: {summary[key]['aesthetic']:.4f} (cached)")
        del ev; gc.collect(); torch.cuda.empty_cache()
    else:
        print("  All cached.")

    # ---- LPIPS ----
    print("\n[4/5] LPIPS (vs baseline)...")
    if any(k in NEEDS_SCORING for k, *_ in CONFIGS):
        ev = LPIPSEvaluator(device)
        for key, _, is_ref, ref_key in CONFIGS:
            if key in NEEDS_SCORING:
                if is_ref:
                    summary[key]["lpips"] = 0.0
                    print(f"  {key}: 0.0000 (reference)")
                else:
                    s = ev.score(all_imgs[key], all_imgs[ref_key])
                    summary[key]["lpips"] = float(np.mean(s))
                    print(f"  {key}: {summary[key]['lpips']:.4f}")
            else:
                print(f"  {key}: {summary[key]['lpips']:.4f} (cached)")
        del ev; gc.collect(); torch.cuda.empty_cache()
    else:
        print("  All cached.")

    # ---- FID ----
    print("\n[5/5] FID (vs baseline)...")
    for key, rel_dir, is_ref, ref_key in CONFIGS:
        if key in NEEDS_SCORING:
            if is_ref:
                summary[key]["fid"] = 0.0
                print(f"  {key}: 0.00 (reference)")
            else:
                ref_dir = os.path.join(ed, ref_key)
                acc_dir = os.path.join(ed, rel_dir)
                val = compute_fid(ref_dir, acc_dir)
                summary[key]["fid"] = float(val)
                print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {summary[key]['fid']:.2f} (cached)")

    # ---- Save updated summary ----
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated → {summary_path}")

    # ---- Print final table ----
    M = ["mean_time", "clip", "pick", "ir", "hps", "aesthetic", "lpips", "edge", "fid"]
    LABELS = ["time", "CLIP", "Pick", "IR", "HPS", "Aesth", "LPIPS", "Edge", "FID"]
    ref_times = {
        "sd3":  summary["sd3/baseline"].get("mean_time"),
        "flux": summary["flux/baseline"].get("mean_time"),
    }

    print("\n" + "=" * 105)
    hdr = f"{'config':<28} {'speedup':>7}"
    for lb in LABELS[1:]:
        hdr += f" {lb:>8}"
    print(hdr)
    print("=" * 105)

    rows = []
    for key, _, _, _ in CONFIGS:
        r = summary[key]
        model = key.split("/")[0]
        ref_t = ref_times[model]
        t = r.get("mean_time")
        spd = f"{ref_t/t:.2f}x" if (t and ref_t) else "N/A"

        line = f"{key:<28} {spd:>7}"
        for mk in ["clip", "pick", "ir", "hps", "aesthetic", "lpips", "edge", "fid"]:
            v = r.get(mk, None)
            line += f" {v:8.4f}" if v is not None else f" {'N/A':>8}"
        print(line)
        rows.append({"config": key, "speedup": spd, **{mk: r.get(mk, "") for mk in
                     ["clip", "pick", "ir", "hps", "aesthetic", "lpips", "edge", "fid"]}})

    print("=" * 105)
    print("LPIPS/FID: lower = closer to baseline. Aesth: higher = more aesthetic.")

    # Save CSV
    csv_path = os.path.join(ed, "summary_full.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
