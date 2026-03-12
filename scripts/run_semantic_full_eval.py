#!/usr/bin/env python3
"""
Full evaluation: baseline vs SASD best config.

Best config: semantic mask + SLIC region method + full_skip_interval=2
20 prompts × 3 seeds = 60 images per config.

Metrics: CLIP, Edge, PickScore, ImageReward, HPSv2, Aesthetic, LPIPS, FID
"""

import os
import sys
import time
import gc
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.dit_wrapper import LuminaDiTWrapper
from src.eval.metrics import compute_clip_score, compute_edge_density

OUTPUT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "semantic_full_eval")

EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "Baseline (dense, CFG=4.0)",
        "method": "baseline",
        "kwargs": {},
    },
    {
        "name": "sasd_semantic_slic_fskip2",
        "desc": "SASD: semantic+SLIC+fskip2 (s_fg=7,s_bg=1)",
        "method": "accel",
        "kwargs": dict(
            mask_type="semantic",
            region_method="slic",
            skip_ratio=0.50,
            s_fg=7.0, s_bg=1.0,
            mask_step=5,
            full_skip_interval=2,
            sparse_ffn=True,
            sparse_blocks=True,
        ),
    },
    {
        "name": "sasd_cfg_mag_fskip2",
        "desc": "SASD: cfg_magnitude+fskip2 (s_fg=7,s_bg=1)",
        "method": "accel",
        "kwargs": dict(
            mask_type="cfg_magnitude",
            skip_ratio=0.50,
            s_fg=7.0, s_bg=1.0,
            mask_step=5,
            full_skip_interval=2,
            sparse_ffn=True,
            sparse_blocks=True,
        ),
    },
]


# ── Metric evaluators ────────────────────────────────────────────────────────

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
                text=[prompt], images=[img], return_tensors="pt", padding=True,
            ).to(self.device)
            out = self.model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
            scores.append((ie @ te.T).item())
        return scores


class ImageRewardEvaluator:
    def __init__(self, device="cuda"):
        import ImageReward as ir_module
        self.model = ir_module.load("ImageReward-v1.0", device=device)

    @torch.no_grad()
    def score(self, images, prompts):
        return [float(self.model.score(p, img)) for img, p in zip(images, prompts)]


class HPSv2Evaluator:
    def __init__(self, device="cuda"):
        import hpsv2
        self.hpsv2 = hpsv2

    @torch.no_grad()
    def score(self, images, prompts, img_dir):
        scores = []
        for img, prompt in zip(images, prompts):
            tmp = os.path.join(img_dir, "_tmp_hps.png")
            img.save(tmp)
            r = self.hpsv2.score(tmp, prompt, hps_version="v2.1")
            scores.append(float(r[0]) if isinstance(r, (list, np.ndarray)) else float(r))
            os.remove(tmp)
        return scores


class AestheticScoreEvaluator:
    def __init__(self, device="cuda"):
        import open_clip, torch.nn as nn
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device)
        self.model.eval()

        cache = os.path.join(os.path.dirname(__file__), "..", ".cache")
        os.makedirs(cache, exist_ok=True)
        weights_path = os.path.join(cache, "aesthetic_mlp_l14.pth")
        if not os.path.exists(weights_path):
            url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
            torch.hub.download_url_to_file(url, weights_path)

        self.mlp = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ).to(device)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        self.mlp.load_state_dict({k.replace("layers.", ""): v for k, v in state.items()})
        self.mlp.eval()

    @torch.no_grad()
    def score(self, images):
        scores = []
        for img in images:
            t = self.preprocess(img).unsqueeze(0).to(self.device)
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            scores.append(self.mlp(f.float()).item())
        return scores


class LPIPSEvaluator:
    def __init__(self, device="cuda"):
        import lpips
        self.fn = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, images, refs):
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                        T.Normalize([0.5]*3, [0.5]*3)])
        return [self.fn(tf(a).unsqueeze(0).to(self.device),
                        tf(b).unsqueeze(0).to(self.device)).item()
                for a, b in zip(images, refs)]


# ── Generation ────────────────────────────────────────────────────────────────

def generate_images(wrapper, prompts, seeds, exp, output_base):
    name = exp["name"]
    img_dir = os.path.join(output_base, name, "images")
    os.makedirs(img_dir, exist_ok=True)

    total_images = len(prompts) * len(seeds)
    print(f"\n=== [{name}] {exp['desc']} ({total_images} images) ===")

    times, count, skipped = [], 0, 0
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            out_path = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
            if os.path.exists(out_path):
                skipped += 1
                count += 1
                if count % 10 == 0 or count == total_images:
                    print(f"  [{count}/{total_images}] skipped (already exists)")
                continue

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if exp["method"] == "baseline":
                img = wrapper.generate(prompt=prompt, seed=seed, cfg_scale=4.0)
            else:
                img = wrapper.generate_accelerated_dual(
                    prompt=prompt, seed=seed, **exp["kwargs"])

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            count += 1

            img.save(out_path)
            if count % 10 == 0 or count == total_images:
                print(f"  [{count}/{total_images}] avg {sum(times)/len(times):.2f}s/img")

            del img
            gc.collect()
            torch.cuda.empty_cache()

    if skipped == total_images:
        print(f"  All {total_images} images already exist, skipping generation.")
        # load times from existing images not available; return 0 as placeholder
        return img_dir, 0.0
    mean_time = sum(times) / len(times) if times else 0.0
    print(f"  Done: {mean_time:.2f}s/img ({skipped} skipped)")
    return img_dir, mean_time


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="prompts/prompts_dev.txt")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip_gen", action="store_true", help="skip generation, only run metrics")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = [l.strip() for l in f if l.strip()][:args.num_prompts]
    seeds = args.seeds
    print(f"Prompts: {len(prompts)}, Seeds: {seeds}, Total/config: {len(prompts)*len(seeds)}")

    # ── Phase 1: Generate ──────────────────────────────────────────────────
    measured_times = {}
    if not args.skip_gen:
        wrapper = LuminaDiTWrapper(dtype="bf16")
        for exp in EXPERIMENTS:
            _, t = generate_images(wrapper, prompts, seeds, exp, OUTPUT_BASE)
            measured_times[exp["name"]] = t
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
    # if some times are 0.0 (all images were pre-existing), mark as N/A
    have_times = all(measured_times.get(exp["name"], 0) > 0 for exp in EXPERIMENTS)

    # ── Phase 2: Load images ───────────────────────────────────────────────
    print("\nLoading images...")
    all_images, all_prompts_list = {}, {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        img_dir = os.path.join(OUTPUT_BASE, name, "images")
        imgs, plist = [], []
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                p = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
                imgs.append(Image.open(p).convert("RGB"))
                plist.append(prompt)
        all_images[name] = imgs
        all_prompts_list[name] = plist
        print(f"  {name}: {len(imgs)} images")

    baseline_images = all_images[EXPERIMENTS[0]["name"]]

    # ── Phase 3: Load evaluators ───────────────────────────────────────────
    evaluators = {}
    for label, cls in [
        ("PickScore",    PickScoreEvaluator),
        ("ImageReward",  ImageRewardEvaluator),
        ("HPSv2",        HPSv2Evaluator),
        ("Aesthetic",    AestheticScoreEvaluator),
        ("LPIPS",        LPIPSEvaluator),
    ]:
        print(f"Loading {label}...", end="", flush=True)
        try:
            evaluators[label] = cls()
            print(" OK")
        except Exception as e:
            print(f" SKIP ({e})")

    # ── Phase 4: Compute metrics ───────────────────────────────────────────
    all_metrics = {}
    baseline_name = EXPERIMENTS[0]["name"]

    for exp in EXPERIMENTS:
        name = exp["name"]
        images = all_images[name]
        plist  = all_prompts_list[name]
        img_dir = os.path.join(OUTPUT_BASE, name, "images")
        m = {}

        print(f"\n--- Metrics: {name} ---")

        print("  CLIP...", end="", flush=True)
        m["clip"] = compute_clip_score(images, plist)
        print(f" {np.mean(m['clip']):.4f}")

        print("  Edge...", end="", flush=True)
        m["edge"] = [compute_edge_density(img) for img in images]
        print(f" {np.mean(m['edge']):.4f}")

        for key, label in [("PickScore","pick"), ("ImageReward","ir"),
                            ("Aesthetic","aesthetic")]:
            if key in evaluators:
                print(f"  {key}...", end="", flush=True)
                m[label] = evaluators[key].score(images, plist) if label != "aesthetic" \
                            else evaluators[key].score(images)
                print(f" {np.mean(m[label]):.4f}")

        if "HPSv2" in evaluators:
            print("  HPSv2...", end="", flush=True)
            m["hps"] = evaluators["HPSv2"].score(images, plist, img_dir)
            print(f" {np.mean(m['hps']):.4f}")

        if "LPIPS" in evaluators and name != baseline_name:
            print("  LPIPS...", end="", flush=True)
            m["lpips"] = evaluators["LPIPS"].score(images, baseline_images)
            print(f" {np.mean(m['lpips']):.4f}")

        all_metrics[name] = m

    del evaluators
    gc.collect()
    torch.cuda.empty_cache()

    # ── FID ────────────────────────────────────────────────────────────────
    fid_scores = {}
    print("\nFID...", end="", flush=True)
    try:
        from cleanfid import fid as cleanfid
        bl_dir = os.path.join(OUTPUT_BASE, baseline_name, "images")
        for exp in EXPERIMENTS[1:]:
            name = exp["name"]
            d = os.path.join(OUTPUT_BASE, name, "images")
            v = cleanfid.compute_fid(bl_dir, d, mode="clean", num_workers=0)
            fid_scores[name] = v
            print(f" {name}={v:.2f}", end="", flush=True)
        print()
    except Exception as e:
        print(f" SKIP ({e})")

    # ── Summary table ──────────────────────────────────────────────────────
    ref_time = measured_times.get(baseline_name, 0)
    print("\n" + "=" * 120)
    print(f"{'Config':<42} {'Time':>7} {'Speed':>6}  CLIP    Edge    Pick      IR     HPS    Aesth   LPIPS    FID")
    print("=" * 120)

    for exp in EXPERIMENTS:
        name = exp["name"]
        t = measured_times.get(name, 0)
        if have_times and ref_time > 0 and t > 0:
            spd_str = f"{ref_time / t:5.2f}x"
            t_str = f"{t:6.2f}s"
        else:
            spd_str = "   N/A"
            t_str = "   N/A "
        m = all_metrics[name]

        def v(key): return f"{np.mean(m[key]):7.4f}" if key in m else "    N/A"

        lpips_str = v("lpips") if name != baseline_name else "  0.000"
        fid_str = f"{fid_scores[name]:7.2f}" if name in fid_scores else "    N/A"

        print(f"  {exp['desc']:<40} {t_str} {spd_str}  "
              f"{v('clip')} {v('edge')} {v('pick')} {v('ir')} {v('hps')} {v('aesthetic')} "
              f"{lpips_str} {fid_str}")

    print("=" * 120)

    # ── Save JSON ──────────────────────────────────────────────────────────
    summary = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        m = all_metrics[name]
        t = measured_times.get(name, 0)
        entry = {"desc": exp["desc"], "mean_time": t,
                 "speedup": (ref_time / t) if (have_times and t > 0 and ref_time > 0) else None}
        for key in ["clip","edge","pick","ir","hps","aesthetic","lpips"]:
            if key in m:
                entry[f"mean_{key}"] = float(np.mean(m[key]))
                entry[f"std_{key}"]  = float(np.std(m[key]))
        if name in fid_scores:
            entry["fid"] = fid_scores[name]
        summary[name] = entry

    out_json = os.path.join(OUTPUT_BASE, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_json}")
    print("=== Done! ===")


if __name__ == "__main__":
    main()
