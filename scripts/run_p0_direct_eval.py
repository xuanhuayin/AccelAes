#!/usr/bin/env python3
"""
P0 Ablation — Direct threshold mask (no SLIC).

对 P0 ablation 实验的完整复现，唯一改动是把 region_method="slic" 换成
region_method="threshold"（直接百分位阈值，不做 superpixel 聚类）。

Baseline 和 fskip_only 图片从 outputs/p0_ablation/ 直接复用（它们不用 mask）。
SASD 变体（attn_only/attn_ffn/sasd_full）重新生成。
输出到 outputs/p0_ablation_direct/

Usage:
  python scripts/run_p0_direct_eval.py              # gen + metrics
  python scripts/run_p0_direct_eval.py --skip_gen   # metrics only (images exist)
"""

import argparse
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

from src.models.dit_wrapper import LuminaDiTWrapper
from src.eval.metrics import compute_clip_score, compute_edge_density

OUTPUT_BASE   = "outputs/p0_ablation_direct"
SLIC_BASE     = "outputs/p0_ablation"          # 复用 baseline / fskip_only

# ── Experiment definitions（region_method="threshold" for SASD configs）───────

EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "Baseline (dense, CFG=4.0)",
        "fn": "reuse",   # 直接从 SLIC_BASE 复用
    },
    {
        "name": "fskip_only",
        "desc": "Step cache only (fskip2, no spatial)",
        "fn": "reuse",
    },
    {
        "name": "attn_only",
        "desc": "Sparse attn + spatial CFG, direct thresh (no FFN, no fskip)",
        "fn": "sasd",
        "kwargs": dict(
            mask_type="semantic", region_method="threshold",
            skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
            full_skip_interval=0, sparse_ffn=False, sparse_blocks=True,
        ),
    },
    {
        "name": "attn_ffn",
        "desc": "Sparse attn+FFN + spatial CFG, direct thresh (no fskip)",
        "fn": "sasd",
        "kwargs": dict(
            mask_type="semantic", region_method="threshold",
            skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
            full_skip_interval=0, sparse_ffn=True, sparse_blocks=True,
        ),
    },
    {
        "name": "sasd_full",
        "desc": "SASD full, direct thresh (attn+ffn+spatial_cfg+fskip2)",
        "fn": "sasd",
        "kwargs": dict(
            mask_type="semantic", region_method="threshold",
            skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
            full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
        ),
    },
]


# ── Metric evaluators (same as run_p0_eval.py) ────────────────────────────────

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
                text=[prompt], images=[img], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
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
        import tempfile
        scores = []
        with tempfile.TemporaryDirectory() as td:
            for i, (img, prompt) in enumerate(zip(images, prompts)):
                tmp = os.path.join(td, f"{i}.png")
                img.save(tmp)
                r = self.hpsv2.score(tmp, prompt, hps_version="v2.1")
                scores.append(float(r[0]) if isinstance(r, (list, np.ndarray)) else float(r))
        return scores


class AestheticScoreEvaluator:
    def __init__(self, device="cuda"):
        import open_clip
        import torch.nn as nn
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device)
        self.model.eval()
        cache = os.path.join(os.path.dirname(__file__), "..", ".cache")
        os.makedirs(cache, exist_ok=True)
        wp = os.path.join(cache, "aesthetic_mlp_l14.pth")
        if not os.path.exists(wp):
            url = ("https://github.com/christophschuhmann/"
                   "improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth")
            torch.hub.download_url_to_file(url, wp)
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1),
        ).to(device)
        state = torch.load(wp, map_location=device, weights_only=True)
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
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        return [
            self.fn(tf(a).unsqueeze(0).to(self.device),
                    tf(b).unsqueeze(0).to(self.device)).item()
            for a, b in zip(images, refs)
        ]


# ── Generation ────────────────────────────────────────────────────────────────

def run_generation(wrapper, prompts, seeds):
    """Generate images; reuse baseline/fskip_only from SLIC_BASE."""
    gen_times = {}
    total = len(prompts) * len(seeds)

    for exp in EXPERIMENTS:
        name = exp["name"]
        img_dir = os.path.join(OUTPUT_BASE, name, "images")
        os.makedirs(img_dir, exist_ok=True)

        if exp["fn"] == "reuse":
            # Symlink / copy from p0_ablation
            src_dir = os.path.join(SLIC_BASE, name, "images")
            if not os.path.isdir(src_dir):
                print(f"  WARNING: {src_dir} not found, skipping {name}")
                gen_times[name] = 0.0
                continue
            # Create symlinks for any missing files
            linked = 0
            for pi, prompt in enumerate(prompts):
                for seed in seeds:
                    fname = f"p{pi:04d}_s{seed:04d}.png"
                    dst = os.path.join(img_dir, fname)
                    src = os.path.abspath(os.path.join(src_dir, fname))
                    if not os.path.exists(dst) and os.path.exists(src):
                        os.symlink(src, dst)
                        linked += 1
            print(f"  [{name}] reused {total} images from SLIC_BASE ({linked} symlinked)")
            gen_times[name] = 0.0   # timing loaded from SLIC_BASE summary
            continue

        # SASD variant: generate fresh
        # Defrag GPU between configs
        if wrapper is not None and hasattr(wrapper, "transformer"):
            wrapper.transformer.to("cpu")
            gc.collect(); torch.cuda.empty_cache()
            wrapper.transformer.to("cuda")

        print(f"\n=== [{name}] {exp['desc']} ({total} images) ===")
        times, count, skipped = [], 0, 0
        kw = exp["kwargs"]

        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                out_path = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
                if os.path.exists(out_path):
                    skipped += 1; count += 1
                    continue

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                img = wrapper.generate_accelerated_dual(prompt=prompt, seed=seed, **kw)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                times.append(elapsed)
                img.save(out_path)
                count += 1

                if count % 10 == 0 or count == total:
                    avg = sum(times) / len(times)
                    print(f"  [{count}/{total}] avg {avg:.2f}s/img")

                del img
                gc.collect(); torch.cuda.empty_cache()

        mean_t = sum(times) / len(times) if times else 0.0
        print(f"  Done: {mean_t:.2f}s/img ({skipped} skipped)")
        gen_times[name] = mean_t

    return gen_times


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_all_metrics(prompts, seeds):
    total = len(prompts) * len(seeds)
    baseline_name = "baseline"

    print("\nLoading images...")
    all_imgs, all_plist = {}, {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        img_dir = os.path.join(OUTPUT_BASE, name, "images")
        imgs, plist = [], []
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                p = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
                imgs.append(Image.open(p).convert("RGB"))
                plist.append(prompt)
        all_imgs[name] = imgs
        all_plist[name] = plist
        print(f"  {name}: {len(imgs)} images")

    baseline_imgs = all_imgs[baseline_name]

    evaluators = {}
    for label, cls in [
        ("PickScore",   PickScoreEvaluator),
        ("ImageReward", ImageRewardEvaluator),
        ("HPSv2",       HPSv2Evaluator),
        ("Aesthetic",   AestheticScoreEvaluator),
        ("LPIPS",       LPIPSEvaluator),
    ]:
        print(f"Loading {label}...", end="", flush=True)
        try:
            evaluators[label] = cls()
            print(" OK")
        except Exception as e:
            print(f" SKIP ({e})")

    all_metrics = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        images  = all_imgs[name]
        plist   = all_plist[name]
        img_dir = os.path.join(OUTPUT_BASE, name, "images")
        m = {}
        print(f"\n--- Metrics: {name} ---")

        print("  CLIP...", end="", flush=True)
        m["clip"] = compute_clip_score(images, plist)
        print(f" {np.mean(m['clip']):.4f}")

        print("  Edge...", end="", flush=True)
        m["edge"] = [compute_edge_density(img) for img in images]
        print(f" {np.mean(m['edge']):.4f}")

        for label, key in [("PickScore", "pick"), ("ImageReward", "ir"), ("Aesthetic", "aesthetic")]:
            if label in evaluators:
                print(f"  {label}...", end="", flush=True)
                if key == "aesthetic":
                    m[key] = evaluators[label].score(images)
                else:
                    m[key] = evaluators[label].score(images, plist)
                print(f" {np.mean(m[key]):.4f}")

        if "HPSv2" in evaluators:
            print("  HPSv2...", end="", flush=True)
            m["hps"] = evaluators["HPSv2"].score(images, plist, img_dir)
            print(f" {np.mean(m['hps']):.4f}")

        if "LPIPS" in evaluators and name != baseline_name:
            print("  LPIPS...", end="", flush=True)
            m["lpips"] = evaluators["LPIPS"].score(images, baseline_imgs)
            print(f" {np.mean(m['lpips']):.4f}")
        elif name == baseline_name:
            m["lpips"] = [0.0] * total

        all_metrics[name] = m

    del evaluators
    gc.collect(); torch.cuda.empty_cache()

    # FID
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

    return all_metrics, fid_scores


# ── Summary ───────────────────────────────────────────────────────────────────

def print_and_save(gen_times, all_metrics, fid_scores, prompts, seeds):
    # Load timing for reused configs from SLIC_BASE summary
    slic_summary_path = os.path.join(SLIC_BASE, "summary.json")
    slic_times = {}
    if os.path.exists(slic_summary_path):
        with open(slic_summary_path) as f:
            slic_data = json.load(f)
        for name, d in slic_data.items():
            slic_times[name] = d.get("mean_time", 0.0)

    # Merge times: reused configs from slic_times, new configs from gen_times
    final_times = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        if exp["fn"] == "reuse":
            final_times[name] = slic_times.get(name, 0.0)
        else:
            final_times[name] = gen_times.get(name, 0.0)

    ref_t = final_times.get("baseline", 0.0)

    print("\n" + "=" * 130)
    print(f"{'Config':<45} {'Time':>7} {'Speed':>6}  "
          "CLIP    Edge    Pick      IR     HPS    Aesth   LPIPS    FID")
    print("=" * 130)

    summary = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        t = final_times[name]
        m = all_metrics[name]

        spd = f"{ref_t / t:5.2f}x" if (ref_t > 0 and t > 0) else "  N/A "
        t_str = f"{t:6.2f}s" if t > 0 else "   N/A "

        def v(key):
            return f"{np.mean(m[key]):7.4f}" if key in m else "    N/A"

        fid = f"{fid_scores[name]:7.2f}" if name in fid_scores else "    N/A"
        print(f"  {exp['desc']:<43} {t_str} {spd}  "
              f"{v('clip')} {v('edge')} {v('pick')} {v('ir')} "
              f"{v('hps')} {v('aesthetic')} {v('lpips')} {fid}")

        entry = {"desc": exp["desc"], "mean_time": t,
                 "speedup": (ref_t / t) if (ref_t > 0 and t > 0) else None}
        for key in ["clip", "edge", "pick", "ir", "hps", "aesthetic", "lpips"]:
            if key in m:
                entry[f"mean_{key}"] = float(np.mean(m[key]))
                entry[f"std_{key}"]  = float(np.std(m[key]))
        if name in fid_scores:
            entry["fid"] = fid_scores[name]
        elif name == "baseline":
            entry["fid"] = 0.0
        summary[name] = entry

    print("=" * 130)

    # Also print SLIC vs direct comparison for sasd_full
    print("\n--- SLIC vs Direct threshold (sasd_full) ---")
    if os.path.exists(slic_summary_path):
        with open(slic_summary_path) as f:
            slic_data = json.load(f)
        if "sasd_full" in slic_data and "sasd_full" in summary:
            s = slic_data["sasd_full"]
            d = summary["sasd_full"]
            print(f"  SLIC:   time={s['mean_time']:.2f}s  IR={s['mean_ir']:.4f}  "
                  f"LPIPS={s['mean_lpips']:.4f}  FID={s.get('fid', 0):.2f}  "
                  f"CLIP={s['mean_clip']:.4f}  Aesth={s['mean_aesthetic']:.4f}")
            print(f"  Direct: time={d['mean_time']:.2f}s  IR={d.get('mean_ir', 0):.4f}  "
                  f"LPIPS={d.get('mean_lpips', 0):.4f}  FID={d.get('fid', 0):.2f}  "
                  f"CLIP={d.get('mean_clip', 0):.4f}  Aesth={d.get('mean_aesthetic', 0):.4f}")
            if d.get("mean_ir") and s.get("mean_ir"):
                delta_ir = (d["mean_ir"] - s["mean_ir"]) / s["mean_ir"] * 100
                delta_lpips = (d.get("mean_lpips", 0) - s.get("mean_lpips", 0))
                print(f"  Δ IR = {delta_ir:+.1f}%   Δ LPIPS = {delta_lpips:+.4f}")

    out_json = os.path.join(OUTPUT_BASE, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_json}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts",     default="prompts/prompts_dev.txt")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--seeds",       nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip_gen",    action="store_true")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = [l.strip() for l in f if l.strip()][:args.num_prompts]
    seeds = args.seeds
    print(f"Prompts: {len(prompts)}, Seeds: {seeds}, Total/config: {len(prompts) * len(seeds)}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"Reusing baseline/fskip_only from: {SLIC_BASE}")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    gen_times = {}
    if not args.skip_gen:
        wrapper = LuminaDiTWrapper(dtype="bf16")
        gen_times = run_generation(wrapper, prompts, seeds)
        del wrapper
        gc.collect(); torch.cuda.empty_cache()
    else:
        print("Skipping generation (--skip_gen)")

    all_metrics, fid_scores = compute_all_metrics(prompts, seeds)
    print_and_save(gen_times, all_metrics, fid_scores, prompts, seeds)


if __name__ == "__main__":
    main()
