"""
Full 8-metric evaluation of step-level caching baselines:
  TeaCache (arXiv 2411.19108) and TaylorSeer (arXiv 2503.06923)
  on Lumina-Next-T2I and FLUX.1-dev.

Lumina (30 steps, 20 prompts × seeds=[42, 1337] = 40 images/config):
  Configs: baseline, teacache_t006, teacache_t010, teacache_t015, teacache_t020,
           taylor_r1, taylor_r2, taylor_r3

FLUX (28 steps, 24 prompts × seed=42 = 24 images/config):
  Configs: baseline, teacache_t010, teacache_t015, teacache_t020, teacache_t030,
           taylor_r1_o2, taylor_r2_o2, taylor_r4_o2

All 8 metrics: CLIP, PickScore, IR, HPSv2, Aesthetic, LPIPS (vs baseline), Edge, FID.

Results:
  outputs/stepcache_lumina/summary.json
  outputs/stepcache_flux/summary.json

Usage:
  python scripts/run_stepcache_fulleval.py --model lumina
  python scripts/run_stepcache_fulleval.py --model flux
  python scripts/run_stepcache_fulleval.py --model all
  python scripts/run_stepcache_fulleval.py --model all --skip_gen   # metrics only
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import compute_clip_score, compute_edge_density


# ── Metric evaluators (copied from run_p0_eval.py) ────────────────────────────

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
    def score(self, images, prompts):
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
                   "improved-aesthetic-predictor/raw/main/"
                   "sac+logos+ava1-l14-linearMSE.pth")
            torch.hub.download_url_to_file(url, wp)
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
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
        return [self.fn(tf(a).unsqueeze(0).to(self.device),
                        tf(b).unsqueeze(0).to(self.device)).item()
                for a, b in zip(images, refs)]


# ── Common metric runner ───────────────────────────────────────────────────────

def compute_all_metrics(experiments, output_base, prompts, seeds):
    """Load images, compute all 8 metrics, return (all_metrics, fid_scores)."""
    baseline_name = experiments[0]["name"]
    total = len(prompts) * len(seeds)

    # Load images
    print("\nLoading images...")
    all_imgs, all_plist = {}, {}
    for exp in experiments:
        name = exp["name"]
        img_dir = os.path.join(output_base, name, "images")
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

    # Load evaluators
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

    # Per-experiment metrics
    all_metrics = {}
    for exp in experiments:
        name = exp["name"]
        images = all_imgs[name]
        plist  = all_plist[name]
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
                m[key] = evaluators[label].score(images) if key == "aesthetic" \
                    else evaluators[label].score(images, plist)
                print(f" {np.mean(m[key]):.4f}")

        if "HPSv2" in evaluators:
            print("  HPSv2...", end="", flush=True)
            m["hps"] = evaluators["HPSv2"].score(images, plist)
            print(f" {np.mean(m['hps']):.4f}")

        if "LPIPS" in evaluators:
            print("  LPIPS...", end="", flush=True)
            m["lpips"] = [0.0] * total if name == baseline_name \
                else evaluators["LPIPS"].score(images, baseline_imgs)
            print(f" {np.mean(m['lpips']):.4f}")

        all_metrics[name] = m

    del evaluators; gc.collect(); torch.cuda.empty_cache()

    # FID
    fid_scores = {}
    print("\nFID...", end="", flush=True)
    try:
        from cleanfid import fid as cleanfid
        bl_dir = os.path.join(output_base, baseline_name, "images")
        for exp in experiments[1:]:
            name = exp["name"]
            d = os.path.join(output_base, name, "images")
            v = cleanfid.compute_fid(bl_dir, d, mode="clean", num_workers=0)
            fid_scores[name] = v
            print(f" {name}={v:.2f}", end="", flush=True)
        print()
    except Exception as e:
        print(f" SKIP ({e})")

    return all_metrics, fid_scores


def save_summary(experiments, output_base, gen_results, all_metrics, fid_scores, prompts, seeds):
    """Build and save summary.json; print final table."""
    baseline_name = experiments[0]["name"]
    baseline_time  = gen_results.get(baseline_name, (None, 0.0))[1] or 0.0

    summary = {}
    for exp in experiments:
        name = exp["name"]
        _, mean_time = gen_results.get(name, (None, 0.0))
        m = all_metrics.get(name, {})
        speedup = (baseline_time / mean_time) if mean_time > 0 else 1.0

        summary[name] = {
            "desc":       exp.get("desc", name),
            "mean_time":  mean_time,
            "speedup":    speedup,
            "mean_clip":  float(np.mean(m["clip"]))      if "clip"      in m else None,
            "mean_pick":  float(np.mean(m["pick"]))      if "pick"      in m else None,
            "mean_ir":    float(np.mean(m["ir"]))        if "ir"        in m else None,
            "mean_hps":   float(np.mean(m["hps"]))       if "hps"       in m else None,
            "mean_aesthetic": float(np.mean(m["aesthetic"])) if "aesthetic" in m else None,
            "mean_lpips": float(np.mean(m["lpips"]))     if "lpips"     in m else None,
            "mean_edge":  float(np.mean(m["edge"]))      if "edge"      in m else None,
            "fid":        fid_scores.get(name, 0.0 if name == baseline_name else None),
        }

    summary_path = os.path.join(output_base, "summary.json")
    os.makedirs(output_base, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {summary_path}")

    # Print table
    base = summary[baseline_name]
    ref_ir   = base.get("mean_ir")   or 1.0
    ref_pick = base.get("mean_pick") or 1.0
    print("\n" + "=" * 110)
    print(f"{'Config':<20} {'Spd':>5} {'CLIP':>7} {'Pick':>7} {'IR':>7} "
          f"{'HPS':>7} {'Aesth':>7} {'LPIPS':>7} {'Edge':>7} {'FID':>7}")
    print("-" * 110)
    for name, s in summary.items():
        spd   = f"{s['speedup']:.2f}×"
        clip_ = f"{s['mean_clip']:.4f}"  if s["mean_clip"]  is not None else "  N/A"
        pick_ = f"{s['mean_pick']:.4f}"  if s["mean_pick"]  is not None else "  N/A"
        ir_   = f"{s['mean_ir']:.4f}"    if s["mean_ir"]    is not None else "  N/A"
        hps_  = f"{s['mean_hps']:.4f}"   if s["mean_hps"]   is not None else "  N/A"
        aes_  = f"{s['mean_aesthetic']:.4f}" if s["mean_aesthetic"] is not None else "  N/A"
        lp_   = f"{s['mean_lpips']:.4f}" if s["mean_lpips"] is not None else "  N/A"
        edg_  = f"{s['mean_edge']:.4f}"  if s["mean_edge"]  is not None else "  N/A"
        fid_  = f"{s['fid']:.2f}"        if s["fid"]        is not None else "  N/A"
        print(f"{name:<20} {spd:>5} {clip_:>7} {pick_:>7} {ir_:>7} "
              f"{hps_:>7} {aes_:>7} {lp_:>7} {edg_:>7} {fid_:>7}")
    print("=" * 110)

    return summary


# ── Lumina evaluation ──────────────────────────────────────────────────────────

def run_lumina(skip_gen: bool = False):
    from src.models.dit_wrapper import LuminaDiTWrapper
    from src.baselines.teacache import generate_teacache_lumina, generate_taylor_lumina

    OUTPUT_BASE = "outputs/stepcache_lumina"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    with open("prompts/prompts_dev.txt") as f:
        prompts = [l.strip() for l in f if l.strip()][:20]
    seeds = [42, 1337]

    EXPERIMENTS = [
        dict(name="baseline",        desc="Baseline (dense, CFG=4.0)",                    fn="baseline"),
        dict(name="teacache_t006",   desc="TeaCache threshold=0.06",                       fn="teacache", threshold=0.06),
        dict(name="teacache_t010",   desc="TeaCache threshold=0.10",                       fn="teacache", threshold=0.10),
        dict(name="teacache_t015",   desc="TeaCache threshold=0.15",                       fn="teacache", threshold=0.15),
        dict(name="teacache_t020",   desc="TeaCache threshold=0.20",                       fn="teacache", threshold=0.20),
        dict(name="taylor_r1",       desc="TaylorSeer run=1 (skip×1, ~1.50×)",             fn="taylor",   taylor_run=1),
        dict(name="taylor_r2",       desc="TaylorSeer run=2 (skip×2, ~2.0×)",              fn="taylor",   taylor_run=2),
        dict(name="taylor_r3",       desc="TaylorSeer run=3 (skip×3, ~2.5×)",              fn="taylor",   taylor_run=3),
    ]

    gen_results = {}
    total = len(prompts) * len(seeds)

    if not skip_gen:
        print("Loading Lumina-Next-T2I...")
        wrapper = LuminaDiTWrapper()

        for exp in EXPERIMENTS:
            name = exp["name"]
            img_dir = os.path.join(OUTPUT_BASE, name, "images")
            os.makedirs(img_dir, exist_ok=True)

            print(f"\n=== [{name}] {exp['desc']} ({total} images) ===")
            times, count, skipped = [], 0, 0

            for pi, prompt in enumerate(prompts):
                for seed in seeds:
                    out_path = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
                    if os.path.exists(out_path):
                        skipped += 1; count += 1
                        continue

                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    fn = exp["fn"]
                    if fn == "baseline":
                        img = wrapper.generate(prompt=prompt, seed=seed, cfg_scale=4.0)
                    elif fn == "teacache":
                        img = generate_teacache_lumina(
                            wrapper, prompt=prompt, seed=seed,
                            threshold=exp["threshold"], steps=30)
                    elif fn == "taylor":
                        img = generate_taylor_lumina(
                            wrapper, prompt=prompt, seed=seed,
                            taylor_run=exp["taylor_run"], steps=30)
                    else:
                        raise ValueError(f"Unknown fn: {fn}")

                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    img.save(out_path)
                    count += 1

                    if count % 10 == 0 or count == total:
                        avg = sum(times) / len(times)
                        print(f"  [{count}/{total}] avg {avg:.2f}s/img  ({skipped} skipped)")

                    del img; gc.collect(); torch.cuda.empty_cache()

            mean_t = sum(times) / len(times) if times else 0.0
            print(f"  Done: mean={mean_t:.2f}s/img  ({skipped} already existed)")
            gen_results[name] = (img_dir, mean_t)

        del wrapper; gc.collect(); torch.cuda.empty_cache()
    else:
        print("Skipping generation (--skip_gen).")
        for exp in EXPERIMENTS:
            name = exp["name"]
            img_dir = os.path.join(OUTPUT_BASE, name, "images")
            gen_results[name] = (img_dir, 0.0)
        # Try to load existing timing from summary
        sp = os.path.join(OUTPUT_BASE, "summary.json")
        if os.path.exists(sp):
            with open(sp) as f:
                prev = json.load(f)
            for name in prev:
                if name in gen_results:
                    gen_results[name] = (gen_results[name][0], prev[name].get("mean_time", 0.0))

    all_metrics, fid_scores = compute_all_metrics(EXPERIMENTS, OUTPUT_BASE, prompts, seeds)
    save_summary(EXPERIMENTS, OUTPUT_BASE, gen_results, all_metrics, fid_scores, prompts, seeds)


# ── FLUX evaluation ────────────────────────────────────────────────────────────

def run_flux(skip_gen: bool = False):
    from src.models.flux_wrapper import FluxDiTWrapper

    OUTPUT_BASE = "outputs/stepcache_flux"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Use the same 24 prompts as the main FLUX eval
    ref_path = "outputs/accelae_topk_flux/scores_fskip2.json"
    with open(ref_path) as f:
        ref = json.load(f)
    tags    = list(ref.keys())
    prompts = [ref[tag]["prompt"] for tag in tags]   # ordered list, index = pi
    seeds   = [42]

    EXPERIMENTS = [
        dict(name="baseline",       desc="Baseline (dense, CFG=3.5)",              fn="baseline"),
        dict(name="accelae_fskip2", desc="AccelAes fskip2 (interval=2, warmup=5)", fn="fskip2",   full_skip_interval=2, warmup_steps=5),
        dict(name="teacache_t010",  desc="TeaCache threshold=0.10",                fn="teacache", threshold=0.10),
        dict(name="teacache_t015",  desc="TeaCache threshold=0.15",                fn="teacache", threshold=0.15),
        dict(name="teacache_t020",  desc="TeaCache threshold=0.20",                fn="teacache", threshold=0.20),
        dict(name="teacache_t030",  desc="TeaCache threshold=0.30",                fn="teacache", threshold=0.30),
        dict(name="taylor_r1_o2",   desc="TaylorSeer run=1, order=2 (~1.63×)",     fn="taylor",   taylor_run=1, taylor_order=2),
        dict(name="taylor_r2_o2",   desc="TaylorSeer run=2, order=2 (~2.12×)",     fn="taylor",   taylor_run=2, taylor_order=2),
        dict(name="taylor_r4_o2",   desc="TaylorSeer run=4, order=2 (~2.74×)",     fn="taylor",   taylor_run=4, taylor_order=2),
    ]

    gen_results = {}
    total = len(prompts) * len(seeds)

    if not skip_gen:
        print("Loading FLUX.1-dev...")
        wrapper = FluxDiTWrapper(dtype="bf16")

        print("Pre-encoding prompts...")
        for prompt in prompts:
            wrapper._encode_prompt(prompt)
        print(f"  {len(prompts)} prompts cached.\n")

        for exp in EXPERIMENTS:
            name = exp["name"]
            img_dir = os.path.join(OUTPUT_BASE, name, "images")
            os.makedirs(img_dir, exist_ok=True)

            print(f"\n=== [{name}] {exp['desc']} ({total} images) ===")
            times, count, skipped = [], 0, 0

            for pi, prompt in enumerate(prompts):
                for seed in seeds:
                    out_path = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
                    if os.path.exists(out_path):
                        skipped += 1; count += 1
                        continue

                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    fn = exp["fn"]
                    if fn == "baseline":
                        img = wrapper.generate(prompt=prompt, seed=seed, steps=28)
                    elif fn == "fskip2":
                        img = wrapper.generate_accelerated(
                            prompt=prompt, seed=seed, steps=28,
                            full_skip_interval=exp["full_skip_interval"],
                            warmup_steps=exp["warmup_steps"])
                    elif fn == "teacache":
                        img = wrapper.generate_teacache(
                            prompt=prompt, seed=seed, steps=28,
                            threshold=exp["threshold"], warmup_steps=3)
                    elif fn == "taylor":
                        img = wrapper.generate_taylor(
                            prompt=prompt, seed=seed, steps=28,
                            taylor_run=exp["taylor_run"],
                            taylor_order=exp["taylor_order"],
                            warmup_steps=5)
                    else:
                        raise ValueError(f"Unknown fn: {fn}")

                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    img.save(out_path)
                    count += 1

                    if count % 8 == 0 or count == total:
                        avg = sum(times) / len(times)
                        print(f"  [{count}/{total}] avg {avg:.2f}s/img  ({skipped} skipped)")

                    del img; gc.collect(); torch.cuda.empty_cache()

            mean_t = sum(times) / len(times) if times else 0.0
            print(f"  Done: mean={mean_t:.2f}s/img  ({skipped} already existed)")
            gen_results[name] = (img_dir, mean_t)

        del wrapper; gc.collect(); torch.cuda.empty_cache()
    else:
        print("Skipping generation (--skip_gen).")
        for exp in EXPERIMENTS:
            name = exp["name"]
            img_dir = os.path.join(OUTPUT_BASE, name, "images")
            gen_results[name] = (img_dir, 0.0)
        sp = os.path.join(OUTPUT_BASE, "summary.json")
        if os.path.exists(sp):
            with open(sp) as f:
                prev = json.load(f)
            for name in prev:
                if name in gen_results:
                    gen_results[name] = (gen_results[name][0], prev[name].get("mean_time", 0.0))

    all_metrics, fid_scores = compute_all_metrics(EXPERIMENTS, OUTPUT_BASE, prompts, seeds)
    save_summary(EXPERIMENTS, OUTPUT_BASE, gen_results, all_metrics, fid_scores, prompts, seeds)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lumina", "flux", "all"], default="all")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip image generation (metrics only, images must already exist)")
    args = parser.parse_args()

    if args.model in ("lumina", "all"):
        print("\n" + "#" * 60)
        print("# TeaCache + TaylorSeer on Lumina (full 8-metric eval)")
        print("#" * 60)
        run_lumina(skip_gen=args.skip_gen)

    if args.model in ("flux", "all"):
        print("\n" + "#" * 60)
        print("# TeaCache + TaylorSeer on FLUX (full 8-metric eval)")
        print("#" * 60)
        run_flux(skip_gen=args.skip_gen)


if __name__ == "__main__":
    main()
