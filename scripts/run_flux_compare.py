#!/usr/bin/env python3
"""
FLUX.1-dev comparison: AccelAes vs step-level and layer-level caching baselines.

24 prompts × seed=42 = 24 images/config (matches stepcache_flux eval).
8 metrics: CLIP, PickScore, ImageReward, HPSv2, Aesthetic, LPIPS (vs baseline), Edge, FID.

Configs:
  baseline        — dense generation (reused from outputs/eval/flux/baseline)
  accelae         — AccelAes fskip2 (reused from outputs/eval/flux/accelae_fskip2)
  teacache_t015   — TeaCache threshold=0.15 (reused from stepcache_flux)
  teacache_t030   — TeaCache threshold=0.30 (reused from stepcache_flux)
  taylor_r1       — TaylorSeer run=1, order=2 (reused from stepcache_flux)
  taylor_r2       — TaylorSeer run=2, order=2 (reused from stepcache_flux)
  delta_dit       — Delta-DiT (3-tier layer caching, 19+38 blocks)
  fora            — FORA (residual reuse, ~1.00× due to pre-hook limitation)

Usage:
  python scripts/run_flux_compare.py
  python scripts/run_flux_compare.py --skip_gen
  python scripts/run_flux_compare.py --only_config delta_dit
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

OUTPUT_DIR         = "outputs/flux_compare"
STEPCACHE_FLUX_DIR = "outputs/stepcache_flux"
EVAL_FLUX_DIR      = "outputs/eval/flux"
NUM_PROMPTS        = 24
SEED               = 42
STEPS              = 28
GUIDANCE           = 3.5

# ── Compare configs ────────────────────────────────────────────────────────────

COMPARE_CONFIGS = [
    {
        "name": "baseline",
        "desc": "Baseline (dense)",
        # stepcache_flux stores images in {config}/images/ subdirectory
        "ref_dir": f"{STEPCACHE_FLUX_DIR}/baseline/images",
        "ref_time_key": "baseline",
    },
    {
        "name": "accelae",
        "desc": "AccelAes (fskip2, ours)",
        "ref_dir": f"{STEPCACHE_FLUX_DIR}/accelae_fskip2/images",
        "ref_time_key": "accelae_fskip2",
    },
    {
        "name": "teacache_t015",
        "desc": "TeaCache threshold=0.15",
        "ref_dir": f"{STEPCACHE_FLUX_DIR}/teacache_t015/images",
        "ref_time_key": "teacache_t015",
    },
    {
        "name": "teacache_t030",
        "desc": "TeaCache threshold=0.30",
        "ref_dir": f"{STEPCACHE_FLUX_DIR}/teacache_t030/images",
        "ref_time_key": "teacache_t030",
    },
    {
        "name": "taylor_r1",
        "desc": "TaylorSeer run=1, order=2",
        "ref_dir": f"{STEPCACHE_FLUX_DIR}/taylor_r1_o2/images",
        "ref_time_key": "taylor_r1_o2",
    },
    {
        "name": "taylor_r2",
        "desc": "TaylorSeer run=2, order=2",
        "ref_dir": f"{STEPCACHE_FLUX_DIR}/taylor_r2_o2/images",
        "ref_time_key": "taylor_r2_o2",
    },
    {
        "name": "delta_dit",
        "desc": "Delta-DiT (3-tier layer cache)",
        "fn": "delta_dit",
        "kwargs": dict(warmup_steps=2),
    },
    {
        "name": "fora",
        "desc": "FORA (residual reuse)",
        "fn": "fora",
        "kwargs": dict(reuse_interval=2, warmup_steps=2),
    },
]


# ── Metric evaluators (shared with other eval scripts) ────────────────────────

class PickScoreEvaluator:
    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, AutoModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = (
            AutoModel.from_pretrained("yuvalkirstain/PickScore_v1")
            .eval().to(device)
        )

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
                scores.append(
                    float(r[0]) if isinstance(r, (list, np.ndarray)) else float(r)
                )
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
        wp = os.path.join(cache, "aesthetic_mlp_l14.pth")
        if not os.path.exists(wp):
            url = ("https://github.com/christophschuhmann/"
                   "improved-aesthetic-predictor/raw/main/"
                   "sac+logos+ava1-l14-linearMSE.pth")
            torch.hub.download_url_to_file(url, wp)
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.Dropout(0.1),
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
        tf = T.Compose([
            T.Resize((256, 256)), T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        return [
            self.fn(
                tf(a).unsqueeze(0).to(self.device),
                tf(b).unsqueeze(0).to(self.device),
            ).item()
            for a, b in zip(images, refs)
        ]


def compute_fid(dir_a, dir_b):
    from cleanfid import fid
    return fid.compute_fid(dir_a, dir_b, mode="clean", num_workers=0)


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_prompts(path, n):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()][:n]


def load_images_single_seed(img_dir, prompts, seed):
    imgs, plist = [], []
    for pi in range(len(prompts)):
        p = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
        imgs.append(Image.open(p).convert("RGB"))
        plist.append(prompts[pi])
    return imgs, plist


def run_generation(wrapper, cfg, prompts, seed, out_dir):
    from src.baselines.delta_dit import generate_delta_dit_flux
    from src.baselines.fora import generate_fora_flux

    fn_map = {
        "delta_dit": generate_delta_dit_flux,
        "fora":      generate_fora_flux,
    }
    gen_fn = fn_map[cfg["fn"]]

    os.makedirs(out_dir, exist_ok=True)
    times, imgs, plist = [], [], []
    total = len(prompts)
    skipped = 0

    for pi, prompt in enumerate(prompts):
        path = os.path.join(out_dir, f"p{pi:04d}_s{seed:04d}.png")
        if os.path.exists(path):
            imgs.append(Image.open(path).convert("RGB"))
            plist.append(prompt)
            skipped += 1
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = gen_fn(wrapper, prompt, seed, steps=STEPS,
                     guidance_scale=GUIDANCE, **cfg["kwargs"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        img.save(path)
        imgs.append(img)
        plist.append(prompt)
        times.append(elapsed)

        print(f"  [{pi+1}/{total}] skip={skipped} last={elapsed:.2f}s")
        del img; gc.collect(); torch.cuda.empty_cache()

    mean_time = float(np.mean(times)) if times else None
    print(f"  Done: skip={skipped}/{total}, mean_time={mean_time}")
    return imgs, plist, mean_time


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_gen",    action="store_true")
    parser.add_argument("--only_config", type=str, default=None)
    parser.add_argument("--prompts_file", default="prompts/prompts_dev.txt")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prompts = load_prompts(args.prompts_file, NUM_PROMPTS)
    print(f"FLUX compare: {len(prompts)} prompts × seed={SEED} = "
          f"{len(prompts)} imgs/config, {len(COMPARE_CONFIGS)} configs")

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # ── Step 1: Generate images ────────────────────────────────────────────────
    gen_times = {}

    if not args.skip_gen:
        wrapper = None

        for cfg in COMPARE_CONFIGS:
            name = cfg["name"]
            if args.only_config and name != args.only_config:
                continue
            if cfg.get("ref_dir"):
                print(f"\n[gen] {name}: reusing {cfg['ref_dir']}")
                continue

            print(f"\n=== [{name}] {cfg['desc']} ===")
            out_dir = os.path.join(OUTPUT_DIR, name)

            all_exist = all(
                os.path.exists(os.path.join(out_dir, f"p{pi:04d}_s{SEED:04d}.png"))
                for pi in range(len(prompts))
            )
            if all_exist:
                print(f"  All {len(prompts)} images cached, skipping.")
                continue

            if wrapper is None:
                from src.models.flux_wrapper import FLUXWrapper
                wrapper = FLUXWrapper(dtype="bf16")

            imgs, plist, mean_time = run_generation(wrapper, cfg, prompts, SEED, out_dir)
            gen_times[name] = mean_time

        if wrapper is not None:
            del wrapper; gc.collect(); torch.cuda.empty_cache()

    # ── Step 2: Load all images ────────────────────────────────────────────────
    all_imgs  = {}
    all_plist = {}

    for cfg in COMPARE_CONFIGS:
        name = cfg["name"]
        img_dir = cfg.get("ref_dir") or os.path.join(OUTPUT_DIR, name)

        if not os.path.isdir(img_dir):
            print(f"  [skip] {name}: no image dir {img_dir}")
            continue

        try:
            imgs, plist = load_images_single_seed(img_dir, prompts, SEED)
            all_imgs[name]  = imgs
            all_plist[name] = plist
            print(f"Loaded {len(imgs)} images: {name}")
        except FileNotFoundError as e:
            print(f"  [skip] {name}: {e}")
            continue

    if "baseline" not in all_imgs:
        print("ERROR: baseline images not found.")
        sys.exit(1)

    # ── Step 3: Compute metrics ────────────────────────────────────────────────
    names_to_score = list(all_imgs.keys())
    device = "cuda"

    for name in names_to_score:
        if name not in summary:
            summary[name] = {}

    # Timing: from generation or stepcache_flux summary
    for name, t in gen_times.items():
        if t is not None:
            summary[name]["mean_time"] = t

    # Pull timing from existing summaries for ref configs
    stepcache_path = f"{STEPCACHE_FLUX_DIR}/summary.json"
    eval_path = "outputs/eval/summary.json"

    # stepcache_flux has steady-state-aware times; prefer those
    if os.path.exists(stepcache_path):
        with open(stepcache_path) as f:
            sc = json.load(f)
        for cfg in COMPARE_CONFIGS:
            name = cfg["name"]
            ref_key = cfg.get("ref_time_key", "")
            sc_key = ref_key.replace("stepcache_flux/", "") if ref_key.startswith("stepcache_flux/") else None
            if sc_key and sc_key in sc and name not in gen_times:
                if "mean_time" not in summary.get(name, {}):
                    t = sc[sc_key].get("mean_time")
                    if t:
                        summary.setdefault(name, {})["mean_time"] = t

    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ev = json.load(f)
        for cfg in COMPARE_CONFIGS:
            name = cfg["name"]
            ref_key = cfg.get("ref_time_key", "")
            ev_key = ref_key.replace("stepcache_flux/", "flux/") if ref_key.startswith("stepcache_flux/") else ref_key
            ev_key2 = ref_key.replace("stepcache_flux/", "") if ref_key.startswith("stepcache_flux/") else ref_key
            for k in [ref_key, ev_key, ev_key2]:
                if k in ev and name not in gen_times:
                    if "mean_time" not in summary.get(name, {}):
                        t = ev[k].get("mean_time")
                        if t:
                            summary.setdefault(name, {})["mean_time"] = t
                        break

    # NOTE: for accelae, use the steady-state value from stepcache_flux (7.31s)
    if "accelae" in names_to_score and "mean_time" not in summary.get("accelae", {}):
        if os.path.exists(stepcache_path):
            sc = json.load(open(stepcache_path))
            if "accelae_fskip2" in sc:
                summary.setdefault("accelae", {})["mean_time"] = sc["accelae_fskip2"]["mean_time"]

    # CLIP + Edge
    print("\n[1/7] CLIP + Edge...")
    for name in names_to_score:
        if "clip" not in summary[name]:
            cs = compute_clip_score(all_imgs[name], all_plist[name])
            es = [compute_edge_density(img) for img in all_imgs[name]]
            summary[name]["clip"]  = float(np.mean(cs))
            summary[name]["edge"]  = float(np.mean(es))
            print(f"  {name}: CLIP={summary[name]['clip']:.4f}  Edge={summary[name]['edge']:.4f}")
        else:
            print(f"  {name}: CLIP={summary[name]['clip']:.4f} (cached)")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # PickScore
    print("\n[2/7] PickScore...")
    needs_pick = [n for n in names_to_score if "pick" not in summary[n]]
    if needs_pick:
        ev = PickScoreEvaluator(device)
        for name in needs_pick:
            s = ev.score(all_imgs[name], all_plist[name])
            summary[name]["pick"] = float(np.mean(s))
            print(f"  {name}: {summary[name]['pick']:.4f}")
        del ev; gc.collect(); torch.cuda.empty_cache()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ImageReward
    print("\n[3/7] ImageReward...")
    needs_ir = [n for n in names_to_score if "ir" not in summary[n]]
    if needs_ir:
        ev = ImageRewardEvaluator(device)
        for name in needs_ir:
            s = ev.score(all_imgs[name], all_plist[name])
            summary[name]["ir"] = float(np.mean(s))
            print(f"  {name}: {summary[name]['ir']:.4f}")
        del ev; gc.collect(); torch.cuda.empty_cache()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # HPSv2
    print("\n[4/7] HPSv2...")
    needs_hps = [n for n in names_to_score if "hps" not in summary[n]]
    if needs_hps:
        ev = HPSv2Evaluator(device)
        for name in needs_hps:
            s = ev.score(all_imgs[name], all_plist[name])
            summary[name]["hps"] = float(np.mean(s))
            print(f"  {name}: {summary[name]['hps']:.4f}")
        del ev; gc.collect(); torch.cuda.empty_cache()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Aesthetic
    print("\n[5/7] Aesthetic Score...")
    needs_aes = [n for n in names_to_score if "aesthetic" not in summary[n]]
    if needs_aes:
        ev = AestheticScoreEvaluator(device)
        for name in needs_aes:
            s = ev.score(all_imgs[name])
            summary[name]["aesthetic"] = float(np.mean(s))
            print(f"  {name}: {summary[name]['aesthetic']:.4f}")
        del ev; gc.collect(); torch.cuda.empty_cache()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # LPIPS (vs baseline)
    print("\n[6/7] LPIPS (vs baseline)...")
    needs_lpips = [n for n in names_to_score if "lpips" not in summary[n]]
    if needs_lpips:
        ev = LPIPSEvaluator(device)
        for name in needs_lpips:
            if name == "baseline":
                summary[name]["lpips"] = 0.0
            else:
                s = ev.score(all_imgs[name], all_imgs["baseline"])
                summary[name]["lpips"] = float(np.mean(s))
            print(f"  {name}: {summary[name]['lpips']:.4f}")
        del ev; gc.collect(); torch.cuda.empty_cache()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # FID (vs baseline)
    print("\n[7/7] FID (vs baseline)...")
    baseline_cfg = COMPARE_CONFIGS[0]
    baseline_dir = baseline_cfg.get("ref_dir") or os.path.join(OUTPUT_DIR, "baseline")
    for cfg in COMPARE_CONFIGS:
        name = cfg["name"]
        if name not in all_imgs:
            continue
        if "fid" not in summary[name]:
            if name == "baseline":
                summary[name]["fid"] = 0.0
                print(f"  {name}: 0.00 (reference)")
            else:
                acc_dir = cfg.get("ref_dir") or os.path.join(OUTPUT_DIR, name)
                val = compute_fid(baseline_dir, acc_dir)
                summary[name]["fid"] = float(val)
                print(f"  {name}: {val:.2f}")
        else:
            print(f"  {name}: {summary[name]['fid']:.2f} (cached)")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print table ────────────────────────────────────────────────────────────
    baseline_time = summary.get("baseline", {}).get("mean_time")

    print("\n" + "=" * 130)
    print(f"{'Config':<30} {'Time':>7} {'Speedup':>8}  "
          f"{'CLIP':>7}  {'Edge':>7}  {'Pick':>7}  {'IR':>7}  "
          f"{'HPS':>7}  {'Aesth':>8}  {'LPIPS':>7}  {'FID':>8}")
    print("=" * 130)

    for cfg in COMPARE_CONFIGS:
        name = cfg["name"]
        if name not in summary:
            continue
        r = summary[name]
        t   = r.get("mean_time")
        spd = f"{baseline_time/t:.3f}×" if (t and baseline_time) else "  N/A "
        ts  = f"{t:.2f}s" if t else "   N/A"
        print(
            f"  {cfg['desc']:<28} {ts:>7} {spd:>8}  "
            f"{r.get('clip',0):7.4f}  {r.get('edge',0):7.4f}  "
            f"{r.get('pick',0):7.4f}  {r.get('ir',0):7.4f}  "
            f"{r.get('hps',0):7.4f}  {r.get('aesthetic',0):8.4f}  "
            f"{r.get('lpips',0):7.4f}  {r.get('fid',0):8.2f}"
        )
    print("=" * 130)
    print(f"\nSaved → {summary_path}")
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
