#!/usr/bin/env python3
"""
Supplementary sensitivity experiments for AccelAes.

Experiment 1: skip_ratio (percentile p) sensitivity
  - skip_ratio ∈ {0.30, 0.40, 0.50, 0.60, 0.70}
  - All other kwargs fixed at AccelAes defaults

Experiment 2: mask_step sensitivity
  - mask_step ∈ {3, 5, 7, 10, 15}
  - All other kwargs fixed at AccelAes defaults

Baseline reused from outputs/p0_ablation_direct/baseline/

Output: outputs/supp_sensitivity/
  summary_ratio.json   — skip_ratio sweep
  summary_maskstep.json — mask_step sweep
"""

import argparse, gc, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.eval.metrics import compute_clip_score, compute_edge_density

OUTPUT_BASE  = "outputs/supp_sensitivity"
BASELINE_DIR = "outputs/p5eval-baseline/images"

# ── AccelAes defaults ──────────────────────────────────────────────────────────
BASE_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    s_fg=7.0, s_bg=1.0,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
    cfg_scale=4.0,
)

# ── Experiment grids ──────────────────────────────────────────────────────────
RATIO_CONFIGS = [
    {"name": "ratio_030", "desc": "skip_ratio=0.30 (p=30%)",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.30, "mask_step": 5}},
    {"name": "ratio_040", "desc": "skip_ratio=0.40 (p=40%)",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.40, "mask_step": 5}},
    {"name": "ratio_050", "desc": "skip_ratio=0.50 (p=50%) [default]",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 5}},
    {"name": "ratio_060", "desc": "skip_ratio=0.60 (p=60%)",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.60, "mask_step": 5}},
    {"name": "ratio_070", "desc": "skip_ratio=0.70 (p=70%)",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.70, "mask_step": 5}},
]

MASKSTEP_CONFIGS = [
    {"name": "maskstep_03", "desc": "mask_step=3",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 3}},
    {"name": "maskstep_05", "desc": "mask_step=5 [default]",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 5}},
    {"name": "maskstep_07", "desc": "mask_step=7",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 7}},
    {"name": "maskstep_10", "desc": "mask_step=10",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 10}},
    {"name": "maskstep_15", "desc": "mask_step=15",
     "kwargs": {**BASE_KWARGS, "skip_ratio": 0.50, "mask_step": 15}},
]


# ── Metric evaluators ─────────────────────────────────────────────────────────

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
        tf = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
        return [
            self.fn(tf(a).unsqueeze(0).to(self.device),
                    tf(b).unsqueeze(0).to(self.device)).item()
            for a, b in zip(images, refs)
        ]


# ── Generation ────────────────────────────────────────────────────────────────

def generate_config(wrapper, config, prompts, seeds):
    img_dir = os.path.join(OUTPUT_BASE, config["name"], "images")
    os.makedirs(img_dir, exist_ok=True)
    total = len(prompts) * len(seeds)
    times, skipped = [], 0

    print(f"\n=== {config['desc']} ===")
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            out_path = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
            if os.path.exists(out_path):
                skipped += 1; continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = wrapper.generate_accelerated_dual(prompt=prompt, seed=seed,
                                                    **config["kwargs"])
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            img.save(out_path)
            del img
            gc.collect(); torch.cuda.empty_cache()

            done = len(times) + skipped
            if done % 15 == 0 or done == total:
                avg = sum(times)/len(times) if times else 0
                print(f"  [{done}/{total}] avg {avg:.2f}s/img")

    mean_t = sum(times)/len(times) if times else None
    if mean_t:
        print(f"  Done: {mean_t:.2f}s/img  ({skipped} skipped)")
    else:
        print(f"  All {skipped} images already exist, skipping generation")
    return mean_t


# ── Scoring ───────────────────────────────────────────────────────────────────

def load_images(config_name, prompts, seeds):
    img_dir = os.path.join(OUTPUT_BASE, config_name, "images")
    imgs, plist = [], []
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            p = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
            imgs.append(Image.open(p).convert("RGB"))
            plist.append(prompt)
    return imgs, plist

def load_baseline_images(prompts, seeds):
    imgs = []
    for pi in range(len(prompts)):
        for seed in seeds:
            p = os.path.join(BASELINE_DIR, f"p{pi:04d}_s{seed:04d}.png")
            imgs.append(Image.open(p).convert("RGB"))
    return imgs

def score_config(config_name, prompts, seeds, baseline_imgs, evaluators):
    imgs, plist = load_images(config_name, prompts, seeds)
    img_dir = os.path.join(OUTPUT_BASE, config_name, "images")
    m = {}
    print(f"  Scoring {config_name}...")
    m["clip"] = float(np.mean(compute_clip_score(imgs, plist)))
    m["edge"] = float(np.mean([compute_edge_density(img) for img in imgs]))
    m["ir"]   = float(np.mean(evaluators["ir"].score(imgs, plist)))
    m["hps"]  = float(np.mean(evaluators["hps"].score(imgs, plist, img_dir)))
    m["aesthetic"] = float(np.mean(evaluators["aes"].score(imgs)))
    m["lpips"] = float(np.mean(evaluators["lpips"].score(imgs, baseline_imgs)))
    print(f"    IR={m['ir']:.4f}  CLIP={m['clip']:.4f}  HPS={m['hps']:.4f}"
          f"  Aesth={m['aesthetic']:.4f}  LPIPS={m['lpips']:.4f}")
    return m

def compute_fid(config_name):
    try:
        from cleanfid import fid as cleanfid
        bl_dir = BASELINE_DIR
        d = os.path.join(OUTPUT_BASE, config_name, "images")
        v = cleanfid.compute_fid(bl_dir, d, mode="clean", num_workers=0)
        print(f"    FID={v:.2f}")
        return v
    except Exception as e:
        print(f"    FID: SKIP ({e})")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sweep(wrapper, configs, prompts, seeds, sweep_name):
    print(f"\n{'='*60}")
    print(f"SWEEP: {sweep_name}  ({len(configs)} configs × {len(prompts)*len(seeds)} imgs)")
    print(f"{'='*60}")

    # 1. Generate
    gen_times = {}
    for cfg in configs:
        t = generate_config(wrapper, cfg, prompts, seeds)
        gen_times[cfg["name"]] = t

    # 2. Load baseline
    print("\nLoading baseline images...")
    baseline_imgs = load_baseline_images(prompts, seeds)

    # 3. Load evaluators
    print("Loading evaluators...")
    evaluators = {
        "ir":   ImageRewardEvaluator(),
        "hps":  HPSv2Evaluator(),
        "aes":  AestheticScoreEvaluator(),
        "lpips": LPIPSEvaluator(),
    }

    # 4. Score
    summary = {}
    for cfg in configs:
        m = score_config(cfg["name"], prompts, seeds, baseline_imgs, evaluators)
        fid = compute_fid(cfg["name"])
        entry = {
            "desc":       cfg["desc"],
            "mean_time":  gen_times.get(cfg["name"]),
            **m,
        }
        if fid is not None:
            entry["fid"] = fid
        summary[cfg["name"]] = entry

    del evaluators
    gc.collect(); torch.cuda.empty_cache()

    # 5. Save
    out_path = os.path.join(OUTPUT_BASE, f"summary_{sweep_name}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")

    # 6. Print table
    print(f"\n--- {sweep_name} results ---")
    print(f"{'Config':<30} {'Time':>7}  IR      CLIP    HPS     Aesth   LPIPS   FID")
    print("-" * 95)
    for cfg in configs:
        e = summary[cfg["name"]]
        t_str = f"{e['mean_time']:.2f}s" if e.get("mean_time") else "  reuse"
        fid_str = f"{e['fid']:.1f}" if "fid" in e else "  N/A"
        print(f"  {cfg['desc']:<28} {t_str}  "
              f"{e['ir']:.4f}  {e['clip']:.4f}  {e['hps']:.4f}  "
              f"{e['aesthetic']:.4f}  {e['lpips']:.4f}  {fid_str}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts",     default="prompts/prompts_dev.txt")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--seeds",       nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip_gen",    action="store_true")
    parser.add_argument("--sweep",       choices=["ratio", "maskstep", "all"], default="all")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = [l.strip() for l in f if l.strip()][:args.num_prompts]
    seeds = args.seeds
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print(f"Prompts: {len(prompts)}, Seeds: {seeds}, "
          f"Total/config: {len(prompts)*len(seeds)}")
    print(f"Baseline dir: {BASELINE_DIR}")

    # Verify baseline exists
    n_baseline = len([f for f in os.listdir(BASELINE_DIR) if f.endswith(".png")])
    print(f"Baseline images found: {n_baseline}")

    if not args.skip_gen:
        wrapper = LuminaDiTWrapper(dtype="bf16")
    else:
        wrapper = None
        print("Skipping generation (--skip_gen)")

    if args.sweep in ("ratio", "all"):
        run_sweep(wrapper, RATIO_CONFIGS, prompts, seeds, "ratio")

    if args.sweep in ("maskstep", "all"):
        run_sweep(wrapper, MASKSTEP_CONFIGS, prompts, seeds, "maskstep")

    if wrapper is not None:
        del wrapper
        gc.collect(); torch.cuda.empty_cache()

    print("\nAll sweeps done.")


if __name__ == "__main__":
    main()
