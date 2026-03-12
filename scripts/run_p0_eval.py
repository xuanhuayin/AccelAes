#!/usr/bin/env python3
"""
P0 Evaluation: Baseline Comparison + Component Ablation on Lumina-Next-T2I.

MODE 1 -- baseline_compare:
  Compares SASD against training-free baselines:
    baseline / delta_dit / fora / sasd_full (semantic+slic+fskip2)

MODE 2 -- ablation:
  Component ablation — each sub-component turned on/off:
    baseline / fskip_only / attn_only / attn_ffn / sasd_full

Both modes: 20 prompts × 3 seeds = 60 images/config.
All 8 metrics: CLIP, PickScore, ImageReward, HPSv2, Aesthetic, LPIPS, Edge, FID.

Usage:
  python scripts/run_p0_eval.py --mode baseline_compare
  python scripts/run_p0_eval.py --mode ablation
  python scripts/run_p0_eval.py --mode all
  python scripts/run_p0_eval.py --mode all --skip_gen  # metrics only
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina

from src.models.dit_wrapper import LuminaDiTWrapper
from src.baselines.delta_dit import generate_delta_dit
from src.baselines.fora import generate_fora
from src.baselines.ras import generate_ras
from src.eval.metrics import compute_clip_score, compute_edge_density
from src.sparse.skip_cache import SkipUpdateCache

OUTPUT_BASE_COMPARE = "outputs/p0_compare"
OUTPUT_BASE_ABLATION = "outputs/p0_ablation"


# ── fskip-only generation (step-level cache, dense spatial) ──────────────────

@torch.no_grad()
def generate_fskip_only(
    wrapper,
    prompt: str,
    seed: int,
    full_skip_interval: int = 2,
    warmup_steps: int = 2,
    cfg_scale: float = 4.0,
    steps: int = 30,
    height: int = 1024,
    width: int = 1024,
) -> Image.Image:
    """
    Step-level caching ONLY — no spatial sparsity, no spatial CFG.

    Every `full_skip_interval`-th step (after warmup), the entire transformer
    forward is skipped and noise_pred is linearly extrapolated from the two
    most recent cached predictions.
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    # ---- 1. Encode prompt ----
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=True,
        negative_prompt="",
        num_images_per_prompt=1,
        device=device,
    )
    prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat(
        [prompt_attention_mask, negative_prompt_attention_mask], dim=0
    )

    # ---- 2. Cross-attention kwargs ----
    default_sample_size = getattr(pipe, "default_sample_size", 128)
    cross_attention_kwargs = {
        "base_sequence_length": (default_sample_size // 2) ** 2
    }
    scaling_factor = math.sqrt(width * height / wrapper.default_image_size ** 2)

    # ---- 3. Prepare latents ----
    latent_h = height // wrapper.vae_scale_factor
    latent_w = width // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=dtype,
    )

    # ---- 4. Scheduler ----
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    head_dim = wrapper.transformer.head_dim
    scaling_watershed = 1.0

    # ---- 5. Step-level cache ----
    cache = SkipUpdateCache(method="linear")
    compute_count = 0  # steps where transformer actually ran

    # ---- 6. Denoising loop ----
    for i, t in enumerate(timesteps):
        # Decide whether to skip this step
        is_skip = (
            full_skip_interval > 0
            and cache.has_cache()
            and i >= warmup_steps
            and (i - warmup_steps) % full_skip_interval == 0
        )

        if is_skip:
            noise_pred = cache.get_prediction(i)
        else:
            latent_model_input = torch.cat([latents] * 2, dim=0)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                current_timestep = torch.tensor(
                    [current_timestep], dtype=torch.float64, device=device
                )
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(device)
            current_timestep = current_timestep.expand(latent_model_input.shape[0])
            current_timestep = (
                1 - current_timestep / pipe.scheduler.config.num_train_timesteps
            )

            if current_timestep[0] < scaling_watershed:
                linear_factor, ntk_factor = scaling_factor, 1.0
            else:
                linear_factor, ntk_factor = 1.0, scaling_factor

            image_rotary_emb = get_2d_rotary_pos_embed_lumina(
                head_dim, 384, 384,
                linear_factor=linear_factor,
                ntk_factor=ntk_factor,
            )

            raw = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            raw = raw.chunk(2, dim=1)[0]
            eps, rest = raw[:, :3], raw[:, 3:]
            eps_cond, eps_uncond = torch.split(eps, len(eps) // 2, dim=0)
            guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            eps = torch.cat([guided, guided], dim=0)
            raw = torch.cat([eps, rest], dim=1)
            noise_pred, _ = raw.chunk(2, dim=0)
            noise_pred = -noise_pred

            cache.store(i, noise_pred)
            compute_count += 1

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # ---- 7. Decode ----
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ── Experiment definitions ────────────────────────────────────────────────────

SASD_FULL_KWARGS = dict(
    mask_type="semantic",
    region_method="threshold",
    skip_ratio=0.50,
    s_fg=7.0, s_bg=1.0,
    mask_step=5,
    full_skip_interval=2,
    sparse_ffn=True,
    sparse_blocks=True,
)

EXPERIMENTS_COMPARE = [
    {
        "name": "baseline",
        "desc": "Baseline (dense, CFG=4.0)",
        "fn": "baseline",
        "kwargs": {},
    },
    {
        "name": "delta_dit",
        "desc": "Delta-DiT (3-tier layer cache, warmup=2)",
        "fn": "delta_dit",
        "kwargs": dict(warmup_steps=2, cfg_scale=4.0),
    },
    {
        "name": "fora",
        "desc": "FORA (block residual reuse, interval=2, warmup=2)",
        "fn": "fora",
        "kwargs": dict(reuse_interval=2, warmup_steps=2, cfg_scale=4.0),
    },
    {
        "name": "ras",
        "desc": "RAS (cross-attn magnitude mask, target_ratio=0.5, warmup=2)",
        "fn": "ras",
        "kwargs": dict(target_ratio=0.5, warmup_steps=2, cfg_scale=4.0),
    },
    {
        "name": "sasd_cfg_fskip",
        "desc": "SASD (spatial CFG + fskip2, no sparse attn/FFN)",
        "fn": "sasd",
        "kwargs": dict(
            mask_type="semantic", region_method="threshold",
            skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
            full_skip_interval=2, sparse_ffn=False, sparse_blocks=False,
        ),
    },
    {
        "name": "sasd_full",
        "desc": "SASD (semantic+slic+fskip2, s_fg=7, s_bg=1)",
        "fn": "sasd",
        "kwargs": SASD_FULL_KWARGS,
    },
]

EXPERIMENTS_ABLATION = [
    {
        "name": "baseline",
        "desc": "Baseline (dense, CFG=4.0)",
        "fn": "baseline",
        "kwargs": {},
    },
    {
        "name": "fskip_only",
        "desc": "Step cache only (fskip2, no spatial)",
        "fn": "fskip_only",
        "kwargs": dict(full_skip_interval=2, warmup_steps=2, cfg_scale=4.0),
    },
    {
        "name": "attn_only",
        "desc": "Sparse attn + spatial CFG only (no FFN, no fskip)",
        "fn": "sasd",
        "kwargs": dict(
            mask_type="semantic", region_method="threshold",
            skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
            full_skip_interval=0, sparse_ffn=False, sparse_blocks=True,
        ),
    },
    {
        "name": "attn_ffn",
        "desc": "Sparse attn + sparse FFN + spatial CFG (no fskip)",
        "fn": "sasd",
        "kwargs": dict(
            mask_type="semantic", region_method="threshold",
            skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
            full_skip_interval=0, sparse_ffn=True, sparse_blocks=True,
        ),
    },
    {
        "name": "sasd_full",
        "desc": "SASD full (attn+ffn+spatial_cfg+fskip2)",
        "fn": "sasd",
        "kwargs": SASD_FULL_KWARGS,
    },
]


# ── Metric evaluators ─────────────────────────────────────────────────────────

class PickScoreEvaluator:
    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, AutoModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
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
    def score(self, images, prompts, img_dir):
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
            "ViT-L-14", pretrained="openai", device=device
        )
        self.model.eval()
        cache = os.path.join(os.path.dirname(__file__), "..", ".cache")
        os.makedirs(cache, exist_ok=True)
        wp = os.path.join(cache, "aesthetic_mlp_l14.pth")
        if not os.path.exists(wp):
            url = (
                "https://github.com/christophschuhmann/"
                "improved-aesthetic-predictor/raw/main/"
                "sac+logos+ava1-l14-linearMSE.pth"
            )
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


# ── Generation helper ─────────────────────────────────────────────────────────

def generate_one(wrapper, exp, prompt, seed):
    fn = exp["fn"]
    kw = exp["kwargs"]
    if fn == "baseline":
        return wrapper.generate(prompt=prompt, seed=seed, cfg_scale=4.0)
    elif fn == "delta_dit":
        return generate_delta_dit(wrapper, prompt=prompt, seed=seed, **kw)
    elif fn == "fora":
        return generate_fora(wrapper, prompt=prompt, seed=seed, **kw)
    elif fn == "ras":
        return generate_ras(wrapper, prompt=prompt, seed=seed, **kw)
    elif fn == "fskip_only":
        return generate_fskip_only(wrapper, prompt=prompt, seed=seed, **kw)
    elif fn == "sasd":
        return wrapper.generate_accelerated_dual(prompt=prompt, seed=seed, **kw)
    else:
        raise ValueError(f"Unknown fn: {fn}")


def run_generation(wrapper, prompts, seeds, experiments, output_base):
    """Generate images for all experiments; return {name: (img_dir, mean_time)}."""
    results = {}
    total_per_config = len(prompts) * len(seeds)

    for exp in experiments:
        name = exp["name"]
        img_dir = os.path.join(output_base, name, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Force defragment GPU memory between configs
        if hasattr(wrapper, 'transformer'):
            wrapper.transformer.to("cpu")
            wrapper._transformer_on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(wrapper, 'transformer'):
            wrapper.transformer.to("cuda")
            wrapper._transformer_on_gpu = True

        print(f"\n=== [{name}] {exp['desc']} ({total_per_config} images) ===")
        times, count, skipped = [], 0, 0

        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                out_path = os.path.join(img_dir, f"p{pi:04d}_s{seed:04d}.png")
                if os.path.exists(out_path):
                    skipped += 1
                    count += 1
                    if count % 10 == 0 or count == total_per_config:
                        print(f"  [{count}/{total_per_config}] skipped")
                    continue

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                img = generate_one(wrapper, exp, prompt, seed)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                times.append(elapsed)
                img.save(out_path)
                count += 1

                if count % 10 == 0 or count == total_per_config:
                    avg = sum(times) / len(times)
                    print(f"  [{count}/{total_per_config}] avg {avg:.2f}s/img")

                del img
                gc.collect()
                torch.cuda.empty_cache()

        if skipped == total_per_config:
            print(f"  All {total_per_config} images already exist.")
            results[name] = (img_dir, 0.0)
        else:
            mean_t = sum(times) / len(times) if times else 0.0
            print(f"  Done: {mean_t:.2f}s/img ({skipped} skipped)")
            results[name] = (img_dir, mean_t)

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_all_metrics(experiments, output_base, prompts, seeds):
    total = len(prompts) * len(seeds)
    baseline_name = experiments[0]["name"]

    # ---- Load images ----
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

    # ---- Load evaluators ----
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

    # ---- Compute per-experiment metrics ----
    all_metrics = {}
    for exp in experiments:
        name = exp["name"]
        images = all_imgs[name]
        plist  = all_plist[name]
        img_dir = os.path.join(output_base, name, "images")
        m = {}
        print(f"\n--- Metrics: {name} ---")

        print("  CLIP...", end="", flush=True)
        m["clip"] = compute_clip_score(images, plist)
        print(f" {np.mean(m['clip']):.4f}")

        print("  Edge...", end="", flush=True)
        m["edge"] = [compute_edge_density(img) for img in images]
        print(f" {np.mean(m['edge']):.4f}")

        for label, key in [
            ("PickScore",   "pick"),
            ("ImageReward", "ir"),
            ("Aesthetic",   "aesthetic"),
        ]:
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
    gc.collect()
    torch.cuda.empty_cache()

    # ---- FID ----
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


# ── Print & save results ──────────────────────────────────────────────────────

def print_and_save(experiments, output_base, gen_results, all_metrics, fid_scores):
    ref_t = gen_results[experiments[0]["name"]][1]
    have_times = ref_t > 0 and all(gen_results[e["name"]][1] > 0 for e in experiments)

    print("\n" + "=" * 130)
    print(
        f"{'Config':<42} {'Time':>7} {'Speed':>6}  "
        "CLIP    Edge    Pick      IR     HPS    Aesth   LPIPS    FID"
    )
    print("=" * 130)

    summary = {}
    for exp in experiments:
        name = exp["name"]
        _, t = gen_results[name]
        m = all_metrics[name]

        if have_times and t > 0:
            spd = f"{ref_t / t:5.2f}x"
            t_str = f"{t:6.2f}s"
        else:
            spd = "  N/A "
            t_str = "   N/A "

        def v(key):
            return f"{np.mean(m[key]):7.4f}" if key in m else "    N/A"

        fid = f"{fid_scores[name]:7.2f}" if name in fid_scores else "    N/A"

        print(
            f"  {exp['desc']:<40} {t_str} {spd}  "
            f"{v('clip')} {v('edge')} {v('pick')} {v('ir')} "
            f"{v('hps')} {v('aesthetic')} {v('lpips')} {fid}"
        )

        entry = {
            "desc": exp["desc"],
            "mean_time": t,
            "speedup": (ref_t / t) if (have_times and t > 0) else None,
        }
        for key in ["clip", "edge", "pick", "ir", "hps", "aesthetic", "lpips"]:
            if key in m:
                entry[f"mean_{key}"] = float(np.mean(m[key]))
                entry[f"std_{key}"]  = float(np.std(m[key]))
        if name in fid_scores:
            entry["fid"] = fid_scores[name]
        elif name == experiments[0]["name"]:
            entry["fid"] = 0.0
        summary[name] = entry

    print("=" * 130)

    out_json = os.path.join(output_base, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_json}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline_compare", "ablation", "all"],
                        default="all")
    parser.add_argument("--prompts",     default="prompts/prompts_dev.txt")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--seeds",       nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip_gen",    action="store_true",
                        help="skip generation, only run metrics on existing images")
    parser.add_argument("--only_config", default=None,
                        help="run generation for a single named config only (gen phase), "
                             "then exit. Use this to run each config in a fresh process "
                             "to avoid GPU memory fragmentation.")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = [l.strip() for l in f if l.strip()][: args.num_prompts]
    seeds = args.seeds
    print(f"Prompts: {len(prompts)}, Seeds: {seeds}, "
          f"Total/config: {len(prompts) * len(seeds)}")

    run_compare  = args.mode in ("baseline_compare", "all")
    run_ablation = args.mode in ("ablation", "all")

    # ── Phase 1: Generation ────────────────────────────────────────────────
    gen_results_compare  = {}
    gen_results_ablation = {}

    if not args.skip_gen:
        # --only_config: single-config subprocess mode (avoids memory fragmentation)
        if args.only_config is not None:
            name = args.only_config
            # Search the list(s) relevant to the current mode
            # (compare list first only when mode allows compare, to avoid
            #  routing ablation-only configs to p0_compare/)
            exp_single = None
            output_base_single = None
            if run_compare:
                for exp in EXPERIMENTS_COMPARE:
                    if exp["name"] == name:
                        exp_single = exp
                        output_base_single = OUTPUT_BASE_COMPARE
                        break
            if exp_single is None and run_ablation:
                for exp in EXPERIMENTS_ABLATION:
                    if exp["name"] == name:
                        exp_single = exp
                        output_base_single = OUTPUT_BASE_ABLATION
                        break
            if exp_single is None and not run_compare:
                # fallback: search compare too (in case mode mismatch)
                for exp in EXPERIMENTS_COMPARE:
                    if exp["name"] == name:
                        exp_single = exp
                        output_base_single = OUTPUT_BASE_COMPARE
                        break
            if exp_single is None:
                print(f"ERROR: config '{name}' not found in EXPERIMENTS_COMPARE or EXPERIMENTS_ABLATION")
                import sys; sys.exit(1)
            wrapper = LuminaDiTWrapper(dtype="bf16")
            run_generation(wrapper, prompts, seeds, [exp_single], output_base_single)
            del wrapper
            gc.collect()
            torch.cuda.empty_cache()
            print(f"=== Single-config '{name}' done. Exiting. ===")
            import sys; sys.exit(0)

        wrapper = LuminaDiTWrapper(dtype="bf16")

        if run_compare:
            print("\n\n" + "#" * 60)
            print("# BASELINE COMPARISON")
            print("#" * 60)
            gen_results_compare = run_generation(
                wrapper, prompts, seeds, EXPERIMENTS_COMPARE, OUTPUT_BASE_COMPARE
            )

        if run_ablation:
            print("\n\n" + "#" * 60)
            print("# COMPONENT ABLATION")
            print("#" * 60)
            gen_results_ablation = run_generation(
                wrapper, prompts, seeds, EXPERIMENTS_ABLATION, OUTPUT_BASE_ABLATION
            )

        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # Recover placeholder times (0.0 = N/A)
        for exp in EXPERIMENTS_COMPARE:
            gen_results_compare[exp["name"]] = (
                os.path.join(OUTPUT_BASE_COMPARE, exp["name"], "images"), 0.0
            )
        for exp in EXPERIMENTS_ABLATION:
            gen_results_ablation[exp["name"]] = (
                os.path.join(OUTPUT_BASE_ABLATION, exp["name"], "images"), 0.0
            )

    # ── Phase 2: Metrics ───────────────────────────────────────────────────
    if run_compare:
        print("\n\n" + "#" * 60)
        print("# METRICS: BASELINE COMPARISON")
        print("#" * 60)
        m_compare, fid_compare = compute_all_metrics(
            EXPERIMENTS_COMPARE, OUTPUT_BASE_COMPARE, prompts, seeds
        )
        print_and_save(
            EXPERIMENTS_COMPARE, OUTPUT_BASE_COMPARE,
            gen_results_compare, m_compare, fid_compare,
        )

    if run_ablation:
        print("\n\n" + "#" * 60)
        print("# METRICS: COMPONENT ABLATION")
        print("#" * 60)
        m_ablation, fid_ablation = compute_all_metrics(
            EXPERIMENTS_ABLATION, OUTPUT_BASE_ABLATION, prompts, seeds
        )
        print_and_save(
            EXPERIMENTS_ABLATION, OUTPUT_BASE_ABLATION,
            gen_results_ablation, m_ablation, fid_ablation,
        )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
