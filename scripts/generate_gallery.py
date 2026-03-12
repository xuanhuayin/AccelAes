#!/usr/bin/env python3
"""
Generate 100-prompt gallery using all three models with AccelAes configuration.

Output layout:
  outputs/gallery/lumina/  — Lumina-Next AccelAes (semantic+fskip2)
  outputs/gallery/sd3/     — SD3-Medium AccelAes (fskip2 only)
  outputs/gallery/flux/    — FLUX.1-dev AccelAes (TaylorSeer r=2)

100 prompts × 3 models = 300 images total, seed=42 for all.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT_FILE  = "prompts/prompts_dev.txt"
NUM_PROMPTS  = 100
SEED         = 42
OUT_BASE     = "outputs/gallery"

# ── Lumina AccelAes kwargs ─────────────────────────────────────────────────
LUMINA_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    skip_ratio=0.50, s_fg=7.0, s_bg=1.0,
    mask_step=5, full_skip_interval=2,
    sparse_ffn=True, sparse_blocks=True,
)

# ── helpers ────────────────────────────────────────────────────────────────
def load_prompts():
    with open(PROMPT_FILE) as f:
        return [l.strip() for l in f if l.strip()][:NUM_PROMPTS]

def run_lumina(prompts):
    from src.models.dit_wrapper import LuminaDiTWrapper
    print("\n=== Lumina-Next AccelAes ===")
    wrapper = LuminaDiTWrapper()
    img_dir = f"{OUT_BASE}/lumina"
    os.makedirs(img_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        out = f"{img_dir}/p{i:04d}.png"
        if os.path.exists(out):
            print(f"  [{i+1:3d}/100] skip"); continue
        t0 = time.time()
        img = wrapper.generate_accelerated_dual(
            prompt=prompt, seed=SEED, **LUMINA_KWARGS)
        img.save(out)
        print(f"  [{i+1:3d}/100] {time.time()-t0:.1f}s  {prompt[:60]}")

    del wrapper
    import torch; torch.cuda.empty_cache()

def run_sd3(prompts):
    from src.models.sd3_wrapper import SD3Wrapper
    # Best SD3 config: fskip2_only (uniform CFG=7.0, no spatial split)
    print("\n=== SD3-Medium fskip2 (best config) ===")
    wrapper = SD3Wrapper()
    img_dir = f"{OUT_BASE}/sd3"
    os.makedirs(img_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        out = f"{img_dir}/p{i:04d}.png"
        if os.path.exists(out):
            print(f"  [{i+1:3d}/100] skip"); continue
        t0 = time.time()
        img = wrapper.generate_accelerated(
            prompt=prompt, seed=SEED,
            cfg_scale=7.0, s_fg=7.0, s_bg=7.0,   # uniform CFG, no spatial split
            full_skip_interval=2)
        img.save(out)
        print(f"  [{i+1:3d}/100] {time.time()-t0:.1f}s  {prompt[:60]}")

    del wrapper
    import torch; torch.cuda.empty_cache()

def run_flux(prompts):
    from src.models.flux_wrapper import FluxDiTWrapper as FluxWrapper
    # Best FLUX config: TaylorSeer r=2 (2.12×, IR +5.8%)
    print("\n=== FLUX.1-dev TaylorSeer r=2 (best config) ===")
    wrapper = FluxWrapper()
    img_dir = f"{OUT_BASE}/flux"
    os.makedirs(img_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        out = f"{img_dir}/p{i:04d}.png"
        if os.path.exists(out):
            print(f"  [{i+1:3d}/100] skip"); continue
        t0 = time.time()
        img = wrapper.generate_taylor(
            prompt=prompt, seed=SEED, taylor_run=2, taylor_order=2)
        img.save(out)
        print(f"  [{i+1:3d}/100] {time.time()-t0:.1f}s  {prompt[:60]}")

    del wrapper
    import torch; torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["lumina", "sd3", "flux"],
                        default=["lumina", "sd3", "flux"],
                        help="Which models to run (default: all three)")
    args = parser.parse_args()

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts. Models: {args.models}")
    print(f"Output: {OUT_BASE}/")

    t_total = time.time()
    if "lumina" in args.models: run_lumina(prompts)
    if "sd3"    in args.models: run_sd3(prompts)
    if "flux"   in args.models: run_flux(prompts)

    print(f"\nAll done in {(time.time()-t_total)/60:.1f} min.")
    print(f"Images saved to {OUT_BASE}/")
