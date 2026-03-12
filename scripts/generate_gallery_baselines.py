#!/usr/bin/env python3
"""
Generate gallery baselines — same 100 prompts, seed=42, for all competing methods.

Outputs under outputs/gallery/:
  lumina_baseline/    — Lumina dense baseline
  lumina_delta_dit/   — Lumina + Δ-DiT
  lumina_fora/        — Lumina + FORA
  lumina_ras/         — Lumina + RAS
  sd3_baseline/       — SD3 dense baseline
  flux_baseline/      — FLUX dense baseline
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT_FILE = "prompts/prompts_dev.txt"
NUM_PROMPTS = 100
SEED        = 42
OUT_BASE    = "outputs/gallery"

def load_prompts():
    with open(PROMPT_FILE) as f:
        return [l.strip() for l in f if l.strip()][:NUM_PROMPTS]

def gen_loop(name, gen_fn, prompts, desc=""):
    img_dir = f"{OUT_BASE}/{name}"
    os.makedirs(img_dir, exist_ok=True)
    print(f"\n=== {desc or name} ===")
    for i, prompt in enumerate(prompts):
        out = f"{img_dir}/p{i:04d}.png"
        if os.path.exists(out):
            print(f"  [{i+1:3d}/100] skip"); continue
        t0 = time.time()
        img = gen_fn(prompt)
        img.save(out)
        print(f"  [{i+1:3d}/100] {time.time()-t0:.1f}s  {prompt[:60]}")

# ── Lumina baselines ────────────────────────────────────────────────────────

def run_lumina_all(prompts):
    from src.models.dit_wrapper import LuminaDiTWrapper
    from src.baselines.delta_dit import generate_delta_dit
    from src.baselines.fora      import generate_fora
    from src.baselines.ras       import generate_ras

    print("\nLoading Lumina...")
    w = LuminaDiTWrapper()

    gen_loop("lumina_baseline",  lambda p: w.generate(prompt=p, seed=SEED),
             prompts, "Lumina dense baseline")

    gen_loop("lumina_delta_dit", lambda p: generate_delta_dit(w, prompt=p, seed=SEED),
             prompts, "Lumina Δ-DiT")

    gen_loop("lumina_fora",      lambda p: generate_fora(w, prompt=p, seed=SEED),
             prompts, "Lumina FORA")

    gen_loop("lumina_ras",       lambda p: generate_ras(w, prompt=p, seed=SEED),
             prompts, "Lumina RAS")

    del w
    import torch; torch.cuda.empty_cache()

# ── SD3 baseline ────────────────────────────────────────────────────────────

def run_sd3_baseline(prompts):
    from src.models.sd3_wrapper import SD3Wrapper
    print("\nLoading SD3...")
    w = SD3Wrapper()

    gen_loop("sd3_baseline", lambda p: w.generate(prompt=p, seed=SEED),
             prompts, "SD3 dense baseline")

    del w
    import torch; torch.cuda.empty_cache()

# ── FLUX baseline ────────────────────────────────────────────────────────────

def run_flux_baseline(prompts):
    from src.models.flux_wrapper import FluxDiTWrapper
    print("\nLoading FLUX...")
    w = FluxDiTWrapper()

    gen_loop("flux_baseline", lambda p: w.generate(prompt=p, seed=SEED),
             prompts, "FLUX dense baseline")

    del w
    import torch; torch.cuda.empty_cache()

# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["lumina", "sd3", "flux"],
                        default=["lumina", "sd3", "flux"])
    args = parser.parse_args()

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts. Models: {args.models}")

    t0 = time.time()
    if "lumina" in args.models: run_lumina_all(prompts)
    if "sd3"    in args.models: run_sd3_baseline(prompts)
    if "flux"   in args.models: run_flux_baseline(prompts)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min.")
    print(f"Saved to {OUT_BASE}/")
