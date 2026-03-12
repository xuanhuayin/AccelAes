#!/usr/bin/env python3
"""
Generate epic 3D stone letter logo "AccelAes" with all three backbones.
Style: dramatic stone/rock 3D letters on cliffs, cinematic sky.

Output: outputs/figures/logo_epic/
  flux_s{seed}.png
  sd3_s{seed}.png
  lumina_s{seed}.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "outputs/figures/logo_epic"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]

# Carefully crafted prompt based on the reference image style
PROMPT = (
    'Epic 3D stone letters spelling "AccelAes" standing tall on dramatic rocky cliff edges, '
    'giant monumental typography carved from ancient weathered rock and stone, '
    'upper letters with rusty iron texture, lower letters with dark granite texture, '
    'dramatic cinematic sky with swirling clouds and glowing stars, '
    'volumetric god rays, misty atmosphere, deep shadow and bright highlights, '
    'photorealistic, 8k, ultra detailed, cinematic composition, '
    'epic fantasy landscape, fog rolling through the cliffs'
)

print(f"Prompt:\n{PROMPT}\n")


def run_flux():
    from src.models.flux_wrapper import FluxDiTWrapper
    print("=== FLUX.1-dev ===")
    w = FluxDiTWrapper()
    for seed in SEEDS:
        out = f"{OUT_DIR}/flux_s{seed}.png"
        if os.path.exists(out):
            print(f"  skip: {out}"); continue
        print(f"  seed={seed} ...", end=" ", flush=True)
        img = w.generate(prompt=PROMPT, seed=seed,
                         guidance_scale=3.5, steps=28,
                         height=1024, width=1024)
        img.save(out)
        print(f"-> {out}")
    del w
    import torch; torch.cuda.empty_cache()


def run_sd3():
    from src.models.sd3_wrapper import SD3Wrapper
    print("=== SD3-Medium ===")
    w = SD3Wrapper()
    for seed in SEEDS:
        out = f"{OUT_DIR}/sd3_s{seed}.png"
        if os.path.exists(out):
            print(f"  skip: {out}"); continue
        print(f"  seed={seed} ...", end=" ", flush=True)
        img = w.generate(prompt=PROMPT, seed=seed,
                         cfg_scale=7.0, steps=28,
                         height=1024, width=1024)
        img.save(out)
        print(f"-> {out}")
    del w
    import torch; torch.cuda.empty_cache()


def run_lumina():
    from src.models.dit_wrapper import LuminaDiTWrapper
    print("=== Lumina-Next-T2I ===")
    w = LuminaDiTWrapper()
    for seed in SEEDS:
        out = f"{OUT_DIR}/lumina_s{seed}.png"
        if os.path.exists(out):
            print(f"  skip: {out}"); continue
        print(f"  seed={seed} ...", end=" ", flush=True)
        img = w.generate(prompt=PROMPT, seed=seed,
                         cfg_scale=4.0, steps=30,
                         height=1024, width=1024)
        img.save(out)
        print(f"-> {out}")
    del w
    import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["flux", "sd3", "lumina"],
                        default=["flux", "sd3", "lumina"])
    args = parser.parse_args()

    if "flux"   in args.models: run_flux()
    if "sd3"    in args.models: run_sd3()
    if "lumina" in args.models: run_lumina()

    print(f"\nDone. Saved to {OUT_DIR}/")
