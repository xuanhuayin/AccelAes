#!/usr/bin/env python3
"""
Use FLUX.1-dev to generate AccelAes logo variations.
FLUX has strong text rendering capability.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.flux_wrapper import FluxDiTWrapper as FluxWrapper

OUT_DIR = "outputs/figures/logo_variants"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = [
    # v1: minimalist typography
    ('v1_typo',
     'A clean minimalist logo with the bold text "AccelAes" in dark charcoal color, '
     'the letters "Aes" in forest green, white background, modern sans-serif font, '
     'professional academic design, no decorations, sharp edges'),

    # v2: glowing neon sci-fi
    ('v2_neon',
     'Glowing neon sign reading "AccelAes", dark background, bright green neon letters, '
     'electric glow effect, futuristic, cyberpunk style, high contrast, photorealistic'),

    # v3: elegant embossed
    ('v3_emboss',
     'Elegant embossed text logo "AccelAes" on a dark green velvet background, '
     'gold metallic lettering, luxury design, high detail, studio lighting'),

    # v4: geometric abstract
    ('v4_geo',
     'Abstract geometric logo design featuring the letters A and E intertwined, '
     'green and dark grey color palette, modern, minimalist, vector art style, '
     'clean lines, white background, text "AccelAes" below'),

    # v5: paper/academic badge
    ('v5_badge',
     'Academic conference paper badge with text "AccelAes" in the center, '
     'circular design, green and white color scheme, clean professional typography, '
     'subtle circuit pattern background, ECCV 2026 at the bottom'),
]

def main():
    print("Loading FLUX...")
    wrapper = FluxWrapper()

    for name, prompt in PROMPTS:
        for seed in [42, 123]:
            out = f"{OUT_DIR}/{name}_s{seed}.png"
            if os.path.exists(out):
                print(f"  skip: {out}"); continue
            print(f"\n  [{name} seed={seed}]")
            print(f"  prompt: {prompt[:80]}...")
            img = wrapper.generate(prompt=prompt, seed=seed,
                                   guidance_scale=3.5, steps=28,
                                   height=768, width=1024)
            img.save(out)
            print(f"  -> {out}")

    print("\nDone. Variants in:", OUT_DIR)

if __name__ == "__main__":
    main()
