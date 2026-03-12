#!/usr/bin/env python3
"""
Replace 6 distorted prompts with cleaner landscape/animal/architecture prompts.

Replacements:
  Set D idx 2 (steampunk→crystal cave)  — maskstep vis_images
  Set D idx 4 (ballet→torii gates)      — maskstep vis_images
  Set E idx 0 (floral portrait→reef)    — ratio vis_images
  Set E idx 4 (astronaut→snow leopard)  — ratio vis_images
  Set C idx 2 (mermaid→autumn forest)   — SD3 supp_qual_assets
  Set C idx 3 (orchids→jade mountain)   — SD3 supp_qual_assets
"""
import os, sys, gc
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── New replacement prompts ───────────────────────────────────────────────────
# No human figures / hybrid creatures — landscape, architecture, animals only

REPL_D2 = (
    "A crystal cave interior with luminescent stalactites and glowing mineral pools, "
    "ethereal bioluminescent atmosphere, photorealistic, stunning, highly detailed, "
    "volumetric lighting, masterpiece"
)
REPL_D4 = (
    "Ancient red torii gates pathway winding through a misty Japanese forest, "
    "soft golden morning light filtering through cedar trees, fallen leaves, "
    "photorealistic, cinematic, masterpiece, highly detailed"
)
REPL_E0 = (
    "Vibrant tropical coral reef teeming with colorful exotic fish, crystal clear "
    "turquoise water, sunlight shafts piercing the surface, photorealistic underwater "
    "photography, stunning, highly detailed, dramatic lighting"
)
REPL_E4 = (
    "A majestic snow leopard resting on a rocky mountain ledge, dramatic winter "
    "Himalayan landscape, soft bokeh background, photorealistic, cinematic, stunning, "
    "dramatic lighting, highly detailed fur"
)
REPL_C2 = (
    "A misty autumn forest with golden and crimson maple leaves, ancient moss-covered "
    "stone lanterns along a winding path, soft volumetric light, photorealistic, "
    "masterpiece, cinematic, highly detailed"
)
REPL_C3 = (
    "A jade mountain peak dramatically emerging from a sea of clouds, ancient Chinese "
    "ink painting meets photorealism, dramatic lighting, cinematic, masterpiece, "
    "highly detailed, stunning atmosphere"
)

SEED = 0

# ── Part 1: Lumina sensitivity vis_images ─────────────────────────────────────

MASKSTEP_CONFIGS = [
    ("maskstep_03", dict(mask_type="semantic", region_method="threshold",
                         skip_ratio=0.50, mask_step=3, s_fg=7.0, s_bg=1.0,
                         full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                         cfg_scale=4.0, steps=30)),
    ("maskstep_05", dict(mask_type="semantic", region_method="threshold",
                         skip_ratio=0.50, mask_step=5, s_fg=7.0, s_bg=1.0,
                         full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                         cfg_scale=4.0, steps=30)),
    ("maskstep_07", dict(mask_type="semantic", region_method="threshold",
                         skip_ratio=0.50, mask_step=7, s_fg=7.0, s_bg=1.0,
                         full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                         cfg_scale=4.0, steps=30)),
    ("maskstep_10", dict(mask_type="semantic", region_method="threshold",
                         skip_ratio=0.50, mask_step=10, s_fg=7.0, s_bg=1.0,
                         full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                         cfg_scale=4.0, steps=30)),
    ("maskstep_15", dict(mask_type="semantic", region_method="threshold",
                         skip_ratio=0.50, mask_step=15, s_fg=7.0, s_bg=1.0,
                         full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                         cfg_scale=4.0, steps=30)),
]

RATIO_CONFIGS = [
    ("ratio_030", dict(mask_type="semantic", region_method="threshold",
                       skip_ratio=0.30, mask_step=5, s_fg=7.0, s_bg=1.0,
                       full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                       cfg_scale=4.0, steps=30)),
    ("ratio_040", dict(mask_type="semantic", region_method="threshold",
                       skip_ratio=0.40, mask_step=5, s_fg=7.0, s_bg=1.0,
                       full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                       cfg_scale=4.0, steps=30)),
    ("ratio_050", dict(mask_type="semantic", region_method="threshold",
                       skip_ratio=0.50, mask_step=5, s_fg=7.0, s_bg=1.0,
                       full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                       cfg_scale=4.0, steps=30)),
    ("ratio_060", dict(mask_type="semantic", region_method="threshold",
                       skip_ratio=0.60, mask_step=5, s_fg=7.0, s_bg=1.0,
                       full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                       cfg_scale=4.0, steps=30)),
    ("ratio_070", dict(mask_type="semantic", region_method="threshold",
                       skip_ratio=0.70, mask_step=5, s_fg=7.0, s_bg=1.0,
                       full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
                       cfg_scale=4.0, steps=30)),
]


def regen_lumina():
    from src.models.dit_wrapper import LuminaDiTWrapper
    print("\n" + "="*55)
    print("Lumina: regenerate Set D idx 2,4 and Set E idx 0,4")
    print("="*55)
    wrapper = LuminaDiTWrapper(dtype="bf16")

    BASE = "outputs/supp_sensitivity"

    # Set D: maskstep configs, indices 2 and 4
    for cfg_name, kwargs in MASKSTEP_CONFIGS:
        vis_dir = os.path.join(BASE, cfg_name, "vis_images")
        for pi, prompt in [(2, REPL_D2), (4, REPL_D4)]:
            dst = os.path.join(vis_dir, f"p{pi:04d}_s0000.png")
            # delete old
            if os.path.exists(dst):
                os.remove(dst)
            print(f"  [{cfg_name}] p{pi:04d}: {prompt[:60]}...")
            img = wrapper.generate_accelerated_dual(prompt=prompt, seed=SEED, **kwargs)
            img.save(dst)
            del img; gc.collect(); torch.cuda.empty_cache()

    # Set E: ratio configs, indices 0 and 4
    for cfg_name, kwargs in RATIO_CONFIGS:
        vis_dir = os.path.join(BASE, cfg_name, "vis_images")
        for pi, prompt in [(0, REPL_E0), (4, REPL_E4)]:
            dst = os.path.join(vis_dir, f"p{pi:04d}_s0000.png")
            if os.path.exists(dst):
                os.remove(dst)
            print(f"  [{cfg_name}] p{pi:04d}: {prompt[:60]}...")
            img = wrapper.generate_accelerated_dual(prompt=prompt, seed=SEED, **kwargs)
            img.save(dst)
            del img; gc.collect(); torch.cuda.empty_cache()

    del wrapper; gc.collect(); torch.cuda.empty_cache()
    print("  ✓ Lumina done")


# ── Part 2: SD3 supp_qual_assets ─────────────────────────────────────────────

def regen_sd3():
    from src.models.sd3_wrapper import SD3Wrapper
    from src.baselines.teacache import generate_teacache_sd3, generate_taylor_sd3
    from src.baselines.delta_dit import generate_delta_dit_sd3
    from src.baselines.fora import generate_fora_sd3

    print("\n" + "="*55)
    print("SD3: regenerate Set C idx 2 (mermaid) and 3 (orchids)")
    print("="*55)

    BASE = "outputs/supp_qual_assets/sd3"

    wrapper = SD3Wrapper(dtype="bf16")

    METHODS = {
        "baseline": lambda w, p: w.generate(
            prompt=p, seed=SEED, steps=28, cfg_scale=7.0),
        "AccelAes": lambda w, p: w.generate_accelerated(
            prompt=p, seed=SEED, full_skip_interval=2,
            cfg_scale=7.0, steps=28),
        "TeaCache": lambda w, p: generate_teacache_sd3(
            w, p, SEED, threshold=0.15, warmup_steps=3,
            cfg_scale=7.0, steps=28),
        "TaylorSeer": lambda w, p: generate_taylor_sd3(
            w, p, SEED, taylor_run=2, warmup_steps=5,
            taylor_order=2, cfg_scale=7.0, steps=28),
        "DeltaDiT": lambda w, p: generate_delta_dit_sd3(
            w, p, SEED, warmup_steps=2, cfg_scale=7.0, steps=28),
    }

    for method, gen_fn in METHODS.items():
        method_dir = os.path.join(BASE, method)
        os.makedirs(method_dir, exist_ok=True)
        for pi, prompt in [(2, REPL_C2), (3, REPL_C3)]:
            dst = os.path.join(method_dir, f"p{pi:02d}.png")
            if os.path.exists(dst):
                os.remove(dst)
            print(f"  [{method}] p{pi:02d}: {prompt[:60]}...")
            img = gen_fn(wrapper, prompt)
            if img is not None:
                img.save(dst)
            del img; gc.collect(); torch.cuda.empty_cache()
        print(f"  ✓ {method}")

    del wrapper; gc.collect(); torch.cuda.empty_cache()
    print("  ✓ SD3 done")


# ── Part 3: Rebuild PDFs ─────────────────────────────────────────────────────

def rebuild_pdfs():
    print("\n" + "="*55)
    print("Rebuilding sensitivity vis PDFs")
    print("="*55)
    import subprocess
    subprocess.run(["python3", "scripts/gen_sensitivity_vis.py"], check=True)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["lumina", "sd3", "pdf", "all"], default="all")
    args = parser.parse_args()

    if args.part in ("lumina", "all"):
        regen_lumina()
    if args.part in ("sd3", "all"):
        regen_sd3()
    if args.part in ("pdf", "all"):
        rebuild_pdfs()

    print("\n✅ Done.")
