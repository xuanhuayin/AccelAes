#!/usr/bin/env python3
"""
Generate 20 aesthetic-anchor-rich prompts × 3 seeds for all 8 methods in teaser_fig.

New images: p0020_s0000 … p0039_s0002  (indices 20–39, seeds 0/1/2)
Methods: baseline / delta_dit / fora / ras / sasd_cfg_fskip / sasd_full /
         teacache_t015 / taylor_r2
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.baselines.delta_dit  import generate_delta_dit
from src.baselines.fora        import generate_fora
from src.baselines.ras         import generate_ras
from src.baselines.teacache    import generate_teacache_lumina, generate_taylor_lumina

OUT_BASE    = "outputs/teaser_fig"
PROMPT_FILE = "prompts/prompts_aesthetic.txt"
SEEDS       = [0, 1, 2]
P_OFFSET    = 20          # filenames start at p0020

prompts = open(PROMPT_FILE).read().splitlines()
assert len(prompts) == 20, f"Expected 20 prompts, got {len(prompts)}"

SASD_FULL_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
)
SASD_CFG_FSKIP_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
    full_skip_interval=2, sparse_ffn=False, sparse_blocks=False,
)

def make_methods(w):
    return [
        ("baseline",
         lambda w, p, s: w.generate(prompt=p, seed=s, cfg_scale=4.0)),
        ("delta_dit",
         lambda w, p, s: generate_delta_dit(w, prompt=p, seed=s,
                                             warmup_steps=2, cfg_scale=4.0)),
        ("fora",
         lambda w, p, s: generate_fora(w, prompt=p, seed=s,
                                        reuse_interval=2, warmup_steps=2, cfg_scale=4.0)),
        ("ras",
         lambda w, p, s: generate_ras(w, prompt=p, seed=s,
                                       target_ratio=0.5, warmup_steps=2, cfg_scale=4.0)),
        ("sasd_cfg_fskip",
         lambda w, p, s: w.generate_accelerated_dual(prompt=p, seed=s, cfg_scale=4.0,
                                                      **SASD_CFG_FSKIP_KWARGS)),
        ("sasd_full",
         lambda w, p, s: w.generate_accelerated_dual(prompt=p, seed=s, cfg_scale=4.0,
                                                      **SASD_FULL_KWARGS)),
        ("teacache_t015",
         lambda w, p, s: generate_teacache_lumina(w, prompt=p, seed=s,
                                                   cfg_scale=4.0, threshold=0.15)),
        ("taylor_r2",
         lambda w, p, s: generate_taylor_lumina(w, prompt=p, seed=s,
                                                 cfg_scale=4.0, taylor_run=2)),
    ]

print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()
methods = make_methods(wrapper)
total   = len(prompts) * len(SEEDS)

for method_name, fn in methods:
    img_dir = os.path.join(OUT_BASE, method_name, "images")
    os.makedirs(img_dir, exist_ok=True)
    done = skipped = 0
    print(f"\n=== {method_name} ===", flush=True)

    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(SEEDS):
            fname = f"p{P_OFFSET + pi:04d}_s{si:04d}.png"
            out   = os.path.join(img_dir, fname)
            if os.path.exists(out):
                skipped += 1
                continue
            img = fn(wrapper, prompt, seed)
            img.save(out)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total}", flush=True)

    print(f"  Done: {done} generated, {skipped} skipped")

print("\nAll done.")
