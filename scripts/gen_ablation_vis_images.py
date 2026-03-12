#!/usr/bin/env python3
"""
Generate high-quality visualization images for ablation figures.

Produces:
  outputs/p0_ablation_direct/{config}/vis_images/p{i:04d}_s0000.png  (C.1, 5 configs)
  outputs/anchor_ablation/{config}/vis_images/p{i:04d}_s0000.png     (C.2, 3 configs)
  outputs/supp_qual_assets/lumina/{method}/vis/p{i:02d}.png           (G.1, 8 methods)

5 aesthetic prompts (same set as sensitivity vis_images).
"""
import os, sys, gc
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

PROMPTS = [
    "A silver wolf howling under moonlight, intricate fur and sharp claws, "
    "dramatic sky background, photorealistic, cinematic, masterpiece",

    "A peacock displaying its intricate iridescent plumage, garden setting, "
    "soft bokeh, photorealistic, stunning, dramatic lighting",

    "A geisha in intricate traditional kimono with detailed embroidery, "
    "cherry blossom background, photorealistic, cinematic, portrait",

    "A phoenix rising from flames with intricate golden feather details, "
    "dramatic lighting, fantasy art, highly detailed, vivid",

    "Grand baroque palace interior with intricate golden decorations, "
    "volumetric lighting, photorealistic, masterpiece, highly detailed",
]
SEED = 0

SASD_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
    cfg_scale=4.0, steps=30,
)


def gen_all(wrapper, out_dir, gen_fn, label):
    os.makedirs(out_dir, exist_ok=True)
    for pi, prompt in enumerate(PROMPTS):
        dst = os.path.join(out_dir, f"p{pi:04d}_s0000.png")
        if os.path.exists(dst):
            print(f"  skip [{label}] p{pi:04d}")
            continue
        print(f"  [{label}] p{pi:04d}: {prompt[:55]}...")
        img = gen_fn(wrapper, prompt)
        img.save(dst)
        del img; gc.collect(); torch.cuda.empty_cache()
    print(f"  ✓ {label}")


# ── C.1 SkipSparse ablation ───────────────────────────────────────────────

def run_c1_ablation():
    from src.models.dit_wrapper import LuminaDiTWrapper
    print("\n" + "="*55)
    print("C.1  SkipSparse Ablation (5 configs)")
    print("="*55)
    wrapper = LuminaDiTWrapper(dtype="bf16")

    BASE = "outputs/p0_ablation_direct"

    # baseline — reuse from sensitivity (same prompt set)
    # (actually generate fresh for correctness)
    gen_all(wrapper,
            os.path.join(BASE, "baseline", "vis_images"),
            lambda w, p: w.generate(prompt=p, seed=SEED,
                                    steps=30, cfg_scale=4.0),
            "baseline")

    # fskip_only: step-skip only, no spatial sparse
    gen_all(wrapper,
            os.path.join(BASE, "fskip_only", "vis_images"),
            lambda w, p: w.generate_accelerated_dual(
                prompt=p, seed=SEED,
                mask_type="semantic", region_method="threshold",
                skip_ratio=0.50, s_fg=4.0, s_bg=4.0, mask_step=5,
                full_skip_interval=2, sparse_ffn=False, sparse_blocks=False,
                cfg_scale=4.0, steps=30),
            "fskip_only")

    # attn_only: spatial sparse attn, no FFN, no fskip
    gen_all(wrapper,
            os.path.join(BASE, "attn_only", "vis_images"),
            lambda w, p: w.generate_accelerated_dual(
                prompt=p, seed=SEED,
                mask_type="semantic", region_method="threshold",
                skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
                full_skip_interval=0, sparse_ffn=False, sparse_blocks=True,
                cfg_scale=4.0, steps=30),
            "attn_only")

    # attn_ffn: spatial sparse attn+FFN, no fskip
    gen_all(wrapper,
            os.path.join(BASE, "attn_ffn", "vis_images"),
            lambda w, p: w.generate_accelerated_dual(
                prompt=p, seed=SEED,
                mask_type="semantic", region_method="threshold",
                skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
                full_skip_interval=0, sparse_ffn=True, sparse_blocks=True,
                cfg_scale=4.0, steps=30),
            "attn_ffn")

    # sasd_full: full AccelAes — reuse from sensitivity vis_images (maskstep_05)
    src_dir = "outputs/supp_sensitivity/maskstep_05/vis_images"
    dst_dir = os.path.join(BASE, "sasd_full", "vis_images")
    os.makedirs(dst_dir, exist_ok=True)
    import shutil
    for pi in range(len(PROMPTS)):
        src = os.path.join(src_dir, f"p{pi:04d}_s0000.png")
        dst = os.path.join(dst_dir, f"p{pi:04d}_s0000.png")
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
    print("  ✓ sasd_full (reused from sensitivity vis_images)")

    del wrapper; gc.collect(); torch.cuda.empty_cache()


# ── C.2 AesMask Token Selection Ablation ─────────────────────────────────

def run_c2_ablation():
    import src.sparse.mask_builders as _mb
    from src.models.dit_wrapper import LuminaDiTWrapper
    print("\n" + "="*55)
    print("C.2  AesMask Token Selection Ablation (3 configs)")
    print("="*55)
    wrapper = LuminaDiTWrapper(dtype="bf16")

    BASE = "outputs/anchor_ablation"

    # sasd_with_anchor — reuse sasd_full vis_images
    import shutil
    src_dir = "outputs/supp_sensitivity/maskstep_05/vis_images"
    dst_dir = os.path.join(BASE, "sasd_with_anchor", "vis_images")
    os.makedirs(dst_dir, exist_ok=True)
    for pi in range(len(PROMPTS)):
        src = os.path.join(src_dir, f"p{pi:04d}_s0000.png")
        dst = os.path.join(dst_dir, f"p{pi:04d}_s0000.png")
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
    print("  ✓ sasd_with_anchor (reused)")

    # sasd_no_anchor — uniform token weights (None → equal weight)
    _orig = _mb.SemanticMaskBuilder.compute_token_importance
    _mb.SemanticMaskBuilder.compute_token_importance = lambda self, **kw: None
    try:
        if hasattr(wrapper, '_token_importance_cache'):
            wrapper._token_importance_cache.clear()
        gen_all(wrapper,
                os.path.join(BASE, "sasd_no_anchor", "vis_images"),
                lambda w, p: w.generate_accelerated_dual(
                    prompt=p, seed=SEED, **SASD_KWARGS),
                "sasd_no_anchor")
    finally:
        _mb.SemanticMaskBuilder.compute_token_importance = _orig
        if hasattr(wrapper, '_token_importance_cache'):
            wrapper._token_importance_cache.clear()

    # sasd_nonaesthetic — reversed importance (non-aesthetic weighting)
    def _reverse_importance(self, **kw):
        result = _orig(self, **kw)
        if result is not None:
            import torch
            result = 1.0 - result / (result.max() + 1e-8)
        return result

    _mb.SemanticMaskBuilder.compute_token_importance = _reverse_importance
    try:
        if hasattr(wrapper, '_token_importance_cache'):
            wrapper._token_importance_cache.clear()
        gen_all(wrapper,
                os.path.join(BASE, "sasd_nonaesthetic", "vis_images"),
                lambda w, p: w.generate_accelerated_dual(
                    prompt=p, seed=SEED, **SASD_KWARGS),
                "sasd_nonaesthetic")
    finally:
        _mb.SemanticMaskBuilder.compute_token_importance = _orig
        if hasattr(wrapper, '_token_importance_cache'):
            wrapper._token_importance_cache.clear()

    del wrapper; gc.collect(); torch.cuda.empty_cache()


# ── supp_qual_assets / Lumina ─────────────────────────────────────────────

def run_qual_lumina():
    from src.models.dit_wrapper import LuminaDiTWrapper
    from src.baselines.ras       import generate_ras
    from src.baselines.teacache  import generate_teacache_lumina, generate_taylor_lumina
    from src.baselines.delta_dit import generate_delta_dit
    from src.baselines.fora      import generate_fora
    import shutil

    print("\n" + "="*55)
    print("supp_qual_assets / Lumina (8 methods)")
    print("="*55)

    OUT = "outputs/supp_qual_assets/lumina"

    def out_dir(method):
        d = os.path.join(OUT, method)
        os.makedirs(d, exist_ok=True)
        return d

    # baseline — reuse from c1 vis_images
    src_dir = "outputs/p0_ablation_direct/baseline/vis_images"
    for pi in range(len(PROMPTS)):
        src = os.path.join(src_dir, f"p{pi:04d}_s0000.png")
        dst = os.path.join(out_dir("baseline"), f"p{pi:02d}.png")
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
    print("  ✓ baseline (reused)")

    # AccelAes — reuse sasd_full vis_images
    src_dir = "outputs/supp_sensitivity/maskstep_05/vis_images"
    for pi in range(len(PROMPTS)):
        src = os.path.join(src_dir, f"p{pi:04d}_s0000.png")
        dst = os.path.join(out_dir("AccelAes"), f"p{pi:02d}.png")
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
    print("  ✓ AccelAes (reused)")

    wrapper = LuminaDiTWrapper(dtype="bf16")

    # SDiT — SLIC-based method
    for pi, prompt in enumerate(PROMPTS):
        dst = os.path.join(out_dir("SDiT"), f"p{pi:02d}.png")
        if os.path.exists(dst):
            continue
        print(f"  [SDiT] p{pi:02d}: {prompt[:55]}...")
        img = wrapper.generate_accelerated_dual(
            prompt=prompt, seed=SEED,
            mask_type="semantic", region_method="slic",
            skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
            full_skip_interval=0, sparse_ffn=False, sparse_blocks=True,
            cfg_scale=4.0, steps=30)
        img.save(dst); del img; gc.collect(); torch.cuda.empty_cache()
    print("  ✓ SDiT")

    # RAS
    gen_all_baseline(wrapper, out_dir("RAS"),
        lambda w, p: generate_ras(w, p, SEED,
            target_ratio=0.5, warmup_steps=2, cfg_scale=4.0, steps=30))

    # DeltaDiT
    gen_all_baseline(wrapper, out_dir("DeltaDiT"),
        lambda w, p: generate_delta_dit(w, p, SEED,
            warmup_steps=2, cfg_scale=4.0, steps=30))

    # FORA
    gen_all_baseline(wrapper, out_dir("FORA"),
        lambda w, p: generate_fora(w, p, SEED,
            reuse_interval=2, warmup_steps=2, cfg_scale=4.0, steps=30))

    # TeaCache
    gen_all_baseline(wrapper, out_dir("TeaCache"),
        lambda w, p: generate_teacache_lumina(w, p, SEED,
            threshold=0.15, warmup_steps=3, cfg_scale=4.0, steps=30))

    # TaylorSeer
    gen_all_baseline(wrapper, out_dir("TaylorSeer"),
        lambda w, p: generate_taylor_lumina(w, p, SEED,
            taylor_run=2, warmup_steps=5, taylor_order=2, cfg_scale=4.0, steps=30))

    del wrapper; gc.collect(); torch.cuda.empty_cache()


def _rename_vis(d):
    """Rename p{i:04d}_s0000.png → p{i:02d}.png"""
    import glob
    for src in sorted(glob.glob(os.path.join(d, "p????_s????.png"))):
        pi = int(os.path.basename(src).split("_")[0][1:])
        dst = os.path.join(d, f"p{pi:02d}.png")
        if not os.path.exists(dst):
            os.rename(src, dst)


def gen_all_baseline(wrapper, out_dir, gen_fn):
    os.makedirs(out_dir, exist_ok=True)
    method = os.path.basename(out_dir)
    for pi, prompt in enumerate(PROMPTS):
        dst = os.path.join(out_dir, f"p{pi:02d}.png")
        if os.path.exists(dst):
            print(f"  skip [{method}] p{pi:02d}")
            continue
        print(f"  [{method}] p{pi:02d}: {prompt[:55]}...")
        img = gen_fn(wrapper, prompt)
        if img is not None:
            img.save(dst)
            del img; gc.collect(); torch.cuda.empty_cache()
    print(f"  ✓ {method}")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["c1", "c2", "lumina", "all"], default="all")
    args = parser.parse_args()

    if args.part in ("c1", "all"):
        run_c1_ablation()
    if args.part in ("c2", "all"):
        run_c2_ablation()
    if args.part in ("lumina", "all"):
        run_qual_lumina()

    print("\n✅ All done.")
