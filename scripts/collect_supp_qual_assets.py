#!/usr/bin/env python3
"""
Collect and generate all qualitative comparison images for supplementary.

Unified seed=0 across all methods and backbones.
Selected prompts: indices 3, 6, 7, 14 from prompts_dev.txt.

Output structure:
  outputs/supp_qual_assets/
    prompts.txt
    lumina/
      baseline/       p03.png p06.png p07.png p14.png
      AccelAes/
      SDiT/
      RAS/
      DeltaDiT/
      FORA/
      TeaCache/
      TaylorSeer/
    flux/
      baseline/
      AccelAes/
      TeaCache_t015/
      TaylorSeer_r1/
      TaylorSeer_r2/
    sd3/
      baseline/
      AccelAes_fskip2/
      TeaCache_t030/
      TaylorSeer_r1/
      DeltaDiT/
"""

import os, sys, gc, shutil
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

PROMPT_FILE = "prompts/prompts_dev.txt"
PROMPT_IDS  = [3, 6, 7, 14]
SEED        = 0
OUT_BASE    = "outputs/supp_qual_assets"

with open(PROMPT_FILE) as f:
    ALL_PROMPTS = [l.strip() for l in f if l.strip()]

SELECTED = [(i, ALL_PROMPTS[i]) for i in PROMPT_IDS]

# ── Utilities ─────────────────────────────────────────────────────────────────

def out_dir(backbone, method):
    d = os.path.join(OUT_BASE, backbone, method)
    os.makedirs(d, exist_ok=True)
    return d

def out_path(backbone, method, pi):
    return os.path.join(out_dir(backbone, method), f"p{pi:02d}.png")

def exists(backbone, method):
    return all(os.path.exists(out_path(backbone, method, pi)) for pi, _ in SELECTED)

def copy_from(backbone, method, src_dir, seed=0):
    """Copy images from src_dir/{p{pi:04d}_s{seed:04d}.png} → out_path."""
    d = out_dir(backbone, method)
    ok = True
    for pi, _ in SELECTED:
        src = os.path.join(src_dir, f"p{pi:04d}_s{seed:04d}.png")
        dst = out_path(backbone, method, pi)
        if os.path.exists(dst):
            continue
        if not os.path.exists(src):
            print(f"  [MISSING] {src}")
            ok = False
            continue
        shutil.copy2(src, dst)
        print(f"  copy {os.path.basename(src)} → {os.path.relpath(dst)}")
    return ok

def save_img(img, backbone, method, pi):
    dst = out_path(backbone, method, pi)
    if not os.path.exists(dst):
        img.save(dst)
        print(f"  saved → {os.path.relpath(dst)}")

# ── Write prompts.txt ─────────────────────────────────────────────────────────

def write_prompts():
    os.makedirs(OUT_BASE, exist_ok=True)
    with open(os.path.join(OUT_BASE, "prompts.txt"), "w") as f:
        f.write("Selected prompts for supplementary qualitative figures\n")
        f.write("Seed = 0 for all methods and backbones\n")
        f.write("=" * 60 + "\n\n")
        for pi, prompt in SELECTED:
            f.write(f"p{pi:02d} (index {pi:04d}):\n{prompt}\n\n")
    print(f"Wrote prompts.txt")


# ── LUMINA ────────────────────────────────────────────────────────────────────

def collect_lumina():
    print("\n" + "=" * 50)
    print("LUMINA – copying existing seed=0 images")
    print("=" * 50)

    copy_from("lumina", "baseline",   "outputs/p5eval-baseline/images",               seed=0)
    copy_from("lumina", "AccelAes",   "outputs/p0_ablation_direct/sasd_full/images",   seed=0)
    copy_from("lumina", "SDiT",       "outputs/sdit_eval/images",                      seed=0)

    # These need generation
    need_gen = []
    for m in ["RAS", "DeltaDiT", "FORA", "TeaCache", "TaylorSeer"]:
        if not exists("lumina", m):
            need_gen.append(m)

    if not need_gen:
        print("  All Lumina methods already present.")
        return

    print(f"\nGenerating Lumina methods with seed={SEED}: {need_gen}")
    from src.models.dit_wrapper import LuminaDiTWrapper
    wrapper = LuminaDiTWrapper(dtype="bf16")

    for m in need_gen:
        print(f"\n  -- {m} --")
        for pi, prompt in SELECTED:
            dst = out_path("lumina", m, pi)
            if os.path.exists(dst):
                continue
            img = _generate_lumina(wrapper, m, prompt, SEED)
            if img is not None:
                save_img(img, "lumina", m, pi)
                del img; gc.collect(); torch.cuda.empty_cache()

    del wrapper; gc.collect(); torch.cuda.empty_cache()


def _generate_lumina(wrapper, method, prompt, seed):
    from src.baselines.ras       import generate_ras
    from src.baselines.teacache  import generate_teacache_lumina, generate_taylor_lumina
    from src.baselines.delta_dit import generate_delta_dit
    from src.baselines.fora      import generate_fora

    if method == "RAS":
        return generate_ras(wrapper, prompt, seed,
                            target_ratio=0.5, warmup_steps=2, cfg_scale=4.0, steps=30)
    elif method == "DeltaDiT":
        return generate_delta_dit(wrapper, prompt, seed,
                                  warmup_steps=2, cfg_scale=4.0, steps=30)
    elif method == "FORA":
        return generate_fora(wrapper, prompt, seed,
                             reuse_interval=2, warmup_steps=2, cfg_scale=4.0, steps=30)
    elif method == "TeaCache":
        return generate_teacache_lumina(wrapper, prompt, seed,
                                        threshold=0.15, warmup_steps=3,
                                        cfg_scale=4.0, steps=30)
    elif method == "TaylorSeer":
        return generate_taylor_lumina(wrapper, prompt, seed,
                                      taylor_run=2, warmup_steps=5, taylor_order=2,
                                      cfg_scale=4.0, steps=30)
    else:
        print(f"  Unknown method: {method}")
        return None


# ── FLUX ──────────────────────────────────────────────────────────────────────

def collect_flux():
    print("\n" + "=" * 50)
    print("FLUX – all methods use seed=42, copying")
    print("=" * 50)

    FLUX_SEED = 42
    flux_base = "outputs/stepcache_flux"
    copy_from("flux", "baseline",       f"{flux_base}/baseline/images",     seed=FLUX_SEED)
    copy_from("flux", "AccelAes",       f"{flux_base}/accelae_fskip2/images",seed=FLUX_SEED)
    copy_from("flux", "TeaCache_t015",  f"{flux_base}/teacache_t015/images", seed=FLUX_SEED)
    copy_from("flux", "TaylorSeer_r1",  f"{flux_base}/taylor_r1_o2/images",  seed=FLUX_SEED)
    copy_from("flux", "TaylorSeer_r2",  f"{flux_base}/taylor_r2_o2/images",  seed=FLUX_SEED)


# ── SD3 ───────────────────────────────────────────────────────────────────────

def collect_sd3():
    print("\n" + "=" * 50)
    print("SD3 – all methods use seed=0, copying")
    print("=" * 50)

    sd3_base = "outputs/sd3_compare"
    copy_from("sd3", "baseline",        f"{sd3_base}/baseline_gen",    seed=0)
    copy_from("sd3", "AccelAes_fskip2", f"{sd3_base}/fskip2_gen",      seed=0)
    copy_from("sd3", "TeaCache_t030",   f"{sd3_base}/teacache_t030",   seed=0)
    copy_from("sd3", "TaylorSeer_r1",   f"{sd3_base}/taylor_r1",       seed=0)
    copy_from("sd3", "DeltaDiT",        f"{sd3_base}/delta_dit",       seed=0)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for backbone in ["lumina", "flux", "sd3"]:
        bdir = os.path.join(OUT_BASE, backbone)
        if not os.path.isdir(bdir):
            continue
        print(f"\n{backbone.upper()}:")
        for method in sorted(os.listdir(bdir)):
            mdir = os.path.join(bdir, method)
            if not os.path.isdir(mdir):
                continue
            imgs = [f for f in os.listdir(mdir) if f.endswith(".png")]
            status = "✅" if len(imgs) == len(PROMPT_IDS) else f"⚠ {len(imgs)}/{len(PROMPT_IDS)}"
            print(f"  {method:<20} {status}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["lumina", "flux", "sd3", "all"],
                        default="all")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip generation, only copy existing images")
    args = parser.parse_args()

    write_prompts()

    if args.backbone in ("lumina", "all"):
        if args.skip_gen:
            # Only copy existing ones
            print("\nLUMINA – copy only (no generation)")
            copy_from("lumina", "baseline",   "outputs/p5eval-baseline/images", seed=0)
            copy_from("lumina", "AccelAes",   "outputs/p0_ablation_direct/sasd_full/images", seed=0)
            copy_from("lumina", "SDiT",       "outputs/sdit_eval/images", seed=0)
        else:
            collect_lumina()

    if args.backbone in ("flux", "all"):
        collect_flux()

    if args.backbone in ("sd3", "all"):
        collect_sd3()

    print_summary()
    print(f"\nAll assets in: {OUT_BASE}/")


if __name__ == "__main__":
    main()
