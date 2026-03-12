#!/usr/bin/env python3
"""
Ablation: CLIP aesthetic anchor weighting vs uniform token weights.

Configs (20 prompts × seeds [0,1,2] = 60 images each):
  sasd_with_anchor  — default: compute_token_importance returns CLIP-weighted scores
  sasd_no_anchor    — uniform: compute_token_importance patched to return None

Same SASD kwargs as p0_ablation_direct/sasd_full.
Output: outputs/anchor_ablation/
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
import src.sparse.mask_builders as _mb

PROMPTS_FILE = "prompts/prompts_dev.txt"
N_PROMPTS    = 20
SEEDS        = [0, 1, 2]
OUT_BASE     = "outputs/anchor_ablation"

SASD_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
)

prompts = open(PROMPTS_FILE).read().splitlines()[:N_PROMPTS]
print(f"Prompts: {len(prompts)}  Seeds: {SEEDS}  Total/config: {len(prompts)*len(SEEDS)}")

# ── Monkey-patch for no-anchor run ────────────────────────────────────────
_orig_compute = _mb.SemanticMaskBuilder.compute_token_importance

def _uniform_token_importance(self, **kwargs):
    """Return None → build_mask_from_cross_attention uses uniform weights."""
    return None

# ── Helpers ───────────────────────────────────────────────────────────────
def run_config(wrapper, name, patch_fn=None):
    img_dir = os.path.join(OUT_BASE, name, "images")
    os.makedirs(img_dir, exist_ok=True)
    times, done, skipped = [], 0, 0
    total = len(prompts) * len(SEEDS)

    if patch_fn is not None:
        _mb.SemanticMaskBuilder.compute_token_importance = patch_fn
    else:
        _mb.SemanticMaskBuilder.compute_token_importance = _orig_compute

    # Reset cached builder so the patch takes effect
    wrapper._semantic_builder = None
    wrapper._token_importance_cache.clear()

    print(f"\n=== {name} ===")
    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(SEEDS):
            fname = f"p{pi:04d}_s{si:04d}.png"
            out   = os.path.join(img_dir, fname)
            if os.path.exists(out):
                skipped += 1; continue
            t0  = time.time()
            img = wrapper.generate_accelerated_dual(
                prompt=prompt, seed=seed, cfg_scale=4.0, **SASD_KWARGS)
            elapsed = time.time() - t0
            img.save(out)
            times.append(elapsed)
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{total}]  avg={sum(times)/len(times):.2f}s  {fname}", flush=True)

    # Restore
    _mb.SemanticMaskBuilder.compute_token_importance = _orig_compute

    avg_t = sum(times)/len(times) if times else None
    print(f"  Done: {done} generated, {skipped} skipped" +
          (f"  avg={avg_t:.2f}s  speedup={12.37/avg_t:.2f}x" if avg_t else ""))
    return avg_t

# ── Main ──────────────────────────────────────────────────────────────────
print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()

t_with = run_config(wrapper, "sasd_with_anchor",  patch_fn=None)
t_no   = run_config(wrapper, "sasd_no_anchor",    patch_fn=_uniform_token_importance)

print("\n=== Timing summary ===")
if t_with: print(f"  with_anchor : {t_with:.2f}s  {12.37/t_with:.2f}x")
if t_no:   print(f"  no_anchor   : {t_no:.2f}s  {12.37/t_no:.2f}x")
print("Done. Run compute metrics separately.")
