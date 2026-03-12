#!/usr/bin/env python3
"""
Visualize the effect of CLIP aesthetic anchor weighting on the spatial mask.

For one aesthetic prompt, captures cross-attention at mask_step and computes:
  - heatmap_uniform : avg cross-attn weighted uniformly over all content tokens
  - heatmap_anchor  : avg cross-attn weighted by CLIP-anchor cosine similarity

Outputs (outputs/anchor_viz/):
  generated.png        — generated image
  heatmap_uniform.png  — uniform token-weight heatmap  (jet colormap, 64×64 block style)
  heatmap_anchor.png   — anchor-weighted heatmap
  mask_uniform.png     — binary mask from uniform heatmap
  mask_anchor.png      — binary mask from anchor heatmap
  token_weights.png    — per-token importance bar chart
  comparison.png       — side-by-side 6-panel figure
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
import src.sparse.mask_builders as _mb

OUT_DIR = "outputs/anchor_viz"
os.makedirs(OUT_DIR, exist_ok=True)

# Aesthetic prompt — rich in anchor-matching keywords
PROMPT = (
    "Portrait of a warrior queen, intricate gold armor, sharp focus, "
    "depth of field, stunning, photorealistic, volumetric lighting, highly detailed"
)
SEED   = 0
RATIO  = 0.50   # fg fraction for mask

# ── Capture hooks ─────────────────────────────────────────────────────────────
_captured = {}   # filled by hook

_orig_build = _mb.SemanticMaskBuilder.build_mask_from_cross_attention

def _hook_build(self, cross_attn_maps, token_importance, latent_h, latent_w):
    num_patches = latent_h * latent_w
    all_attn = torch.stack([m.mean(dim=0) for m in cross_attn_maps])
    avg_attn  = all_attn.mean(dim=0)   # (patches, seq_len)

    # ----- uniform heatmap (equal weight over ALL tokens) -----
    unif_w  = torch.ones(avg_attn.shape[1], device=avg_attn.device,
                         dtype=avg_attn.dtype) / avg_attn.shape[1]
    hm_u    = (avg_attn * unif_w.unsqueeze(0)).sum(-1)
    hm_u    = hm_u[:num_patches].reshape(latent_h, latent_w)
    hm_u    = (hm_u - hm_u.min()) / (hm_u.max() - hm_u.min() + 1e-8)

    # ----- anchor-weighted heatmap -----
    ti = token_importance
    if ti is None:
        ti = unif_w
    hm_a = (avg_attn * ti.unsqueeze(0)).sum(-1)
    hm_a = hm_a[:num_patches].reshape(latent_h, latent_w)
    hm_a = (hm_a - hm_a.min()) / (hm_a.max() - hm_a.min() + 1e-8)

    _captured["avg_attn"]          = avg_attn.float().cpu()
    _captured["token_importance"]  = ti.float().cpu()
    _captured["heatmap_uniform"]   = hm_u.float().cpu()
    _captured["heatmap_anchor"]    = hm_a.float().cpu()
    _captured["latent_h"]          = latent_h
    _captured["latent_w"]          = latent_w

    # continue with original to actually build the mask
    return _orig_build(self, cross_attn_maps, token_importance, latent_h, latent_w)

_mb.SemanticMaskBuilder.build_mask_from_cross_attention = _hook_build

# ── Also capture token-level importance & words ────────────────────────────────
_orig_compute = _mb.SemanticMaskBuilder.compute_token_importance

def _hook_compute(self, prompt, prompt_mask, input_ids, tokenizer):
    importance = _orig_compute(self, prompt, prompt_mask, input_ids, tokenizer)
    # Decode tokens to words for the bar chart
    ids = input_ids[0]
    special = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    words, imps = [], []
    for i, tid in enumerate(ids):
        tid_v = tid.item()
        if tid_v in special or tid_v == 0:
            continue
        tok = tokenizer.decode([tid_v], skip_special_tokens=True).strip()
        if tok:
            words.append(tok)
            imps.append(importance[i].item() if importance is not None else 1.0)
    _captured["words"]      = words
    _captured["token_imps"] = imps
    return importance

_mb.SemanticMaskBuilder.compute_token_importance = _hook_compute

# ── Generate ───────────────────────────────────────────────────────────────────
print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()

print(f"Generating: {PROMPT[:60]}...")
img = wrapper.generate_accelerated_dual(
    prompt=PROMPT, seed=SEED, cfg_scale=4.0,
    mask_type="semantic", region_method="threshold",
    skip_ratio=RATIO, s_fg=7.0, s_bg=1.0, mask_step=5,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
)
img.save(f"{OUT_DIR}/generated.png")
print(f"Generated image saved. Captured keys: {list(_captured.keys())}")

# Restore patches
_mb.SemanticMaskBuilder.build_mask_from_cross_attention = _orig_build
_mb.SemanticMaskBuilder.compute_token_importance = _orig_compute

if "heatmap_uniform" not in _captured:
    print("ERROR: hook did not fire — mask may have been skipped. Exiting.")
    sys.exit(1)

hm_u = _captured["heatmap_uniform"].numpy()
hm_a = _captured["heatmap_anchor"].numpy()
H, W  = hm_u.shape

# ── Binary masks (threshold at 1-ratio quantile) ───────────────────────────────
def to_mask(hm, ratio=RATIO):
    thr = np.quantile(hm, 1.0 - ratio)
    return (hm >= thr).astype(np.float32)

mask_u = to_mask(hm_u)
mask_a = to_mask(hm_a)

# ── Pixel-block style heatmap (NEAREST upscale to 512) ────────────────────────
def render_heatmap(hm, size=512):
    """Render heatmap with jet colormap, NEAREST upscale for pixel-block look."""
    cm = plt.get_cmap("jet")
    rgb = (cm(hm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)

def render_mask_overlay(base_img, mask, size=512, fg_color=(255,80,80), bg_color=(80,120,255), alpha=0.45):
    """Overlay fg/bg color on image according to binary mask."""
    base = np.array(base_img.resize((size, size))).astype(float)
    mask_up = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    ).astype(float) / 255.0
    overlay = np.zeros_like(base)
    for c, v in enumerate(fg_color):
        overlay[:, :, c] = mask_up * v
    for c, v in enumerate(bg_color):
        overlay[:, :, c] += (1 - mask_up) * v
    result = base * (1 - alpha) + overlay * alpha
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))

hm_img_u = render_heatmap(hm_u)
hm_img_a = render_heatmap(hm_a)
ov_u     = render_mask_overlay(img, mask_u)
ov_a     = render_mask_overlay(img, mask_a)

hm_img_u.save(f"{OUT_DIR}/heatmap_uniform.png")
hm_img_a.save(f"{OUT_DIR}/heatmap_anchor.png")
ov_u.save(f"{OUT_DIR}/mask_uniform.png")
ov_a.save(f"{OUT_DIR}/mask_anchor.png")

# ── Token importance bar chart ─────────────────────────────────────────────────
words = _captured.get("words", [])
imps  = _captured.get("token_imps", [])

if words:
    fig_bar, ax = plt.subplots(figsize=(max(6, len(words)*0.4), 2.8))
    colors = ["#e74c3c" if v > 0.6 else "#f39c12" if v > 0.3 else "#95a5a6"
              for v in imps]
    ax.bar(range(len(words)), imps, color=colors, width=0.7)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CLIP anchor similarity", fontsize=9)
    ax.set_title("Per-token importance (CLIP × aesthetic anchors)", fontsize=10)
    ax.axhline(0.6, color="red", ls="--", lw=1, label="threshold 0.60")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig_bar.tight_layout()
    fig_bar.savefig(f"{OUT_DIR}/token_weights.png", dpi=150)
    plt.close(fig_bar)
    print(f"Token bar chart saved ({len(words)} tokens).")

# ── 6-panel comparison figure ─────────────────────────────────────────────────
SIZE = 512
fig = plt.figure(figsize=(14, 5.5))
gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.04)

panels = [
    (img.resize((SIZE, SIZE)),     "Generated image"),
    (hm_img_u,                     "Heatmap\n(uniform weights)"),
    (ov_u,                         "Mask\n(uniform weights)"),
    (hm_img_a,                     "Heatmap\n(anchor weights)"),
    (ov_a,                         "Mask\n(anchor weights)"),
]

for i, (im, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    ax.imshow(im)
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis("off")

# Add dividing line between uniform and anchor columns
fig.patches.append(plt.Rectangle(
    (2.5/6 - 0.005, 0.02), 0.010, 0.96,
    transform=fig.transFigure, color="black", lw=0, zorder=5
))
fig.text(1.5/6, 0.97, "Without anchor weighting", ha="center", va="top",
         fontsize=11, color="#555555")
fig.text(4/6,   0.97, "With CLIP anchor weighting", ha="center", va="top",
         fontsize=11, color="#c0392b", fontweight="bold")

fig.suptitle(f'"{PROMPT[:70]}..."', fontsize=9, y=1.01, color="#333333")
fig.savefig(f"{OUT_DIR}/comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll outputs saved to {OUT_DIR}/")
print(f"  heatmap_uniform.png  heatmap_anchor.png")
print(f"  mask_uniform.png     mask_anchor.png")
print(f"  token_weights.png    comparison.png")
