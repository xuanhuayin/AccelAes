#!/usr/bin/env python3
"""
Lumina anchor comparison visualization.

For each subject: capture cross-attention avg_attn at mask_step,
compute 4 heatmap variants, save individual figures.

Output: outputs/accelae_figures_lumina/lumina/{tag}/
  heatmap_{anchor,nonaesthetic,funcword,uniform}.png
  mask_binary_{anchor,nonaesthetic,funcword,uniform}.png
  mask_overlay_{anchor,nonaesthetic,funcword,uniform}.png
  token_weights.png
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import src.sparse.mask_builders as _mb
from transformers import CLIPTokenizer, CLIPTextModel

SUBJECTS = [
    ("dragon",
     "An ancient dragon with intricate scales perched on a mountain peak, "
     "dramatic lighting, cinematic fantasy art"),
    ("warrior",
     "A samurai warrior in intricate armor standing in a bamboo forest, "
     "cinematic lighting, photorealistic, sharp focus"),
    ("portrait",
     "Close up portrait of an elegant woman with intricate floral headpiece, "
     "photorealistic, studio lighting, detailed skin texture"),
]

OUT_BASE  = "outputs/accelae_figures_lumina/lumina"
SEED      = 42
RATIO     = 0.50
THRESHOLD = 0.60
VIZ_SIZE  = 512

SASD_KWARGS = dict(
    mask_type="semantic", region_method="threshold",
    skip_ratio=RATIO, s_fg=7.0, s_bg=1.0, mask_step=5,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
)

ANCHORS = [
    "photorealistic","realistic","detailed","sharp focus","depth of field",
    "volumetric lighting","intricate","highly detailed","cinematic",
    "stunning","beautiful","elegant","artistic","masterpiece",
    "professional photography","vivid","vibrant","soft bokeh","bokeh",
    "dramatic lighting","fantasy art","studio lighting",
    "portrait","close-up","full body","main subject","character",
]

STOPWORDS = {"a","an","the","in","on","of","for","with","and","or","but",
             "is","are","was","were","her","his","its","my","our","their",
             "this","that","these","those","by","at","to","as","from","up"}

# ── Rendering ─────────────────────────────────────────────────────────────────
def render_heatmap(hm, size=VIZ_SIZE):
    rgb = (plt.get_cmap("jet")(hm)[:,:,:3]*255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.BILINEAR)

def render_binary_mask(mask_np, size=VIZ_SIZE):
    hard = (mask_np > 0.5).astype(np.uint8) * 255
    return Image.fromarray(hard).resize((size, size), Image.NEAREST)

def render_overlay(base_img, mask_np, size=VIZ_SIZE, alpha=0.50, blur_radius=14):
    base  = np.array(base_img.resize((size, size))).astype(float)
    m_pil = Image.fromarray((mask_np*255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    m_pil = m_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    m_up  = np.array(m_pil).astype(float)/255.0
    fg    = np.array([255, 140,  60], dtype=float)
    bg    = np.array([ 60,  80, 220], dtype=float)
    ov    = m_up[:,:,None]*fg + (1-m_up[:,:,None])*bg
    return Image.fromarray((base*(1-alpha)+ov*alpha).clip(0,255).astype(np.uint8))

# ── Hook into build_mask_from_cross_attention ─────────────────────────────────
_captured = {}
_orig_build = _mb.SemanticMaskBuilder.build_mask_from_cross_attention

def _hook_build(self, cross_attn_maps, token_importance, latent_h, latent_w):
    num_patches = latent_h * latent_w
    all_attn = torch.stack([m.mean(dim=0) for m in cross_attn_maps])
    avg_attn  = all_attn.mean(dim=0).float()   # (patches, seq_len)
    _captured["avg_attn"]       = avg_attn[:num_patches].cpu()
    _captured["token_importance"] = (token_importance.float().cpu()
                                     if token_importance is not None else None)
    _captured["latent_h"] = latent_h
    _captured["latent_w"] = latent_w
    return _orig_build(self, cross_attn_maps, token_importance, latent_h, latent_w)

_mb.SemanticMaskBuilder.build_mask_from_cross_attention = _hook_build

# ── Load Lumina ───────────────────────────────────────────────────────────────
from src.models.dit_wrapper import LuminaDiTWrapper
print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()
tokenizer = wrapper.pipe.tokenizer   # Gemma SentencePiece

# ── Load CLIP ─────────────────────────────────────────────────────────────────
CLIP_ID = "openai/clip-vit-large-patch14"
print("Loading CLIP...")
clip_tok = CLIPTokenizer.from_pretrained(CLIP_ID)
clip_enc = CLIPTextModel.from_pretrained(CLIP_ID).eval().to("cuda")

with torch.no_grad():
    anc_toks = clip_tok(ANCHORS, padding=True, truncation=True, return_tensors="pt").to("cuda")
    anc_out  = clip_enc(**anc_toks)
    mask_a   = anc_toks["attention_mask"].float()
    for sid in (clip_tok.bos_token_id, clip_tok.eos_token_id):
        if sid is not None:
            mask_a[anc_toks["input_ids"] == sid] = 0
    anc_emb  = (anc_out.last_hidden_state * mask_a.unsqueeze(-1)).sum(1)
    anc_emb  = F.normalize(anc_emb / (mask_a.sum(1, keepdim=True)+1e-8), dim=-1)
clip_enc.to("cpu"); torch.cuda.empty_cache()

# ── Helper: map word importance → Gemma token weights ─────────────────────────
def gemma_weights_for_words(target_words, gemma_ids, tokenizer, seq_len, hi=1.0, lo=0.005):
    target_ids = set()
    for w in target_words:
        w_clean = w.strip(",.").strip()
        if not w_clean: continue
        enc = tokenizer(w_clean, add_special_tokens=False, return_tensors="pt")
        for tid in enc["input_ids"][0].tolist():
            target_ids.add(tid)
    special = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, 0}
    w_arr = np.zeros(seq_len, dtype=np.float32)
    for i, tid in enumerate(gemma_ids[:seq_len]):
        if tid in special: continue
        w_arr[i] = hi if tid in target_ids else lo
    return w_arr

# ── Process each subject ──────────────────────────────────────────────────────
for tag, prompt in SUBJECTS:
    out_dir = f"{OUT_BASE}/{tag}"
    print(f"\n{'='*60}\n{tag.upper()}: {prompt[:65]}...")

    sasd_img = Image.open(f"{out_dir}/result.png").convert("RGB")

    # Run generation — hook fires at mask_step
    _captured.clear()
    _ = wrapper.generate_accelerated_dual(
        prompt=prompt, seed=SEED, cfg_scale=4.0, **SASD_KWARGS
    )

    if "avg_attn" not in _captured:
        print(f"  WARNING: hook did not fire, skipping.")
        continue

    avg_attn = _captured["avg_attn"].numpy()   # (patches, seq_len)
    latent_h = _captured["latent_h"]
    latent_w = _captured["latent_w"]
    seq_len  = avg_attn.shape[1]
    patch_h  = latent_h
    patch_w  = latent_w
    print(f"  avg_attn: {avg_attn.shape}  grid: {patch_h}×{patch_w}")

    # ── CLIP similarity per word ───────────────────────────────────────────────
    words_raw = prompt.replace(",", " ,").split()
    words_raw = [w for w in words_raw if w.strip()]

    clip_enc.to("cuda")
    with torch.no_grad():
        w_toks = clip_tok(words_raw, padding=True, truncation=True, return_tensors="pt").to("cuda")
        w_out  = clip_enc(**w_toks)
        mask_w = w_toks["attention_mask"].float()
        for sid in (clip_tok.bos_token_id, clip_tok.eos_token_id):
            if sid is not None:
                mask_w[w_toks["input_ids"] == sid] = 0
        w_emb  = (w_out.last_hidden_state * mask_w.unsqueeze(-1)).sum(1)
        w_emb  = F.normalize(w_emb / (mask_w.sum(1, keepdim=True)+1e-8), dim=-1)
        sim_np = (w_emb @ anc_emb.T).max(dim=-1).values.cpu().float().numpy()
    clip_enc.to("cpu"); torch.cuda.empty_cache()

    word_imp = np.where(sim_np >= THRESHOLD, sim_np, 0.0)

    print("  Per-word similarity:")
    for w, s, imp in zip(words_raw, sim_np, word_imp):
        mark = "★" if imp > 0 else " "
        print(f"    {mark} {w:30s} sim={s:.3f}")

    # ── Gemma token weights ────────────────────────────────────────────────────
    gemma_enc  = tokenizer(prompt, return_tensors="pt")
    gemma_ids  = gemma_enc["input_ids"][0].tolist()

    high_words = [w.strip(",.") for w, imp in zip(words_raw, word_imp) if imp > 0]
    nona_words = [w for w, s in zip(words_raw, sim_np)
                  if s < THRESHOLD and w.strip(",.").lower() not in STOPWORDS
                  and len(w.strip(",.")) > 1]
    func_words = [w for w in words_raw
                  if w.strip(",.").lower() in STOPWORDS or w.strip() == ","]

    print(f"  Anchor:      {high_words}")
    print(f"  Non-aes:     {nona_words}")
    print(f"  Func/stop:   {func_words}")

    w_anchor = gemma_weights_for_words(high_words, gemma_ids, tokenizer, seq_len)
    w_nonaes = gemma_weights_for_words(nona_words, gemma_ids, tokenizer, seq_len)
    w_func   = gemma_weights_for_words(func_words, gemma_ids, tokenizer, seq_len)

    # ── Compute heatmaps ───────────────────────────────────────────────────────
    def make_heatmap(attn_np, w=None):
        if w is None:
            hm = attn_np.mean(axis=-1)
        else:
            wn = w / (w.sum() + 1e-8)
            hm = (attn_np * wn[None, :]).sum(axis=-1)
        hm = hm.reshape(patch_h, patch_w)
        return (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    def to_mask(hm):
        thr = np.quantile(hm, 1.0 - RATIO)
        return (hm >= thr).astype(np.float32)

    variants = {
        "anchor":       make_heatmap(avg_attn, w_anchor),
        "nonaesthetic": make_heatmap(avg_attn, w_nonaes),
        "funcword":     make_heatmap(avg_attn, w_func),
        "uniform":      make_heatmap(avg_attn, None),
    }
    masks = {k: to_mask(v) for k, v in variants.items()}

    # ── Save ──────────────────────────────────────────────────────────────────
    for vname, hm in variants.items():
        m = masks[vname]
        render_heatmap(hm).save(f"{out_dir}/heatmap_{vname}.png")
        render_binary_mask(m).save(f"{out_dir}/mask_binary_{vname}.png")
        render_overlay(sasd_img, m).save(f"{out_dir}/mask_overlay_{vname}.png")
    print(f"  Saved 3×4 = 12 figures to {out_dir}/")

    # ── Quantitative diff vs anchor ────────────────────────────────────────────
    ref_hm   = variants["anchor"]
    ref_mask = masks["anchor"]
    print("  Mask IoU vs anchor:")
    for vname in ["nonaesthetic","funcword","uniform"]:
        m = masks[vname]
        iou = (ref_mask.astype(bool) & m.astype(bool)).sum() / \
              (ref_mask.astype(bool) | m.astype(bool)).sum()
        hm_diff = np.abs(variants[vname] - ref_hm).mean()
        print(f"    {vname:15s}: IoU={iou:.3f}  hm_diff={hm_diff:.4f}")

    # ── Token weights bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(words_raw)*0.55), 3))
    colors  = ["#e74c3c" if s >= THRESHOLD else "#95a5a6" for s in sim_np]
    ax.bar(range(len(words_raw)), sim_np, color=colors, width=0.7)
    ax.set_xticks(range(len(words_raw)))
    ax.set_xticklabels([w.replace(",","") for w in words_raw],
                       rotation=40, ha="right", fontsize=9)
    ax.axhline(THRESHOLD, color="#e74c3c", ls="--", lw=1.2, label=f"threshold {THRESHOLD}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CLIP cosine similarity\nwith aesthetic anchors", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title(f"{tag} — per-token CLIP anchor similarity (Lumina/Gemma)", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/token_weights.png", dpi=150)
    plt.close(fig)
    print(f"  Saved token_weights.png")

_mb.SemanticMaskBuilder.build_mask_from_cross_attention = _orig_build
print("\nAll done.")
