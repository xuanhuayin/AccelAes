#!/usr/bin/env python3
"""
FLUX anchor comparison visualization (multi-subject).

For each subject: capture T5-cross-attention avg_attn at mask_step,
compute 4 heatmap variants, save individual figures.

Output: outputs/paper_figures/flux/{tag}/
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

from src.models.flux_wrapper import FluxDiTWrapper
from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
from transformers import CLIPTokenizer, CLIPTextModel

SUBJECTS = [
    ("lion",
     "A majestic lion at golden hour in the savanna, photorealistic, "
     "cinematic lighting, intricate fur detail"),
    ("astronaut",
     "Portrait of an astronaut with intricate spacesuit details, "
     "photorealistic, cinematic"),
    ("city",
     "A futuristic Tokyo street at night with neon lights and intricate "
     "reflections on wet pavement, cinematic"),
    ("dragon",
     "An ancient dragon with intricate scales perched on a mountain peak, "
     "dramatic lighting, cinematic fantasy art"),
    ("flowers",
     "Macro photograph of orchids with morning dew drops, "
     "photorealistic, intricate petals, soft bokeh"),
]

OUT_BASE  = "outputs/paper_figures/flux"
SEED      = 42
RATIO     = 0.50
THRESHOLD = 0.60
VIZ_SIZE  = 512

ANCHORS = [
    "photorealistic", "realistic", "detailed", "sharp focus", "depth of field",
    "volumetric lighting", "intricate", "highly detailed", "cinematic",
    "stunning", "beautiful", "elegant", "artistic", "masterpiece",
    "professional photography", "vivid", "vibrant", "soft bokeh", "bokeh",
    "dramatic lighting", "fantasy art",
    "portrait", "close-up", "full body", "main subject", "character",
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

# ── Hook into SD3SemanticMaskBuilder.build_mask ────────────────────────────────
_captured = {}
_orig_build = SD3SemanticMaskBuilder.build_mask

def _hook_build(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
    if affinity_maps:
        stacked = torch.stack(affinity_maps, dim=0)   # (L, B, heads, N_img, N_txt)
        B = stacked.shape[1]
        cond = stacked[:, B//2:, :, :, :]             # conditional half
        avg  = cond.mean(dim=[0, 1, 2]).float().cpu()  # (N_img, N_txt)
        _captured["avg_attn"] = avg
        _captured["patch_h"]  = patch_h
        _captured["patch_w"]  = patch_w
    return _orig_build(self, affinity_maps, patch_h, patch_w, clip_token_weights)

SD3SemanticMaskBuilder.build_mask = _hook_build

# ── Load FLUX ─────────────────────────────────────────────────────────────────
print("Loading FLUX wrapper...")
wrapper = FluxDiTWrapper(dtype="bf16")
t5_tokenizer = wrapper.pipe.tokenizer_2

# ── Load CLIP for anchor similarity ───────────────────────────────────────────
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

# ── Helper: map word importance → T5 token weights ────────────────────────────
def t5_weights_for_words(target_words, t5_ids_list, N_txt, hi=1.0, lo=0.005):
    target_ids = set()
    for w in target_words:
        w_clean = w.strip(",.").strip()
        if not w_clean: continue
        enc = t5_tokenizer(w_clean, add_special_tokens=False, return_tensors="pt")
        for tid in enc["input_ids"][0].tolist():
            target_ids.add(tid)
    pad = t5_tokenizer.pad_token_id
    eos = t5_tokenizer.eos_token_id
    w_arr = np.zeros(N_txt, dtype=np.float32)
    for i, tid in enumerate(t5_ids_list[:N_txt]):
        if tid in (pad, eos, 0): continue
        w_arr[i] = hi if tid in target_ids else lo
    return w_arr

# ── Process each subject ──────────────────────────────────────────────────────
for tag, prompt in SUBJECTS:
    out_dir = f"{OUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n{tag.upper()}: {prompt[:65]}...")

    sasd_img = Image.open(f"{out_dir}/sasd.png").convert("RGB")

    # Run generation — hook fires at mask_step
    _captured.clear()
    _ = wrapper.generate_accelerated_sparse(
        prompt, seed=SEED, steps=28,
        mask_step=8, skip_ratio=RATIO, n_segments=64,
        region_method="threshold",
        sparse_blocks=True,
        full_skip_interval=2, warmup_steps=5,
    )

    if "avg_attn" not in _captured:
        print(f"  WARNING: hook did not fire, skipping.")
        continue

    avg_attn = _captured["avg_attn"].numpy()   # (N_img, N_txt)
    patch_h  = _captured["patch_h"]
    patch_w  = _captured["patch_w"]
    N_txt    = avg_attn.shape[1]
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

    # ── T5 token weights ───────────────────────────────────────────────────────
    t5_enc_out = t5_tokenizer(
        prompt, return_tensors="pt",
        padding="max_length", max_length=N_txt, truncation=True,
    )
    t5_ids_list = t5_enc_out["input_ids"][0].tolist()

    high_words = [w.strip(",.") for w, imp in zip(words_raw, word_imp) if imp > 0]
    nona_words = [w for w, s in zip(words_raw, sim_np)
                  if s < THRESHOLD and w.strip(",.").lower() not in STOPWORDS
                  and len(w.strip(",.")) > 1]
    func_words = [w for w in words_raw
                  if w.strip(",.").lower() in STOPWORDS or w.strip() == ","]

    print(f"  Anchor:      {high_words}")
    print(f"  Non-aes:     {nona_words}")
    print(f"  Func/stop:   {func_words}")

    w_anchor = t5_weights_for_words(high_words, t5_ids_list, N_txt)
    w_nonaes = t5_weights_for_words(nona_words, t5_ids_list, N_txt)
    w_func   = t5_weights_for_words(func_words, t5_ids_list, N_txt)

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
    ax.set_title(f"{tag} — per-token CLIP anchor similarity (FLUX/T5)", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/token_weights.png", dpi=150)
    plt.close(fig)
    print(f"  Saved token_weights.png")

SD3SemanticMaskBuilder.build_mask = _orig_build
print("\nAll done.")
