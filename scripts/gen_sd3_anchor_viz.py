#!/usr/bin/env python3
"""
For SD3 dragon and flowers: generate per-component anchor comparison figures.

Per subject, saves:
  heatmap_{anchor,nonaesthetic,funcword,uniform}.png   — jet colormap, NEAREST upscale
  mask_binary_{anchor,nonaesthetic,funcword,uniform}.png — clean black/white threshold mask
  mask_overlay_{anchor,nonaesthetic,funcword,uniform}.png — overlay on sasd.png
  token_weights.png                                      — CLIP anchor similarity bar chart

Output dirs:
  outputs/paper_figures/sd3/dragon/
  outputs/paper_figures/sd3/flowers/
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.sd3_wrapper import SD3Wrapper
from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
from transformers import CLIPTokenizer, CLIPTextModel

SUBJECTS = [
    {
        "name":   "dragon",
        "prompt": ("An ancient dragon with intricate scales perched on a mountain peak, "
                   "dramatic lighting, cinematic fantasy art"),
        "out_dir": "outputs/paper_figures/sd3/dragon",
    },
    {
        "name":   "flowers",
        "prompt": ("Macro photograph of orchids with morning dew drops, "
                   "photorealistic, intricate petals, soft bokeh"),
        "out_dir": "outputs/paper_figures/sd3/flowers",
    },
]

SEED      = 42
RATIO     = 0.50
THRESHOLD = 0.60
VIZ_SIZE  = 512

ANCHORS = [
    "photorealistic","realistic","detailed","sharp focus","depth of field",
    "volumetric lighting","intricate","highly detailed","cinematic",
    "stunning","beautiful","elegant","artistic","masterpiece",
    "professional photography","vivid","vibrant","soft bokeh","bokeh",
    "dramatic lighting","fantasy art",
    "portrait","close-up","full body","main subject","character",
]

STOPWORDS = {"a","an","the","in","on","of","for","with","and","or","but",
             "is","are","was","were","her","his","its","my","our","their",
             "this","that","these","those","by","at","to","as","from"}

# ── Rendering helpers ──────────────────────────────────────────────────────────
def render_heatmap(hm, size=VIZ_SIZE):
    cm  = plt.get_cmap("jet")
    rgb = (cm(hm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.BILINEAR)

def render_binary_mask(mask_np, size=VIZ_SIZE):
    """Clean hard binary mask — white=FG, black=BG, no blur."""
    hard = (mask_np > 0.5).astype(np.uint8) * 255
    return Image.fromarray(hard).resize((size, size), Image.NEAREST)

def render_overlay(base_img, mask_np, size=VIZ_SIZE, alpha=0.45):
    base   = np.array(base_img.resize((size, size))).astype(float)
    m_up   = np.array(
        Image.fromarray((mask_np * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    ).astype(float) / 255.0
    fg_col = np.array([255,  80,  80], dtype=float)
    bg_col = np.array([ 80, 120, 255], dtype=float)
    overlay = m_up[:,:,None] * fg_col + (1 - m_up[:,:,None]) * bg_col
    return Image.fromarray((base*(1-alpha)+overlay*alpha).clip(0,255).astype(np.uint8))

def make_heatmap(avg_attn, weights=None):
    """avg_attn: (N_img, N_txt) numpy. Returns normalized (H, W) heatmap."""
    if weights is None:
        hm = avg_attn.mean(axis=-1)
    else:
        w  = weights / (weights.sum() + 1e-8)
        hm = (avg_attn * w[None, :]).sum(axis=-1)
    hm = hm.reshape(patch_h, patch_w)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm

def to_mask(hm, ratio=RATIO):
    thr = np.quantile(hm, 1.0 - ratio)
    return (hm >= thr).astype(np.float32)

def t5_weights_for_words(target_words, t5_ids_list, t5_tokenizer, N_txt, hi=1.0, lo=0.005):
    target_ids = set()
    for w in target_words:
        w_clean = w.strip(",.").strip()
        if not w_clean:
            continue
        enc = t5_tokenizer(w_clean, add_special_tokens=False, return_tensors="pt")
        for tid in enc["input_ids"][0].tolist():
            target_ids.add(tid)
    w_arr = np.zeros(N_txt, dtype=np.float32)
    pad   = t5_tokenizer.pad_token_id
    eos   = t5_tokenizer.eos_token_id
    for i, tid in enumerate(t5_ids_list):
        if tid in (pad, eos, 0):
            continue
        w_arr[i] = hi if tid in target_ids else lo
    return w_arr

# ── Hook ──────────────────────────────────────────────────────────────────────
_captured = {}
_orig_build = SD3SemanticMaskBuilder.build_mask

def _hook_build(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
    if affinity_maps:
        stacked = torch.stack(affinity_maps, dim=0)
        B = stacked.shape[1]
        cond = stacked[:, B//2:, :, :, :]
        avg  = cond.mean(dim=[0, 1, 2]).float().cpu().numpy()
        _captured["avg_attn"] = avg
        _captured["patch_h"]  = patch_h
        _captured["patch_w"]  = patch_w
    return _orig_build(self, affinity_maps, patch_h, patch_w, clip_token_weights)

SD3SemanticMaskBuilder.build_mask = _hook_build

# ── Load SD3 ──────────────────────────────────────────────────────────────────
print("Loading SD3 wrapper...")
wrapper = SD3Wrapper()

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
    anc_emb  = F.normalize(anc_emb / (mask_a.sum(1, keepdim=True) + 1e-8), dim=-1)

clip_enc.to("cpu")
torch.cuda.empty_cache()

t5_tokenizer = wrapper.pipe.tokenizer_2

# ── Process each subject ──────────────────────────────────────────────────────
for subj in SUBJECTS:
    name   = subj["name"]
    prompt = subj["prompt"]
    out_dir = subj["out_dir"]
    print(f"\n{'='*60}\n{name.upper()}: {prompt[:60]}...")

    # Load reference images
    sasd_img = Image.open(f"{out_dir}/sasd.png").convert("RGB")

    # Run SD3 AccelAes to capture affinity maps
    _captured.clear()
    _ = wrapper.generate_accelerated(
        prompt=prompt, seed=SEED, cfg_scale=7.0,
        mask_type="semantic",
        skip_ratio=RATIO, s_fg=9.0, s_bg=2.0,
        mask_step=5, full_skip_interval=2,
        sparse_attn=True,
    )

    if "avg_attn" not in _captured:
        print(f"  WARNING: hook did not fire for {name}, skipping.")
        continue

    avg_attn = _captured["avg_attn"]   # (N_img, N_txt)
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
        w_emb  = F.normalize(w_emb / (mask_w.sum(1, keepdim=True) + 1e-8), dim=-1)
        sim_np = (w_emb @ anc_emb.T).max(dim=-1).values.cpu().float().numpy()
    clip_enc.to("cpu")
    torch.cuda.empty_cache()

    word_importance = np.where(sim_np >= THRESHOLD, sim_np, 0.0)

    print("  Per-word similarity:")
    for w, s, imp in zip(words_raw, sim_np, word_importance):
        mark = "★" if imp > 0 else " "
        print(f"    {mark} {w:30s} sim={s:.3f}")

    # ── T5 token weights ───────────────────────────────────────────────────────
    t5_enc_out = t5_tokenizer(
        prompt, return_tensors="pt",
        padding="max_length", max_length=N_txt, truncation=True,
    )
    t5_ids      = t5_enc_out["input_ids"][0]
    t5_ids_list = t5_ids.tolist()

    high_imp_words  = [w.strip(",.") for w, imp in zip(words_raw, word_importance) if imp > 0]
    non_aes_words   = [w for w, s in zip(words_raw, sim_np)
                       if s < THRESHOLD and w.strip(",.").lower() not in STOPWORDS
                       and len(w.strip(",.")) > 1]
    func_words      = [w for w in words_raw
                       if w.strip(",.").lower() in STOPWORDS or w.strip() == ","]

    print(f"  Anchor words:       {high_imp_words}")
    print(f"  Non-aes content:    {non_aes_words}")
    print(f"  Function/stop:      {func_words}")

    w_anchor   = t5_weights_for_words(high_imp_words, t5_ids_list, t5_tokenizer, N_txt)
    w_nonaes   = t5_weights_for_words(non_aes_words,  t5_ids_list, t5_tokenizer, N_txt)
    w_func     = t5_weights_for_words(func_words,     t5_ids_list, t5_tokenizer, N_txt)

    # ── Compute heatmaps ───────────────────────────────────────────────────────
    hm = {
        "anchor":       make_heatmap(avg_attn, w_anchor),
        "nonaesthetic": make_heatmap(avg_attn, w_nonaes),
        "funcword":     make_heatmap(avg_attn, w_func),
        "uniform":      make_heatmap(avg_attn, None),
    }
    masks = {k: to_mask(v) for k, v in hm.items()}

    # ── Save figures ───────────────────────────────────────────────────────────
    for tag, h in hm.items():
        m = masks[tag]
        render_heatmap(h).save(f"{out_dir}/heatmap_{tag}.png")
        render_binary_mask(m).save(f"{out_dir}/mask_binary_{tag}.png")
        render_overlay(sasd_img, m).save(f"{out_dir}/mask_overlay_{tag}.png")
        print(f"  saved heatmap_{tag}.png  mask_binary_{tag}.png  mask_overlay_{tag}.png")

    # Also overwrite old binary_mask.png with clean anchor-threshold version
    render_binary_mask(masks["anchor"]).save(f"{out_dir}/binary_mask.png")
    print(f"  updated binary_mask.png (clean threshold, anchor weighting)")

    # ── Token weight bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(words_raw)*0.55), 3))
    colors  = ["#e74c3c" if s >= THRESHOLD else "#95a5a6" for s in sim_np]
    ax.bar(range(len(words_raw)), sim_np, color=colors, width=0.7)
    ax.set_xticks(range(len(words_raw)))
    ax.set_xticklabels([w.replace(",","") for w in words_raw],
                       rotation=40, ha="right", fontsize=9)
    ax.axhline(THRESHOLD, color="#e74c3c", ls="--", lw=1.2,
               label=f"threshold {THRESHOLD}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CLIP cosine similarity\nwith aesthetic anchors", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title(f"{name} — per-token CLIP anchor similarity", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/token_weights.png", dpi=150)
    plt.close(fig)
    print(f"  saved token_weights.png")

SD3SemanticMaskBuilder.build_mask = _orig_build
print("\nAll done.")
