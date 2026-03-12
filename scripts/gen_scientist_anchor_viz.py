#!/usr/bin/env python3
"""
Generate per-component figures for scientist anchor comparison:
  generated.png       — FLUX AccelAes generated image (already exists, reuse)
  heatmap_uniform.png — cross-attn heatmap, uniform text-token weights
  heatmap_anchor.png  — cross-attn heatmap, CLIP anchor-weighted tokens
  mask_uniform.png    — binary mask overlay (uniform)
  mask_anchor.png     — binary mask overlay (anchor)
  token_weights.png   — per-token CLIP similarity bar chart

All saved to outputs/paper_figures/flux/scientist/
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

OUT_DIR = "outputs/paper_figures/flux/scientist"
PROMPT  = (
    "A Victorian-era scientist in her detailed laboratory, photorealistic, "
    "intricate steampunk equipment"
)
SEED    = 0
RATIO   = 0.50   # fg fraction

# ── Load existing generated image ─────────────────────────────────────────────
base_img = Image.open(f"{OUT_DIR}/baseline.png").convert("RGB")
sasd_img = Image.open(f"{OUT_DIR}/sasd.png").convert("RGB")

# ── Load model & capture affinity maps ────────────────────────────────────────
from src.models.flux_wrapper import FluxDiTWrapper
from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder

_captured = {}

_orig_build = SD3SemanticMaskBuilder.build_mask

def _hook_build(self, affinity_maps, patch_h, patch_w, clip_token_weights=None):
    if not affinity_maps:
        return _orig_build(self, affinity_maps, patch_h, patch_w, clip_token_weights)

    stacked = torch.stack(affinity_maps, dim=0)           # (L, B, heads, N_img, N_txt)
    B = stacked.shape[1]
    stacked = stacked[:, B // 2:, :, :, :]               # cond half
    avg = stacked.mean(dim=[0, 1, 2]).float().cpu()       # (N_img, N_txt)

    _captured["avg_attn"] = avg
    _captured["patch_h"]  = patch_h
    _captured["patch_w"]  = patch_w

    return _orig_build(self, affinity_maps, patch_h, patch_w, clip_token_weights)

SD3SemanticMaskBuilder.build_mask = _hook_build

print("Loading FLUX wrapper...")
wrapper = FluxDiTWrapper(dtype="bf16")

print(f"Running FLUX AccelAes on scientist prompt...")
img = wrapper.generate_accelerated_sparse(
    PROMPT, seed=SEED, steps=28,
    mask_step=8, skip_ratio=RATIO, n_segments=64,
    region_method="threshold",
    sparse_blocks=True,
    full_skip_interval=2, warmup_steps=5,
)

SD3SemanticMaskBuilder.build_mask = _orig_build

if "avg_attn" not in _captured:
    print("Hook did not fire — exiting.")
    sys.exit(1)

avg_attn = _captured["avg_attn"]   # (N_img, N_txt)
patch_h  = _captured["patch_h"]
patch_w  = _captured["patch_w"]
N_txt    = avg_attn.shape[1]
print(f"Captured avg_attn: {avg_attn.shape}  patch grid: {patch_h}×{patch_w}")

# ── Compute CLIP anchor weights for T5 tokens ─────────────────────────────────
# Load CLIP (same as SemanticMaskBuilder for Lumina)
from transformers import CLIPTokenizer, CLIPTextModel

ANCHORS = [
    # quality / photorealism
    "photorealistic", "realistic", "detailed", "sharp focus", "depth of field",
    "volumetric lighting", "intricate", "highly detailed", "cinematic",
    # style / aesthetic
    "stunning", "beautiful", "elegant", "artistic", "masterpiece",
    "professional photography", "vivid", "vibrant",
    # subject
    "portrait", "close-up", "full body", "main subject", "character",
]

clip_model_id = "openai/clip-vit-large-patch14"
print("Loading CLIP tokenizer/encoder for anchor weights...")
clip_tok = CLIPTokenizer.from_pretrained(clip_model_id)
clip_enc = CLIPTextModel.from_pretrained(clip_model_id).eval().to("cuda")

with torch.no_grad():
    # Encode anchors
    anc_toks = clip_tok(ANCHORS, padding=True, truncation=True, return_tensors="pt").to("cuda")
    anc_out  = clip_enc(**anc_toks)
    # Pool over non-special tokens
    mask_a   = anc_toks["attention_mask"].float()
    for sid in (clip_tok.bos_token_id, clip_tok.eos_token_id):
        if sid is not None:
            mask_a[anc_toks["input_ids"] == sid] = 0
    anc_emb  = (anc_out.last_hidden_state * mask_a.unsqueeze(-1)).sum(1)
    anc_emb  = F.normalize(anc_emb / (mask_a.sum(1, keepdim=True) + 1e-8), dim=-1)

    # Tokenize the prompt with CLIP to get word-level scores
    words_raw  = PROMPT.replace(",", " ,").split()
    # filter empty
    words_raw  = [w for w in words_raw if w.strip()]
    word_toks  = clip_tok(words_raw, padding=True, truncation=True, return_tensors="pt").to("cuda")
    word_out   = clip_enc(**word_toks)
    mask_w     = word_toks["attention_mask"].float()
    for sid in (clip_tok.bos_token_id, clip_tok.eos_token_id):
        if sid is not None:
            mask_w[word_toks["input_ids"] == sid] = 0
    word_emb   = (word_out.last_hidden_state * mask_w.unsqueeze(-1)).sum(1)
    word_emb   = F.normalize(word_emb / (mask_w.sum(1, keepdim=True) + 1e-8), dim=-1)

    # Per-word cosine similarity with anchors
    sim        = (word_emb @ anc_emb.T).max(dim=-1).values   # (num_words,)
    sim_np     = sim.cpu().float().numpy()

clip_enc.to("cpu")
torch.cuda.empty_cache()

# Threshold at 0.60
THRESHOLD = 0.60
word_importance = np.where(sim_np >= THRESHOLD, sim_np, 0.0)

print("\nPer-word CLIP anchor similarity:")
for w, s, imp in zip(words_raw, sim_np, word_importance):
    mark = "★" if imp > 0 else " "
    print(f"  {mark} {w:25s}  sim={s:.3f}  imp={imp:.3f}")

# Map word importance to actual T5 token positions via token ID matching
t5_tokenizer = wrapper.pipe.tokenizer_2   # T5 tokenizer

# Full-prompt tokenization (padded to N_txt)
t5_enc = t5_tokenizer(
    PROMPT, return_tensors="pt",
    padding="max_length", max_length=N_txt, truncation=True,
)
t5_ids = t5_enc["input_ids"][0]  # (N_txt,)

# High-importance words (sim >= threshold)
high_imp_words = [w.strip(",.") for w, imp in zip(words_raw, word_importance) if imp > 0]
print(f"\nHigh-importance words: {high_imp_words}")

# Get token IDs for each high-importance word (tokenize each word alone)
high_imp_ids = set()
for hw in high_imp_words:
    enc = t5_tokenizer(hw, add_special_tokens=False, return_tensors="pt")
    for tid in enc["input_ids"][0].tolist():
        high_imp_ids.add(tid)
print(f"High-importance token IDs: {high_imp_ids}")
print(f"Decoded: {[t5_tokenizer.decode([tid]) for tid in high_imp_ids]}")

# Assign weights: high for anchor tokens, small baseline for content, 0 for padding
BASELINE = 0.02
t5_weights = np.zeros(N_txt, dtype=np.float32)
pad_id = t5_tokenizer.pad_token_id
eos_id = t5_tokenizer.eos_token_id
for i, tid in enumerate(t5_ids.tolist()):
    if tid in (pad_id, eos_id, 0):
        continue
    t5_weights[i] = 1.0 if tid in high_imp_ids else BASELINE

anchor_weights = torch.from_numpy(t5_weights).float()
anchor_weights = anchor_weights / (anchor_weights.sum() + 1e-8)

print(f"T5 high-weight tokens: {(t5_weights > 0.5).sum()}")
for i, w in enumerate(t5_weights[:30]):
    tok_str = t5_tokenizer.decode([t5_ids[i].item()]).strip()
    flag = "★" if w > 0.5 else " "
    if w > 0:
        print(f"  {flag} [{i:2d}] '{tok_str}'  w={w:.4f}")

# ── Compute heatmaps ──────────────────────────────────────────────────────────
def make_heatmap(attn, weights=None):
    """attn: (N_img, N_txt).  weights: (N_txt,) or None (uniform)."""
    if weights is None:
        hm = attn.mean(dim=-1)
    else:
        w  = weights.to(attn.device, dtype=attn.dtype)
        hm = (attn * w.unsqueeze(0)).sum(dim=-1)
    hm = hm.reshape(patch_h, patch_w).numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm

hm_u = make_heatmap(avg_attn, weights=None)
hm_a = make_heatmap(avg_attn, weights=anchor_weights)

# ── Non-aesthetic word weights (content words below threshold) ─────────────────
# Two variants:
#   non_content: content words only (scientist, Victorian-era, laboratory, steampunk, equipment)
#   func_only:   function / stop words only (A, in, her, ,)
STOPWORDS = {"a","an","the","in","on","of","for","with","and","or","but",
             "is","are","was","were","her","his","its","my","our","their",
             "this","that","these","those","by","at","to","as","from"}

# Build T5 token weights for each variant
def build_t5_weights_for_words(target_words, t5_ids_list, t5_tokenizer, N_txt):
    """Assign weight=1 to T5 tokens belonging to target_words, 0.005 elsewhere."""
    target_words_clean = [w.strip(",.").lower() for w in target_words]
    target_ids = set()
    for w in target_words_clean:
        if w:
            enc = t5_tokenizer(w, add_special_tokens=False, return_tensors="pt")
            for tid in enc["input_ids"][0].tolist():
                target_ids.add(tid)
    w_arr = np.zeros(N_txt, dtype=np.float32)
    pad_id = t5_tokenizer.pad_token_id
    eos_id = t5_tokenizer.eos_token_id
    for i, tid in enumerate(t5_ids_list):
        if tid in (pad_id, eos_id, 0):
            continue
        w_arr[i] = 1.0 if tid in target_ids else 0.005
    t = torch.from_numpy(w_arr).float()
    return t / (t.sum() + 1e-8)

t5_ids_list = t5_ids.tolist()

# Non-aesthetic content words: all words NOT above threshold and NOT stopwords
non_aes_words = [w for w, s in zip(words_raw, sim_np)
                 if s < THRESHOLD and w.strip(",.").lower() not in STOPWORDS
                 and len(w.strip(",.")) > 1]
# Function/stop words: only stopwords present in the prompt
func_words    = [w for w in words_raw
                 if w.strip(",.").lower() in STOPWORDS or w.strip() == ","]

print(f"\nNon-aesthetic content words: {non_aes_words}")
print(f"Function/stop words:         {func_words}")

non_aes_weights = build_t5_weights_for_words(non_aes_words, t5_ids_list, t5_tokenizer, N_txt)
func_weights    = build_t5_weights_for_words(func_words,    t5_ids_list, t5_tokenizer, N_txt)

hm_na = make_heatmap(avg_attn, weights=non_aes_weights)   # non-aesthetic content
hm_fn = make_heatmap(avg_attn, weights=func_weights)      # function words

# ── Build binary masks ────────────────────────────────────────────────────────
def to_binary_mask(hm, ratio=RATIO):
    thr = np.quantile(hm, 1.0 - ratio)
    return (hm >= thr).astype(np.float32)

mask_u  = to_binary_mask(hm_u)
mask_a  = to_binary_mask(hm_a)
mask_na = to_binary_mask(hm_na)
mask_fn = to_binary_mask(hm_fn)

# ── Rendering helpers ─────────────────────────────────────────────────────────
VIZ_SIZE = 512

def render_heatmap(hm, size=VIZ_SIZE):
    cm  = plt.get_cmap("jet")
    rgb = (cm(hm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)

def render_mask_overlay(base, mask_np, size=VIZ_SIZE, alpha=0.45):
    base_r = np.array(base.resize((size, size))).astype(float)
    m_img  = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    m_np   = np.array(m_img).astype(float) / 255.0
    fg_col = np.array([255,  80,  80], dtype=float)
    bg_col = np.array([ 80, 120, 255], dtype=float)
    overlay = m_np[:, :, None] * fg_col + (1 - m_np[:, :, None]) * bg_col
    result  = base_r * (1 - alpha) + overlay * alpha
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))

def render_clean_mask(mask_np, size=VIZ_SIZE):
    """White fg, dark grey bg — clean binary mask."""
    m = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    m_np = np.array(m).astype(float) / 255.0
    canvas = np.ones((size, size, 3), dtype=float) * 255
    canvas[m_np < 0.5] = 50
    return Image.fromarray(canvas.astype(np.uint8))

# ── Save individual figures ───────────────────────────────────────────────────
renders = {
    "heatmap_uniform":       render_heatmap(hm_u),
    "heatmap_anchor":        render_heatmap(hm_a),
    "heatmap_nonaesthetic":  render_heatmap(hm_na),
    "heatmap_funcword":      render_heatmap(hm_fn),
    "mask_overlay_uniform":      render_mask_overlay(sasd_img, mask_u),
    "mask_overlay_anchor":       render_mask_overlay(sasd_img, mask_a),
    "mask_overlay_nonaesthetic": render_mask_overlay(sasd_img, mask_na),
    "mask_overlay_funcword":     render_mask_overlay(sasd_img, mask_fn),
    "mask_clean_uniform":        render_clean_mask(mask_u),
    "mask_clean_anchor":         render_clean_mask(mask_a),
    "mask_clean_nonaesthetic":   render_clean_mask(mask_na),
    "mask_clean_funcword":       render_clean_mask(mask_fn),
}
for fname, img_out in renders.items():
    img_out.save(f"{OUT_DIR}/{fname}.png")
    print(f"saved {fname}.png")

# ── Token weight bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
colors = ["#e74c3c" if s >= THRESHOLD else "#95a5a6" for s in sim_np]
bars = ax.bar(range(len(words_raw)), sim_np, color=colors, width=0.7)
ax.set_xticks(range(len(words_raw)))
ax.set_xticklabels(
    [w.replace(",", "") for w in words_raw],
    rotation=40, ha="right", fontsize=9
)
ax.axhline(THRESHOLD, color="#e74c3c", ls="--", lw=1.2, label=f"threshold {THRESHOLD}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("CLIP cosine similarity\nwith aesthetic anchors", fontsize=9)
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/token_weights.png", dpi=150)
plt.close(fig)
print("saved token_weights.png")

print(f"\nDone. All figures in {OUT_DIR}/")
print("  heatmap_uniform.png   heatmap_anchor.png")
print("  mask_overlay_uniform  mask_overlay_anchor")
print("  mask_clean_uniform    mask_clean_anchor")
print("  token_weights.png")
