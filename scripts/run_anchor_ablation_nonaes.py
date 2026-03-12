#!/usr/bin/env python3
"""
Add the third anchor-ablation variant: non-aesthetic token weighting.
Inverts CLIP importance: content words with LOW anchor similarity get HIGH weight.
Then scores all 3 configs and draws the comparison bar chart.
"""
import os, sys, time, json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

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

# ── Non-aesthetic patch: invert CLIP importance ─────────────────────────────
_orig_compute = _mb.SemanticMaskBuilder.compute_token_importance

def _nonaesthetic_token_importance(self, prompt, prompt_mask, input_ids, tokenizer):
    """High weight for content words with LOW anchor similarity (non-aesthetic)."""
    importance = _orig_compute(self, prompt, prompt_mask, input_ids, tokenizer)
    # importance has high values for anchor words, low/zero for others.
    # Invert: non-anchor content words get weight 1.0, anchor words get 0.02.
    seq_len = prompt_mask.shape[1]
    ids = input_ids[0]
    special_ids = self._get_special_ids(tokenizer)

    # Build content mask (non-special)
    content = torch.zeros(seq_len, device=importance.device)
    for i in range(len(ids)):
        if ids[i].item() not in special_ids:
            content[i] = 1.0

    # Positions where original importance > 0.5 → these are anchor tokens
    anchor_pos = importance > 0.5

    # Non-anchor content tokens: content=1 AND NOT anchor → weight 1.0
    # Anchor tokens → weight 0.02; special/pad → 0
    inv = torch.where(anchor_pos, torch.full_like(importance, 0.02), content)
    return inv

# ── Helpers ──────────────────────────────────────────────────────────────────
def run_config(wrapper, name, patch_fn=None):
    img_dir = os.path.join(OUT_BASE, name, "images")
    os.makedirs(img_dir, exist_ok=True)
    times, done, skipped = [], 0, 0
    total = len(prompts) * len(SEEDS)

    if patch_fn is not None:
        _mb.SemanticMaskBuilder.compute_token_importance = patch_fn
    else:
        _mb.SemanticMaskBuilder.compute_token_importance = _orig_compute

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
                print(f"  [{done}/{total}]  avg={sum(times)/len(times):.2f}s", flush=True)

    _mb.SemanticMaskBuilder.compute_token_importance = _orig_compute
    avg_t = sum(times)/len(times) if times else None
    print(f"  Done: {done} gen, {skipped} skipped" +
          (f"  avg={avg_t:.2f}s" if avg_t else ""))
    return avg_t

# ── Generate ──────────────────────────────────────────────────────────────────
from src.models.dit_wrapper import LuminaDiTWrapper
print("Loading Lumina-Next-T2I...")
wrapper = LuminaDiTWrapper()

run_config(wrapper, "sasd_nonaesthetic", patch_fn=_nonaesthetic_token_importance)

print("\nGeneration done. Now scoring...")

# ── Score all 3 configs ───────────────────────────────────────────────────────
from PIL import Image

class PickScoreEval:
    def __init__(self):
        from transformers import AutoProcessor, AutoModel
        self.proc  = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to("cuda")
    @torch.no_grad()
    def score(self, img, prompt):
        inp = self.proc(text=[prompt], images=[img], return_tensors="pt", padding=True).to("cuda")
        out = self.model(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        return (ie @ te.T).item()

class IREval:
    def __init__(self):
        import ImageReward as ir_module
        self.model = ir_module.load("ImageReward-v1.0", device="cuda")
    @torch.no_grad()
    def score(self, img, prompt):
        return float(self.model.score(prompt, img))

class AestheticEval:
    def __init__(self):
        import sys
        sys.path.insert(0, "scripts")
        # inline from run_metrics_only.py
        import importlib.util, types
        spec = importlib.util.spec_from_file_location("_rm", "scripts/run_metrics_only.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.model = mod.AestheticScoreEvaluator(device="cuda")
    def score(self, img):
        return float(self.model.score([img])[0])

class CLIPEval:
    def __init__(self):
        from src.eval.metrics import compute_clip_score
        self.fn = compute_clip_score
    def score(self, img, prompt):
        return float(self.fn([img], [prompt])[0])

print("Loading scorers...")
ir_eval  = IREval()
ps_eval  = PickScoreEval()
ae_eval  = AestheticEval()
cl_eval  = CLIPEval()

configs = ["sasd_with_anchor", "sasd_no_anchor", "sasd_nonaesthetic"]
results = {}

for cfg in configs:
    img_dir = os.path.join(OUT_BASE, cfg, "images")
    if not os.path.isdir(img_dir):
        print(f"  SKIP {cfg} — no images dir")
        continue
    irs, pss, aes, cls_ = [], [], [], []
    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(SEEDS):
            path = os.path.join(img_dir, f"p{pi:04d}_s{si:04d}.png")
            if not os.path.exists(path): continue
            img = Image.open(path).convert("RGB")
            irs.append(ir_eval.score(img, prompt))
            pss.append(ps_eval.score(img, prompt))
            aes.append(ae_eval.score(img))
            cls_.append(cl_eval.score(img, prompt))
    results[cfg] = dict(
        ir=np.mean(irs), pick=np.mean(pss),
        aesthetic=np.mean(aes), clip=np.mean(cls_),
    )
    print(f"  {cfg}: IR={results[cfg]['ir']:.3f}  Pick={results[cfg]['pick']:.4f}"
          f"  Aes={results[cfg]['aesthetic']:.3f}  CLIP={results[cfg]['clip']:.4f}")

# Save
score_path = os.path.join(OUT_BASE, "scores_3way.json")
with open(score_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nScores saved to {score_path}")
