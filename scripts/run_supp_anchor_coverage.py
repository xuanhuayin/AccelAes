#!/usr/bin/env python3
"""
Supplementary: Anchor word coverage statistics.

For each prompt in the evaluation set, compute:
  - Whether any token exceeds the anchor similarity threshold (0.60)
  - How many anchor tokens are matched
  - Which anchor words are most frequently matched

Output: outputs/supp_sensitivity/anchor_coverage.json
"""

import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

CLIP_ID   = "openai/clip-vit-large-patch14"
THRESHOLD = 0.60
OUTPUT_BASE = "outputs/supp_sensitivity"

# 23 aesthetic anchors (same as AesMask)
# 34 anchors — exactly matching SemanticMaskBuilder.DEFAULT_ANCHORS in src/sparse/mask_builders.py
ANCHORS = [
    # 风格/渲染质量
    "photorealistic", "realistic", "cinematic", "highly detailed", "artistic",
    "masterpiece", "professional photography", "soft bokeh", "bokeh",
    "dramatic lighting", "fantasy art", "studio lighting",
    # 细节/清晰度
    "detailed", "intricate", "sharp focus", "sharp",
    # 审美判断
    "stunning", "beautiful", "elegant", "vivid", "vibrant",
    # 摄影/空间构图
    "depth of field", "volumetric lighting", "close up", "portrait", "full body",
    # 主体指称
    "main subject", "foreground", "focused",
    # 内容级兜底
    "subject", "character", "object", "figure", "person",
]

def main():
    # Load prompts (dev set + full set)
    prompt_files = {
        "dev":  ("prompts/prompts_dev.txt",  200),
        "all":  ("prompts/prompts_all.txt",  10000),
    }
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Load CLIP
    print("Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)
    encoder   = CLIPTextModel.from_pretrained(CLIP_ID).eval().to("cuda")

    # Encode anchors
    with torch.no_grad():
        anc_toks = tokenizer(ANCHORS, padding=True, truncation=True,
                             return_tensors="pt").to("cuda")
        anc_out  = encoder(**anc_toks)
        mask_a   = anc_toks["attention_mask"].float()
        for sid in (tokenizer.bos_token_id, tokenizer.eos_token_id):
            if sid is not None:
                mask_a[anc_toks["input_ids"] == sid] = 0
        anc_emb = (anc_out.last_hidden_state * mask_a.unsqueeze(-1)).sum(1)
        anc_emb = F.normalize(anc_emb / (mask_a.sum(1, keepdim=True) + 1e-8), dim=-1)

    results_all = {}

    for split, (fpath, max_n) in prompt_files.items():
        if not os.path.exists(fpath):
            print(f"  {fpath} not found, skipping {split}")
            continue

        with open(fpath) as f:
            prompts = [l.strip() for l in f if l.strip()][:max_n]
        print(f"\n{split}: {len(prompts)} prompts")

        n_triggered    = 0  # prompts with ≥1 anchor match
        token_counts   = []  # n matched anchor tokens per prompt
        anchor_hits    = {a: 0 for a in ANCHORS}  # per-anchor hit count
        max_sim_per_prompt = []

        for i, prompt in enumerate(prompts):
            words = prompt.replace(",", " ,").split()
            words = [w for w in words if w.strip()]
            if not words:
                continue

            with torch.no_grad():
                w_toks = tokenizer(words, padding=True, truncation=True,
                                   return_tensors="pt").to("cuda")
                w_out  = encoder(**w_toks)
                mask_w = w_toks["attention_mask"].float()
                for sid in (tokenizer.bos_token_id, tokenizer.eos_token_id):
                    if sid is not None:
                        mask_w[w_toks["input_ids"] == sid] = 0
                w_emb = (w_out.last_hidden_state * mask_w.unsqueeze(-1)).sum(1)
                w_emb = F.normalize(w_emb / (mask_w.sum(1, keepdim=True) + 1e-8), dim=-1)
                # (n_words, n_anchors)
                sim   = (w_emb @ anc_emb.T).cpu().float().numpy()

            # max similarity per word (across all anchors)
            word_max_sim = sim.max(axis=1)  # (n_words,)
            max_sim_per_prompt.append(float(word_max_sim.max()))

            matched_words = [w.strip(",.") for w, s in zip(words, word_max_sim)
                             if s >= THRESHOLD]
            n_matched = len(matched_words)
            token_counts.append(n_matched)

            if n_matched > 0:
                n_triggered += 1
                # per-anchor hits
                for ai, anchor in enumerate(ANCHORS):
                    if sim[:, ai].max() >= THRESHOLD:
                        anchor_hits[anchor] += 1

            if (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{len(prompts)}]  triggered so far: "
                      f"{n_triggered}/{i+1} ({n_triggered/(i+1)*100:.1f}%)")

        n_total  = len(token_counts)
        coverage = n_triggered / n_total * 100
        avg_tok  = np.mean(token_counts)
        avg_max  = np.mean(max_sim_per_prompt)

        print(f"\n  === {split.upper()} coverage ===")
        print(f"  Prompts with ≥1 anchor match (triggered): "
              f"{n_triggered}/{n_total} = {coverage:.1f}%")
        print(f"  Avg anchor tokens per prompt (triggered only): "
              f"{np.mean([x for x in token_counts if x > 0]):.2f}")
        print(f"  Avg anchor tokens per prompt (all):            {avg_tok:.2f}")
        print(f"  Avg max similarity per prompt:                 {avg_max:.3f}")

        # Top-10 most triggered anchors
        top_anchors = sorted(anchor_hits.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top-10 most triggered anchors:")
        for anchor, cnt in top_anchors:
            pct = cnt / n_triggered * 100 if n_triggered > 0 else 0
            print(f"    {anchor:<30} {cnt:4d} ({pct:.1f}% of triggered prompts)")

        results_all[split] = {
            "n_total":           n_total,
            "n_triggered":       n_triggered,
            "coverage_pct":      round(coverage, 2),
            "avg_matched_tokens_all":       round(avg_tok, 3),
            "avg_matched_tokens_triggered": round(
                float(np.mean([x for x in token_counts if x > 0])), 3
            ) if n_triggered > 0 else 0,
            "avg_max_similarity": round(avg_max, 4),
            "anchor_hit_counts": {a: int(anchor_hits[a]) for a in ANCHORS},
            "top10_anchors": [(a, int(c)) for a, c in top_anchors],
        }

    out_path = os.path.join(OUTPUT_BASE, "anchor_coverage.json")
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")

    encoder.to("cpu")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
