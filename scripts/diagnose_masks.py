#!/usr/bin/env python3
"""Diagnose mask quality: log anchor confidence per prompt."""

import sys, os
sys.path.insert(0, "/home/runkai/xuanhua/SASD")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers",
                                     subfolder="tokenizer")

with open("prompts/prompts_dev.txt") as f:
    prompts = [l.strip() for l in f if l.strip()]

from src.sparse.mask_builders import SemanticMaskBuilder
builder = SemanticMaskBuilder(ratio=0.25, blur_sigma=1.0)
builder.compute_anchor_embeddings(device="cuda")

print(f"{'#':>3} | {'Prompt':50s} | {'Best':>6} | {'Anch?':>5} | {'Top Word→Anchor':35s} | {'#Cntnt':>6}")
print("-" * 125)

anchor_count = 0
total_count = 0

for pi, prompt in enumerate(prompts[:50]):
    if not prompt or prompt == ",":
        continue
    total_count += 1

    encoded = tok(prompt, return_tensors="pt", padding="max_length", max_length=256)
    input_ids = encoded["input_ids"]       # (1, seq_len)
    prompt_mask = encoded["attention_mask"] # (1, seq_len)
    ids = input_ids[0]

    # Group into words
    words, word_indices = builder._group_tokens_to_words(ids, tok)

    # Identify stopwords
    is_stopword = [
        w.strip(",.;:!?'\"()[]{}").lower() in builder.STOPWORDS
        or len(w.strip(",.;:!?'\"()[]{}")) <= 1
        for w in words
    ]
    content_word_idx = [i for i, stop in enumerate(is_stopword) if not stop]
    n_content = len(content_word_idx)

    if content_word_idx:
        content_words = [words[i] for i in content_word_idx]
        with torch.no_grad():
            clip_toks = builder._clip_tokenizer(
                content_words, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            clip_out = builder._clip_encoder(**clip_toks)
            clip_embeds = clip_out.last_hidden_state.float()
            clip_content = clip_toks["attention_mask"].clone()
            for sid in (builder._clip_tokenizer.bos_token_id,
                        builder._clip_tokenizer.eos_token_id):
                if sid is not None:
                    clip_content[clip_toks["input_ids"] == sid] = 0
            cmask = clip_content.unsqueeze(-1).float()
            cmask_sum = cmask.sum(dim=1).clamp(min=1.0)
            word_pooled = (clip_embeds * cmask).sum(dim=1) / cmask_sum
            word_pooled = F.normalize(word_pooled, dim=-1)
            sim = word_pooled @ builder._anchor_embeddings.T
            cw_importance = sim.max(dim=-1).values
            best_match = cw_importance.max().item()

            top_idx = cw_importance.argmax().item()
            top_word = content_words[top_idx]
            anchor_idx = sim[top_idx].argmax().item()
            top_anchor = builder.DEFAULT_ANCHORS[anchor_idx]
            anchored = best_match >= builder.confidence_threshold
            if anchored:
                anchor_count += 1
    else:
        best_match = 0.0
        top_word = "N/A"
        top_anchor = "N/A"
        anchored = False

    top_info = f"{top_word}→{top_anchor}" if anchored else f"({top_word}, below)"
    print(f"{pi:3d} | {prompt[:50]:50s} | {best_match:6.3f} | {'YES' if anchored else 'no':>5} | {top_info:35s} | {n_content:>6}")

print(f"\nAnchored: {anchor_count}/{total_count} ({anchor_count/total_count*100:.0f}%)")
print(f"Confidence threshold: {builder.confidence_threshold}")

# Mask stats from saved debug runs
print("\n\n=== Mask Statistics (first 10 prompts) ===")
print(f"{'#':>3} | {'Prompt':45s} | {'Sem r25':>8} {'std':>6} | {'Sem r40':>8} {'std':>6} | {'Cplx r25':>8} {'std':>6}")
print("-" * 110)

for pi in range(10):
    prompt_short = prompts[pi][:45] if pi < len(prompts) else "?"
    parts = []
    for exp in ["debug_sem_r25", "debug_sem_r40", "debug_cplx_r25"]:
        mask_path = f"outputs/{exp}/mask_visualizations/p{pi:04d}_s0000_mask.png"
        if os.path.exists(mask_path):
            m = np.array(Image.open(mask_path).convert("L")).astype(float) / 255.0
            fg = (m > 0.5).mean()
            std = m.std()
            parts.append(f"{fg:7.0%} {std:6.3f}")
        else:
            parts.append(f"{'N/A':>7} {'N/A':>6}")
    print(f"{pi:3d} | {prompt_short:45s} | {parts[0]} | {parts[1]} | {parts[2]}")
