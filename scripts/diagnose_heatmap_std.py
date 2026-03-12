#!/usr/bin/env python3
"""Measure heatmap std before thresholding to calibrate MIN_HEATMAP_STD."""

import sys, os
sys.path.insert(0, "/home/runkai/xuanhua/SASD")

import torch
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers",
                                     subfolder="tokenizer")

with open("prompts/prompts_dev.txt") as f:
    prompts = [l.strip() for l in f if l.strip()]

from src.sparse.mask_builders import SemanticMaskBuilder
from src.models.dit_wrapper import LuminaDiTWrapper

wrapper = LuminaDiTWrapper(model_name="Alpha-VLLM/Lumina-Next-SFT-diffusers")

builder = SemanticMaskBuilder(ratio=0.25, blur_sigma=1.0)
builder.compute_anchor_embeddings(device="cuda")

# Monkey-patch to capture heatmap std
heatmap_std_log = []
_orig = SemanticMaskBuilder.build_mask_from_cross_attention

def patched_build(self, cross_attn_maps, token_importance, latent_h, latent_w):
    num_patches = latent_h * latent_w
    all_attn = torch.stack([m.mean(dim=0) for m in cross_attn_maps])
    avg_attn = all_attn.mean(dim=0)
    weighted = avg_attn * token_importance.unsqueeze(0)
    heatmap = weighted.sum(dim=-1)
    heatmap = heatmap[:num_patches].reshape(latent_h, latent_w)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    std_val = heatmap.std().item()
    heatmap_std_log.append(std_val)
    # Build mask normally (bypass quality check)
    threshold = torch.quantile(heatmap.flatten(), 1.0 - self.ratio)
    mask = (heatmap >= threshold).float()
    if self.blur_sigma > 0:
        mask = self._gaussian_blur(mask, sigma=self.blur_sigma)
    return mask

SemanticMaskBuilder.build_mask_from_cross_attention = patched_build

print(f"{'#':>3} | {'Prompt':50s} | {'HeatmapStd':>10} | {'MaskStd':>8} | {'Quality':>8}")
print("-" * 95)

for pi, prompt in enumerate(prompts[:20]):
    if not prompt or prompt == ",":
        continue

    heatmap_std_log.clear()

    image = wrapper.generate_with_auto_mask(
        prompt=prompt,
        seed=0,
        mask_builder=builder,
        s_fg=5.0, s_bg=3.5,
    )

    h_std = heatmap_std_log[0] if heatmap_std_log else -1

    # Read saved mask for comparison
    mask_path = f"outputs/debug_sem_r25/mask_visualizations/p{pi:04d}_s0000_mask.png"
    if os.path.exists(mask_path):
        from PIL import Image
        m = np.array(Image.open(mask_path).convert("L")).astype(float) / 255.0
        m_std = m.std()
    else:
        m_std = -1

    quality = "GOOD" if h_std >= 0.20 else ("WEAK" if h_std >= 0.12 else "SKIP")
    print(f"{pi:3d} | {prompt[:50]:50s} | {h_std:10.4f} | {m_std:8.4f} | {quality:>8}")
