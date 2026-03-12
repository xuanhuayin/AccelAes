#!/usr/bin/env python3
"""
Evaluate generated images using the metrics reported in the AccelAes paper:
  - CLIP Score      (text-image alignment)
  - ImageReward     (human preference)
  - HPSv2           (human preference score v2)
  - Aesthetic Score (LAION aesthetic predictor)
  - Edge Density    (structural detail proxy)
  - PickScore       (pairwise human preference)

Usage:
    python scripts/eval_metrics.py \
        --image_dir outputs/my_run \
        --prompts prompts/prompts_dev.txt \
        --output_json outputs/my_run/scores.json
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_images_and_prompts(image_dir: str, prompts_file: str):
    prompt_lines = Path(prompts_file).read_text().splitlines()
    prompts = [p.strip() for p in prompt_lines if p.strip()]

    images, used_prompts = [], []
    for i, prompt in enumerate(prompts):
        path = os.path.join(image_dir, f"p{i:04d}_s0000.png")
        if not os.path.exists(path):
            path = os.path.join(image_dir, f"p{i:02d}.png")
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
            used_prompts.append(prompt)

    print(f"  Loaded {len(images)} images from {image_dir}")
    return images, used_prompts


# ── CLIP Score ────────────────────────────────────────────────────────────────

def compute_clip_scores(images, prompts, device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    scores = []
    import torch
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            t = preprocess(img).unsqueeze(0).to(device)
            tok = tokenizer([prompt]).to(device)
            f_img = model.encode_image(t)
            f_txt = model.encode_text(tok)
            f_img = f_img / f_img.norm(dim=-1, keepdim=True)
            f_txt = f_txt / f_txt.norm(dim=-1, keepdim=True)
            scores.append((f_img @ f_txt.T).item())
    return scores


# ── ImageReward ───────────────────────────────────────────────────────────────

def compute_image_reward(images, prompts, device="cuda"):
    import ImageReward as ir
    model = ir.load("ImageReward-v1.0", device=device)
    scores = []
    for img, prompt in zip(images, prompts):
        scores.append(model.score(prompt, img))
    return scores


# ── HPSv2 ─────────────────────────────────────────────────────────────────────

def compute_hps(images, prompts, device="cuda"):
    import hpsv2
    scores = []
    for img, prompt in zip(images, prompts):
        score = hpsv2.score(img, prompt, hps_version="v2.1")[0]
        scores.append(float(score))
    return scores


# ── LAION Aesthetic Score ─────────────────────────────────────────────────────

def compute_aesthetic(images, device="cuda"):
    import torch
    import clip
    from aesthetic_predictor import AestheticPredictor

    predictor = AestheticPredictor()
    predictor.eval().to(device)

    _, preprocess = clip.load("ViT-L/14", device=device)
    model_clip, _ = clip.load("ViT-L/14", device=device)

    scores = []
    with torch.no_grad():
        for img in images:
            t = preprocess(img).unsqueeze(0).to(device)
            features = model_clip.encode_image(t)
            features = features / features.norm(dim=-1, keepdim=True)
            score = predictor(features).item()
            scores.append(score)
    return scores


# ── Edge Density ──────────────────────────────────────────────────────────────

def compute_edge_density(images):
    from scipy.ndimage import convolve
    sx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
    sy = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
    scores = []
    for img in images:
        gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
        mag = np.sqrt(convolve(gray, sx) ** 2 + convolve(gray, sy) ** 2)
        scores.append(float(mag.mean()))
    return scores


# ── PickScore ─────────────────────────────────────────────────────────────────

def compute_pick_score(images, prompts, device="cuda"):
    import torch
    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    scores = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            inp = processor(
                text=prompt, images=img,
                return_tensors="pt", padding=True
            ).to(device)
            logits = model(**inp).logits_per_image
            scores.append(logits.item())
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--prompts", default="prompts/prompts_dev.txt")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--metrics", nargs="+",
        default=["clip", "ir", "hps", "aesthetic", "edge", "pickscore"],
        choices=["clip", "ir", "hps", "aesthetic", "edge", "pickscore"],
    )
    args = parser.parse_args()

    images, prompts = load_images_and_prompts(args.image_dir, args.prompts)
    if not images:
        print("No images found. Check --image_dir and --prompts.")
        sys.exit(1)

    results = {}

    if "clip" in args.metrics:
        print("Computing CLIP Score...")
        s = compute_clip_scores(images, prompts, args.device)
        results["clip_score"] = float(np.mean(s))
        print(f"  CLIP Score: {results['clip_score']:.4f}")

    if "ir" in args.metrics:
        print("Computing ImageReward...")
        s = compute_image_reward(images, prompts, args.device)
        results["image_reward"] = float(np.mean(s))
        print(f"  ImageReward: {results['image_reward']:.4f}")

    if "hps" in args.metrics:
        print("Computing HPSv2...")
        s = compute_hps(images, prompts, args.device)
        results["hpsv2"] = float(np.mean(s))
        print(f"  HPSv2: {results['hpsv2']:.4f}")

    if "aesthetic" in args.metrics:
        print("Computing Aesthetic Score...")
        s = compute_aesthetic(images, args.device)
        results["aesthetic_score"] = float(np.mean(s))
        print(f"  Aesthetic Score: {results['aesthetic_score']:.4f}")

    if "edge" in args.metrics:
        print("Computing Edge Density...")
        s = compute_edge_density(images)
        results["edge_density"] = float(np.mean(s))
        print(f"  Edge Density: {results['edge_density']:.4f}")

    if "pickscore" in args.metrics:
        print("Computing PickScore...")
        s = compute_pick_score(images, prompts, args.device)
        results["pick_score"] = float(np.mean(s))
        print(f"  PickScore: {results['pick_score']:.4f}")

    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
