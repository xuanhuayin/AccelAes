"""
Evaluation metrics for image generation quality.

Implements:
  - CLIPScore: text-image alignment
  - Edge density: background hallucination proxy (Canny/Scharr)
"""

import os
import csv
import torch
import numpy as np
from PIL import Image


def compute_clip_score(images: list, prompts: list, model_name: str = "ViT-L-14", device: str = "cuda") -> list:
    """
    Compute CLIP score for each (image, prompt) pair.

    Args:
        images: list of PIL.Image
        prompts: list of str
        model_name: OpenCLIP model name
        device: torch device

    Returns:
        list of float scores
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    scores = []
    for img, prompt in zip(images, prompts):
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        text_tokens = tokenizer([prompt]).to(device)

        with torch.no_grad():
            img_features = model.encode_image(img_tensor)
            text_features = model.encode_text(text_tokens)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = (img_features @ text_features.T).item()

        scores.append(score)

    return scores


def compute_edge_density(image: Image.Image, region_mask: np.ndarray = None) -> float:
    """
    Compute edge density as a proxy for background hallucination.

    Uses Scharr operator on grayscale image.

    Args:
        image: PIL Image
        region_mask: optional binary mask (H, W) to restrict computation.
                     If None, uses full image.

    Returns:
        float: mean edge magnitude in the region.
    """
    from scipy.ndimage import convolve

    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0

    # Scharr kernels
    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
    scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)

    gx = convolve(gray, scharr_x)
    gy = convolve(gray, scharr_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    if region_mask is not None:
        region_mask = region_mask.astype(bool)
        if region_mask.sum() == 0:
            return 0.0
        return float(magnitude[region_mask].mean())
    return float(magnitude.mean())


def save_metrics_csv(records: list, path: str):
    """
    Save metric records to CSV.

    Args:
        records: list of dicts with metric values.
        path: output CSV path.
    """
    if not records:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
