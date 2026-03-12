"""
SD3 Semantic Mask Builder.

Builds a foreground mask from SD3's joint attention image→text affinity maps.

Pipeline:
  1. Aggregate affinity maps across hooked layers and attention heads.
     → per-patch importance: (N_img,) = mean over layers, heads, text tokens
  2. Optionally weight text tokens by CLIP-L importance (semantic reweighting).
  3. Reshape to (patch_h, patch_w), apply SLIC region segmentation.
  4. Select top-ratio regions as foreground.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic

from src.sparse.boundary_ops import soft_mask_pipeline


class SD3SemanticMaskBuilder:
    """
    Semantic foreground mask for SD3 MMDiT via joint attention affinity.

    Args:
        ratio: fraction of patches marked as foreground.
        n_segments: SLIC target segment count.
        compactness: SLIC compactness (lower = follow affinity contours more).
        blur_sigma: soft mask edge blur.
        use_clip_weights: if True, weight text tokens by CLIP importance before
            aggregation (requires prompt and CLIP model at build time).
        region_method: "threshold" (direct percentile, default) or "slic"
            (superpixel grouping). "threshold" is faster and more accurate.
    """

    def __init__(
        self,
        ratio: float = 0.5,
        n_segments: int = 64,
        compactness: float = 0.1,
        blur_sigma: float = 1.5,
        use_clip_weights: bool = False,
        region_method: str = "threshold",
    ):
        self.ratio = ratio
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_sigma = blur_sigma
        self.use_clip_weights = use_clip_weights
        self.region_method = region_method

    def build_mask(
        self,
        affinity_maps: list,
        patch_h: int,
        patch_w: int,
        clip_token_weights=None,
    ) -> torch.Tensor:
        """
        Build (patch_h, patch_w) foreground mask from joint attention affinities.

        Args:
            affinity_maps: list of (B, heads, N_img, N_txt) tensors from SD3JointAttnHookManager.
                           N_img = patch_h * patch_w.  B=2 (uncond+cond); we use the cond half.
            patch_h, patch_w: spatial dimensions in patch space.
            clip_token_weights: optional (N_txt,) tensor of per-token importance weights.
                                If None, all text tokens are weighted equally.

        Returns:
            mask: (patch_h, patch_w) float tensor, 1=foreground, 0=background.
        """
        if not affinity_maps:
            # Fallback: all foreground
            return torch.ones(patch_h, patch_w)

        # Stack layers: (n_layers, B, heads, N_img, N_txt)
        stacked = torch.stack(affinity_maps, dim=0)

        # Use only the cond half (last half of batch)
        B = stacked.shape[1]
        stacked = stacked[:, B // 2:, :, :, :]  # (n_layers, 1, heads, N_img, N_txt)

        # Mean over layers and heads → (N_img, N_txt)
        importance = stacked.mean(dim=[0, 1, 2])  # (N_img, N_txt)

        # Optionally weight text tokens by CLIP importance
        if clip_token_weights is not None:
            w = clip_token_weights.to(importance.device, dtype=importance.dtype)
            w = w / (w.sum() + 1e-8)
            importance = (importance * w.unsqueeze(0)).sum(dim=-1)  # (N_img,)
        else:
            importance = importance.mean(dim=-1)  # (N_img,) — mean over text tokens

        # Reshape to spatial
        importance = importance.reshape(patch_h, patch_w)

        imp_np = importance.cpu().float().numpy()
        imin, imax = imp_np.min(), imp_np.max()
        if imax - imin > 1e-8:
            imp_norm = (imp_np - imin) / (imax - imin)
        else:
            imp_norm = np.ones_like(imp_np) * 0.5

        if self.region_method == "slic":
            # SLIC superpixel segmentation
            imp_3ch = np.stack([imp_norm, imp_norm, imp_norm], axis=-1)
            labels = slic(
                imp_3ch,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=1.0,
                start_label=0,
            )

            region_scores = {}
            for lbl in np.unique(labels):
                region_scores[lbl] = imp_np[labels == lbl].mean()

            sorted_lbls = sorted(region_scores, key=region_scores.get, reverse=True)
            n_fg = max(1, int(len(sorted_lbls) * self.ratio))
            fg_set = set(sorted_lbls[:n_fg])

            mask_np = np.zeros_like(imp_np, dtype=np.float32)
            for lbl in fg_set:
                mask_np[labels == lbl] = 1.0
        else:
            # Direct percentile threshold (faster, preserves true affinity distribution)
            thresh = np.percentile(imp_np, (1.0 - self.ratio) * 100)
            mask_np = (imp_np >= thresh).astype(np.float32)

        device = affinity_maps[0].device
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)

        if self.blur_sigma > 0:
            mask = soft_mask_pipeline(mask, dilation_radius=0, blur_sigma=self.blur_sigma)

        return mask
