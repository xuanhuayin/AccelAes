"""
SLIC superpixel region mask builder (Phase 4).

Replaces fixed grid blocks with connected superpixel regions for more
semantically coherent foreground/background segmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic

from .boundary_ops import soft_mask_pipeline


class RegionMaskBuilder:
    """
    SLIC superpixel-based complexity mask.

    Pipeline:
      1. Compute per-pixel complexity via Scharr gradients (same as ComplexityMaskBuilder)
      2. Segment latent into superpixel regions via SLIC
      3. Score each region by top-q% mean complexity
      4. Select top regions by complexity (ratio fraction = foreground)
      5. Optionally apply boundary smoothing (dilation + blur)
    """

    def __init__(
        self,
        n_segments: int = 64,
        compactness: float = 10.0,
        top_q: float = 0.75,
        ratio: float = 0.25,
        dilation_radius: int = 2,
        blur_sigma: float = 1.5,
    ):
        """
        Args:
            n_segments: target number of SLIC superpixels.
            compactness: SLIC compactness (higher = more regular shapes).
            top_q: quantile for region complexity scoring (top-q% pixels).
            ratio: fraction of regions to mark as foreground.
            dilation_radius: morphological dilation radius for boundary smoothing (0=off).
            blur_sigma: Gaussian blur sigma for boundary smoothing (0=off).
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.top_q = top_q
        self.ratio = ratio
        self.dilation_radius = dilation_radius
        self.blur_sigma = blur_sigma

    def compute_complexity(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Per-pixel complexity via Scharr gradient magnitude.

        Args:
            latent: (B, C, H, W) tensor.

        Returns:
            (B, 1, H, W) gradient magnitude map.
        """
        gray = latent.mean(dim=1, keepdim=True)

        scharr_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
            dtype=latent.dtype, device=latent.device,
        ).reshape(1, 1, 3, 3)
        scharr_y = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
            dtype=latent.dtype, device=latent.device,
        ).reshape(1, 1, 3, 3)

        gx = F.conv2d(gray, scharr_x, padding=1)
        gy = F.conv2d(gray, scharr_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def build_regions(self, complexity_map: np.ndarray) -> np.ndarray:
        """
        Segment the complexity map into superpixel regions using SLIC.

        Args:
            complexity_map: (H, W) numpy array of complexity values.

        Returns:
            labels: (H, W) integer array of region labels.
        """
        # Normalize to [0, 1] for SLIC
        cmap = complexity_map.copy()
        cmin, cmax = cmap.min(), cmap.max()
        if cmax - cmin > 1e-8:
            cmap = (cmap - cmin) / (cmax - cmin)
        else:
            cmap = np.zeros_like(cmap)

        # SLIC expects (H, W) or (H, W, C); single-channel is fine
        labels = slic(
            cmap,
            n_segments=self.n_segments,
            compactness=self.compactness,
            channel_axis=None,  # grayscale
            start_label=0,
        )
        return labels

    def score_regions(
        self, complexity_map: np.ndarray, labels: np.ndarray
    ) -> dict:
        """
        Score each region by top-q% mean complexity.

        Args:
            complexity_map: (H, W) numpy array.
            labels: (H, W) integer region labels.

        Returns:
            Dict mapping region_id -> score.
        """
        scores = {}
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            pixels = complexity_map[labels == lbl]
            k = max(1, int(len(pixels) * self.top_q))
            # Top-q% mean: partition and take top k
            topk = np.partition(pixels, -k)[-k:]
            scores[lbl] = topk.mean()
        return scores

    def build_mask(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Build SLIC region-based complexity mask with optional boundary smoothing.

        Args:
            latent: (B, C, H, W) latent tensor.

        Returns:
            mask: (H, W) float tensor, 1=foreground, 0=background.
        """
        complexity = self.compute_complexity(latent)  # (B, 1, H, W)
        c_tensor = complexity[0, 0]  # (H, W)
        c_np = c_tensor.cpu().float().numpy()

        # SLIC segmentation (runs on CPU, ~5ms for 128x128)
        labels = self.build_regions(c_np)
        scores = self.score_regions(c_np, labels)

        # Select top-ratio regions as foreground
        sorted_labels = sorted(scores.keys(), key=lambda l: scores[l], reverse=True)
        num_fg = max(1, int(len(sorted_labels) * self.ratio))
        fg_labels = set(sorted_labels[:num_fg])

        # Build binary mask
        mask_np = np.zeros_like(c_np, dtype=np.float32)
        for lbl in fg_labels:
            mask_np[labels == lbl] = 1.0

        mask = torch.from_numpy(mask_np).to(device=latent.device, dtype=latent.dtype)

        # Boundary smoothing
        if self.dilation_radius > 0 or self.blur_sigma > 0:
            mask = soft_mask_pipeline(mask, self.dilation_radius, self.blur_sigma)

        return mask

    def build_mask_with_viz(self, latent: torch.Tensor) -> tuple:
        """Build mask and return (mask, complexity_map) for visualization."""
        complexity = self.compute_complexity(latent)
        c_tensor = complexity[0, 0]
        mask = self.build_mask(latent)
        return mask, c_tensor
