"""
SDiT-style and CFG-magnitude mask builders.

Implements two training-free mask strategies:
  1. SDiTMaskBuilder: gradient complexity on predicted clean latent with
     SLIC region segmentation (not grid). Follows the SDiT paper's
     edge + Laplacian + residual complexity score.
  2. CFGMagnitudeMaskBuilder: uses |cond_pred - uncond_pred| as importance.
     This signal is completely free since CFG already computes both branches.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic

from src.sparse.boundary_ops import soft_mask_pipeline


class SDiTMaskBuilder:
    """
    SDiT-style region-adaptive mask using predicted clean latent.

    Complexity = alpha * edge_strength + beta * |Laplacian| + gamma * residual_noise
    Regions via SLIC superpixels (not grid blocks).

    Args:
        n_segments: SLIC target superpixel count.
        compactness: SLIC compactness.
        ratio: fraction of regions marked as foreground.
        top_q: quantile for region scoring.
        alpha, beta, gamma: weights for edge, Laplacian, residual terms.
        blur_sigma: Gaussian blur for soft mask edges.
    """

    def __init__(
        self,
        n_segments: int = 64,
        compactness: float = 10.0,
        ratio: float = 0.5,
        top_q: float = 0.75,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        blur_sigma: float = 1.5,
    ):
        self.n_segments = n_segments
        self.compactness = compactness
        self.ratio = ratio
        self.top_q = top_q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.blur_sigma = blur_sigma

    def compute_complexity(
        self,
        latent: torch.Tensor,
        noise_pred: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        SDiT complexity score on predicted clean latent.

        x_pred = x_t - sigma * noise_pred

        Args:
            latent: (B, C, H, W) noisy latent x_t.
            noise_pred: (B, C, H, W) predicted noise (after CFG, negated).
            sigma: current noise level.

        Returns:
            (H, W) complexity map.
        """
        # Predicted clean latent
        x_pred = latent + sigma * noise_pred  # noise_pred is already negated in loop

        # Average across channels
        gray = x_pred[0].mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

        # Edge strength (Scharr)
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
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)[0, 0]  # (H, W)

        # Laplacian
        laplacian_k = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=latent.dtype, device=latent.device,
        ).reshape(1, 1, 3, 3)
        lap = F.conv2d(gray, laplacian_k, padding=1).abs()[0, 0]  # (H, W)

        # Residual noise magnitude
        residual = (latent[0] - x_pred[0]).abs().mean(dim=0)  # (H, W)

        # Normalize each component to [0, 1]
        def norm01(t):
            tmin, tmax = t.min(), t.max()
            return (t - tmin) / (tmax - tmin + 1e-8)

        complexity = (
            self.alpha * norm01(edge)
            + self.beta * norm01(lap)
            + self.gamma * norm01(residual)
        )
        return complexity

    def build_mask(
        self,
        latent: torch.Tensor,
        noise_pred: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        Build SDiT-style region mask.

        Returns:
            mask: (H, W) float tensor, 1=foreground, 0=background.
        """
        complexity = self.compute_complexity(latent, noise_pred, sigma)
        c_np = complexity.cpu().float().numpy()

        # SLIC region segmentation
        c_norm = c_np.copy()
        cmin, cmax = c_norm.min(), c_norm.max()
        if cmax - cmin > 1e-8:
            c_norm = (c_norm - cmin) / (cmax - cmin)
        labels = slic(
            c_norm, n_segments=self.n_segments,
            compactness=self.compactness, channel_axis=None, start_label=0,
        )

        # Score regions by top-q% complexity
        scores = {}
        for lbl in np.unique(labels):
            pixels = c_np[labels == lbl]
            k = max(1, int(len(pixels) * self.top_q))
            topk = np.partition(pixels, -k)[-k:]
            scores[lbl] = topk.mean()

        # Select top regions as foreground
        sorted_labels = sorted(scores, key=lambda l: scores[l], reverse=True)
        n_fg = max(1, int(len(sorted_labels) * self.ratio))
        fg_labels = set(sorted_labels[:n_fg])

        mask_np = np.zeros_like(c_np, dtype=np.float32)
        for lbl in fg_labels:
            mask_np[labels == lbl] = 1.0

        mask = torch.from_numpy(mask_np).to(device=latent.device, dtype=latent.dtype)

        if self.blur_sigma > 0:
            mask = soft_mask_pipeline(mask, dilation_radius=0, blur_sigma=self.blur_sigma)

        return mask


class CFGMagnitudeMaskBuilder:
    """
    CFG-magnitude mask: uses |cond_pred - uncond_pred| as importance.

    Regions where CFG makes a big difference are prompt-relevant (foreground).
    This signal is completely free since CFG already computes both branches.

    Uses SLIC region segmentation for semantic boundaries.
    """

    def __init__(
        self,
        n_segments: int = 64,
        compactness: float = 10.0,
        ratio: float = 0.5,
        top_q: float = 0.75,
        blur_sigma: float = 1.5,
    ):
        self.n_segments = n_segments
        self.compactness = compactness
        self.ratio = ratio
        self.top_q = top_q
        self.blur_sigma = blur_sigma

    def build_mask(
        self,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build mask from CFG magnitude.

        Args:
            cond_pred: (1, C, H, W) conditional noise prediction.
            uncond_pred: (1, C, H, W) unconditional noise prediction.

        Returns:
            mask: (H, W) float tensor, 1=foreground, 0=background.
        """
        # Per-pixel CFG magnitude (averaged across channels)
        magnitude = (cond_pred - uncond_pred).abs().mean(dim=[0, 1])  # (H, W)
        mag_np = magnitude.cpu().float().numpy()

        # SLIC region segmentation
        m_norm = mag_np.copy()
        mmin, mmax = m_norm.min(), m_norm.max()
        if mmax - mmin > 1e-8:
            m_norm = (m_norm - mmin) / (mmax - mmin)
        labels = slic(
            m_norm, n_segments=self.n_segments,
            compactness=self.compactness, channel_axis=None, start_label=0,
        )

        # Score regions
        scores = {}
        for lbl in np.unique(labels):
            pixels = mag_np[labels == lbl]
            k = max(1, int(len(pixels) * self.top_q))
            topk = np.partition(pixels, -k)[-k:]
            scores[lbl] = topk.mean()

        # Select top regions as foreground
        sorted_labels = sorted(scores, key=lambda l: scores[l], reverse=True)
        n_fg = max(1, int(len(sorted_labels) * self.ratio))
        fg_labels = set(sorted_labels[:n_fg])

        mask_np = np.zeros_like(mag_np, dtype=np.float32)
        for lbl in fg_labels:
            mask_np[labels == lbl] = 1.0

        mask = torch.from_numpy(mask_np).to(
            device=cond_pred.device, dtype=cond_pred.dtype,
        )

        if self.blur_sigma > 0:
            mask = soft_mask_pipeline(mask, dilation_radius=0, blur_sigma=self.blur_sigma)

        return mask
