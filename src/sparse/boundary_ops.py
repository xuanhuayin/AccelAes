"""
Boundary smoothing operations for mask post-processing (Phase 4).

Provides morphological dilation and Gaussian blur to create soft
transition zones at foreground/background boundaries.
"""

import torch
import torch.nn.functional as F


def dilate_mask(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """
    Morphological dilation via max_pool2d.

    Args:
        mask: (H, W) binary or soft mask tensor.
        radius: dilation radius in pixels.

    Returns:
        Dilated mask, same shape and device.
    """
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    # max_pool2d expects (B, C, H, W)
    x = mask.unsqueeze(0).unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=radius)
    return x.squeeze(0).squeeze(0)


def gaussian_blur(mask: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """
    Apply Gaussian blur to a 2D mask.

    Args:
        mask: (H, W) float tensor.
        sigma: Gaussian standard deviation.

    Returns:
        Blurred mask, clamped to [0, 1].
    """
    if sigma <= 0:
        return mask
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.reshape(1, 1, kernel_size, kernel_size)

    padded = mask.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    padded = F.pad(padded, [pad, pad, pad, pad], mode="reflect")
    blurred = F.conv2d(padded, kernel_2d)
    return blurred.squeeze(0).squeeze(0).clamp(0, 1)


def soft_mask_pipeline(
    binary_mask: torch.Tensor,
    dilation_radius: int = 2,
    blur_sigma: float = 1.5,
) -> torch.Tensor:
    """
    Combined boundary smoothing: dilate then blur.

    Converts a hard binary mask into a soft mask with smooth transitions
    at the foreground/background boundary.

    Args:
        binary_mask: (H, W) binary mask (1=foreground, 0=background).
        dilation_radius: pixels to expand foreground before blurring.
        blur_sigma: Gaussian blur sigma for smooth falloff.

    Returns:
        Soft mask (H, W), values in [0, 1].
    """
    mask = dilate_mask(binary_mask, radius=dilation_radius)
    mask = gaussian_blur(mask, sigma=blur_sigma)
    return mask
