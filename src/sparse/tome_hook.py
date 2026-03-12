"""
Token merging hook manager for Lumina DiT.

Manages installation and removal of ToMe self-attention processors
on the transformer's attn1 (self-attention) layers. Mirrors the
CrossAttnHookManager pattern used in Phase 2.
"""

import torch

from src.sparse.tome_processor import ToMeSelfAttnProcessor


class ToMeHookManager:
    """
    Manages token merging processors on self-attention (attn1) layers.

    Usage:
        mgr = ToMeHookManager(transformer, patch_h=64, patch_w=64)
        mgr.install(fg_mask=mask, merge_ratio=0.5)
        # ... run denoising steps ...
        mgr.update_ratio(0.3)  # optionally adjust ratio per step
        # ... more steps ...
        mgr.remove()  # restore original processors
    """

    def __init__(self, transformer, patch_h: int, patch_w: int, layer_indices=None):
        """
        Args:
            transformer: LuminaNextDiT2DModel instance.
            patch_h: spatial height of the token grid.
            patch_w: spatial width of the token grid.
            layer_indices: which layers to hook (default: all).
        """
        self.transformer = transformer
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_layers = len(transformer.layers)
        self.layer_indices = layer_indices or list(range(self.num_layers))
        self._original_processors = {}
        self._fg_mask = None
        self._merge_ratio = 0.0

    def install(self, fg_mask: torch.Tensor | None, merge_ratio: float):
        """
        Install ToMe processors on all attn1 layers.

        Args:
            fg_mask: (N,) bool tensor, True = foreground (protected).
                     None = uniform mode (all tokens can be merged).
            merge_ratio: fraction of background tokens to merge (0-1).
        """
        self._fg_mask = fg_mask
        self._merge_ratio = merge_ratio

        for idx in self.layer_indices:
            block = self.transformer.layers[idx]
            self._original_processors[idx] = block.attn1.processor
            block.attn1.processor = ToMeSelfAttnProcessor(
                fg_mask=fg_mask,
                merge_ratio=merge_ratio,
                patch_h=self.patch_h,
                patch_w=self.patch_w,
            )

    def remove(self):
        """Restore original attn1 processors."""
        for idx, proc in self._original_processors.items():
            self.transformer.layers[idx].attn1.processor = proc
        self._original_processors.clear()

    def update_ratio(self, ratio: float):
        """
        Update merge ratio on all installed processors.

        Useful for schedule-based merging where ratio changes per step.
        """
        self._merge_ratio = ratio
        for idx in self.layer_indices:
            proc = self.transformer.layers[idx].attn1.processor
            if isinstance(proc, ToMeSelfAttnProcessor):
                proc.merge_ratio = ratio
