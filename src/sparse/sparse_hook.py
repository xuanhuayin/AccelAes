"""
Sparse forward hook manager for Lumina DiT.

Manages installation and removal of SparseAttnProcessor on the
transformer's attn1 (self-attention) layers. Mirrors the
ToMeHookManager pattern used in Phase 3.
"""

import torch

from src.sparse.sparse_processor import SparseAttnProcessor


class SparseHookManager:
    """
    Manages sparse attention processors on self-attention (attn1) layers.

    All layers share a single cache dict so memory is centrally managed.

    Usage:
        mgr = SparseHookManager(transformer)
        mgr.install(fg_mask=mask)
        # ... run denoising steps ...
        mgr.remove()  # restore original processors, clear cache
    """

    def __init__(self, transformer, layer_indices=None):
        """
        Args:
            transformer: LuminaNextDiT2DModel instance.
            layer_indices: which layers to hook (default: all).
        """
        self.transformer = transformer
        self.num_layers = len(transformer.layers)
        self.layer_indices = layer_indices or list(range(self.num_layers))
        self._original_processors = {}
        self._cache = {}  # shared across all layers: {layer_idx: Tensor}

    def install(self, fg_mask: torch.Tensor, dense_mode: bool = False):
        """
        Install SparseAttnProcessor on all attn1 layers.

        Args:
            fg_mask: (N,) bool tensor, True = foreground (computed via SDPA).
            dense_mode: if True, processors run full SDPA to warm up cache.
        """
        self._fg_mask = fg_mask
        for idx in self.layer_indices:
            block = self.transformer.layers[idx]
            self._original_processors[idx] = block.attn1.processor
            block.attn1.processor = SparseAttnProcessor(
                fg_mask=fg_mask,
                layer_idx=idx,
                cache=self._cache,
                dense_mode=dense_mode,
            )

    def set_dense_mode(self, dense: bool):
        """Toggle all processors between dense (cache warm-up) and sparse mode."""
        for idx in self.layer_indices:
            proc = self.transformer.layers[idx].attn1.processor
            if isinstance(proc, SparseAttnProcessor):
                proc.dense_mode = dense

    def update_mask(self, fg_mask: torch.Tensor):
        """Update fg_mask in all processors and clear attn cache (mask changed)."""
        self._fg_mask = fg_mask
        self._cache.clear()
        for idx in self.layer_indices:
            proc = self.transformer.layers[idx].attn1.processor
            if isinstance(proc, SparseAttnProcessor):
                proc.fg_mask = fg_mask

    def remove(self):
        """Restore original attn1 processors and clear cache."""
        for idx, proc in self._original_processors.items():
            self.transformer.layers[idx].attn1.processor = proc
        self._original_processors.clear()
        self._cache.clear()

    def cache_is_warm(self):
        """Check if cache is populated for all layers."""
        return len(self._cache) == len(self.layer_indices)

    @property
    def cache_memory_mb(self):
        """Estimate current cache memory usage in MB."""
        total = 0
        for v in self._cache.values():
            total += v.nelement() * v.element_size()
        return total / (1024 * 1024)
