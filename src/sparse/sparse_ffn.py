"""
Sparse FFN wrapper and block-level manager for Lumina DiT.

SparseFeedForward wraps the original LuminaFeedForward so that only
foreground tokens pass through the SwiGLU MLP.  Background tokens
reuse the cached output from the previous denoising step.

SparseBlockManager coordinates both sparse attn1 (via SparseHookManager)
and sparse FFN across all transformer layers.
"""

import torch
import torch.nn as nn

from src.sparse.sparse_hook import SparseHookManager
from src.sparse.sparse_processor import SparseAttnProcessor


class SparseFeedForward(nn.Module):
    """
    Wrapper around LuminaFeedForward that only computes MLP for
    foreground tokens.  Background tokens reuse cached FFN output.

    Args:
        original_ff: the original feed_forward module to wrap.
        fg_mask: (N,) bool tensor, True = foreground.
        layer_idx: integer layer index for cache lookup.
        ffn_cache: shared dict {layer_idx: Tensor(B, N, D)}.
    """

    def __init__(self, original_ff: nn.Module, fg_mask: torch.Tensor,
                 layer_idx: int, ffn_cache: dict):
        super().__init__()
        self.ff = original_ff
        self.fg_mask = fg_mask
        self.layer_idx = layer_idx
        self.ffn_cache = ffn_cache
        self.dense_mode = False

    def forward(self, x):
        # x: (B, N, hidden_size)
        if self.dense_mode:
            out = self.ff(x)
            self.ffn_cache[self.layer_idx] = out.detach().clone()
            return out

        fg_idx = self.fg_mask.nonzero(as_tuple=True)[0]  # (N_fg,)
        bg_idx = (~self.fg_mask).nonzero(as_tuple=True)[0]  # (N_bg,)

        # Foreground: compute FFN on subset
        x_fg = x[:, fg_idx]        # (B, N_fg, D)
        out_fg = self.ff(x_fg)     # (B, N_fg, D)

        # Scatter: fg = new, bg = cached
        out = torch.zeros_like(x)
        out[:, fg_idx] = out_fg
        if self.layer_idx in self.ffn_cache:
            out[:, bg_idx] = self.ffn_cache[self.layer_idx][:, bg_idx]

        self.ffn_cache[self.layer_idx] = out.detach().clone()
        return out


class SparseBlockManager:
    """
    Manages both SparseAttnProcessor (attn1) and SparseFeedForward (FFN)
    across all transformer layers.

    Extends SparseHookManager by additionally wrapping each block's
    feed_forward with SparseFeedForward.

    Usage:
        mgr = SparseBlockManager(transformer)
        mgr.install(fg_mask=mask)
        # ... denoising loop ...
        mgr.remove()
    """

    def __init__(self, transformer, layer_indices=None):
        self.transformer = transformer
        self.num_layers = len(transformer.layers)
        self.layer_indices = layer_indices or list(range(self.num_layers))

        # Attn1 sparse manager (reuse existing)
        self._sparse_hook_mgr = SparseHookManager(transformer, layer_indices)

        # FFN state
        self._original_ffs = {}      # {layer_idx: original feed_forward module}
        self._ffn_cache = {}         # shared across layers: {layer_idx: Tensor}
        self._sparse_ffs = {}        # {layer_idx: SparseFeedForward wrapper}

        self._sparse_ffn_enabled = True  # can disable FFN sparsity independently

    def install(self, fg_mask: torch.Tensor, dense_mode: bool = False,
                sparse_ffn: bool = True):
        """
        Install sparse processors on attn1 and FFN wrappers.

        Args:
            fg_mask: (N,) bool tensor, True = foreground.
            dense_mode: if True, run full compute to warm up caches.
            sparse_ffn: if False, only install sparse attn1 (no FFN wrapping).
        """
        self._fg_mask = fg_mask
        self._sparse_ffn_enabled = sparse_ffn

        # 1. Install sparse attn1 processors
        self._sparse_hook_mgr.install(fg_mask=fg_mask, dense_mode=dense_mode)

        # 2. Install SparseFeedForward wrappers
        if sparse_ffn:
            for idx in self.layer_indices:
                block = self.transformer.layers[idx]
                self._original_ffs[idx] = block.feed_forward
                wrapper = SparseFeedForward(
                    original_ff=block.feed_forward,
                    fg_mask=fg_mask,
                    layer_idx=idx,
                    ffn_cache=self._ffn_cache,
                )
                wrapper.dense_mode = dense_mode
                block.feed_forward = wrapper
                self._sparse_ffs[idx] = wrapper

    def set_dense_mode(self, dense: bool):
        """Toggle all processors and FFN wrappers between dense and sparse."""
        self._sparse_hook_mgr.set_dense_mode(dense)
        for wrapper in self._sparse_ffs.values():
            wrapper.dense_mode = dense

    def update_mask(self, fg_mask: torch.Tensor):
        """Update fg_mask in all processors and wrappers; clears attn+FFN caches."""
        self._fg_mask = fg_mask
        self._sparse_hook_mgr.update_mask(fg_mask)
        self._ffn_cache.clear()
        for wrapper in self._sparse_ffs.values():
            wrapper.fg_mask = fg_mask

    def remove(self):
        """Restore original attn1 processors and feed_forward modules, clear caches."""
        # Restore attn1
        self._sparse_hook_mgr.remove()

        # Restore FFN
        for idx, original_ff in self._original_ffs.items():
            self.transformer.layers[idx].feed_forward = original_ff
        self._original_ffs.clear()
        self._sparse_ffs.clear()
        self._ffn_cache.clear()

    def cache_is_warm(self):
        """Check if both attn1 and FFN caches are populated for all layers."""
        attn_warm = self._sparse_hook_mgr.cache_is_warm()
        if not self._sparse_ffn_enabled:
            return attn_warm
        ffn_warm = len(self._ffn_cache) == len(self.layer_indices)
        return attn_warm and ffn_warm

    @property
    def cache_memory_mb(self):
        """Estimate total cache memory (attn1 + FFN) in MB."""
        attn_mb = self._sparse_hook_mgr.cache_memory_mb
        ffn_total = 0
        for v in self._ffn_cache.values():
            ffn_total += v.nelement() * v.element_size()
        return attn_mb + ffn_total / (1024 * 1024)
