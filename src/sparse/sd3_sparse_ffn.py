"""
Sparse FFN for SD3 JointTransformerBlock.

Mirrors Lumina's SparseFeedForward / SparseBlockManager but targets
SD3's transformer_blocks[i].ff (image feed-forward only).

Each JointTransformerBlock forward:
  norm_hidden → gate/shift affine → self.ff(norm_hidden) → gate_mlp * ff_out → residual

We wrap self.ff so that:
  - foreground tokens: compute fresh FF output
  - background tokens: reuse cached output from previous dense step
  - dense_mode: compute all tokens and update cache (warmup)
"""

import torch
import torch.nn as nn


class SD3SparseFeedForward(nn.Module):
    """
    Wrapper around SD3 JointTransformerBlock.ff for background token caching.

    Args:
        original_ff: the original FeedForward module.
        fg_mask: (N_img,) bool tensor. True = foreground token.
        layer_idx: integer index (used as cache key).
        ffn_cache: shared dict {layer_idx: Tensor(B, N_img, D)}.
    """

    def __init__(self, original_ff: nn.Module, fg_mask: torch.Tensor,
                 layer_idx: int, ffn_cache: dict):
        super().__init__()
        self.ff = original_ff
        self.fg_mask = fg_mask
        self.layer_idx = layer_idx
        self.ffn_cache = ffn_cache
        self.dense_mode = True  # start dense to warm cache

    def forward(self, x):
        # x: (B, N_img, D)
        if self.dense_mode:
            out = self.ff(x)
            self.ffn_cache[self.layer_idx] = out.detach().clone()
            return out

        fg_idx = self.fg_mask.nonzero(as_tuple=True)[0]   # (N_fg,)
        bg_idx = (~self.fg_mask).nonzero(as_tuple=True)[0]  # (N_bg,)

        # Foreground: fresh computation on subset
        out_fg = self.ff(x[:, fg_idx])  # (B, N_fg, D)

        # Assemble output: fg=new, bg=cached
        out = torch.zeros_like(x)
        out[:, fg_idx] = out_fg
        if self.layer_idx in self.ffn_cache:
            out[:, bg_idx] = self.ffn_cache[self.layer_idx][:, bg_idx]

        self.ffn_cache[self.layer_idx] = out.detach().clone()
        return out


class SD3SparseBlockManager:
    """
    Installs SD3SparseFeedForward on selected SD3 transformer blocks.

    Usage:
        mgr = SD3SparseBlockManager(transformer)
        mgr.install(fg_mask)          # dense_mode=True by default (warmup)
        # ... 2 warmup steps ...
        mgr.set_dense_mode(False)     # switch to sparse
        # ... remaining steps ...
        mgr.remove()
    """

    def __init__(self, transformer, layer_indices=None):
        self.transformer = transformer
        blocks = transformer.transformer_blocks
        self.num_layers = len(blocks)
        self.layer_indices = layer_indices if layer_indices is not None \
            else list(range(self.num_layers))

        self._original_ffs: dict = {}   # {idx: original ff module}
        self._ffn_cache: dict = {}       # {idx: Tensor(B, N_img, D)}
        self._sparse_ffs: dict = {}      # {idx: SD3SparseFeedForward}

    def install(self, fg_mask: torch.Tensor, dense_mode: bool = True):
        """
        Replace self.ff in selected blocks with SD3SparseFeedForward.

        Args:
            fg_mask: (N_img,) bool tensor. True = foreground.
            dense_mode: start in dense mode to populate cache.
        """
        self._ffn_cache.clear()
        for idx in self.layer_indices:
            block = self.transformer.transformer_blocks[idx]
            orig = block.ff
            self._original_ffs[idx] = orig
            wrapper = SD3SparseFeedForward(
                original_ff=orig,
                fg_mask=fg_mask,
                layer_idx=idx,
                ffn_cache=self._ffn_cache,
            )
            wrapper.dense_mode = dense_mode
            block.ff = wrapper
            self._sparse_ffs[idx] = wrapper

    def set_dense_mode(self, dense: bool):
        """Toggle all wrappers between dense (cache update) and sparse (bg reuse)."""
        for wrapper in self._sparse_ffs.values():
            wrapper.dense_mode = dense

    def remove(self):
        """Restore original ff modules and clear all caches."""
        for idx, orig in self._original_ffs.items():
            self.transformer.transformer_blocks[idx].ff = orig
        self._original_ffs.clear()
        self._sparse_ffs.clear()
        self._ffn_cache.clear()
