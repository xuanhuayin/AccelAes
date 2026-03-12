"""
SD3 sparse block manager.

Manages installation of SparseJointAttnProcessor and SparseFeedForward
across all JointTransformerBlocks in SD3Transformer2DModel.

SD3 architecture:
  - Flat list: transformer.transformer_blocks[0..17] (18 layers for SD3-Medium)
  - Single spatial resolution: 64×64 patches (1024/8/2 = 64)
  - Joint attention (MMDiT): image + text tokens concatenated
  - Last block (17): context_pre_only=True (no text output)
  - Each block has: attn (joint), ff (image FFN), ff_context (text FFN)
"""

import torch
import torch.nn as nn

from src.sparse.sd3_sparse_processor import SparseJointAttnProcessor
from src.sparse.sparse_ffn import SparseFeedForward


class SD3SparseBlockManager:
    """
    Manages sparse joint-attention + sparse FFN across all SD3 transformer blocks.

    Usage:
        mgr = SD3SparseBlockManager(transformer)
        mgr.install(fg_mask=mask, dense_mode=True)
        # ... denoising loop ...
        mgr.set_dense_mode(False)
        # ...
        mgr.remove()
    """

    def __init__(self, transformer):
        """
        Args:
            transformer: SD3Transformer2DModel instance.
        """
        self.transformer = transformer
        self.num_layers = len(transformer.transformer_blocks)

        self._original_processors = {}  # {layer_idx: original processor}
        self._original_ffs = {}         # {layer_idx: original ff}
        self._attn_cache = {}           # shared: {layer_idx: (img_out, ctx_out)}
        self._ffn_cache = {}            # shared: {layer_idx: Tensor}
        self._sparse_ffs = {}           # {layer_idx: SparseFeedForward wrapper}
        self._sparse_ffn_enabled = True

    def install(self, fg_mask: torch.Tensor, dense_mode: bool = False,
                sparse_ffn: bool = True):
        """
        Install sparse processors on all blocks.

        Args:
            fg_mask: (N_img,) bool tensor at patch level (e.g., 64*64=4096).
                True = foreground image token.
            dense_mode: if True, run full compute to warm up caches.
            sparse_ffn: if True, also wrap ff with SparseFeedForward.
        """
        self._sparse_ffn_enabled = sparse_ffn

        for idx in range(self.num_layers):
            block = self.transformer.transformer_blocks[idx]

            # ---- Joint attention ----
            self._original_processors[idx] = block.attn.processor
            block.attn.processor = SparseJointAttnProcessor(
                fg_mask=fg_mask,
                layer_idx=idx,
                cache=self._attn_cache,
                dense_mode=dense_mode,
            )

            # ---- Image FFN (block.ff) ----
            # Note: we do NOT sparsify ff_context (text FFN) since text tokens
            # are always fully computed.
            if sparse_ffn and hasattr(block, 'ff') and block.ff is not None:
                self._original_ffs[idx] = block.ff
                wrapper = SparseFeedForward(
                    original_ff=block.ff,
                    fg_mask=fg_mask,
                    layer_idx=idx,
                    ffn_cache=self._ffn_cache,
                )
                wrapper.dense_mode = dense_mode
                block.ff = wrapper
                self._sparse_ffs[idx] = wrapper

    def set_dense_mode(self, dense: bool):
        """Toggle all processors and FFN wrappers between dense and sparse."""
        for idx in range(self.num_layers):
            block = self.transformer.transformer_blocks[idx]
            proc = block.attn.processor
            if isinstance(proc, SparseJointAttnProcessor):
                proc.dense_mode = dense

        for wrapper in self._sparse_ffs.values():
            wrapper.dense_mode = dense

    def remove(self):
        """Restore original processors and FFN modules, clear caches."""
        for idx, proc in self._original_processors.items():
            self.transformer.transformer_blocks[idx].attn.processor = proc

        for idx, ff in self._original_ffs.items():
            self.transformer.transformer_blocks[idx].ff = ff

        self._original_processors.clear()
        self._original_ffs.clear()
        self._sparse_ffs.clear()
        self._attn_cache.clear()
        self._ffn_cache.clear()

    def cache_is_warm(self):
        """Check if all caches are populated."""
        n = self.num_layers
        attn_warm = len(self._attn_cache) == n
        if not self._sparse_ffn_enabled:
            return attn_warm
        # FFN cache might have fewer entries if some blocks have no ff
        ffn_warm = len(self._ffn_cache) == len(self._sparse_ffs)
        return attn_warm and ffn_warm

    @property
    def block_info(self):
        """Return list of (name, has_dual_attn, context_pre_only) for debugging."""
        info = []
        for idx in range(self.num_layers):
            block = self.transformer.transformer_blocks[idx]
            info.append((
                f"block_{idx}",
                getattr(block, 'use_dual_attention', False),
                getattr(block, 'context_pre_only', False),
            ))
        return info
