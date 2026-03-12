"""
SDXL sparse block manager.

Manages installation of SparseAttnProcessorSDXL (attn1) and
SparseFeedForward (FFN) across all BasicTransformerBlocks in the
SDXL U-Net (down_blocks + mid_block + up_blocks).

Key difference from Lumina's SparseBlockManager:
  - Nested block enumeration: unet.down_blocks[i].attentions[j].transformer_blocks[k]
  - Multi-resolution mask: fg_mask_latent is downsampled to each block's spatial resolution
  - Standard MHA (no GQA/RoPE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sparse.sdxl_sparse_processor import SparseAttnProcessorSDXL
from src.sparse.sparse_ffn import SparseFeedForward


class SDXLSparseBlockManager:
    """
    Manages sparse attn1 + sparse FFN across all SDXL transformer blocks.

    Handles the multi-resolution structure of U-Net:
      - down_blocks: progressively smaller spatial resolutions
      - mid_block: smallest resolution
      - up_blocks: progressively larger spatial resolutions

    Usage:
        mgr = SDXLSparseBlockManager(unet)
        mgr.install(fg_mask_latent=mask, dense_mode=True)
        # ... denoising loop ...
        mgr.set_dense_mode(False)
        # ...
        mgr.remove()
    """

    def __init__(self, unet):
        self.unet = unet
        self._blocks = []  # [(block_ref, spatial_h, spatial_w, name), ...]
        self._original_processors = {}  # {name: original_processor}
        self._original_ffs = {}  # {name: original_ff}
        self._attn_cache = {}  # shared across all attn1 layers
        self._ffn_cache = {}  # shared across all FFN layers
        self._sparse_ffs = {}  # {name: SparseFeedForward wrapper}
        self._sparse_ffn_enabled = True

        self._enumerate_blocks()

    def _enumerate_blocks(self):
        """Enumerate all BasicTransformerBlocks in down/mid/up blocks."""
        self._blocks = []
        latent_h, latent_w = 128, 128  # SDXL latent is 128x128 for 1024x1024

        # Down blocks
        current_h, current_w = latent_h, latent_w
        for i, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, 'attentions'):
                for j, attn_layer in enumerate(down_block.attentions):
                    if hasattr(attn_layer, 'transformer_blocks'):
                        for k, tb in enumerate(attn_layer.transformer_blocks):
                            name = f"down_{i}_attn_{j}_tb_{k}"
                            self._blocks.append((tb, current_h, current_w, name))
            # Downsample after each down_block (except the last one)
            if hasattr(down_block, 'downsamplers') and down_block.downsamplers is not None:
                current_h //= 2
                current_w //= 2

        # Mid block
        if hasattr(self.unet, 'mid_block') and self.unet.mid_block is not None:
            mid = self.unet.mid_block
            if hasattr(mid, 'attentions'):
                for j, attn_layer in enumerate(mid.attentions):
                    if hasattr(attn_layer, 'transformer_blocks'):
                        for k, tb in enumerate(attn_layer.transformer_blocks):
                            name = f"mid_attn_{j}_tb_{k}"
                            self._blocks.append((tb, current_h, current_w, name))

        # Up blocks
        # Up blocks go from smallest to largest resolution
        # Need to figure out the resolution schedule
        # SDXL up_blocks mirror down_blocks in reverse
        up_h, up_w = current_h, current_w
        for i, up_block in enumerate(self.unet.up_blocks):
            if hasattr(up_block, 'attentions'):
                for j, attn_layer in enumerate(up_block.attentions):
                    if hasattr(attn_layer, 'transformer_blocks'):
                        for k, tb in enumerate(attn_layer.transformer_blocks):
                            name = f"up_{i}_attn_{j}_tb_{k}"
                            self._blocks.append((tb, up_h, up_w, name))
            # Upsample after each up_block (except the last one)
            if hasattr(up_block, 'upsamplers') and up_block.upsamplers is not None:
                up_h *= 2
                up_w *= 2

    def _get_mask_for_resolution(self, fg_mask_latent, target_h, target_w, device):
        """
        Downsample/resize fg_mask_latent to target spatial resolution,
        then flatten to token-level bool mask.

        Args:
            fg_mask_latent: (H, W) float tensor at latent resolution.
            target_h, target_w: target spatial dimensions for this block.

        Returns:
            (target_h * target_w,) bool tensor.
        """
        src_h, src_w = fg_mask_latent.shape
        if src_h == target_h and src_w == target_w:
            mask_resized = fg_mask_latent
        else:
            mask_resized = F.interpolate(
                fg_mask_latent.unsqueeze(0).unsqueeze(0).float(),
                size=(target_h, target_w),
                mode="nearest",
            ).squeeze(0).squeeze(0)

        return (mask_resized.reshape(-1) > 0.5).to(device=device)

    def install(self, fg_mask_latent: torch.Tensor, dense_mode: bool = False,
                sparse_ffn: bool = True):
        """
        Install sparse processors on all attn1 layers and FFN wrappers.

        Args:
            fg_mask_latent: (H, W) float tensor at latent resolution (e.g. 128x128).
                Will be automatically resized to each block's spatial resolution.
            dense_mode: if True, run full compute to warm up caches.
            sparse_ffn: if False, only install sparse attn1.
        """
        self._sparse_ffn_enabled = sparse_ffn
        device = fg_mask_latent.device

        # Precompute masks for each resolution
        resolution_masks = {}  # {(h, w): bool_mask}

        for block, sp_h, sp_w, name in self._blocks:
            key = (sp_h, sp_w)
            if key not in resolution_masks:
                resolution_masks[key] = self._get_mask_for_resolution(
                    fg_mask_latent, sp_h, sp_w, device
                )

        # Install processors
        for idx, (block, sp_h, sp_w, name) in enumerate(self._blocks):
            fg_mask = resolution_masks[(sp_h, sp_w)]

            # attn1 (self-attention)
            self._original_processors[name] = block.attn1.processor
            block.attn1.processor = SparseAttnProcessorSDXL(
                fg_mask=fg_mask,
                layer_idx=idx,
                cache=self._attn_cache,
                dense_mode=dense_mode,
            )

            # FFN
            if sparse_ffn:
                self._original_ffs[name] = block.ff
                wrapper = SparseFeedForward(
                    original_ff=block.ff,
                    fg_mask=fg_mask,
                    layer_idx=idx,
                    ffn_cache=self._ffn_cache,
                )
                wrapper.dense_mode = dense_mode
                block.ff = wrapper
                self._sparse_ffs[name] = wrapper

    def set_dense_mode(self, dense: bool):
        """Toggle all processors and FFN wrappers between dense and sparse."""
        for block, _, _, name in self._blocks:
            proc = block.attn1.processor
            if isinstance(proc, SparseAttnProcessorSDXL):
                proc.dense_mode = dense

        for wrapper in self._sparse_ffs.values():
            wrapper.dense_mode = dense

    def remove(self):
        """Restore original attn1 processors and FFN modules, clear caches."""
        for block, _, _, name in self._blocks:
            if name in self._original_processors:
                block.attn1.processor = self._original_processors[name]
            if name in self._original_ffs:
                block.ff = self._original_ffs[name]

        self._original_processors.clear()
        self._original_ffs.clear()
        self._sparse_ffs.clear()
        self._attn_cache.clear()
        self._ffn_cache.clear()

    def cache_is_warm(self):
        """Check if all caches are populated."""
        n_blocks = len(self._blocks)
        attn_warm = len(self._attn_cache) == n_blocks
        if not self._sparse_ffn_enabled:
            return attn_warm
        ffn_warm = len(self._ffn_cache) == n_blocks
        return attn_warm and ffn_warm

    @property
    def num_blocks(self):
        return len(self._blocks)

    @property
    def block_info(self):
        """Return list of (name, spatial_h, spatial_w) for debugging."""
        return [(name, h, w) for _, h, w, name in self._blocks]
