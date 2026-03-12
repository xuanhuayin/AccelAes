"""
Cross-attention hook for capturing attention maps from Lumina DiT.

The default LuminaAttnProcessor2_0 uses F.scaled_dot_product_attention (SDPA)
which doesn't return attention weights. This module provides:

  1. CrossAttnCaptureProcessor: replaces SDPA with manual attention to capture maps
  2. Hook manager to install/remove hooks on specific steps
"""

import math
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class CrossAttnCaptureProcessor:
    """
    Drop-in replacement for LuminaAttnProcessor2_0 that captures attention maps.

    Only used on cross-attention (attn2) layers. Computes attention weights
    manually instead of using SDPA, stores them in `storage`.
    """

    def __init__(self, storage: list, layer_idx: int):
        """
        Args:
            storage: shared list to append attention maps to
            layer_idx: which layer this processor belongs to
        """
        self.storage = storage
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        base_sequence_length=None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype
        kv_heads = inner_dim // head_dim

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if query_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # GQA expansion
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # (batch, heads, seq_q, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Manual attention computation to capture weights
        scale = softmax_scale if softmax_scale is not None else (head_dim ** -0.5)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
            attn_mask = attn_mask.expand(-1, attn.heads, sequence_length, -1)
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

        attn_weights = attn_weights.softmax(dim=-1).to(dtype)

        # Store attention map: only the cond branch (first half of batch)
        # Shape: (heads, num_patches, seq_len_kv)
        cond_attn = attn_weights[0].detach()  # first item in batch = cond
        self.storage.append(cond_attn)

        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).to(dtype)

        return hidden_states


class CrossAttnHookManager:
    """
    Manages cross-attention hooks on the Lumina transformer.

    Usage:
        mgr = CrossAttnHookManager(transformer)
        mgr.install()           # replace attn2 processors with capture versions
        # ... run one forward pass ...
        maps = mgr.get_maps()   # retrieve captured attention maps
        mgr.remove()            # restore original processors
    """

    def __init__(self, transformer, layer_indices=None):
        """
        Args:
            transformer: the LuminaNextDiT2DModel
            layer_indices: which layers to hook (default: all)
        """
        self.transformer = transformer
        self.num_layers = len(transformer.layers)
        self.layer_indices = layer_indices or list(range(self.num_layers))
        self._original_processors = {}
        self._storage = []

    def install(self):
        """Install capture processors on cross-attention (attn2) layers."""
        self._storage.clear()
        for idx in self.layer_indices:
            block = self.transformer.layers[idx]
            self._original_processors[idx] = block.attn2.processor
            block.attn2.processor = CrossAttnCaptureProcessor(
                storage=self._storage,
                layer_idx=idx,
            )

    def remove(self):
        """Restore original processors."""
        for idx, proc in self._original_processors.items():
            self.transformer.layers[idx].attn2.processor = proc
        self._original_processors.clear()

    def get_maps(self) -> list:
        """
        Get captured cross-attention maps.

        Returns:
            list of (num_heads, num_patches, seq_len_kv) tensors,
            one per hooked layer.
        """
        return list(self._storage)

    def clear(self):
        """Clear stored maps without removing hooks."""
        self._storage.clear()
