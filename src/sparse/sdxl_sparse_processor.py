"""
Sparse self-attention processor for SDXL U-Net.

Replaces the default AttnProcessor2_0 on attn1 (self-attention) layers
in SDXL's BasicTransformerBlock. Only foreground tokens participate in
SDPA; background tokens reuse cached attention output.

Key differences from Lumina's SparseAttnProcessor:
  - No RoPE (SDXL uses absolute/sinusoidal positional encoding)
  - No GQA (standard multi-head attention)
  - attn1.to_out is a real Linear + Dropout (not Identity)
  - Output shape: (B, N, inner_dim) after to_out projection
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class SparseAttnProcessorSDXL:
    """
    Self-attention processor with foreground-only SDPA for SDXL.

    Drop-in replacement for AttnProcessor2_0 on attn1 layers.
    Background tokens skip O(N^2) attention, using cached output instead.

    Args:
        fg_mask: (N,) bool tensor, True = foreground (computed).
        layer_idx: integer layer index for cache lookup.
        cache: shared dict {layer_idx: Tensor(B, N, inner_dim)}.
        dense_mode: if True, run full SDPA (for cache warm-up).
    """

    def __init__(self, fg_mask: torch.Tensor, layer_idx: int, cache: dict,
                 dense_mode: bool = False):
        self.fg_mask = fg_mask
        self.layer_idx = layer_idx
        self.cache = cache
        self.dense_mode = dense_mode

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # For self-attention, encoder_hidden_states is None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # QKV projections on all tokens (O(N) linear ops)
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        inner_dim = query.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # QK normalization (if present)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ===== Dense mode: full SDPA (cache warm-up) =====
        if self.dense_mode:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, scale=attn.scale
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

            # to_out: Linear + Dropout
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            # Cache the output (after to_out)
            self.cache[self.layer_idx] = hidden_states.detach().clone()

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

        # ===== Sparse mode: Q_fg @ (K_all, V_all) =====
        fg_indices = self.fg_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_fg,)
        n_fg = fg_indices.shape[0]

        # Gather foreground queries only
        # query shape: (B, heads, N, head_dim) -> gather on dim=2
        idx_q = fg_indices.view(1, 1, -1, 1).expand(batch_size, attn.heads, n_fg, head_dim)
        q_fg = query.gather(2, idx_q)  # (B, heads, N_fg, head_dim)

        # SDPA: (B, heads, N_fg, head_dim) @ (B, heads, N, head_dim)
        attn_out_fg = F.scaled_dot_product_attention(
            q_fg, key, value, attn_mask=None, scale=attn.scale
        )
        # (B, heads, N_fg, head_dim) -> (B, N_fg, inner_dim)
        attn_out_fg = attn_out_fg.transpose(1, 2).reshape(batch_size, n_fg, inner_dim)

        # Apply to_out to foreground only
        attn_out_fg = attn.to_out[0](attn_out_fg)
        attn_out_fg = attn.to_out[1](attn_out_fg)

        # Scatter: fg = new, bg = cached
        if self.layer_idx in self.cache:
            output = self.cache[self.layer_idx].clone()
        else:
            output = torch.zeros(
                batch_size, sequence_length, inner_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )

        idx_out = fg_indices.view(1, -1, 1).expand(batch_size, n_fg, inner_dim)
        output.scatter_(1, idx_out, attn_out_fg)

        # Update cache
        self.cache[self.layer_idx] = output.detach().clone()

        hidden_states = output

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
