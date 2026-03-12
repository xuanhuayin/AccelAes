"""
Sparse joint-attention processor for SD3 (MMDiT).

Replaces JointAttnProcessor2_0 on each JointTransformerBlock's `attn`.
SD3 uses joint attention: image and text tokens are concatenated before SDPA.

Key differences from Lumina/SDXL:
  - Joint attention: Q = [Q_img, Q_text], K = [K_img, K_text], V = [V_img, V_text]
  - Sparse only on image tokens; text tokens always computed fully
  - Separate projections: to_q/to_k/to_v for image, add_q_proj/add_k_proj/add_v_proj for text
  - Output split back into image and text after SDPA
  - to_out for image, to_add_out for text (except last block: context_pre_only)
"""

import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention


class SparseJointAttnProcessor:
    """
    Joint attention processor with foreground-only SDPA for SD3's MMDiT.

    In SD3, attention concatenates image and text tokens:
        Q = [Q_img(N_img), Q_text(N_text)]
        K = [K_img(N_img), K_text(N_text)]
        V = [V_img(N_img), V_text(N_text)]

    Sparse mode:
        - Text tokens: always fully computed
        - Foreground image tokens: fully computed (Q_fg @ K_all, V_all)
        - Background image tokens: reuse cached output

    Args:
        fg_mask: (N_img,) bool tensor, True = foreground image token.
        layer_idx: integer layer index for cache lookup.
        cache: shared dict {layer_idx: (img_output, ctx_output)}.
        dense_mode: if True, run full SDPA (for cache warm-up).
    """

    def __init__(self, fg_mask: torch.Tensor, layer_idx: int, cache: dict,
                 dense_mode: bool = False):
        self.fg_mask = fg_mask          # (N_img,) bool
        self.layer_idx = layer_idx
        self.cache = cache
        self.dense_mode = dense_mode

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        n_img = hidden_states.shape[1]

        # ---- Image projections ----
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ---- Text projections ----
        has_context = encoder_hidden_states is not None
        if has_context:
            n_text = encoder_hidden_states.shape[1]
            ctx_query = attn.add_q_proj(encoder_hidden_states)
            ctx_key = attn.add_k_proj(encoder_hidden_states)
            ctx_value = attn.add_v_proj(encoder_hidden_states)

            ctx_query = ctx_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ctx_key = ctx_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ctx_value = ctx_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                ctx_query = attn.norm_added_q(ctx_query)
            if attn.norm_added_k is not None:
                ctx_key = attn.norm_added_k(ctx_key)

        # ===== Dense mode: full joint SDPA =====
        if self.dense_mode:
            if has_context:
                q_full = torch.cat([query, ctx_query], dim=2)
                k_full = torch.cat([key, ctx_key], dim=2)
                v_full = torch.cat([value, ctx_value], dim=2)
            else:
                q_full = query
                k_full = key
                v_full = value

            out = F.scaled_dot_product_attention(q_full, k_full, v_full, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
            out = out.to(query.dtype)

            if has_context:
                img_out = out[:, :n_img]
                ctx_out = out[:, n_img:]
                if not attn.context_pre_only:
                    ctx_out = attn.to_add_out(ctx_out)
            else:
                img_out = out
                ctx_out = None

            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)

            # Cache
            self.cache[self.layer_idx] = (
                img_out.detach().clone(),
                ctx_out.detach().clone() if ctx_out is not None else None,
            )

            if has_context:
                return img_out, ctx_out
            return img_out

        # ===== Sparse mode: foreground image Q + all text Q =====
        fg_indices = self.fg_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_fg,)
        n_fg = fg_indices.shape[0]

        # Build full K, V (all image + all text)
        if has_context:
            k_full = torch.cat([key, ctx_key], dim=2)      # (B, heads, N_img+N_text, head_dim)
            v_full = torch.cat([value, ctx_value], dim=2)
        else:
            k_full = key
            v_full = value

        # ---- Foreground image queries ----
        idx_q = fg_indices.view(1, 1, -1, 1).expand(batch_size, attn.heads, n_fg, head_dim)
        q_fg = query.gather(2, idx_q)  # (B, heads, N_fg, head_dim)

        # ---- Text queries (always fully computed) ----
        if has_context:
            q_sparse = torch.cat([q_fg, ctx_query], dim=2)  # (B, heads, N_fg+N_text, head_dim)
        else:
            q_sparse = q_fg

        # ---- Sparse SDPA ----
        attn_out = F.scaled_dot_product_attention(q_sparse, k_full, v_full, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        attn_out = attn_out.to(query.dtype)

        # Split image foreground and text
        if has_context:
            attn_img_fg = attn_out[:, :n_fg]       # (B, N_fg, D)
            attn_ctx = attn_out[:, n_fg:]           # (B, N_text, D)
            if not attn.context_pre_only:
                ctx_out = attn.to_add_out(attn_ctx)
            else:
                ctx_out = attn_ctx
        else:
            attn_img_fg = attn_out
            ctx_out = None

        # Apply to_out to foreground image tokens
        attn_img_fg = attn.to_out[0](attn_img_fg)
        attn_img_fg = attn.to_out[1](attn_img_fg)

        # Scatter: fg=new, bg=cached
        cached = self.cache.get(self.layer_idx, (None, None))
        if cached[0] is not None:
            img_out = cached[0].clone()
        else:
            img_out = torch.zeros(
                batch_size, n_img, inner_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )

        idx_out = fg_indices.view(1, -1, 1).expand(batch_size, n_fg, inner_dim)
        img_out.scatter_(1, idx_out, attn_img_fg)

        # Update cache
        self.cache[self.layer_idx] = (
            img_out.detach().clone(),
            ctx_out.detach().clone() if ctx_out is not None else None,
        )

        if has_context:
            return img_out, ctx_out
        return img_out
