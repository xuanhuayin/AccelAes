"""
Sparse self-attention processor for Lumina DiT.

Replaces LuminaAttnProcessor2_0 on attn1 (self-attention) layers.
Only foreground tokens participate in SDPA; background tokens reuse
cached attention output from the previous denoising step.

The processor output shape is identical to the original: (B, N, heads, head_dim).
"""

import math
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class SparseAttnProcessor:
    """
    Self-attention processor with foreground-only SDPA.

    Drop-in replacement for LuminaAttnProcessor2_0 on attn1 layers.
    Background tokens skip the O(N^2) attention computation entirely,
    using per-layer cached output from the previous step instead.

    Args:
        fg_mask: (N,) bool tensor, True = foreground (computed).
        layer_idx: integer layer index for cache lookup.
        cache: shared dict {layer_idx: Tensor(B, N, heads, head_dim)}.
    """

    def __init__(self, fg_mask: torch.Tensor, layer_idx: int, cache: dict,
                 dense_mode: bool = False):
        self.fg_mask = fg_mask          # (N,) bool
        self.layer_idx = layer_idx
        self.cache = cache              # shared across layers
        self.dense_mode = dense_mode    # True = run full SDPA (for cache warm-up)

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

        # --- 1. Compute Q, K, V on full N tokens (O(N) linear projections) ---
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype
        kv_heads = inner_dim // head_dim

        # --- 2. QK normalization ---
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # --- 3. Apply RoPE ---
        if query_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # --- 4. Proportional attention scale (use original N for consistency) ---
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # --- 5. GQA expansion ---
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # ===== Dense mode: full SDPA on all tokens (cache warm-up) =====
        if self.dense_mode:
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, scale=softmax_scale
            )
            output = output.transpose(1, 2).to(dtype)  # (B, N, heads, head_dim)

            # Fill cache for all layers
            self.cache[self.layer_idx] = output.detach().clone()
            return output

        # ===== Sparse mode: Q_fg @ (K_all, V_all) =====
        # Foreground queries attend to ALL keys/values (full context preserved).
        # Background queries are skipped entirely (use cache).
        # Compute: O(N_fg * N) instead of O(N * N).

        # --- 6. Gather foreground Q only; K/V stay full ---
        fg_indices = self.fg_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_fg,)
        n_fg = fg_indices.shape[0]

        idx_q = fg_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, n_fg, attn.heads * head_dim
        )
        q_fg = query.reshape(batch_size, sequence_length, -1).gather(1, idx_q).reshape(batch_size, n_fg, attn.heads, head_dim)

        # --- 7. SDPA: (B, heads, N_fg, head_dim) @ (B, heads, N, head_dim) ---
        q_fg = q_fg.transpose(1, 2)            # (B, heads, N_fg, head_dim)
        k_all = key.transpose(1, 2)            # (B, heads, N,    head_dim)
        v_all = value.transpose(1, 2)          # (B, heads, N,    head_dim)

        attn_out_fg = F.scaled_dot_product_attention(
            q_fg, k_all, v_all, attn_mask=None, scale=softmax_scale
        )
        attn_out_fg = attn_out_fg.transpose(1, 2).to(dtype)  # (B, N_fg, heads, head_dim)

        # --- 8. Scatter back: foreground = new, background = cached ---
        if self.layer_idx in self.cache:
            output = self.cache[self.layer_idx].clone()
        else:
            output = torch.zeros(
                batch_size, sequence_length, attn.heads, head_dim,
                device=hidden_states.device, dtype=dtype,
            )

        idx_out = fg_indices.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, n_fg, attn.heads * head_dim
        )
        output_flat = output.reshape(batch_size, sequence_length, -1)
        output_flat.scatter_(1, idx_out, attn_out_fg.reshape(batch_size, n_fg, -1))
        output = output_flat.reshape(batch_size, sequence_length, attn.heads, head_dim)

        # --- 9. Update cache ---
        self.cache[self.layer_idx] = output.detach().clone()

        return output
