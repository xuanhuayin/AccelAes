"""
Token-merging self-attention processor for Lumina DiT.

Replaces LuminaAttnProcessor2_0 on attn1 (self-attention) layers.
Merges background tokens before SDPA to reduce the O(N²) cost,
then unmerges to restore full sequence length for downstream cross-attention.

The processor output shape is identical to the original: (B, N, heads, head_dim).
"""

import math
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class ToMeSelfAttnProcessor:
    """
    Self-attention processor with token merging.

    Drop-in replacement for LuminaAttnProcessor2_0 on attn1 layers.
    Uses a foreground mask to protect important tokens from being merged.

    The merge/unmerge happens after RoPE application and before SDPA,
    so positional information is preserved in the merged representation.
    """

    def __init__(self, fg_mask: torch.Tensor | None, merge_ratio: float, patch_h: int, patch_w: int):
        """
        Args:
            fg_mask: (N,) bool tensor, True = foreground (never merged).
                     None means all tokens can be merged (uniform mode).
            merge_ratio: fraction of background tokens to merge (0-1).
            patch_h: spatial height of patch grid.
            patch_w: spatial width of patch grid.
        """
        self.fg_mask = fg_mask
        self.merge_ratio = merge_ratio
        self.patch_h = patch_h
        self.patch_w = patch_w

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
        from src.sparse.token_merge import bipartite_soft_matching_with_mask

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        # --- 1. Compute Q, K, V ---
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

        # --- 4. Compute merge/unmerge from K (head 0) ---
        # Use K after RoPE as the similarity metric (position-aware features)
        # Shape: (B, N, kv_heads, head_dim) → take first KV head
        merge_metric = key[:, :, 0, :]  # (B, N, head_dim)

        # Compute number of tokens to merge
        if self.fg_mask is not None:
            n_background = (~self.fg_mask).sum().item()
        else:
            n_background = sequence_length
        r = int(n_background * self.merge_ratio)

        if r > 0:
            merge_fn, unmerge_fn = bipartite_soft_matching_with_mask(
                metric=merge_metric,
                fg_mask=self.fg_mask,
                r=r,
                patch_h=self.patch_h,
                patch_w=self.patch_w,
            )
        else:
            merge_fn = lambda x: x
            unmerge_fn = lambda x: x

        # --- 5. Merge Q, K, V ---
        # Q, K, V are (B, N, heads/kv_heads, head_dim)
        # merge_fn expects (B, N, ...) — works with any trailing dims
        query = merge_fn(query)    # (B, N-r, heads, head_dim)
        key = merge_fn(key)        # (B, N-r, kv_heads, head_dim)
        value = merge_fn(value)    # (B, N-r, kv_heads, head_dim)

        merged_len = query.shape[1]

        # --- 6. Proportional attention scale ---
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # --- 7. GQA expansion ---
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # --- 8. SDPA on merged tokens ---
        # (B, N-r, heads, head_dim) → (B, heads, N-r, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Build attention mask for merged sequence
        if attention_mask is not None:
            # Self-attention mask: (B, 1, 1, N) → need (B, heads, N-r, N-r)
            # For self-attention, we create a simple causal-free mask for merged tokens
            # Since all tokens can attend to all other tokens in self-attn, use no mask
            attn_mask = None
        else:
            attn_mask = None

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, scale=softmax_scale
        )

        # (B, heads, N-r, head_dim) → (B, N-r, heads, head_dim)
        hidden_states = hidden_states.transpose(1, 2).to(dtype)

        # --- 9. Unmerge: (B, N-r, heads, head_dim) → (B, N, heads, head_dim) ---
        hidden_states = unmerge_fn(hidden_states)

        return hidden_states
