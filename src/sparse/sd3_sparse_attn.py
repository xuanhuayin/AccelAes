"""
Spatial Attention Caching for SD3 JointTransformerBlock.

For each block, we replace the attention processor with one that:
  - dense_mode=True  : computes full joint attention, caches image-token output
  - dense_mode=False : computes attention ONLY for foreground image tokens (fresh),
                       reuses cached output for background image tokens.
                       Text tokens always get fresh attention.

Speedup source:
  Normal SDPA: Q[(N_img + N_txt)] × K[(N_img + N_txt)]  → O((N_img+N_txt)^2)
  Sparse SDPA: Q[(N_fg  + N_txt)] × K[(N_img + N_txt)]  → O((N_fg+N_txt)*(N_img+N_txt))
  With N_fg ≈ 0.5·N_img, attention FLOPs ≈ 50% cheaper per sparse step.

Unlike sparse FFN, this does NOT depend on the specific AdaLayerNorm modulation
of the block input: the attention output is what we cache, and the gate_msa
modulation is applied outside the attention call (in JointTransformerBlock.forward),
so the current step's gate is always applied correctly.
"""

import torch
import torch.nn.functional as F


class _SD3SparseAttnProcessor:
    """
    Replaces JointAttnProcessor2_0 with spatial caching.

    Args:
        fg_mask   : (N_img,) bool tensor. True = foreground token.
        layer_idx : block index (cache key).
        attn_cache: shared dict {layer_idx: Tensor(B, N_img, inner_dim)}.
        dense_mode: if True, compute all tokens and update cache.
    """

    def __init__(self, fg_mask: torch.Tensor, layer_idx: int, attn_cache: dict):
        self.fg_mask = fg_mask
        self.layer_idx = layer_idx
        self.attn_cache = attn_cache
        self.dense_mode = True

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        batch = hidden_states.shape[0]
        head_dim = attn.inner_dim // attn.heads

        def _proj(proj, norm, x):
            out = proj(x)
            if norm is not None:
                out = norm(out)
            # (B, N, heads, head_dim) → (B, heads, N, head_dim)
            return out.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        # Image token projections
        q_img = _proj(attn.to_q,       getattr(attn, "norm_q",       None), hidden_states)
        k_img = _proj(attn.to_k,       getattr(attn, "norm_k",       None), hidden_states)
        v_img = _proj(attn.to_v,       None,                                 hidden_states)

        # Text token projections
        q_txt = _proj(attn.add_q_proj, getattr(attn, "norm_added_q", None), encoder_hidden_states)
        k_txt = _proj(attn.add_k_proj, getattr(attn, "norm_added_k", None), encoder_hidden_states)
        v_txt = _proj(attn.add_v_proj, None,                                 encoder_hidden_states)

        n_img = hidden_states.shape[1]

        # Joint K and V always include all tokens (needed for correct attention)
        k_joint = torch.cat([k_img, k_txt], dim=2)  # (B, heads, N_img+N_txt, head_dim)
        v_joint = torch.cat([v_img, v_txt], dim=2)

        if self.dense_mode:
            # Full joint attention
            q_joint = torch.cat([q_img, q_txt], dim=2)
            out = F.scaled_dot_product_attention(
                q_joint, k_joint, v_joint, dropout_p=0.0, is_causal=False
            )
            out = out.transpose(1, 2).reshape(batch, -1, attn.inner_dim).to(hidden_states.dtype)
            out_img = out[:, :n_img]
            out_txt = out[:, n_img:]

            # Cache raw image-token attention output (before gate_msa scaling)
            self.attn_cache[self.layer_idx] = out_img.detach().clone()

        else:
            # Sparse: compute Q only for foreground image tokens
            fg_idx = self.fg_mask.nonzero(as_tuple=True)[0]   # (N_fg,)
            bg_idx = (~self.fg_mask).nonzero(as_tuple=True)[0]

            q_fg = q_img[:, :, fg_idx, :]  # (B, heads, N_fg, head_dim)

            # Joint Q: fg image tokens + all text tokens
            q_sparse = torch.cat([q_fg, q_txt], dim=2)  # (B, heads, N_fg+N_txt, head_dim)

            out_sparse = F.scaled_dot_product_attention(
                q_sparse, k_joint, v_joint, dropout_p=0.0, is_causal=False
            )
            out_sparse = out_sparse.transpose(1, 2).reshape(
                batch, -1, attn.inner_dim
            ).to(hidden_states.dtype)

            n_fg = fg_idx.shape[0]
            out_fg_img = out_sparse[:, :n_fg]    # (B, N_fg, inner_dim)
            out_txt    = out_sparse[:, n_fg:]    # (B, N_txt, inner_dim)

            # Assemble image output: fg=fresh, bg=cached
            out_img = torch.zeros(
                batch, n_img, attn.inner_dim,
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            out_img[:, fg_idx] = out_fg_img
            if self.layer_idx in self.attn_cache:
                out_img[:, bg_idx] = self.attn_cache[self.layer_idx][:, bg_idx]

            # Update cache with combined output
            self.attn_cache[self.layer_idx] = out_img.detach().clone()

        # Output projections (same as JointAttnProcessor2_0)
        out_img = attn.to_out[0](out_img)
        out_img = attn.to_out[1](out_img)

        if hasattr(attn, "to_add_out") and not attn.context_pre_only:
            out_txt = attn.to_add_out(out_txt)

        return out_img, out_txt


class SD3SparseAttnManager:
    """
    Installs _SD3SparseAttnProcessor on selected SD3 transformer blocks.

    Usage:
        mgr = SD3SparseAttnManager(transformer)
        mgr.install(fg_mask, dense_mode=True)   # warmup
        # ... 2 warmup steps ...
        mgr.set_dense_mode(False)               # switch to sparse
        # ... remaining steps ...
        mgr.remove()
    """

    def __init__(self, transformer, layer_indices=None):
        self.transformer = transformer
        blocks = transformer.transformer_blocks
        self.num_layers = len(blocks)
        self.layer_indices = (
            layer_indices if layer_indices is not None else list(range(self.num_layers))
        )
        self._orig_procs: dict = {}    # {idx: original processor}
        self._sparse_procs: dict = {}  # {idx: _SD3SparseAttnProcessor}
        self._attn_cache: dict = {}    # {idx: Tensor(B, N_img, inner_dim)}

    def install(self, fg_mask: torch.Tensor, dense_mode: bool = True):
        """
        Replace attention processors on selected blocks.

        Args:
            fg_mask   : (N_img,) bool. True = foreground.
            dense_mode: start dense for cache warmup.
        """
        self._attn_cache.clear()
        blocks = self.transformer.transformer_blocks
        for idx in self.layer_indices:
            block = blocks[idx]
            # Skip context_pre_only blocks (last block, no text output)
            # — they still have attention but with a different structure.
            # We can still apply caching; context_pre_only is handled in __call__.
            orig_proc = block.attn.processor
            self._orig_procs[idx] = orig_proc

            proc = _SD3SparseAttnProcessor(
                fg_mask=fg_mask,
                layer_idx=idx,
                attn_cache=self._attn_cache,
            )
            proc.dense_mode = dense_mode
            block.attn.set_processor(proc)
            self._sparse_procs[idx] = proc

    def set_dense_mode(self, dense: bool):
        """Toggle all processors between dense (cache update) and sparse (bg reuse)."""
        for proc in self._sparse_procs.values():
            proc.dense_mode = dense

    def remove(self):
        """Restore original processors and clear cache."""
        blocks = self.transformer.transformer_blocks
        for idx, orig in self._orig_procs.items():
            blocks[idx].attn.set_processor(orig)
        self._orig_procs.clear()
        self._sparse_procs.clear()
        self._attn_cache.clear()
