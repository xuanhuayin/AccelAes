"""
FLUX Sparse Computation Manager.

Applies sparse self-attention to FLUX's single-stream transformer blocks
(single_transformer_blocks, 38 blocks).

Architecture notes (FluxSingleTransformerBlock):
  - Input to attn: hidden_states = cat([text, image], dim=1), shape [B, N_txt+N_img, D]
  - Q,K,V format: [B, N, heads, head_dim] (unflatten, NOT transposed)
  - Norms (norm_q, norm_k) act on head_dim (last dim of [B,N,heads,head_dim])
  - RoPE: apply_rotary_emb(..., sequence_dim=1) for [B,N,heads,head_dim] tensors
  - No to_out — attn module has pre_only=True, returns raw SDPA output [B,N,heads*head_dim]
  - proj_out (attn_output + mlp_output → hidden) is at the BLOCK level, not attn level

Strategy:
  - Text tokens: always fresh (essential for conditioning fidelity)
  - Fg image tokens: fresh SDPA(Q_fg, K_all, V_all)
  - Bg image tokens: reuse cached attention output from previous non-skipped step
"""

import torch
import torch.nn.functional as F


class _FluxSparseAttnProcessor:
    """
    Sparse attention processor for FLUX single-stream blocks.

    Replaces FluxAttnProcessor. Handles text+image concatenated input.

    Dense mode:  full SDPA → update image token cache.
    Sparse mode: text + fg image → fresh SDPA(Q_fresh, K_all, V_all)
                 bg image → reuse cached output from previous step.

    Args:
        fg_mask:   (N_img,) bool tensor, True = foreground image token.
        cache:     dict {block_idx: Tensor(B, N_img, heads*head_dim)} shared cache.
        block_idx: integer index of this block.
        n_txt:     number of text tokens prepended in the block input.
    """

    def __init__(self, fg_mask: torch.Tensor, cache: dict, block_idx: int, n_txt: int):
        self.fg_mask = fg_mask
        self.cache = cache
        self.block_idx = block_idx
        self.n_txt = n_txt
        self.dense_mode = True

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,        # [B, N_txt+N_img, D]
        encoder_hidden_states=None,          # None for single-stream
        attention_mask=None,
        image_rotary_emb=None,
        *args,
        **kwargs,
    ):
        from diffusers.models.embeddings import apply_rotary_emb

        B, N_total, _ = hidden_states.shape
        head_dim = attn.inner_dim // attn.heads
        n_txt = self.n_txt

        # ---- Project Q, K, V for ALL tokens ----
        # FluxAttention convention: unflatten to [B, N, heads, head_dim]
        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, head_dim))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, head_dim))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, head_dim))

        # Norms act on head_dim (last dim of [B, N, heads, head_dim])
        q = attn.norm_q(q)
        k = attn.norm_k(k)

        # RoPE with sequence_dim=1 for [B, N, heads, head_dim] tensors
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        # Transpose to [B, heads, N, head_dim] for PyTorch SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.dense_mode or self.block_idx not in self.cache:
            # ---- Dense mode: full SDPA, update image token cache ----
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            # [B, heads, N_total, head_dim] → [B, N_total, heads*head_dim]
            out = out.transpose(1, 2).flatten(2, 3).to(hidden_states.dtype)
            # Cache only image token portion
            self.cache[self.block_idx] = out[:, n_txt:].detach().clone()
            return out

        else:
            # ---- Sparse mode: text + fg image fresh, bg image from cache ----
            fg_idx = self.fg_mask.nonzero(as_tuple=True)[0]       # [N_fg]

            # Indices in the full sequence: text tokens + fg image tokens
            txt_idx = torch.arange(n_txt, device=hidden_states.device)
            fg_full_idx = fg_idx + n_txt                           # shift by n_txt
            fresh_idx = torch.cat([txt_idx, fg_full_idx])          # [n_txt + N_fg]

            # Fresh SDPA for text + fg image
            q_fresh = q[:, :, fresh_idx, :]   # [B, heads, n_txt+N_fg, head_dim]
            out_fresh = F.scaled_dot_product_attention(
                q_fresh, k, v, dropout_p=0.0, is_causal=False
            )
            out_fresh = out_fresh.transpose(1, 2).flatten(2, 3).to(hidden_states.dtype)
            # [B, n_txt+N_fg, heads*head_dim]

            # Update image cache: fg positions get fresh values
            cached_img = self.cache[self.block_idx].clone()        # [B, N_img, heads*head_dim]
            cached_img[:, fg_idx] = out_fresh[:, n_txt:]
            self.cache[self.block_idx] = cached_img.detach()

            # Assemble: text (fresh) + image (fg fresh, bg cached)
            out_full = torch.cat([
                out_fresh[:, :n_txt],   # text tokens
                cached_img,             # image tokens
            ], dim=1)                   # [B, N_total, heads*head_dim]

            return out_full


class FluxSparseManager:
    """
    Installs sparse attention processors on FLUX single-stream transformer blocks.

    Usage:
        n_txt = prompt_embeds.shape[1]
        mgr = FluxSparseManager(transformer)
        mgr.install(fg_mask, dense_mode=True, n_txt=n_txt)
        # denoising loop ...
        mgr.set_dense_mode(False)   # switch to sparse mode
        # more steps ...
        mgr.remove()
    """

    def __init__(self, transformer, layer_indices=None):
        self.transformer = transformer
        blocks = transformer.single_transformer_blocks
        if layer_indices is None:
            layer_indices = list(range(len(blocks)))
        self.layer_indices = layer_indices
        self._procs = []
        self._orig_procs = []
        self._cache: dict = {}

    def install(self, fg_mask: torch.Tensor, dense_mode: bool = True, n_txt: int = 77):
        """
        Install sparse attention processors.

        Args:
            fg_mask:    (N_img,) bool tensor, True = foreground.
            dense_mode: if True, start in dense mode (populate cache first).
            n_txt:      number of text tokens in the block input (prepended to image).
        """
        self._procs.clear()
        self._orig_procs.clear()
        self._cache.clear()

        blocks = self.transformer.single_transformer_blocks
        for idx in self.layer_indices:
            proc = _FluxSparseAttnProcessor(fg_mask, self._cache, idx, n_txt)
            proc.dense_mode = dense_mode
            self._procs.append(proc)
            self._orig_procs.append(blocks[idx].attn.processor)
            blocks[idx].attn.set_processor(proc)

    def set_dense_mode(self, dense: bool):
        for proc in self._procs:
            proc.dense_mode = dense

    def remove(self):
        blocks = self.transformer.single_transformer_blocks
        for idx, orig in zip(self.layer_indices, self._orig_procs):
            blocks[idx].attn.set_processor(orig)
        self._procs.clear()
        self._orig_procs.clear()
        self._cache.clear()
