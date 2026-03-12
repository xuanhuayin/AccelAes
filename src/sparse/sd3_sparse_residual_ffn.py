"""
Sparse Residual FFN for SD3 JointTransformerBlock — Option 2.

Problem with naive sparse FFN (sd3_sparse_ffn.py):
  block.ff receives norm_h = norm2(h) * (1+scale_mlp) + shift_mlp
  This input is strongly timestep-dependent → cached ff(norm_h_prev) is stale.

Fix: cache the FULL residual contribution  gate_mlp * ff(norm_h)
  Dense mode : compute full FF, cache gate_mlp_t * ff(norm_h_t) for bg tokens.
  Sparse mode:
    - fg tokens: fresh  gate_mlp_t * ff(norm_h_fg_t)
    - bg tokens: cached gate_mlp_prev * ff(norm_h_bg_prev)  (reused verbatim)

The bg residual is added directly; gate_mlp is NOT applied again.
This avoids the mismatch of applying the current gate to a stale ff_output.

Implementation: monkey-patch block.forward (closure over block, mask, cache).
"""

import torch
import torch.nn.functional as F


def _make_sparse_residual_forward(block, fg_mask: torch.Tensor,
                                   layer_idx: int, cache: dict,
                                   dense_ref: list):
    """
    Build a replacement forward method for a JointTransformerBlock that
    caches gate_mlp * ff(norm_h) for background image tokens.

    Args:
        block     : JointTransformerBlock instance.
        fg_mask   : (N_img,) bool, True = foreground token.
        layer_idx : int, cache key.
        cache     : shared dict {layer_idx: Tensor(B, N_bg, D)}.
        dense_ref : list of length 1 holding current dense_mode bool.
    """

    def _try_chunked_ff(ff_module, x, block_):
        if getattr(block_, '_chunk_size', None) is not None:
            try:
                from diffusers.models.transformers.transformer_sd3 import _chunked_feed_forward
                return _chunked_feed_forward(ff_module, x, block_._chunk_dim, block_._chunk_size)
            except ImportError:
                pass
        return ff_module(x)

    def patched_forward(hidden_states, encoder_hidden_states, temb,
                        joint_attention_kwargs=None):
        joint_attention_kwargs = joint_attention_kwargs or {}

        # ─── Attention (identical to JointTransformerBlock.forward) ───
        if block.use_dual_attention:
            (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp,
             norm_hidden_states2, gate_msa2) = block.norm1(hidden_states, emb=temb)
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                block.norm1(hidden_states, emb=temb)

        if block.context_pre_only:
            norm_encoder_hidden_states = block.norm1_context(encoder_hidden_states, temb)
        else:
            (norm_encoder_hidden_states, c_gate_msa,
             c_shift_mlp, c_scale_mlp, c_gate_mlp) = \
                block.norm1_context(encoder_hidden_states, emb=temb)

        attn_output, context_attn_output = block.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if block.use_dual_attention:
            attn_output2 = block.attn2(
                hidden_states=norm_hidden_states2, **joint_attention_kwargs)
            hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

        # ─── Image FFN with sparse residual caching ───
        fg_idx = fg_mask.nonzero(as_tuple=True)[0]   # (N_fg,)
        bg_idx = (~fg_mask).nonzero(as_tuple=True)[0]

        norm_h = block.norm2(hidden_states)
        norm_h = norm_h * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if dense_ref[0] or (layer_idx not in cache):
            # Dense: full computation, cache bg residual
            ff_out = _try_chunked_ff(block.ff, norm_h, block)
            ff_residual = gate_mlp.unsqueeze(1) * ff_out  # (B, N_img, D)
            cache[layer_idx] = ff_residual[:, bg_idx].detach().clone()
            hidden_states = hidden_states + ff_residual
        else:
            # Sparse: fg = current gate × fresh ff, bg = cached full residual
            ff_out_fg = block.ff(norm_h[:, fg_idx])                  # (B, N_fg, D)
            ff_residual_fg = gate_mlp.unsqueeze(1) * ff_out_fg       # apply current gate to fg

            hs = hidden_states.clone()
            hs[:, fg_idx] = hs[:, fg_idx] + ff_residual_fg
            hs[:, bg_idx] = hs[:, bg_idx] + cache[layer_idx]
            hidden_states = hs

        # ─── Context (text) FFN (identical to original) ───
        if block.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_enc = block.norm2_context(encoder_hidden_states)
            norm_enc = norm_enc * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            ctx_ff_out = _try_chunked_ff(block.ff_context, norm_enc, block)
            encoder_hidden_states = (encoder_hidden_states
                                     + c_gate_mlp.unsqueeze(1) * ctx_ff_out)

        return encoder_hidden_states, hidden_states

    return patched_forward


class SD3SparseResidualFFNManager:
    """
    Installs sparse residual FFN on selected SD3 transformer blocks by
    monkey-patching block.forward with a closure that caches gate*ff outputs
    for background tokens.

    Usage:
        mgr = SD3SparseResidualFFNManager(transformer)
        mgr.install(fg_mask, dense_mode=True)   # warmup
        # ... 2 warmup steps ...
        mgr.set_dense_mode(False)               # switch to sparse
        # ... remaining denoising steps ...
        mgr.remove()
    """

    def __init__(self, transformer, layer_indices=None):
        self.transformer = transformer
        blocks = transformer.transformer_blocks
        self.num_layers = len(blocks)
        self.layer_indices = (layer_indices if layer_indices is not None
                              else list(range(self.num_layers)))
        self._orig_forwards: dict = {}   # {idx: original forward}
        self._cache: dict = {}           # {idx: Tensor(B, N_bg, D)}
        self._dense_ref: list = [True]   # mutable for closure

    def install(self, fg_mask: torch.Tensor, dense_mode: bool = True):
        self._cache.clear()
        self._dense_ref[0] = dense_mode
        blocks = self.transformer.transformer_blocks
        for idx in self.layer_indices:
            block = blocks[idx]
            self._orig_forwards[idx] = block.forward
            block.forward = _make_sparse_residual_forward(
                block, fg_mask, idx, self._cache, self._dense_ref
            )

    def set_dense_mode(self, dense: bool):
        self._dense_ref[0] = dense

    def remove(self):
        blocks = self.transformer.transformer_blocks
        for idx, orig in self._orig_forwards.items():
            blocks[idx].forward = orig
        self._orig_forwards.clear()
        self._cache.clear()
