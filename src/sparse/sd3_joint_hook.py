"""
SD3 Joint Attention Hook.

SD3's MMDiT uses joint attention where image tokens and text tokens
attend to each other simultaneously. To build a semantic mask, we
capture the image→text affinity (q_img @ k_txt^T) from selected blocks.

This avoids computing the full N×N attention matrix by only computing
the image→text cross-attention portion as an unnormalized affinity.
"""

import torch
from diffusers.models.attention_processor import Attention


class _SD3AffinityCapture:
    """
    Temporary replacement for JointAttnProcessor2_0.

    Computes the same output as the original processor but also captures
    the image→text affinity map (q_img @ k_txt.T / sqrt(head_dim)).

    Shape: (B, heads, N_img, N_txt)
    """

    def __init__(self):
        self.affinity = None  # set during forward

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        batch = hidden_states.shape[0]
        head_dim = attn.inner_dim // attn.heads

        def proj_and_reshape(proj, norm, x):
            out = proj(x)
            if norm is not None:
                out = norm(out)
            return out.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        # Image token projections
        q_img = proj_and_reshape(attn.to_q, getattr(attn, "norm_q", None), hidden_states)
        k_img = proj_and_reshape(attn.to_k, getattr(attn, "norm_k", None), hidden_states)
        v_img = proj_and_reshape(attn.to_v, None, hidden_states)

        # Text token projections
        q_txt = proj_and_reshape(attn.add_q_proj, getattr(attn, "norm_added_q", None), encoder_hidden_states)
        k_txt = proj_and_reshape(attn.add_k_proj, getattr(attn, "norm_added_k", None), encoder_hidden_states)
        v_txt = proj_and_reshape(attn.add_v_proj, None, encoder_hidden_states)

        # Capture image→text affinity (unnormalized, memory-efficient)
        # shape: (B, heads, N_img, N_txt)
        scale = head_dim ** -0.5
        with torch.no_grad():
            self.affinity = (q_img @ k_txt.transpose(-2, -1) * scale).detach()

        # Joint keys/values for attention output
        k_joint = torch.cat([k_img, k_txt], dim=2)
        v_joint = torch.cat([v_img, v_txt], dim=2)
        q_joint = torch.cat([q_img, q_txt], dim=2)

        # Standard SDPA for output (efficient, no weight storage)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_joint, k_joint, v_joint, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(hidden_states.dtype)

        n_img = hidden_states.shape[1]
        out_img = out[:, :n_img, :]
        out_txt = out[:, n_img:, :]

        # Output projections
        out_img = attn.to_out[0](out_img)
        out_img = attn.to_out[1](out_img)

        if hasattr(attn, "to_add_out") and attn.context_pre_only is False:
            out_txt = attn.to_add_out(out_txt)

        return out_img, out_txt


class SD3JointAttnHookManager:
    """
    Installs affinity-capture processors on selected SD3 transformer blocks.

    Usage:
        hook_mgr = SD3JointAttnHookManager(transformer, layer_indices=range(0, 24, 4))
        hook_mgr.install()
        # ... run transformer forward ...
        affinity_maps = hook_mgr.get_affinity_maps()  # list of (B, heads, N_img, N_txt)
        hook_mgr.remove()
    """

    def __init__(self, transformer, layer_indices=None):
        self.transformer = transformer
        blocks = transformer.transformer_blocks
        if layer_indices is None:
            layer_indices = range(0, len(blocks), 4)
        self.layer_indices = list(layer_indices)
        self._captures = []   # list of _SD3AffinityCapture instances
        self._orig_procs = []  # saved original processors

    def install(self):
        self._captures.clear()
        self._orig_procs.clear()
        blocks = self.transformer.transformer_blocks
        for idx in self.layer_indices:
            cap = _SD3AffinityCapture()
            self._captures.append(cap)
            self._orig_procs.append(blocks[idx].attn.processor)
            blocks[idx].attn.set_processor(cap)

    def remove(self):
        blocks = self.transformer.transformer_blocks
        for idx, orig in zip(self.layer_indices, self._orig_procs):
            blocks[idx].attn.set_processor(orig)
        self._orig_procs.clear()

    def get_affinity_maps(self):
        """Return list of captured (B, heads, N_img, N_txt) affinity tensors."""
        return [cap.affinity for cap in self._captures if cap.affinity is not None]
