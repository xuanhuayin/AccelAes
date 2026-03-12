"""
FLUX Joint Attention Hook.

Captures image→text attention affinity from FLUX double-stream blocks
(FluxTransformerBlock) for building semantic foreground masks.

Strategy: Use register_forward_pre_hook on the Attention module.
This does NOT replace the attention processor (avoids RoPE implementation
differences), and captures pre-RoPE Q_img @ K_txt^T affinity as a proxy
for semantic relevance. Pre-RoPE is fine for mask building (semantic
structure is preserved even without positional encoding).
"""

import torch


class FluxJointAttnHookManager:
    """
    Installs pre-hooks on selected FLUX double-stream block attention modules.

    The hook computes Q_img @ K_txt^T (pre-RoPE) without modifying the actual
    forward pass. This gives an image→text affinity map for mask building.

    Usage:
        hook_mgr = FluxJointAttnHookManager(transformer, layer_indices=[0, 3, 6, ...])
        hook_mgr.install()
        # run transformer forward (any number of times)
        affinity_maps = hook_mgr.get_affinity_maps()  # list of (B, heads, N_img, N_txt)
        hook_mgr.remove()
    """

    def __init__(self, transformer, layer_indices=None):
        self.transformer = transformer
        blocks = transformer.transformer_blocks
        if layer_indices is None:
            # Sample every 3rd block (6-7 blocks from 19 total)
            layer_indices = list(range(0, len(blocks), 3))
        self.layer_indices = list(layer_indices)
        self._captures = []   # list of {"affinity": tensor or None}
        self._hooks = []

    def install(self):
        self._captures.clear()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        blocks = self.transformer.transformer_blocks
        for idx in self.layer_indices:
            capture = {"affinity": None}
            self._captures.append(capture)

            def _make_hook(cap):
                def pre_hook(module, args, kwargs):
                    """
                    Pre-hook on Attention.forward().
                    Computes Q_img @ K_txt^T affinity using pre-RoPE projections.
                    Does NOT modify args/kwargs — returns None so forward proceeds normally.
                    """
                    hidden_states = kwargs.get("hidden_states")
                    encoder_hidden_states = kwargs.get("encoder_hidden_states")
                    if hidden_states is None or encoder_hidden_states is None:
                        return None

                    with torch.no_grad():
                        B = hidden_states.shape[0]
                        head_dim = module.inner_dim // module.heads

                        # Image Q (pre-RoPE) — reshape before norm (norm acts on head_dim)
                        q_img = module.to_q(hidden_states)
                        q_img = q_img.view(B, -1, module.heads, head_dim).transpose(1, 2)
                        if hasattr(module, "norm_q") and module.norm_q is not None:
                            q_img = module.norm_q(q_img)
                        # shape: (B, heads, N_img, head_dim)

                        # Text K (pre-RoPE) — reshape before norm
                        k_txt = module.add_k_proj(encoder_hidden_states)
                        k_txt = k_txt.view(B, -1, module.heads, head_dim).transpose(1, 2)
                        if hasattr(module, "norm_added_k") and module.norm_added_k is not None:
                            k_txt = module.norm_added_k(k_txt)
                        # shape: (B, heads, N_txt, head_dim)

                        # Image→text affinity: (B, heads, N_img, N_txt)
                        scale = head_dim ** -0.5
                        cap["affinity"] = (
                            q_img @ k_txt.transpose(-2, -1) * scale
                        ).detach().cpu()  # to CPU to save GPU memory

                    return None  # don't modify the forward call

                return pre_hook

            # with_kwargs=True requires torch >= 2.0
            h = blocks[idx].attn.register_forward_pre_hook(
                _make_hook(capture), with_kwargs=True
            )
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captures.clear()

    def get_affinity_maps(self):
        """
        Return list of captured (B, heads, N_img, N_txt) affinity tensors.
        Only returns maps that were actually captured (non-None).
        Moves tensors back to GPU if needed.
        """
        maps = []
        for cap in self._captures:
            if cap["affinity"] is not None:
                maps.append(cap["affinity"])
        return maps
