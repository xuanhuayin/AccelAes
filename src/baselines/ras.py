"""
RAS (Region-Adaptive Sampling) baseline implementation.

RAS accelerates DiT inference by identifying spatially active regions
via cross-attention activation magnitude and restricting full computation
to those regions. Inactive (background) regions reuse cached noise_pred
from the previous denoising step.

Key properties:
  - Mask is updated every step from cross-attention maps (dynamic, not fixed)
  - Progressive bg ratio: fraction of skipped tokens increases linearly
    from 0 at warmup end to target_ratio at the final step
  - Uses token-level sparsity (SparseBlockManager) for internal computation
  - Plus output-level bg blending for noise_pred

Reference: Region-Adaptive Sampling (arXiv 2502.10389, Feb 2025)
  "Focuses the full denoising computation on the regions the model is
  currently 'looking at' (high cross-attention activation), while other
  regions are updated using cached noise from the previous step."

Used as a comparison baseline for SASD.
"""

import math
import torch
from typing import Optional
from PIL import Image


def _build_fg_mask_from_attn(
    attn_maps: list,
    skip_ratio: float,
    patch_h: int,
    patch_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """
    Build a (N,) bool fg_mask and a (1,1,H,W) float spatial mask from
    cross-attention maps captured during a single forward pass.

    Args:
        attn_maps: list of (num_heads, num_patches, kv_len) tensors from
                   CrossAttnCaptureProcessor (one per hooked layer).
        skip_ratio: fraction of tokens to treat as background (0 = all fg).
        patch_h, patch_w: spatial patch grid dimensions.
        device, dtype: output tensor device/dtype.

    Returns:
        fg_token_mask: (N,) bool, True = foreground.
        fg_spatial:    (1, 1, patch_h, patch_w) float in [0,1].
    """
    N = patch_h * patch_w
    if not attn_maps or skip_ratio <= 0.0:
        fg_token_mask = torch.ones(N, dtype=torch.bool, device=device)
        fg_spatial = torch.ones(1, 1, patch_h, patch_w, device=device, dtype=dtype)
        return fg_token_mask, fg_spatial

    # Average importance over all captured layers and all heads.
    # Per-patch importance = mean over heads and kv-tokens of attn weight.
    importance_list = []
    for amap in attn_maps:
        # amap: (heads, patches, kv_len)  — cond branch only
        imp = amap.mean(dim=(0, 2))  # (patches,)
        importance_list.append(imp)

    importance = torch.stack(importance_list, dim=0).mean(dim=0)  # (N,)

    # Threshold: keep top (1 - skip_ratio) fraction as foreground
    n_fg = max(1, int(round(N * (1.0 - skip_ratio))))
    if n_fg >= N:
        fg_token_mask = torch.ones(N, dtype=torch.bool, device=device)
        fg_spatial = torch.ones(1, 1, patch_h, patch_w, device=device, dtype=dtype)
        return fg_token_mask, fg_spatial

    threshold = torch.topk(importance, n_fg, largest=True).values[-1]
    fg_token_mask = (importance >= threshold).to(device)

    fg_spatial = fg_token_mask.float().reshape(1, 1, patch_h, patch_w).to(dtype)
    return fg_token_mask, fg_spatial


@torch.no_grad()
def generate_ras(
    wrapper,
    prompt: str,
    seed: int,
    target_ratio: float = 0.5,
    warmup_steps: int = 2,
    cfg_scale: float = 4.0,
    steps: int = 30,
    height: int = 1024,
    width: int = 1024,
    hook_every: int = 4,
    sparse_ffn: bool = True,
) -> Image.Image:
    """
    Generate an image using RAS (Region-Adaptive Sampling).

    Args:
        wrapper: LuminaDiTWrapper instance.
        prompt: text prompt.
        seed: random seed.
        target_ratio: maximum bg fraction (0.5 = skip 50% at the end).
        warmup_steps: initial dense steps (no sparsity).
        cfg_scale: classifier-free guidance scale.
        steps: denoising steps.
        height, width: output resolution.
        hook_every: subsample every N layers for cross-attn capture.
        sparse_ffn: if True, also sparsify FFN (not just attn1).

    Returns:
        PIL Image.
    """
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
    from src.sparse.cross_attn_hook import CrossAttnHookManager
    from src.sparse.sparse_ffn import SparseBlockManager

    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    # ---- 1. Encode prompt ----
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=True,
        negative_prompt="",
        num_images_per_prompt=1,
        device=device,
    )
    prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat(
        [prompt_attention_mask, negative_prompt_attention_mask], dim=0
    )

    # ---- 2. Cross-attention kwargs ----
    cross_attention_kwargs = {}
    default_sample_size = getattr(pipe, "default_sample_size", 128)
    cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

    scaling_factor = math.sqrt(width * height / wrapper.default_image_size ** 2)

    # ---- 3. Prepare latents ----
    latent_h = height // wrapper.vae_scale_factor
    latent_w = width // wrapper.vae_scale_factor
    patch_h = latent_h // 2  # Lumina patch size = 2
    patch_w = latent_w // 2
    latent_channels = wrapper.transformer.config.in_channels
    shape = (1, latent_channels, latent_h, latent_w)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # ---- 4. Scheduler ----
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    head_dim = wrapper.transformer.head_dim
    num_layers = len(wrapper.transformer.layers)
    hook_layers = list(range(0, num_layers, hook_every))

    # ---- 5. State ----
    block_mgr = None
    noise_pred_cache = None  # cached output noise_pred for bg blending
    scaling_watershed = 1.0

    # ---- 6. Denoising loop ----
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor(
                [current_timestep], dtype=torch.float64, device=device
            )
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        current_timestep = (
            1 - current_timestep / pipe.scheduler.config.num_train_timesteps
        )

        if current_timestep[0] < scaling_watershed:
            linear_factor = scaling_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scaling_factor

        image_rotary_emb = get_2d_rotary_pos_embed_lumina(
            head_dim, 384, 384,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
        )

        # Install cross-attn capture on every step
        hook_mgr = CrossAttnHookManager(
            wrapper.transformer, layer_indices=hook_layers
        )
        hook_mgr.install()

        # Sparse mode after warm-up
        if block_mgr is not None and i > warmup_steps:
            block_mgr.set_dense_mode(False)

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_mask=prompt_attention_mask,
            image_rotary_emb=image_rotary_emb,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        hook_mgr.remove()

        # ---- Build/update mask from this step's cross-attn ----
        active_steps = steps - warmup_steps
        steps_past_warmup = i - warmup_steps
        if i <= warmup_steps or active_steps <= 0:
            current_ratio = 0.0
        else:
            current_ratio = target_ratio * min(1.0, steps_past_warmup / active_steps)

        attn_maps = hook_mgr.get_maps()
        fg_token_mask, fg_spatial = _build_fg_mask_from_attn(
            attn_maps, current_ratio, patch_h, patch_w, device, dtype
        )

        # Install or update SparseBlockManager
        if block_mgr is None:
            if i == warmup_steps:
                # First non-warmup step: install with dense_mode=True (populate cache)
                block_mgr = SparseBlockManager(wrapper.transformer)
                block_mgr.install(
                    fg_mask=fg_token_mask,
                    dense_mode=True,
                    sparse_ffn=sparse_ffn,
                )
        else:
            # Update mask in-place (clears attn+FFN caches so next step repopulates)
            block_mgr.update_mask(fg_token_mask)

        # ---- CFG ----
        noise_pred = noise_pred.chunk(2, dim=1)[0]
        noise_pred_eps = noise_pred[:, :3]
        noise_pred_rest = noise_pred[:, 3:]
        noise_pred_cond, noise_pred_uncond = torch.split(
            noise_pred_eps, len(noise_pred_eps) // 2, dim=0
        )
        guided = noise_pred_uncond + cfg_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        noise_pred_eps = torch.cat([guided, guided], dim=0)
        noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
        noise_pred, _ = noise_pred.chunk(2, dim=0)
        noise_pred = -noise_pred

        # ---- Output-level bg blending (RAS: inactive region uses cached noise) ----
        # fg_spatial is at patch resolution; noise_pred is at latent resolution (2×)
        if noise_pred_cache is not None and i > warmup_steps:
            fg_latent = torch.nn.functional.interpolate(
                fg_spatial, size=(latent_h, latent_w), mode="nearest"
            )
            noise_pred = fg_latent * noise_pred + (1.0 - fg_latent) * noise_pred_cache

        noise_pred_cache = noise_pred.detach().clone()

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # ---- 7. Cleanup ----
    if block_mgr is not None:
        block_mgr.remove()

    # ---- 8. Decode ----
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
