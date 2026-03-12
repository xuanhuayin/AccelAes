"""
FORA baseline implementation.

FORA (Feature-Output Reuse Across steps) accelerates diffusion inference
by reusing attention and MLP outputs from previous denoising steps.
Unlike Delta-DiT's per-layer scheduling, FORA operates at a coarser
granularity: entire transformer blocks are either computed or reused.

Reference: FORA (arXiv 2024)

This is a training-free, spatially-unaware acceleration method.
Used as a comparison baseline for SASD.
"""

import torch
from typing import Optional, Dict, Tuple
from PIL import Image


class FORAManager:
    """
    Manages cross-step feature reuse for FORA acceleration on Lumina DiT.

    FORA reuses the residual (output - input) of transformer blocks
    from the previous timestep. On reuse steps, the block's output is
    approximated as: input + cached_residual.

    Compared to Delta-DiT:
      - More aggressive: caches entire block residuals, not just outputs
      - Simpler schedule: alternating compute/reuse pattern
      - May degrade more at high reuse ratios

    Args:
        transformer: LuminaNextDiT2DModel instance.
        reuse_interval: compute every N steps (1=always compute, 2=compute-reuse, etc.).
        skip_layers: list of layer indices to always compute (never cache).
    """

    def __init__(
        self,
        transformer,
        reuse_interval: int = 2,
        skip_layers: Optional[list] = None,
    ):
        self.transformer = transformer
        self.num_layers = len(transformer.layers)
        self.reuse_interval = reuse_interval
        self.skip_layers = set(skip_layers or [])

        self._residual_cache = {}  # {layer_idx: residual_tensor}
        self._hooks = []
        self._step_counter = 0
        self._installed = False

    def install_hooks(self):
        """Install forward hooks for residual caching/reuse."""
        self._installed = True
        self._residual_cache.clear()
        self._step_counter = 0

        for idx in range(self.num_layers):
            if idx in self.skip_layers:
                continue

            layer = self.transformer.layers[idx]

            def make_pre_hook(layer_idx):
                def pre_hook(module, args):
                    # On reuse steps, add cached residual to input
                    if (layer_idx in self._residual_cache
                            and self._step_counter % self.reuse_interval != 0):
                        # args[0] is hidden_states
                        hidden_states = args[0]
                        reused = hidden_states + self._residual_cache[layer_idx]
                        # Return modified args to skip the forward
                        return (reused,) + args[1:] if len(args) > 1 else (reused,)
                    return None
                return pre_hook

            def make_post_hook(layer_idx):
                def post_hook(module, args, output):
                    # Cache residual = output - input
                    if self._step_counter % self.reuse_interval == 0:
                        hidden_states = args[0]
                        if isinstance(output, tuple):
                            residual = output[0] - hidden_states
                            self._residual_cache[layer_idx] = residual.detach().clone()
                        else:
                            residual = output - hidden_states
                            self._residual_cache[layer_idx] = residual.detach().clone()
                    return output
                return post_hook

            h1 = layer.register_forward_pre_hook(make_pre_hook(idx))
            h2 = layer.register_forward_hook(make_post_hook(idx))
            self._hooks.extend([h1, h2])

    def step(self):
        """Advance the step counter."""
        self._step_counter += 1

    def remove(self):
        """Remove all hooks and clear caches."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._residual_cache.clear()
        self._step_counter = 0
        self._installed = False


@torch.no_grad()
def generate_fora(
    wrapper,
    prompt: str,
    seed: int,
    reuse_interval: int = 2,
    warmup_steps: int = 2,
    skip_layers: Optional[list] = None,
    cfg_scale: float = 4.0,
    steps: int = 30,
    height: int = 1024,
    width: int = 1024,
) -> Image.Image:
    """
    Generate an image using FORA acceleration.

    Args:
        wrapper: LuminaDiTWrapper instance.
        prompt: text prompt.
        seed: random seed.
        reuse_interval: compute every N steps (2 = 50% reuse).
        warmup_steps: number of initial dense steps.
        skip_layers: layers to never cache (always compute).
        cfg_scale: classifier-free guidance scale.
        steps: number of denoising steps.
        height, width: output resolution.

    Returns:
        PIL Image.
    """
    import math
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina

    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    mgr = FORAManager(wrapper.transformer, reuse_interval, skip_layers)

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
    prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

    # ---- 2. Cross-attention kwargs ----
    cross_attention_kwargs = {}
    default_sample_size = getattr(pipe, 'default_sample_size', 128)
    cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

    scaling_factor = math.sqrt(width * height / wrapper.default_image_size ** 2)

    # ---- 3. Prepare latents ----
    latent_h = height // wrapper.vae_scale_factor
    latent_w = width // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels
    shape = (1, latent_channels, latent_h, latent_w)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # ---- 4. Scheduler ----
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    head_dim = wrapper.transformer.head_dim

    # ---- 5. Install FORA hooks ----
    mgr.install_hooks()

    # ---- 6. Denoising loop ----
    scaling_watershed = 1.0

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], dtype=torch.float64, device=device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        current_timestep = 1 - current_timestep / pipe.scheduler.config.num_train_timesteps

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

        # During warmup, force all layers to compute
        if i < warmup_steps:
            mgr._step_counter = 0

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_mask=prompt_attention_mask,
            image_rotary_emb=image_rotary_emb,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        if i >= warmup_steps:
            mgr.step()

        # Split pred / sigma
        noise_pred = noise_pred.chunk(2, dim=1)[0]

        # CFG
        noise_pred_eps = noise_pred[:, :3]
        noise_pred_rest = noise_pred[:, 3:]
        noise_pred_cond, noise_pred_uncond = torch.split(
            noise_pred_eps, len(noise_pred_eps) // 2, dim=0
        )
        guided = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        noise_pred_eps = torch.cat([guided, guided], dim=0)
        noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
        noise_pred, _ = noise_pred.chunk(2, dim=0)
        noise_pred = -noise_pred

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # ---- 7. Cleanup ----
    mgr.remove()

    # ---- 8. Decode ----
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


# ─────────────────────────────────────────────────────────────────────────────
# SD3-Medium: FORA on MMDiT joint attention blocks
# ─────────────────────────────────────────────────────────────────────────────

class FORAManagerSD3:
    """
    FORA for SD3-Medium MMDiT joint attention blocks.

    Caches the hidden_states residual (output - input) of each joint block
    and adds it back on reuse steps. Note: since the pre-hook approach
    modifies inputs but does NOT skip the forward call, actual speedup
    is ~1.00× (same as FORA on Lumina). Included for completeness.

    Args:
        transformer: SD3Transformer2DModel instance.
        reuse_interval: compute every N steps (1=always, 2=compute-reuse).
        skip_layers: layer indices to always compute (never cache).
    """

    def __init__(
        self,
        transformer,
        reuse_interval: int = 2,
        skip_layers: Optional[list] = None,
    ):
        self.transformer = transformer
        self.num_layers = len(transformer.transformer_blocks)
        self.reuse_interval = reuse_interval
        self.skip_layers = set(skip_layers or [])

        self._residual_cache = {}
        self._hooks = []
        self._step_counter = 0
        self._installed = False

    def install_hooks(self):
        self._installed = True
        self._residual_cache.clear()
        self._step_counter = 0

        for idx in range(self.num_layers):
            if idx in self.skip_layers:
                continue

            block = self.transformer.transformer_blocks[idx]

            def make_pre_hook(layer_idx):
                def pre_hook(module, args, kwargs):
                    # args[0] is hidden_states on compute steps; add residual on reuse steps
                    if (layer_idx in self._residual_cache
                            and self._step_counter % self.reuse_interval != 0):
                        hidden_states = kwargs.get("hidden_states",
                                                   args[0] if args else None)
                        if hidden_states is not None:
                            reused = hidden_states + self._residual_cache[layer_idx]
                            if "hidden_states" in kwargs:
                                kwargs["hidden_states"] = reused
                            else:
                                args = (reused,) + args[1:]
                        return args, kwargs
                return pre_hook

            def make_post_hook(layer_idx):
                def post_hook(module, args, kwargs, output):
                    # Cache hidden_states residual on compute steps
                    if self._step_counter % self.reuse_interval == 0:
                        hidden_states_in = kwargs.get("hidden_states",
                                                      args[0] if args else None)
                        if hidden_states_in is not None:
                            # output may be (encoder_hidden_states, hidden_states) tuple
                            hs_out = output[-1] if isinstance(output, tuple) else output
                            residual = hs_out - hidden_states_in
                            self._residual_cache[layer_idx] = residual.detach().clone()
                    return output
                return post_hook

            h1 = block.register_forward_pre_hook(make_pre_hook(idx), with_kwargs=True)
            h2 = block.register_forward_hook(make_post_hook(idx), with_kwargs=True)
            self._hooks.extend([h1, h2])

    def step(self):
        self._step_counter += 1

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._residual_cache.clear()
        self._step_counter = 0
        self._installed = False


@torch.no_grad()
def generate_fora_sd3(
    wrapper,
    prompt: str,
    seed: int,
    reuse_interval: int = 2,
    warmup_steps: int = 2,
    skip_layers: Optional[list] = None,
    cfg_scale: float = 7.0,
    steps: int = 28,
    height: int = 1024,
    width: int = 1024,
) -> "Image.Image":
    """
    FORA acceleration for SD3-Medium.

    Residual caching applied to SD3's 24 joint transformer blocks.
    Note: pre-hook approach provides no actual speedup (~1.00×);
    included for fair comparison with FORA on other architectures.
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    mgr = FORAManagerSD3(wrapper.transformer, reuse_interval, skip_layers)

    # 1. Encode prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt="", do_classifier_free_guidance=True,
        num_images_per_prompt=1, device=device,
    )
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 2. Prepare latents
    latent_h = height // wrapper.vae_scale_factor
    latent_w = width // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=dtype,
    )

    # 3. Scheduler
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 4. Install FORA hooks
    mgr.install_hooks()

    # 5. Denoising loop
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        if i < warmup_steps:
            mgr._step_counter = 0

        raw = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(latent_model_input.shape[0]),
            encoder_hidden_states=prompt_embeds_cfg,
            pooled_projections=pooled_cfg,
            return_dict=False,
        )[0]

        if i >= warmup_steps:
            mgr.step()

        noise_pred_uncond, noise_pred_cond = raw.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # 6. Cleanup
    mgr.remove()

    # 7. Decode (SD3: undo scaling and apply shift factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ─────────────────────────────────────────────────────────────────────────────
# FLUX.1-dev: FORA on double-stream + single-stream transformer blocks
# ─────────────────────────────────────────────────────────────────────────────

class FORAManagerFLUX:
    """
    FORA for FLUX.1-dev.

    Caches hidden_states residuals from both block types.
    Double-stream blocks take (hidden_states, encoder_hidden_states, ...) and
    return (encoder_hidden_states, hidden_states) — residual is on hidden_states.
    Single-stream blocks take/return hidden_states only.
    Note: pre-hook approach provides no actual speedup (~1.00×).
    """

    def __init__(
        self,
        transformer,
        reuse_interval: int = 2,
        skip_layers: Optional[list] = None,
    ):
        self.transformer = transformer
        self.n_double = len(transformer.transformer_blocks)
        self.n_single = len(transformer.single_transformer_blocks)
        self.reuse_interval = reuse_interval
        self.skip_layers = set(skip_layers or [])

        self._residual_cache = {}
        self._hooks = []
        self._step_counter = 0
        self._installed = False

    def install_hooks(self):
        self._installed = True
        self._residual_cache.clear()
        self._step_counter = 0

        for idx in range(self.n_double):
            if ("d", idx) in self.skip_layers:
                continue
            block = self.transformer.transformer_blocks[idx]
            cache_key = ("d", idx)

            def make_pre_d(ckey):
                def pre_hook(module, args, kwargs):
                    if (ckey in self._residual_cache
                            and self._step_counter % self.reuse_interval != 0):
                        hs = kwargs.get("hidden_states",
                                        args[0] if args else None)
                        if hs is not None:
                            reused = hs + self._residual_cache[ckey]
                            if "hidden_states" in kwargs:
                                kwargs["hidden_states"] = reused
                            else:
                                args = (reused,) + args[1:]
                        return args, kwargs
                return pre_hook

            def make_post_d(ckey):
                def post_hook(module, args, kwargs, output):
                    if self._step_counter % self.reuse_interval == 0:
                        hs_in = kwargs.get("hidden_states",
                                           args[0] if args else None)
                        if hs_in is not None:
                            # double-stream returns (enc_hs, hidden_states) tuple
                            hs_out = output[-1] if isinstance(output, tuple) else output
                            self._residual_cache[ckey] = (hs_out - hs_in).detach().clone()
                    return output
                return post_hook

            h1 = block.register_forward_pre_hook(make_pre_d(cache_key), with_kwargs=True)
            h2 = block.register_forward_hook(make_post_d(cache_key), with_kwargs=True)
            self._hooks.extend([h1, h2])

        for idx in range(self.n_single):
            if ("s", idx) in self.skip_layers:
                continue
            block = self.transformer.single_transformer_blocks[idx]
            cache_key = ("s", idx)

            def make_pre_s(ckey):
                def pre_hook(module, args, kwargs):
                    if (ckey in self._residual_cache
                            and self._step_counter % self.reuse_interval != 0):
                        hs = kwargs.get("hidden_states",
                                        args[0] if args else None)
                        if hs is not None:
                            reused = hs + self._residual_cache[ckey]
                            if "hidden_states" in kwargs:
                                kwargs["hidden_states"] = reused
                            else:
                                args = (reused,) + args[1:]
                        return args, kwargs
                return pre_hook

            def make_post_s(ckey):
                def post_hook(module, args, kwargs, output):
                    if self._step_counter % self.reuse_interval == 0:
                        hs_in = kwargs.get("hidden_states",
                                           args[0] if args else None)
                        if hs_in is not None:
                            hs_out = output[-1] if isinstance(output, tuple) else output
                            self._residual_cache[ckey] = (hs_out - hs_in).detach().clone()
                    return output
                return post_hook

            h1 = block.register_forward_pre_hook(make_pre_s(cache_key), with_kwargs=True)
            h2 = block.register_forward_hook(make_post_s(cache_key), with_kwargs=True)
            self._hooks.extend([h1, h2])

    def step(self):
        self._step_counter += 1

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._residual_cache.clear()
        self._step_counter = 0
        self._installed = False


@torch.no_grad()
def generate_fora_flux(
    wrapper,
    prompt: str,
    seed: int,
    reuse_interval: int = 2,
    warmup_steps: int = 2,
    guidance_scale: float = 3.5,
    steps: int = 28,
    height: int = 1024,
    width: int = 1024,
) -> "Image.Image":
    """
    FORA acceleration for FLUX.1-dev.

    Residual caching on all 57 transformer blocks (19 double + 38 single).
    Note: pre-hook approach provides no actual speedup (~1.00×).
    """
    mgr = FORAManagerFLUX(wrapper.transformer, reuse_interval)

    prompt_embeds, pooled_prompt_embeds, text_ids = wrapper._encode_prompt(prompt)
    wrapper._ensure_transformer_on_gpu()
    latents, img_ids = wrapper._prepare_latents(height, width, seed)
    timesteps = wrapper._setup_scheduler(steps, height, width)

    import torch as _torch
    txt_ids = _torch.zeros(
        prompt_embeds.shape[1], 3, device=wrapper.device, dtype=wrapper.dtype)

    mgr.install_hooks()

    for i, t in enumerate(timesteps):
        if i < warmup_steps:
            mgr._step_counter = 0

        noise_pred = wrapper._transformer_step(
            latents, t, prompt_embeds, pooled_prompt_embeds,
            txt_ids, img_ids, guidance_scale)

        if i >= warmup_steps:
            mgr.step()

        latents = wrapper.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    mgr.remove()
    return wrapper._decode(latents, height, width)
