"""
TeaCache baseline implementation for Lumina-Next-T2I.

TeaCache (arXiv 2411.19108) is an adaptive step-caching method that tracks
the change in model outputs between consecutive timesteps and skips recomputation
when the predicted change falls below a threshold.

Our implementation uses the per-step change in noise_pred as the skip criterion:
  - Maintain the last two computed noise predictions.
  - Estimate per-step change rate as ||p_last - p_prev|| / ||p_prev||.
  - Accumulate expected change since the last compute step.
  - Skip the transformer when accumulated expected change < threshold.
  - Reuse the last computed noise_pred (0th-order copy).

Threshold calibration (approximate, 30 steps, Lumina):
  threshold=0.06  → ~2×  speedup
  threshold=0.12  → ~3×  speedup

Reference: TeaCache — Timestep Embedding Aware Cache for Video/Image DiTs
  arXiv 2411.19108 (CVPR 2025 Highlight)
"""

import math
import torch
from PIL import Image


@torch.no_grad()
def generate_teacache_lumina(
    wrapper,
    prompt: str,
    seed: int,
    cfg_scale: float = 4.0,
    steps: int = 30,
    height: int = 1024,
    width: int = 1024,
    threshold: float = 0.06,
    warmup_steps: int = 3,
) -> Image.Image:
    """
    TeaCache adaptive step caching for Lumina-Next-T2I.

    Args:
        wrapper:      LuminaDiTWrapper instance.
        prompt:       Text prompt.
        seed:         Random seed.
        cfg_scale:    Classifier-free guidance scale.
        steps:        Number of denoising steps.
        height/width: Output resolution.
        threshold:    Accumulated change threshold for skipping (default 0.06 ≈ 2×).
        warmup_steps: Number of initial fully-computed steps.

    Returns:
        PIL Image.
    """
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina

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
    default_sample_size = getattr(pipe, "default_sample_size", 128)
    cross_attention_kwargs = {
        "base_sequence_length": (default_sample_size // 2) ** 2
    }
    scaling_factor = math.sqrt(width * height / wrapper.default_image_size ** 2)

    # ---- 3. Prepare latents ----
    latent_channels = wrapper.transformer.config.in_channels
    shape = (1, latent_channels, height // wrapper.vae_scale_factor,
             width // wrapper.vae_scale_factor)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # ---- 4. Scheduler ----
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    head_dim = wrapper.transformer.head_dim
    scaling_watershed = 1.0

    # ---- 5. TeaCache state ----
    # Track last two COMPUTED noise_preds to estimate per-step change rate
    prev_pred_1 = None   # most recent computed noise_pred
    prev_pred_2 = None   # second-most-recent computed noise_pred
    acc = 0.0            # accumulated expected change since last compute

    # ---- 6. Denoising loop ----
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor(
                [current_timestep], dtype=torch.float64, device=device)
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

        # ---- Estimate per-step change rate (from last two computed) ----
        per_step_delta = 0.0
        if prev_pred_2 is not None:
            per_step_delta = float(
                (prev_pred_1 - prev_pred_2).norm() /
                (prev_pred_2.norm() + 1e-8)
            )

        # ---- Skip decision ----
        should_skip = (
            prev_pred_1 is not None
            and i >= warmup_steps
            and acc + per_step_delta < threshold
        )

        if should_skip:
            # Reuse last computed noise_pred
            noise_pred = prev_pred_1
            acc += per_step_delta
        else:
            # Full transformer forward
            raw = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Split pred / sigma
            raw = raw.chunk(2, dim=1)[0]

            # Uniform CFG
            raw_eps = raw[:, :3]
            raw_rest = raw[:, 3:]
            cond, uncond = torch.split(raw_eps, len(raw_eps) // 2, dim=0)
            guided = uncond + cfg_scale * (cond - uncond)
            raw_eps = torch.cat([guided, guided], dim=0)
            raw = torch.cat([raw_eps, raw_rest], dim=1)
            raw, _ = raw.chunk(2, dim=0)
            noise_pred = -raw

            # Update TeaCache state
            prev_pred_2 = prev_pred_1
            prev_pred_1 = noise_pred.clone()
            acc = 0.0

        latents = pipe.scheduler.step(
            noise_pred, t, latents, return_dict=False)[0]

    # ---- 7. Decode ----
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.no_grad()
def generate_taylor_lumina(
    wrapper,
    prompt: str,
    seed: int,
    cfg_scale: float = 4.0,
    steps: int = 30,
    height: int = 1024,
    width: int = 1024,
    taylor_run: int = 2,
    warmup_steps: int = 5,
    taylor_order: int = 2,
) -> Image.Image:
    """
    TaylorSeer-style fixed-cycle step caching for Lumina-Next-T2I (arXiv 2503.06923).

    Scheduling: first `warmup_steps` steps always computed; then [compute, skip×taylor_run]
    cycles repeat. Skipped steps use TaylorSkipCache (Taylor extrapolation of noise_pred).
    No spatial CFG — uniform guidance throughout (matches TeaCache standalone setup).

    Args:
        wrapper:      LuminaDiTWrapper instance.
        prompt:       Text prompt.
        seed:         Random seed.
        cfg_scale:    Classifier-free guidance scale (default 4.0).
        steps:        Number of denoising steps (default 30).
        taylor_run:   Skip steps per compute step (default 2, i.e. 2× speedup in cycle).
        warmup_steps: Fully-computed warm-up steps before cycling (default 5).
        taylor_order: Taylor expansion order: 1=linear, 2=quadratic (default).

    Returns:
        PIL Image.
    """
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
    from src.sparse.skip_cache import TaylorSkipCache

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
    default_sample_size = getattr(pipe, "default_sample_size", 128)
    cross_attention_kwargs = {
        "base_sequence_length": (default_sample_size // 2) ** 2
    }
    scaling_factor = math.sqrt(width * height / wrapper.default_image_size ** 2)

    # ---- 3. Prepare latents ----
    latent_channels = wrapper.transformer.config.in_channels
    shape = (1, latent_channels, height // wrapper.vae_scale_factor,
             width // wrapper.vae_scale_factor)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # ---- 4. Scheduler ----
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    head_dim = wrapper.transformer.head_dim
    scaling_watershed = 1.0

    # ---- 5. TaylorSeer cache ----
    cache = TaylorSkipCache(order=taylor_order)
    warmup = max(2, warmup_steps)
    run_len = 1 + taylor_run   # [compute, skip×taylor_run] cycle length

    # ---- 6. Denoising loop ----
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor(
                [current_timestep], dtype=torch.float64, device=device)
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

        # ---- TaylorSeer skip decision ----
        if i >= warmup and cache.ready():
            pos_in_run = (i - warmup) % run_len
            if pos_in_run > 0:
                # Skip step: Taylor-extrapolate
                noise_pred = cache.predict(target_step=i)
                latents = pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]
                continue

        # Full transformer forward
        raw = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_mask=prompt_attention_mask,
            image_rotary_emb=image_rotary_emb,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # Split pred / sigma
        raw = raw.chunk(2, dim=1)[0]

        # Uniform CFG
        raw_eps = raw[:, :3]
        raw_rest = raw[:, 3:]
        cond, uncond = torch.split(raw_eps, len(raw_eps) // 2, dim=0)
        guided = uncond + cfg_scale * (cond - uncond)
        raw_eps = torch.cat([guided, guided], dim=0)
        raw = torch.cat([raw_eps, raw_rest], dim=1)
        raw, _ = raw.chunk(2, dim=0)
        noise_pred = -raw

        cache.store(i, noise_pred)
        latents = pipe.scheduler.step(
            noise_pred, t, latents, return_dict=False)[0]

    # ---- 7. Decode ----
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ─────────────────────────────────────────────────────────────────────────────
# SD3-Medium: TeaCache and TaylorSeer
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_teacache_sd3(
    wrapper,
    prompt: str,
    seed: int,
    cfg_scale: float = 7.0,
    steps: int = 28,
    height: int = 1024,
    width: int = 1024,
    threshold: float = 0.15,
    warmup_steps: int = 3,
) -> "Image.Image":
    """
    TeaCache adaptive step caching for SD3-Medium (arXiv 2411.19108).

    Tracks noise_pred change between consecutive computed steps.
    When accumulated change < threshold, reuses the last noise_pred.
    Uniform CFG throughout (no spatial CFG).
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    # 1. Encode prompt
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt="", do_classifier_free_guidance=True,
        num_images_per_prompt=1, device=device,
    )
    # SD3 batch order: [uncond, cond]
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 2. Latents
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

    # 4. TeaCache state
    prev_pred_1 = None   # most-recent computed noise_pred
    prev_pred_2 = None   # second-most-recent
    acc = 0.0

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        per_step_delta = 0.0
        if prev_pred_2 is not None:
            per_step_delta = float(
                (prev_pred_1 - prev_pred_2).norm() / (prev_pred_2.norm() + 1e-8)
            )

        should_skip = (
            prev_pred_1 is not None
            and i >= warmup_steps
            and acc + per_step_delta < threshold
        )

        if should_skip:
            noise_pred = prev_pred_1
            acc += per_step_delta
        else:
            raw = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=prompt_embeds_cfg,
                pooled_projections=pooled_cfg,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = raw.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            prev_pred_2 = prev_pred_1
            prev_pred_1 = noise_pred.clone()
            acc = 0.0

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Decode (SD3: undo scaling and apply shift factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.no_grad()
def generate_taylor_sd3(
    wrapper,
    prompt: str,
    seed: int,
    cfg_scale: float = 7.0,
    steps: int = 28,
    height: int = 1024,
    width: int = 1024,
    taylor_run: int = 2,
    warmup_steps: int = 5,
    taylor_order: int = 2,
) -> "Image.Image":
    """
    TaylorSeer fixed-cycle step caching for SD3-Medium (arXiv 2503.06923).

    Scheduling: first `warmup_steps` always computed; then [compute, skip×taylor_run].
    Skipped steps use Taylor extrapolation of noise_pred. Uniform CFG.
    """
    from src.sparse.skip_cache import TaylorSkipCache

    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    # 1. Encode prompt
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt="", do_classifier_free_guidance=True,
        num_images_per_prompt=1, device=device,
    )
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 2. Latents
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

    # 4. TaylorSeer cache
    cache = TaylorSkipCache(order=taylor_order)
    warmup = max(2, warmup_steps)
    run_len = 1 + taylor_run

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # Skip decision
        if i >= warmup and cache.ready():
            pos_in_run = (i - warmup) % run_len
            if pos_in_run > 0:
                noise_pred = cache.predict(target_step=i)
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                continue

        # Full transformer forward
        raw = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(latent_model_input.shape[0]),
            encoder_hidden_states=prompt_embeds_cfg,
            pooled_projections=pooled_cfg,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_cond = raw.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        cache.store(i, noise_pred)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Decode (SD3: undo scaling and apply shift factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
