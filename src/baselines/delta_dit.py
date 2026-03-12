"""
Delta-DiT baseline implementation.

Delta-DiT accelerates DiT inference by caching transformer layer outputs
at different intervals depending on layer depth. Lower layers (which change
slowly across timesteps) are cached more aggressively, while upper layers
are recomputed more frequently.

Reference: Delta-DiT (arXiv 2024)

This is a training-free, spatially-unaware acceleration method.
Used as a comparison baseline for SASD.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from PIL import Image


class DeltaDiTManager:
    """
    Manages layer-level caching for Delta-DiT acceleration on Lumina DiT.

    Each transformer layer has a cache interval: the layer output is
    recomputed every `interval` steps and cached output is reused in between.

    Lower layers (early in the network) change less across timesteps and
    get larger cache intervals. Upper layers get smaller intervals.

    Args:
        transformer: LuminaNextDiT2DModel instance.
        cache_schedule: dict mapping layer index ranges to cache intervals.
            E.g., {(0, 8): 3, (8, 16): 2, (16, 24): 1}
            means layers 0-7 cache every 3 steps, 8-15 every 2, 16-23 every 1.
    """

    def __init__(
        self,
        transformer,
        cache_schedule: Optional[Dict[Tuple[int, int], int]] = None,
    ):
        self.transformer = transformer
        self.num_layers = len(transformer.layers)

        # Default schedule: 3 tiers
        if cache_schedule is None:
            third = self.num_layers // 3
            cache_schedule = {
                (0, third): 3,
                (third, 2 * third): 2,
                (2 * third, self.num_layers): 1,
            }

        # Expand schedule to per-layer intervals
        self._layer_intervals = {}
        for (start, end), interval in cache_schedule.items():
            for idx in range(start, min(end, self.num_layers)):
                self._layer_intervals[idx] = interval

        # Fill any missing layers with interval=1 (always compute)
        for idx in range(self.num_layers):
            if idx not in self._layer_intervals:
                self._layer_intervals[idx] = 1

        self._cache = {}          # {layer_idx: cached_output}
        self._original_forwards = {}  # {layer_idx: original forward method}
        self._step_counter = 0
        self._installed = False

    def install_hooks(self):
        """Patch forward methods on all transformer layers for caching.

        Uses method patching (not register_forward_pre_hook) because pre-hooks
        replace module *inputs*, not *outputs*. We need to return cached output
        directly and skip the forward computation entirely.
        """
        self._installed = True
        self._cache.clear()
        self._step_counter = 0
        self._original_forwards = {}

        for idx in range(self.num_layers):
            layer = self.transformer.layers[idx]
            self._original_forwards[idx] = layer.forward

            def make_patched_forward(layer_idx, orig_fwd):
                def patched_forward(*args, **kwargs):
                    interval = self._layer_intervals[layer_idx]
                    if (layer_idx in self._cache
                            and self._step_counter % interval != 0):
                        # Return cached output, skip computation
                        return self._cache[layer_idx]
                    output = orig_fwd(*args, **kwargs)
                    self._cache[layer_idx] = output
                    return output
                return patched_forward

            layer.forward = make_patched_forward(idx, self._original_forwards[idx])

    def step(self):
        """Advance the step counter (call once per denoising step)."""
        self._step_counter += 1

    def remove(self):
        """Restore original forward methods and clear caches."""
        for idx, orig_fwd in self._original_forwards.items():
            self.transformer.layers[idx].forward = orig_fwd
        self._original_forwards = {}
        self._cache.clear()
        self._step_counter = 0
        self._installed = False

    @property
    def cache_schedule_str(self) -> str:
        """Human-readable cache schedule."""
        intervals = {}
        for idx, interval in sorted(self._layer_intervals.items()):
            if interval not in intervals:
                intervals[interval] = []
            intervals[interval].append(idx)
        parts = []
        for interval, layers in sorted(intervals.items()):
            parts.append(f"interval={interval}: layers {layers[0]}-{layers[-1]}")
        return ", ".join(parts)


@torch.no_grad()
def generate_delta_dit(
    wrapper,
    prompt: str,
    seed: int,
    cache_schedule: Optional[Dict[Tuple[int, int], int]] = None,
    warmup_steps: int = 2,
    cfg_scale: float = 4.0,
    steps: int = 30,
    height: int = 1024,
    width: int = 1024,
) -> Image.Image:
    """
    Generate an image using Delta-DiT acceleration.

    Wraps the standard Lumina denoising loop with Delta-DiT layer caching.
    The first `warmup_steps` are always fully computed to populate caches.

    Args:
        wrapper: LuminaDiTWrapper instance.
        prompt: text prompt.
        seed: random seed.
        cache_schedule: per-layer-range cache intervals.
        warmup_steps: number of initial dense steps.
        cfg_scale: classifier-free guidance scale.
        steps: number of denoising steps.
        height, width: output resolution.

    Returns:
        PIL Image.
    """
    import math
    from PIL import Image
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina

    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    mgr = DeltaDiTManager(wrapper.transformer, cache_schedule)

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

    # ---- 5. Install Delta-DiT hooks ----
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

        # Rotary embeddings
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

        # During warmup, ensure all layers compute (no caching)
        if i < warmup_steps:
            mgr._step_counter = 0  # Force cache miss on all layers

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_mask=prompt_attention_mask,
            image_rotary_emb=image_rotary_emb,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # Advance step counter (after warmup)
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
# SD3-Medium: Delta-DiT on MMDiT joint attention blocks
# ─────────────────────────────────────────────────────────────────────────────

class DeltaDiTManagerSD3:
    """
    Delta-DiT for SD3-Medium MMDiT joint attention blocks.

    Same tier-based caching logic as DeltaDiTManager, but targets
    transformer.transformer_blocks (24 JointTransformerBlock layers).
    Block forward outputs are (encoder_hidden_states, hidden_states) tuples;
    the entire tuple is cached and returned on skip steps.
    """

    def __init__(
        self,
        transformer,
        cache_schedule: Optional[Dict[Tuple[int, int], int]] = None,
    ):
        self.transformer = transformer
        self.num_layers = len(transformer.transformer_blocks)

        if cache_schedule is None:
            third = self.num_layers // 3
            cache_schedule = {
                (0, third): 3,
                (third, 2 * third): 2,
                (2 * third, self.num_layers): 1,
            }

        self._layer_intervals = {}
        for (start, end), interval in cache_schedule.items():
            for idx in range(start, min(end, self.num_layers)):
                self._layer_intervals[idx] = interval
        for idx in range(self.num_layers):
            if idx not in self._layer_intervals:
                self._layer_intervals[idx] = 1

        self._cache = {}
        self._original_forwards = {}
        self._step_counter = 0
        self._installed = False

    def install_hooks(self):
        self._installed = True
        self._cache.clear()
        self._step_counter = 0
        self._original_forwards = {}

        for idx in range(self.num_layers):
            block = self.transformer.transformer_blocks[idx]
            self._original_forwards[idx] = block.forward

            def make_patched_forward(layer_idx, orig_fwd):
                def patched_forward(*args, **kwargs):
                    interval = self._layer_intervals[layer_idx]
                    if (layer_idx in self._cache
                            and self._step_counter % interval != 0):
                        return self._cache[layer_idx]
                    output = orig_fwd(*args, **kwargs)
                    self._cache[layer_idx] = output
                    return output
                return patched_forward

            block.forward = make_patched_forward(idx, self._original_forwards[idx])

    def step(self):
        self._step_counter += 1

    def remove(self):
        for idx, orig_fwd in self._original_forwards.items():
            self.transformer.transformer_blocks[idx].forward = orig_fwd
        self._original_forwards = {}
        self._cache.clear()
        self._step_counter = 0
        self._installed = False


@torch.no_grad()
def generate_delta_dit_sd3(
    wrapper,
    prompt: str,
    seed: int,
    cache_schedule: Optional[Dict[Tuple[int, int], int]] = None,
    warmup_steps: int = 2,
    cfg_scale: float = 7.0,
    steps: int = 28,
    height: int = 1024,
    width: int = 1024,
) -> "Image.Image":
    """
    Delta-DiT acceleration for SD3-Medium.

    Layer-level caching applied to SD3's 24 joint transformer blocks.
    First `warmup_steps` always computed fully. Uniform CFG throughout.
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    mgr = DeltaDiTManagerSD3(wrapper.transformer, cache_schedule)

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
    # SD3 batch order: [uncond, cond]
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

    # 4. Install Delta-DiT hooks
    mgr.install_hooks()

    # 5. Denoising loop
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # During warmup, force all caches to miss (step_counter=0 hits nothing)
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
# FLUX.1-dev: Delta-DiT on double-stream + single-stream transformer blocks
# ─────────────────────────────────────────────────────────────────────────────

class DeltaDiTManagerFLUX:
    """
    Delta-DiT for FLUX.1-dev.

    FLUX has two block types:
      - transformer.transformer_blocks (19 double-stream FluxTransformerBlock)
        returns (encoder_hidden_states, hidden_states)
      - transformer.single_transformer_blocks (38 single-stream blocks)
        returns hidden_states (or a tuple)

    Each group gets its own 3-tier cache schedule.
    """

    def __init__(
        self,
        transformer,
        double_schedule: Optional[Dict[Tuple[int, int], int]] = None,
        single_schedule: Optional[Dict[Tuple[int, int], int]] = None,
    ):
        self.transformer = transformer
        self.n_double = len(transformer.transformer_blocks)
        self.n_single = len(transformer.single_transformer_blocks)

        def _make_schedule(n, schedule):
            if schedule is None:
                third = n // 3
                schedule = {
                    (0, third): 3,
                    (third, 2 * third): 2,
                    (2 * third, n): 1,
                }
            intervals = {}
            for (start, end), interval in schedule.items():
                for idx in range(start, min(end, n)):
                    intervals[idx] = interval
            for idx in range(n):
                if idx not in intervals:
                    intervals[idx] = 1
            return intervals

        self._double_intervals = _make_schedule(self.n_double, double_schedule)
        self._single_intervals = _make_schedule(self.n_single, single_schedule)

        self._cache = {}          # keyed by ("d", idx) or ("s", idx)
        self._original_forwards = {}
        self._step_counter = 0
        self._installed = False

    def install_hooks(self):
        self._installed = True
        self._cache.clear()
        self._step_counter = 0
        self._original_forwards = {}

        for idx in range(self.n_double):
            block = self.transformer.transformer_blocks[idx]
            self._original_forwards[("d", idx)] = block.forward
            key = ("d", idx)

            def make_patched(cache_key, intervals_key, orig_fwd):
                def patched_forward(*args, **kwargs):
                    interval = self._double_intervals[intervals_key]
                    if (cache_key in self._cache
                            and self._step_counter % interval != 0):
                        return self._cache[cache_key]
                    output = orig_fwd(*args, **kwargs)
                    self._cache[cache_key] = output
                    return output
                return patched_forward

            block.forward = make_patched(key, idx, self._original_forwards[key])

        for idx in range(self.n_single):
            block = self.transformer.single_transformer_blocks[idx]
            self._original_forwards[("s", idx)] = block.forward
            key = ("s", idx)

            def make_patched_single(cache_key, intervals_key, orig_fwd):
                def patched_forward(*args, **kwargs):
                    interval = self._single_intervals[intervals_key]
                    if (cache_key in self._cache
                            and self._step_counter % interval != 0):
                        return self._cache[cache_key]
                    output = orig_fwd(*args, **kwargs)
                    self._cache[cache_key] = output
                    return output
                return patched_forward

            block.forward = make_patched_single(key, idx, self._original_forwards[key])

    def step(self):
        self._step_counter += 1

    def remove(self):
        for idx in range(self.n_double):
            key = ("d", idx)
            if key in self._original_forwards:
                self.transformer.transformer_blocks[idx].forward = self._original_forwards[key]
        for idx in range(self.n_single):
            key = ("s", idx)
            if key in self._original_forwards:
                self.transformer.single_transformer_blocks[idx].forward = self._original_forwards[key]
        self._original_forwards = {}
        self._cache.clear()
        self._step_counter = 0
        self._installed = False


@torch.no_grad()
def generate_delta_dit_flux(
    wrapper,
    prompt: str,
    seed: int,
    warmup_steps: int = 2,
    guidance_scale: float = 3.5,
    steps: int = 28,
    height: int = 1024,
    width: int = 1024,
) -> "Image.Image":
    """
    Delta-DiT acceleration for FLUX.1-dev.

    Layer-level caching on all 57 transformer blocks (19 double + 38 single).
    3-tier schedule per group. First `warmup_steps` always fully computed.
    """
    mgr = DeltaDiTManagerFLUX(wrapper.transformer)

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
