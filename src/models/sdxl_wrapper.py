"""
SDXL U-Net Wrapper for SASD.

Wraps the diffusers StableDiffusionXLPipeline to expose:
  - Standard generation (baseline)
  - SASD accelerated generation: spatial CFG + sparse attn + sparse FFN + step skip

Proves SASD's generality beyond DiT architectures.
"""

import math
import gc
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler


class SDXLWrapper:
    """Wrapper around SDXL for sparse compute experiments."""

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        dtype: str = "bf16",
        device: str = "cuda",
        cache_dir: str = None,
    ):
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.device = device
        self.dtype = torch_dtype

        print(f"Loading SDXL: {model_name} ({dtype})...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            use_safetensors=True,
        ).to(device)

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = self.pipe.vae_scale_factor  # 8

        print(f"SDXL loaded. VAE scale factor: {self.vae_scale_factor}")

    def _prepare_denoise(self, prompt, seed, cfg_scale, steps, height, width):
        """Shared initialization for both generate() and generate_accelerated().
        Returns all tensors needed for the denoising loop, guaranteeing
        identical latent noise for the same seed."""
        pipe = self.pipe
        device = self.device

        # Encode prompt (SDXL dual text encoders)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )

        # Timesteps
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # Latents — deterministic from seed
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        num_channels = self.unet.config.in_channels
        shape = (1, num_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)
        latents = latents * pipe.scheduler.init_noise_sigma

        # SDXL added conditions
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
        add_time_ids = pipe._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=self.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(device)

        # CFG: concat uncond + cond
        prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {
            "text_embeds": add_text_embeds_cfg,
            "time_ids": add_time_ids_cfg,
        }

        return latents, timesteps, prompt_embeds_cfg, added_cond_kwargs, latent_h, latent_w

    def _decode_latents(self, latents):
        """Decode latents to PIL image, handling VAE fp16 upcast."""
        pipe = self.pipe
        latents = latents / self.vae.config.scaling_factor
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            pipe.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: int,
        cfg_scale: float = 7.5,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """Standard SDXL baseline generation (manual loop, seed-consistent)."""
        pipe = self.pipe
        latents, timesteps, prompt_embeds_cfg, added_cond_kwargs, _, _ = \
            self._prepare_denoise(prompt, seed, cfg_scale, steps, height, width)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds_cfg,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return self._decode_latents(latents)

    @torch.no_grad()
    def generate_accelerated(
        self,
        prompt: str,
        seed: int,
        skip_ratio: float = 0.5,
        mask_type: str = "cfg_magnitude",
        s_fg: float = 9.0,
        s_bg: float = 2.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        cfg_scale: float = 7.5,
        sparse_ffn: bool = True,
        sparse_blocks: bool = True,
        sparse_start_step: int = -1,
        full_skip_interval: int = 0,
        blur_sigma: float = 1.5,
        save_mask_callback=None,
    ) -> Image.Image:
        """
        SASD accelerated generation for SDXL.

        Applies the same four-layer acceleration as DiT version:
          1. Spatial CFG: per-pixel guidance scale (fg high, bg low)
          2. Sparse self-attention: fg-only SDPA in attn1
          3. Sparse FFN: fg-only MLP computation
          4. Step-level caching: skip entire UNet forward passes

        Args:
            skip_ratio: fraction of tokens treated as background.
            mask_type: "cfg_magnitude" (recommended), "complexity", "uniform".
            s_fg: CFG scale for foreground.
            s_bg: CFG scale for background.
            mask_step: denoising step at which to build the mask.
            cfg_scale: base CFG scale (used before mask is built).
            sparse_ffn: if True, also sparsify FFN.
            sparse_blocks: if True, install sparse processors.
            sparse_start_step: step to switch from dense to sparse (-1 = mask_step+1).
            full_skip_interval: skip entire UNet every N steps (0=disabled).
            blur_sigma: Gaussian blur for mask edges.
            save_mask_callback: optional fn(mask_tensor).
        """
        from src.sparse.sdxl_block_manager import SDXLSparseBlockManager
        from src.sparse.sdit_mask_builder import CFGMagnitudeMaskBuilder
        from src.sparse.mask_builders import ComplexityMaskBuilder
        from src.sparse.skip_cache import SkipUpdateCache

        pipe = self.pipe
        device = self.device

        # ---- 1-4. Shared initialization (same latent noise as generate()) ----
        latents, timesteps, prompt_embeds_cfg, added_cond_kwargs, latent_h, latent_w = \
            self._prepare_denoise(prompt, seed, cfg_scale, steps, height, width)

        # ---- 5. Prepare mask builder ----
        use_cfg_magnitude = mask_type == "cfg_magnitude"
        use_complexity = mask_type == "complexity"
        use_uniform = mask_type == "uniform"

        mask_builder = None
        if use_cfg_magnitude:
            mask_builder = CFGMagnitudeMaskBuilder(
                n_segments=64, compactness=10.0,
                ratio=1.0 - skip_ratio, blur_sigma=blur_sigma,
            )
        elif use_complexity:
            mask_builder = ComplexityMaskBuilder(
                grid_size=8, ratio=1.0 - skip_ratio,
            )

        # ---- 6. Initialize caches ----
        full_skip_cache = SkipUpdateCache(method="linear") if full_skip_interval > 0 else None

        # ---- 7. Denoising loop ----
        block_mgr = None
        sparse_activated = False
        s_map = None  # (1, 1, latent_h, latent_w) per-pixel CFG scale
        actual_sparse_start = sparse_start_step if sparse_start_step >= 0 else mask_step + 1
        fg_mask_spatial = None

        for i, t in enumerate(timesteps):
            # --- Full skip: skip entire UNet, extrapolate noise_pred ---
            is_full_skip = False
            if (full_skip_interval > 0
                    and full_skip_cache is not None
                    and full_skip_cache.has_cache()
                    and sparse_activated):
                steps_since_sparse = i - actual_sparse_start
                if steps_since_sparse >= 2 and steps_since_sparse % full_skip_interval == 0:
                    is_full_skip = True

            if is_full_skip:
                noise_pred = full_skip_cache.get_prediction(i)
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                continue

            # Set block manager mode
            if block_mgr is not None and sparse_activated:
                block_mgr.set_dense_mode(False)

            # Scale latents for scheduler
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # Forward through UNet
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # ---- CFG: split cond / uncond ----
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

            # ---- Build mask at mask_step ----
            if i == mask_step and block_mgr is None:
                if use_cfg_magnitude:
                    spatial_mask = mask_builder.build_mask(
                        noise_pred_cond, noise_pred_uncond,
                    )
                elif use_uniform:
                    # Uniform random mask
                    spatial_mask = torch.ones(latent_h, latent_w,
                                              device=device, dtype=self.dtype)
                    n_total = latent_h * latent_w
                    n_bg = int(n_total * skip_ratio)
                    perm = torch.randperm(n_total, device=device)[:n_bg]
                    spatial_mask.view(-1)[perm] = 0.0
                elif use_complexity and mask_builder is not None:
                    spatial_mask = mask_builder.build_mask(latents)
                else:
                    # Fallback: center mask
                    spatial_mask = torch.zeros(latent_h, latent_w,
                                               device=device, dtype=self.dtype)
                    margin_h = int(latent_h * skip_ratio / 2)
                    margin_w = int(latent_w * skip_ratio / 2)
                    spatial_mask[margin_h:latent_h-margin_h, margin_w:latent_w-margin_w] = 1.0

                spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)

                if save_mask_callback:
                    save_mask_callback(spatial_mask)

                # fg_mask_spatial for spatial CFG: (1, 1, H, W)
                fg_mask_spatial = spatial_mask.unsqueeze(0).unsqueeze(0)

                # Build per-pixel CFG scale map
                s_map = s_bg + fg_mask_spatial * (s_fg - s_bg)

                # Install sparse block manager
                if sparse_blocks:
                    block_mgr = SDXLSparseBlockManager(self.unet)
                    block_mgr.install(
                        fg_mask_latent=spatial_mask,
                        dense_mode=True,
                        sparse_ffn=sparse_ffn,
                    )

            # Activate sparse mode
            if block_mgr is not None and not sparse_activated and i >= actual_sparse_start:
                sparse_activated = True

            # ---- Apply CFG: spatial or uniform ----
            if s_map is not None:
                noise_pred = noise_pred_uncond + s_map * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            # Store in full-skip cache
            if full_skip_cache is not None and sparse_activated:
                full_skip_cache.store(i, noise_pred)

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 8. Cleanup ----
        if block_mgr is not None:
            block_mgr.remove()

        # ---- 9. Decode ----
        return self._decode_latents(latents)
