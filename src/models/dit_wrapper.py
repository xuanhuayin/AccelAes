"""
Lumina-Next-T2I DiT Wrapper.

Wraps the diffusers LuminaText2ImgPipeline to expose:
  - Standard generation (baseline)
  - Spatial CFG: per-patch guidance scale (Phase 1)
  - Access to internal transformer for future hooks
"""

import math
import torch
from PIL import Image
try:
    from diffusers import LuminaPipeline as LuminaText2ImgPipeline
except ImportError:
    from diffusers import LuminaText2ImgPipeline
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina


class LuminaDiTWrapper:
    """Wrapper around Lumina-Next-T2I for sparse compute experiments."""

    def __init__(
        self,
        model_name: str = "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        dtype: str = "bf16",
        device: str = "cuda",
        cache_dir: str = None,
    ):
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.device = device
        self.dtype = torch_dtype

        print(f"Loading {model_name} ({dtype})...")
        self.pipe = LuminaText2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        ).to(device)

        # Expose internal components
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.image_processor = self.pipe.image_processor

        # Constants
        self.vae_scale_factor = self.pipe.vae_scale_factor  # 8
        self.default_image_size = getattr(self.pipe, 'default_image_size', 1024)

        # Semantic mask builder cache (lazy-initialized, reused across generate calls)
        # Avoids reloading CLIP (~340MB) and recomputing anchor embeddings per image
        self._semantic_builder = None
        self._token_importance_cache: dict = {}  # (prompt, seq_len) -> importance tensor

        print(f"Model loaded. VAE scale factor: {self.vae_scale_factor}")

    def generate(
        self,
        prompt: str,
        seed: int,
        cfg_scale: float = 4.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """Standard baseline generation."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="pil",
        )
        return result.images[0]

    @torch.no_grad()
    def generate_spatial_cfg(
        self,
        prompt: str,
        seed: int,
        mask: torch.Tensor,
        s_fg: float = 7.0,
        s_bg: float = 1.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
    ) -> Image.Image:
        """
        Spatial CFG generation (Phase 1).

        Replicates the exact Lumina denoising loop but replaces the uniform
        guidance_scale with a per-pixel scale map:
          s_i = s_bg + mask_i * (s_fg - s_bg)

        Only the first 3 channels of noise_pred are guided (Lumina convention).

        Args:
            mask: (H_latent, W_latent) float tensor in [0,1].
                  1 = foreground, 0 = background.
            s_fg: guidance scale for foreground regions.
            s_bg: guidance scale for background regions.
        """
        pipe = self.pipe
        device = self.device

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
        # Concat [cond, uncond] (Lumina order)
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Build per-pixel guidance scale map ----
        mask = mask.to(device=device, dtype=self.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        s_map = s_bg + mask * (s_fg - s_bg)  # (1, 1, H_lat, W_lat)

        # ---- 6. Denoising loop (matches pipeline_lumina.py exactly) ----
        head_dim = self.transformer.head_dim

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
            current_timestep = t
            if not torch.is_tensor(current_timestep):
                current_timestep = torch.tensor([current_timestep], dtype=torch.float64, device=device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(device)
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            # Reverse timestep (flow-matching: t=0 is noise, t=1 is image)
            current_timestep = 1 - current_timestep / pipe.scheduler.config.num_train_timesteps

            # Time-aware rotary embeddings
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

            # Forward through transformer
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Split pred / sigma (learn_sigma model outputs 8 channels)
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # --- Spatial CFG (replaces uniform guidance_scale) ---
            # noise_pred shape: (2, C, H, W) where batch dim = [cond, uncond]
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]

            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )
            # Per-pixel guidance: s_map is (1, 1, H_lat, W_lat)
            noise_pred_half = noise_pred_uncond_eps + s_map * (
                noise_pred_cond_eps - noise_pred_uncond_eps
            )
            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)

            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)

            # Negate (Lumina convention)
            noise_pred = -noise_pred

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 7. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def generate_with_auto_mask(
        self,
        prompt: str,
        seed: int,
        mask_builder,
        s_fg: float = 7.0,
        s_bg: float = 1.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
        save_mask_callback=None,
    ) -> Image.Image:
        """
        Spatial CFG with auto-generated mask (Phase 2).

        For ComplexityMaskBuilder: mask is built from the latent at `mask_step`.
        For SemanticMaskBuilder: mask is built from cross-attn maps at `mask_step`.

        The mask is computed once at `mask_step` and reused for the rest.

        Args:
            mask_builder: a ComplexityMaskBuilder or SemanticMaskBuilder instance
            mask_step: which denoising step to compute the mask at
            save_mask_callback: optional fn(mask, heatmap) called when mask is built
        """
        from src.sparse.mask_builders import ComplexityMaskBuilder, SemanticMaskBuilder
        from src.sparse.cross_attn_hook import CrossAttnHookManager

        pipe = self.pipe
        device = self.device

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

        # Keep separate copies for semantic mask
        cond_embeds = prompt_embeds.clone()
        cond_mask = prompt_attention_mask.clone()

        # Concat [cond, uncond]
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Prepare semantic mask builder if needed ----
        use_semantic = isinstance(mask_builder, SemanticMaskBuilder)
        hook_mgr = None
        if use_semantic:
            mask_builder.compute_anchor_embeddings(device=device)
            # Tokenize prompt with model tokenizer to get input_ids for
            # word grouping. Pad to same length as cond_embeds.
            seq_len = cond_embeds.shape[1]
            cond_tokens = self.tokenizer(
                prompt, padding="max_length", max_length=seq_len,
                truncation=True, return_tensors="pt",
            ).to(device)
            token_importance = mask_builder.compute_token_importance(
                prompt=prompt,
                prompt_mask=cond_mask,
                input_ids=cond_tokens["input_ids"],
                tokenizer=self.tokenizer,
            )

        # ---- 6. Denoising loop ----
        head_dim = self.transformer.head_dim
        patch_size = self.transformer.config.patch_size  # typically 2
        patch_h = latent_h // patch_size  # patch-level spatial dims
        patch_w = latent_w // patch_size
        s_map = None  # will be set at mask_step

        for i, t in enumerate(timesteps):
            # Install cross-attn hooks just before the mask_step
            if use_semantic and i == mask_step:
                hook_mgr = CrossAttnHookManager(self.transformer)
                hook_mgr.install()

            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
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

            # Forward
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Remove hooks after capture step
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # ---- Build mask at mask_step ----
            if i == mask_step and s_map is None:
                if use_semantic:
                    cross_attn_maps = hook_mgr.get_maps()
                    mask = mask_builder.build_mask_from_cross_attention(
                        cross_attn_maps, token_importance,
                        patch_h, patch_w,
                    )
                else:
                    # ComplexityMaskBuilder: use current latent
                    mask = mask_builder.build_mask(latents)

                if mask is None:
                    # Mask builder decided the heatmap lacks spatial structure.
                    # Fall back to uniform CFG for the rest of the steps.
                    s_map = torch.full(
                        (1, 1, latent_h, latent_w), s_fg,
                        device=device, dtype=self.dtype,
                    )
                    if save_mask_callback:
                        # Save an all-zeros mask to indicate fallback
                        fallback_mask = torch.zeros(patch_h, patch_w, device=device, dtype=self.dtype)
                        save_mask_callback(fallback_mask, None)
                else:
                    mask = mask.to(device=device, dtype=self.dtype)
                    if save_mask_callback:
                        if use_semantic:
                            save_mask_callback(mask, None)
                        else:
                            _, complexity_map = mask_builder.build_mask_with_viz(latents)
                            save_mask_callback(mask, complexity_map)

                    # Upscale mask from patch-level to latent-level if needed
                    if mask.shape[0] != latent_h or mask.shape[1] != latent_w:
                        mask = torch.nn.functional.interpolate(
                            mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(latent_h, latent_w),
                            mode="nearest",
                        ).squeeze(0).squeeze(0).to(dtype=self.dtype)

                    mask_4d = mask.unsqueeze(0).unsqueeze(0)
                    s_map = s_bg + mask_4d * (s_fg - s_bg)

            # Split pred / sigma
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # Spatial CFG (or uniform if mask not yet built)
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )

            if s_map is not None:
                noise_pred_half = noise_pred_uncond_eps + s_map * (
                    noise_pred_cond_eps - noise_pred_uncond_eps
                )
            else:
                # Before mask_step: use uniform s_fg
                noise_pred_half = noise_pred_uncond_eps + s_fg * (
                    noise_pred_cond_eps - noise_pred_uncond_eps
                )

            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)
            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
            noise_pred = -noise_pred

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 7. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def generate_tome(
        self,
        prompt: str,
        seed: int,
        merge_ratio: float = 0.5,
        mask_type: str = "complexity",
        cfg_scale: float = 4.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        mask_ratio: float = 0.25,
        sparsity_schedule: str = "constant",
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
        grid_size: int = 8,
        blur_sigma: float = 1.0,
        save_mask_callback=None,
    ) -> Image.Image:
        """
        Generation with token merging for inference acceleration (Phase 3).

        Runs full computation until mask_step, builds a foreground mask,
        installs ToMe processors on self-attention layers, then continues
        denoising with merged background tokens. Uses standard uniform CFG.

        Args:
            merge_ratio: fraction of background tokens to merge (0-1).
            mask_type: "complexity", "semantic", or "uniform".
                "uniform" = no foreground protection (all tokens mergeable).
            cfg_scale: uniform classifier-free guidance scale.
            mask_step: denoising step at which to compute mask and start merging.
            mask_ratio: fraction of patches marked as foreground (for mask builder).
            sparsity_schedule: "constant" or "cosine" merge ratio schedule.
            grid_size: grid size for complexity mask.
            blur_sigma: Gaussian blur sigma for semantic mask.
            save_mask_callback: optional fn(mask, heatmap) called when mask is built.
        """
        from src.sparse.mask_builders import ComplexityMaskBuilder, SemanticMaskBuilder
        from src.sparse.cross_attn_hook import CrossAttnHookManager
        from src.sparse.tome_hook import ToMeHookManager
        from src.sparse.token_merge import compute_merge_ratio_schedule

        pipe = self.pipe
        device = self.device

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

        cond_embeds = prompt_embeds.clone()
        cond_mask = prompt_attention_mask.clone()

        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Prepare mask builder ----
        head_dim = self.transformer.head_dim
        patch_size = self.transformer.config.patch_size
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        use_semantic = mask_type == "semantic"
        use_complexity = mask_type == "complexity"
        use_uniform = mask_type == "uniform"

        mask_builder = None
        hook_mgr = None
        token_importance = None

        if use_complexity:
            mask_builder = ComplexityMaskBuilder(grid_size=grid_size, ratio=mask_ratio)
        elif use_semantic:
            mask_builder = SemanticMaskBuilder(ratio=mask_ratio, blur_sigma=blur_sigma)
            mask_builder.compute_anchor_embeddings(device=device)
            seq_len = cond_embeds.shape[1]
            cond_tokens = self.tokenizer(
                prompt, padding="max_length", max_length=seq_len,
                truncation=True, return_tensors="pt",
            ).to(device)
            token_importance = mask_builder.compute_token_importance(
                prompt=prompt,
                prompt_mask=cond_mask,
                input_ids=cond_tokens["input_ids"],
                tokenizer=self.tokenizer,
            )

        # Compute per-step merge ratio schedule
        ratio_schedule = compute_merge_ratio_schedule(
            total_steps=steps,
            mask_step=mask_step,
            base_ratio=merge_ratio,
            schedule=sparsity_schedule,
        )

        # ---- 6. Denoising loop ----
        tome_mgr = None
        fg_token_mask = None  # (patch_h * patch_w,) bool, True = foreground

        for i, t in enumerate(timesteps):
            # Install cross-attn hooks for semantic mask capture
            if use_semantic and i == mask_step:
                hook_mgr = CrossAttnHookManager(self.transformer)
                hook_mgr.install()

            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
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

            # Update ToMe ratio for this step (if installed)
            if tome_mgr is not None:
                tome_mgr.update_ratio(ratio_schedule[i])

            # Forward
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Remove cross-attn hooks after capture
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # ---- Build mask and install ToMe at mask_step ----
            if i == mask_step and tome_mgr is None:
                if use_uniform:
                    # No foreground protection — all tokens can be merged
                    fg_token_mask = None
                    spatial_mask = None
                elif use_semantic:
                    cross_attn_maps = hook_mgr.get_maps()
                    spatial_mask = mask_builder.build_mask_from_cross_attention(
                        cross_attn_maps, token_importance,
                        patch_h, patch_w,
                    )
                else:
                    # Complexity mask: built from latent, returns (latent_h, latent_w)
                    spatial_mask = mask_builder.build_mask(latents)

                # Convert spatial mask to per-token foreground indicator
                if use_uniform:
                    fg_token_mask = None
                elif spatial_mask is None:
                    # Mask builder fallback: no spatial structure → uniform merging
                    fg_token_mask = None
                    spatial_mask = None
                else:
                    spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)

                    if save_mask_callback:
                        if use_semantic:
                            save_mask_callback(spatial_mask, None)
                        else:
                            _, cmap = mask_builder.build_mask_with_viz(latents)
                            save_mask_callback(spatial_mask, cmap)

                    # Resize mask to patch-level if needed
                    if spatial_mask.shape[0] != patch_h or spatial_mask.shape[1] != patch_w:
                        spatial_mask = torch.nn.functional.interpolate(
                            spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(patch_h, patch_w),
                            mode="nearest",
                        ).squeeze(0).squeeze(0).to(dtype=self.dtype)

                    # Threshold to binary: >0.5 = foreground
                    fg_token_mask = (spatial_mask.reshape(-1) > 0.5).to(device=device)

                # Install ToMe processors
                tome_mgr = ToMeHookManager(
                    self.transformer,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
                tome_mgr.install(fg_mask=fg_token_mask, merge_ratio=ratio_schedule[i])

            # Split pred / sigma
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # Standard uniform CFG
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )

            noise_pred_half = noise_pred_uncond_eps + cfg_scale * (
                noise_pred_cond_eps - noise_pred_uncond_eps
            )

            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)
            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
            noise_pred = -noise_pred

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 7. Cleanup ----
        if tome_mgr is not None:
            tome_mgr.remove()

        # ---- 8. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def generate_skip_update(
        self,
        prompt: str,
        seed: int,
        skip_ratio: float = 0.25,
        mask_type: str = "complexity",
        extrapolation: str = "copy",
        refresh_interval: int = 0,
        cfg_scale: float = 4.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        grid_size: int = 8,
        blur_sigma: float = 1.0,
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
        save_mask_callback=None,
    ) -> Image.Image:
        """
        Skip-update generation (Phase 3).

        Runs the FULL forward pass every step, but after mask_step, replaces
        background regions of noise_pred with cached/extrapolated values
        before feeding to the scheduler. This validates that skipping
        background updates doesn't destroy quality.

        Args:
            skip_ratio: fraction of patches treated as background (skipped).
            mask_type: "complexity", "semantic", or "uniform".
            extrapolation: "copy" (reuse last) or "linear" (extrapolate from last 2).
            refresh_interval: full refresh every N steps after mask_step (0=never).
            cfg_scale: uniform classifier-free guidance scale.
            mask_step: denoising step at which to compute mask and start skipping.
            save_mask_callback: optional fn(mask, heatmap) called when mask is built.
        """
        from src.sparse.mask_builders import ComplexityMaskBuilder, SemanticMaskBuilder
        from src.sparse.cross_attn_hook import CrossAttnHookManager
        from src.sparse.skip_cache import SkipUpdateCache

        pipe = self.pipe
        device = self.device

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

        cond_embeds = prompt_embeds.clone()
        cond_mask = prompt_attention_mask.clone()

        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Prepare mask builder ----
        head_dim = self.transformer.head_dim
        patch_size = self.transformer.config.patch_size
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        use_semantic = mask_type == "semantic"
        use_complexity = mask_type == "complexity"
        use_uniform = mask_type == "uniform"

        mask_builder = None
        hook_mgr = None
        token_importance = None

        if use_complexity:
            # fg fraction = 1 - skip_ratio (skip_ratio is bg fraction)
            mask_builder = ComplexityMaskBuilder(grid_size=grid_size, ratio=1.0 - skip_ratio)
        elif use_semantic:
            mask_builder = SemanticMaskBuilder(ratio=1.0 - skip_ratio, blur_sigma=blur_sigma)
            mask_builder.compute_anchor_embeddings(device=device)
            seq_len = cond_embeds.shape[1]
            cond_tokens = self.tokenizer(
                prompt, padding="max_length", max_length=seq_len,
                truncation=True, return_tensors="pt",
            ).to(device)
            token_importance = mask_builder.compute_token_importance(
                prompt=prompt,
                prompt_mask=cond_mask,
                input_ids=cond_tokens["input_ids"],
                tokenizer=self.tokenizer,
            )

        # ---- 6. Initialize skip cache ----
        cache = SkipUpdateCache(method=extrapolation)
        fg_mask_spatial = None  # (1, 1, latent_h, latent_w), 1=foreground

        # ---- 7. Denoising loop ----
        for i, t in enumerate(timesteps):
            # Install cross-attn hooks for semantic mask capture
            if use_semantic and i == mask_step:
                hook_mgr = CrossAttnHookManager(self.transformer)
                hook_mgr.install()

            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
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

            # Forward (always full)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Remove cross-attn hooks after capture
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # ---- Build mask at mask_step ----
            if i == mask_step and fg_mask_spatial is None:
                if use_uniform:
                    # Uniform: skip_ratio fraction is background everywhere
                    fg_mask_spatial = torch.full(
                        (1, 1, latent_h, latent_w), 1.0 - skip_ratio,
                        device=device, dtype=self.dtype,
                    )
                elif use_semantic:
                    cross_attn_maps = hook_mgr.get_maps()
                    spatial_mask = mask_builder.build_mask_from_cross_attention(
                        cross_attn_maps, token_importance,
                        patch_h, patch_w,
                    )
                    if spatial_mask is None:
                        # No spatial structure — no skip-update
                        fg_mask_spatial = torch.ones(
                            1, 1, latent_h, latent_w,
                            device=device, dtype=self.dtype,
                        )
                        if save_mask_callback:
                            fallback_mask = torch.zeros(patch_h, patch_w, device=device, dtype=self.dtype)
                            save_mask_callback(fallback_mask, None)
                    else:
                        spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)
                        if save_mask_callback:
                            save_mask_callback(spatial_mask, None)
                        # Upscale to latent resolution
                        if spatial_mask.shape[0] != latent_h or spatial_mask.shape[1] != latent_w:
                            spatial_mask = torch.nn.functional.interpolate(
                                spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(latent_h, latent_w),
                                mode="nearest",
                            ).squeeze(0).squeeze(0).to(dtype=self.dtype)
                        fg_mask_spatial = spatial_mask.unsqueeze(0).unsqueeze(0)
                else:
                    # Complexity mask
                    spatial_mask = mask_builder.build_mask(latents)
                    spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)
                    if save_mask_callback:
                        _, cmap = mask_builder.build_mask_with_viz(latents)
                        save_mask_callback(spatial_mask, cmap)
                    # Upscale if needed
                    if spatial_mask.shape[0] != latent_h or spatial_mask.shape[1] != latent_w:
                        spatial_mask = torch.nn.functional.interpolate(
                            spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(latent_h, latent_w),
                            mode="nearest",
                        ).squeeze(0).squeeze(0).to(dtype=self.dtype)
                    fg_mask_spatial = spatial_mask.unsqueeze(0).unsqueeze(0)

            # Split pred / sigma
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # Standard uniform CFG
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )
            noise_pred_half = noise_pred_uncond_eps + cfg_scale * (
                noise_pred_cond_eps - noise_pred_uncond_eps
            )
            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)
            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
            noise_pred = -noise_pred

            # ---- Skip-update: blend with cached prediction for background ----
            noise_pred_full = noise_pred.clone()  # save before blending

            if fg_mask_spatial is not None and cache.has_cache() and i > mask_step:
                is_refresh = (
                    refresh_interval > 0
                    and (i - mask_step) % refresh_interval == 0
                )
                if not is_refresh:
                    cached_pred = cache.get_prediction(i)
                    bg_mask = 1.0 - fg_mask_spatial
                    noise_pred = fg_mask_spatial * noise_pred + bg_mask * cached_pred

            cache.store(i, noise_pred_full)  # always cache the FULL (unblended) prediction

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 8. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def generate_skip_update_v2(
        self,
        prompt: str,
        seed: int,
        skip_ratio: float = 0.25,
        mask_type: str = "complexity",
        extrapolation: str = "copy",
        refresh_interval: int = 0,
        cfg_scale: float = 4.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        grid_size: int = 8,
        blur_sigma: float = 1.0,
        # Phase 4 new params
        region_mask: bool = False,
        n_segments: int = 64,
        slic_compactness: float = 10.0,
        dilation_radius: int = 0,
        mask_blur_sigma: float = 0.0,
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
        save_mask_callback=None,
    ) -> Image.Image:
        """
        Skip-update generation v2 (Phase 4).

        Extends generate_skip_update with:
          - SLIC superpixel region masks (--region_mask)
          - Boundary smoothing (dilation + blur) on any mask type
          - Velocity-space and x0-space extrapolation

        Args:
            skip_ratio: fraction of patches treated as background (skipped).
            mask_type: "complexity", "semantic", or "uniform".
            extrapolation: "copy", "linear", "velocity", or "x0".
            refresh_interval: full refresh every N steps after mask_step (0=never).
            region_mask: if True, use SLIC region mask instead of grid blocks.
            n_segments: SLIC target superpixel count.
            slic_compactness: SLIC compactness parameter.
            dilation_radius: boundary dilation radius (0=off).
            mask_blur_sigma: boundary Gaussian blur sigma (0=off).
            save_mask_callback: optional fn(mask, heatmap) called when mask is built.
        """
        from src.sparse.mask_builders import ComplexityMaskBuilder, SemanticMaskBuilder
        from src.sparse.cross_attn_hook import CrossAttnHookManager
        from src.sparse.extrapolators import make_extrapolator
        from src.sparse.boundary_ops import soft_mask_pipeline

        pipe = self.pipe
        device = self.device

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

        cond_embeds = prompt_embeds.clone()
        cond_mask = prompt_attention_mask.clone()

        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Prepare mask builder ----
        head_dim = self.transformer.head_dim
        patch_size = self.transformer.config.patch_size
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        use_semantic = mask_type == "semantic"
        use_complexity = mask_type == "complexity"
        use_uniform = mask_type == "uniform"

        mask_builder = None
        hook_mgr = None
        token_importance = None

        if use_complexity:
            if region_mask:
                from src.sparse.region_mask_builder import RegionMaskBuilder
                mask_builder = RegionMaskBuilder(
                    n_segments=n_segments,
                    compactness=slic_compactness,
                    ratio=1.0 - skip_ratio,
                    dilation_radius=dilation_radius,
                    blur_sigma=mask_blur_sigma,
                )
            else:
                mask_builder = ComplexityMaskBuilder(
                    grid_size=grid_size, ratio=1.0 - skip_ratio,
                )
        elif use_semantic:
            mask_builder = SemanticMaskBuilder(ratio=1.0 - skip_ratio, blur_sigma=blur_sigma)
            mask_builder.compute_anchor_embeddings(device=device)
            seq_len = cond_embeds.shape[1]
            cond_tokens = self.tokenizer(
                prompt, padding="max_length", max_length=seq_len,
                truncation=True, return_tensors="pt",
            ).to(device)
            token_importance = mask_builder.compute_token_importance(
                prompt=prompt,
                prompt_mask=cond_mask,
                input_ids=cond_tokens["input_ids"],
                tokenizer=self.tokenizer,
            )

        # ---- 6. Initialize extrapolator ----
        extrapolator = make_extrapolator(extrapolation)
        fg_mask_spatial = None  # (1, 1, latent_h, latent_w), 1=foreground

        # ---- 7. Denoising loop ----
        for i, t in enumerate(timesteps):
            # Install cross-attn hooks for semantic mask capture
            if use_semantic and i == mask_step:
                hook_mgr = CrossAttnHookManager(self.transformer)
                hook_mgr.install()

            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
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

            # Forward (always full)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Remove cross-attn hooks after capture
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # ---- Build mask at mask_step ----
            if i == mask_step and fg_mask_spatial is None:
                if use_uniform:
                    fg_mask_spatial = torch.full(
                        (1, 1, latent_h, latent_w), 1.0 - skip_ratio,
                        device=device, dtype=self.dtype,
                    )
                elif use_semantic:
                    cross_attn_maps = hook_mgr.get_maps()
                    spatial_mask = mask_builder.build_mask_from_cross_attention(
                        cross_attn_maps, token_importance,
                        patch_h, patch_w,
                    )
                    if spatial_mask is None:
                        fg_mask_spatial = torch.ones(
                            1, 1, latent_h, latent_w,
                            device=device, dtype=self.dtype,
                        )
                        if save_mask_callback:
                            fallback_mask = torch.zeros(patch_h, patch_w, device=device, dtype=self.dtype)
                            save_mask_callback(fallback_mask, None)
                    else:
                        spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)
                        # Apply boundary smoothing if requested
                        if dilation_radius > 0 or mask_blur_sigma > 0:
                            spatial_mask = soft_mask_pipeline(
                                spatial_mask, dilation_radius, mask_blur_sigma,
                            )
                        if save_mask_callback:
                            save_mask_callback(spatial_mask, None)
                        if spatial_mask.shape[0] != latent_h or spatial_mask.shape[1] != latent_w:
                            spatial_mask = torch.nn.functional.interpolate(
                                spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(latent_h, latent_w),
                                mode="nearest",
                            ).squeeze(0).squeeze(0).to(dtype=self.dtype)
                        fg_mask_spatial = spatial_mask.unsqueeze(0).unsqueeze(0)
                else:
                    # Complexity mask (grid or SLIC region)
                    spatial_mask = mask_builder.build_mask(latents)
                    spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)
                    # Apply boundary smoothing for grid masks (SLIC handles it internally)
                    if not region_mask and (dilation_radius > 0 or mask_blur_sigma > 0):
                        spatial_mask = soft_mask_pipeline(
                            spatial_mask, dilation_radius, mask_blur_sigma,
                        )
                    if save_mask_callback:
                        _, cmap = mask_builder.build_mask_with_viz(latents)
                        save_mask_callback(spatial_mask, cmap)
                    if spatial_mask.shape[0] != latent_h or spatial_mask.shape[1] != latent_w:
                        spatial_mask = torch.nn.functional.interpolate(
                            spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(latent_h, latent_w),
                            mode="nearest",
                        ).squeeze(0).squeeze(0).to(dtype=self.dtype)
                    fg_mask_spatial = spatial_mask.unsqueeze(0).unsqueeze(0)

            # Split pred / sigma
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # Standard uniform CFG
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )
            noise_pred_half = noise_pred_uncond_eps + cfg_scale * (
                noise_pred_cond_eps - noise_pred_uncond_eps
            )
            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)
            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
            noise_pred = -noise_pred

            # ---- Skip-update: blend with extrapolated prediction for background ----
            noise_pred_full = noise_pred.clone()

            # Read sigma for extrapolator (sigmas[i] corresponds to step i)
            sigma = pipe.scheduler.sigmas[i].item()

            if fg_mask_spatial is not None and extrapolator.can_extrapolate() and i > mask_step:
                is_refresh = (
                    refresh_interval > 0
                    and (i - mask_step) % refresh_interval == 0
                )
                if not is_refresh:
                    cached = extrapolator.extrapolate(i, sigma, latents)
                    bg_mask = 1.0 - fg_mask_spatial
                    noise_pred = fg_mask_spatial * noise_pred + bg_mask * cached

            extrapolator.store(i, noise_pred_full, sigma, latents)

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 8. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def generate_sparse_forward(
        self,
        prompt: str,
        seed: int,
        skip_ratio: float = 0.25,
        mask_type: str = "complexity",
        cfg_scale: float = 4.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        grid_size: int = 8,
        blur_sigma: float = 1.0,
        refresh_interval: int = 0,
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
        save_mask_callback=None,
    ) -> Image.Image:
        """
        Sparse forward generation (Phase 5).

        Steps 0 → mask_step: dense forward (standard).
        At mask_step: build foreground mask, install SparseAttnProcessor in
            dense_mode to warm up per-layer cache.
        Steps mask_step+1 → end: sparse forward — attn1 only computes
            foreground tokens, background tokens reuse cached attn output.

        Cross-attention (attn2) stays dense throughout (text tokens ~16, cheap).

        Args:
            skip_ratio: fraction of tokens treated as background (skipped in attn1).
            mask_type: "complexity", "semantic", or "uniform".
            cfg_scale: uniform classifier-free guidance scale.
            mask_step: denoising step at which to compute mask and start sparse.
            grid_size: grid size for complexity mask.
            blur_sigma: Gaussian blur sigma for semantic mask.
            refresh_interval: re-run dense every N steps to refresh cache (0=never).
            save_mask_callback: optional fn(mask, heatmap) called when mask is built.
        """
        from src.sparse.mask_builders import ComplexityMaskBuilder, SemanticMaskBuilder
        from src.sparse.cross_attn_hook import CrossAttnHookManager
        from src.sparse.sparse_hook import SparseHookManager

        pipe = self.pipe
        device = self.device

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

        cond_embeds = prompt_embeds.clone()
        cond_mask = prompt_attention_mask.clone()

        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Prepare mask builder ----
        head_dim = self.transformer.head_dim
        patch_size = self.transformer.config.patch_size
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        use_semantic = mask_type == "semantic"
        use_complexity = mask_type == "complexity"
        use_uniform = mask_type == "uniform"

        mask_builder = None
        hook_mgr = None
        token_importance = None

        if use_complexity:
            mask_builder = ComplexityMaskBuilder(grid_size=grid_size, ratio=1.0 - skip_ratio)
        elif use_semantic:
            mask_builder = SemanticMaskBuilder(ratio=1.0 - skip_ratio, blur_sigma=blur_sigma)
            mask_builder.compute_anchor_embeddings(device=device)
            seq_len = cond_embeds.shape[1]
            cond_tokens = self.tokenizer(
                prompt, padding="max_length", max_length=seq_len,
                truncation=True, return_tensors="pt",
            ).to(device)
            token_importance = mask_builder.compute_token_importance(
                prompt=prompt,
                prompt_mask=cond_mask,
                input_ids=cond_tokens["input_ids"],
                tokenizer=self.tokenizer,
            )

        # ---- 6. Denoising loop ----
        sparse_mgr = None

        for i, t in enumerate(timesteps):
            # Install cross-attn hooks for semantic mask capture
            if use_semantic and i == mask_step:
                hook_mgr = CrossAttnHookManager(self.transformer)
                hook_mgr.install()

            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
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

            # Periodic dense refresh: switch to dense mode temporarily
            if sparse_mgr is not None and refresh_interval > 0:
                steps_since_mask = i - mask_step
                if steps_since_mask > 1 and steps_since_mask % refresh_interval == 0:
                    sparse_mgr.set_dense_mode(True)

            # Forward
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # After dense refresh step, switch back to sparse
            if sparse_mgr is not None and refresh_interval > 0:
                steps_since_mask = i - mask_step
                if steps_since_mask > 1 and steps_since_mask % refresh_interval == 0:
                    sparse_mgr.set_dense_mode(False)

            # Remove cross-attn hooks after capture
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # ---- Build mask at mask_step and install sparse processors ----
            if i == mask_step and sparse_mgr is None:
                fg_token_mask = self._build_fg_token_mask(
                    mask_builder=mask_builder,
                    hook_mgr=hook_mgr,
                    token_importance=token_importance,
                    latents=latents,
                    use_semantic=use_semantic,
                    use_uniform=use_uniform,
                    skip_ratio=skip_ratio,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    device=device,
                    save_mask_callback=save_mask_callback,
                )

                # Install sparse processors in dense_mode for this step
                # (the forward already ran for mask_step, so we install for next step)
                sparse_mgr = SparseHookManager(self.transformer)
                sparse_mgr.install(fg_mask=fg_token_mask, dense_mode=True)
                # Note: mask_step forward already completed with original processors.
                # The next step (mask_step+1) will run in dense_mode to warm cache,
                # then we switch to sparse.

            # After cache warm-up step (mask_step+1), switch to sparse mode
            if sparse_mgr is not None and i == mask_step + 1:
                sparse_mgr.set_dense_mode(False)

            # Split pred / sigma
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # Standard uniform CFG
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )
            noise_pred_half = noise_pred_uncond_eps + cfg_scale * (
                noise_pred_cond_eps - noise_pred_uncond_eps
            )
            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)
            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
            noise_pred = -noise_pred

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 7. Cleanup ----
        if sparse_mgr is not None:
            sparse_mgr.remove()

        # ---- 8. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    def _build_fg_token_mask(
        self,
        mask_builder,
        hook_mgr,
        token_importance,
        latents,
        use_semantic: bool,
        use_uniform: bool,
        skip_ratio: float,
        patch_h: int,
        patch_w: int,
        latent_h: int,
        latent_w: int,
        device,
        save_mask_callback=None,
    ) -> torch.Tensor:
        """
        Build a (N,) bool foreground token mask from spatial mask builders.

        Returns:
            fg_token_mask: (patch_h * patch_w,) bool tensor on device.
                True = foreground (always computed), False = background (cached).
        """
        import torch.nn.functional as F

        if use_uniform:
            # Random uniform mask: skip_ratio fraction is background
            n_tokens = patch_h * patch_w
            n_fg = int(n_tokens * (1.0 - skip_ratio))
            perm = torch.randperm(n_tokens, device=device)
            fg_token_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
            fg_token_mask[perm[:n_fg]] = True
            return fg_token_mask

        if use_semantic:
            cross_attn_maps = hook_mgr.get_maps()
            spatial_mask = mask_builder.build_mask_from_cross_attention(
                cross_attn_maps, token_importance,
                patch_h, patch_w,
            )
        else:
            # Complexity mask
            spatial_mask = mask_builder.build_mask(latents)

        if spatial_mask is None:
            # No spatial structure — treat everything as foreground (no skip)
            if save_mask_callback:
                fallback = torch.zeros(patch_h, patch_w, device=device, dtype=self.dtype)
                save_mask_callback(fallback, None)
            return torch.ones(patch_h * patch_w, dtype=torch.bool, device=device)

        spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)

        if save_mask_callback:
            if use_semantic:
                save_mask_callback(spatial_mask, None)
            else:
                _, cmap = mask_builder.build_mask_with_viz(latents)
                save_mask_callback(spatial_mask, cmap)

        # Resize to patch grid if needed
        if spatial_mask.shape[0] != patch_h or spatial_mask.shape[1] != patch_w:
            spatial_mask = F.interpolate(
                spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(patch_h, patch_w),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(dtype=self.dtype)

        # Threshold to binary: >0.5 = foreground
        fg_token_mask = (spatial_mask.reshape(-1) > 0.5).to(device=device)
        return fg_token_mask

    def _build_masks_for_dual(
        self,
        mask_builder,
        hook_mgr,
        token_importance,
        latents,
        use_semantic: bool,
        use_uniform: bool,
        skip_ratio: float,
        patch_h: int,
        patch_w: int,
        latent_h: int,
        latent_w: int,
        device,
        save_mask_callback=None,
    ):
        """
        Build both token-level bool mask (for sparse attn) and spatial float
        mask (for spatial CFG).

        Returns:
            fg_token_mask: (patch_h * patch_w,) bool, True = foreground.
            fg_mask_spatial: (1, 1, latent_h, latent_w) float in [0,1], 1 = foreground.
        """
        import torch.nn.functional as F

        if use_uniform:
            n_tokens = patch_h * patch_w
            n_fg = int(n_tokens * (1.0 - skip_ratio))
            perm = torch.randperm(n_tokens, device=device)
            fg_token_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
            fg_token_mask[perm[:n_fg]] = True
            # For spatial CFG: upscale token mask to latent resolution
            spatial_float = fg_token_mask.float().reshape(1, 1, patch_h, patch_w)
            fg_mask_spatial = F.interpolate(
                spatial_float, size=(latent_h, latent_w), mode="nearest",
            ).to(dtype=self.dtype)
            return fg_token_mask, fg_mask_spatial

        if use_semantic:
            cross_attn_maps = hook_mgr.get_maps()
            spatial_mask = mask_builder.build_mask_from_cross_attention(
                cross_attn_maps, token_importance,
                patch_h, patch_w,
            )
        else:
            spatial_mask = mask_builder.build_mask(latents)

        if spatial_mask is None:
            if save_mask_callback:
                fallback = torch.zeros(patch_h, patch_w, device=device, dtype=self.dtype)
                save_mask_callback(fallback, None)
            fg_token_mask = torch.ones(patch_h * patch_w, dtype=torch.bool, device=device)
            fg_mask_spatial = torch.ones(
                1, 1, latent_h, latent_w, device=device, dtype=self.dtype,
            )
            return fg_token_mask, fg_mask_spatial

        spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)

        if save_mask_callback:
            if use_semantic:
                save_mask_callback(spatial_mask, None)
            else:
                _, cmap = mask_builder.build_mask_with_viz(latents)
                save_mask_callback(spatial_mask, cmap)

        # Resize to patch grid if needed
        if spatial_mask.shape[0] != patch_h or spatial_mask.shape[1] != patch_w:
            spatial_mask = F.interpolate(
                spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(patch_h, patch_w),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(dtype=self.dtype)

        # Token mask: binary threshold
        fg_token_mask = (spatial_mask.reshape(-1) > 0.5).to(device=device)

        # Spatial mask: upscale to latent resolution for CFG blending
        fg_spatial = spatial_mask.unsqueeze(0).unsqueeze(0)  # (1,1,patch_h,patch_w)
        if patch_h != latent_h or patch_w != latent_w:
            fg_mask_spatial = F.interpolate(
                fg_spatial.float(), size=(latent_h, latent_w), mode="nearest",
            ).to(dtype=self.dtype)
        else:
            fg_mask_spatial = fg_spatial

        return fg_token_mask, fg_mask_spatial

    @torch.no_grad()
    def generate_dual_degradation(
        self,
        prompt: str,
        seed: int,
        skip_ratio: float = 0.5,
        mask_type: str = "complexity",
        s_fg: float = 7.0,
        s_bg: float = 1.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        grid_size: int = 8,
        blur_sigma: float = 1.0,
        refresh_interval: int = 0,
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
        save_mask_callback=None,
    ) -> Image.Image:
        """
        Dual-Degradation generation (SASD core method).

        Combines two complementary operations for background regions:
          - Operation A (Sparse Forward): background tokens skip attn1 SDPA,
            reuse per-layer cached output → O(N_fg * N) instead of O(N^2).
          - Operation B (CFG Suppression): background regions use low CFG scale
            (s_bg ≈ 1.0), suppressing semantic hallucination and softening
            sparse computation artifacts into natural bokeh.

        Foreground regions get full computation + enhanced CFG (s_fg).

        Args:
            skip_ratio: fraction of tokens treated as background.
            mask_type: "complexity", "semantic", or "uniform".
            s_fg: CFG scale for foreground (aesthetic focal) regions.
            s_bg: CFG scale for background (suppression) regions.
            mask_step: denoising step at which to build mask.
            grid_size: grid size for complexity mask.
            blur_sigma: blur sigma for semantic mask.
            refresh_interval: dense refresh every N steps (0=never).
            save_mask_callback: optional fn(mask, heatmap).
        """
        from src.sparse.mask_builders import ComplexityMaskBuilder, SemanticMaskBuilder
        from src.sparse.cross_attn_hook import CrossAttnHookManager
        from src.sparse.sparse_hook import SparseHookManager

        pipe = self.pipe
        device = self.device

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

        cond_embeds = prompt_embeds.clone()
        cond_mask = prompt_attention_mask.clone()

        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Prepare mask builder ----
        head_dim = self.transformer.head_dim
        patch_size = self.transformer.config.patch_size
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        use_semantic = mask_type == "semantic"
        use_complexity = mask_type == "complexity"
        use_uniform = mask_type == "uniform"

        mask_builder = None
        hook_mgr = None
        token_importance = None

        if use_complexity:
            mask_builder = ComplexityMaskBuilder(grid_size=grid_size, ratio=1.0 - skip_ratio)
        elif use_semantic:
            mask_builder = SemanticMaskBuilder(ratio=1.0 - skip_ratio, blur_sigma=blur_sigma)
            mask_builder.compute_anchor_embeddings(device=device)
            seq_len = cond_embeds.shape[1]
            cond_tokens = self.tokenizer(
                prompt, padding="max_length", max_length=seq_len,
                truncation=True, return_tensors="pt",
            ).to(device)
            token_importance = mask_builder.compute_token_importance(
                prompt=prompt,
                prompt_mask=cond_mask,
                input_ids=cond_tokens["input_ids"],
                tokenizer=self.tokenizer,
            )

        # ---- 6. Denoising loop ----
        sparse_mgr = None
        s_map = None  # (1, 1, latent_h, latent_w) per-pixel CFG scale

        for i, t in enumerate(timesteps):
            # Install cross-attn hooks for semantic mask capture
            if use_semantic and i == mask_step:
                hook_mgr = CrossAttnHookManager(self.transformer)
                hook_mgr.install()

            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
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

            # Periodic dense refresh
            if sparse_mgr is not None and refresh_interval > 0:
                steps_since_mask = i - mask_step
                if steps_since_mask > 1 and steps_since_mask % refresh_interval == 0:
                    sparse_mgr.set_dense_mode(True)

            # Forward
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # After dense refresh, switch back to sparse
            if sparse_mgr is not None and refresh_interval > 0:
                steps_since_mask = i - mask_step
                if steps_since_mask > 1 and steps_since_mask % refresh_interval == 0:
                    sparse_mgr.set_dense_mode(False)

            # Remove cross-attn hooks after capture
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # ---- Build masks at mask_step ----
            if i == mask_step and sparse_mgr is None:
                fg_token_mask, fg_mask_spatial = self._build_masks_for_dual(
                    mask_builder=mask_builder,
                    hook_mgr=hook_mgr,
                    token_importance=token_importance,
                    latents=latents,
                    use_semantic=use_semantic,
                    use_uniform=use_uniform,
                    skip_ratio=skip_ratio,
                    patch_h=patch_h, patch_w=patch_w,
                    latent_h=latent_h, latent_w=latent_w,
                    device=device,
                    save_mask_callback=save_mask_callback,
                )

                # Build per-pixel CFG scale map (Operation B)
                s_map = s_bg + fg_mask_spatial * (s_fg - s_bg)  # (1,1,H,W)

                # Install sparse processors in dense mode for cache warm-up (Operation A)
                sparse_mgr = SparseHookManager(self.transformer)
                sparse_mgr.install(fg_mask=fg_token_mask, dense_mode=True)

            # After cache warm-up step, switch to sparse
            if sparse_mgr is not None and i == mask_step + 1:
                sparse_mgr.set_dense_mode(False)

            # Split pred / sigma
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # ---- CFG: spatial (if mask built) or uniform ----
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )

            if s_map is not None:
                # Spatial CFG: s_fg for foreground, s_bg for background
                noise_pred_half = noise_pred_uncond_eps + s_map * (
                    noise_pred_cond_eps - noise_pred_uncond_eps
                )
            else:
                # Before mask_step: uniform s_fg
                noise_pred_half = noise_pred_uncond_eps + s_fg * (
                    noise_pred_cond_eps - noise_pred_uncond_eps
                )

            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)
            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
            noise_pred = -noise_pred

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 7. Cleanup ----
        if sparse_mgr is not None:
            sparse_mgr.remove()

        # ---- 8. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    @torch.no_grad()
    def generate_accelerated_dual(
        self,
        prompt: str,
        seed: int,
        skip_ratio: float = 0.5,
        mask_type: str = "complexity",
        s_fg: float = 7.0,
        s_bg: float = 1.0,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        grid_size: int = 8,
        blur_sigma: float = 1.0,
        refresh_interval: int = 0,
        skip_step_interval: int = 0,
        sparse_ffn: bool = True,
        sparse_blocks: bool = True,
        sparse_start_step: int = -1,
        full_skip_interval: int = 0,
        taylor_run: int = 0,
        cfg_scale: float = 4.0,
        scaling_watershed: float = 1.0,
        proportional_attn: bool = True,
        save_mask_callback=None,
        region_method: str = "threshold",
    ) -> Image.Image:
        """
        Accelerated Dual-Degradation generation (Phase 6).

        Extends generate_dual_degradation with two additional acceleration layers:
          1. Sparse FFN: background tokens skip the SwiGLU MLP, reuse cached output.
          2. Step-level caching: skip entire transformer forward passes at regular
             intervals, reuse/extrapolate noise_pred.

        Args:
            skip_ratio: fraction of tokens treated as background.
            mask_type: "complexity", "semantic", "uniform", "sdit", or "cfg_magnitude".
            s_fg: CFG scale for foreground regions.
            s_bg: CFG scale for background regions.
            mask_step: denoising step at which to build mask.
            grid_size: grid size for complexity mask.
            blur_sigma: blur sigma for semantic mask.
            refresh_interval: dense refresh every N steps (0=never).
            skip_step_interval: skip transformer forward every N steps after
                warm-up (0=disabled). E.g. 3 means compute-compute-skip pattern.
            sparse_ffn: if True, also sparsify FFN (not just attn1).
            sparse_blocks: if True, install SparseBlockManager (component-level
                sparsity on attn1/FFN). If False, all transformer forwards are
                dense — acceleration comes only from step-level caching + spatial CFG.
            sparse_start_step: step at which to switch from dense to sparse mode.
                -1 (default) = mask_step + 1 (immediate). Use higher values for
                hybrid: dense early steps (quality) + sparse late steps (speed).
            full_skip_interval: skip entire transformer forward every N steps
                during sparse phase (0=disabled). On skipped steps, noise_pred
                is fully extrapolated from cache. Combined with Plan A sparse
                for maximum speedup.
            taylor_run: TaylorSeer-style run-based scheduling (0=disabled).
                When > 0, overrides full_skip_interval with [compute, skip×run]
                cycles and uses 2nd-order Taylor extrapolation. E.g. taylor_run=2
                → [C,S,S] cycles after sparse warm-up (expected ~2.5× on Lumina).
            save_mask_callback: optional fn(mask, heatmap).
        """
        import torch.nn.functional as F
        from src.sparse.mask_builders import ComplexityMaskBuilder, SemanticMaskBuilder
        from src.sparse.cross_attn_hook import CrossAttnHookManager
        from src.sparse.sparse_ffn import SparseBlockManager
        from src.sparse.skip_cache import SkipUpdateCache
        from src.sparse.sdit_mask_builder import SDiTMaskBuilder, CFGMagnitudeMaskBuilder

        pipe = self.pipe
        device = self.device

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

        cond_embeds = prompt_embeds.clone()
        cond_mask = prompt_attention_mask.clone()

        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

        # ---- 2. Cross-attention kwargs ----
        cross_attention_kwargs = {}
        if proportional_attn:
            default_sample_size = getattr(pipe, 'default_sample_size', 128)
            cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

        scaling_factor = math.sqrt(width * height / self.default_image_size ** 2)

        # ---- 3. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        shape = (1, latent_channels, latent_h, latent_w)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.dtype)

        # ---- 4. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 5. Prepare mask builder ----
        head_dim = self.transformer.head_dim
        patch_size = self.transformer.config.patch_size
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        use_semantic = mask_type == "semantic"
        use_complexity = mask_type == "complexity"
        use_uniform = mask_type == "uniform"
        use_sdit = mask_type == "sdit"
        use_cfg_magnitude = mask_type == "cfg_magnitude"

        mask_builder = None
        hook_mgr = None
        token_importance = None

        if use_complexity:
            mask_builder = ComplexityMaskBuilder(grid_size=grid_size, ratio=1.0 - skip_ratio)
        elif use_semantic:
            # Reuse cached builder so CLIP model + anchor embeddings are loaded only once
            # per wrapper lifetime. Rebuild only when hyperparams change.
            if (self._semantic_builder is None
                    or self._semantic_builder.ratio != 1.0 - skip_ratio
                    or self._semantic_builder.blur_sigma != blur_sigma
                    or self._semantic_builder.region_method != region_method):
                self._semantic_builder = SemanticMaskBuilder(
                    ratio=1.0 - skip_ratio, blur_sigma=blur_sigma,
                    region_method=region_method,
                )
                self._semantic_builder.compute_anchor_embeddings(device=device)
                self._token_importance_cache.clear()
            mask_builder = self._semantic_builder

            # Cache token importance per (prompt, seq_len):
            # same prompt with different seeds has identical token importance.
            seq_len = cond_embeds.shape[1]
            cache_key = (prompt, seq_len)
            if cache_key not in self._token_importance_cache:
                cond_tokens = self.tokenizer(
                    prompt, padding="max_length", max_length=seq_len,
                    truncation=True, return_tensors="pt",
                ).to(device)
                self._token_importance_cache[cache_key] = mask_builder.compute_token_importance(
                    prompt=prompt,
                    prompt_mask=cond_mask,
                    input_ids=cond_tokens["input_ids"],
                    tokenizer=self.tokenizer,
                )
            token_importance = self._token_importance_cache[cache_key]
        elif use_sdit:
            mask_builder = SDiTMaskBuilder(
                n_segments=64, compactness=10.0,
                ratio=1.0 - skip_ratio, blur_sigma=blur_sigma,
            )
        elif use_cfg_magnitude:
            mask_builder = CFGMagnitudeMaskBuilder(
                n_segments=64, compactness=10.0,
                ratio=1.0 - skip_ratio, blur_sigma=blur_sigma,
            )

        # ---- 6. Initialize step-level cache ----
        step_cache = SkipUpdateCache(method="linear") if (skip_step_interval > 0 or full_skip_interval > 0) else None
        # Full-skip cache: for extrapolating entire noise_pred when skipping transformer
        if taylor_run > 0:
            from src.sparse.skip_cache import TaylorSkipCache
            full_skip_cache = TaylorSkipCache(order=2)
        elif full_skip_interval > 0:
            full_skip_cache = SkipUpdateCache(method="linear")
        else:
            full_skip_cache = None

        # ---- 7. Denoising loop ----
        block_mgr = None
        sparse_activated = False  # tracks whether sparse mode has been turned on
        s_map = None  # (1, 1, latent_h, latent_w) per-pixel CFG scale
        actual_sparse_start = sparse_start_step if sparse_start_step >= 0 else mask_step + 1
        fg_mask_spatial = None  # set at mask_step, used for bg merge

        for i, t in enumerate(timesteps):
            # --- Full skip: skip entire transformer, extrapolate noise_pred ---
            is_full_skip = False
            if (full_skip_cache is not None
                    and full_skip_cache.has_cache()
                    and sparse_activated):
                steps_since_sparse = i - actual_sparse_start
                if taylor_run > 0:
                    # Run-based: [compute, skip×taylor_run] after 2-step warm-up
                    if steps_since_sparse >= 2:
                        pos_in_run = (steps_since_sparse - 2) % (1 + taylor_run)
                        if pos_in_run != 0:
                            is_full_skip = True
                elif full_skip_interval > 0:
                    # Interval-based (fskip2): every full_skip_interval-th step
                    if steps_since_sparse >= 2 and steps_since_sparse % full_skip_interval == 0:
                        is_full_skip = True

            if is_full_skip:
                # Skip transformer entirely, use extrapolated noise_pred
                noise_pred = full_skip_cache.get_prediction(i)
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                continue

            # --- Decide if bg needs refresh (dense) or can use cache ---
            # bg_refresh=True: run dense transformer, store fresh noise_pred
            # bg_refresh=False: run sparse transformer (fg fresh, bg cached),
            #   then override bg noise_pred with output-level extrapolation
            is_bg_refresh = True  # default: full compute
            if (skip_step_interval > 0
                    and step_cache is not None
                    and step_cache.has_cache()
                    and sparse_activated):
                steps_since_sparse = i - actual_sparse_start
                # First 2 sparse steps are always refresh (warmup for extrapolation)
                if steps_since_sparse >= 2 and steps_since_sparse % skip_step_interval != 0:
                    is_bg_refresh = False

            # Set block manager mode: dense for bg refresh, sparse for bg skip
            if block_mgr is not None and sparse_activated:
                block_mgr.set_dense_mode(is_bg_refresh)

            # Run transformer forward (fg computed fresh every step)

            # Install cross-attn hooks for semantic mask capture
            # Only hook every 4th layer (6 layers instead of 24) — cross-attention
            # patterns are consistent across layers, subsampling reduces overhead ~4x.
            if use_semantic and i == mask_step:
                num_layers = len(self.transformer.layers)
                hook_layers = list(range(0, num_layers, 4))
                hook_mgr = CrossAttnHookManager(self.transformer, layer_indices=hook_layers)
                hook_mgr.install()

            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Timestep
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

            # Forward through transformer
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Remove cross-attn hooks after capture
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # ---- Build masks at mask_step (complexity/semantic/uniform) ----
            if i == mask_step and block_mgr is None and not use_sdit and not use_cfg_magnitude:
                fg_token_mask, fg_mask_spatial = self._build_masks_for_dual(
                    mask_builder=mask_builder,
                    hook_mgr=hook_mgr,
                    token_importance=token_importance,
                    latents=latents,
                    use_semantic=use_semantic,
                    use_uniform=use_uniform,
                    skip_ratio=skip_ratio,
                    patch_h=patch_h, patch_w=patch_w,
                    latent_h=latent_h, latent_w=latent_w,
                    device=device,
                    save_mask_callback=save_mask_callback,
                )

                # Build per-pixel CFG scale map
                s_map = s_bg + fg_mask_spatial * (s_fg - s_bg)

                # Install SparseBlockManager (attn1 + optionally FFN)
                if sparse_blocks:
                    block_mgr = SparseBlockManager(self.transformer)
                    block_mgr.install(
                        fg_mask=fg_token_mask,
                        dense_mode=True,
                        sparse_ffn=sparse_ffn,
                    )

            # Activate sparse mode (hybrid: delay via sparse_start_step)
            # Works even when sparse_blocks=False (fskip-only mode: no block_mgr but
            # full_skip_interval still needs sparse_activated to start caching)
            if not sparse_activated and i >= actual_sparse_start:
                if block_mgr is not None or full_skip_interval > 0 or taylor_run > 0:
                    sparse_activated = True

            # Split pred / sigma
            noise_pred = noise_pred.chunk(2, dim=1)[0]

            # ---- CFG: spatial (if mask built) or uniform ----
            noise_pred_eps, noise_pred_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_pred_cond_eps, noise_pred_uncond_eps = torch.split(
                noise_pred_eps, len(noise_pred_eps) // 2, dim=0
            )

            # ---- Build masks at mask_step (sdit/cfg_magnitude — need CFG split) ----
            if i == mask_step and block_mgr is None and (use_sdit or use_cfg_magnitude):
                if use_sdit:
                    sdit_pred = torch.cat(
                        [-noise_pred_cond_eps, noise_pred_rest[:1]], dim=1,
                    )
                    spatial_mask = mask_builder.build_mask(
                        latents, sdit_pred, float(t),
                    )
                else:  # cfg_magnitude
                    spatial_mask = mask_builder.build_mask(
                        noise_pred_cond_eps, noise_pred_uncond_eps,
                    )

                spatial_mask = spatial_mask.to(device=device, dtype=self.dtype)

                if save_mask_callback:
                    save_mask_callback(spatial_mask, None)

                # Resize to patch grid for token mask
                if spatial_mask.shape[0] != patch_h or spatial_mask.shape[1] != patch_w:
                    spatial_mask = F.interpolate(
                        spatial_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(patch_h, patch_w), mode="nearest",
                    ).squeeze(0).squeeze(0).to(dtype=self.dtype)

                fg_token_mask = (spatial_mask.reshape(-1) > 0.5).to(device=device)

                # Upscale to latent resolution for spatial CFG
                fg_spatial = spatial_mask.unsqueeze(0).unsqueeze(0)
                if patch_h != latent_h or patch_w != latent_w:
                    fg_mask_spatial = F.interpolate(
                        fg_spatial.float(), size=(latent_h, latent_w), mode="nearest",
                    ).to(dtype=self.dtype)
                else:
                    fg_mask_spatial = fg_spatial

                s_map = s_bg + fg_mask_spatial * (s_fg - s_bg)

                if sparse_blocks:
                    block_mgr = SparseBlockManager(self.transformer)
                    block_mgr.install(
                        fg_mask=fg_token_mask,
                        dense_mode=True,
                        sparse_ffn=sparse_ffn,
                    )

            if s_map is not None:
                noise_pred_half = noise_pred_uncond_eps + s_map * (
                    noise_pred_cond_eps - noise_pred_uncond_eps
                )
            else:
                # Before mask is built, use base cfg_scale (same as baseline)
                noise_pred_half = noise_pred_uncond_eps + cfg_scale * (
                    noise_pred_cond_eps - noise_pred_uncond_eps
                )

            noise_pred_eps = torch.cat([noise_pred_half, noise_pred_half], dim=0)
            noise_pred = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
            noise_pred = -noise_pred

            # ---- Output-level bg merge ----
            # On sparse steps: fg uses fresh noise_pred, bg uses extrapolation
            # from step_cache (updated every compute step after mask is built).
            if (not is_bg_refresh
                    and step_cache is not None
                    and step_cache.has_cache()
                    and fg_mask_spatial is not None):
                bg_pred = step_cache.get_prediction(i)
                noise_pred = fg_mask_spatial * noise_pred + (1 - fg_mask_spatial) * bg_pred

            # On bg-refresh steps: store fresh noise_pred for future extrapolation
            if (is_bg_refresh
                    and step_cache is not None
                    and fg_mask_spatial is not None):
                step_cache.store(i, noise_pred)

            # Store in full-skip cache (for full step skipping)
            if full_skip_cache is not None and sparse_activated:
                full_skip_cache.store(i, noise_pred)

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- 8. Cleanup ----
        if block_mgr is not None:
            block_mgr.remove()

        # ---- 9. Decode ----
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    def make_center_mask(
        self,
        latent_h: int,
        latent_w: int,
        fg_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Create a simple center-region foreground mask (Phase 1 debugging).

        Returns: (latent_h, latent_w) float tensor, 1=foreground, 0=background.
        """
        mask = torch.zeros(latent_h, latent_w)
        margin_h = int(latent_h * (1 - fg_ratio) / 2)
        margin_w = int(latent_w * (1 - fg_ratio) / 2)
        mask[margin_h:latent_h - margin_h, margin_w:latent_w - margin_w] = 1.0
        return mask
