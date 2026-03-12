"""
Stable Diffusion 3 (MMDiT) Wrapper.

Wraps StableDiffusion3Pipeline to expose:
  - Standard generation (baseline)
  - Accelerated generation: spatial CFG (cfg_magnitude mask) + step-level caching
"""

import torch
import torch.nn.functional as F  # noqa: F401 (used inside generate_accelerated)
from PIL import Image


class SD3DiTWrapper:
    """Wrapper around SD3 medium for SASD experiments."""

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        dtype: str = "bf16",
        device: str = "cuda",
        cache_dir: str = None,
    ):
        from diffusers import StableDiffusion3Pipeline

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.device = device
        self.dtype = torch_dtype

        print(f"Loading {model_name} ({dtype})...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        ).to(device)

        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = self.pipe.vae_scale_factor  # 8

        print(f"Model loaded. VAE scale factor: {self.vae_scale_factor}")

    def generate(
        self,
        prompt: str,
        seed: int,
        cfg_scale: float = 7.0,
        steps: int = 28,
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
    def generate_accelerated(
        self,
        prompt: str,
        seed: int,
        mask_type: str = "cfg_magnitude",
        cfg_scale: float = 7.0,
        s_fg: float = 9.0,
        s_bg: float = 2.0,
        steps: int = 28,
        height: int = 1024,
        width: int = 1024,
        mask_step: int = 5,
        skip_ratio: float = 0.5,
        full_skip_interval: int = 0,
        full_skip_consecutive: int = 0,
        sparse_ffn: bool = False,
        sparse_attn: bool = False,
        sparse_residual_ffn: bool = False,
        n_segments: int = 64,
    ) -> Image.Image:
        """
        Accelerated SD3 generation: spatial CFG + sparse attn/FFN + step caching.

        Args:
            mask_type: "cfg_magnitude" or "semantic".
              - cfg_magnitude: uses |cond - uncond| magnitude + SLIC regions.
              - semantic: uses SD3 joint attention image→text affinity + SLIC.
            s_fg/s_bg: guidance scale for foreground/background regions.
            mask_step: denoising step at which to build the mask.
            skip_ratio: fraction of patches treated as background (low CFG).
            full_skip_interval: alternating skip — skip every N-th sparse step (0=off).
                e.g. interval=2 → compute,SKIP,compute,SKIP,... → ~50% sparse skipped.
            full_skip_consecutive: consecutive skip — skip C steps then compute 1 (0=off).
                e.g. consecutive=2 → SKIP,SKIP,compute,SKIP,SKIP,compute,... → ~67% skipped.
                Uses quadratic extrapolation; requires warmup of C+1 steps.
                Mutually exclusive with full_skip_interval.
            sparse_ffn: if True, skip FF for background tokens (reuse cache).
                        Note: may cause quality issues due to AdaLayerNorm modulation.
            sparse_attn: if True, skip attention for background tokens (reuse cache).
                         More stable than sparse_ffn; gate_msa is applied outside.
            sparse_residual_ffn: if True, cache gate_mlp*ff(norm_h) for bg tokens.
                         Avoids the stale-ff issue by caching the full residual.
            n_segments: SLIC segment count for region segmentation.
        """
        from src.sparse.sdit_mask_builder import CFGMagnitudeMaskBuilder
        from src.sparse.sd3_joint_hook import SD3JointAttnHookManager
        from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
        from src.sparse.sd3_sparse_ffn import SD3SparseBlockManager
        from src.sparse.sd3_sparse_attn import SD3SparseAttnManager
        from src.sparse.sd3_sparse_residual_ffn import SD3SparseResidualFFNManager
        from src.sparse.skip_cache import SkipUpdateCache

        pipe = self.pipe
        device = self.device
        use_semantic = (mask_type == "semantic")

        # ---- 1. Encode prompt ----
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt="",
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
            device=device,
        )

        # SD3 batch order: [uncond, cond]
        prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # ---- 2. Prepare latents ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels  # 16 for SD3

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator, device=device, dtype=self.dtype,
        )

        # ---- 3. Scheduler ----
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # ---- 4. Mask builder & hook setup ----
        patch_size = self.transformer.config.patch_size  # 2 for SD3
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        if use_semantic:
            mask_builder = SD3SemanticMaskBuilder(
                ratio=1.0 - skip_ratio,
                n_segments=n_segments,
                compactness=0.1,
                blur_sigma=1.5,
            )
            num_layers = len(self.transformer.transformer_blocks)
            hook_layers = list(range(0, num_layers, 4))
            hook_mgr = SD3JointAttnHookManager(self.transformer, layer_indices=hook_layers)
        else:
            mask_builder = CFGMagnitudeMaskBuilder(
                n_segments=n_segments,
                compactness=10.0,
                ratio=1.0 - skip_ratio,
                blur_sigma=1.5,
            )
            hook_mgr = None

        # ---- 5. Sparse managers (installed after mask is built) ----
        block_mgr    = SD3SparseBlockManager(self.transformer)       if sparse_ffn          else None
        attn_mgr     = SD3SparseAttnManager(self.transformer)        if sparse_attn         else None
        residual_mgr = SD3SparseResidualFFNManager(self.transformer) if sparse_residual_ffn else None

        # ---- 6. Full-skip cache ----
        assert not (full_skip_interval > 0 and full_skip_consecutive > 0), \
            "full_skip_interval and full_skip_consecutive are mutually exclusive"
        if full_skip_consecutive > 0:
            full_skip_cache = SkipUpdateCache(method="quadratic")
            _skip_warmup = full_skip_consecutive + 1   # need C+1 points for quadratic
        elif full_skip_interval > 0:
            full_skip_cache = SkipUpdateCache(method="linear")
            _skip_warmup = 2
        else:
            full_skip_cache = None
            _skip_warmup = 2

        sparse_activated = False
        s_map = None        # (1, 1, latent_h, latent_w) per-pixel CFG scale
        fg_token_mask = None  # (N_img,) bool, True=foreground
        actual_sparse_start = mask_step + 1

        # ---- 7. Denoising loop ----
        for i, t in enumerate(timesteps):

            # --- Full skip: extrapolate noise_pred, skip transformer ---
            if (full_skip_cache is not None
                    and full_skip_cache.has_cache()
                    and sparse_activated):
                steps_since_sparse = i - actual_sparse_start

                if full_skip_consecutive > 0:
                    # Pattern: SKIP×C, compute, SKIP×C, compute, ...
                    if steps_since_sparse >= _skip_warmup:
                        steps_after_warmup = steps_since_sparse - _skip_warmup
                        period = full_skip_consecutive + 1
                        pos = steps_after_warmup % period
                        if pos < full_skip_consecutive:
                            lookahead = pos + 1
                            noise_pred = full_skip_cache.get_prediction(lookahead=lookahead)
                            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                            continue

                elif full_skip_interval > 0:
                    # Pattern: compute, SKIP, compute, SKIP, ...
                    if steps_since_sparse >= _skip_warmup and steps_since_sparse % full_skip_interval == 0:
                        noise_pred = full_skip_cache.get_prediction(lookahead=1)
                        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                        continue

            # --- Sparse ops: dense mode during warmup, then switch to sparse ---
            if sparse_activated:
                steps_since_sparse = i - actual_sparse_start
                is_warmup = steps_since_sparse < _skip_warmup
                if block_mgr is not None:
                    block_mgr.set_dense_mode(is_warmup)
                if attn_mgr is not None:
                    attn_mgr.set_dense_mode(is_warmup)
                if residual_mgr is not None:
                    residual_mgr.set_dense_mode(is_warmup)

            # Install hooks for semantic mask capture at mask_step
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.install()

            # Duplicate latents for CFG
            latent_model_input = torch.cat([latents] * 2, dim=0)

            # Transformer forward
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=prompt_embeds_cfg,
                pooled_projections=pooled_cfg,
                return_dict=False,
            )[0]

            # Remove hooks after capture step
            if use_semantic and i == mask_step and hook_mgr is not None:
                hook_mgr.remove()

            # Split uncond / cond (SD3 batch order: [uncond, cond])
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

            # --- Build mask at mask_step ---
            if i == mask_step and s_map is None:
                if use_semantic:
                    affinity_maps = hook_mgr.get_affinity_maps()
                    mask = mask_builder.build_mask(
                        affinity_maps=affinity_maps,
                        patch_h=patch_h,
                        patch_w=patch_w,
                    )
                    # mask is at patch resolution; upsample to latent for s_map
                    mask_latent = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(latent_h, latent_w), mode="nearest",
                    ).squeeze(0).squeeze(0)
                else:
                    mask = mask_builder.build_mask(
                        cond_pred=noise_pred_cond,
                        uncond_pred=noise_pred_uncond,
                    )
                    # cfg_magnitude mask is already at latent resolution
                    # downsample to patch space for token mask
                    mask_latent = mask
                    mask = F.avg_pool2d(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        kernel_size=patch_size, stride=patch_size,
                    ).squeeze(0).squeeze(0)

                mask = mask.to(device=device, dtype=self.dtype)
                mask_latent = mask_latent.to(device=device, dtype=self.dtype)

                # Spatial CFG scale map (latent resolution)
                s_map = s_bg + mask_latent.unsqueeze(0).unsqueeze(0) * (s_fg - s_bg)

                # Token fg mask for sparse FFN (patch resolution, flattened)
                fg_token_mask = (mask > 0.5).reshape(-1)  # (N_img,)

                # Install sparse FFN blocks (start in dense mode for warmup)
                if block_mgr is not None:
                    block_mgr.install(fg_mask=fg_token_mask, dense_mode=True)

                # Install sparse attention (start in dense mode for warmup)
                if attn_mgr is not None:
                    attn_mgr.install(fg_mask=fg_token_mask, dense_mode=True)

                # Install sparse residual FFN (start in dense mode for warmup)
                if residual_mgr is not None:
                    residual_mgr.install(fg_mask=fg_token_mask, dense_mode=True)

                sparse_activated = True

            # --- Spatial CFG ---
            if s_map is not None:
                guided = noise_pred_uncond + s_map * (noise_pred_cond - noise_pred_uncond)
            else:
                guided = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            # Store in full-skip cache
            if full_skip_cache is not None and sparse_activated:
                full_skip_cache.store(i, guided)

            latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]

        # ---- 8. Cleanup ----
        if block_mgr is not None:
            block_mgr.remove()
        if attn_mgr is not None:
            attn_mgr.remove()
        if residual_mgr is not None:
            residual_mgr.remove()

        # ---- 9. Decode ----
        # SD3 VAE: undo scaling and apply shift factor
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image


# Alias for backward compatibility with existing scripts
SD3Wrapper = SD3DiTWrapper
