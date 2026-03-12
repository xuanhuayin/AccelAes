"""
FLUX.1-dev Wrapper.

内存策略：
  - 所有组件先加载到 CPU
  - encode 阶段：T5(9.2GB) + CLIP(0.3GB) 临时上 GPU，encode 完移回 CPU
  - denoising 阶段：transformer(22GB) + VAE(0.3GB) 在 GPU

加速方案：
  FLUX.1-dev 使用 guidance distillation（单 pass，guidance embedding），
  不走双 pass CFG，因此不适合 spatial CFG。

  已实现加速组件：
  1. step-level caching (fskip2): 步级预测外推，架构无关。
  2. semantic sparse attention: 语义前景 mask（来自 double-stream joint attention affinity）
     + Q_fg @ K_all, V_all 稀疏注意力（应用于 single-stream blocks）。
     注意：不做 sparse FFN（AdaLN 耦合导致 FFN 输出跨步缓存无效，同 SD3）。
     注意：不做 spatial CFG（无双 pass CFG）。
"""

import numpy as np
import torch
from PIL import Image


class FluxDiTWrapper:
    """Wrapper around FLUX.1-dev for SASD experiments."""

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        dtype: str = "bf16",
        device: str = "cuda",
        cache_dir: str = None,
    ):
        from diffusers import FluxPipeline

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.device = device
        self.dtype = torch_dtype
        self.model_name = model_name

        print(f"Loading {model_name} ({dtype}) ...")
        # 全部先加载到 CPU，denoising 时再按需上 GPU
        self.pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = self.pipe.vae_scale_factor  # 8

        # VAE 常驻 GPU，transformer 按需上 GPU（encode 时需先移回 CPU）
        self.vae.to(device)
        self._transformer_on_gpu = False
        self._prompt_cache: dict = {}   # {prompt: (embeds, pooled, text_ids)}
        print(f"VAE on GPU. GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # ------------------------------------------------------------------ #
    #  内部工具                                                             #
    # ------------------------------------------------------------------ #

    def _encode_prompt(self, prompt: str):
        """
        encode prompt，带缓存避免重复编码。
        transformer(22GB) + T5(9.2GB) 不能同时上 GPU，
        所以 encode 时先把 transformer 移回 CPU，encode 完再移回 GPU。
        """
        if prompt in self._prompt_cache:
            return self._prompt_cache[prompt]

        pipe = self.pipe
        device = self.device

        # transformer 先让位
        if self._transformer_on_gpu:
            self.transformer.to("cpu")
            self._transformer_on_gpu = False
            torch.cuda.empty_cache()

        pipe.text_encoder.to(device)
        pipe.text_encoder_2.to(device)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
            )

        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        self._prompt_cache[prompt] = (prompt_embeds, pooled_prompt_embeds, text_ids)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _ensure_transformer_on_gpu(self):
        if not self._transformer_on_gpu:
            self.transformer.to(self.device)
            self._transformer_on_gpu = True

    def _setup_scheduler(self, num_steps: int, height: int, width: int):
        """FLUX 动态 sigma shifting，返回 timesteps。"""
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        latent_h = height // self.vae_scale_factor
        latent_w = width  // self.vae_scale_factor
        # FLUX 在 transformer 前做 2×2 pack，seq_len = (H//2)*(W//2)
        image_seq_len = (latent_h // 2) * (latent_w // 2)

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)
        return self.scheduler.timesteps

    def _prepare_latents(self, height: int, width: int, seed: int):
        num_channels = self.transformer.config.in_channels // 4  # 16
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents, img_ids = self.pipe.prepare_latents(
            1, num_channels, height, width,
            self.dtype, self.device, generator,
        )
        return latents, img_ids

    def _transformer_step(self, latents, t, prompt_embeds,
                          pooled_prompt_embeds, txt_ids, img_ids,
                          guidance_scale: float):
        t_in = t.expand(latents.shape[0])
        with torch.no_grad():
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=t_in / 1000,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                guidance=torch.tensor(
                    [guidance_scale], device=self.device, dtype=self.dtype),
                return_dict=False,
            )[0]
        return noise_pred

    def _decode(self, latents, height: int, width: int) -> Image.Image:
        latents = self.pipe._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        seed: int,
        guidance_scale: float = 3.5,
        steps: int = 28,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """标准 baseline 生成。"""
        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(prompt)
        self._ensure_transformer_on_gpu()
        latents, img_ids = self._prepare_latents(height, width, seed)
        timesteps = self._setup_scheduler(steps, height, width)

        txt_ids = torch.zeros(
            prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        for t in timesteps:
            noise_pred = self._transformer_step(
                latents, t, prompt_embeds, pooled_prompt_embeds,
                txt_ids, img_ids, guidance_scale)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return self._decode(latents, height, width)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_accelerated(
        self,
        prompt: str,
        seed: int,
        guidance_scale: float = 3.5,
        steps: int = 28,
        height: int = 1024,
        width: int = 1024,
        full_skip_interval: int = 0,
        warmup_steps: int = 2,
    ) -> Image.Image:
        """
        加速生成：step-level caching (fskip)。

        FLUX 使用 guidance distillation (单 pass)，不做空间 CFG。
        加速完全来自 step-level noise_pred 缓存与外推。

        Args:
            full_skip_interval: 隔 N 步跳一步（0=关闭）。2 = 交替跳，~1.87x。
            warmup_steps: 开始跳步前先积累的步数（至少2，用于线性外推）。
        """
        from src.sparse.skip_cache import SkipUpdateCache

        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(prompt)
        self._ensure_transformer_on_gpu()
        latents, img_ids = self._prepare_latents(height, width, seed)
        timesteps = self._setup_scheduler(steps, height, width)

        txt_ids = torch.zeros(
            prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        cache = SkipUpdateCache(method="linear") if full_skip_interval > 0 else None
        warmup = max(2, warmup_steps)

        for i, t in enumerate(timesteps):
            # Step skip 判断
            if cache is not None and cache.has_cache() and i >= warmup:
                if (i - warmup) % full_skip_interval == 0:
                    noise_pred = cache.get_prediction(lookahead=1)
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    continue

            noise_pred = self._transformer_step(
                latents, t, prompt_embeds, pooled_prompt_embeds,
                txt_ids, img_ids, guidance_scale)

            if cache is not None:
                cache.store(i, noise_pred)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return self._decode(latents, height, width)

    @torch.no_grad()
    def generate_taylor(
        self,
        prompt: str,
        seed: int,
        guidance_scale: float = 3.5,
        steps: int = 28,
        height: int = 1024,
        width: int = 1024,
        taylor_run: int = 2,
        warmup_steps: int = 5,
        taylor_order: int = 2,
    ) -> Image.Image:
        """
        TaylorSeer-style step-level caching for FLUX (arXiv 2503.06923).

        调度方式：warmup_steps 个完整计算步，之后以 [compute, skip×taylor_run] 为周期循环。
        预测方式：TaylorSkipCache，用历史 noise_pred 做 Taylor 展开外推（默认二阶）。

        理论加速比（28 steps, warmup=5）：
          taylor_run=1  → ~1.65×  (17 compute steps)
          taylor_run=2  → ~2.15×  (13 compute steps)
          taylor_run=4  → ~2.80×  (10 compute steps)
          taylor_run=5  → ~3.11×  ( 9 compute steps)

        Args:
            taylor_run   : 每个 compute 步后连续跳过的步数。
            warmup_steps : 开始跳步前的热身计算步数（最少 2）。
            taylor_order : 1=线性外推，2=二阶 Taylor 外推（默认）。
        """
        from src.sparse.skip_cache import TaylorSkipCache

        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(prompt)
        self._ensure_transformer_on_gpu()
        latents, img_ids = self._prepare_latents(height, width, seed)
        timesteps = self._setup_scheduler(steps, height, width)

        txt_ids = torch.zeros(
            prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        cache = TaylorSkipCache(order=taylor_order)
        warmup = max(2, warmup_steps)
        run_len = 1 + taylor_run   # [compute, skip×taylor_run] cycle length

        for i, t in enumerate(timesteps):
            # After warmup: position within current run
            if i >= warmup:
                pos_in_run = (i - warmup) % run_len
            else:
                pos_in_run = 0  # warmup → always compute

            if i >= warmup and pos_in_run > 0 and cache.ready():
                # Skip step: Taylor-extrapolate from stored history
                noise_pred = cache.predict(target_step=i)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                continue

            # Compute step: run transformer + store
            noise_pred = self._transformer_step(
                latents, t, prompt_embeds, pooled_prompt_embeds,
                txt_ids, img_ids, guidance_scale)
            cache.store(i, noise_pred)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return self._decode(latents, height, width)

    @torch.no_grad()
    def generate_teacache(
        self,
        prompt: str,
        seed: int,
        guidance_scale: float = 3.5,
        steps: int = 28,
        height: int = 1024,
        width: int = 1024,
        threshold: float = 0.10,
        warmup_steps: int = 3,
    ) -> Image.Image:
        """
        TeaCache-style adaptive step caching for FLUX (arXiv 2411.19108).

        Skip criterion: accumulated per-step estimated noise_pred change
        (measured between last two compute steps) < threshold.

        Threshold calibration (approximate, 28 steps):
          threshold=0.10  → ~2×  speedup
          threshold=0.20  → ~3×  speedup
          threshold=0.30  → ~4×  speedup
        """
        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(prompt)
        self._ensure_transformer_on_gpu()
        latents, img_ids = self._prepare_latents(height, width, seed)
        timesteps = self._setup_scheduler(steps, height, width)

        txt_ids = torch.zeros(
            prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        prev_pred_1 = None   # noise_pred from the most recent compute step
        prev_pred_2 = None   # noise_pred from the second-most-recent compute step
        acc = 0.0            # accumulated estimated change since last compute

        for i, t in enumerate(timesteps):
            # Estimate per-step change rate from last two computed predictions
            per_step_delta = 0.0
            if prev_pred_2 is not None:
                per_step_delta = float(
                    (prev_pred_1 - prev_pred_2).norm() /
                    (prev_pred_2.norm() + 1e-8)
                )

            # Skip decision: accumulated predicted change is below threshold
            should_skip = (
                prev_pred_1 is not None
                and i >= warmup_steps
                and acc + per_step_delta < threshold
            )

            if should_skip:
                noise_pred = prev_pred_1   # 0th-order reuse
                acc += per_step_delta
            else:
                noise_pred = self._transformer_step(
                    latents, t, prompt_embeds, pooled_prompt_embeds,
                    txt_ids, img_ids, guidance_scale)
                prev_pred_2 = prev_pred_1
                prev_pred_1 = noise_pred.clone()
                acc = 0.0

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return self._decode(latents, height, width)

    @torch.no_grad()
    def generate_accelerated_sparse(
        self,
        prompt: str,
        seed: int,
        guidance_scale: float = 3.5,
        steps: int = 28,
        height: int = 1024,
        width: int = 1024,
        # Semantic mask
        mask_step: int = 8,
        skip_ratio: float = 0.5,
        n_segments: int = 64,
        region_method: str = "threshold",
        # Sparse attention
        sparse_blocks: bool = True,
        # Step-level caching
        full_skip_interval: int = 2,
        warmup_steps: int = 5,
    ) -> Image.Image:
        """
        加速生成：semantic mask + sparse attention（single-stream blocks）+ step-level cache。

        组件：
          1. 语义 mask：在 mask_step 从 double-stream joint attention 提取 image→text affinity，
             通过 SLIC 分割得到前景二值 mask。
          2. 稀疏注意力：Q_fg @ K_all, V_all，应用于所有 single-stream blocks（38 个）。
             bg token 复用缓存的 attention 输出。
          3. 步级缓存：全局 noise_pred 线性外推（full_skip_interval=2）。

        无 spatial CFG（FLUX 无双 pass CFG），无 sparse FFN（AdaLN 耦合）。
        """
        from src.sparse.flux_joint_hook import FluxJointAttnHookManager
        from src.sparse.flux_sparse_manager import FluxSparseManager
        from src.sparse.sd3_semantic_mask import SD3SemanticMaskBuilder
        from src.sparse.skip_cache import SkipUpdateCache

        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt(prompt)
        self._ensure_transformer_on_gpu()
        latents, img_ids = self._prepare_latents(height, width, seed)
        timesteps = self._setup_scheduler(steps, height, width)

        txt_ids = torch.zeros(
            prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype
        )

        # FLUX 2×2 spatial packing → packed latent is (H/8/2, W/8/2)
        latent_h = height // self.vae_scale_factor
        latent_w = width  // self.vae_scale_factor
        patch_h = latent_h // 2
        patch_w = latent_w // 2

        # Step-level cache
        step_cache = SkipUpdateCache(method="linear") if full_skip_interval > 0 else None
        warmup = max(2, warmup_steps)

        # Mask builder
        mask_builder = SD3SemanticMaskBuilder(
            ratio=1.0 - skip_ratio,
            n_segments=n_segments,
            compactness=0.1,
            blur_sigma=1.5,
            region_method=region_method,
        ) if sparse_blocks else None

        hook_mgr = None
        sparse_mgr = None
        sparse_activated = False

        for i, t in enumerate(timesteps):
            # ---- Full skip (step-level cache) ----
            if (
                step_cache is not None
                and step_cache.has_cache()
                and i >= warmup
                and (i - warmup) % full_skip_interval == 0
            ):
                noise_pred = step_cache.get_prediction(lookahead=1)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                continue

            # ---- Install joint attention hook at mask_step ----
            if sparse_blocks and i == mask_step and hook_mgr is None:
                hook_mgr = FluxJointAttnHookManager(
                    self.transformer,
                    layer_indices=list(range(0, len(self.transformer.transformer_blocks), 3)),
                )
                hook_mgr.install()

            # ---- Transformer forward ----
            noise_pred = self._transformer_step(
                latents, t, prompt_embeds, pooled_prompt_embeds,
                txt_ids, img_ids, guidance_scale,
            )

            # ---- Build mask from joint affinity (captured during mask_step forward) ----
            if sparse_blocks and i == mask_step and hook_mgr is not None and sparse_mgr is None:
                affinity_maps = hook_mgr.get_affinity_maps()
                hook_mgr.remove()
                hook_mgr = None

                if affinity_maps and mask_builder is not None:
                    mask_spatial = mask_builder.build_mask(affinity_maps, patch_h, patch_w)
                    mask_spatial = mask_spatial.to(device=self.device, dtype=self.dtype)
                    fg_token_mask = (mask_spatial.reshape(-1) > 0.5)

                    n_txt = prompt_embeds.shape[1]
                    sparse_mgr = FluxSparseManager(self.transformer)
                    sparse_mgr.install(fg_token_mask, dense_mode=True, n_txt=n_txt)

            # ---- Activate sparse mode 1 step after mask_step (cache is now populated) ----
            if sparse_mgr is not None and not sparse_activated and i > mask_step:
                sparse_activated = True
                sparse_mgr.set_dense_mode(False)

            # ---- Store in step cache ----
            if step_cache is not None:
                step_cache.store(i, noise_pred)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if sparse_mgr is not None:
            sparse_mgr.remove()

        return self._decode(latents, height, width)
