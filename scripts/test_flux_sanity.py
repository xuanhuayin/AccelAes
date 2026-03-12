"""
FLUX.1-dev 基本可行性测试：
1. 跑通一张 baseline 图
2. 打印 transformer 架构关键信息
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from diffusers import FluxPipeline

MODEL = "black-forest-labs/FLUX.1-dev"
print(f"Loading {MODEL} ...")
t0 = time.time()

# 全部先加载到 CPU
pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
print(f"Load time: {time.time()-t0:.1f}s")

# 显存策略：
#   text encode 阶段：T5(9.2GB)+CLIP(0.3GB) 上 GPU，encode 完移回 CPU
#   denoising 阶段：transformer(24GB) 上 GPU
#   decode 阶段：VAE(0.3GB) 上 GPU
# 峰值约 26GB，32GB 显卡可以跑通

# ---- 打印关键架构信息（CPU 上读，不占显存）----
tr = pipe.transformer
print(f"\n=== FluxTransformer2DModel ===")
print(f"  double-stream blocks : {len(tr.transformer_blocks)}")
print(f"  single-stream blocks : {len(tr.single_transformer_blocks)}")
print(f"  in_channels          : {tr.config.in_channels}")
print(f"  patch_size           : {tr.config.patch_size}")
print(f"  VAE scale factor     : {pipe.vae_scale_factor}")
print(f"  scheduler            : {type(pipe.scheduler).__name__}")

# ---- 分阶段上 GPU 生成一张图 ----
print("\n--- Stage 1: encode prompt (T5 + CLIP on GPU) ---")
pipe.text_encoder.to("cuda")
pipe.text_encoder_2.to("cuda")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

with torch.no_grad():
    (prompt_embeds, pooled_prompt_embeds,
     text_ids) = pipe.encode_prompt(
        prompt="a golden retriever puppy playing in a field of sunflowers",
        prompt_2=None,
        device="cuda",
        num_images_per_prompt=1,
    )

# 编码完把 text encoder 移回 CPU，腾出显存
pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")
torch.cuda.empty_cache()
print(f"  GPU mem after offload: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

print("\n--- Stage 2: denoising (transformer on GPU) ---")
pipe.transformer.to("cuda")
pipe.vae.to("cuda")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# 准备 latents（FLUX packed format）
height, width = 1024, 1024
latent_h = height // pipe.vae_scale_factor   # 128
latent_w = width  // pipe.vae_scale_factor   # 128

generator = torch.Generator(device="cuda").manual_seed(42)
# num_channels_latents = in_channels // 4 (unpacked VAE channels = 16)
num_channels_latents = tr.config.in_channels // 4   # 16
# prepare_latents 返回 (packed_latents, img_ids)
latents, img_ids = pipe.prepare_latents(
    1, num_channels_latents, height, width,
    torch.bfloat16, "cuda", generator,
)

import numpy as np
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

num_steps = 28
# FLUX 的 latent 经过 2×2 pack 后送入 transformer
# seq_len = (latent_h // 2) * (latent_w // 2)
image_seq_len = (latent_h // 2) * (latent_w // 2)   # 64*64=4096 for 1024×1024

mu = calculate_shift(
    image_seq_len,
    pipe.scheduler.config.base_image_seq_len,
    pipe.scheduler.config.max_image_seq_len,
    pipe.scheduler.config.base_shift,
    pipe.scheduler.config.max_shift,
)
sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device="cuda")
timesteps = pipe.scheduler.timesteps

# txt_ids for RoPE (img_ids already returned by prepare_latents)
txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device="cuda", dtype=torch.bfloat16)

t1 = time.time()
with torch.no_grad():
    for i, t in enumerate(timesteps):
        t_expand = t.expand(latents.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=t_expand / 1000,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=torch.tensor([3.5], device="cuda", dtype=torch.bfloat16),
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

elapsed = time.time() - t1
print(f"  Denoising: {elapsed:.1f}s  ({elapsed/28*1000:.0f}ms/step)")

print("\n--- Stage 3: decode ---")
latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
with torch.no_grad():
    image = pipe.vae.decode(latents, return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="pil")[0]

os.makedirs("outputs/flux_test", exist_ok=True)
image.save("outputs/flux_test/baseline.png")
print(f"Saved → outputs/flux_test/baseline.png")
print(f"Peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
