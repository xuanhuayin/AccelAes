"""
生成 FLUX 对比图：baseline vs fskip2_w5
3 prompts × 1 seed，保存到 outputs/flux_visual/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.flux_wrapper import FluxDiTWrapper
from pathlib import Path

PROMPTS = [
    ("dog",     "a golden retriever puppy playing in a field of sunflowers"),
    ("city",    "a futuristic city at night with neon lights reflecting on wet streets"),
    ("portrait","portrait of an elderly woman with kind eyes, photorealistic"),
]
SEED = 42
OUT = Path("outputs/flux_visual")
OUT.mkdir(parents=True, exist_ok=True)

wrapper = FluxDiTWrapper(dtype="bf16", device="cuda")

# 预先 encode
for _, prompt in PROMPTS:
    wrapper._encode_prompt(prompt)
wrapper._ensure_transformer_on_gpu()

configs = [
    dict(name="baseline",     fn="generate",
         kwargs=dict(steps=28, guidance_scale=3.5)),
    dict(name="fskip2_w5",   fn="generate_accelerated",
         kwargs=dict(steps=28, guidance_scale=3.5,
                     full_skip_interval=2, warmup_steps=5)),
]

for tag, prompt in PROMPTS:
    print(f"\n=== {tag} ===")
    for cfg in configs:
        if cfg["fn"] == "generate":
            img = wrapper.generate(prompt, seed=SEED, **cfg["kwargs"])
        else:
            img = wrapper.generate_accelerated(prompt, seed=SEED, **cfg["kwargs"])
        path = OUT / f"{tag}_{cfg['name']}.png"
        img.save(path)
        print(f"  saved → {path}")

print(f"\n全部图片已保存至 {OUT.resolve()}")
