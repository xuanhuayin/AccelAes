"""
生成对比图：baseline vs semantic_fskip2 vs semantic_consec2
3 prompts × 1 seed，保存到 outputs/visual_compare/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.sd3_wrapper import SD3DiTWrapper
from pathlib import Path

PROMPTS = [
    ("dog",     "a golden retriever puppy playing in a field of sunflowers"),
    ("city",    "a futuristic city at night with neon lights reflecting on wet streets"),
    ("portrait","portrait of an elderly woman with kind eyes, photorealistic"),
]
SEED = 42

OUT = Path("outputs/visual_compare")
OUT.mkdir(parents=True, exist_ok=True)

wrapper = SD3DiTWrapper(dtype="bf16", device="cuda")

configs = [
    dict(name="baseline",          fn="generate",
         kwargs=dict(steps=28, cfg_scale=7.0)),
    dict(name="semantic_fskip2",   fn="generate_accelerated",
         kwargs=dict(steps=28, mask_type="semantic", skip_ratio=0.5,
                     full_skip_interval=2, n_segments=64)),
    dict(name="semantic_consec2",  fn="generate_accelerated",
         kwargs=dict(steps=28, mask_type="semantic", skip_ratio=0.5,
                     full_skip_consecutive=2, n_segments=64)),
]

for tag, prompt in PROMPTS:
    print(f"\n=== {tag}: {prompt[:50]} ===")
    for cfg in configs:
        if cfg["fn"] == "generate":
            img = wrapper.generate(prompt, seed=SEED, **cfg["kwargs"])
        else:
            img = wrapper.generate_accelerated(prompt, seed=SEED, **cfg["kwargs"])
        path = OUT / f"{tag}_{cfg['name']}.png"
        img.save(path)
        print(f"  saved → {path}")

print(f"\n全部图片已保存至 {OUT.resolve()}")
