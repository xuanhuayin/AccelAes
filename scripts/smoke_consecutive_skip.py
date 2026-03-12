"""
Smoke test: compare fskip2 (linear, alternating) vs fskip_consec2 (quadratic, consecutive).
3 prompts × 2 seeds = 6 images per config.
"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import CLIPProcessor, CLIPModel
from src.models.sd3_wrapper import SD3DiTWrapper

PROMPTS = [
    "a golden retriever puppy playing in a field of sunflowers",
    "a futuristic city at night with neon lights reflecting on wet streets",
    "portrait of an elderly woman with kind eyes, photorealistic",
]
SEEDS = [42, 1337]

wrapper = SD3DiTWrapper(dtype="bf16", device="cuda")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").half()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def clip_score(image, text):
    inputs = clip_proc(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = clip_model(**inputs)
    return out.logits_per_image.item() / 100.0

# 理论跳步分析 (steps=28, mask_step=5, actual_sparse_start=6)
# fskip2 (interval=2):      warmup=2, sparse 20 steps, skip 10 → 18 computes → ~1.56x
# consec2 (consecutive=2):  warmup=3, sparse 19 steps, skip 13 → 15 computes → ~1.87x

configs = [
    dict(name="baseline",
         fn="generate",
         kwargs=dict(steps=28, cfg_scale=7.0)),
    dict(name="cfg_mag_fskip2",
         fn="generate_accelerated",
         kwargs=dict(steps=28, mask_type="cfg_magnitude", skip_ratio=0.5,
                     full_skip_interval=2, n_segments=64)),
    dict(name="cfg_mag_consec2",
         fn="generate_accelerated",
         kwargs=dict(steps=28, mask_type="cfg_magnitude", skip_ratio=0.5,
                     full_skip_consecutive=2, n_segments=64)),
]

results = {}
for cfg in configs:
    scores = []
    t0 = time.time()
    for prompt in PROMPTS:
        for seed in SEEDS:
            if cfg["fn"] == "generate":
                img = wrapper.generate(prompt, seed=seed, **cfg["kwargs"])
            else:
                img = wrapper.generate_accelerated(prompt, seed=seed, **cfg["kwargs"])
            s = clip_score(img, prompt)
            scores.append(s)
    elapsed = time.time() - t0
    avg = sum(scores) / len(scores)
    results[cfg["name"]] = dict(avg_clip=avg, elapsed=elapsed)
    print(f"[{cfg['name']:25s}]  avg_clip={avg:.4f}  total={elapsed:.1f}s")

print("\n=== Summary ===")
base_clip = results["baseline"]["avg_clip"]
base_time = results["baseline"]["elapsed"]
for name, r in results.items():
    drop = (r["avg_clip"] - base_clip) / base_clip * 100
    speedup = base_time / r["elapsed"] if name != "baseline" else 1.0
    print(f"  {name:25s}  clip={r['avg_clip']:.4f} ({drop:+.1f}%)  speedup={speedup:.2f}x")
