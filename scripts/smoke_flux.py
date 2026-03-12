"""
FLUX.1-dev smoke test: baseline vs fskip2 (different warmup settings).
3 prompts × 2 seeds = 6 images/config
"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import CLIPProcessor, CLIPModel
from src.models.flux_wrapper import FluxDiTWrapper

PROMPTS = [
    "a golden retriever puppy playing in a field of sunflowers",
    "a futuristic city at night with neon lights reflecting on wet streets",
    "portrait of an elderly woman with kind eyes, photorealistic",
]
SEEDS = [42, 1337]

wrapper = FluxDiTWrapper(dtype="bf16", device="cuda")

# 预先 encode 所有 prompt（transformer 暂时在 CPU），完成后 transformer 上 GPU
# 这样计时时不含搬运开销，所有 config 公平对比
print("Pre-encoding prompts...")
for p in PROMPTS:
    wrapper._encode_prompt(p)
wrapper._ensure_transformer_on_gpu()
print(f"Prompts cached, transformer on GPU. "
      f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").half()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def clip_score(image, text):
    inputs = clip_proc(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = clip_model(**inputs)
    return out.logits_per_image.item() / 100.0

# 理论加速分析:
# warmup=2: skip at 2,4,...,26 → 13 skips, 15 computes → 28/15 = 1.87x
# warmup=5: skip at 5,7,...,27 → 12 skips, 16 computes → 28/16 = 1.75x
configs = [
    dict(name="baseline",
         fn="generate",
         kwargs=dict(steps=28, guidance_scale=3.5)),
    dict(name="flux_fskip2_w5",
         fn="generate_accelerated",
         kwargs=dict(steps=28, guidance_scale=3.5,
                     full_skip_interval=2, warmup_steps=5)),
    dict(name="flux_fskip2_w2",
         fn="generate_accelerated",
         kwargs=dict(steps=28, guidance_scale=3.5,
                     full_skip_interval=2, warmup_steps=2)),
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
    print(f"[{cfg['name']:20s}]  avg_clip={avg:.4f}  total={elapsed:.1f}s")

print("\n=== Summary ===")
base_clip = results["baseline"]["avg_clip"]
base_time = results["baseline"]["elapsed"]
for name, r in results.items():
    drop = (r["avg_clip"] - base_clip) / base_clip * 100
    speedup = base_time / r["elapsed"] if name != "baseline" else 1.0
    print(f"  {name:20s}  clip={r['avg_clip']:.4f} ({drop:+.1f}%)  speedup={speedup:.2f}x")
