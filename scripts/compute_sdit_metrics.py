"""Compute metrics for SDIT evaluation images."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
import torch
import ImageReward as RM
import open_clip
import lpips as lpips_lib
from cleanfid import fid as cleanfid

device = "cuda"
ROOT        = "/home/runkai/xuanhua/SASD"
SDIT_DIR    = f"{ROOT}/outputs/sdit_eval/images"
BASE_DIR    = f"{ROOT}/outputs/semantic_full_eval/baseline/images"
PROMPT_FILE = f"{ROOT}/prompts/prompts_dev.txt"
N_PROMPTS, SEEDS = 20, [0, 1, 2]

prompts = open(PROMPT_FILE).read().splitlines()[:N_PROMPTS]

print("Loading IR model...")
ir_model = RM.load("ImageReward-v1.0")

print("Loading CLIP model...")
clip_model, _, clip_prep = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai", device=device)
clip_tok = open_clip.get_tokenizer("ViT-L-14")
clip_model.eval()

print("Loading LPIPS model...")
lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()

def to_t(img, size=256):
    img = img.resize((size, size)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

ir_scores, clip_scores, lpips_scores = [], [], []

print("Computing IR + CLIP + LPIPS...")
for pi, prompt in enumerate(prompts):
    for si, seed in enumerate(SEEDS):
        fname  = f"p{pi:04d}_s{si:04d}.png"
        gen_p  = f"{SDIT_DIR}/{fname}"
        base_p = f"{BASE_DIR}/{fname}"
        if not os.path.exists(gen_p):
            continue
        gen_img = Image.open(gen_p).convert("RGB")

        ir_scores.append(ir_model.score(prompt, gen_img))

        with torch.no_grad():
            tokens = clip_tok([prompt]).to(device)
            img_t  = clip_prep(gen_img).unsqueeze(0).to(device)
            ig = clip_model.encode_image(img_t); ig /= ig.norm(dim=-1, keepdim=True)
            tg = clip_model.encode_text(tokens); tg /= tg.norm(dim=-1, keepdim=True)
            clip_scores.append((ig * tg).sum().item())

        if os.path.exists(base_p):
            g = to_t(gen_img).to(device) * 2 - 1
            b = to_t(Image.open(base_p)).to(device) * 2 - 1
            with torch.no_grad():
                lpips_scores.append(lpips_fn(g, b).item())

print("Computing FID...")
fid_val = cleanfid.compute_fid(BASE_DIR, SDIT_DIR, mode="clean", num_workers=0)

timing = json.load(open(f"{ROOT}/outputs/sdit_eval/timing.json"))
results = {
    "method":     "sdit_repro",
    "skip_run":   timing["skip_run"],
    "ratio":      0.5,
    "avg_time_s": timing["avg_time_s"],
    "speedup":    timing["speedup"],
    "IR":    round(float(np.mean(ir_scores)), 3),
    "CLIP":  round(float(np.mean(clip_scores)), 3),
    "LPIPS": round(float(np.mean(lpips_scores)), 3),
    "FID":   round(fid_val, 1),
    "n":     len(ir_scores),
}
print("\n=== SDIT Results ===")
for k, v in results.items():
    print(f"  {k}: {v}")

json.dump(results, open(f"{ROOT}/outputs/sdit_eval/summary.json", "w"), indent=2)
print(f"\nSaved to {ROOT}/outputs/sdit_eval/summary.json")
