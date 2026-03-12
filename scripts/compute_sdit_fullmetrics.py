"""
Compute full metrics for SDiT repro images:
  CLIP, PickScore, IR, HPSv2, Aesthetic, LPIPS, Edge, FID
Images: outputs/sdit_eval/images/  (20 prompts × seeds=[0,1,2] = 60 imgs)
Baseline: outputs/semantic_full_eval/baseline/images/
Saves: outputs/sdit_eval/summary.json
"""
import gc, json, os, sys
import numpy as np
import torch
from PIL import Image

ROOT = "/home/runkai/xuanhua/SASD"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.eval.metrics import compute_clip_score, compute_edge_density

SDIT_DIR     = f"{ROOT}/outputs/sdit_eval/images"
BASELINE_DIR = f"{ROOT}/outputs/semantic_full_eval/baseline/images"
PROMPT_FILE  = f"{ROOT}/prompts/prompts_dev.txt"
OUT_JSON     = f"{ROOT}/outputs/sdit_eval/summary.json"

PROMPTS = open(PROMPT_FILE).read().splitlines()[:20]
SEEDS   = [0, 1, 2]

# ── load images ───────────────────────────────────────────────────────────────
sdit_imgs, base_imgs, plist = [], [], []
for pi, prompt in enumerate(PROMPTS):
    for si in range(len(SEEDS)):
        fname = f"p{pi:04d}_s{si:04d}.png"
        sdit_imgs.append(Image.open(f"{SDIT_DIR}/{fname}").convert("RGB"))
        base_imgs.append(Image.open(f"{BASELINE_DIR}/{fname}").convert("RGB"))
        plist.append(prompt)

print(f"Loaded {len(sdit_imgs)} SDiT images, {len(base_imgs)} baseline images")

m = {}

# ── CLIP ──────────────────────────────────────────────────────────────────────
print("CLIP...", end="", flush=True)
m["clip"] = compute_clip_score(sdit_imgs, plist)
print(f" {np.mean(m['clip']):.4f}")

# ── Edge ──────────────────────────────────────────────────────────────────────
print("Edge...", end="", flush=True)
m["edge"] = [compute_edge_density(img) for img in sdit_imgs]
print(f" {np.mean(m['edge']):.4f}")

# ── PickScore ─────────────────────────────────────────────────────────────────
print("PickScore...", end="", flush=True)
try:
    from transformers import AutoProcessor, AutoModel
    device = "cuda"
    proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    scores = []
    with torch.no_grad():
        for img, prompt in zip(sdit_imgs, plist):
            inp = proc(text=[prompt], images=[img], return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(device)
            out = pick_model(**inp)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
            scores.append((ie @ te.T).item())
    m["pick"] = scores
    del pick_model; gc.collect(); torch.cuda.empty_cache()
    print(f" {np.mean(m['pick']):.4f}")
except Exception as e:
    print(f" SKIP ({e})")

# ── ImageReward ───────────────────────────────────────────────────────────────
print("ImageReward...", end="", flush=True)
try:
    import ImageReward as RM
    ir_model = RM.load("ImageReward-v1.0", device="cuda")
    with torch.no_grad():
        m["ir"] = [float(ir_model.score(p, img)) for img, p in zip(sdit_imgs, plist)]
    del ir_model; gc.collect(); torch.cuda.empty_cache()
    print(f" {np.mean(m['ir']):.4f}")
except Exception as e:
    print(f" SKIP ({e})")

# ── HPSv2 ─────────────────────────────────────────────────────────────────────
print("HPSv2...", end="", flush=True)
try:
    import hpsv2, tempfile
    scores = []
    with tempfile.TemporaryDirectory() as td:
        for i, (img, prompt) in enumerate(zip(sdit_imgs, plist)):
            tmp = os.path.join(td, f"{i}.png")
            img.save(tmp)
            r = hpsv2.score(tmp, prompt, hps_version="v2.1")
            scores.append(float(r[0]) if isinstance(r, (list, np.ndarray)) else float(r))
    m["hps"] = scores
    gc.collect(); torch.cuda.empty_cache()
    print(f" {np.mean(m['hps']):.4f}")
except Exception as e:
    print(f" SKIP ({e})")

# ── Aesthetic ─────────────────────────────────────────────────────────────────
print("Aesthetic...", end="", flush=True)
try:
    import open_clip
    import torch.nn as nn
    device = "cuda"
    aes_clip, _, aes_prep = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    aes_clip.eval()
    wp = f"{ROOT}/.cache/aesthetic_mlp_l14.pth"
    mlp = nn.Sequential(
        nn.Linear(768, 1024), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64),  nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    ).to(device)
    state = torch.load(wp, map_location=device, weights_only=True)
    mlp.load_state_dict({k.replace("layers.", ""): v for k, v in state.items()})
    mlp.eval()
    scores = []
    with torch.no_grad():
        for img in sdit_imgs:
            t = aes_prep(img).unsqueeze(0).to(device)
            f = aes_clip.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            scores.append(mlp(f.float()).item())
    m["aesthetic"] = scores
    del aes_clip, mlp; gc.collect(); torch.cuda.empty_cache()
    print(f" {np.mean(m['aesthetic']):.4f}")
except Exception as e:
    print(f" SKIP ({e})")

# ── LPIPS ─────────────────────────────────────────────────────────────────────
print("LPIPS...", end="", flush=True)
try:
    import lpips
    import torchvision.transforms as T
    device = "cuda"
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    tf = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    with torch.no_grad():
        m["lpips"] = [lpips_fn(tf(a).unsqueeze(0).to(device),
                               tf(b).unsqueeze(0).to(device)).item()
                      for a, b in zip(sdit_imgs, base_imgs)]
    del lpips_fn; gc.collect(); torch.cuda.empty_cache()
    print(f" {np.mean(m['lpips']):.4f}")
except Exception as e:
    print(f" SKIP ({e})")

# ── FID ───────────────────────────────────────────────────────────────────────
print("FID...", end="", flush=True)
fid_val = None
try:
    from cleanfid import fid as cleanfid
    fid_val = cleanfid.compute_fid(BASELINE_DIR, SDIT_DIR, mode="clean", num_workers=0)
    print(f" {fid_val:.2f}")
except Exception as e:
    print(f" SKIP ({e})")

# ── save ─────────────────────────────────────────────────────────────────────
existing = {}
if os.path.exists(OUT_JSON):
    existing = json.load(open(OUT_JSON))

existing.update({
    "mean_clip":      float(np.mean(m["clip"]))       if "clip"      in m else existing.get("mean_clip"),
    "mean_pick":      float(np.mean(m["pick"]))       if "pick"      in m else existing.get("mean_pick"),
    "mean_ir":        float(np.mean(m["ir"]))         if "ir"        in m else existing.get("mean_ir"),
    "mean_hps":       float(np.mean(m["hps"]))        if "hps"       in m else existing.get("mean_hps"),
    "mean_aesthetic": float(np.mean(m["aesthetic"]))  if "aesthetic" in m else existing.get("mean_aesthetic"),
    "mean_lpips":     float(np.mean(m["lpips"]))      if "lpips"     in m else existing.get("mean_lpips"),
    "mean_edge":      float(np.mean(m["edge"]))       if "edge"      in m else existing.get("mean_edge"),
    "fid":            fid_val                         if fid_val is not None else existing.get("fid"),
})

json.dump(existing, open(OUT_JSON, "w"), indent=2)
print(f"\nSaved → {OUT_JSON}")

print("\n=== SDiT Full Metrics ===")
for k in ["mean_clip","mean_pick","mean_ir","mean_hps","mean_aesthetic","mean_lpips","mean_edge","fid"]:
    v = existing.get(k)
    print(f"  {k:20s}: {v:.4f}" if v is not None else f"  {k:20s}: N/A")
