import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.baselines.sdit import generate_sdit_lumina

OUT_DIR     = "/home/runkai/xuanhua/SASD/outputs/sdit_eval"
PROMPT_FILE = "/home/runkai/xuanhua/SASD/prompts/prompts_dev.txt"
SEEDS       = [0, 1, 2]
N_PROMPTS   = 20

os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
prompts = open(PROMPT_FILE).read().splitlines()[:N_PROMPTS]

print("Loading model...")
wrapper = LuminaDiTWrapper()
times, done, skipped = [], 0, 0
total = N_PROMPTS * len(SEEDS)

print(f"\n=== SDIT (skip_run=1, ratio=0.5) ===")
for pi, prompt in enumerate(prompts):
    for si, seed in enumerate(SEEDS):
        fname = f"p{pi:04d}_s{si:04d}.png"
        out   = f"{OUT_DIR}/images/{fname}"
        if os.path.exists(out):
            skipped += 1; continue
        t0  = time.time()
        img = generate_sdit_lumina(wrapper, prompt=prompt, seed=seed, cfg_scale=4.0, skip_run=1, ratio=0.5)
        elapsed = time.time() - t0
        img.save(out)
        times.append(elapsed)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{total}  avg={sum(times)/len(times):.2f}s", flush=True)

print(f"  Done: {done} generated, {skipped} skipped")
avg_t = sum(times)/len(times) if times else None
if avg_t:
    print(f"  Avg time: {avg_t:.2f}s  speedup: {12.37/avg_t:.2f}x")
    json.dump({"avg_time_s": avg_t, "speedup": round(12.37/avg_t,2), "skip_run": 1},
              open(f"{OUT_DIR}/timing.json", "w"), indent=2)
