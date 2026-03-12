"""
Taylor acceleration experiment on Lumina-Next-T2I (SASD full pipeline).

Compares 20 prompts × 2 seeds = 40 images per config:
  - sasd_fskip2   (current: full_skip_interval=2, reference)
  - sasd_taylor_r2 (taylor_run=2, expected ~2.50×)
  - sasd_taylor_r3 (taylor_run=3, expected ~2.78×)
  - sasd_taylor_r4 (taylor_run=4, expected ~2.97×)

Metrics: ImageReward + latency.
Results saved to outputs/taylor_lumina/scores_taylor.json
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dit_wrapper import LuminaDiTWrapper

OUT_DIR = "outputs/taylor_lumina"
os.makedirs(OUT_DIR, exist_ok=True)
SCORES_OUT = os.path.join(OUT_DIR, "scores_taylor.json")

# ── Prompts ───────────────────────────────────────────────────────────────────
with open("prompts/prompts_dev.txt") as f:
    ALL_PROMPTS = [l.strip() for l in f if l.strip()]
PROMPTS = ALL_PROMPTS[:20]
SEEDS = [42, 1337]

# ── Shared SASD kwargs (same as SASD_FULL_KWARGS in run_p0_eval.py) ──────────
BASE_KWARGS = dict(
    mask_type="semantic",
    region_method="threshold",
    skip_ratio=0.50,
    s_fg=7.0, s_bg=1.0,
    mask_step=5,
    sparse_ffn=True,
    sparse_blocks=True,
    steps=30,
)

CONFIGS = [
    dict(name="sasd_fskip2",    full_skip_interval=2, taylor_run=0),
    dict(name="sasd_taylor_r2", full_skip_interval=0, taylor_run=2),
    dict(name="sasd_taylor_r3", full_skip_interval=0, taylor_run=3),
    dict(name="sasd_taylor_r4", full_skip_interval=0, taylor_run=4),
]


def main():
    print("Loading Lumina-Next-T2I...")
    wrapper = LuminaDiTWrapper()

    import ImageReward as RM
    reward_model = RM.load("ImageReward-v1.0")
    reward_model.eval()

    results = {}   # prompt → {config_name → {ir_list, latency_list}}

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n=== Config: {name} ===")
        latencies = []

        for pi, prompt in enumerate(PROMPTS):
            ir_scores = []
            lat_scores = []

            for seed in SEEDS:
                t0 = time.perf_counter()
                img = wrapper.generate_accelerated_dual(
                    prompt=prompt,
                    seed=seed,
                    full_skip_interval=cfg["full_skip_interval"],
                    taylor_run=cfg["taylor_run"],
                    **BASE_KWARGS,
                )
                lat = time.perf_counter() - t0

                with torch.no_grad():
                    ir = reward_model.score(prompt, [img])
                    ir = ir[0] if isinstance(ir, list) else float(ir)

                ir_scores.append(ir)
                lat_scores.append(lat)
                latencies.append(lat)

            key = f"{pi:02d}_{prompt[:40]}"
            if key not in results:
                results[key] = {"prompt": prompt}
            results[key][name] = {
                "ir_mean": sum(ir_scores) / len(ir_scores),
                "ir_list": ir_scores,
                "lat_mean": sum(lat_scores) / len(lat_scores),
            }
            avg_ir = sum(ir_scores) / len(ir_scores)
            print(f"  [{pi:02d}] IR={avg_ir:.3f}  t={lat_scores[0]:.2f}s  prompt={prompt[:40]}")

        avg_lat = sum(latencies) / len(latencies)
        avg_ir_all = sum(
            results[f"{pi:02d}_{p[:40]}"][name]["ir_mean"]
            for pi, p in enumerate(PROMPTS)
        ) / len(PROMPTS)
        print(f"  → avg latency: {avg_lat:.2f}s  avg IR: {avg_ir_all:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Config':<20}  {'Avg IR':>8}  {'Avg lat':>9}  {'vs fskip2 IR':>13}")
    print("-" * 72)

    ref_ir = sum(
        results[f"{pi:02d}_{p[:40]}"]["sasd_fskip2"]["ir_mean"]
        for pi, p in enumerate(PROMPTS)
    ) / len(PROMPTS)
    ref_lat = sum(
        results[f"{pi:02d}_{p[:40]}"]["sasd_fskip2"]["lat_mean"]
        for pi, p in enumerate(PROMPTS)
    ) / len(PROMPTS)

    for cfg in CONFIGS:
        name = cfg["name"]
        irs = [results[f"{pi:02d}_{p[:40]}"][name]["ir_mean"] for pi, p in enumerate(PROMPTS)]
        lats = [results[f"{pi:02d}_{p[:40]}"][name]["lat_mean"] for pi, p in enumerate(PROMPTS)]
        avg_ir = sum(irs) / len(irs)
        avg_lat = sum(lats) / len(lats)
        delta = (avg_ir - ref_ir) / abs(ref_ir) * 100 if ref_ir != 0 else 0
        print(f"{name:<20}  {avg_ir:>8.3f}  {avg_lat:>8.2f}s  {delta:>+12.1f}%")

    print(f"\nNote: Baseline Lumina is ~12.37s. Real speedups (vs 12.37s):")
    for cfg in CONFIGS:
        name = cfg["name"]
        lats = [results[f"{pi:02d}_{p[:40]}"][name]["lat_mean"] for pi, p in enumerate(PROMPTS)]
        avg_lat = sum(lats) / len(lats)
        speedup = 12.37 / avg_lat
        print(f"  {name:<20}  {avg_lat:.2f}s  {speedup:.2f}×")

    with open(SCORES_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved → {SCORES_OUT}")


if __name__ == "__main__":
    main()
