"""
TaylorSeer-style acceleration experiment on FLUX.1-dev.

Compares on all 24 prompts from scores_fskip2.json:
  - baseline
  - fskip2  (linear, run=1 equivalent, generate_accelerated with interval=2, warmup=5)
  - taylor_r1_o2  (run=1, order=2)
  - taylor_r2_o2  (run=2, order=2)  — primary target
  - taylor_r4_o2  (run=4, order=2)
  - taylor_r5_o2  (run=5, order=2)  — aggressive

Results saved to outputs/accelae_topk_flux/scores_taylor.json
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.flux_wrapper import FluxDiTWrapper

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_DIR = "outputs/accelae_topk_flux"
os.makedirs(OUT_DIR, exist_ok=True)
SCORES_OUT = os.path.join(OUT_DIR, "scores_taylor.json")

# ── Prompts (24 prompts from fskip2 run) ─────────────────────────────────────
with open(os.path.join(OUT_DIR, "scores_fskip2.json")) as f:
    fskip2_scores = json.load(f)

PROMPTS = {tag: v["prompt"] for tag, v in fskip2_scores.items()}

SEED = 42
STEPS = 28
WARMUP = 5

# ── Configs to evaluate ───────────────────────────────────────────────────────
CONFIGS = [
    dict(name="baseline",    method="baseline"),
    dict(name="fskip2",      method="fskip2",  full_skip_interval=2, warmup_steps=WARMUP),
    dict(name="taylor_r1_o2", method="taylor", taylor_run=1, warmup_steps=WARMUP, taylor_order=2),
    dict(name="taylor_r2_o2", method="taylor", taylor_run=2, warmup_steps=WARMUP, taylor_order=2),
    dict(name="taylor_r4_o2", method="taylor", taylor_run=4, warmup_steps=WARMUP, taylor_order=2),
    dict(name="taylor_r5_o2", method="taylor", taylor_run=5, warmup_steps=WARMUP, taylor_order=2),
]


def generate_one(wrapper: FluxDiTWrapper, prompt: str, cfg: dict) -> tuple:
    """Return (image, latency_sec)."""
    method = cfg["method"]
    t0 = time.perf_counter()

    if method == "baseline":
        img = wrapper.generate(prompt, seed=SEED, steps=STEPS)
    elif method == "fskip2":
        img = wrapper.generate_accelerated(
            prompt, seed=SEED, steps=STEPS,
            full_skip_interval=cfg["full_skip_interval"],
            warmup_steps=cfg["warmup_steps"],
        )
    elif method == "taylor":
        img = wrapper.generate_taylor(
            prompt, seed=SEED, steps=STEPS,
            taylor_run=cfg["taylor_run"],
            warmup_steps=cfg["warmup_steps"],
            taylor_order=cfg["taylor_order"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    latency = time.perf_counter() - t0
    return img, latency


def main():
    print("Loading FLUX...")
    wrapper = FluxDiTWrapper(dtype="bf16")

    # ── Pre-encode all prompts ────────────────────────────────────────────────
    print("Pre-encoding prompts...")
    for tag, prompt in PROMPTS.items():
        wrapper._encode_prompt(prompt)
    print(f"  {len(PROMPTS)} prompts encoded and cached.\n")

    # ── Run all configs ───────────────────────────────────────────────────────
    import ImageReward as RM
    reward_model = RM.load("ImageReward-v1.0")
    reward_model.eval()

    results = {}   # tag → {config_name → {ir, latency}}

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n=== Config: {name} ===")
        config_latencies = []

        for tag, prompt in PROMPTS.items():
            img, latency = generate_one(wrapper, prompt, cfg)
            config_latencies.append(latency)

            # Score with ImageReward
            with torch.no_grad():
                score = reward_model.score(prompt, [img])
                ir = score[0] if isinstance(score, list) else float(score)

            if tag not in results:
                results[tag] = {}
            results[tag][name] = {"ir": ir, "latency": latency}

            print(f"  [{tag}] IR={ir:.3f}  t={latency:.2f}s")

        avg_lat = sum(config_latencies) / len(config_latencies)
        print(f"  → avg latency: {avg_lat:.2f}s")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Config':<18}  {'Avg IR':>8}  {'Avg lat':>9}  {'Speedup':>8}  {'Pos ΔIR':>8}")
    print("-" * 80)

    base_lat = sum(
        results[tag]["baseline"]["latency"] for tag in PROMPTS
    ) / len(PROMPTS)

    for cfg in CONFIGS:
        name = cfg["name"]
        irs = [results[tag][name]["ir"] for tag in PROMPTS]
        lats = [results[tag][name]["latency"] for tag in PROMPTS]
        base_irs = [results[tag]["baseline"]["ir"] for tag in PROMPTS]

        avg_ir = sum(irs) / len(irs)
        avg_lat = sum(lats) / len(lats)
        speedup = base_lat / avg_lat
        n_positive = sum(1 for i, b in zip(irs, base_irs) if i > b)
        print(
            f"{name:<18}  {avg_ir:>8.3f}  {avg_lat:>8.2f}s  {speedup:>7.2f}×  "
            f"{n_positive:>3}/{len(PROMPTS)}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(SCORES_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved → {SCORES_OUT}")


if __name__ == "__main__":
    main()
