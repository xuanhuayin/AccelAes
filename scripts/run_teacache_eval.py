"""
TeaCache baseline evaluation — Lumina + FLUX.

Lumina (30 steps, 20 prompts × 2 seeds):
  Configs: baseline, teacache_t006, teacache_t010, teacache_t015, teacache_t020

FLUX (28 steps, 24 prompts, seed=42):
  Configs: baseline, teacache_t010, teacache_t015, teacache_t020, teacache_t030

Results:
  outputs/teacache_lumina/scores.json
  outputs/teacache_flux/scores.json
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_image_reward():
    import ImageReward as RM
    rm = RM.load("ImageReward-v1.0")
    rm.eval()
    return rm


def _ir_score(rm, prompt, img):
    with torch.no_grad():
        s = rm.score(prompt, [img])
    return s[0] if isinstance(s, list) else float(s)


# ── Lumina evaluation ──────────────────────────────────────────────────────────

def run_lumina(output_dir: str):
    from src.models.dit_wrapper import LuminaDiTWrapper
    from src.baselines.teacache import generate_teacache_lumina

    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, "scores.json")

    with open("prompts/prompts_dev.txt") as f:
        prompts = [l.strip() for l in f if l.strip()][:20]
    seeds = [42, 1337]

    CONFIGS = [
        dict(name="baseline",        method="baseline"),
        dict(name="teacache_t006",   method="teacache", threshold=0.06),
        dict(name="teacache_t010",   method="teacache", threshold=0.10),
        dict(name="teacache_t015",   method="teacache", threshold=0.15),
        dict(name="teacache_t020",   method="teacache", threshold=0.20),
    ]

    print("Loading Lumina-Next-T2I...")
    wrapper = LuminaDiTWrapper()
    rm = _load_image_reward()

    results = {}  # prompt_key → {config → {ir_mean, ir_list, lat_mean}}

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n=== Config: {name} ===")
        latencies = []

        for pi, prompt in enumerate(prompts):
            ir_list, lat_list = [], []
            for seed in seeds:
                t0 = time.perf_counter()
                if cfg["method"] == "baseline":
                    img = wrapper.generate(prompt, seed=seed, steps=30)
                else:
                    img = generate_teacache_lumina(
                        wrapper, prompt, seed=seed,
                        threshold=cfg["threshold"], steps=30)
                lat = time.perf_counter() - t0

                ir = _ir_score(rm, prompt, img)
                ir_list.append(ir)
                lat_list.append(lat)
                latencies.append(lat)

            key = f"{pi:02d}_{prompt[:40]}"
            if key not in results:
                results[key] = {"prompt": prompt}
            results[key][name] = {
                "ir_mean": sum(ir_list) / len(ir_list),
                "ir_list": ir_list,
                "lat_mean": sum(lat_list) / len(lat_list),
            }
            avg_ir = results[key][name]["ir_mean"]
            avg_lat = results[key][name]["lat_mean"]
            print(f"  [{pi:02d}] IR={avg_ir:.3f}  t={avg_lat:.2f}s  {prompt[:40]}")

        avg_lat_all = sum(latencies) / len(latencies)
        print(f"  → avg latency: {avg_lat_all:.2f}s")

    # Summary
    print("\n" + "=" * 72)
    print(f"{'Config':<20}  {'Avg IR':>8}  {'Avg lat':>9}  {'Speedup':>8}  {'vs baseline IR':>14}")
    print("-" * 72)

    ref_ir = sum(
        results[f"{pi:02d}_{p[:40]}"]["baseline"]["ir_mean"]
        for pi, p in enumerate(prompts)
    ) / len(prompts)
    ref_lat = sum(
        results[f"{pi:02d}_{p[:40]}"]["baseline"]["lat_mean"]
        for pi, p in enumerate(prompts)
    ) / len(prompts)

    for cfg in CONFIGS:
        name = cfg["name"]
        irs = [results[f"{pi:02d}_{p[:40]}"][name]["ir_mean"] for pi, p in enumerate(prompts)]
        lats = [results[f"{pi:02d}_{p[:40]}"][name]["lat_mean"] for pi, p in enumerate(prompts)]
        avg_ir = sum(irs) / len(irs)
        avg_lat = sum(lats) / len(lats)
        speedup = ref_lat / avg_lat
        delta_ir = (avg_ir - ref_ir) / abs(ref_ir) * 100 if ref_ir != 0 else 0
        print(f"{name:<20}  {avg_ir:>8.3f}  {avg_lat:>8.2f}s  {speedup:>7.2f}×  {delta_ir:>+13.1f}%")

    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved → {scores_path}")


# ── FLUX evaluation ────────────────────────────────────────────────────────────

def run_flux(output_dir: str):
    from src.models.flux_wrapper import FluxDiTWrapper

    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, "scores.json")

    # Use the same 24 prompts as the main FLUX eval
    ref_path = "outputs/accelae_topk_flux/scores_fskip2.json"
    with open(ref_path) as f:
        ref = json.load(f)
    PROMPTS = {tag: v["prompt"] for tag, v in ref.items()}

    SEED = 42
    STEPS = 28
    WARMUP = 3

    CONFIGS = [
        dict(name="baseline",       method="baseline"),
        dict(name="teacache_t010",  method="teacache", threshold=0.10),
        dict(name="teacache_t015",  method="teacache", threshold=0.15),
        dict(name="teacache_t020",  method="teacache", threshold=0.20),
        dict(name="teacache_t030",  method="teacache", threshold=0.30),
    ]

    print("Loading FLUX...")
    wrapper = FluxDiTWrapper(dtype="bf16")

    print("Pre-encoding prompts...")
    for prompt in PROMPTS.values():
        wrapper._encode_prompt(prompt)
    print(f"  {len(PROMPTS)} prompts cached.\n")

    rm = _load_image_reward()

    results = {}  # tag → {config → {ir, latency}}

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n=== Config: {name} ===")
        config_latencies = []

        for tag, prompt in PROMPTS.items():
            t0 = time.perf_counter()
            if cfg["method"] == "baseline":
                img = wrapper.generate(prompt, seed=SEED, steps=STEPS)
            else:
                img = wrapper.generate_teacache(
                    prompt, seed=SEED, steps=STEPS,
                    threshold=cfg["threshold"],
                    warmup_steps=WARMUP)
            lat = time.perf_counter() - t0

            ir = _ir_score(rm, prompt, img)
            config_latencies.append(lat)

            if tag not in results:
                results[tag] = {}
            results[tag][name] = {"ir": ir, "latency": lat}
            print(f"  [{tag}] IR={ir:.3f}  t={lat:.2f}s")

        avg_lat = sum(config_latencies) / len(config_latencies)
        print(f"  → avg latency: {avg_lat:.2f}s")

    # Summary
    print("\n" + "=" * 72)
    print(f"{'Config':<18}  {'Avg IR':>8}  {'Avg lat':>9}  {'Speedup':>8}  {'Pos ΔIR':>8}")
    print("-" * 72)

    base_lat = sum(results[tag]["baseline"]["latency"] for tag in PROMPTS) / len(PROMPTS)
    base_irs = [results[tag]["baseline"]["ir"] for tag in PROMPTS]

    for cfg in CONFIGS:
        name = cfg["name"]
        irs = [results[tag][name]["ir"] for tag in PROMPTS]
        lats = [results[tag][name]["latency"] for tag in PROMPTS]
        avg_ir = sum(irs) / len(irs)
        avg_lat = sum(lats) / len(lats)
        speedup = base_lat / avg_lat
        n_pos = sum(1 for i, b in zip(irs, base_irs) if i > b)
        print(f"{name:<18}  {avg_ir:>8.3f}  {avg_lat:>8.2f}s  {speedup:>7.2f}×  {n_pos:>3}/{len(PROMPTS)}")

    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved → {scores_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lumina", "flux", "all"], default="all",
                        help="Which model to evaluate (default: all)")
    args = parser.parse_args()

    if args.model in ("lumina", "all"):
        print("\n" + "#" * 60)
        print("# TeaCache on Lumina")
        print("#" * 60)
        run_lumina("outputs/teacache_lumina")

    if args.model in ("flux", "all"):
        print("\n" + "#" * 60)
        print("# TeaCache on FLUX")
        print("#" * 60)
        run_flux("outputs/teacache_flux")


if __name__ == "__main__":
    main()
