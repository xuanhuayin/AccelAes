#!/usr/bin/env python3
"""
Phase 3 交付物: 失败案例集 (artifact list)
对比 baseline 与 r=0.5 skip-update，找出质量下降最明显的样本。
"""

import csv
import os
import json

def load_metrics(exp_dir):
    """Load metrics.csv as dict keyed by (prompt_idx, seed)"""
    csv_path = os.path.join(exp_dir, "metrics.csv")
    records = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["prompt_idx"]), int(row["seed"]))
            records[key] = {
                "filename": row["filename"],
                "clip_score": float(row["clip_score"]),
                "edge_density": float(row["edge_density"]),
            }
    return records

def load_prompts(path, max_prompts=50):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
                if len(prompts) >= max_prompts:
                    break
    return prompts

base_dir = "outputs/p3b_baseline_full"
skip_dir = "outputs/p3b_full_r05_linear_norefresh"

base_metrics = load_metrics(base_dir)
skip_metrics = load_metrics(skip_dir)
prompts = load_prompts("prompts/prompts_dev.txt", max_prompts=50)

# Compute per-sample deltas
deltas = []
for key in base_metrics:
    if key not in skip_metrics:
        continue
    b = base_metrics[key]
    s = skip_metrics[key]
    clip_delta = s["clip_score"] - b["clip_score"]
    edge_delta = s["edge_density"] - b["edge_density"]
    clip_delta_pct = clip_delta / b["clip_score"] * 100 if b["clip_score"] > 0 else 0
    edge_delta_pct = edge_delta / b["edge_density"] * 100 if b["edge_density"] > 0 else 0
    deltas.append({
        "prompt_idx": key[0],
        "seed": key[1],
        "prompt": prompts[key[0]] if key[0] < len(prompts) else "?",
        "filename": b["filename"],
        "base_clip": b["clip_score"],
        "skip_clip": s["clip_score"],
        "clip_delta_pct": clip_delta_pct,
        "base_edge": b["edge_density"],
        "skip_edge": s["edge_density"],
        "edge_delta_pct": edge_delta_pct,
    })

# Sort by worst CLIP degradation
deltas.sort(key=lambda x: x["clip_delta_pct"])

print("=" * 80)
print("Phase 3 Artifact List: r=0.5 linear norefresh vs baseline")
print("=" * 80)

print("\n### Top 10 Worst CLIP Score Degradation ###\n")
print(f"{'Rank':<5} {'File':<20} {'Base CLIP':<12} {'Skip CLIP':<12} {'Δ%':<8} {'Prompt (truncated)'}")
print("-" * 80)
for i, d in enumerate(deltas[:10]):
    prompt_short = d["prompt"][:50] + "..." if len(d["prompt"]) > 50 else d["prompt"]
    print(f"{i+1:<5} {d['filename']:<20} {d['base_clip']:<12.4f} {d['skip_clip']:<12.4f} {d['clip_delta_pct']:<8.2f} {prompt_short}")

# Sort by worst edge density degradation
deltas.sort(key=lambda x: x["edge_delta_pct"])

print("\n### Top 10 Worst Edge Density Degradation ###\n")
print(f"{'Rank':<5} {'File':<20} {'Base Edge':<12} {'Skip Edge':<12} {'Δ%':<8} {'Prompt (truncated)'}")
print("-" * 80)
for i, d in enumerate(deltas[:10]):
    prompt_short = d["prompt"][:50] + "..." if len(d["prompt"]) > 50 else d["prompt"]
    print(f"{i+1:<5} {d['filename']:<20} {d['base_edge']:<12.4f} {d['skip_edge']:<12.4f} {d['edge_delta_pct']:<8.2f} {prompt_short}")

# Summary statistics
clip_deltas = [d["clip_delta_pct"] for d in deltas]
edge_deltas = [d["edge_delta_pct"] for d in deltas]

print("\n### Summary Statistics ###\n")
print(f"Total samples: {len(deltas)}")
print(f"CLIP Δ%:  mean={sum(clip_deltas)/len(clip_deltas):.2f}%, "
      f"min={min(clip_deltas):.2f}%, max={max(clip_deltas):.2f}%, "
      f"std={((sum((x - sum(clip_deltas)/len(clip_deltas))**2 for x in clip_deltas) / len(clip_deltas))**0.5):.2f}%")
print(f"Edge Δ%:  mean={sum(edge_deltas)/len(edge_deltas):.2f}%, "
      f"min={min(edge_deltas):.2f}%, max={max(edge_deltas):.2f}%, "
      f"std={((sum((x - sum(edge_deltas)/len(edge_deltas))**2 for x in edge_deltas) / len(edge_deltas))**0.5):.2f}%")

# Count how many samples have >2% degradation
clip_bad = sum(1 for d in clip_deltas if d < -2)
edge_bad = sum(1 for d in edge_deltas if d < -2)
print(f"\nSamples with >2% CLIP degradation: {clip_bad}/{len(deltas)} ({clip_bad/len(deltas)*100:.1f}%)")
print(f"Samples with >2% Edge degradation: {edge_bad}/{len(deltas)} ({edge_bad/len(deltas)*100:.1f}%)")

# Save artifact list as JSON
artifact_list = {
    "experiment": "p3b_full_r05_linear_norefresh vs p3b_baseline_full",
    "config": {"skip_ratio": 0.5, "extrapolation": "linear", "refresh": 0},
    "total_samples": len(deltas),
    "summary": {
        "clip_delta_pct": {"mean": sum(clip_deltas)/len(clip_deltas), "min": min(clip_deltas), "max": max(clip_deltas)},
        "edge_delta_pct": {"mean": sum(edge_deltas)/len(edge_deltas), "min": min(edge_deltas), "max": max(edge_deltas)},
        "clip_bad_2pct": clip_bad,
        "edge_bad_2pct": edge_bad,
    },
    "worst_clip_cases": deltas[:10] if deltas[0]["clip_delta_pct"] == min(clip_deltas) else sorted(deltas, key=lambda x: x["clip_delta_pct"])[:10],
    "worst_edge_cases": sorted(deltas, key=lambda x: x["edge_delta_pct"])[:10],
}

out_path = "outputs/phase3_artifact_list.json"
with open(out_path, "w") as f:
    json.dump(artifact_list, f, indent=2, ensure_ascii=False)
print(f"\nSaved artifact list to {out_path}")
