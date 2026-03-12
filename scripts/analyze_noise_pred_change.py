#!/usr/bin/env python3
"""
Noise prediction change analysis: fg vs bg regions.

For each denoising step, computes |noise_pred_t - noise_pred_{t-1}|
separately for foreground and background regions.

Expected result: bg change amplitude is 3-5x smaller than fg,
justifying the background caching strategy.

Outputs:
  - Per-step change magnitudes (JSON + CSV)
  - Summary statistics
  - Matplotlib plot (if available)

Usage:
    python scripts/analyze_noise_pred_change.py
    python scripts/analyze_noise_pred_change.py --num_prompts 10
"""

import os
import sys
import json
import argparse
import math

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dit_wrapper import LuminaDiTWrapper
from src.utils.io_utils import load_config, load_prompts
from src.utils.seed import set_seed
from src.eval.speed import setup_torch_determinism
from src.sparse.sdit_mask_builder import CFGMagnitudeMaskBuilder
from src.sparse.boundary_ops import soft_mask_pipeline

# Suppress matplotlib backend warning
import warnings
warnings.filterwarnings("ignore")


def analyze_single_prompt(wrapper, prompt, seed, steps, resolution, cfg_scale,
                          mask_step=5, skip_ratio=0.5):
    """
    Run denoising and track noise_pred changes per step in fg/bg regions.

    Returns:
        dict with per-step fg/bg change magnitudes.
    """
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
    import torch.nn.functional as F

    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    # Encode prompt
    (
        prompt_embeds, prompt_attention_mask,
        negative_prompt_embeds, negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt, do_classifier_free_guidance=True,
        negative_prompt="", num_images_per_prompt=1, device=device,
    )
    prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)

    cross_attention_kwargs = {}
    default_sample_size = getattr(pipe, 'default_sample_size', 128)
    cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

    scaling_factor = math.sqrt(resolution * resolution / wrapper.default_image_size ** 2)

    latent_h = resolution // wrapper.vae_scale_factor
    latent_w = resolution // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels
    shape = (1, latent_channels, latent_h, latent_w)
    head_dim = wrapper.transformer.head_dim

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    mask_builder = CFGMagnitudeMaskBuilder(
        n_segments=64, compactness=10.0,
        ratio=1.0 - skip_ratio, blur_sigma=1.5,
    )

    prev_noise_pred = None
    fg_mask = None
    results = {"steps": [], "fg_change": [], "bg_change": [], "total_change": []}

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], dtype=torch.float64, device=device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        current_timestep = 1 - current_timestep / pipe.scheduler.config.num_train_timesteps

        scaling_watershed = 1.0
        if current_timestep[0] < scaling_watershed:
            linear_factor = scaling_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scaling_factor
        image_rotary_emb = get_2d_rotary_pos_embed_lumina(
            head_dim, 384, 384,
            linear_factor=linear_factor, ntk_factor=ntk_factor,
        )

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_mask=prompt_attention_mask,
            image_rotary_emb=image_rotary_emb,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        noise_pred = noise_pred.chunk(2, dim=1)[0]

        # CFG
        noise_pred_eps = noise_pred[:, :3]
        noise_pred_rest = noise_pred[:, 3:]
        noise_pred_cond, noise_pred_uncond = torch.split(
            noise_pred_eps, len(noise_pred_eps) // 2, dim=0
        )

        # Build mask at mask_step
        if i == mask_step:
            spatial_mask = mask_builder.build_mask(noise_pred_cond, noise_pred_uncond)
            spatial_mask = spatial_mask.to(device=device, dtype=dtype)
            fg_mask = spatial_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        guided = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        noise_pred_eps = torch.cat([guided, guided], dim=0)
        noise_pred_full = torch.cat([noise_pred_eps, noise_pred_rest], dim=1)
        noise_pred_full, _ = noise_pred_full.chunk(2, dim=0)
        noise_pred_full = -noise_pred_full

        # Compute change from previous step
        if prev_noise_pred is not None and fg_mask is not None:
            change = (noise_pred_full - prev_noise_pred).abs()
            # change: (1, C, H, W)
            change_magnitude = change.mean(dim=1, keepdim=True)  # (1, 1, H, W)

            fg_change = (change_magnitude * fg_mask).sum() / (fg_mask.sum() + 1e-8)
            bg_mask_val = 1.0 - fg_mask
            bg_change = (change_magnitude * bg_mask_val).sum() / (bg_mask_val.sum() + 1e-8)
            total_change = change_magnitude.mean()

            results["steps"].append(i)
            results["fg_change"].append(float(fg_change))
            results["bg_change"].append(float(bg_change))
            results["total_change"].append(float(total_change))

        prev_noise_pred = noise_pred_full.clone()

        latents = pipe.scheduler.step(noise_pred_full, t, latents, return_dict=False)[0]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/noise_pred_analysis")
    args = parser.parse_args()

    config = load_config("configs/base.yaml")
    setup_torch_determinism()

    wrapper = LuminaDiTWrapper(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bf16"),
    )

    prompts = load_prompts(config["paths"]["prompts_dev"])[:args.num_prompts]
    resolution = config["generation"].get("resolution", 1024)
    steps = config["sampler"].get("steps", 30)
    cfg_scale = config["generation"].get("cfg_scale", 4.0)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output_dir,
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Noise Prediction Change Analysis ===")
    print(f"Prompts: {len(prompts)}, Seed: {args.seed}")
    print()

    all_results = []

    for pidx, prompt in enumerate(prompts):
        print(f"[{pidx+1}/{len(prompts)}] {prompt[:60]}...")

        with torch.no_grad():
            result = analyze_single_prompt(
                wrapper, prompt, args.seed, steps, resolution, cfg_scale,
            )

        result["prompt_idx"] = pidx
        result["prompt"] = prompt
        all_results.append(result)

        # Print per-prompt summary
        if result["fg_change"]:
            avg_fg = np.mean(result["fg_change"])
            avg_bg = np.mean(result["bg_change"])
            ratio = avg_fg / (avg_bg + 1e-8)
            print(f"  fg_change={avg_fg:.6f}, bg_change={avg_bg:.6f}, ratio={ratio:.2f}x")

    # Aggregate across prompts
    all_fg = []
    all_bg = []
    all_ratios = []

    for r in all_results:
        if r["fg_change"]:
            avg_fg = np.mean(r["fg_change"])
            avg_bg = np.mean(r["bg_change"])
            all_fg.append(avg_fg)
            all_bg.append(avg_bg)
            all_ratios.append(avg_fg / (avg_bg + 1e-8))

    print()
    print("=" * 60)
    print("Aggregate Results")
    print("=" * 60)
    print(f"  Mean fg change:   {np.mean(all_fg):.6f} ± {np.std(all_fg):.6f}")
    print(f"  Mean bg change:   {np.mean(all_bg):.6f} ± {np.std(all_bg):.6f}")
    print(f"  Mean fg/bg ratio: {np.mean(all_ratios):.2f}x ± {np.std(all_ratios):.2f}")
    print("=" * 60)

    # Save results
    results_path = os.path.join(output_dir, "noise_pred_change.json")
    with open(results_path, "w") as f:
        json.dump({
            "per_prompt": all_results,
            "aggregate": {
                "mean_fg_change": float(np.mean(all_fg)),
                "mean_bg_change": float(np.mean(all_bg)),
                "mean_ratio": float(np.mean(all_ratios)),
                "std_ratio": float(np.std(all_ratios)),
            },
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Per-step plot (averaged across prompts)
        max_steps = max(len(r["steps"]) for r in all_results)
        fg_by_step = [[] for _ in range(max_steps)]
        bg_by_step = [[] for _ in range(max_steps)]

        for r in all_results:
            for j, s in enumerate(range(len(r["steps"]))):
                fg_by_step[j].append(r["fg_change"][j])
                bg_by_step[j].append(r["bg_change"][j])

        step_indices = list(range(len(fg_by_step)))
        fg_means = [np.mean(x) if x else 0 for x in fg_by_step]
        bg_means = [np.mean(x) if x else 0 for x in bg_by_step]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(step_indices, fg_means, 'r-o', label='Foreground', markersize=3)
        ax.plot(step_indices, bg_means, 'b-o', label='Background', markersize=3)
        ax.set_xlabel("Denoising Step")
        ax.set_ylabel("|noise_pred_t - noise_pred_{t-1}|")
        ax.set_title("Noise Prediction Change: Foreground vs Background")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(output_dir, "noise_pred_change.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
