# AccelAes

**AccelAes** is a training-free DiT inference acceleration framework combining aesthetic-aware sparse attention (AesMask) with frame-level step skipping (StepSkip), achieving **2.1× speedup** with improved image quality on Lumina-Next-T2I, FLUX.1, and SD3.

## Method Overview

AccelAes consists of two complementary components:

- **AesMask**: Semantic-aware token selection using CLIP anchor words to identify aesthetically significant regions. Applies differential CFG guidance (higher scale on foreground, lower on background) via a one-shot sparse attention mask.
- **StepSkip (fskip)**: Full-step skipping every *k* denoising steps, caching and reusing noise predictions for background regions while refreshing foreground tokens.

Supported backbones: **Lumina-Next-T2I**, **FLUX.1-dev**, **SD3-Medium**

## Results

| Method | Speedup | IR | LPIPS | FID |
|---|---|---|---|---|
| Baseline | 1.00× | 0.752 | 0.000 | 0.0 |
| Δ-DiT | 1.52× | 0.485 | 0.172 | 97.4 |
| RAS | 1.47× | 0.788 | 0.025 | 20.0 |
| TeaCache | 1.49× | 0.670 | 0.046 | 37.7 |
| TaylorSeer | 2.05× | 0.408 | 0.170 | 112.2 |
| **AccelAes (ours)** | **2.11×** | **0.841** | 0.057 | 46.6 |

## Repository Structure

```
src/
  models/          # Model wrappers (Lumina, FLUX, SD3, SDXL)
  sparse/          # AesMask, sparse attention, skip cache
  baselines/       # Reproduced baselines (RAS, DeltaDiT, FORA, TeaCache, TaylorSeer)
  eval/            # Metrics (IR, CLIP, LPIPS, FID, HPSv2, PickScore)
scripts/
  run_p0_eval.py           # Main Lumina evaluation
  run_sd3_compare.py       # SD3 baseline comparison
  run_stepcache_fulleval.py # FLUX + TeaCache + TaylorSeer evaluation
  run_supp_sensitivity.py  # Sensitivity analysis (mask_step, skip_ratio)
  run_anchor_ablation.py   # AesMask ablation
  gen_supp_qual_vis.py     # Supplementary qualitative figures
  gen_sensitivity_vis.py   # Sensitivity visualization figures
prompts/
  prompts_dev.txt          # 200-prompt general evaluation set
  prompts_all.txt          # 40-prompt aesthetic evaluation set
configs/
  base.yaml                # Default hyperparameters
```

## Key Hyperparameters

```python
# AccelAes default (Lumina)
mask_type        = "semantic"
region_method    = "threshold"
skip_ratio       = 0.50
s_fg             = 7.0      # foreground CFG scale
s_bg             = 1.0      # background CFG scale
mask_step        = 5        # mask refresh interval
full_skip_interval = 2      # skip every 2nd step
sparse_ffn       = True
sparse_blocks    = True
cfg_scale        = 4.0
steps            = 30
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- diffusers, transformers, accelerate
- CLIP (`openai/clip-vit-large-patch14`)
- ImageReward, HPSv2, LPIPS, pytorch-fid

## Usage

```python
from src.models.dit_wrapper import LuminaDiTWrapper

wrapper = LuminaDiTWrapper(dtype="bf16")
img = wrapper.generate_accelerated_dual(
    prompt="A peacock displaying its intricate iridescent plumage, photorealistic",
    seed=0,
    mask_type="semantic", region_method="threshold",
    skip_ratio=0.50, s_fg=7.0, s_bg=1.0, mask_step=5,
    full_skip_interval=2, sparse_ffn=True, sparse_blocks=True,
    cfg_scale=4.0, steps=30,
)
img.save("output.png")
```
