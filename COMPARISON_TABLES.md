# Baseline Comparison Tables

8 metrics: CLIP↑ · PickScore↑ · ImageReward↑ · HPSv2↑ · Aesthetic↑ · LPIPS↓ · Edge↑ · FID↓

---

## Table 1 — Lumina-Next-T2I

20 prompts × 3 seeds = 60 images/config · 30 steps · CFG = 4.0

| Method | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|-----:|--------:|------:|------:|----:|-----:|-------:|-------:|------:|-----:|
| Baseline | 12.37s | 1.00× | 0.2531 | 0.2185 | 0.752 | 0.2710 | 5.941 | 0.000 | 0.583 | 0.0 |
| Δ-DiT [2406.01125] | 8.12s | 1.52× | 0.2523 | 0.2158 | 0.485 | 0.2540 | 5.873 | 0.172 | 0.522 | 97.4 |
| FORA [2407.01425] | 12.36s | 1.00× | 0.2463 | 0.2128 | 0.517 | 0.2496 | 5.766 | 0.242 | 0.565 | 148.0 |
| RAS [2502.10389] | 8.38s | 1.47× | 0.2550 | 0.2185 | 0.788 | 0.2713 | 5.927 | 0.025 | 0.581 | 20.0 |
| **AccelAes (ours)** | **5.86s** | **2.11×** | **0.2540** | **0.2188** | **0.841** | **0.2740** | **5.941** | 0.057 | **0.629** | 46.6 |
| Δ vs baseline | — | — | +0.4% | +0.1% | **+11.9%** | **+1.1%** | +0.0% | — | **+7.8%** | — |

> AccelAes 配置：semantic mask (threshold) + sparse attn/FFN + spatial CFG (s_fg=7, s_bg=1) + fskip2 (interval=2)
> RAS 为 output-level blending 复现；官方 2.51× 需 flash_attn+Triton 内核

---

## Table 2 — SD3-Medium

30 prompts × 2 seeds = 60 images/config · 28 steps · CFG = 7.0

| Method | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|-----:|--------:|------:|------:|----:|-----:|-------:|-------:|------:|-----:|
| Baseline | 3.52s | 1.00× | 0.2662 | 0.2198 | 0.879 | 0.2895 | 5.733 | 0.000 | 0.916 | 0.0 |
| TeaCache [2411.19108] (t=0.15) | 3.29s | 1.07× | 0.2662 | — | 0.876 | 0.2893 | 5.729 | **0.007** | 0.916 | **10.3** |
| TeaCache [2411.19108] (t=0.30) | 2.49s | 1.41× | 0.2667 | — | 0.847 | 0.2863 | 5.717 | 0.056 | 0.881 | 39.5 |
| TaylorSeer [2503.06923] (r=1) | 2.21s | 1.59× | 0.2682 | — | **0.898** | 0.2852 | 5.676 | 0.116 | 0.935 | 70.2 |
| TaylorSeer [2503.06923] (r=2) | 1.74s | **2.02×** | 0.2624 | — | 0.729 | 0.2723 | 5.561 | 0.281 | 0.998† | 132.6 |
| Δ-DiT [2406.01125] | 2.33s | 1.51× | 0.2634 | — | 0.828 | 0.2804 | 5.656 | 0.278 | 0.850 | 116.9 |
| FORA [2407.01425] | 3.54s | 0.99× | 0.2663 | — | 0.703 | 0.2643 | 5.522 | 0.282 | 0.830 | 146.1 |
| Step-cache only (fskip2) | 2.34s | 1.50× | **0.2688** | 0.2191 | 0.891 | 0.2867 | 5.663 | 0.058 | **0.960** | 44.3 |
| **AccelAes (ours)** | **2.34s** | **1.50×** | 0.2644 | 0.2187 | 0.804 | 0.2814 | 5.690 | 0.111 | 0.944 | 70.7 |
| Δ vs step-cache only | — | — | −1.7% | −0.2% | −9.8% | −1.8% | +0.5% | worse | −1.7% | worse |

> AccelAes 配置：semantic mask (joint attn affinity) + spatial CFG (s_fg=9, s_bg=2) + fskip2（无 sparse FFN，AdaLN 耦合排除）
> PickScore 仅对 fskip2_only 和 AccelAes 记录（其余 — 为未测）
> SD3 上未复现 RAS（需 MMDiT 专用 PIT kernel）
> †TaylorSeer r=2 Edge=0.998 为伪高值：图像崩溃后大量高频噪声使 Canny 计数虚高，非真实细节保留

---

## Table 3 — FLUX.1-dev

24 prompts × seed=42 = 24 images/config · 28 steps · guidance = 3.5

| Method | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|-----:|--------:|------:|------:|----:|-----:|-------:|-------:|------:|-----:|
| Baseline | 12.66s | 1.00× | 0.2753 | 0.2290 | 1.233 | 0.3200 | 6.243 | 0.000 | 0.560 | 0.0 |
| TeaCache [2411.19108] (t=0.15) | 7.80s | 1.62× | 0.2765 | 0.2292 | 1.283 | 0.3174 | 6.268 | 0.032 | 0.535 | 21.9 |
| TeaCache [2411.19108] (t=0.30) | 5.92s | 2.14× | 0.2777 | 0.2291 | 1.225 | 0.3154 | 6.301 | 0.071 | 0.529 | 39.6 |
| TaylorSeer [2503.06923] (r=1) | 7.74s | 1.63× | 0.2763 | 0.2292 | 1.295 | 0.3196 | 6.261 | **0.018** | 0.565 | **13.4** |
| TaylorSeer [2503.06923] (r=2) | 5.96s | **2.12×** | 0.2765 | 0.2291 | **1.304** | **0.3217** | **6.309** | 0.055 | 0.572 | 32.6 |
| **AccelAes (ours)** | **7.31s** | **1.73×** | 0.2752 | 0.2290 | 1.267 | 0.3214 | 6.272 | 0.030 | **0.590** | 19.5 |
| Δ vs baseline | — | — | −0.0% | +0.0% | **+2.7%** | +0.4% | +0.5% | — | **+5.4%** | — |

> AccelAes 配置：fskip2 only（无 spatial CFG：guidance distillation 无双 pass；无 sparse attn/FFN：2×2 spatial packing 引入跨 patch 混叠，实验排除）
> baseline 时间取 steady-state ≈ 12.66s（首样本 GPU warmup ~116s 排除）；AccelAes 取 last-8 steady-state ≈ 7.31s

---

## 跨模型汇总

| Model | Arch | AccelAes 配置 | Speedup | IR Δ | LPIPS | Edge Δ |
|-------|------|--------------|--------:|-----:|------:|-------:|
| Lumina-Next-T2I | No AdaLN · CFG | Sparse attn+FFN + Spatial CFG + fskip2 | **2.11×** | **+11.9%** | 0.057 | **+7.8%** |
| SD3-Medium | AdaLN · CFG | Spatial CFG + fskip2 | 1.50× | −8.5%* | 0.111 | +3.1% |
| FLUX.1-dev | AdaLN · no CFG | fskip2 only | 1.73× | +2.7% | 0.030 | +5.4% |

> *SD3 IR 下降源于 spatial CFG (s_bg=2.0)，非 fskip2；step-cache only 在 SD3 上 IR +1.4%
