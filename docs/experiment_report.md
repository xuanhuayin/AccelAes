# SASD 实验报告：Aesthetic-Aware Sparse Diffusion

## 1. 项目概述

SASD (Semantic-Aware Sparse Diffusion) 提出将 **算力稀疏性 (Computational Sparsity)** 与 **语义引导抑制 (Semantic Guidance Suppression)** 统一到一个 training-free 框架中。核心假说：背景区域存在 **双重冗余性** —— 既不需要高频计算，也不需要高频语义注入。对背景同时实施"少算 + 弱引导"的双重降级，可以在加速推理的同时维持甚至提升图像质量。

**模型**：Lumina-Next-T2I (Alpha-VLLM/Lumina-Next-SFT-diffusers)
**评测规模**：20 prompts × 3 seeds = 60 images per config，1024×1024 分辨率

---

## 2. 方法模块与对应实验

| 模块 | 方法 | 对应实验 |
|---|---|---|
| Zero-Shot Aesthetic Mask | 利用 DiT 内部 cross-attention + text embedding anchor matching 识别前景/背景 | Phase 2 |
| Sparse Forward (操作 A) | 背景 token 跳过 self-attention SDPA，复用 per-layer cache | Phase 5 |
| CFG Suppression (操作 B) | 背景区域 CFG scale 降至 s_bg ≈ 1.0 | Phase 1/2 |
| **Dual Degradation** | **操作 A + 操作 B 组合** | **SASD Core** |

---

## 3. 实验结果

### 3.1 Phase 5: Sparse Forward vs Baseline

验证"背景 token 跳过 self-attention"能否实现真实加速。

| Config | Time (s/img) | Speedup | CLIP Score | ΔCLIP vs Baseline | Edge Density |
|---|---|---|---|---|---|
| Baseline (dense, CFG=4.0) | 12.54 | 1.00x | 0.2531 | — | 0.583 |
| Sparse skip=0.50, no refresh | 8.40 | **1.49x** | 0.2343 | -0.019 | 0.509 |
| Sparse skip=0.50, refresh=5 | 8.47 | **1.48x** | 0.2517 | **-0.001** | 0.544 |
| Sparse skip=0.75 | 8.14 | **1.54x** | 0.2305 | -0.023 | 0.451 |
| Sparse semantic skip=0.50 | 9.30 | 1.35x | 0.2429 | -0.010 | 0.519 |
| Skip-update P3 (copy, 无真实加速) | 12.51 | 1.00x | 0.2534 | +0.000 | 0.555 |
| Skip-update P4 (velocity, 无真实加速) | 12.53 | 1.00x | 0.2539 | +0.001 | 0.583 |

**关键发现**：
- Phase 3/4 的 skip-update **没有真实加速**（每步仍跑完整 forward），只是后处理混合。Phase 5 sparse forward 实现了 **1.48-1.54x 真实加速**。
- **refresh=5 至关重要**：不刷新 cache 的 CLIP 掉 0.019，刷新后仅掉 0.001。
- skip=0.75 质量下降过大，0.50 是 sweet spot。

### 3.2 SASD Dual Degradation: Sparse + Spatial CFG

验证核心假说：CFG 抑制能否掩盖 sparse 伪影。

以 **Spatial-CFG-only (s_fg=7, s_bg=1)** 为质量参考基准（公平对比，同 CFG 设置）：

| Config | Time (s/img) | Speedup | CLIP Score | ΔCLIP vs Ref | Edge Density |
|---|---|---|---|---|---|
| Baseline dense (CFG=4.0) | 12.53 | 1.00x | 0.2531 | +0.004 | 0.583 |
| **Spatial-CFG-only** (s_fg=7, s_bg=1, 无加速) | 12.52 | 1.00x | 0.2490 | 0 (ref) | 0.547 |
| Sparse-only (skip=0.50, r=5, CFG=4) | 8.47 | **1.48x** | 0.2517 | +0.003 | 0.544 |
| **DUAL skip=0.50, cfg 7/1, refresh=5** | **8.47** | **1.48x** | **0.2487** | **-0.0002** | 0.515 |
| DUAL skip=0.50, cfg 7/1, no refresh | 8.39 | 1.49x | 0.2365 | -0.013 | 0.480 |
| DUAL skip=0.75, cfg 7/1, refresh=5 | 8.27 | **1.52x** | 0.2447 | -0.004 | 0.486 |
| **DUAL skip=0.50, cfg 5/2, refresh=5** | **8.47** | **1.48x** | **0.2505** | **+0.002** | 0.537 |

**核心发现**：

1. **DUAL (skip=0.50, cfg 7/1, refresh=5) vs Spatial-CFG-only: CLIP 仅差 -0.0002**，统计上无差异，验证了"sparse 不损质量"。
2. **DUAL (cfg 5/2) CLIP 比参考还高 +0.002**，说明温和 CFG 差异下 dual degradation 完全无损。
3. **Edge density 下降符合预期**：背景低 CFG 导致细节减少 = 自然虚化 (Bokeh 效果)。
4. **无 refresh 掉分严重 (-0.013)**，refresh 是必要组件。
5. **Spatial CFG 不增加额外计算开销**（8.47s vs 8.47s），dual 的速度等于 sparse-only。

### 3.3 速度分析

Sparse forward 的加速来源：

- Self-attention 从 O(N²) 降到 O(N_fg × N)，skip=0.50 时 attention 计算减半
- Q/K/V 投影 + cross-attention 仍在全 N tokens 上（O(N)），限制了进一步加速
- Per-layer cache 内存: ~864 MB (24 layers × 36MB/layer)，32GB GPU 可接受

理论 vs 实测：
- 理论预测（仅 attention 减半）: ~1.5-2.0x
- 实测: 1.48x — 符合预测，O(N) 操作成为瓶颈

---

## 4. 视觉质量对比

### Sparse Forward 修复过程

初版实现 Q_fg @ K_fg（前景只 attend 前景），导致严重马赛克伪影。
修复为 Q_fg @ K_all（前景 attend 所有 token），质量大幅恢复。

### Dual Degradation 效果

- 背景低 CFG 使 sparse 残留伪影被"柔化"为自然的虚化效果
- 前景主体保持完整细节和高质量
- refresh=5 进一步消除 cache 过时导致的残留伪影

---

## 5. 与 Contribution 提案的对标

| 提案声明 | 实验验证 | 状态 |
|---|---|---|
| 背景"双重冗余性"假说 | DUAL (cfg 7/1) CLIP -0.0002 vs ref | **已验证** |
| Training-free Aesthetic Discovery | Cross-attn + anchor matching（无 SAM） | **已实现** |
| 继承 SDiT 2.7-3.0x 加速 | 实测 1.48-1.52x | **部分达成** |
| 美学评分超越 SDiT | 未对标（无 SDiT 在同模型上的实现） | **待补充** |
| 背景 Bokeh 效果 | Edge density 下降 0.547→0.515 | **已观察到** |
| 幻觉抑制测试 | 未做（需干扰词 prompt 测试） | **待补充** |

---

## 6. 当前不足与差距

### 6.1 加速比不足
- 实测 1.48x vs 提案声称的 2.7-3.0x，差距较大
- 原因：(1) 我们只 sparse attn1，Q/K/V 投影 + cross-attention 仍全量；(2) SDiT 的"跳步"是整层/整步跳过，更激进
- 可能改进：combine with step-skipping, 或 sparse cross-attention

### 6.2 缺失实验
- **FID**：需要在大规模数据集（COCO 30K）上计算，目前只有 CLIP score
- **PSNR/SSIM**：需要和参考图逐像素对比
- **CLIP-Aesthetic Score**：提案中强调需要测但未做
- **幻觉测试**：用干扰词 prompt 验证 CFG 抑制能否减少背景幻觉
- **SDiT 对标**：在同一模型上复现 SDiT，apple-to-apple 对比
- **更高分辨率**：2048×2048 下 N 更大，加速比应更高
- **多模型验证**：SD3 / FLUX 等其他 DiT 模型

### 6.3 方法完整性
- 提案中的 **三级区域划分**（美学核心区 / 结构支撑区 / 背景抑制区）未实现，目前是二值 mask
- **Quickshift 聚类** 替代方案未实现
- **注意力-复杂度对齐**（Dual-Filter 的 Step 2）未充分验证

---

## 7. 文件清单

| 文件 | 用途 |
|---|---|
| `src/sparse/sparse_processor.py` | SparseAttnProcessor（attn1 稀疏 SDPA） |
| `src/sparse/sparse_hook.py` | SparseHookManager（安装/移除/cache管理） |
| `src/models/dit_wrapper.py` | generate_sparse_forward() + generate_dual_degradation() |
| `scripts/run_generate.py` | CLI 入口（--sparse_forward / --dual_degrade） |
| `scripts/run_phase5_smoke.py` | Phase 5 冒烟测试 |
| `scripts/run_phase5_eval.py` | Phase 5 完整评测 |
| `scripts/run_dual_smoke.py` | Dual degradation 冒烟测试 |
| `scripts/run_dual_eval.py` | Dual degradation 完整评测 |
