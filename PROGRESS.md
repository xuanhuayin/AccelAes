# SASD 项目进展文档

> 最后更新: 2026-02-18
> 目的: 给 Claude Code 和自己看的完整实验状态记录

---

## 一、项目目标 & 路线图（对照 SASD_plan.pdf）

在 Lumina-Next-T2I (DiT) 上实现**基于语义感知的稀疏计算推理加速**。

| Phase | SASD_plan 描述 | 状态 | 说明 |
|---|---|---|---|
| Phase 1 | Spatial CFG Field (不省算) | **✅ 完成** | 结论：对质量无提升 |
| Phase 2 | 自动 mask: 语义/美学感知 (不省算) | **✅ 完成** | mask 基础设施可靠，被后续 Phase 复用 |
| Phase 3 | "跳更新"原型 (仍不省算) | **✅ 实验完成** | 12 组 Quick Sweep + 4 组 Full Scale |
| Phase 3 交付物 | 质量曲线 vs ratio + 失败案例集 | **✅ 完成** | 图表 + artifact list + 失败模式分析 |
| Phase 4 | SDiT baseline 对标 (分割升级 + 边界 + velocity 外推) | **❌ 未开始** | |
| Phase 5 | 真正省算: Sparse Forward Compute | **❌ 未开始** | |

> **注:** Phase 3a (ToMe) 不在原始 SASD_plan 中，是额外尝试的加速路径，代码已完成但优先级低于主线。

---

## 二、模型与环境

| 项 | 值 |
|---|---|
| 模型 | `Alpha-VLLM/Lumina-Next-SFT-diffusers` |
| 文本编码器 | Gemma (SentencePiece, BOS=2, EOS=1) |
| CLIP(语义计算) | `openai/clip-vit-large-patch14` (fp16, 按需加载) |
| VAE scale factor | 8 |
| Patch size | 2 → 1024px 图像 = 128×128 latent = **64×64 = 4096 patches** |
| 自注意力 | 32 query heads, 8 KV heads (GQA), head_dim=72 |
| Transformer 层数 | 24 层 |
| 采样 | 30 步, CFG=4.0, DPM-Solver++ |
| 模型预测头 | eps |
| Baseline 速度 | **~12.5s/张** (1024×1024, RTX 5090) |
| Dev prompts | `prompts/prompts_dev.txt`, **200 条** |
| 配置文件 | `configs/base.yaml` |

---

## 三、Phase 1: 空间 CFG（✅ 完成）

**目标（plan P1）:** 证明"空间引导强度场"在 DiT 上稳定可用。

在前景/背景区域使用不同 CFG scale，结论是**对图像质量没有提升**。

| 实验 | CLIP Score | Edge Density |
|---|---|---|
| `p1_baseline_cfg7` (25p×1s) | 0.2580 | 0.7723 |
| `p1_spatial_cfg_bg0.0` | — | — |
| `p1_spatial_cfg_bg0.5` | — | — |
| `p1_spatial_cfg_bg1.0` | — | — |

---

## 四、Phase 2: 自动 Mask（✅ 完成）

**目标（plan P2）:** 把 mask 从"人为"变成"可解释、可重复"。

构建了两种 mask builder（Complexity + Semantic），验证了 mask 质量。空间 CFG 质量依然无提升，但 **mask 基础设施被 Phase 3 复用**。

| 实验 | N | CLIP Score | Edge Density | 说明 |
|---|---|---|---|---|
| `p2_baseline_cfg4` | 150 | **0.2540** | **0.5344** | Phase 2 基线 |
| `p2_semantic_dev` | 150 | 0.2468 | 0.4968 | 语义 mask + 空间 CFG |
| `p2_complexity_dev` | 150 | 0.2451 | 0.4947 | 复杂度 mask + 空间 CFG |
| `p2v3_sem_r35_fg50_bg40` | 150 | 0.2532 | 0.5337 | 最新调参版 |

**关键发现：**
- Gemma 嵌入不适合余弦相似度匹配（所有 token 聚在 0.63-0.85 之间）
- 改用 CLIP 文本编码器计算 token 重要性，阈值 0.60 有效分离真匹配和噪声
- ~20% 的 prompt 触发 anchor 加权，~80% 回退到纯 cross-attention
- Complexity mask 和 Semantic mask 都能可靠地识别前景区域

---

## 五、Phase 3: 跳更新原型（✅ 实验完成，⏳ 交付物待产出）

**目标（plan P3）:** 验证"跳过部分区域更新 + 外推/缓存"不会明显破坏质量。

> 注意: 此阶段 forward 仍全量，先验证数值与观感稳定。

### 5.1 核心思路

工作在 **noise_pred 空间级别**（CFG 之后、scheduler.step 之前）。每步仍然运行完整的 forward pass，但在背景区域用缓存/外推的 noise_pred 替换当前值。

```
for i, t in enumerate(timesteps):
    noise_pred = transformer(...)          # 完整 forward（每步都跑）
    noise_pred = apply_cfg(noise_pred)     # 标准 CFG
    noise_pred_full = noise_pred.clone()   # 在 masking 前保存

    if cache.has_cache() and i > mask_step and not is_refresh_step:
        cached = cache.get_prediction(i)   # copy 或 linear 外推
        noise_pred = fg_mask * noise_pred + (1-fg_mask) * cached  # 混合

    cache.store(i, noise_pred_full)        # 总是缓存完整预测
    latents = scheduler.step(noise_pred, t, latents)
```

### 5.2 已完成的代码

| 文件 | 说明 |
|---|---|
| `src/sparse/skip_cache.py` | SkipUpdateCache: 缓存 + copy/linear 外推 |
| `src/models/dit_wrapper.py` | `generate_skip_update()` 方法 |
| `scripts/run_generate.py` | `--skip_update` 系列参数 |

### 5.3 实验矩阵 & 结果

**plan 要求:** ratio ∈ {0.125, 0.25, 0.5}, extrapolation ∈ {copy, linear}, refresh ∈ {none, every 5 steps}

#### Quick Sweep（12 组 × 10 prompts × 3 seeds = 360 图）

| skip_ratio | extrapolation | refresh | CLIP Score | Edge Density | CLIP Δ% | Edge Δ% |
|---|---|---|---|---|---|---|
| — (baseline) | — | — | **0.2579** | **0.7488** | — | — |
| 0.125 | copy | 0 | 0.2572 | 0.7321 | -0.27% | -2.23% |
| 0.125 | copy | 5 | 0.2574 | 0.7346 | -0.19% | -1.90% |
| **0.125** | **linear** | **0** | **0.2582** | **0.7481** | **+0.12%** | **-0.09%** |
| 0.125 | linear | 5 | 0.2579 | 0.7483 | 0.00% | -0.07% |
| 0.25 | copy | 0 | 0.2564 | 0.7150 | -0.58% | -4.52% |
| 0.25 | copy | 5 | 0.2571 | 0.7203 | -0.31% | -3.81% |
| **0.25** | **linear** | **0** | **0.2580** | **0.7469** | **+0.04%** | **-0.25%** |
| 0.25 | linear | 5 | 0.2582 | 0.7473 | +0.12% | -0.20% |
| 0.5 | copy | 0 | 0.2562 | 0.6791 | -0.66% | -9.31% |
| 0.5 | copy | 5 | 0.2570 | 0.6905 | -0.35% | -7.79% |
| 0.5 | linear | 0 | 0.2572 | 0.7455 | -0.27% | -0.44% |
| 0.5 | linear | 5 | 0.2574 | 0.7458 | -0.19% | -0.40% |

#### Full Scale（最佳 4 配置 × 50 prompts × 3 seeds = 150 图/组）

| 实验 | skip_ratio | extrapolation | refresh | CLIP Score | Edge Density | CLIP Δ% | Edge Δ% |
|---|---|---|---|---|---|---|---|
| `p3b_baseline_full` | — | — | — | **0.2540** | **0.5344** | — | — |
| `p3b_full_r0125_linear_norefresh` | 0.125 | linear | 0 | 0.2537 | 0.5312 | -0.12% | -0.60% |
| `p3b_full_r025_linear_norefresh` | 0.25 | linear | 0 | 0.2533 | 0.5303 | -0.28% | -0.77% |
| `p3b_full_r025_linear_refresh5` | 0.25 | linear | 5 | **0.2538** | 0.5306 | **-0.08%** | -0.71% |
| `p3b_full_r05_linear_norefresh` | 0.5 | linear | 0 | 0.2532 | 0.5288 | -0.31% | -1.05% |

### 5.4 Phase 3 关键结论

1. **Linear 外推完胜 Copy** — 在所有 ratio 下，linear 的 CLIP 和 edge density 都明显优于 copy。Copy 在 r=0.5 时 edge density 降 9.3%，linear 仅降 0.4%。
2. **Linear 外推几乎无损** — r=0.25 linear 的 CLIP 降幅 <0.3%，edge density 降幅 <0.8%，在统计噪声范围内。
3. **Refresh 对 linear 帮助不大** — linear 外推已经足够好，refresh 带来的增益微乎其微。对 copy 有一定帮助。
4. **"背景可以跳更新"假设验证通过** — 即使 r=0.5 (跳过 50% 背景 token 的更新)，质量依然可接受。

### 5.5 Phase 3 交付物（✅ 已完成）

#### 质量曲线 vs ratio

- 图表文件: `outputs/phase3_quality_curves.png`, `outputs/phase3_quality_delta.png`
- 绘图脚本: `scripts/plot_phase3_curves.py`
- **核心结论:** linear 外推曲线几乎平坦（edge density Δ <1%），copy 外推随 ratio 增大急剧下降（r=0.5 时 edge Δ ≈ -9%）

#### 失败案例集 (artifact list)

- 数据文件: `outputs/phase3_artifact_list.json`
- 分析脚本: `scripts/find_artifacts.py`
- 对比: `p3b_full_r05_linear_norefresh` vs `p3b_baseline_full` (150 图)

**统计概览 (r=0.5 linear norefresh):**

| 指标 | 均值 Δ% | 最差 Δ% | >2% 退化比例 |
|---|---|---|---|
| CLIP Score | -0.22% | -17.62% | 12.7% (19/150) |
| Edge Density | -1.22% | -32.58% | 8.7% (13/150) |

**最严重的失败模式:**

1. **prompt #37 "limes"** — CLIP 降 6-17%，edge 降 25-33%。这是一个低复杂度的静物场景，全图内容同质，mask 难以区分前景/背景，导致"背景"其实就是主体本身。
2. **prompt #10 ","（逗号）** — 无语义内容的 prompt，模型生成随机图案，edge 降 7-18%。mask 在没有语义信号时退化严重。
3. **prompt #22 "rainy window"** — 全图均匀纹理（雨滴），edge 降 7-8%。均匀纹理场景不适合前景/背景分割。

**结论:** 大部分样本（~87%）edge 退化 <2%，失败集中在：(a) 低复杂度同质场景，(b) 无语义 prompt，(c) 全图均匀纹理。这些是 mask 本身的局限性，不是外推策略的问题。

---

## 5.5b、Phase 3a: Token 合并 ToMe（代码完成，优先级低）

> 不在原始 SASD_plan 中，作为额外加速路径。

核心思路: 在自注意力中合并相似的背景 token，减少序列长度获得 O(N²) 级加速。

| 文件 | 说明 |
|---|---|
| `src/sparse/token_merge.py` | 核心合并/反合并算法（ToMe 棋盘格分区 + mask 保护） |
| `src/sparse/tome_processor.py` | ToMe 自注意力 Processor（替换 attn1） |
| `src/sparse/tome_hook.py` | Hook 管理器 |

速度测试已跑，质量实验待进一步验证。非主线任务，暂不优先。

---

## 六、下一步计划（对照 SASD_plan）

### 当前位置: Phase 3 完成 → 进入 Phase 4

Phase 3 所有实验和交付物已完成。下一步进入 Phase 4。

### 下一步: Phase 4 — SDiT Baseline 对标（plan P4）

**目标:** 建立强 baseline，明确我们方法超越它的空间。

| 子任务 | plan 章节 | 说明 |
|---|---|---|
| P4.1 分割升级 | Quickshift 或 SLIC superpixels | 关键是"连通区域"而非散点 mask |
| P4.2 边界膨胀 + soft mask | dilation (r=1-2) + mask blur | 降低边界断裂 |
| P4.3 velocity-space 外推 | SDiT-style | eps/v/x0 统一到中间变量空间外推 |

**交付物:**
- SDiT-like baseline: ratio=0.125/0.25 的质量与速度
- 与 Phase 3 简化 baseline 的差异分析

### Step C: Phase 5 — 真正省算（plan P5）

从"更新稀疏"升级为"计算稀疏"，实现真实 wall-clock 加速。

两条路径并行评估:
- **Path A:** token 子集计算（只对 active 子集做 attention/MLP）
- **Path B:** 块稀疏 block-wise skipping（以 4×4 block 为单位 gather/scatter）

**验收标准:** speedup ≥ 1.5×，质量下降在可接受范围内。

---

## 七、代码结构速查

```
SASD/
├── configs/base.yaml              # 模型和采样配置
├── prompts/prompts_dev.txt        # 200 条 dev prompts
├── src/
│   ├── models/dit_wrapper.py      # 核心：generate(), generate_spatial_cfg(),
│   │                              #        generate_with_auto_mask(), generate_tome(),
│   │                              #        generate_skip_update()
│   ├── sparse/
│   │   ├── mask_builders.py       # ComplexityMaskBuilder, SemanticMaskBuilder
│   │   ├── cross_attn_hook.py     # CrossAttnHookManager
│   │   ├── token_merge.py         # [Phase 3a] bipartite_soft_matching_with_mask()
│   │   ├── tome_processor.py      # [Phase 3a] ToMeSelfAttnProcessor
│   │   ├── tome_hook.py           # [Phase 3a] ToMeHookManager
│   │   └── skip_cache.py          # [Phase 3b] SkipUpdateCache (copy/linear 外推)
│   ├── eval/
│   │   ├── metrics.py             # CLIPScore, Edge Density
│   │   └── speed.py               # SpeedBenchmark
│   └── utils/
│       ├── io_utils.py            # 配置加载、输出目录
│       ├── log.py                 # JsonLogger
│       └── seed.py                # 种子管理
├── scripts/
│   ├── run_generate.py            # 主生成脚本（所有 phase）
│   ├── run_speed.py               # 速度基准测试
│   ├── run_eval.py                # 质量评估（CLIPScore + Edge Density）
│   ├── diagnose_masks.py          # Mask 诊断
│   └── viz_masks.py               # Mask 可视化
└── outputs/                       # 实验输出
```

---

## 八、关键 CLI 命令速查

```bash
# 生成（baseline）
python scripts/run_generate.py --config configs/base.yaml \
    --prompts prompts/prompts_dev.txt --seeds 0,1,2 --exp_name NAME

# 生成（Phase 3b Skip-Update）
python scripts/run_generate.py --config configs/base.yaml \
    --prompts prompts/prompts_dev.txt --seeds 0,1,2 \
    --skip_update --skip_ratio 0.25 --extrapolation linear \
    --refresh_interval 0 --skip_mask_type complexity \
    --exp_name NAME

# 生成（Phase 3a ToMe）
python scripts/run_generate.py --config configs/base.yaml \
    --prompts prompts/prompts_dev.txt --seeds 0,1,2 \
    --tome --merge_ratio 0.5 --tome_mask_type complexity \
    --exp_name NAME

# 速度测试
python scripts/run_speed.py --config configs/base.yaml \
    --tome --merge_ratio 0.5 --tome_mask_type uniform \
    --exp_name NAME

# 质量评估
python scripts/run_eval.py --exp_dir outputs/NAME

# Mask 可视化
python scripts/run_generate.py ... --save_masks
```
