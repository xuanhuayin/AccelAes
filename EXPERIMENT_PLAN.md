# SASD 实验方案 & 代码迁移指南

> 2026-02-22 | 目标：准备顶会投稿所需的全部实验数据
> 最后更新：2026-02-22

---

## 零、已完成实验结果

### Mask Type 对比实验 ✅
> Lumina-Next-SFT-diffusers, 20 prompts × 1 seed, 1024×1024, skip_ratio=0.5, full_skip_interval=4, s_fg=7.0, s_bg=1.0

| Config | Time/img | CLIP↑ | Edge↑ | Speedup |
|--------|----------|-------|-------|---------|
| baseline | 12.32s | 0.2534 | 0.5674 | 1.00x |
| sasd_semantic | 8.50s | **0.2578** | 0.5735 | 1.45x |
| sasd_cfg_magnitude | 7.37s | 0.2571 | 0.5836 | **1.67x** |
| sasd_complexity | 7.37s | 0.2514 | 0.5737 | 1.67x |
| sasd_uniform | 7.36s | 0.2528 | 0.5682 | 1.67x |

**关键发现：**
- Semantic mask CLIP 分最高（0.2578），甚至超过 baseline（0.2534）——语义感知增强了前景质量
- Semantic mask 较慢（1.45x vs 1.67x），原因：CLIP 每张图重新加载（~1.1s 额外开销）
- cfg_magnitude 在速度和质量之间最平衡
- uniform/complexity 质量明显低于内容感知方法

**待优化：** Semantic mask 速度（见 TODO P0 - semantic 优化）

---

## 一、实验总览

| 实验 | 模型 | 目的 | 状态 | 优先级 |
|------|------|------|------|--------|
| E0. Mask type 对比 | Lumina-Next | 选最优 mask | ✅ 完成 (20p×1s) | — |
| E1. Lumina baseline | Lumina-Next | 基准线 | ⏳ 待跑 (200p×3s) | P0 |
| E2. Lumina SASD | Lumina-Next | 主实验 | ⏳ 待跑 (200p×3s) | P0 |
| E3. Lumina 消融 | Lumina-Next | ablation table | ⏳ 待跑 | P0 |
| E4. Lumina baseline对比 | Lumina-Next | vs Delta-DiT, FORA, ToMe | ⏳ 待跑 | P0 |
| E5. SD3 baseline | SD3-Medium | 跨架构 baseline | 🔒 blocked (HF login) | P0 |
| E6. SD3 SASD | SD3-Medium | 跨架构验证 | 🔒 blocked | P0 |
| E7. noise_pred 分析 | Lumina-Next | motivation figure | ⏳ 待跑 | P1 |
| E8. 速度基准 | 全部 | wall-clock timing | ⏳ 待跑 | P1 |
| E9. PixArt-α (可选) | PixArt-α | 第3个架构 | ⏳ 可选 | P2 |
| E10. PartiPrompts | Lumina | 大规模 FID | ⏳ 可选 | P2 |

---

## 二、详细实验配置

### E1 & E2: Lumina-Next-T2I 主实验

**当前 CFG 参数说明：**
- 小规模测试（mask compare）：`s_fg=7.0, s_bg=1.0`（Lumina 默认 cfg_scale=4）
- 主实验推荐：`s_fg=9.0, s_bg=2.0`（更强的前景增强）
- 可选变体：`s_fg=7.0, s_bg=1.0` / `s_fg=12.0, s_bg=1.0`（在消融中覆盖）

```bash
# E1: Baseline
python scripts/run_generate.py \
    --method baseline \
    --prompts prompts/prompts_dev.txt \
    --num_prompts 200 --num_seeds 3 \
    --output_dir outputs/lumina_eval/baseline

# E2: SASD 完整方法 (cfg_magnitude，速度/质量最优)
python scripts/run_generate.py \
    --method accelerated_dual \
    --mask_type cfg_magnitude \
    --skip_ratio 0.50 \
    --s_fg 9.0 --s_bg 2.0 \
    --mask_step 5 \
    --sparse_ffn \
    --full_skip_interval 4 \
    --num_prompts 200 --num_seeds 3 \
    --output_dir outputs/lumina_eval/sasd_cfg_mag

# E2b: SASD semantic (最高质量)
python scripts/run_generate.py \
    --method accelerated_dual \
    --mask_type semantic \
    --skip_ratio 0.50 \
    --s_fg 9.0 --s_bg 2.0 \
    --mask_step 5 \
    --sparse_ffn \
    --full_skip_interval 4 \
    --num_prompts 200 --num_seeds 3 \
    --output_dir outputs/lumina_eval/sasd_semantic
```

**注意**：需要确认 `run_generate.py` 支持这些参数，或者写一个新的统一评测脚本。

### E3: Lumina 消融实验

```bash
python scripts/run_ablation.py --num_prompts 200 --num_seeds 3
```

**消融矩阵**（每行一个实验）：

| 实验名 | mask_type | skip_ratio | sparse_attn | sparse_ffn | skip_interval | s_fg/s_bg |
|--------|-----------|------------|-------------|------------|---------------|-----------|
| ablation_mask_cfg | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_mask_complexity | complexity | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_mask_uniform | uniform | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_ratio_25 | cfg_magnitude | 0.25 | ✓ | ✓ | 0 | 9/2 |
| ablation_ratio_50 | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_ratio_75 | cfg_magnitude | 0.75 | ✓ | ✓ | 0 | 9/2 |
| ablation_attn_only | cfg_magnitude | 0.50 | ✓ | ✗ | 0 | 9/2 |
| ablation_ffn_only | cfg_magnitude | 0.50 | ✗ | ✓ | 0 | 9/2 |
| ablation_both | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_skip0 | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_skip3 | cfg_magnitude | 0.50 | ✓ | ✓ | 3 | 9/2 |
| ablation_skip4 | cfg_magnitude | 0.50 | ✓ | ✓ | 4 | 9/2 |
| ablation_skip5 | cfg_magnitude | 0.50 | ✓ | ✓ | 5 | 9/2 |
| ablation_cfg_uniform | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 7.5/7.5 |
| ablation_cfg_spatial_7_1 | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 7/1 |
| ablation_cfg_spatial_9_2 | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_cfg_extreme | cfg_magnitude | 0.50 | ✓ | ✓ | 0 | 12/1 |
| ablation_maskstep3 | cfg_magnitude | 0.50 | ✓ | ✓ | 4 | 9/2 |
| ablation_maskstep5 | cfg_magnitude | 0.50 | ✓ | ✓ | 4 | 9/2 |
| ablation_maskstep8 | cfg_magnitude | 0.50 | ✓ | ✓ | 4 | 9/2 |
| ablation_maskstep10 | cfg_magnitude | 0.50 | ✓ | ✓ | 4 | 9/2 |
| ablation_extrap_copy | cfg_magnitude | 0.50 | ✓ | ✓ | 4 | 9/2 |
| ablation_extrap_linear | cfg_magnitude | 0.50 | ✓ | ✓ | 4 | 9/2 |
| ablation_mask_semantic | **semantic** | 0.50 | ✓ | ✓ | 0 | 9/2 |
| ablation_sem_fullskip4 | **semantic** | 0.50 | ✓ | ✓ | 4 | 9/2 |

**已完成（小规模验证）**：
| 实验名 | 结果 |
|--------|------|
| baseline | 12.32s, CLIP=0.2534, Edge=0.5674 |
| sasd_semantic | 8.50s, CLIP=**0.2578**, Edge=0.5735, **1.45x** |
| sasd_cfg_magnitude | 7.37s, CLIP=0.2571, Edge=0.5836, **1.67x** |
| sasd_complexity | 7.37s, CLIP=0.2514, Edge=0.5737, 1.67x |
| sasd_uniform | 7.36s, CLIP=0.2528, Edge=0.5682, 1.67x |

### E4: Baseline 对比

```bash
# Delta-DiT
python scripts/run_baseline_compare.py --method delta_dit --num_prompts 200 --num_seeds 3

# FORA
python scripts/run_baseline_compare.py --method fora --num_prompts 200 --num_seeds 3

# ToMe (需要实现/集成)
python scripts/run_baseline_compare.py --method tome --merge_ratio 0.5 --num_prompts 200 --num_seeds 3
```

### E5 & E6: SD3-Medium

```bash
# E5: SD3 Baseline
python scripts/run_sd3_eval.py \
    --num_prompts 200 --num_seeds 3 \
    --skip_baseline false \
    --output_dir outputs/sd3_eval

# E6: SD3 SASD
# (包含在同一个脚本中，自动跑 baseline + SASD configs)
```

### E7: Noise Pred 变化分析

```bash
python scripts/analyze_noise_pred_change.py \
    --num_prompts 10 --num_seeds 1 \
    --output_dir outputs/noise_analysis
```

输出：
- `fg_bg_change.json` — 每步 fg/bg 变化幅度
- `fg_bg_change_plot.png` — 折线图

### E8: 速度基准

```bash
# Lumina
python scripts/run_speed_benchmark.py --model lumina --warmup 5 --measure 30

# SD3
python scripts/run_speed_benchmark.py --model sd3 --warmup 5 --measure 30
```

---

## 三、评估指标计算

### CLIP Score
```python
from src.eval.metrics import compute_clip_score
scores = compute_clip_score(images, prompts)  # 使用 openai/clip-vit-large-patch14
```

### FID (需要安装 pytorch-fid)
```bash
pip install pytorch-fid
python -m pytorch_fid outputs/baseline/images outputs/sasd/images
```

### LPIPS (需要安装)
```bash
pip install lpips
python scripts/compute_lpips.py --ref outputs/baseline/images --gen outputs/sasd/images
```

### Edge Density
```python
from src.eval.metrics import compute_edge_density
density = compute_edge_density(image)  # Canny edge ratio
```

---

## 四、代码迁移指南

### 需要迁移的文件清单

```
SASD/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dit_wrapper.py              # Lumina DiT wrapper (生成核心)
│   │   ├── sdxl_wrapper.py             # SDXL wrapper (可选)
│   │   └── sd3_wrapper.py              # SD3 wrapper (新建)
│   ├── sparse/
│   │   ├── __init__.py
│   │   ├── sparse_processor.py         # ★ Lumina 稀疏 attn processor
│   │   ├── sparse_hook.py              # ★ Lumina hook manager
│   │   ├── sparse_ffn.py              # ★ 稀疏 FFN + SparseBlockManager
│   │   ├── sdit_mask_builder.py       # ★ CFGMagnitude + SDiT mask builder
│   │   ├── mask_builders.py           # Complexity + Semantic mask builder
│   │   ├── skip_cache.py             # ★ 步级缓存
│   │   ├── boundary_ops.py           # soft mask 操作
│   │   ├── cross_attn_hook.py        # cross-attention heatmap
│   │   ├── token_merge.py            # ToMe (可选)
│   │   ├── tome_processor.py         # ToMe processor (可选)
│   │   ├── tome_hook.py              # ToMe hook (可选)
│   │   ├── sd3_sparse_processor.py    # ★ SD3 稀疏 processor (新建)
│   │   ├── sd3_block_manager.py       # ★ SD3 block manager (新建)
│   │   ├── sdxl_sparse_processor.py   # SDXL processor (可选)
│   │   └── sdxl_block_manager.py      # SDXL block manager (可选)
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── delta_dit.py              # Delta-DiT baseline
│   │   └── fora.py                   # FORA baseline
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                # CLIP score + edge density
│   │   └── speed.py                  # SpeedBenchmark
│   └── utils/
│       ├── __init__.py
│       └── seed.py                   # set_seed()
├── scripts/
│   ├── run_generate.py               # Lumina 生成脚本
│   ├── run_sdxl_eval.py              # SDXL 评测
│   ├── run_sd3_eval.py               # SD3 评测 (新建)
│   ├── run_sdxl_smoke.py             # SDXL smoke test
│   ├── run_sd3_smoke.py              # SD3 smoke test (新建)
│   ├── run_ablation.py               # 消融实验
│   ├── run_baseline_compare.py       # baseline 对比 (待写)
│   └── analyze_noise_pred_change.py  # noise_pred 分析
├── prompts/
│   └── prompts_dev.txt               # 200 条测试 prompts
├── configs/
│   └── base.yaml
├── PAPER_OUTLINE.md
├── EXPERIMENT_PLAN.md                # 本文档
└── requirements.txt                  # 依赖列表
```

**★ 标记的文件是核心必须迁移的。**

### 环境依赖

```bash
# requirements.txt
torch>=2.1.0
diffusers>=0.28.0
transformers>=4.38.0
accelerate
safetensors
sentencepiece
scikit-image          # for SLIC segmentation
open-clip-torch       # for CLIP score
Pillow
numpy
matplotlib
tqdm
pytorch-fid           # for FID computation
lpips                 # for LPIPS computation
```

### 模型下载

```bash
# Lumina-Next-T2I (需要 HuggingFace 访问)
# 会在首次运行时自动下载到 ~/.cache/huggingface/
# 模型 ID: Alpha-VLLM/Lumina-Next-SFT-diffusers

# SD3-Medium (需要 HuggingFace 账号 + 同意 license)
# 模型 ID: stabilityai/stable-diffusion-3-medium-diffusers
# 登录方式:
huggingface-cli login  # 输入 access token

# PixArt-α (可选)
# 模型 ID: PixArt-alpha/PixArt-XL-2-1024-MS
```

### 迁移步骤

```bash
# 1. 克隆代码
scp -r runkai@source_machine:/home/runkai/xuanhua/SASD /path/to/target/

# 2. 安装环境
pip install -r requirements.txt

# 3. 验证 GPU
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name())"

# 4. 跑 smoke test 验证
# Lumina
python scripts/run_generate.py --method baseline --num_prompts 3 --num_seeds 1

# SD3
python scripts/run_sd3_smoke.py --dtype fp16 --steps 28
```

---

## 五、TODO 清单

### 🔴 P0 — 投稿必须

- [ ] **Semantic mask 速度优化** ⚡ 当前瓶颈
  - 现状：semantic 8.50s (1.45x) vs cfg_magnitude 7.37s (1.67x)，差 ~1.1s/image
  - 原因 1：每张图重新实例化 `SemanticMaskBuilder`，导致 CLIP 模型重复加载
  - 原因 2：Cross-attention hook 用手动 matmul 替代 SDPA，带来额外开销
  - 原因 3：Anchor embeddings 每张图重新计算（23 条短语）
  - [ ] 把 CLIP 模型和 anchor embeddings 挂载到 `DitWrapper.__init__`，跨 image 复用
  - [ ] 把 `SemanticMaskBuilder` 实例缓存在 wrapper 上，避免每次重建
  - [ ] 验证优化后 semantic 速度是否接近 cfg_magnitude（目标 <7.8s）

- [x] **SD3 代码完成** ✅ 已写完，等解锁
  - [x] `sd3_wrapper.py` — SD3 wrapper
  - [x] `sd3_sparse_processor.py` — SD3 joint attention 稀疏 processor
  - [x] `sd3_block_manager.py` — SD3 block manager
  - [x] `run_sd3_smoke.py` — smoke test
  - [x] `run_sd3_eval.py` — 完整评测脚本
  - [ ] **🔒 解锁 SD3**：访问 huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers 同意 license，然后 `huggingface-cli login`

- [ ] **Lumina 主实验**（需先完成 semantic 优化）
  - [ ] E1: baseline 200p×3s
  - [ ] E2a: SASD cfg_magnitude 200p×3s（s_fg=9.0, s_bg=2.0, skip_ratio=0.5, full_skip_interval=4）
  - [ ] E2b: SASD semantic 200p×3s（同上参数）
  - [ ] 计算 CLIP score, edge density, speedup
  - [ ] 人工检查 10+ 组图像质量（对比 baseline vs semantic vs cfg_magnitude）

- [ ] **消融实验**
  - [ ] E3: 22 个消融配置 × 200p×1s（节约时间用 1 seed）
  - 重点消融维度：mask_type / skip_ratio / s_fg vs s_bg / full_skip_interval / sparse_ffn on/off
  - [ ] 整理成 ablation table（LaTeX 格式）

- [ ] **Baseline 对比**
  - [ ] E4: Delta-DiT 200p×3s
  - [ ] E4: FORA 200p×3s
  - [ ] (可选) ToMe 200p×3s
  - [ ] 统一格式的 comparison table

- [ ] **SD3 实验**（解锁后）
  - [ ] E5: SD3 baseline 200p×3s
  - [ ] E6: SD3 SASD 200p×3s
  - [ ] cross-architecture table（Lumina + SD3 两行）

- [ ] **速度基准**
  - [ ] E8: Lumina 速度（warmup 5, measure 30）
  - [ ] E8: SD3 速度

### 🟡 P1 — 强烈建议

- [ ] **Noise pred 分析**（motivation figure）
  - [ ] E7: 10 prompts，每步记录 fg/bg 变化幅度
  - [ ] 生成折线图：x=denoising step, y=L2 change, 两条曲线（fg/bg）

- [ ] **Mask 可视化 figure**
  - [ ] 选 5-6 个有代表性的 prompt
  - [ ] 并排展示：原图 | cfg_magnitude heatmap | semantic heatmap | 生成结果
  - [ ] 用 matplotlib 生成 figure

- [ ] **CFG 参数消融**（s_fg / s_bg 敏感度）
  - 当前只测了 s_fg=7.0, s_bg=1.0（小规模）
  - [ ] 对比 (7/1), (9/2), (9/1), (12/1) 这 4 组，各 50p×1s
  - [ ] 目标：确定论文中推荐的最优参数

- [ ] **FID 计算**
  - [ ] 生成 5000+ 张图（baseline + SASD semantic + SASD cfg_mag）
  - [ ] pytorch-fid 计算

- [ ] **LPIPS 计算**
  - [ ] baseline vs SASD 逐对比较

### 🟢 P2 — 可选增强

- [ ] PixArt-α 第3个架构验证
- [ ] PartiPrompts 大规模评测（10K prompts）
- [ ] SASD + DeepCache 叠加实验
- [ ] SASD + ToMe 叠加实验
- [ ] 不同步数（20/25/30/50）的效果对比
- [ ] 不同分辨率（512/768/1024）的速度曲线

---

## 六、两台机器分工建议

**机器 A (RTX 5090 32GB):**
- SD3 实验 (E5, E6) — 需要较大显存
- SDXL 实验 (如果需要)
- 速度基准 (E8)

**机器 B (另一台):**
- Lumina 主实验 (E1, E2)
- 消融实验 (E3)
- Baseline 对比 (E4)
- Noise pred 分析 (E7)

**并行策略：** 两台机器同时跑不同模型的实验，最终汇总结果。

---

## 七、结果汇总格式

每个实验输出到 `outputs/<experiment_name>/`：
```
outputs/lumina_eval/baseline/
├── images/           # p0000_s0000.png ~ p0199_s0002.png
├── logs.jsonl        # 每张图的 prompt, seed, time
├── metrics.csv       # CLIP score, edge density per image
└── summary.json      # 聚合统计 (mean CLIP, mean edge density, avg time)
```

最终汇总表格：
```
outputs/summary_table.csv
method, model, clip_mean, clip_std, edge_mean, fid, lpips, speedup, time_per_img
```
