# SASD Mask Methods 详细文档

本文档详细介绍 SASD 项目中所有 mask 构建方法的原理、流程和实现细节。

---

## 总览

| Mask 类型 | 重要性信号来源 | 区域划分方式 | 额外模型开销 | 加速比 | 质量损失 (dCLIP) |
|-----------|---------------|-------------|-------------|--------|-----------------|
| **Complexity** | Scharr 梯度 | 固定网格 (8x8) | 无 | ~2.4x | -0.012 |
| **Semantic** | CLIP anchors + cross-attention | 隐式热力图 | CLIP ViT-L/14 (~340MB) | ~2.0x | -0.008 |
| **Region** | Scharr 梯度 | SLIC 超像素 | 无 | ~2.4x | — |
| **SDiT** | 边缘 + Laplacian + 残差噪声 | SLIC 超像素 | 无 | ~2.4x | -0.019 |
| **CFG Magnitude** | \|cond_pred - uncond_pred\| | SLIC 超像素 | 无 | ~2.4x | **-0.005** |

所有 mask 最终产出 `(H, W)` 的 float tensor（值域 [0, 1]），1=前景，0=背景。

---

## 1. Complexity Mask (网格复杂度)

**文件**: `src/sparse/mask_builders.py` — `ComplexityMaskBuilder`

### 核心思想

用 Scharr 梯度算子检测 latent 空间中纹理复杂的区域。纹理越复杂 → 越可能是前景主体 → 需要更多计算资源。

### 详细流程

```
输入: latents (1, C, 128, 128)
     ↓
[Step 1] Scharr 梯度计算
  - 跨 channel 平均: gray = latents.mean(dim=1)  → (128, 128)
  - Scharr-x 卷积核: [[-3,0,3],[-10,0,10],[-3,0,3]]
  - Scharr-y 卷积核: [[-3,-10,-3],[0,0,0],[3,10,3]]
  - gx = conv2d(gray, scharr_x)
  - gy = conv2d(gray, scharr_y)
  - complexity = sqrt(gx² + gy²)  → (128, 128)
     ↓
[Step 2] 固定网格划分
  - 将 128x128 划分为 grid_size x grid_size = 8x8 个 block
  - 每个 block 大小 = 16x16 像素
     ↓
[Step 3] 区域评分
  - 对每个 block，取其中 top 75% 复杂度像素的均值作为得分
  - 用 top-q 而非全部均值，避免边缘低复杂度像素拉低得分
     ↓
[Step 4] 前景选择
  - 按得分降序排列所有 block
  - 选得分最高的前 ratio (默认 25%) 的 block 作为前景
     ↓
输出: mask (8, 8) → 上采样到 (128, 128)
```

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `grid_size` | 8 | 网格划分数 (8x8 = 64 blocks) |
| `top_q` | 0.75 | 区域内取 top 75% 像素评分 |
| `ratio` | 0.25 | 前 25% 的 block 为前景 |

### 优缺点

- **优点**: 实现简单，无额外开销
- **缺点**: 网格边界是人为的，不跟随语义边界；检测的是「纹理复杂度」而非「语义重要性」，复杂纹理的背景（如树叶、砖墙）也会被标为前景

---

## 2. Semantic Mask (语义 mask)

**文件**: `src/sparse/mask_builders.py` — `SemanticMaskBuilder`

### 核心思想

结合两个信号：
1. **CLIP 文本编码器**：找出 prompt 中哪些词是「美学/质量相关」的（重点词）
2. **Cross-attention 热力图**：看这些重点词在图像空间中 attend 到了哪些位置

两者结合 → 「prompt 中美学关键词重点关注的图像区域」= 前景。

### 详细流程

```
输入: prompt, cross_attention_maps, tokenizer
     ↓
[Phase 1] CLIP 文本编码
  - 加载 openai/clip-vit-large-patch14 (~340MB, 懒加载)
  - 将 prompt 分词 (Gemma SentencePiece)
  - 用 leading-space heuristic 将 subword 合并为 whole words
    例: "▁photo" "▁of" "▁a" "▁beau" "tiful" → ["photo", "of", "a", "beautiful"]
  - 每个 whole word 单独送入 CLIP text encoder 得到 embedding
     ↓
[Phase 2] 锚点匹配 → token importance
  - 23 个美学锚点 (从 Pick-a-Pic 37K prompts 数据驱动):
    5 大类:
      质量修饰: detailed(4634), beautiful(2321), realistic(2137), intricate(1353), ...
      技术参数: sharp focus(601), high resolution(475), 8k(390), ...
      艺术风格: digital art(1027), oil painting(535), concept art(470), ...
      人物构图: portrait(1992), full body(549), close up(469), ...
      内容主体: main subject(新增), character(新增), object(新增), ...
  - 对每个 word，计算与所有 anchor 的 cosine similarity
  - 取最大 similarity 作为该 word 的 importance
  - 阈值判断: 如果 max similarity < 0.60 → importance = 0 (过滤虚词噪声)
    例: "of" vs "main subject" = 0.52 < 0.60 → 过滤
        "detailed" vs "detailed" = 1.0 > 0.60 → 保留
  - 47 个停用词 (a, an, the, of, in, ...) 强制 importance = 0
  - 将 word-level importance 映射回 Gemma token positions
     ↓
[Phase 3] Cross-attention 加权聚合
  - 输入: 24 层 transformer 的 cross-attention maps
    每层: (32 heads, 4096 patches, seq_len) → 只取 conditional branch
  - 跨层平均 → (32, 4096, seq_len)
  - 跨 head 平均 → (4096, seq_len)
  - 用 token importance 加权: heatmap = attn @ importance
    → (4096,) 的 per-token 重要性分数
  - reshape 到 (64, 64) 的空间热力图
     ↓
[Phase 4] 热力图质量检查
  - 检查 heatmap 的 std > 0.12 (确保有空间结构)
  - 如果 std 太小 → 说明没有清晰的前景/背景分离 → 返回 None (fallback 到全 dense)
     ↓
[Phase 5] 二值化 + 平滑
  - 按 ratio (默认 0.5) 做阈值: 取 top 50% 像素为前景
  - Gaussian blur (sigma=1.0) 平滑边界
     ↓
输出: mask (64, 64) → 上采样到 (128, 128)
```

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `ratio` | 0.5 | 前景占比 50% |
| `blur_sigma` | 1.0 | 边界平滑 sigma |
| `confidence_threshold` | 0.60 | CLIP 相似度阈值 |
| `heatmap_std_threshold` | 0.12 | 热力图质量检查 |

### 优缺点

- **优点**: 真正基于语义的 mask，理解 prompt 含义；美学关键词权重高，纯内容词权重低
- **缺点**: 需要额外加载 CLIP 模型 (~340MB)；需要替换 SDPA 为手动 Q@K^T 来捕获 attention weights（SDPA 不返回 attention weights）；整体速度最慢 (2.04x vs 2.4x)
- **覆盖率**: ~20% 的 prompt 触发锚点加权，~80% fallback 到纯 cross-attention

---

## 3. Region Mask (SLIC 区域复杂度)

**文件**: `src/sparse/region_mask_builder.py` — `RegionMaskBuilder`

### 核心思想

是 Complexity Mask 的升级版：用 SLIC 超像素替代固定网格，让区域边界更贴合语义。

### 详细流程

```
输入: latents (1, C, 128, 128)
     ↓
[Step 1] Scharr 梯度计算 (同 Complexity)
  - complexity = sqrt(gx² + gy²)  → (128, 128)
     ↓
[Step 2] SLIC 超像素分割
  - 将 complexity map 归一化到 [0, 1]
  - SLIC 算法:
    · 初始化 64 个均匀分布的聚类中心
    · 在 5D 空间 (x, y, intensity) 迭代聚类
    · compactness=10 控制空间 vs 强度的权重
  - 输出: labels (128, 128) 每个像素的区域 ID
     ↓
[Step 3] 区域评分
  - 每个区域: top 75% 像素的 complexity 均值
     ↓
[Step 4] 前景选择
  - 得分最高的前 25% 区域 → 前景
     ↓
[Step 5] 边界平滑
  - 形态学膨胀 (radius=2): max_pool2d 扩展前景边界 2 像素
  - Gaussian blur (sigma=1.5): 平滑 0/1 边界为 [0,1] 渐变
     ↓
输出: soft mask (128, 128)
```

### vs Complexity Mask 的区别

| 对比 | Complexity (网格) | Region (SLIC) |
|------|------------------|---------------|
| 区域划分 | 固定 8x8 方格 | 自适应超像素 |
| 边界质量 | 人为方块切割 | 贴合纹理边界 |
| 边界处理 | 无 | 膨胀 + 高斯模糊 |
| 区域数量 | 64 (8x8) | ~64 (可配置) |

---

## 4. SDiT Mask (梯度复杂度 + SLIC)

**文件**: `src/sparse/sdit_mask_builder.py` — `SDiTMaskBuilder`

### 核心思想

参考 SDiT 论文的方法：利用 denoising 过程中的信息估计 predicted clean latent，在其上计算多维度复杂度（边缘 + 拉普拉斯 + 残差噪声），结合 SLIC 分割。

### 详细流程

```
输入: latents x_t (1, C, 128, 128),
      noise_pred (1, C, 128, 128) (已取反),
      sigma (当前噪声水平)
     ↓
[Step 1] 估计 clean latent
  - x_pred = x_t + sigma * noise_pred
    (noise_pred 已取反，所以 + 号)
  - gray = x_pred 跨 channel 平均 → (1, 1, 128, 128)
     ↓
[Step 2] 三维度复杂度计算

  (a) 边缘强度 (Scharr, alpha=1.0):
    - gx = conv2d(gray, scharr_x)
    - gy = conv2d(gray, scharr_y)
    - edge = sqrt(gx² + gy²)

  (b) 拉普拉斯 (beta=0.5):
    - kernel = [[0,1,0],[1,-4,1],[0,1,0]]
    - lap = |conv2d(gray, laplacian_kernel)|
    - 检测亮度突变点（物体边界）

  (c) 残差噪声幅度 (gamma=0.3):
    - residual = |x_t - x_pred| 跨 channel 平均
    - 噪声残留大 → 该区域预测不确定 → 需要更多计算

  - 各自归一化到 [0, 1]
  - complexity = 1.0*edge + 0.5*lap + 0.3*residual
     ↓
[Step 3] SLIC 分割 + 评分 + 选择
  - (同 CFG Magnitude，64 superpixels, top-75%, ratio=0.5)
     ↓
[Step 4] Gaussian blur 平滑 (sigma=1.5)
     ↓
输出: soft mask (128, 128)
```

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `alpha` | 1.0 | 边缘项权重 |
| `beta` | 0.5 | 拉普拉斯项权重 |
| `gamma` | 0.3 | 残差噪声项权重 |
| `n_segments` | 64 | SLIC 目标区域数 |
| `compactness` | 10.0 | SLIC 空间紧凑度 |
| `ratio` | 0.5 | 前景占比 |

### 为什么效果最差

实测 dCLIP = -0.019，IR = -0.704。原因：

1. **纹理复杂度 ≠ 语义重要性**。砖墙、草地等复杂纹理的背景也会被标为前景
2. **残差噪声在早期步骤 (mask_step=5) 还不够 discriminative**，信噪比低
3. 三个分量的线性组合权重是手动设定的，不同 prompt 的最优权重不同

---

## 5. CFG Magnitude Mask (CFG 差异 + SLIC)

**文件**: `src/sparse/sdit_mask_builder.py` — `CFGMagnitudeMaskBuilder`

### 核心思想

Classifier-Free Guidance 每步计算两个分支：
- `cond_pred`: 有文字引导的噪声预测
- `uncond_pred`: 无文字引导的噪声预测

差异 `|cond_pred - uncond_pred|` 直接表示 **prompt 对该区域的影响程度**：
- 差异大 → prompt 在重点描述这个区域 → 前景
- 差异小 → prompt 不关心这个区域 → 背景

这个信号完全免费：CFG 本身就需要计算两个分支。

### 详细流程

```
输入: cond_pred (1, 3, 128, 128)     ← 条件分支 noise prediction (eps)
      uncond_pred (1, 3, 128, 128)   ← 无条件分支 noise prediction (eps)
     ↓
[Step 1] 逐像素 CFG 差异
  - magnitude = |cond_pred - uncond_pred|.mean(dim=[0,1])
  - 跨 batch 和 channel 平均 → (128, 128)
  - 值越大 = 该 latent 位置越受 prompt 影响
     ↓
[Step 2] SLIC 超像素分割
  - 归一化 magnitude map 到 [0, 1]
  - SLIC(n_segments=64, compactness=10.0)
  - 产出 ~64 个自适应区域，边界贴合 magnitude 的梯度
     ↓
[Step 3] 区域评分
  - 每个区域: 取 top 75% 像素的 magnitude 均值
  - 用 top-q 而非均值，避免区域边缘混入的低 magnitude 像素拉低得分
     ↓
[Step 4] 前景选择
  - 按得分降序排列
  - 取前 50% 区域 (ratio=0.5) 为前景
  - 生成二值 mask: 前景=1, 背景=0
     ↓
[Step 5] Gaussian blur 边界平滑
  - sigma=1.5
  - 硬 0/1 边界 → [0, 1] 渐变过渡带
  - 避免前景/背景接缝处出现伪影
     ↓
输出: soft mask (128, 128)
```

### 在生成循环中的集成

```python
# 在 mask_step (默认第5步) 时触发:

# 1. Transformer forward pass (dense mode)
noise_pred = transformer(latent_input, timestep, ...)

# 2. 分离 pred 和 sigma
noise_pred = noise_pred.chunk(2, dim=1)[0]

# 3. CFG 分支拆分
noise_pred_eps = noise_pred[:, :3]
cond_eps, uncond_eps = split(noise_pred_eps, batch_dim)

# 4. 在这里构建 mask (CFG 分支拆分后、CFG 合并前)
spatial_mask = cfg_magnitude_builder.build_mask(cond_eps, uncond_eps)

# 5. 转换为 token mask + spatial CFG map
fg_token_mask = (spatial_mask > 0.5)     # (4096,) bool
s_map = s_bg + spatial_mask * (s_fg - s_bg)  # 前景 CFG=7.0, 背景 CFG=1.0

# 6. 安装 sparse block manager
block_mgr.install(fg_mask=fg_token_mask, sparse_ffn=True)

# 后续步骤:
#   - mask_step+1: 切换到 sparse mode
#   - 前景 token: 完整计算 attn1 SDPA + SwiGLU FFN
#   - 背景 token: 复用上一步的 cached 输出
#   - 每 2 步跳 1 步 (线性外推 noise_pred)
```

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `n_segments` | 64 | SLIC 目标区域数 |
| `compactness` | 10.0 | SLIC 空间紧凑度 (越大区域越规则) |
| `ratio` | 0.5 | 前景占总区域的比例 |
| `top_q` | 0.75 | 区域内取 top 75% 像素评分 |
| `blur_sigma` | 1.5 | 边界 Gaussian 平滑 sigma |

### 为什么效果最好

| 对比对象 | CFG Magnitude 的优势 |
|---------|---------------------|
| Complexity/Region (梯度) | CFG magnitude 反映「prompt 关注度」，梯度只反映「纹理复杂度」。复杂纹理的背景不会被误标为前景 |
| SDiT (多维复杂度) | 单一信号更稳定，不需要手动调 alpha/beta/gamma 权重 |
| Semantic (CLIP + cross-attn) | 不需要额外加载 CLIP 模型 (~340MB)；不需要替换 SDPA 捕获 attention weights；计算零开销 |
| 所有网格方法 | SLIC 超像素贴合语义边界，不会方块切割主体 |

---

## 6. 辅助组件

### 6.1 SLIC 超像素分割

所有 SLIC-based 方法共享同一套参数逻辑：

```
SLIC 算法 (Simple Linear Iterative Clustering):
  1. 在图像上均匀撒 K 个初始中心点
  2. 对每个像素，在 5D 空间 (x, y, intensity) 中找最近的中心
  3. 迭代更新中心位置
  4. 收敛后输出每个像素的区域标签

compactness 参数:
  - 控制空间距离 vs 强度差异的相对权重
  - 大 (>20): 区域接近正方形，空间上紧凑
  - 小 (<5): 区域形状不规则，完全跟随强度边界
  - 默认 10: 平衡
```

### 6.2 边界平滑 (boundary_ops.py)

```
soft_mask_pipeline(binary_mask, dilation_radius, blur_sigma):

  Step 1: 形态学膨胀
    - 用 max_pool2d(kernel=2r+1) 扩展前景边界
    - 效果: 前景区域向外生长 r 像素
    - 目的: 确保前景边缘不被切到

  Step 2: Gaussian blur
    - kernel_size = 6σ + 1 (自适应)
    - padding = reflect mode (边缘镜像)
    - 效果: 硬 0/1 边界 → [0, 1] 渐变
    - 目的: spatial CFG 和 sparse attention 在边界处平滑过渡
```

### 6.3 Cross-Attention 捕获 (cross_attn_hook.py)

仅 Semantic mask 使用：

```
问题: Lumina 用 F.scaled_dot_product_attention (SDPA)，不返回 attention weights
解决: 临时替换 attn2 processor，手动计算 Q@K^T

CrossAttnCaptureProcessor:
  1. 接收 hidden_states (图像 tokens) 和 encoder_states (文本 tokens)
  2. 计算 Q = linear(hidden), K = linear(encoder), V = linear(encoder)
  3. GQA 展开: 32 query heads, 8 KV heads → 每个 KV head 被 4 个 Q head 共享
  4. 手动 attention: weights = softmax(Q @ K^T / sqrt(d))
  5. 只存 conditional branch (CFG batch 的前半)
  6. 输出 attention map: (32 heads, 4096 patches, seq_len)
```

---

## 7. 实验结果对比

20 prompts x 3 seeds = 60 images/config，全部使用 sparse attn1 + sparse FFN + skip_interval=2：

```
Config                          Time  Speedup   CLIP    dCLIP   Pick    dPick     IR     dIR    Edge
─────────────────────────────────────────────────────────────────────────────────────────────────────
Baseline (dense, CFG=4.0)      12.60s  1.00x  0.2531  +0.000  0.2185  +0.000  0.752  +0.000  0.583
Semantic (cross-attn + CLIP)    6.19s  2.04x  0.2449  -0.008  0.2095  -0.009  0.428  -0.324  0.514
SDiT (complexity + SLIC)        5.25s  2.40x  0.2347  -0.019  0.2020  -0.017  0.048  -0.704  0.419
CFG magnitude + SLIC            5.25s  2.40x  0.2487  -0.005  0.2100  -0.009  0.465  -0.287  0.521
```

**结论**: CFG Magnitude 在所有指标上都是最优的 mask 策略 — 最小的质量损失 + 最快的速度 + 零额外计算开销。
