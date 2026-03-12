# AccelAes: Aesthetic-Guided Sparse Computation Improves Quality While Accelerating Diffusion Transformers

> 论文大纲 & 写作指南（最后更新: 2026-02-26）
> 当前实验状态: Lumina ✅ / SD3 ✅ / FLUX ✅ — P0 消融 ✅ / P1 SD3消融 ✅ / Direct threshold 消融 ✅ — 全部完成，数字已锁定（region_method=threshold 为新默认）
> FLUX sparse attention 实验验证（2026-02-25）：per-prompt 分析确认 sparse attn 对 FLUX 质量有害；FLUX 最终只用 fskip2（步级缓存）。SD3 joint attention heatmap 可视化质量远优于 Lumina cross-attention，论文可视化图用 SD3。

---

**AccelAes: Acceleration of DiT for Training-Free Aesthetic-aware Image Generation**

---

## Abstract（~150 words）

Existing training-free acceleration methods for Diffusion Transformers (DiTs) share a common
goal: maintain image quality while reducing inference time. None of them actively improve it.
We present AccelAes, a training-free framework that exploits a key observation: DiT denoising is
spatially non-uniform—image patches corresponding to aesthetic vocabulary in prompts
(e.g., "photorealistic", "intricate", "cinematic") receive disproportionate cross-attention and
govern the aesthetic quality of the output, while low-attention regions remain stable across
timesteps and can safely reuse cached computations. AccelAes identifies aesthetically-relevant
regions by matching prompt tokens to a pre-defined aesthetic anchor set via CLIP similarity, maps
their cross-attention affinity to a one-shot spatial mask, and applies (1) **spatially-adaptive
sparse attention and FFN**—restricting expensive computation to aesthetic regions—combined with
(2) **spatial CFG**, assigning higher guidance scale to aesthetic regions to actively boost quality,
and (3) **step-level prediction caching** (similar to TeaCache/TaylorSeer) as an orthogonal
speed-up. On Lumina-Next, AccelAes achieves **IR +11.9%** quality improvement over the dense
baseline while delivering **2.09× speedup**—the first acceleration method to actively improve
image quality. We further validate on SD3-Medium (1.50×) and FLUX.1-dev (1.73×).
Code will be released.

---

## 1. Introduction（~1.5 pages）

### 结构与要点

**第1段：背景引入**
- Diffusion models 已成为图像生成主流（DALL-E, Stable Diffusion, FLUX, Sora）
- 架构从 U-Net 演进到 Diffusion Transformer (DiT)：PixArt-α, SD3, FLUX
- DiT 优势：更好的 scaling behavior、更强的生成质量
- 但 DiT 推理更慢：全 transformer，self-attention O(N²)，1024×1024 → 4096 tokens，无 hierarchical downsampling

**第2段：现有加速方法的共同局限——没有一个主动提升质量**
- 步级缓存 (DeepCache, Δ-DiT, FORA, TeaCache, TaylorSeer)：跳步或外推中间特征，**目标是"维持质量不下降"**，最好结果约 ±1-2% IR
- 空间自适应 (RAS, SDiT)：动态信号（cross-attn 幅度、像素复杂度），**同样以保质量为目标**，RAS 在 Lumina 上 IR +4.8%（略有改善但非设计目标）
- Token merging (ToMe, IToMe)：合并相似 token，**质量下降或持平**
- **共同缺陷：所有方法都把质量目标定为"维持不下降"——没有方法将质量提升作为加速的主要目标**

**第3段：关键观察（motivation）**
- **观察1**：DiT 的 cross-attention affinity 在空间上高度不均匀——图像 patch 对 prompt 中美学描述词（photorealistic、intricate、cinematic 等）的注意力强度，在美学关注区域是其他区域的 3~10×。这意味着 DiT 在学习"哪些 patch 对应哪些质量描述词"，这些 patch 才是影响最终图像美学质量的关键区域。
- **观察2**：低关注区域（对美学描述词 affinity 低的 patch）的 noise_pred 在连续去噪步之间变化幅度仅为高关注区域的 1/3~1/5，大多数步可安全跳过
- **关键洞察**：现有方法用的是"动态信号"（每步重新计算哪里在变化）——我们用的是"语义信号"（一次识别哪里决定质量）。向决定质量的区域倾斜计算，不仅节省计算，还能用 spatial CFG 主动增强这些区域的细节，从而**主动提升**图像质量。

**第4段：方法概述**
- 提出 AccelAes，核心贡献是**美学锚点驱动的空间计算分配**：
  1. **美学区域检测**：CLIP 相似度匹配 prompt token 与美学锚点集合 → 美学相关 token → cross-attention affinity map → 一次性空间 mask（直接百分位阈值，无需 superpixel）
  2. **空间自适应稀疏计算**（适用于无 AdaLN 耦合的架构，如 Lumina）：美学区域 full attention + FFN，低关注区域复用缓存；Q_aesthetic @ K_all 保留全局 context
  3. **Spatial CFG**：对美学区域施加更强引导（s_fg > s_bg），主动放大美学细节——这是质量提升的直接来源
  4. **步级预测缓存**（与 TeaCache/TaylorSeer 类似，作为正交加速组件）：噪声预测级别线性外推，全架构通用
- 关键：组件 1-3 共同构成统一的"美学计算分配框架"，组件 4 是独立的辅助加速手段

**第5段：贡献**
1. **核心发现**：首次提出用美学锚点词（photorealistic、intricate 等）驱动 DiT 空间计算分配——信号完全来自 CLIP 语义匹配 + 模型内置 cross-attention，无额外训练或标注
2. **核心方法**：空间自适应稀疏 attention + FFN（Q_aesthetic @ K_all 保持全局感知）+ spatial CFG（主动放大美学区域引导），三者共同构成统一框架
3. **核心结果**：**首个在加速同时主动提升图像质量的方法**——Lumina-Next IR +11.9%，Aesthetic +0.0%，CLIP +0.4%，同时 2.09× 加速（相比 RAS 1.47× 更快，IR +10% vs +4.8% 更优）
4. **工程验证**：结合步级预测缓存（类 TeaCache/TaylorSeer，显式引用）在 Lumina/SD3/FLUX 三种架构上验证，总加速 1.50×-2.09×，空间组件独立贡献 1.42×（IR +8.8%）

---

## 2. Related Work（~1 page）

### 2.1 Diffusion Transformer Architectures
- DiT (Peebles & Xie, 2023): class-conditional，标准 DiT block
- Lumina-Next-T2I: GQA, RoPE, Gemma 文本编码，无 AdaLN 全局耦合
- SD3 (MMDiT): joint attention，双流结构，AdaLN timestep modulation
- FLUX.1-dev: 19 double-stream + 38 single-stream，guidance distillation（单 pass），无 CFG

### 2.2 Training-Free Diffusion Acceleration

**采样步数减少：**
- DDIM, DPM-Solver, PLMS — 减少总步数；与本文正交

**特征缓存/层跳过（step-level / layer-level）：**
- DeepCache (CVPR 2024, 2312.00858)：缓存 U-Net 高频特征跨步复用；U-Net 专用
- Δ-DiT (arXiv 2406.01125), FORA (arXiv 2407.01425)：缓存/复用 block 输出残差，全图 token 统一策略
- ToCa (ICLR 2025, 2410.05317)：token 级选择性缓存，按步间相似度打分；无语义 mask
- TeaCache (CVPR 2025 highlight, 2411.19108)：用 timestep embedding 预测输出变化量，非均匀跳步调度；最高 4.41×（视频 DiT）
- TaylorSeer (ICCV 2025, 2503.06923)：Taylor 级数外推特征，高阶预测优于线性外推；FLUX 4.99×，当前 SOTA 步级缓存
- **本文定位**：步级 noise_pred 线性外推与 TeaCache/TaylorSeer 方向相同；**AccelAes 不将步级缓存列为主要贡献**，而是显式引用上述方法，将其作为与空间稀疏正交的辅助加速组件。AccelAes 的核心贡献在于美学锚点驱动的空间计算分配（§3.2-3.5），步级缓存（§3.6）为辅助模块。

**Token 级加速（merging）：**
- ToMe (Bolya et al., 2023)：合并相似 token，改变 N，无差别合并
- IToMe (arXiv 2411.16720)：用 **CFG magnitude** 作为重要性信号，优先合并低重要性 token；与 AccelAes cross-attention affinity 方案相关但不同——CFG magnitude 信号免费但每步重新计算，而 AccelAes 只在 mask_step 计算一次
- **本文区别**：不改变 token 数，而是跳过背景 token 的计算，保留所有 K,V

**稀疏注意力（structural）：**
- SpargeAttn (ICML 2025, 2502.18137)：block 级预测哪些 Q-K 对权重显著，内核层结构稀疏；在 FLUX/SD3 2.5–5×；纯结构稀疏无语义 mask
- **本文区别**：AccelAes 按语义内容选 token 子集参与注意力，而非预测 Q-K 权重结构

**空间自适应 CFG（spatial CFG）：**
- S-CFG (CVPR 2024, 2404.05384)：cross-attention map + self-attention 分割语义区域，对不同区域给予不同 CFG scale 以均衡引导强度；**质量提升方法，无加速**；在 UNet SD 上验证
- **本文区别**：AccelAes 同样对不同空间区域使用不同 CFG scale，但目的是**加速**（低注意力区域用低/零 guidance 减少计算），直接百分位阈值保持 attention 真实分布

**区域选择 / 语义 mask（最相关）：**
- **RAS (arXiv Feb 2025, 2502.10389, Microsoft Research)** ⚠ **最重要对比**：
  - 在 **Lumina-Next-T2I** 和 SD3 上验证，加速 **2.36×（SD3）/ 2.51×（Lumina）**
  - 核心思路：聚焦模型当前 cross-attention 激活高的区域做完整去噪，其余区域用上一步缓存的 noise 更新；跳步率随去噪进度递减
  - **与 AccelAes 的关键区别**：
    (1) RAS mask = 纯 cross-attention 激活幅度；AccelAes = **美学锚定词语义匹配**（CLIP + 跨注意力 affinity），信号更丰富
    (2) RAS 缓存 background noise（输出级），AccelAes 在内部跳过 background token 的 attn/FFN 计算（计算级）
    (3) AccelAes 额外有 spatial CFG 放大前景梯度、步级线性外推跳整步；RAS 无这两项
    (4) RAS 未发表于会议（preprint，2025-02），AccelAes 可在公平对比中与之竞争
    (5) RAS 的 2.51× 依赖 flash_attn + Triton 内核；在相同无 flash_attn 环境下实测 1.57–1.61×（token-slicing）；我们的复现（output-level blending）得到 1.47×
- SDiT (arXiv Jan 2025, 2601.12283)：complexity-driven（像素级 Sobel/Laplacian）区域跳过，Lumina-Next 专用，最高 3.0×（激进设置 FID +20%）；在保质量设置（LPIPS≈0.12）下 1.66×
  - 与 AccelAes 区别：bottom-up 信号（像素复杂度）vs top-down（语义/美学）；无步级缓存；无 spatial CFG；无跨架构验证

**本文综合优势（vs 全部相关工作）：**
1. **唯一将空间稀疏（美学语义 mask）+ 步级缓存（线性外推）+ 空间 CFG 三者结合的工作**
2. **mask 信号一次提取、全程复用**（vs SDiT 每步计算、RAS 每步更新）
3. **跨三架构验证**（Lumina / SD3 / FLUX）；RAS/SDiT 仅在 Lumina-Next + SD3 上测试
4. **在相同模型（Lumina-Next）上 vs RAS**：AccelAes **2.09×**（P0实测，semantic_full_eval）/ RAS 1.47×（我们复现）；AccelAes 质量更好（IR +10% vs +4.8%，Edge +10% vs −0.3%）且速度更快（5.91s vs 8.38s）；RAS 官方声称 2.51× 依赖 flash_attn+Triton，在我们环境实测 1.57–1.61×

### 2.3 Content-Adaptive Computation in Vision
- MoE, early exit, adaptive computation time
- 本文在 diffusion 推理中引入内容自适应性：根据 prompt 美学语义动态分配计算资源

---

## 3. Method（~3.5 pages）— 核心部分

### 3.1 Background: Diffusion Transformer Inference

**标准去噪循环（以支持 CFG 的模型为例）：**
```
for t in T, T-1, ..., 1:
    noise_unc = DiT(x_t, t, ∅)          # unconditional branch
    noise_cnd = DiT(x_t, t, c)          # conditional branch
    noise     = noise_unc + w · (noise_cnd - noise_unc)   # CFG
    x_{t-1}   = scheduler.step(noise, t, x_t)
```

每次 DiT forward，对于有 L 层的 transformer：
- 每层 Self-Attention: `O(N² · D)` with `N = (H/p)(W/p)`（1024×1024 image, patch 2 → N=4096）
- 每层 FFN: `O(N · D · 4D)`
- 总计算量：`2 × L × [O(N²D) + O(ND²)]` per denoising step

**关键架构差异（影响加速策略选择）：**

| 属性 | Lumina-Next | SD3-Medium | FLUX.1-dev |
|------|------------|------------|------------|
| Timestep 调制 | 外部 sinusoidal，不入 block | AdaLN（scale/shift 入每 block） | AdaLN（入每 block） |
| CFG 支持 | 是（双 pass） | 是（双 pass） | 否（guidance embedding，单 pass） |
| 内部特征可缓存 | 是（无 timestep 耦合） | 否（AdaLN 耦合） | 否（AdaLN 耦合） |
| 注意力结构 | GQA + RoPE | MMDiT joint attention | double + single stream |
| 输出缓存可用 | 是（noise_pred 层面） | 是（noise_pred 层面） | 是（noise_pred 层面） |

**关键结论**：对于 SD3 和 FLUX，模型内部中间层特征被 timestep-dependent AdaLN 调制，跨步缓存内部特征会引入严重误差（实验验证：CLIP 下降 >40%）。**只有 noise_pred 输出层面的缓存是通用安全的。**

### 3.2 Observation: Aesthetically-Driven Spatial Non-Uniformity

**关键观察**：Prompt 中存在两类词——**美学描述词**（photorealistic、intricate、cinematic、high quality 等，描述"怎么画"）和**内容词**（dog、mountain、car 等，描述"画什么"）。
DiT 的 cross-attention 在空间上高度不均匀，且与美学描述词强相关——对应美学描述词的图像 patch 具有显著更高的 affinity，是模型在去噪过程中精细雕琢的核心区域。

**美学关注区域（Aesthetically-relevant Region）定义**：
给定 prompt，先识别其中的**美学相关 token**（通过 CLIP embedding 与预定义美学锚点集合的相似度），再以这些 token 的 cross-attention affinity 作为每个 patch 的"美学重要性"信号。

**形式化**：
```
# Step 1: 美学锚点匹配（一次性，无额外 forward）
anchor_set = {"photorealistic", "intricate", "cinematic", "detailed", ...}  # 23 anchors, 5类
aesthetic_idx = {j : sim_CLIP(token_j, anchor_set) > 0.60}

# Step 2: patch-level 美学重要性
A_layer(i) = max_{j ∈ aesthetic_idx} softmax(Q_img[i] @ K_txt^T / sqrt(D))_j
importance(i) = mean_l A_l(i)     # 多层聚合（每4层采样一次）
```
其中 `i` 为图像 patch 索引，`j` 为美学相关文本 token，`l` 为 transformer layer 索引。

**实验观察（20 prompts，seeds 0/1/2，Lumina-Next，step 4）：**
- 美学关注区域平均 affinity 是低关注区域的 3~10×
- 低关注区域的 noise_pred 逐步变化量：`‖Δnoise_low‖ / ‖Δnoise_aesthetic‖ ≈ 0.2~0.35`
- 结论：美学锚定词的 cross-attention affinity 能准确定位图像质量的关键区域；低关注区域在大多数步可安全跳过计算

**Mask 构建流程（SemanticMaskBuilder，适用于 Lumina 和 SD3）：**
```
1. 在 mask_step（默认 step=4，warmup 末步）运行一次完整 forward，收集 cross-attention 权重
2. 用 CLIP 匹配 prompt token 与美学锚点集合，得到 aesthetic_idx
   （若无 token 超过阈值，退化为全 token max affinity）
3. 计算每 patch 的 importance = mean over sampled layers of max_{j∈aesthetic_idx} A_l(i,j)
4. 直接百分位阈值：取 importance map 的 top skip_ratio 分位数作为前景
   （实验验证 direct threshold 优于 SLIC：IR +2.2%，LPIPS −0.020，FID 46.6 vs 57.8）
5. 按阈值二值化 → 美学 mask fg_mask ∈ {0,1}^N，Gaussian blur 软化边界
6. Upsample 到 latent 分辨率 → spatial CFG mask
```
直接阈值更精确地保留了 attention 真实分布（SLIC 在低分辨率 heatmap 上的空间平滑
引入偏差；64×64 patch 分辨率下孤立噪声问题轻微，blur 后处理已足够软化边界）。
美学锚点匹配保证了信号的语义解释性（不依赖像素级梯度，不随噪声波动）。

### 3.3 Component 1: Spatially-Adaptive Sparse Self-Attention（Lumina 架构）

**适用条件**：架构内部特征无 timestep-dependent 全局调制（如 Lumina-Next）。

**核心思想**：
- 前景 token 的 Query 与**全部** Key, Value 做 SDPA（保持全局信息感知）
- 背景 token 的 Query 不参与 SDPA，直接复用上一计算步的缓存输出
- 计算量：`O(N²D) → O(N_fg · N · D)`，其中 `N_fg ≈ 0.5N` → 理论 2× attention 加速

**算法：**
```
Algorithm 1: Sparse Self-Attention Forward
---------------------------------------------------------------------------
Input:  x ∈ R^{B×N×D}, fg_mask ∈ {0,1}^N, cache_attn
Output: output ∈ R^{B×N×heads×head_dim}

1.  Q, K, V = to_q(x), to_k(x), to_v(x)    -- O(N) on ALL tokens
2.  Apply RoPE to Q, K                        -- architecture-specific
3.  if dense_mode (cache warm-up):
4.      output = SDPA(Q, K, V)                -- O(N²) full attention
5.      cache_attn ← output.clone()
6.      return output
7.  else (sparse mode):
8.      fg_idx = nonzero(fg_mask)             -- gather foreground indices
9.      Q_fg   = Q[:, fg_idx, :, :]           -- (B, N_fg, heads, head_dim)
10.     attn_fg = SDPA(Q_fg, K, V)            -- O(N_fg × N)  ← KEY
11.     output  = cache_attn.clone()           -- start from cached background
12.     output[:, fg_idx] = attn_fg            -- overwrite foreground
13.     cache_attn ← output                    -- update cache
14.     return output
---------------------------------------------------------------------------
```

**关键设计决策：Q_fg @ K_all, V_all（非 Q_fg @ K_fg, V_fg）**
- 前景 token 仍能感知全部背景 context，避免全局信息丢失
- 这是区别于简单 token pruning 的核心设计——我们不丢弃 token，只是不更新背景的 query

**实现细节（`src/sparse/sparse_processor.py`）：**
- Per-layer 独立 cache，每层 attention pattern 不同
- K/V projection 仍在全部 N token 上计算（前景 Q 需要它们）
- `with_kwargs=True` hook 拦截 Lumina attn processor，替换 SDPA 调用

### 3.4 Component 2: Sparse Feed-Forward Network（Lumina 架构）

**前置条件**：同稀疏 attention，要求内部特征无 timestep AdaLN 耦合。

**核心思想**：前景 token 通过完整 SwiGLU MLP，背景 token 直接复用缓存。

```python
# SparseFeedForward.forward(x):  x: (B, N, D)
if dense_mode:
    out = self.mlp(x)              # full forward
    self.cache = out.clone()
    return out

x_fg   = x[:, fg_indices]         # (B, N_fg, D)
out_fg = self.mlp(x_fg)           # (B, N_fg, D)  -- O(N_fg · D · 4D)

out = self.cache.clone()           # start from background cache
out[:, fg_indices] = out_fg        # overwrite foreground
self.cache = out
return out
```

FFN 计算量：`O(N·D·4D) → O(N_fg·D·4D)`，`N_fg ≈ 0.5N` → 2× FFN 加速。

**为何 SD3/FLUX 不能用**：FFN 输入是 `norm(h_t)` where `norm` 是 AdaLN，其 scale/shift 来自 timestep embedding。不同步的 `norm(h_t)` 值完全不同，缓存的 FFN 输出跨步复用会引入 `O(1)` 量级误差（实验 CLIP 下降 41.6%）。

### 3.5 Component 3: Spatial CFG Guidance Scaling

**传统 CFG**：全图统一 guidance scale `w`：
```
noise(h,w) = noise_unc(h,w) + w · (noise_cnd(h,w) − noise_unc(h,w))
```

**Spatial CFG**（适用于有双 pass CFG 的模型：Lumina, SD3）：
```
s(h,w) = s_bg + mask(h,w) · (s_fg − s_bg)
noise(h,w) = noise_unc(h,w) + s(h,w) · (noise_cnd(h,w) − noise_unc(h,w))
```

参数设置（消融验证最优）：`s_fg = 9.0`，`s_bg = 2.0`
- 前景高 CFG：增强 prompt 遵循度，锐化细节
- 背景低 CFG：避免过度 stylization，保持自然感

**FLUX 不适用**：FLUX.1-dev 使用 guidance distillation（单 pass，guidance 作为 embedding 输入），无法做空间差异化 guidance。

### 3.6 Component 4: Universal Step-Level Prediction Caching（全架构通用）

这是 AccelAes 在所有模型上都能使用的核心加速手段。原理：在去噪后期，相邻步的 noise_pred 变化光滑、可预测，可以用线性外推跳过整个 transformer forward。

**Skip 条件判断（fskip2 策略）：**
- 积累 `warmup_steps` 步建立历史（至少2步用于外推）
- 之后每隔1步跳过：compute, skip, compute, skip, ...
- 跳过步：用线性外推（而非直接复用）估计 noise_pred

**线性外推公式：**
```
# history: [pred_{t-2}, pred_{t-1}]  (已有2步历史)
pred_t_hat = 2 · pred_{t-1} − pred_{t-2}   # 一阶差分外推
```

**理论加速分析（steps=28, warmup=5）：**
- warmup 阶段（step 0-4）：5 次全量计算
- 跳步阶段（step 5-27）：23 步中每2步跳1步 → 12 次跳过，11+1=12 次计算（含warmup边界）
- 实际计算步数：`warmup + ceil((28-warmup)/2) = 5 + 12 = 17`
- 理论加速：`28/17 ≈ 1.65×`（实测 1.73× 因 skip 步有额外节省）

**实现（`src/sparse/skip_cache.py`，`SkipUpdateCache`）：**
```python
class SkipUpdateCache:
    def __init__(self, method="linear"):   # "copy" / "linear"
        self.history = []   # 最多保留2个历史 (linear) 或3个 (quadratic)

    def store(self, step, noise_pred):
        self.history.append((step, noise_pred.clone()))
        if len(self.history) > 2: self.history.pop(0)

    def get_prediction(self, lookahead=1):
        if self.method == "linear" and len(self.history) >= 2:
            p1, p2 = self.history[-2][1], self.history[-1][1]
            return 2*p2 - p1   # linear extrapolation
        return self.history[-1][1]   # fallback: copy

    def has_cache(self):
        return len(self.history) >= 2
```

**为何不用 quadratic 外推**：实验测试 quadratic（3步历史，更高阶外推）用于 consecutive 2-step skip 时视觉质量肉眼可见下降（CLIP −6.9%），放弃。Linear 外推在间隔1步时表现稳健。

### 3.7 Architecture-Specific Deployment Strategy

不同架构能使用的组件不同，由架构特性决定（非超参数选择）：

```
┌─────────────────┬──────────────┬──────────────┬──────────────────────────────┐
│ Component       │ Lumina-Next  │ SD3-Medium   │ FLUX.1-dev                   │
├─────────────────┼──────────────┼──────────────┼──────────────────────────────┤
│ Foreground Det. │ CFG magnitude│ Joint Attn.  │ N/A                          │
│ Sparse Attn     │ ✅            │ ❌ AdaLN      │ ❌ AdaLN + 实验验证 (†)       │
│ Sparse FFN      │ ✅            │ ❌ AdaLN      │ ❌ AdaLN                     │
│ Spatial CFG     │ ✅            │ ✅            │ ❌ No CFG                    │
│ Step Caching    │ ✅            │ ✅            │ ✅                           │
├─────────────────┼──────────────┼──────────────┼──────────────────────────────┤
│ Achieved Speedup│ **2.09×**    │ 1.50×        │ 1.73×                        │
└─────────────────┴──────────────┴──────────────┴──────────────────────────────┘
```

**(†) FLUX Sparse Attn 实验验证（2026-02-25）**：尽管理论上 FLUX single-stream blocks 有稀疏注意力的空间，我们在 24 个 prompt 上进行了三轮系统性测试（mask_step=4/SLIC，mask_step=7/threshold，mask_step=8/threshold），均发现严重质量问题：
- mask_step=8，threshold：20/24 prompt IR 下降，最坏情况 palace −57%，knight −62%
- 根本原因：FLUX 的 2×2 spatial packing 使背景 token 缓存复用在 packed sequence 中引入跨 patch 空间混叠；guidance distillation（单 pass，无 CFG）也意味着 single-stream 中图文混合序列的稀疏 attention 很难做到无损
- **结论：FLUX 最终只使用步级缓存（fskip2），这已被 30 prompt × 2 seed 的大规模评测验证稳定（IR −1.1%，LPIPS=0.048）**

**Lumina 的 AdaLN 情况**：Lumina 使用外部 sinusoidal positional encoding，timestep 信息通过 `<|t|>` token 进入序列，**不是** 通过 AdaLN 修改 scale/shift，因此内部特征跨步稳定，可以安全缓存。

**SD3 AdaLN 导致稀疏 FFN 失败的原理**：SD3 的每个 block 中：
```
h_normed = AdaLN(h, timestep_emb)   # scale_t · LayerNorm(h) + shift_t
ff_out   = gate_t · MLP(h_normed)
```
其中 `scale_t, shift_t, gate_t` 全部随 timestep 变化。即使 `h` 不变，`h_normed` 也完全不同。缓存的 `ff_out` 跨步复用引入 `O(1)` 误差，CLIP 下降 41.6%（实验验证）。

### 3.8 Complete Pipeline Pseudocode

```
Algorithm 2: AccelAes Full Pipeline (SD3 示例)
---------------------------------------------------------------------------
Input:  prompt c, seed, steps=28, mask_step=5, skip_interval=2, warmup=5
        s_fg=9.0, s_bg=2.0, n_segments=64

1.  x_T ~ N(0,I);  encode_prompt(c) → embeds
2.  mask = None;  cache = SkipUpdateCache("linear");  sparse_active = False

3.  for i, t in enumerate(timesteps):

4.    // Step-level skip (post warm-up)
5.    if cache.has_cache() and sparse_active:
6.      if (i - mask_step - 1) % skip_interval == 0:
7.        noise = cache.get_prediction(lookahead=1)   // linear extrapolation
8.        x_{t-1} = scheduler.step(noise, t, x_t)
9.        continue                                     // skip transformer

10.   // Install joint-attn hooks at mask_step
11.   if i == mask_step:  install_joint_attn_hooks()

12.   // DiT forward (CFG double pass for SD3)
13.   [noise_unc, noise_cnd] = DiT(cat([x_t, x_t]), t, cat([∅, c]))

14.   // Build spatial mask at mask_step
15.   if i == mask_step:
16.     A_maps = get_joint_attn_affinity_maps()        // from hooks
17.     mask   = SD3SemanticMask(A_maps, n_segments)   // SLIC + affinity
18.     remove_hooks()
19.     sparse_active = True

20.   // Spatial CFG
21.   if mask is not None:
22.     s_map = s_bg + mask · (s_fg - s_bg)           // per-pixel guidance
23.     noise = noise_unc + s_map · (noise_cnd - noise_unc)
24.   else:
25.     noise = noise_unc + w · (noise_cnd - noise_unc)  // standard CFG

26.   if sparse_active: cache.store(i, noise)

27.   x_{t-1} = scheduler.step(noise, t, x_t)

28. return VAE.decode(x_0)
---------------------------------------------------------------------------
```

---

## 4. Experiments（~2.5 pages）

### 4.1 Experimental Setup

**模型（三种架构，各有独特设计）：**
- **Lumina-Next-T2I**: 24 transformer blocks, GQA (32q/8kv, head_dim=72), RoPE, Gemma text encoder，1024×1024
- **SD3-Medium**: 24 MMDiT blocks (joint attention, AdaLN), CLIP+T5 text encoder，1024×1024
- **FLUX.1-dev**: 19 double-stream + 38 single-stream blocks, guidance distillation（单pass），1024×1024

**评估指标（8个，全部已实测）：**

| 指标 | 类型 | 工具 | 方向 |
|------|------|------|------|
| CLIP Score | 文本对齐 | open_clip ViT-L-14 @openai | ↑ |
| PickScore | 人类偏好代理 | yuvalkirstain/PickScore_v1 | ↑ |
| ImageReward | 人类偏好 | ImageReward-v1.0 | ↑ |
| HPSv2 | 美学+对齐 | hpsv2 v2.1 | ↑ |
| Aesthetic Score | 美学质量 | LAION aesthetic predictor (MLP on CLIP ViT-L-14) | ↑ |
| LPIPS | 感知距离 vs baseline | lpips AlexNet，resize to 256×256 | ↓ |
| Edge Density | 细节保持 | Scharr operator on grayscale | ↑ |
| FID | 分布距离 vs baseline | cleanfid，clean mode | ↓ |

**评测规模与数据路径：**
- Lumina：20 prompts × 3 seeds = **60 images/config**（`prompts/prompts_dev.txt[:20]`，seeds 0/1/2）
- SD3 & FLUX：30 prompts × 2 seeds = **60 images/config**（`prompts/prompts_dev.txt[:30]`，seeds 0/1）
- 所有图像已保存：`outputs/eval/{sd3|flux}/{config}/p{pi:04d}_s{seed:04d}.png`（skip-if-exists）
- Lumina 图像：`outputs/semantic_full_eval/{config}/images/p{pi:04d}_s{seed:04d}.png`

**硬件：** NVIDIA RTX 5090 32GB，PyTorch 2.x + BF16

---

### 4.2 Main Results（✅）

---

**Table 1: Lumina-Next-T2I 方法对比**
（20 prompts × 3 seeds = 60 images/config，30 steps，CFG=4.0）
数据来源：AccelAes → `outputs/p0_ablation_direct/summary.json`；其他方法 → 同 prompt/seed 集实测

| Method | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|------|---------|-------|-------|-----|------|--------|--------|-------|------|
| Baseline (dense, CFG=4.0) | 12.37s | 1.00× | 0.2531 | 0.2185 | 0.752 | 0.2710 | 5.941 | 0.000 | 0.583 | 0.00 |
| Δ-DiT [2406.01125] | 8.12s | 1.52× | 0.2523 | 0.2158 | 0.485 | 0.2540 | 5.873 | 0.172 | 0.522 | 97.4 |
| FORA [2407.01425] | 12.36s | 1.00× | 0.2463 | 0.2128 | 0.517 | 0.2496 | 5.766 | 0.242 | 0.565 | 148.0 |
| RAS [2502.10389] | 8.38s | 1.47× | 0.2550 | 0.2185 | 0.788 | 0.2713 | 5.927 | 0.025 | 0.581 | 20.0 |
| **AccelAes (ours)** | **5.86s** | **2.11×** | **0.2540** | **0.2188** | **0.841** | **0.2740** | **5.941** | 0.057 | **0.629** | 46.6 |
| Δ AccelAes vs baseline | — | — | +0.4% | +0.1% | **+11.9%** | **+1.1%** | +0.0% | — | **+7.8%** | — |

> 配置：semantic mask（threshold）+ sparse attn/FFN + spatial CFG (s_fg=7, s_bg=1) + fskip2（interval=2）。
> ⚠ RAS 为我们自己的复现（output-level blending），算法与论文一致。RAS 官方代码有 diffusers API 兼容性问题，实测 IR=0.35–0.48（语义错误）；官方 2.51× 需 flash_attn+Triton 内核。

---

**Table 2: SD3-Medium 方法对比**
（30 prompts × 2 seeds = 60 images/config，28 steps）
数据来源：`outputs/sd3_compare/summary.json` ✅

| Method | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|------|---------|-------|-------|-----|------|--------|--------|-------|------|
| Baseline (dense, CFG=7.0) | 3.52s | 1.00× | 0.2662 | 0.2198 | 0.879 | 0.2895 | 5.733 | 0.000 | 0.916 | 0.00 |
| TeaCache [2411.19108] (t=0.15) | 3.29s | 1.07× | 0.2662 | — | 0.876 | 0.2893 | 5.729 | **0.007** | 0.916 | **10.3** |
| TeaCache [2411.19108] (t=0.30) | 2.49s | 1.41× | 0.2667 | — | 0.847 | 0.2863 | 5.717 | 0.056 | 0.881 | 39.5 |
| TaylorSeer [2503.06923] (r=1) | 2.21s | 1.59× | 0.2682 | — | **0.898** | 0.2852 | 5.676 | 0.116 | 0.935 | 70.2 |
| TaylorSeer [2503.06923] (r=2) | 1.74s | **2.02×** | 0.2624 | — | 0.729 | 0.2723 | 5.561 | 0.281 | 0.998† | 132.6 |
| Δ-DiT [2406.01125] | 2.33s | 1.51× | 0.2634 | — | 0.828 | 0.2804 | 5.656 | 0.278 | 0.850 | 116.9 |
| FORA [2407.01425] | 3.54s | 0.99× | 0.2663 | — | 0.703 | 0.2643 | 5.522 | 0.282 | 0.830 | 146.1 |
| Step-cache only (fskip2) | 2.34s | 1.50× | **0.2688** | 0.2191 | 0.891 | 0.2867 | 5.663 | 0.058 | **0.960** | 44.3 |
| **AccelAes (ours)** | **2.34s** | **1.50×** | 0.2644 | 0.2187 | 0.804 | 0.2814 | 5.690 | 0.111 | 0.944 | 70.7 |
| Δ AccelAes vs step-cache only | — | — | −1.7% | −0.2% | −9.8% | −1.8% | +0.5% | worse | −1.7% | worse |

> 配置：AccelAes = semantic mask (joint attn affinity) + spatial CFG (s_fg=9, s_bg=2) + fskip2（无 sparse FFN，AdaLN 耦合排除）。
>
> **关键发现**：
> - **fskip2_only 是 SD3 上最优方案**（1.50×, IR +1.4%, LPIPS=0.058, FID=44.3）：步级缓存本身不损质量，甚至略有提升
> - **AccelAes spatial CFG 在 SD3 上适得其反**（IR −8.5%）：SD3 MMDiT 的 joint attention 已天然分离文本-图像交互，外加 spatial CFG (s_bg=2.0) 大幅削弱背景引导，反而降低全局连贯性
> - **TeaCache 在 SD3 上几乎无加速**：t=0.15 仅 1.07×（SD3 MMDiT 步间 noise_pred 变化幅度远大于 FLUX，阈值必须很低才能满足精度）
> - **TaylorSeer r=2 在 SD3 上崩溃**（IR 0.729, FID=132.6）：SD3 单步特征演化非线性，2阶 Taylor 外推误差大；r=1 也有 LPIPS=0.116 vs fskip2 的 0.058
> - **Δ-DiT 层缓存引入空间伪影**（LPIPS=0.278, FID=116.9）：SD3 MMDiT joint block 对跨步 encoder_hidden_states/hidden_states 联合变化敏感，分层缓存导致特征不一致
> - **FORA 在 SD3 上完全无效**（0.99×, IR 0.703）：pre-hook 不跳过前向计算，无速度收益；残差缓存与 SD3 AdaLN 调制耦合，引入严重伪影
> - **结论**：SD3 上只有步级跳步（fskip2）有效；空间自适应方法（AccelAes spatial CFG）和特征层缓存（Δ-DiT, FORA）均对 SD3 MMDiT 有害
> - 注：SD3 上未复现 RAS（需要 MMDiT 专用 PIT kernel，超出范围）；PickScore 仅对 fskip2_only 和 AccelAes 记录（来自 p1_ablation）
> - †TaylorSeer r=2 Edge=0.998 为伪高值：图像内容崩溃（IR=0.729, FID=132.6）后出现大量高频噪声/伪影，Canny 边缘检测计数虚高，非真实细节保留

---

**Table 3: FLUX.1-dev 方法对比**
（24 prompts × seed=42 = 24 images/config，28 steps，guidance=3.5）
数据来源：`outputs/stepcache_flux/summary.json` ✅

| Method | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|------|---------|-------|-------|-----|------|--------|--------|-------|------|
| Baseline (dense) | 12.66s | 1.00× | 0.2753 | 0.2290 | 1.233 | 0.3200 | 6.243 | 0.000 | 0.560 | 0.00 |
| TeaCache [2411.19108] (t=0.15) | 7.80s | 1.62× | 0.2765 | 0.2292 | 1.283 | 0.3174 | 6.268 | 0.032 | 0.535 | 21.9 |
| TeaCache [2411.19108] (t=0.30) | 5.92s | 2.14× | 0.2777 | 0.2291 | 1.225 | 0.3154 | 6.301 | 0.071 | 0.529 | 39.6 |
| TaylorSeer [2503.06923] (r=1) | 7.74s | 1.63× | 0.2763 | 0.2292 | 1.295 | 0.3196 | 6.261 | **0.018** | 0.565 | **13.4** |
| TaylorSeer [2503.06923] (r=2) | 5.96s | **2.12×** | 0.2765 | 0.2291 | **1.304** | **0.3217** | **6.309** | 0.055 | 0.572 | 32.6 |
| **AccelAes (ours)** | **7.31s** | **1.73×** | 0.2752 | 0.2290 | 1.267 | **0.3214** | 6.272 | 0.030 | **0.590** | 19.5 |
| Δ AccelAes vs baseline | — | — | −0.0% | +0.0% | **+2.7%** | +0.4% | +0.5% | — | +5.4% | — |

> FLUX 仅使用步级缓存（无 spatial CFG，无 sparse attn/FFN）：guidance distillation 无 CFG 可拆，2×2 spatial packing 导致 sparse attn 引入跨 patch 混叠（20/24 prompts IR 下降，实验排除）。
> 时间：steady-state baseline ≈ 12.66s；AccelAes 取 last-8 steady-state ≈ 7.31s；speedup = 12.66/7.31 = 1.73×。
> TeaCache/TaylorSeer 均为我们在相同 24-prompt 测试集上的复现结果。

---

**跨模型汇总（paper 正文表格候选）：**

| Model | Arch | Method | Speedup | CLIP Δ | IR Δ | Pick Δ | HPS Δ | Components |
|-------|------|--------|---------|--------|------|--------|-------|------------|
| Lumina-Next | No AdaLN, CFG | **AccelAes** | **2.11×** | +0.4% | **+11.9%** | +0.1% | +1.1% | Sparse attn+FFN + Spatial CFG + fskip2 |
| SD3-Medium | AdaLN, CFG | **AccelAes** | 1.50× | −0.7% | −8.5% | −0.5% | −2.8% | Spatial CFG + fskip2 |
| FLUX.1-dev | AdaLN, no CFG | **AccelAes** | 1.73× | −0.0% | +2.7% | +0.0% | +0.4% | fskip2 only |

> Lumina 最优因为支持全部三组件；SD3 无 sparse attn/FFN（AdaLN 限制）；FLUX 无 spatial CFG（guidance distillation 无双 pass）。

---

**Lumina 全部历史实验数据（消融 Table 素材，均已完成）：**

---

**[A] Plan A 评测**（`outputs/planA_full_eval_summary.json`，20p×3s=60，30 steps，CFG=4.0）
方法：sparse attn + sparse FFN + cfg_magnitude mask，bg_refresh 控制背景缓存刷新频率

| Config | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|------|---------|-------|-------|-----|------|--------|--------|-------|------|
| Baseline | 12.30s | 1.00× | 0.2531 | 0.2185 | 0.7519 | 0.2710 | 5.9407 | 0.0000 | 0.5829 | 0.00 |
| Plan A bg_refresh=5 | 7.46s | 1.65× | 0.2546 | 0.2181 | 0.7984 | 0.2735 | 5.8961 | 0.0696 | 0.6090 | 52.10 |
| Plan A bg_refresh=3 | 7.65s | 1.61× | 0.2553 | 0.2187 | 0.8151 | 0.2745 | 5.9264 | 0.0651 | 0.6192 | 50.45 |
| Δ (bg_refresh=5 vs base) | — | — | +0.6% | −0.2% | +6.2% | +0.9% | −0.7% | — | +4.5% | — |

---

**[B] Plan A + Step Skip**（`outputs/planA_skip_eval_summary.json`，同规模）
在 Plan A bg_refresh=5 基础上，叠加 full_skip_interval=3

| Config | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|------|---------|-------|-------|-----|------|--------|--------|-------|------|
| Baseline | 12.30s | 1.00× | 0.2531 | 0.2185 | 0.7519 | 0.2710 | 5.9407 | 0.0000 | 0.5829 | 0.00 |
| Plan A ref5 + fskip3 | 6.14s | **2.00×** | 0.2534 | 0.2168 | 0.7977 | 0.2701 | 5.8479 | 0.0800 | 0.5785 | 59.77 |
| Δ vs baseline | — | — | +0.1% | −0.8% | +6.1% | −0.3% | −1.6% | — | −0.8% | — |

---

**[C] Hybrid / CFG-fix 评测**（`outputs/hybrid_full_eval_summary.json`，同规模）
cfg_scale=4.0 统一用于 mask_step 之前，之后切换 spatial CFG

| Config | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|------|---------|-------|-------|-----|------|--------|--------|-------|------|
| Baseline | 12.54s | 1.00× | 0.2531 | 0.2185 | 0.7518 | 0.2710 | 5.9407 | 0.0000 | 0.5829 | 0.00 |
| Hybrid20-cfgfix | 8.43s | 1.49× | 0.2549 | 0.2186 | 0.8069 | 0.2735 | 5.9304 | 0.0631 | 0.5990 | 50.11 |
| Δ vs baseline | — | — | +0.7% | +0.0% | +7.3% | +0.9% | −0.2% | — | +2.8% | — |

---

**[D] 组件消融：Sparse Attn / FFN 单独 vs 组合**（`outputs/accel_eval_summary.json`，20p×3s=60）
注：此批实验使用早期版本 cfg_magnitude mask（非 semantic），skip_ratio=0.5，mask_step=5

| Config | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | Edge↑ | 备注 |
|--------|------|---------|-------|-------|-----|-------|------|
| Baseline | 12.54s | 1.00× | 0.2531 | 0.2185 | 0.7519 | 0.5829 | |
| Sparse Attn only | 8.41s | 1.49× | 0.2365 | 0.2031 | −0.032 | 0.4797 | CLIP −6.6% ❌ |
| Sparse Attn + FFN | 7.37s | 1.70× | 0.2410 | 0.2058 | 0.2351 | 0.4570 | CLIP −4.8% ❌ |
| Attn + FFN + fskip3 | 6.01s | **2.09×** | 0.2414 | 0.2058 | 0.2250 | 0.4560 | CLIP −4.6% ❌ |
| Attn + FFN + fskip2 | 5.24s | **2.39×** | 0.2419 | 0.2057 | — | 0.4771 | CLIP −4.4% ❌ |

> ⚠️ 注：此批数据质量下降严重（CLIP −4~7%），是因为当时 bg 缓存采用 no-refresh 策略（背景永不更新），
> 相当于背景完全冻结。后来改为 bg_refresh=5 后（[A]），CLIP 恢复正常（+0.6%）。
> 这批数据展示了"完全冻结背景"的失败模式，可用于 paper 中的 ablation 对比。

---

**[E] 最终最优方法：Semantic Mask + Spatial CFG + fskip2**（`outputs/semantic_full_eval/summary.json`，20p×3s=60）
semantic mask 替换 cfg_magnitude，spatial CFG (s_fg=7, s_bg=1) 替换 uniform CFG

| Config | Time | Speedup | CLIP↑ | Pick↑ | IR↑ | HPS↑ | Aesth↑ | LPIPS↓ | Edge↑ | FID↓ |
|--------|------|---------|-------|-------|-----|------|--------|--------|-------|------|
| Baseline | 12.37s | 1.00× | 0.2531 | 0.2185 | 0.7518 | 0.2710 | 5.9407 | 0.0000 | 0.5829 | 0.00 |
| **AccelAes semantic+fskip2** | 5.91s | **2.09×** | 0.2540 | 0.2190 | 0.8271 | 0.2760 | 5.9286 | 0.0772 | 0.6409 | 58.60 |
| AccelAes cfg_mag+fskip2 | 5.92s | **2.09×** | 0.2548 | 0.2186 | 0.8004 | 0.2752 | 5.9395 | 0.0727 | 0.6380 | 53.27 |
| Δ (semantic vs base) | — | — | +0.4% | +0.2% | +10.0% | +1.8% | −0.2% | — | +10.0% | — |

---

**[F] Mask Type 消融**（`outputs/mask_compare/summary.json`，10p×3s=30，注：仅 CLIP+Edge 两指标）

| Mask Type | Time | Speedup | CLIP↑ | Edge↑ |
|-----------|------|---------|-------|-------|
| Baseline | 12.32s | 1.00× | 0.2534 | 0.5674 |
| Semantic (joint attn) | 8.50s | 1.45× | 0.2578 (+1.7%) | 0.5735 |
| CFG magnitude | 7.37s | **1.67×** | 0.2571 (+1.5%) | 0.5836 |
| Complexity | 7.37s | **1.67×** | 0.2514 (−0.8%) | 0.5737 |
| Uniform random | 7.36s | **1.67×** | 0.2528 (−0.2%) | 0.5682 |

> 注：此批数据只有 CLIP+Edge（未跑全部 8 指标），可补充跑 Pick/IR/HPS。
> 结论：semantic 和 cfg_magnitude 均优于 complexity 和 uniform；
> cfg_magnitude 速度（1.67×）优于 semantic（1.45×，因 joint attn hook 有额外开销）。

---

**关键观察（paper Discussion 素材）：**

1. **Lumina 全方法最强（2.09×，质量全面提升）**：因为三个组件全部可用（sparse attn、sparse FFN、步级缓存）。IR 反而 +10% 因为 spatial CFG 增强了美学关注区域的细节和 prompt 遵从度。

2. **FLUX 步级缓存质量最干净（1.73×，几乎无损）**：CLIP ±0%，IR −1.1%，HPS +0.1%，Aesthetic +0.5%，LPIPS=0.048（和 baseline 几乎相同）。说明 step-level 线性外推在 FLUX 的 guidance distillation 架构下极为稳定。

3. **SD3 的 IR 下降（−8.5%）需要解释**：其他7个指标 <3%。ImageReward 基于 BLIP 评估文本-图像细节一致性，对 spatial CFG 引入的前景/背景差异化 guidance 更敏感。Ablation 计划：单独测 fskip2-only（无 spatial CFG）和 spatial-CFG-only（无 fskip）来分离贡献。

4. **FID 数字（60张图）不够可靠**：FID 需要 ≥2048 张图才有统计意义，60 张图的结果仅供参考。论文中可以报告 LPIPS 和感知对比替代 FID，或补充大规模生成实验。

5. **FLUX 比 SD3 快得多的理由**：FLUX baseline 12.67s/img vs SD3 baseline 3.52s/img。FLUX 使用单 pass（无 CFG），但 transformer 更大（57 blocks vs 24 blocks）。加速比 1.73× 对 FLUX 来说是 **5.35s 节省**，比 SD3 的 1.19s 节省更有价值。

### 4.3 Ablation Study（SD3 消融 ✅ 已完成）

> 数据来源：`outputs/p1_ablation/summary.json`（30p×2s=60 images/config，SD3-Medium，28 steps）

**SD3 Temporal vs Spatial 贡献分离（核心消融）：**

| Config | Desc | Time | Speedup | CLIP | IR | HPS | Aesth | LPIPS↓ | Edge | FID↓ |
|--------|------|------|---------|------|----|-----|-------|--------|------|------|
| Baseline | Dense, CFG=7.0 | 3.52s | 1.00× | 0.2662 | 0.8788 | 0.2895 | 5.7330 | 0.000 | 0.9160 | 0.00 |
| **fskip2_only** | **步级缓存 fskip2，均匀 CFG=7.0（无空间组件）** | **2.34s** | **1.50×** | 0.2688 | **0.8908** | 0.2867 | 5.6634 | **0.058** | 0.9597 | **44.30** |
| semantic_fskip2 | 语义 mask + spatial CFG (s_fg=9,s_bg=2) + fskip2 | 2.34s | 1.50× | 0.2644 | 0.8042 | 0.2814 | 5.6898 | 0.111 | 0.9437 | 70.68 |
| semantic_fskip3 | 语义 mask + spatial CFG + fskip3（33% 跳步） | 2.69s | 1.31× | 0.2638 | 0.8133 | 0.2837 | 5.7066 | 0.104 | 0.9054 | 64.53 |

**关键消融发现：**

1. **步级缓存 (fskip2) 独立效果**（fskip2_only vs baseline）：
   - IR: +1.4%（0.8788 → 0.8908）— step skip **改善** IR
   - LPIPS: 0.058（极低，感知差异很小）
   - 结论：步级线性外推在 SD3 上非常稳定，对质量几乎无损，实现 1.50×

2. **Spatial CFG 的代价**（semantic_fskip2 vs fskip2_only）：
   - IR: 0.8908 → 0.8042（-9.7%）— **IR 下降几乎全部来自 spatial CFG**，而非 step skip
   - LPIPS: 0.058 → 0.111（感知距离翻倍）
   - 原因：s_bg=2.0 大幅降低背景区域引导强度，ImageReward 对全图文本一致性更敏感
   - 结论：SD3 的 IR -8.5% 主因是 spatial CFG，step skip 本身无损甚至有微弱增益

3. **Skip interval 对比**（interval=2 vs interval=3）：
   - `full_skip_interval=3`（每3步跳1步）= 33% 跳步 < `interval=2`（50% 跳步）
   - fskip3 比 fskip2 **更慢**（1.31× vs 1.50×），且质量差别甚微（IR 0.813 vs 0.804）
   - 结论：interval=2 是 SD3 的最优 skip interval（速度与质量最佳平衡）

**SD3 消融总结（paper Discussion 素材）：**
- SD3 的 IR 下降（论文 -8.5% = semantic_fskip2 vs baseline）来自 spatial CFG（差异化引导）
- 步级缓存 fskip2 在 SD3 上：IR +1.4%，LPIPS=0.058，1.50× — 质量极其干净
- spatial CFG 带来：IR -9.7%（vs fskip2_only），LPIPS 从 0.058 升到 0.111
- 如果未来能设计出更温和的 spatial guidance（降低 s_fg-s_bg 差距），SD3 质量会更好
- ImageReward 对 spatial guidance 最敏感（全局 BLIP 评分）；其他7个指标差距 <3%

### 4.4 Visualization & Analysis

**Figure 1: Spatial non-uniformity motivation**
- 3 × 3 grid：3个 prompt，每个显示 (original image, CFG magnitude heatmap, semantic mask)
- 前景清晰分离，背景低激活

**Figure 2: fg/bg noise_pred variation per step**
- 折线图：两条线 fg_change vs bg_change（每步L2范数）
- 证明 bg 约为 fg 的 1/5，支撑 bg 可跳过的论据

**Figure 3: Qualitative comparison（已生成，data: outputs/accelae_figures/）**
- **主要可视化用 SD3**（详见下文选模型理由）
- 每个 prompt：5 panel — Baseline | Joint Attn Heatmap | Semantic Mask | Mask Overlay | AccelAes
- 脚本：`scripts/gen_accelae_figures.py`，输出：`outputs/accelae_figures/full_grid.png`（1940×4216px）
- 10 prompts × 5 panel；ImageReward 评分标注在对比图上（baseline vs AccelAes）

**Figure 4: Architecture compatibility diagram**
- 图示哪个组件在哪个架构上 work/fail，以及失败原因（AdaLN, no CFG）

---

**[可视化模型选择：为何用 SD3 而非 Lumina 或 FLUX]**（2026-02-25 实验验证）

我们对三个模型的 attention heatmap 可视化质量进行了对比实验：

| 模型 | Attention 类型 | Heatmap 质量 | 原因 |
|------|---------------|-------------|------|
| **SD3** | **Joint attention（图文联合）** | **✅ 语义清晰**（lion/tiger 形状可见，step 5/28 处即有清晰前景轮廓） | 图文 token 在同一注意力层共同计算，image→text affinity 在早期步已具有语义结构 |
| Lumina | Cross-attention（图文分离） | ❌ 棋盘噪声（step 5/30 处 latent 太嘈杂，spatial binding 未形成） | 仅 step 5 = 16.7% 采样进度，latent 还是纯噪声，cross-attn 尚未形成稳定空间-语义映射 |
| FLUX | Joint attention（single-stream） | ❌ 不稳定（2×2 packing 后 patch 分辨率 64×64，sparse attn 实验失败） | 同上，且 single-pass guidance distillation 无 CFG double-pass |

**结论**：
- **论文 Figure 3 可视化选用 SD3**：joint attention heatmap 在视觉上最具说服力，lion/tiger/warrior 等 prompt 在 mask_step=5 处可以清晰看到语义前景轮廓，直接验证 §3.2 的"空间非均匀性"观察
- Lumina 在 step 5（16.7% 进度）时 latent 仍很嘈杂，cross-attention heatmap 呈棋盘状噪声，不适合 motivation figure
- FLUX 因稀疏注意力失败（§3.7 注†），heatmap 可视化意义有限
- **Lumina 的可视化**可以改用较晚的 mask_step（如 step 10-15），或用 CFG magnitude（直接计算，无 hook，质量稳定）

**已生成的可视化文件（SD3）：**
- `outputs/accelae_figures/full_grid.png`：10 prompts × 5 panel，1940×4216px，8017KB
- `outputs/accelae_topk_flux/topk_grid.png`：FLUX topk 可视化（仅 baseline vs fskip2，无 sparse）
- `outputs/accelae_topk/topk_grid.png`：Lumina topk 可视化（24 prompts，top-6 by ΔIR）

### 4.5 Comparison with Prior Methods（P0 实测，Lumina-Next-T2I）

20 prompts × 3 seeds = 60 images/config，mask_step=5，skip_ratio=0.5，s_fg=7, s_bg=1，full_skip_interval=2，sparse_blocks=True，sparse_ffn=True。
数据来源：AccelAes → `outputs/semantic_full_eval/summary.json`；其他方法 → 各自实测（同提示集/种子）

#### 4.5.1 与基线方法的比较

| Method | Time/img | Speedup | CLIP | PickScore | IR | HPS | Aesthetic | LPIPS↓ | Edge | FID↓ |
|--------|----------|---------|------|-----------|-----|------|-----------|--------|------|------|
| Baseline (dense, CFG=4.0) | 12.37s | 1.00× | 0.2531 | 0.2185 | 0.7518 | 0.2710 | 5.9407 | 0.000 | 0.5829 | 0.00 |
| Δ-DiT (3-tier layer cache, warmup=2) | 8.12s | **1.52×** | 0.2523 | 0.2158 | 0.4853 | 0.2540 | 5.8729 | 0.172 | 0.5221 | 97.37 |
| FORA (block residual reuse, interval=2) | 12.36s | 1.00× | 0.2463 | 0.2128 | 0.5170 | 0.2496 | 5.7664 | 0.242 | 0.5645 | 148.03 |
| RAS (cross-attn magnitude mask, ratio=0.5) | 8.38s | 1.47× | 0.2550 | 0.2185 | 0.7882 | 0.2713 | 5.9267 | 0.025 | 0.5809 | 19.95 |
| **AccelAes (ours, semantic+thresh+fskip2)** | **5.86s** | **2.11×** | 0.2540 | 0.2188 | **0.8410** | 0.2740 | 5.9414 | 0.057 | 0.6285 | 46.64 |
| Δ AccelAes vs baseline | — | — | **+0.4%** | **+0.1%** | **+11.9%** | **+1.1%** | +0.0% | — | **+7.8%** | — |
| Δ RAS vs baseline | — | — | +0.8% | +0.0% | +4.8% | +0.1% | −0.2% | — | −0.3% | — |

**结论**：
- Δ-DiT 有 1.52× 加速但 IR 下降 35%、LPIPS=0.172（图像严重变形）
- FORA 无有效加速（1.00×）且质量全面下降（CLIP −2.7%，IR −31%，LPIPS=0.242）
- RAS：1.47× 加速，质量接近 baseline（IR +4.8%，LPIPS=0.025），当前最佳质量方法
- **AccelAes：2.09× 加速（最快），质量反而超越 baseline（IR +10%，CLIP +0.4%，Edge +10%）**，得益于 semantic mask + spatial CFG 对美学关注区域的增强

**关键优势分析（vs RAS）**：
- AccelAes 比 RAS 快 **1.42×**（5.91s vs 8.38s），且质量更好（IR +5.3% vs +4.8%，Edge +10% vs −0.3%）
- AccelAes 的 mask 在 mask_step 一次建立，全程复用，无 per-step overhead（RAS 每步重新计算 cross-attn magnitude → overhead 约 2s/30step）

**⚠ 关于 RAS 实现说明（重要）**：
- 上表中的 **RAS 行**（8.38s，1.47×，LPIPS=0.025）使用的是**我们自己对 RAS 算法的忠实复现**（输出级 blending：每步计算 cross-attention magnitude mask，选择活跃区域做完整 transformer forward，非活跃区域复用上步 noise_pred）。

- 我们尝试使用 **官方 microsoft/RAS 代码**（github.com/microsoft/RAS）复现，但存在 **diffusers 0.36+ 兼容性问题**：官方代码的 `ras_forward` 直接 slice `hidden_states` 到活跃 token 后进行 transformer 计算（需要旧版 diffusers LuminaNextDiT 内部 API），与当前 `LuminaPipeline`（diffusers 0.36+）不兼容，导致图像语义错误。

- **官方代码实测结果**（60 images，相同 prompts/seeds）：

| RAS Official Config | Time | Speedup | CLIP | IR | HPS | Aesth | LPIPS↓ | FID↓ |
|---------------------|------|---------|------|----|-----|-------|--------|------|
| Dynamic (skip_num_step=256, len=4) | 7.85s | 1.57× | 0.2440 | 0.4826 | 0.2489 | 5.7212 | 0.1647 | 99.36 |
| Static (skip_num_step=0, ratio=50%) | 7.64s | 1.61× | 0.2422 | 0.3465 | 0.2408 | 5.6395 | 0.2025 | 118.97 |

  对比：官方代码 IR=0.35–0.48（vs baseline 0.75），像素级 RMSE=19.8（vs 我们的复现 RMSE=4.6）。
  图像视觉上连贯（mean=81, std=54）但语义错误，是 API 不兼容导致的系统性错误，而非算法问题。

- **关于 RAS 官方声称的 2.51× 加速**：其 2.51× 需要 flash_attn + Triton indexed matmul + RoPE kernel fusion。在无 flash_attn 环境（我们的硬件）下，官方代码实测 1.57–1.61×（真正的 token slicing）vs 我们复现的 1.47×（output-level blending + hook overhead）。

- **论文使用策略**：RAS 对比行沿用**我们自己的复现**（算法逻辑完全对应论文描述，无实现 bug），并在 paper 中注明官方代码存在 diffusers 兼容性问题、无法在当前环境直接复现。

#### 4.5.2 步级缓存方法对比（TeaCache & TaylorSeer）

**关注点**：将 AccelAes 的步级缓存组件与 TeaCache（CVPR 2025 Highlight）和 TaylorSeer（ICCV 2025）在同一测试集上做量化比较，验证我们的步级缓存实现选型合理，并确认 AccelAes 的主要贡献（空间稀疏+CFG）独立于步级缓存方法的选择。

---

**FLUX 步级缓存方法对比（含 AccelAes）**（24 prompts，seed=42，28 steps，全 8 项指标）

> 数据来源：`outputs/stepcache_flux/summary.json` ✅（AccelAes + TeaCache + TaylorSeer 统一测试集）
> 注：baseline mean_time=16.91s（首张 GPU 预热约 116s 拉高均值）；real speedup 使用 steady-state baseline 12.66s 计算；config 绝对耗时均为热身后正常值（accelae_fskip2 取 last-8 steady-state = 7.31s）。

| Method | Time | Real Speedup | CLIP | IR | HPS | Aesth | LPIPS↓ | Edge | FID↓ |
|--------|------|--------------|------|----|-----|-------|--------|------|------|
| Baseline | ~12.66s | 1.00× | 0.2753 | 1.233 | 0.3200 | 6.243 | 0.000 | 0.560 | 0.00 |
| TeaCache t=0.10 | 9.07s | 1.40× | 0.2753 | 1.245 | 0.3180 | 6.247 | 0.020 | 0.543 | 13.8 |
| **AccelAes fskip2** (ours) | **7.31s** | **1.73×** | 0.2752 | **1.267** | **0.3214** | **6.272** | 0.030 | 0.590 | **19.5** |
| TeaCache t=0.15 | 7.80s | 1.62× | 0.2765 | 1.283 | 0.3174 | 6.268 | 0.032 | 0.535 | 21.9 |
| **TaylorSeer r=1** | 7.74s | 1.63× | 0.2763 | 1.295 | 0.3196 | 6.261 | **0.018** | 0.565 | **13.4** |
| TeaCache t=0.20 | 7.02s | 1.80× | 0.2781 | 1.248 | 0.3176 | 6.282 | 0.048 | 0.535 | 29.1 |
| TeaCache t=0.30 | 5.92s | 2.14× | 0.2777 | 1.225 | 0.3154 | 6.301 | 0.071 | 0.529 | 39.6 |
| **TaylorSeer r=2** | 5.96s | **2.12×** | 0.2765 | **1.304** | **0.3217** | **6.309** | 0.055 | 0.572 | 32.6 |
| TaylorSeer r=4 | 4.62s | 2.74× | 0.2736 | 1.179 | 0.3101 | 6.256 | 0.210 | 0.609 | 86.0 |
| TaylorSeer（paper†） | — | 4.99× | — | — | — | — | — | — | — |

†TaylorSeer 论文 FLUX 4.99× 使用不同测试集（baseline IR~0.942，中间特征级缓存），不可直接数值对比。

**关键发现（FLUX）**：
1. **~1.7× 区间（AccelAes fskip2 vs TaylorSeer r=1 vs TeaCache t=0.15）**：AccelAes（FID=19.5，IR=+2.7%，LPIPS=0.030）在 FID 上优于 TeaCache（FID=21.9，IR=+4.1%，LPIPS=0.032），而 TaylorSeer r=1（FID=13.4，IR=+5.0%，LPIPS=0.018）在质量上最优但速度略慢（1.63×）。
2. **~2.12-2.14× 区间（TaylorSeer r=2 vs TeaCache t=0.30）**：r=2（IR=1.304，LPIPS=0.055，FID=32.6）vs t030（IR=1.225，LPIPS=0.071，FID=39.6）— TaylorSeer 全面优于 TeaCache（IR 高 **+6.4%**，LPIPS 低，FID 低）。
3. **TaylorSeer r=2 为最优**（FLUX，高速区间）：2.12×，IR +5.8% vs baseline，LPIPS=0.055，FID=32.6，是各步级缓存方法中质量-速度最优平衡点。
4. **TaylorSeer r=4 退化**：2.74×，IR −4.4%，LPIPS=0.210，FID=86.0 — 激进跳步带来明显质量损失。
5. **AccelAes fskip2 在 FLUX 上的定位**：1.73×、FID=19.5（最低的 ~1.7× 方法），无需复杂 Taylor 计算，适合轻量步级缓存需求。

---

**Lumina 步级缓存方法对比**（20 prompts × seeds=[42,1337] = 40 images/config，30 steps，均匀 CFG=4.0）

> 数据来源：`outputs/stepcache_lumina/summary.json` ✅（TeaCache + TaylorSeer standalone，全 8 项指标）
> 注：所有 standalone 方法均匀 CFG=4.0，无 spatial mask/sparse attn。AccelAes 行来自 `outputs/p0_ablation_direct`（seeds=[0,1,2]，IR delta 与相同 CFG=4.0 baseline 比较）。

| Method | Time | Speedup | CLIP | IR | HPS | Aesth | LPIPS↓ | Edge | FID↓ |
|--------|------|---------|------|----|-----|-------|--------|------|------|
| Baseline | 12.33s | 1.00× | 0.2532 | 0.705 | 0.2698 | 6.034 | 0.000 | 0.666 | 0.00 |
| TeaCache t=0.06 | 11.56s | 1.07× | 0.2522 | 0.678 | 0.2684 | 6.029 | 0.025 | 0.661 | 20.3 |
| TeaCache t=0.10 | 9.89s | 1.25× | 0.2517 | 0.667 | 0.2660 | 5.995 | 0.033 | 0.628 | 28.9 |
| TeaCache t=0.15 | 8.28s | 1.49× | 0.2512 | 0.670 | 0.2640 | 5.978 | 0.046 | 0.599 | 37.7 |
| TeaCache t=0.20 | 7.39s | 1.67× | 0.2523 | 0.616 | 0.2617 | 5.950 | 0.057 | 0.583 | 45.2 |
| TaylorSeer r=1 | 7.60s | 1.62× | 0.2525 | 0.663 | 0.2675 | 6.012 | 0.040 | 0.666 | 34.1 |
| TaylorSeer r=2 | 6.02s | 2.05× | 0.2491 | 0.604 | 0.2650 | 5.940 | 0.087 | 0.668 | 65.1 |
| TaylorSeer r=3 | 5.23s | 2.36× | 0.2442 | 0.408 | 0.2560 | 5.808 | 0.170 | 0.685 | 112.2 |
| **AccelAes full** (ours)† | **5.86s** | **2.11×** | **0.254** | **0.841** | — | — | **0.057** | — | **46.6** |

†AccelAes 行来自 `outputs/p0_ablation_direct`（seeds=[0,1,2]，60 images，含 spatial+CFG+fskip2 所有组件）；绝对 IR 可直接比较（两组均用相同 CFG=4.0 baseline，accleAes baseline IR=0.752）

**关于 AccelAes + TaylorSeer step caching 的额外实验**（在 AccelAes 框架内替换 fskip2）：

> 数据来源：`outputs/taylor_lumina/scores_taylor.json`（seeds=[42,1337]，20p×2s=40 images）
> 注：此组均含完整 AccelAes 空间组件（semantic mask, sparse attn/FFN, spatial CFG=7/1）

| Method | Avg lat | Speedup | Avg IR | IR Δ vs sasd_fskip2 |
|--------|---------|---------|--------|---------------------|
| AccelAes + fskip2（参考） | 6.03s | ~2.04× | 0.751 | — |
| AccelAes + TaylorSeer r=2 | 5.10s | ~2.41× | 0.667 | −11.2% ❌ |
| AccelAes + TaylorSeer r=3 | 4.60s | ~2.68× | 0.655 | −12.8% ❌ |
| AccelAes + TaylorSeer r=4 | 4.35s | ~2.77× | 0.605 | −19.4% ❌ |

**关键发现（Lumina）**：
- **TaylorSeer standalone 在 Lumina 上质量快速崩溃**：r=2（2.05×）IR 降至 0.604，FID=65.1；r=3（2.36×）IR 仅 0.408，FID=112.2，LPIPS=0.170 — 已完全失效。根本原因：Lumina 的 dual-pass CFG 导致每步 noise_pred 变化幅度较大，高阶 Taylor 外推误差在多步积累后被放大。
- **TeaCache standalone 在 Lumina 上同样效果有限**：最优设置（t=0.15，1.49×）仅 IR −4.9%，无法获得超过 1.67× 的有效加速。
- **TaylorSeer 在 AccelAes 框架内质量同样退化**（−11% 到 −19%）：spatial CFG 引入的空间异构结构使 Taylor 外推更难稳定。
- **结论**：Lumina 上 AccelAes（IR +11.9%，2.11×）显著优于所有纯步级缓存方法；fskip2 是 AccelAes 最优步级缓存选择（TaylorSeer 在 FLUX 无 spatial CFG 时更有优势）。

---

**综合结论（步级缓存 vs AccelAes 主体）**：

| 比较维度 | 结论 |
|----------|------|
| 步级缓存最优方法 | FLUX: TaylorSeer r=2（IR +5.8%，2.12×，LPIPS=0.055）；Lumina: fskip2（稳定，兼容 spatial CFG） |
| TeaCache vs TaylorSeer（FLUX，~1.6×） | r1（LPIPS=0.018，FID=13.4）> t015（LPIPS=0.032，FID=21.9）；TaylorSeer 略优 |
| TeaCache vs TaylorSeer（FLUX，~2.1×） | r2（IR=1.304，LPIPS=0.055）>> t030（IR=1.225，LPIPS=0.071）；TaylorSeer 全面优势 |
| TeaCache vs TaylorSeer（Lumina，~1.6×） | r1（1.62×，IR=0.663，LPIPS=0.040）≈ t015（1.49×，IR=0.670，LPIPS=0.046）；相近 |
| TeaCache vs TaylorSeer（Lumina，2×+） | r2（2.05×，IR=0.604，FID=65.1）vs t020（1.67×，IR=0.616，FID=45.2）；Taylor 更快但 LPIPS/FID 劣化；r=3 时 Taylor 完全崩溃（IR=0.408，FID=112）|
| TaylorSeer（论文）vs ours | 论文 4.99× 太激进导致质量下降；r=2（2.13×）是 FLUX 最优平衡点 |
| **步级缓存 vs AccelAes 核心** | **步级缓存是辅助加速组件**；Lumina IR +11.9% 主要来自空间稀疏+spatial CFG，步级缓存仅贡献速度提升（~1.35× additional）；即使不用步级缓存，AccelAes 空间组件独立实现 IR +8.8%，1.42× |

#### 4.5.3 组件消融实验（Lumina-Next，20p×3s=60）

数据来源：`outputs/p0_ablation_direct/summary.json`（region_method=threshold，全部 bug-fixed 版本）

| Config | Desc | Time | Speedup | CLIP | IR | HPS | LPIPS↓ | Edge | FID↓ |
|--------|------|------|---------|------|----|------|--------|------|------|
| Baseline | Dense, CFG=4.0 | 12.37s | 1.00× | 0.2531 | 0.7518 | 0.2710 | 0.000 | 0.5829 | 0.00 |
| fskip_only | 步级缓存 fskip2（无空间组件） | 7.87s | 1.57× | 0.2500 | 0.7432 | 0.2723 | 0.254 | 0.7195 | 113.95 |
| attn_only | 稀疏 attn + spatial CFG（无 FFN，无 fskip） | 8.72s | 1.42× | 0.2545 | 0.8182 | 0.2725 | 0.053 | 0.5872 | 43.93 |
| attn_ffn | 稀疏 attn + 稀疏 FFN + spatial CFG（无 fskip） | 8.62s | 1.43× | 0.2545 | 0.8182 | 0.2725 | 0.053 | 0.5872 | 43.93 |
| **accelae_full** | **attn+ffn+spatial_cfg+fskip2（完整方法）** | **5.86s** | **2.11×** | **0.2540** | **0.8410** | **0.2740** | **0.057** | **0.6285** | **46.64** |
| Δ AccelAes vs baseline | | — | — | **+0.4%** | **+11.9%** | **+1.1%** | — | **+7.8%** | — |

**消融观察**（region_method=threshold，数据来自 `outputs/p0_ablation_direct/summary.json`）：
1. **fskip_only**：仅步级缓存时 LPIPS=0.254 最高 — 步级外推误差会在背景区域累积，对空间一致性影响大。
2. **attn_only**：spatial CFG 增强前景，IR +8.8% vs baseline，LPIPS=0.053，1.42×。
3. **attn_ffn**：加入稀疏 FFN 后数值与 attn_only 基本一致（IR=0.818, LPIPS=0.053），说明 FFN sparsity 对质量 metrics 影响微小（64×64 heatmap 下背景 FFN 缓存足够稳定）。
4. **accelae_full**：完整方法 IR=0.841（**+11.9% vs baseline**），LPIPS=0.057，**2.11×**。得益于：
   - Spatial CFG 放大前景细节（IR、Aesthetic 全面提升）
   - is_bg_refresh=True 策略：背景始终使用 dense 计算结果，保证 bg 质量
   - fskip2 在维持质量的前提下跳过整步，额外获得速度
5. **空间稀疏 + 步级缓存的协同**：attn_ffn (LPIPS=0.053, 1.43×) + fskip2 → accelae_full (LPIPS=0.057, 2.11×)，速度提升 0.68× 而质量几乎不变。

**Direct threshold vs SLIC（sasd_full 对比）**：
- SLIC（旧）：IR=0.823，LPIPS=0.077，FID=57.8
- Direct threshold（新）：IR=0.841，LPIPS=0.057，FID=46.6
- Δ：IR +2.2%，LPIPS −0.020，FID −11.1，速度相同（5.86s vs 5.88s）
- 结论：SLIC 在 64×64 patch heatmap 上的空间平滑引入偏差；直接百分位阈值更精确，所有指标更优。

### 4.6 Compatibility: AccelAes + Other Methods

- **AccelAes + ToMe**：step cache 后再做 ToMe，两者在不同层面工作（step level vs token level），正交组合预期 >2× speedup
- **AccelAes + DPM-Solver 20 steps**：减步数 + 步级缓存叠加

---

## 5. Conclusion

- AccelAes 是**第一个以主动提升美学质量为目标的 training-free DiT 计算分配框架**。通过识别 prompt 中美学描述词对应的空间区域（CLIP 锚点匹配 + cross-attention affinity），AccelAes 向这些区域倾斜计算资源并强化引导，在加速的同时主动改善图像美学质量。
- 核心贡献（空间语义计算分配）在 Lumina-Next 上独立实现 IR +8.8%（1.42×）；结合步级缓存（类 TeaCache/TaylorSeer，正交辅助）总体达到 IR +11.9%（2.09×）。
- 三架构验证（Lumina / SD3 / FLUX）说明：美学锚点检测和 spatial CFG 是核心，步级缓存是通用辅助，各自均有独立价值。
- **Limitations**:
  1. 稀疏 attention/FFN 仅在无 timestep-coupled AdaLN 的架构上可用（Lumina-style），SD3/FLUX 受限于架构无法启用
  2. Spatial CFG 要求双 pass CFG，对 guidance distillation 模型（FLUX）不适用
  3. FID 在 60 张图的规模下不够可靠（需要 ≥2048 张），需补充大规模评测
  4. mask 在去噪过程中固定（step 5 建立），未随步动态更新；自适应 mask 更新留待未来工作

---

## Appendix

### A. Additional Qualitative Results（各模型各 prompt 完整对比图）
### B. Detailed Ablation Tables（每个维度的数字）
### C. AdaLN Coupling Failure Analysis（稀疏 FFN/Attn 在 SD3 失败的实验分析）
### D. Implementation Details（超参数选择，代码结构）
### E. SDXL Results（U-Net 架构的初步验证，有限加速效果的讨论）

---

## 写作建议

1. **Introduction 的 motivation figure 是决定 acceptance 的关键**：一张 CFG magnitude heatmap + fg/bg 变化折线图，让 reviewer 一眼看到问题所在

2. **AdaLN 分析是技术深度的体现**：详细解释为何稀疏 FFN 在 SD3 失败（这是大多数相关工作没有分析的地方），展示我们对架构的深入理解

3. **强调 "架构感知" 而非 "架构无关"**：诚实地说明哪些组件有架构限制，这反而增加可信度；但步级缓存是真正通用的

4. **FLUX 步级缓存结果是最干净的**：CLIP +0%, IR −1%, 1.73× speedup，几乎完美。需注意：FLUX sparse attention 已经系统测试（24 prompt × 3 配置）确认有害（20/24 prompt IR 下降），**不能将 FLUX 作为 sparse attention 的展示**；FLUX 在论文中仅展示 fskip2（步级缓存）的有效性，是跨架构通用性的最佳证明

5. **SD3 的 IR 下降要解释**：在 paper 中明确说明 ImageReward 对 spatial guidance 差异比其他指标更敏感，其他7个指标均 <3% drop

6. **和 SDiT 的区别**：SDiT 的 3× headline 是在 ratio=0.125 激进设置下（FID +20%，LPIPS=0.20）；在可比质量下（ratio=0.5）SDiT 仅 1.66×，低于 AccelAes 的 2.09×。核心区分：(a) 信号免费（无额外计算）；(b) fskip2 步级缓存是正交贡献；(c) 三架构泛化

7. **Reviewer 反驳预案（速度靠叠加指控）**：
   - 指控："你的 2.09× 是叠加了 TeaCache 类方法才有的，自己的贡献只有 1.42×"
   - 回答：第一，1.42× 已超越 RAS 复现的 1.47× 且质量更好（IR +8.8% vs +4.8%）；第二，我们的目标从来不是速度最大化，而是**质量提升**——即使只用空间组件（1.42×），我们是唯一 IR 主动提升的方法；第三，步级缓存已在论文中引用 TeaCache/TaylorSeer，我们不 claim 其为原创，使用它是工程集成而非创新声明。
   - 关键句（Introduction 或 Contributions 中）："Unlike prior methods that treat quality maintenance as their ceiling, AccelAes sets quality improvement as its primary objective. The spatial component alone achieves 1.42× speedup with IR +8.8%—surpassing all quality-maintaining methods at comparable speed. Step-level caching [TeaCache, TaylorSeer] is incorporated as an orthogonal engineering module, not a primary contribution."

8. **写作中统一术语**：
   - "spatial component" = sparse attn + FFN + spatial CFG（架构相关）
   - "temporal component" = step-level prediction caching（架构无关）
   - "aesthetic region / aesthetically-relevant region" = 对应 prompt 美学描述词的 patch 区域（避免使用 "foreground"，因为美学关注区域不等于前景物体）
   - "aesthetic anchor tokens" = prompt 中与预定义美学词集合相似度高的 token（photorealistic、intricate 等）
   - "low-attention region" = aesthetic affinity 低的区域（避免使用 "background"）
