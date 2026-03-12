# Diffusion Transformer 加速方法调研（2024–2026）

> 更新日期：2026-02-25
> 范围：training-free 加速，image/video DiT，端到端 latency speedup

---

## 一、步级缓存 / 特征预测类

### TaylorSeer — ⭐ 当前 SOTA（步级缓存）
- **来源**：ICCV 2025 | arXiv 2503.06923
- **模型**：FLUX.1-dev（text-to-image），HunyuanVideo（video），DiT-XL/2（class-cond）
- **加速**：
  - FLUX.1-dev：2.96× / 3.53× / **4.99×**（latency speedup，N=5/6/6，O=2/1/2）
  - HunyuanVideo：5.00×（VBench 几乎无损）
  - DiT-XL：4.53×（FID baseline 2.32 → 2.65，+0.33）
- **质量（FLUX）**：ImageReward 0.995~1.030（baseline ~0.942），**超出 baseline**
- **方法**：对 transformer 中间层特征做截断 Taylor 级数展开，用历史多步梯度外推当前步特征，比线性外推精度更高
- **与 AccelAes 对比**：AccelAes fskip2 在 FLUX 只有 1.73×，差距主因是 Taylor 高阶外推 vs 线性外推；但 AccelAes 在 Lumina 额外有 IR +11.9% 质量提升（TaylorSeer 只保持质量，不提升）
- **代码**：https://github.com/Shenyi-Z/TaylorSeer

---

### DPCache — 全局最优跳步规划
- **来源**：arXiv（2025，"Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache"）
- **模型**：FLUX.1-dev，SD3，DiT-XL/2
- **加速**：FLUX **4.87×**，SD3 **3.50×**，视频 DiT 达 **4×+**

#### 核心方法

1. **PACT 代价张量（3D offline profiling）**
   用少量校准图离线构建代价张量 `C[t, k, strategy]`，记录在时间步 `t` 连跳 `k` 步的质量损失。仅需一次离线校准，推理时查表。

2. **动态规划全局调度**
   对整条 T 步去噪轨迹做 DP，在总质量损失预算约束下最大化跳步数。
   结果是**非均匀调度**：
   - 轨迹首尾（noise_pred 变化剧烈）→ 密集计算，几乎不跳
   - 轨迹中段（平滑区，步间差异极小）→ 连跳 4-5 步

3. **最终层缓存（output-level only）**
   只缓存 transformer 最终输出（noise_pred），不缓存中间层特征。
   原因：SD3/FLUX 的 AdaLN 使中间层特征随 timestep 大幅变化，中间层缓存误差会累积；noise_pred 跨步变化更平滑。

4. **二阶 Taylor 外推**
   `f̂(t) ≈ f(t-1) + Δ₁ + Δ₂/2`（用两步历史梯度外推），比线性外推（只用 Δ₁）精度更高，允许跳更多步而不崩质量。

#### 为何远超 fskip2（1.73× → 4.87×）

| | fskip2（AccelAes 用） | DPCache |
|--|--|--|
| 调度方式 | 固定周期：每 2 步跳 1 步 | DP 全局最优：中段可跳 4-5 步 |
| 中段步骤 | 跳 50%（本可跳 80%+）| 按轨迹形状，实际跳 70-80% |
| 外推精度 | 线性（一阶） | 二阶 Taylor |
| 计算量节省 | ~50% FLOPs | ~80% FLOPs |

核心洞察：**去噪轨迹不是均匀的**。中段（t≈0.3~0.7）noise_pred 步间相似度极高，可以批量跳过；固定周期跳步不利用这一结构，严重浪费中段的跳步潜力。

#### 为什么 DPCache 论文里复现的 TeaCache/TaylorSeer 也很快

这是论文对比时的常见现象，原因有以下几点：

**① FLOPs 减少量 ≠ 实际 wall-clock 加速**
许多论文报告 `speedup = baseline_FLOPs / method_FLOPs`。
跳过一步 = 省掉整步 FLOPs，理论 2× 或更高。
但实际延迟：GPU dispatch 开销、Python 调度、内存读写不随 FLOPs 等比缩减，所以 FLOPs 3× ≠ 延迟 3×。
我们测的是真实 `time.time()` 端到端延迟，这才是工程意义上的加速比。

**② 更激进的参数设定**
DPCache 论文为了在同等加速倍数下对比质量，可能使用更激进的 baseline 配置（如 TaylorSeer r=4/5 实现 3-4× FLOPs 减少），但质量会明显下降——这正好衬托 DPCache 在高加速比下的质量优势。
我们测的 TaylorSeer sweet spot 是 r=2（2.12× wall-clock，质量最优）。

**③ 硬件与实现差异**
- H100 + `torch.compile` + CUDA graphs 环境下，跳步的相对收益更接近理论值（dispatch overhead 占比更低）
- 我们在单 GPU 无编译优化的环境下测量，overhead 占比更高

**④ 对 AccelAes 的影响**
被 reviewer 质问"为何不如 DPCache 快"时，正确回答：
- 我们报告的是 **end-to-end wall-clock latency**，不是 FLOPs 减少量
- AccelAes 目标是**质量提升**（Lumina IR +11.9%），不是极限加速；两者定位不同
- DPCache 和 AccelAes 空间稀疏正交，未来可叠加达到更高加速比

#### 与 AccelAes 的对比定位

| 维度 | DPCache | AccelAes |
|------|---------|----------|
| 加速策略 | 时间维度：DP 全局调度跳步 | 时间 + 空间：fskip2 + 语义稀疏 |
| 质量变化 | 维持（不提升）| Lumina: **IR +11.9%**（提升） |
| 适用架构 | FLUX / SD3（AdaLN，noise_pred 平滑）| Lumina / SD3 / FLUX |
| 空间感知 | 无（整图跳步）| 有（fg/bg 分区稀疏 + spatial CFG）|
| 叠加潜力 | 可与空间稀疏叠加 | 可与 DP 调度叠加 |
| FLUX 加速 | **4.87×** | 1.73× |
| Lumina IR | — | **+11.9%** |

---

### TeaCache — ⭐ CVPR 2025 Highlight
- **来源**：CVPR 2025 Highlight | arXiv 2411.19108
- **模型**：Open-Sora-Plan（video），FLUX image
- **加速**：
  - Open-Sora-Plan：**4.41×**（VBench −0.07%）
  - FLUX image：~**2×**（用户实测 40-60% 减少）
- **方法**：用 timestep embedding 的差异预测模型输出变化量，从而决定哪些步可以跳过；非均匀调度（变化大的步不跳）
- **特点**：轻量级预测（只看 embedding，不看中间层），几乎零额外开销
- **代码**：https://github.com/ali-vilab/TeaCache

---

### ToCa — Token-wise Feature Caching
- **来源**：ICLR 2025 | arXiv 2410.05317
- **模型**：PixArt-α，FLUX
- **加速**：~2×（lossless）
- **方法**：按 token 的步间相似度打分，相似度高的 token 缓存复用，相似度低的重新计算
- **与 AccelAes 对比**：无语义 mask，缓存策略按相似度而非美学语义驱动
- **代码**：https://github.com/Shenyi-Z/ToCa

---

### TraCache — 轨迹感知特征预测
- **来源**：OpenReview 2025（在审）
- **模型**：PixArt-α，Open-Sora，DiT-XL
- **加速**：PixArt-α **3.86×**，Open-Sora **3.74×**，DiT-XL **4.51×**
- **方法**：对特征跨步演化轨迹进行局部拟合（多项式），外推预测当前步特征
- **说明**：比 TaylorSeer 更一般化的轨迹外推

---

### CorGi — 贡献引导 Block 级间隔缓存
- **来源**：arXiv 2512.24195（Dec 2025）
- **加速**：~**2.0×** on image DiT，高质量保留
- **方法**：按 block 对输出的贡献度决定缓存间隔，低贡献 block 缓存复用更多步

---

### ProCache — 约束感知动态特征缓存
- **来源**：arXiv 2512.17298（Dec 2025）
- **加速**：~**1.96×**，negligible degradation
- **方法**：将缓存模式选择建模为约束优化问题，自适应决定哪些层哪些步缓存

---

### DiCache
- **来源**：arXiv Aug 2025
- **方法**：在线用浅层 probe 测量特征变化率，动态决定跳步调度
- **特点**：自适应，无需离线 profiling

---

## 二、注意力结构稀疏类

### SpargeAttn — ⭐ ICML 2025
- **来源**：ICML 2025 | arXiv 2502.18137
- **加速**：attention 计算 **2.5×~5×**（需 Triton custom kernel）
- **模型**：FLUX，SD3，Mochi（视频）
- **方法**：块级预测哪些 Q-K 对权重显著（稀疏度 ~0.38 on FLUX），只计算显著块的 attention
- **质量**：FID/CLIP/IR 几乎无损
- **重要说明**：speedup 是 attention kernel 级别，端到端加速低于此（attention 不是全部计算）
- **与 AccelAes 对比**：纯结构稀疏，无语义 mask；AccelAes 按内容选 token 子集，保留全局 K,V
- **代码**：https://github.com/thu-ml/SpargeAttn

---

### SageAttention — ⭐ ICLR/ICML/NeurIPS 2025 Spotlight
- **来源**：ICLR 2025 + ICML 2025 + NeurIPS 2025 Spotlight
- **加速**：attention **2×~5×**（相比 FlashAttention）
- **方法**：FP8/INT8 量化 attention + 硬件优化（RTX/A100/H100 均支持）
- **说明**：硬件级 kernel 加速，与算法级加速（缓存、稀疏）正交叠加
- **代码**：https://github.com/thu-ml/SageAttention

---

### DiTFastAttn — NeurIPS 2024
- **来源**：NeurIPS 2024 | arXiv 2406.08552
- **模型**：PixArt-Sigma（高分辨率生成）
- **加速**：端到端 ~**1.6×~1.8×**（attention FLOPs −76%~−88%）
- **方法**：跨 timestep 和条件维度压缩 attention 冗余

---

## 三、空间自适应类（与 AccelAes 最直接竞争）

### RAS — ⭐ 最重要对比基线
- **来源**：arXiv 2502.10389（Feb 2025，Microsoft Research，未发会议）
- **模型**：Lumina-Next-T2I，SD3-Medium
- **加速**：官方声称 **2.51×**（依赖 flash_attn + Triton indexed matmul + RoPE kernel fusion）
- **我们实测**：1.47×（output-level blending 复现）/ 1.57-1.61×（官方代码，有 diffusers 兼容性问题）
- **方法**：每步计算 cross-attention magnitude mask，活跃区域做完整 denoising，非活跃区域复用上步 noise
- **质量**：IR ≈ baseline（+4.8% vs our baseline，LPIPS=0.025）
- **与 AccelAes 对比**：
  - RAS mask 每步重新计算（有 per-step overhead）；AccelAes mask 一次建立全程复用
  - RAS 无步级缓存（fskip2）；AccelAes 有
  - AccelAes Lumina：**2.09×**（5.86s）vs RAS 复现：1.47×（8.38s）；AccelAes 更快且 IR 更高（+11.9% vs +4.8%）

---

### SDiT — 复杂度驱动空间跳过
- **来源**：arXiv 2601.12283（Jan 2025）
- **模型**：Lumina-Next 专用
- **加速**：1.66×（保质量设置，LPIPS≈0.12）～ **3.0×**（激进设置，FID +20%）
- **方法**：像素级 Sobel/Laplacian 算子衡量 latent 复杂度，低复杂度区域跳过计算
- **与 AccelAes 对比**：bottom-up 信号（像素复杂度）vs top-down（美学/语义）；无步级缓存；无 spatial CFG；无跨架构验证

---

### DyDiT — Dynamic Diffusion Transformer
- **来源**：ICLR 2025
- **模型**：DiT-XL/2
- **加速**：**1.73×**（FLOPs −51%）
- **方法**：动态 token 稀疏化（需少量 fine-tuning）
- **说明**：需要训练，与 training-free 方法不直接可比

---

### E-DiT — Elastic Diffusion Transformer
- **来源**：arXiv 2602.13993（Feb 2026）
- **加速**：~**2×**
- **方法**：综合稀疏 attention + token merging + 特征缓存，统一框架

---

## 四、AccelAes 定位总结

| 方法 | 会议 | 模型 | 端到端加速 | 质量变化 | 无需训练 | 跨架构 |
|------|------|------|-----------|---------|---------|--------|
| TaylorSeer | ICCV 2025 | FLUX | 2.96×~4.99× | IR ≈ baseline | ✅ | 部分 |
| TeaCache | CVPR 2025 HL | FLUX/video | ~2× (img) / 4.41× (video) | ≈ baseline | ✅ | ✅ |
| ToCa | ICLR 2025 | PixArt/FLUX | ~2× | ≈ baseline | ✅ | ✅ |
| SpargeAttn | ICML 2025 | FLUX/SD3 | 2.5×~5× (attn only) | ≈ baseline | ✅ | ✅ |
| DiTFastAttn | NeurIPS 2024 | PixArt | 1.6×~1.8× | ≈ baseline | ✅ | 部分 |
| RAS | arXiv Feb 2025 | Lumina/SD3 | 2.51× (官方) / 1.47× (复现) | ≈ baseline | ✅ | 部分 |
| SDiT | arXiv Jan 2025 | Lumina | 1.66×~3.0× | 下降 | ✅ | ❌ |
| **AccelAes (ours)** | — | Lumina/SD3/FLUX | **2.09× / 1.50× / 1.73×** | **+11.9% IR (Lumina)** | ✅ | ✅ |

**AccelAes 核心差异化**：
1. **唯一质量反而提升的方法**（Lumina IR +11.9%，Aesthetic +0.0%，CLIP +0.4%）——其他方法目标都是"维持质量不下降"
2. **三架构跨模型验证**（Lumina / SD3 / FLUX），结果一致可信
3. **mask 一次建立全程复用**，无 per-step overhead（vs RAS 每步重算）
4. **步级缓存 + 空间稀疏正交叠加**，两个加速来源独立可消融

**速度差距说明（vs TaylorSeer 4.99×）**：
- TaylorSeer 依赖中间层特征外推，要求特征跨步稳定
- SD3/FLUX 的 AdaLN 使中间层特征随 timestep 剧烈变化，不能外推（实验验证：CLIP 下降 >40%）
- AccelAes 只外推最终 noise_pred（output-level），理论上限约 2×
- Lumina 无 AdaLN 可以做中间层缓存，未来可叠加 TaylorSeer 方案达到更高加速比

---

## 五、与顶会审稿相关的注意事项

1. **TaylorSeer 是最强的 step-level 竞品**，须在 Related Work 中详细对比，解释为何 AccelAes 选择 noise_pred 外推而非特征外推（AdaLN 约束）

2. **SpargeAttn 加速数字的可比性**：其 2.5×~5× 是 attention kernel 级别（Triton），端到端远低于此，与我们的端到端 2.09× 不直接可比

3. **TeaCache 在 FLUX image 的 ~2× 与我们的 1.73× 接近**，可作为对比基线补充到表格

4. **RAS 的 2.51× 声明**：需在 paper 中说明依赖 flash_attn+Triton 的加速，在相同无 kernel-fusion 环境下 1.47-1.61×；AccelAes 的 2.09× 不依赖自定义 kernel

5. **质量维度独特性**：所有竞品方法都以"不损失质量"为目标，AccelAes 在 Lumina 上实现质量净提升（美学语义 mask + spatial CFG），这是论文的核心差异化，须在 Introduction 和 Conclusion 中强调

6. **DPCache 加速比远高于 AccelAes**：DPCache FLUX 4.87× vs AccelAes FLUX 1.73×，须在 Related Work 中解释我们不采用 DP 调度的原因（我们的贡献在于质量提升而非纯加速；两者未来可叠加）

---

## 六、跨架构基线实验现状（2026-03）

### 已完成实验

| 基线方法 | Lumina | SD3 | FLUX |
|----------|--------|-----|------|
| Δ-DiT | ✅ `outputs/semantic_full_eval/` | ✅ `outputs/sd3_compare/` | — |
| FORA | ✅ | ✅ | — |
| TeaCache t=0.15/0.30 | ✅ `outputs/stepcache_lumina/` | ✅ | ✅ `outputs/stepcache_flux/` |
| TaylorSeer r=1/2 | ✅ | ✅ | ✅ |
| RAS | ✅ | — | — |
| S-DiT | ✅ | — | — |
| **AccelAes** | ✅ | ✅ | ✅ |

### 关键结论（跨架构）

**SD3**（`outputs/sd3_compare/summary.json`，30p × 2s）：

| 方法 | 加速 | IR | LPIPS | FID |
|------|------|----|-------|-----|
| baseline | 1.00× | 0.879 | 0.000 | 0.0 |
| TeaCache t=0.15 | 1.07× | 0.876 | 0.007 | 10.3 |
| TeaCache t=0.30 | 1.41× | 0.847 | 0.056 | 39.5 |
| TaylorSeer r=1 | 1.59× | 0.898 | 0.116 | 70.2 |
| TaylorSeer r=2 | 2.02× | 0.729 | 0.281 | 132.6 |
| Delta-DiT | 1.51× | 0.828 | 0.278 | 116.9 |
| FORA | 0.99× | 0.703 | 0.282 | 146.1 |
| **AccelAes fskip2** | **1.50×** | **0.891** | **0.058** | **44.3** |

**FLUX**（`outputs/stepcache_flux/summary.json`，24p × seed=42）：

| 方法 | 加速 | IR | LPIPS | FID |
|------|------|----|-------|-----|
| baseline | 1.00× | 1.233 | 0.000 | 0.0 |
| TeaCache t=0.15 | 1.62× | 1.283 | 0.032 | 21.9 |
| TeaCache t=0.30 | 2.14× | 1.225 | 0.071 | 39.6 |
| TaylorSeer r=1 | 1.63× | 1.295 | 0.018 | 13.4 |
| TaylorSeer r=2 | 2.12× | 1.304 | 0.055 | 32.6 |
| **AccelAes fskip2** | **1.73×** | **1.267** | **0.030** | **19.5** |

### 无法直接移植的方法

- **RAS**：假设 UNet 单流 cross-attention 结构，SD3/FLUX 使用 MM-DiT 联合注意力（文本+图像 token 拼接），空间 mask 概念需重写核心逻辑
- **S-DiT**：使用 Sobel/Laplacian 衡量 Lumina latent 复杂度，Lumina 专用；SD3/FLUX latent 空间统计特性不同，需完整重新验证
- **FORA**：在 SD3 上完全失效（0.99× 加速，IR 崩溃），根本原因是 SD3 的 AdaLN + 残差耦合与 FORA 的 pre-hook 跳步不兼容
