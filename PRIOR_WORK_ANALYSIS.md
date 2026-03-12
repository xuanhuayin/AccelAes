# SASD 贡献点新颖性分析 & 前人工作调研

> 2026-02-22

---

## 总览

| # | 贡献点 | 新颖？ | 最相关前人工作 | 区分点 |
|---|--------|--------|----------------|--------|
| 1 | CFG-magnitude mask | **否** | Importance-based ToMe (Wu et al., 2024) | 相同信号，不同应用 |
| 2 | CLIP anchor semantic mask | **是**（组合新颖） | SToRI, OVAM, S-CFG | 新 pipeline: anchor集 + CLIP匹配 + 加权cross-attn heatmap |
| 3 | Q_fg @ K_all 稀疏注意力 | **部分** | RAS (Microsoft, 2025) | RAS 用缓存的 K,V; SASD 用当前步新鲜计算的 K,V |
| 4 | 空间自适应 CFG | **部分** | S-CFG (CVPR 2024) | S-CFG 目标是均衡质量; SASD 故意差异化以加速 |
| 5 | 四层统一加速框架 | **是** | DyDiT, RAS, PAB, SDiT | 无前人统一 mask→sparse attn→sparse FFN→step skip |

---

## 详细分析

### 贡献 1: CFG-Magnitude Mask（前景检测）

**信号**: `|noise_pred_cond - noise_pred_uncond|` 作为免费的前景重要性信号

**结论: 不新颖 — 已有前人工作**

**关键前人工作:**

1. **Importance-Based Token Merging for Diffusion Models** (Wu et al., arXiv 2411.16720, Nov 2024)
   - 使用 **完全相同的信号**: `importance = |ε_θ(x_t|y,t) - ε_θ(x_t,t)|`
   - 用于 token merging: 创建 "important token pool"，保护高 CFG-magnitude 区域不被激进合并
   - 也强调 "零额外计算开销"

2. **RAS: Region-Adaptive Sampling** (Microsoft, arXiv 2502.10389, Feb 2025)
   - 使用相关但不同的信号: noise 标准差来区分 fast/slow update 区域

3. **SDiT** (arXiv 2601.12283, Jan 2026)
   - 使用梯度+拉普拉斯+残差噪声的混合复杂度度量，非 CFG difference

**SASD 的区分点**: 信号本身不新颖，但应用方式不同（用于稀疏 attn/FFN/step-skip 而非 token merging）。**必须引用 Wu et al.**

---

### 贡献 2: CLIP Anchor Semantic Mask

**方法**: CLIP 文本编码器 → 匹配 prompt tokens 与预定义 aesthetic/content anchors → 计算 token importance → 加权 cross-attention heatmap → 空间 mask

**结论: 组合新颖**

**相关前人工作:**

1. **SToRI** (arXiv 2410.08469, Oct 2024) — 基于上下文重要性重新加权 CLIP tokens，但用于通用 CLIP embedding，不是 diffusion 加速
2. **OVAM** (CVPR 2024, arXiv 2403.14291) — 用 cross-attention maps 生成空间 heatmap 做语义分割，不是加速
3. **S-CFG** (CVPR 2024, arXiv 2404.05384) — 用 cross-attention maps 分割语义区域做 CFG 重缩放
4. **T-GATE** (arXiv 2404.02747, 2024) — 分析 cross-attention 时间收敛性，不做 token 重要性加权

**SASD 的新颖之处**: 三步 pipeline（CLIP anchor 匹配 → token importance → importance-weighted cross-attn heatmap → spatial mask）**在加速场景下是全新的**。Cross-attention heatmap 存在，CLIP token 重加权存在，但通过 aesthetic/content anchor 集组合起来做加速 mask 是新的。

**这是最强的新颖性点之一，建议作为主要贡献。**

---

### 贡献 3: Q_fg @ K_all, V_all 稀疏自注意力

**方法**: 前景 token 的 Q 与全部 K,V 做 attention（保持全局上下文），背景 token 复用缓存

**结论: 部分新颖**

**关键前人工作:**

1. **RAS** (Microsoft, arXiv 2502.10389, Feb 2025)
   - `O_active = softmax(Q_active @ [K_active, K_cached_inactive]^T) @ [V_active, V_cached_inactive]`
   - Active queries attend to all K,V，但 inactive 的 K,V 是**上一步缓存的**（不是当前步新算的）
   - **非常接近 SASD 的方法**

2. **ToSA: Token Selective Attention** (arXiv 2406.08816, 2024) — 选择重要 token 做 attention，未选中的完全跳过（不是 attend to all K,V）

3. **DyDiT** (ICLR 2025, arXiv 2410.03456) — 只对 MLP 做选择性计算，attention 仍然全量

4. **标准 token pruning (ToMe 等)** — 直接删除 token，被删 token 从 Q 和 K,V 中都消失

**SASD vs RAS 的关键区别:**
- RAS: `Q_active @ [K_active, K_cached]` — inactive K,V 来自上一步缓存（有 staleness）
- SASD: `Q_fg @ K_all, V_all` — K,V 在当前步对全部 token 新鲜计算（无 staleness）
- SASD 的 K,V 投影仍在全部 N tokens 上做（O(N) linear），真正省的是 SDPA 中 O(N²) → O(N_fg × N)

**必须引用 RAS，明确说明 K,V 是新鲜计算的而非缓存。**

---

### 贡献 4: 空间自适应 CFG（逐像素 guidance scale）

**方法**: `s(h,w) = s_bg + mask(h,w) × (s_fg - s_bg)`

**结论: 部分新颖**

**关键前人工作:**

1. **S-CFG** (Shen et al., CVPR 2024, arXiv 2404.05384)
   - 逐区域自适应 CFG scale，使用 cross-attention 和 self-attention 做语义分割
   - 但目标是**均衡化** guidance（让所有区域获得相似的有效 guidance），不是为了加速
   - 公式: `γ_i = γ × |m_b · η| / |m_i · η|`

2. **LF-CFG** (arXiv 2506.21452, June 2025) — 空间 varying mask 降低低频冗余区域权重，用于质量改善

3. **Pixel-wise Guidance** (arXiv 2212.02024, Dec 2022) — 逐像素 guidance 用于编辑，非加速

**SASD 的区分点**: S-CFG 均衡化质量，SASD 故意差异化（前景高 CFG=质量，背景低 CFG=允许更激进的缓存/跳过）。目标完全不同。**必须引用 S-CFG 并明确区分 motivation。**

---

### 贡献 5: 四层统一加速框架

**方法**: content-adaptive mask → sparse attention → sparse FFN → step-level caching

**结论: 作为统一框架是新颖的**

**前人各组件:**

| 组件 | 前人工作 |
|------|----------|
| Content-adaptive mask | Importance-based ToMe (2024), RAS (2025), SDiT (2026) |
| Sparse attention | DiTFastAttn (NeurIPS 2024), RAS (2025), ToSA (2024) |
| Selective FFN | DyDiT (ICLR 2025) |
| Step-level caching | DeepCache (CVPR 2024), FORA (2024), Delta-DiT (2024), PAB (2024) |

**最接近的组合方法:**

1. **DyDiT** (ICLR 2025): timestep-adaptive width + spatial-selective MLP, 无 sparse attention
2. **E-DiT** (Feb 2026): block skipping + MLP width reduction, 无 content-driven spatial mask
3. **SDiT** (Jan 2026): semantic clustering + complexity scheduling, 区域级调度非算子级
4. **RAS** (Feb 2025): mask + sparse attention + step caching, **无 sparse FFN**
5. **PAB** (2024): attention caching + MLP caching + step skipping, **全 token 统一处理**（非 content-adaptive）

**没有前人将所有四层在同一个 content-adaptive mask 下统一起来。这是另一个强新颖性点。**

---

## 论文定位建议

### 主要贡献（按强度排序）:

1. **🟢 最强: CLIP anchor semantic mask** — 完全新颖的 pipeline，建议作为 contribution #1
2. **🟢 最强: 四层统一框架** — 组合新颖，建议作为 contribution #2
3. **🟡 中等: Q_fg @ K_all (fresh K,V)** — 与 RAS 相近但有区别，作为技术贡献
4. **🟡 中等: 空间自适应 CFG** — 与 S-CFG 相近但目标不同，作为技术贡献
5. **🔴 最弱: CFG-magnitude mask** — 不新颖，作为实现细节或 ablation 选项

### 必须引用的论文:

| 论文 | 原因 |
|------|------|
| **Importance-Based Token Merging** (Wu et al., 2024) | CFG-magnitude 信号的先驱 |
| **S-CFG** (Shen et al., CVPR 2024) | 空间 varying CFG 的先驱 |
| **RAS** (Microsoft, 2025) | Q_active @ K_all 的先驱 |
| **DyDiT** (ICLR 2025) | Selective FFN 的先驱 |
| **SDiT** (2026) | Content-adaptive 区域调度 |
| **DeepCache** (CVPR 2024) | Step-level caching |
| **Delta-DiT** (NeurIPS 2024) | Layer-level caching |
| **FORA** (ECCV 2024) | 残差复用 |
| **PAB** (2024) | 多层 attention broadcasting |
| **DiTFastAttn** (NeurIPS 2024) | DiT attention 加速 |
| **ToMe** (2023) | Token merging |

---

## 参考文献链接

- [Importance-Based Token Merging](https://arxiv.org/abs/2411.16720)
- [S-CFG (CVPR 2024)](https://arxiv.org/abs/2404.05384)
- [RAS (Microsoft)](https://arxiv.org/abs/2502.10389)
- [SDiT](https://arxiv.org/abs/2601.12283)
- [DyDiT (ICLR 2025)](https://arxiv.org/abs/2410.03456)
- [Delta-DiT](https://arxiv.org/abs/2406.01125)
- [PAB](https://arxiv.org/abs/2408.12588)
- [DiTFastAttn (NeurIPS 2024)](https://arxiv.org/abs/2406.08552)
- [DeepCache (CVPR 2024)](https://arxiv.org/abs/2312.00858)
- [FORA](https://arxiv.org/abs/2407.01425)
- [T-GATE](https://arxiv.org/abs/2404.02747)
- [ToSA](https://arxiv.org/abs/2406.08816)
- [E-DiT](https://arxiv.org/html/2602.13993v1)
- [FasterDiffusion (NeurIPS 2024)](https://arxiv.org/abs/2312.09608)
- [SToRI](https://arxiv.org/html/2410.08469v1)
- [OVAM (CVPR 2024)](https://arxiv.org/abs/2403.14291)
- [LF-CFG](https://arxiv.org/abs/2506.21452)
- [ToMe for SD](https://arxiv.org/abs/2303.17604)
- [QDM](https://arxiv.org/abs/2503.12015)
