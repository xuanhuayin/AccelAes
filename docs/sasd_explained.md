# SASD 大白话解释

## 1. 核心观察：并非所有区域都对图像质量有相同贡献

一句 prompt 比如 "a photorealistic portrait with intricate jewelry and bokeh background"，里面的词可以分成两类：

- **美学描述词**：photorealistic、intricate、detailed、cinematic、high quality——这些词告诉模型"怎么画"、"画到什么质量水平"
- **内容/背景词**：portrait、jewelry、bokeh background——这些词告诉模型"画什么"

在去噪过程中，对应美学描述词影响的图像区域，模型每步都在精细调整细节——这些是**美学关注区域**，必须精细计算。对应纯内容或背景的区域，去噪到一定程度就趋于稳定，中后期变化很小——这些是**低关注区域**，可以复用缓存。

现有方法把所有 patch 一视同仁，每步都全量计算——大量算力花在了对最终质量几乎没有边际贡献的区域上。

---

## 2. 怎么找"美学关注区域"（Semantic Mask）

**核心思路：先找 prompt 里的美学关键词，再看它们影响哪些图像 patch。**

### 第一步：识别 prompt 中的美学锚定词

预先定义一组**美学锚点词集合**（从 Pick-a-Pic 37K 条真实用户 prompt 数据中挖掘），涵盖 5 类：
- 纹理/细节：detailed、intricate、fine texture、sharp
- 质感/真实感：photorealistic、high quality、4K、cinematic
- 风格/美学：beautiful、stunning、masterpiece、artistic
- 光影：dramatic lighting、bokeh（主体处）、volumetric
- 人物/结构：portrait、full body、face detail

用 CLIP 文本编码器将每个 prompt token 的 embedding 与锚点词集合做相似度匹配，**相似度 > 0.60 的 token 被标为"美学相关 token"**（约 20% 的 prompt 能触发明确的美学锚定；其余退化为全局 cross-attention 均匀权重）。

### 第二步：用 cross-attention 映射到 patch 空间

在第 **4 步**（warmup 期末）正常做一次 forward，同时捞出 cross-attention affinity 矩阵 `Q_img @ K_text^T`：
- 对每个图像 patch，只看它对"美学相关 token"列的 attention 分数，取 max
- 这个分数 = "这个 patch 受 prompt 中美学描述词影响的强度"

### 第三步：空间聚合 + 生成 mask

1. 用 **SLIC 超像素分割**把相邻 patch 聚成约 64 个空间连贯区域，每区域取平均分数
2. 按分数排序，取前 50% 为**美学关注区域**（需要精细计算），后 50% 为**低关注区域**（可复用缓存）
3. 这个 mask **固定一次，后续所有步骤复用**

信号是"免费的"——cross-attention 本来就是 transformer forward 的一部分，锚点匹配只做一次，无额外推理开销。

---

## 3. 怎么对低关注区域省计算

得到 mask 后，从**第 5 步**开始：

**Sparse Attention（仅 Lumina）：**
- 美学关注区域 token：正常算 `Q @ K_all, V_all`（能看到全局信息）
- 低关注区域 token：直接复用上一步缓存的 attention 输出，不重算

**Sparse FFN（仅 Lumina）：**
- 美学关注区域：正常过 FFN
- 低关注区域：复用上一步 FFN 输出

**Spatial CFG（仅 Lumina + SD3）：**
- 美学关注区域：用高 guidance scale（`s_fg=9.0`），严格遵从 prompt 美学描述
- 低关注区域：用低 guidance scale（`s_bg=2.0`），保持自然感

---

## 4. 步级缓存（fskip2）——独立的第二层加速

**原理：** 相邻去噪步的 noise_pred 变化近似线性，可以外推跳步。

- **前 5 步（warmup）**：每步正常跑，存下 noise_pred
- **第 5 步之后**：每隔 1 步直接用前两步 noise_pred 线性外推，跳过整个 transformer forward
- 28 步实际只跑 ~16 步，节省约 43% 的 forward 次数

这个加速和上面的 sparse 计算是**正交叠加**的：sparse 让每次 forward 更快，fskip2 让 forward 次数更少。

---

## 5. 完整时间线（Lumina 28 步）

```
步  0-3 : warmup，全量计算，存 noise_pred
步  4   : 全量计算 + 提取 cross-attention affinity
          → 识别 prompt 中美学锚定词 → 生成美学关注 mask
步  5   : 安装 sparse processors（先跑一次 dense 填充缓存）
步  6   : fskip2 跳过（线性外推，不跑 transformer）
步  7   : sparse mode — 美学关注区域新鲜算，低关注区域复用缓存
步  8   : fskip2 跳过
步  9   : sparse mode
...
步 27   : fskip2 跳过
```

两层加速叠加 → **2.09× 总加速**（Lumina）

---

## 6. 各模型支持情况

| 组件 | Lumina | SD3 | FLUX |
|---|---|---|---|
| 美学 mask（锚定词 + cross/joint attn + SLIC） | ✅ | ✅ | ✅ |
| Sparse attention | ✅ | ❌ AdaLN耦合 | ❌ overhead过高 |
| Spatial CFG（s_fg / s_bg） | ✅ | ✅ | ❌ 无双路CFG |
| Sparse FFN | ✅ | ❌ AdaLN耦合 | ❌ AdaLN耦合 |
| Step cache（fskip2） | ✅ | ✅ | ✅ |
| **实测加速** | **2.09×** | **1.50×** | **1.73×** |

---

## 7. 和 SDiT 的区别

SDiT（arxiv 2601.12283）同样做区域选择性计算，但：

| | SASD | SDiT |
|---|---|---|
| 区域重要性信号 | **美学锚定词** → cross-attention affinity（一次提取，免费） | x_pred 像素梯度（每步重新算，有额外开销） |
| 信号语义 | prompt 中美学描述词对 patch 的影响强度 | 像素级空间复杂度（Laplacian + Sobel） |
| 步级缓存 | ✅ fskip2（正交叠加） | ❌ 没有 |
| 可比质量下速度 | **2.09×** | 1.66×（ratio=0.5） |
| SDiT 的 3× | — | 激进设置（FID +20%，LPIPS=0.20） |
| 测试模型 | Lumina + SD3 + FLUX | 仅 Lumina |
