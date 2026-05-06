# Masked Diffusion 深度解读：图像生成的第三条路

> 作者：王磊 & 霉霉 ✨
> 日期：2026-05-06
> 关联文档：[原生多模态模型技术指南](./native-multimodal-guide.md)

---

## 目录

1. [定位：三种图像生成范式](#1-定位三种图像生成范式)
2. [什么是 Masked Diffusion](#2-什么是-masked-diffusion)
3. [数学原理](#3-数学原理)
4. [与连续扩散模型的对比](#4-与连续扩散模型的对比)
5. [与 AR 模型的对比](#5-与-ar-模型的对比)
6. [代表性模型详解](#6-代表性模型详解)
7. [并行解码的理论分析](#7-并行解码的理论分析)
8. [对 Infra 的影响](#8-对-infra-的影响)
9. [总结与展望](#9-总结与展望)

---

## 1. 定位：三种图像生成范式

```
                    图像生成模型
                   /      |      \
                  /       |       \
     连续扩散模型    AR 模型    Masked Diffusion
     (Diffusion)  (Autoregressive)  (离散扩散)
         |            |              |
    连续 latent   离散 token      离散 token
    加噪→去噪    从左到右逐个生成  随机 mask→并行填充
    UNet/DiT     Transformer      Transformer (双向)
         |            |              |
   SDXL/DALL-E3  Emu3/GPT-4o    Dynin-Omni/Omni-Diffusion
```

Masked Diffusion 不是凭空出现的，它的思想脉络：

- **BERT (2018)**：Masked Language Modeling，随机遮住 15% token 让模型预测
- **MaskGIT (2022, Google)**：把 MLM 做成迭代式，用于图像生成
- **MUSE (2023, Google)**：MaskGIT 思路的 text-to-image 模型
- **MDLM (2024, NeurIPS)**：严格证明 masked prediction = 离散扩散
- **LLaDA (2025)**：8B 规模 masked diffusion 语言模型，接近 LLaMA3
- **Dynin-Omni / Omni-Diffusion / Muddit (2026)**：多模态统一生成

---

## 2. 什么是 Masked Diffusion

### 直觉

把 BERT 的单步 mask-predict 改成**多步迭代**过程：

```
步骤0（全mask）：  [M] [M] [M] [M] [M] [M] [M] [M]  ← 什么都不知道
步骤1（揭露20%）：[M] [M] 🌊 [M] ☀️ [M] [M] [M]    ← 高置信度的先出
步骤2（揭露50%）：🏖️ [M] 🌊 🌴 ☀️ [M] 🐚 [M]     ← 越来越多
步骤3（揭露80%）：🏖️ 👙 🌊 🌴 ☀️ 🌅 🐚 [M]     ← 接近完成
步骤4（全揭露）：  🏖️ 👙 🌊 🌴 ☀️ 🌅 🐚 🦀      ← Done!
```

对比 AR（必须严格从左到右）：
```
步骤0：🏖️ → 步骤1：👙 → 步骤2：🌊 → ... → 步骤7：🦀
（8 个 token 必须 8 步，每步只出 1 个）
```

**关键区别**：Masked Diffusion 每步可以**同时填多个位置**，且填充顺序不固定（高置信度优先）。

### 与 BERT 的关系

| | BERT MLM | Masked Diffusion |
|---|---|---|
| mask 比例 | 固定 15% | 从 100% 逐步降到 0% |
| 步数 | 单步预测 | 多步迭代（8-64 步） |
| 用途 | 训练 encoder | 训练生成模型 |
| 推理 | 不生成 | 迭代生成 |

本质：Masked Diffusion = **多步 BERT**，把训练时的 mask-predict 直接变成推理时的生成策略。

---

## 3. 数学原理

### 前向过程（Forward Process）

在连续扩散中，前向过程是加高斯噪声：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

在 Masked Diffusion 中，前向过程是**随机 mask**：

```
q(x_t | x_0) = 每个 token 独立地：
  - 以概率 α(t) 保留原始 token
  - 以概率 1-α(t) 替换为 [MASK]

其中 α(t) 从 1（无 mask）单调递减到 0（全 mask）
```

### 反向过程（Reverse Process）

模型 pθ 预测每个 [MASK] 位置应该是什么：

```
p_θ(x_0 | x_t) = Transformer(x_t)  → 对每个 [MASK] 位置输出 vocabulary 分布
```

采样策略（每步）：
1. 模型预测所有 [MASK] 位置的 token 分布
2. 按置信度排序
3. 揭露 top-k 个高置信度位置（k 随步数递增）
4. 其余保持 [MASK]

### 训练目标

MDLM (NeurIPS 2024) 的关键证明：

```
最优训练 loss = Σ_t  λ(t) · L_MLM(mask_rate = 1-α(t))

即：不同 mask rate 下的 MLM loss 的加权和
```

这意味着训练极其简单 —— 就是用**不同 mask 比例**跑 BERT 的 MLM loss，然后加权求和。

### 与连续扩散的统一视角

| 概念 | 连续扩散 | Masked Diffusion |
|---|---|---|
| 前向噪声 | 高斯噪声 N(0,1) | mask（替换为特殊 token） |
| 噪声调度 | β(t), ᾱ(t) | α(t) mask 保留率 |
| 去噪网络 | 预测噪声 ε 或 x_0 | 预测原始 token |
| 损失函数 | MSE (连续) | Cross-entropy (离散) |
| ELBO | 连续变分下界 | 离散变分下界 |

两者在数学框架上是**同构的**，只是一个在连续空间，一个在离散空间。

---

## 4. 与连续扩散模型的对比

| 维度 | 连续扩散（SDXL/DiT） | Masked Diffusion |
|---|---|---|
| 操作空间 | 连续 float latent | 离散 token（和文本共享词表） |
| 前向噪声 | 高斯噪声 | 随机 mask |
| 骨干网络 | UNet 或 DiT | Transformer（类 BERT） |
| 解码器 | VAE decoder | VQ decoder |
| 多模态统一 | ❌ 困难（文本离散、图像连续） | ✅ 天然统一（都是离散 token） |
| 图像质量 | ⭐⭐⭐⭐⭐ 目前最强 | ⭐⭐⭐⭐ 接近但有差距 |
| 条件注入 | Cross-attention / CFG | Token 级混合，双向 attention |
| 可控编辑 | Inpainting/ControlNet 成熟 | 天然支持（直接 mask 要编辑的区域） |

### Masked Diffusion 的优势

1. **多模态统一**：文本、图像、语音都是离散 token，可以在同一个模型中联合建模
2. **天然支持编辑**：想修改图片的某个区域？直接 mask 那些 token 重新生成即可
3. **训练简单**：本质就是不同 mask rate 的 MLM，工程实现比连续扩散简单
4. **不需要 VAE**：直接在 VQ token 空间操作，减少一层信息损失

### 连续扩散的优势

1. **质量天花板更高**：连续空间无量化损失，细节保留更好
2. **成熟的加速手段**：一致性蒸馏、LCM、SDXL-Turbo 等可以 1-4 步出图
3. **生态完善**：ControlNet、IP-Adapter、LoRA 等工具链成熟
4. **分辨率灵活**：不受 codebook 大小限制

---

## 5. 与 AR 模型的对比

| 维度 | AR（GPT-4o/Emu3） | Masked Diffusion |
|---|---|---|
| 生成顺序 | 严格从左到右 | 无序，并行填充 |
| 注意力方向 | 单向 causal mask | **双向 full attention** |
| 每步输出 | 1 个 token | **多个 token（batch unmask）** |
| 生成步数 | O(N)，N=总 token 数 | O(T)，T=扩散步数（8-64） |
| 上下文 | 只能看已生成的（左侧） | 能看所有已揭露的（全局） |
| 文本推理 | 强（本身就是 LLM） | 需额外训练（LLaDA 证明可行） |
| 逆向任务 | 差（reversal curse） | 好（双向建模无方向偏置） |

### 速度对比实例

生成 1024 个图像 token：

```
AR 模型：
  → 1024 步 forward pass
  → 每步 KV cache append + 1 token decode
  → 总延迟 = 1024 × per_token_latency

Masked Diffusion（32 步）：
  → 32 步 full-sequence forward pass
  → 每步 batch predict + 揭露约 32 个位置
  → 总延迟 = 32 × full_seq_forward_latency
  → 虽然每步计算量更大（full attention vs causal），
    但步数少 32 倍，总体快 5-10x
```

### 全局一致性

AR 模型生成图片时，画到右下角时"忘了"左上角的内容（只能靠 KV cache 中的信息）。

Masked Diffusion 每步都对整张图做 full attention：
- 早期步骤：确定整体构图和色调
- 中期步骤：填充主体结构
- 后期步骤：精修细节和纹理

这类似于人类画画的过程：先画草稿 → 上色 → 精修，而不是从左上角画到右下角。

### Reversal Curse 的解决

AR 模型有 "reversal curse"（学了 A→B 不等于学了 B→A），因为单向注意力引入了方向偏置。

Masked Diffusion 使用双向注意力，对 A→B 和 B→A 的建模是对称的。LLaDA 论文实验显示，在反向诗词补全任务上超过 GPT-4o。

---

## 6. 代表性模型详解

### 6.1 MaskGIT (Google, 2022) — 奠基之作

**论文**：Masked Generative Image Transformer (CVPR 2022)

**核心思路**：
- 用 VQGAN 把图像编码为离散 token
- 训练一个双向 Transformer，随机 mask 一部分 token 让模型预测
- 推理时：从全 mask 开始，每步揭露最高置信度的 token

**意义**：第一次把 "iterative mask-predict" 用于图像生成，证明不需要 AR 也能生成高质量图像。

### 6.2 MDLM (NeurIPS 2024) — 理论突破

**论文**：Simple and Effective Masked Diffusion Language Models

**关键贡献**：
1. 证明 masked prediction 在理论上**等价于**离散扩散
2. 提出 Rao-Blackwellized 目标函数，简化训练
3. 用现代工程实践训练 encoder-only 模型，在语言建模上接近 AR

**训练 recipe**：
- 目标 = 不同 mask rate 的 MLM loss 的混合
- 支持 semi-autoregressive 采样（按 chunk 生成）
- 可以生成任意长度文本

### 6.3 LLaDA (2025, NeurIPS) — 规模化验证

**论文**：Large Language Diffusion Models
**规模**：8B 参数，从头训练

**关键发现**：
- Masked diffusion 在 8B 规模上与 LLaMA3 8B 性能接近
- 支持 pre-training + SFT 范式（和 AR LLM 一样的训练流程）
- In-context learning 能力有效
- **解决了 reversal curse**（双向建模的天然优势）

**架构**：标准 Transformer，但用双向 attention（不是 causal mask）

**意义**：证明 masked diffusion **不是玩具**，在 LLM 规模上也 work。

### 6.4 Dynin-Omni (Samsung, 2026.3) — 全模态统一

**论文**：Omnimodal Unified Large Diffusion Language Model
**arXiv**：2604.00007

**架构要点**：
- 统一处理 文本 + 图像 + 语音 + 视频理解
- 所有模态 tokenize 到共享离散空间
- 用 masked diffusion 做所有模态的生成
- Multi-stage training + model-merging modality expansion

**训练策略**：
```
Stage 1: 纯文本 masked diffusion pre-training
Stage 2: Model merge 引入图像能力
Stage 3: Model merge 引入语音能力
Stage 4: Omnimodal alignment（全模态对齐微调）
```

**成绩（19 个 benchmark）**：
- GSM8K: 87.6（数学推理）
- MME-P: 1733.6（多模态理解）
- VideoMME: 61.4（视频理解）
- GenEval: 0.87（图像生成质量）
- LibriSpeech WER: 2.1（语音识别）

**核心结论**：masked diffusion 作为统一范式，在所有模态上都能做到 competitive。

### 6.5 Omni-Diffusion (VITA-MLLM, 2026.3) — 纯粹路线

**论文**：Unified Multimodal Understanding and Generation with Masked Discrete Diffusion
**arXiv**：2603.06577

**与 Dynin-Omni 的区别**：
- 更"纯粹"：100% masked discrete diffusion，不混合 AR
- 直接建模联合分布 p(text, image, speech)
- Any-to-any：任意模态组合输入 → 任意模态组合输出

**技术细节**：
- 统一 mask-based discrete diffusion 处理所有模态
- 支持双模态和更复杂的多模态场景
- 在多模态 benchmark 上匹配或超越现有系统

### 6.6 Muddit (PKU/Skywork, ICLR 2026) — 工程落地

**论文**：Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model
**arXiv**：2505.23606

**创新点**：
- 第二代 Meissonic 模型
- 不从头训练，而是**复用预训练的 text-to-image backbone 的视觉先验**
- 加一个轻量 text decoder，实现多模态统一生成
- 纯离散扩散 Transformer

**工程价值**：
- 证明可以用预训练模型初始化（不需要从头烧钱训练）
- 在质量和效率上 competitive with 更大的 AR 模型
- ICLR 2026 accepted

---

## 7. 并行解码的理论分析

### Easy-First 策略

最优的揭露顺序是什么？信息论分析（arXiv:2602.00286）给出答案：

```
Easy-First：优先揭露低熵（高置信度）的位置

直觉：先把"容易的"定下来，给"难的"提供更多上下文
数学：当模型误差越大时，Easy-First 的收益越显著
```

### 并行解码的风险

每步同时揭露多个位置 = 假设这些位置**条件独立**。但实际上它们之间有相关性：

```
真实分布：p(x_i, x_j | context)  ← 联合分布
并行采样：p(x_i | context) · p(x_j | context)  ← 假设独立

偏差 = Total Correlation（总相关）
```

这种偏差会导致"不连贯"（incoherence）：
- 相邻 token 之间缺乏一致性
- 在空间上表现为图像局部不协调

### 缓解策略

| 策略 | 做法 | 效果 | 代价 |
|---|---|---|---|
| Remasking | 每步揭露后，低置信度的重新 mask | 提升一致性 | 有效步数减少 |
| Verification | 揭露后验证联合概率，不满足就重采样 | 理论最优 | 指数级计算开销 |
| 小 batch 揭露 | 每步只揭露少量位置 | 减少独立性假设偏差 | 总步数增加 |
| 分组策略 | 空间上相距远的位置一起揭露 | 减少局部相关性 | 需要启发式设计 |

**实践中**：大多数模型用 Easy-First + Remasking 的组合，在速度和质量间取得平衡。

---

## 8. 对 Infra 的影响

### 推理 Pattern 差异

| | 连续扩散 | AR | Masked Diffusion |
|---|---|---|---|
| 前向模式 | 固定步数 UNet/DiT forward | KV cache prefill + decode | 多轮 full-attention forward |
| KV Cache | 不需要 | 核心瓶颈 | 不需要（每步独立） |
| 计算特征 | 每步计算量固定 | 每步递增（seq 变长） | 每步计算量固定 |
| 内存特征 | 模型权重 + 中间 latent | 模型权重 + KV cache | 模型权重 + token 序列 |
| Batch 友好度 | 高 | 中（KV cache 碎片化） | 高（固定序列长度） |

### Serving 架构设计

```
传统 AR 模型 Serving：
  Prefill（长）→ Decode（多步，每步短）→ 需要 KV cache 管理
  → PD disaggregation、PagedAttention 等优化

Masked Diffusion Serving：
  每步都是 Full Forward（类似 Prefill）
  → 不需要 KV cache！
  → 不需要 PD disaggregation
  → 类似 batch inference，每步固定计算量
  → 天然适合 tensor parallelism
```

### 具体优势

1. **无 KV cache**：不需要管理动态内存，vLLM/SGLang 的核心复杂度消失
2. **固定计算图**：每步 FLOPs 相同，GPU 利用率稳定
3. **Batch 效率高**：所有序列长度相同，无 padding 浪费
4. **步数-质量可调**：latency SLA 紧就跑 8 步，质量优先跑 64 步

### 潜在挑战

1. **每步计算量大**：full attention over 全序列（比 AR 单步 decode 重）
2. **步数仍然多于蒸馏后的连续扩散**：8-64 步 vs 1-4 步（LCM/Turbo）
3. **缺乏成熟框架**：目前没有类似 vLLM 的 Masked Diffusion 专用推理引擎
4. **混合模态 serving**：如果同时支持文本理解（可能需要 AR）和图像生成（Masked Diffusion），需要路由策略

---

## 9. 总结与展望

### 三种范式的定位

| 范式 | 擅长 | 短板 | 代表 |
|---|---|---|---|
| 连续扩散 | 极致图像质量 | 难与文本统一 | SDXL, DALL-E 3, Flux |
| AR | 文本推理 + 多模态统一 | 生成速度慢 | GPT-4o, Emu3, Janus |
| Masked Diffusion | 速度 + 多模态统一 | 质量略逊、生态不成熟 | Dynin-Omni, LLaDA, Muddit |

### 2026 年的趋势判断

1. **Masked Diffusion 正在成为统一多模态的主流架构选择之一**
   - Dynin-Omni 和 Omni-Diffusion 证明了可行性
   - Muddit 证明了可以复用预训练权重（降低训练成本）

2. **AR + Masked Diffusion 的混合架构可能是最优解**
   - 文本理解/推理用 AR（利用 LLM 的强大能力）
   - 图像/语音生成用 Masked Diffusion（利用并行速度优势）
   - 共享 backbone，通过 attention mask 切换模式

3. **Infra 需要适配**
   - 现有推理框架（vLLM/SGLang/TGI）都是为 AR 设计的
   - Masked Diffusion 需要新的 serving 范式
   - 混合架构需要在同一 serving 实例中支持两种推理模式

### 关键论文列表

| 论文 | 会议/时间 | 贡献 |
|---|---|---|
| MaskGIT | CVPR 2022 | 开创 iterative mask-predict 图像生成 |
| MUSE | ICML 2023 | MaskGIT 思路的 text-to-image |
| MDLM | NeurIPS 2024 | 证明 masked prediction = 离散扩散 |
| LLaDA | NeurIPS 2025 | 8B 规模验证，接近 LLaMA3 |
| Muddit | ICLR 2026 | 复用预训练权重的统一离散扩散 |
| Dynin-Omni | arXiv 2026.3 | 首个 masked-diffusion omnimodal 模型 |
| Omni-Diffusion | arXiv 2026.3 | 纯 masked diffusion any-to-any |
| Generation Order (Zhang et al.) | arXiv 2026.1 | 并行解码的信息论分析 |

---

*Last updated: 2026-05-06*
