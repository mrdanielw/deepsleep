# 原生多模态模型技术指南：从 VQ-VAE 到 AR 图像生成

> 作者：王磊 & 霉霉 ✨  
> 日期：2026-05-01  
> 仓库：[mrdanielw/deepsleep](https://github.com/mrdanielw/deepsleep)

---

## 目录

1. [行业趋势概览](#1-行业趋势概览)
2. [核心概念：VQ (Vector Quantisation)](#2-核心概念vq-vector-quantisation)
3. [VQ-VAE 架构详解](#3-vq-vae-架构详解)
4. [VQ Decoder 模型结构](#4-vq-decoder-模型结构)
5. [AR Transformer 与普通 Transformer 的区别](#5-ar-transformer-与普通-transformer-的区别)
6. [原生多模态 vs 扩散模型对比](#6-原生多模态-vs-扩散模型对比)
7. [为什么 AR 模型画文字更精确](#7-为什么-ar-模型画文字更精确)
8. [代表性模型与论文](#8-代表性模型与论文)

---

## 1. 行业趋势概览

多模态模型正在从"拼接架构"转向"原生架构"：

| 维度 | 传统方案 | 原生多模态 |
|------|----------|-----------|
| **输入理解** | CLIP/SigLIP encoder → 连接层 → LLM | VQ tokenizer → 直接进 LLM vocabulary |
| **图像生成** | LLM 出 caption → 调用 Diffusion 模型 | LLM 直接预测 visual tokens → VQ decoder 解码 |
| **对齐** | 需要大规模 image-text 对比学习 | 不需要，联合训练自然学到对齐 |
| **一致性** | 理解和生成是两套系统 | 同一套权重，上下文天然打通 |

**一句话**：行业从"理解用 CLIP、生成用 Diffusion、中间靠对齐层粘合"→"一个 Transformer 吃所有模态的 token、一次 forward pass 出所有模态"。

---

## 2. 核心概念：VQ (Vector Quantisation)

**VQ = Vector Quantisation = 向量量化**

直觉：把连续的向量"四舍五入"到最近的码本条目上。

```
Codebook（码本）：一本"词典"，有 N 个固定向量（如 N=8192）

输入：连续向量 [0.23, -0.41, 0.87, ...]
操作：找到 codebook 里距离最近的码字
输出：离散 index（如 #3742）

→ 实现了 连续空间 → 离散空间 的转换
```

### 为什么需要离散化？

- LLM（语言模型）本质是在离散 token 上做 next-token prediction
- 图像原本是连续像素值
- VQ 是连接两者的桥梁：把图像变成"单词"让 LLM 能处理

### Codebook 有限，怎么表达千变万化的图像？

不是一个 token 表示整张图，是一组 token 的**排列组合**：

```
Codebook 大小 8192，一张图产生 256 个 token
表达能力：8192^256 ≈ 10^1001 种组合（远超宇宙原子数）
```

类似 26 个英文字母可以写出无穷多篇文章。

实践中提升表达力的方法：

| 技术 | 做法 | 效果 |
|------|------|------|
| 加大 codebook | 8K → 16K → 262K（MAGVIT-v2） | 更细腻的表达 |
| 多层量化（RQ/FSQ） | 一个 patch 用多个 codebook 级联表示 | 逐层补细节 |
| 减小 patch 尺寸 | 16×16 → 8×8 → 4×4 | token 更多但更精确 |
| Lookup-Free (LFQ) | 二进制化，不存显式码本 | 等效超大 codebook |

---

## 3. VQ-VAE 架构详解

**VQ-VAE = VQ Tokenizer（Encoder + 量化层）+ VQ Decoder，端到端训练。**

```
┌─────────────────── VQ-VAE ───────────────────┐
│                                               │
│  [原始图像 256×256×3]                          │
│      ↓                                        │
│  Encoder（CNN/ViT）→ 连续 feature map          │
│      ↓                                        │
│  Vector Quantisation → 离散 token 序列         │  ← VQ Tokenizer
│      ↓                                        │
│  Decoder（CNN/ViT）→ 重建图像                  │  ← VQ Decoder
│                                               │
└───────────────────────────────────────────────┘
```

### 训练方式

训练目标：让重建图像尽量接近原始图像（自监督，无需标注）。

```
Loss = L_reconstruction + L_codebook + L_commitment

L_reconstruction: ||原图 - 重建图||²
L_codebook:       ||sg[encoder输出] - codebook向量||²  (让码本靠近编码)
L_commitment:     ||encoder输出 - sg[codebook向量]||²  (让编码靠近码本)
```

> `sg` = stop gradient，用 straight-through estimator 绕过量化不可导问题

### Token 的本质

**每个 token 不对应固定像素块，而是某种图像局部语义特征。**

```
同一个 token #3742：
  周围是 [天空, #3742, 草地] → Decoder 还原成地平线渐变
  周围是 [墙壁, #3742, 窗户] → Decoder 还原成墙面边角

→ 具体像素取决于上下文（类似"bank"在不同句子里含义不同）
```

---

## 4. VQ Decoder 模型结构

以 VQGAN（主流 VQ-VAE 变体）为例：

```
输入：16×16 token grid → 查 codebook → 16×16×D feature map
      ↓
[Post-Quant Conv] 1×1 卷积（调整通道数）
      ↓  16×16×256
[Mid Block] ResBlock → Self-Attention → ResBlock  ← 全局上下文融合
      ↓  16×16×256
[UpSample Block 1] ResBlock × 2 → Upsample 2×
      ↓  32×32×256
[UpSample Block 2] ResBlock × 2 → Upsample 2×
      ↓  64×64×256
[UpSample Block 3] ResBlock × 2 → Upsample 2×
      ↓  128×128×128
[UpSample Block 4] ResBlock × 2 → Upsample 2×
      ↓  256×256×128
[Output Conv] 3×3 Conv → 3 channels + tanh
      ↓  256×256×3 (RGB)
```

### 各组件职责

| 组件 | 作用 | 边界连续性贡献 |
|------|------|---------------|
| **Self-Attention** | 16×16 上全局信息交互 | 每个 token 看到所有其他 token |
| **ResBlock (3×3 Conv)** | 局部特征精炼 | 感受野覆盖相邻 patch 边界 |
| **Upsample** | 空间分辨率 ×2 | 插值+卷积自然平滑 |
| **多层堆叠** | 感受野逐层扩大 | 最终每个像素受一大片 token 影响 |

### 全程是特征空间，只有最后一层变成像素

```
16×16×256  ← 特征空间（不是像素）
32×32×256  ← 特征空间
64×64×256  ← 特征空间
128×128×128 ← 特征空间
256×256×128 ← 特征空间
      ↓ Conv 3×3 (128→3)  ← 唯一从特征→像素的转换
256×256×3   ← RGB 像素
```

### 与 Stable Diffusion 的关系

VQ Decoder 和 Stable Diffusion 的 VAE Decoder **结构几乎完全相同**（都出自 CompVis/LDM 代码库）：

```
扩散模型：  UNet 去噪 → 连续 latent → [VAE Decoder] → 像素
AR 模型：   Transformer 预测 → 离散 token → [VQ Decoder] → 像素
                                            ↑
                                      结构几乎一样
                                      都是 "潜空间 → 像素" 的翻译器
```

差异仅在输入是连续 latent 还是离散 codebook lookup 的结果。

---

## 5. AR Transformer 与普通 Transformer 的区别

**AR = AutoRegressive = 自回归**：一个 token 一个 token 按顺序生成，每次只看已生成的部分。

| 维度 | AR Transformer (Decoder-only) | 普通 Transformer (Encoder) |
|------|------|------|
| **注意力** | Causal Mask（只看左边/前面） | 双向注意力（看全局） |
| **生成方式** | P(x₁) → P(x₂\|x₁) → P(x₃\|x₁,x₂) → ... | 一次看到全部输入 |
| **代表** | GPT、LLaMA、Janus | BERT、T5 encoder |
| **训练目标** | Next-token prediction | MLM、Seq2Seq 等 |

```
AR Transformer 的 Causal Attention Mask（下三角）：

Token:  t1  t2  t3  t4  t5
t1:     ✓   ✗   ✗   ✗   ✗
t2:     ✓   ✓   ✗   ✗   ✗
t3:     ✓   ✓   ✓   ✗   ✗
t4:     ✓   ✓   ✓   ✓   ✗
t5:     ✓   ✓   ✓   ✓   ✓

→ 每个位置只 attend 到自己和之前的 token
```

在多模态场景中，文本和图像 token 混在同一序列里，模型从左到右生成，天然支持交错输出。

---

## 6. 原生多模态 vs 扩散模型对比

### 图像质量

| 维度 | 扩散模型 | AR 原生多模态 |
|------|---------|-------------|
| 照片写实 | ⭐⭐⭐⭐⭐ 目前最强 | ⭐⭐⭐⭐ 接近但有差距 |
| 细节/纹理 | 极其精细 | 轻微模糊（量化损失） |
| 文字渲染 | ⭐⭐ 很差 | ⭐⭐⭐⭐⭐ 天然优势 |
| 语义准确性 | ⭐⭐⭐ 容易搞混 | ⭐⭐⭐⭐⭐ LLM 理解力加持 |
| 构图/布局 | 容易空间错乱 | 更合理（逻辑推理帮忙） |

### 生成速度

| | 扩散模型 | AR 原生多模态 |
|---|---|---|
| 典型步数 | 20-50 步去噪 | 256-1024 步（逐 token） |
| 每步计算 | 整张图 UNet forward（重） | 一个 token Transformer forward（轻） |
| 并行性 | 每步内高度并行 | 严格串行 |
| 512×512 时间 | ~2-5 秒 (A100) | ~5-15 秒 (A100) |
| 加速手段 | 蒸馏到 1-4 步 | Speculative decoding |

### 根本差异原因

```
扩散模型质量高：连续 latent 无信息损失 + 迭代精炼
AR 模型更准确：LLM 语言理解力 + 每步都 attend 文本
AR 模型速度瓶颈：分辨率↑ → token 数二次方增长
```

### 融合方案（前沿方向）

```
方案一：AR 出草图 + Diffusion 精修
方案二：AR 低分辨率 + Diffusion 超分辨率
方案三：Masked Diffusion（离散 token + 并行解码）
  → 代表：Dynin-Omni、Omni-Diffusion (2026)
```

---

## 7. 为什么 AR 模型画文字更精确

### 扩散模型画文字的问题

```
"咖啡杯上写着 Hello World"
    ↓
CLIP 编码 → 一个模糊的语义向量
    ↓
UNet 去噪时：
  - 不知道 H-e-l-l-o 是有序序列
  - 把每个字母当独立视觉 pattern
  - 去噪过程中无法纠偏
  → 经常写出 "Helo Wrold"
```

### AR 模型画文字的优势

```
"咖啡杯上写着 Hello World"
    ↓
LLM tokenizer：每个字/词是独立 text token
    ↓
生成图像 token 时，每一步都 attend 到原始文本 token
    → 知道现在该画哪个字
    → 知道画到哪了
    → 持续的文本信号约束 = 持续纠偏
```

**核心区别**：

| | 扩散模型 | AR 模型 |
|---|---|---|
| 文字理解 | CLIP 只知"有字"，不知具体结构 | LLM 直接持有 text token |
| 生成中纠偏 | 无（条件固定） | 每步 attention 回看文本 |
| 字形来源 | 从噪声"猜"字形 | 从训练数据"记住" pattern |

**比喻**：扩散模型是"蒙眼画字"，AR 模型是"看着字帖抄写"。

### 中文字符的挑战

AR 模型画中文本质不是"理解笔画后画出来"，而是"训练时见过足够多该字出现在图片中的样子"：

- 常见字（国、中、人）→ pattern 学得好，非常精准
- 罕见字（龘、靐）→ 训练数据少，可能变形

---

## 8. 代表性模型与论文

### 输入端：原生多模态 Tokenization

| 模型 | 机构 | 时间 | 核心做法 |
|------|------|------|----------|
| Emu3 | BAAI 智源 | 2024.9 | 图像/视频/文本全 tokenize，单一 Transformer，不用 CLIP 不用 Diffusion |
| Janus / Janus-Pro | DeepSeek | 2024.10/2025.1 | 解耦视觉编码（理解用 SigLIP，生成用 VQ），共享 AR backbone |
| LongCat-Next | 美团 | 2026.3 | 各模态词化为离散 token 直接进 LM |
| Emu3.5 | BAAI | 2025.10 | Emu3 升级，Native Multimodal World Learners |

### 输出端：文字+图像同时生成

| 模型 | 机构 | 时间 | 特点 |
|------|------|------|------|
| GPT-4o Native Image | OpenAI | 2025.3 | 商用落地，文字渲染极强 |
| Gemini 2.0 Flash | Google | 2024.12 | 原生图像+TTS输出 |
| Dynin-Omni | Samsung | 2026.3 | Masked diffusion omnimodal |
| Omni-Diffusion | - | 2026.3 | Masked discrete diffusion 统一理解+生成 |
| NextFlow | - | 2026.1 | Unified decoder-only sequential modeling |

### 关键论文

- [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869) (BAAI, 2024)
- [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848) (DeepSeek, 2024)
- [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://arxiv.org/abs/2501.17811) (DeepSeek, 2025)
- [VQ-VAE: Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (DeepMind, 2017)
- [VQGAN: Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) (CompVis, 2021)

---

## 对 Infra 的影响

1. **KV Cache 暴涨**：图像 token 化后 sequence 极长，对 serving 的 KV cache 管理压力很大
2. **推理延迟**：AR 逐 token 生成图片较慢，masked diffusion 是并行化的 tradeoff
3. **统一 Serving**：不再需要分别部署 LLM + Diffusion pipeline，一个 endpoint 搞定
4. **Tokenizer 关键**：VQ-VAE/MAGVIT 质量决定图像质量，codebook 大小影响 vocabulary 膨胀
5. **未来趋势**：serving pipeline 可能需同时支持 AR + Diffusion 两种推理模式

---

*Last updated: 2026-05-01*
