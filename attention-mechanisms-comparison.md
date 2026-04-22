# Attention 机制最新进展与多维度对比（2024-2025）

> 整理时间：2026-04-22
> 
> 覆盖范围：稀疏 Attention、线性 Attention、Recurrent/SSM、Latent Attention、混合架构、计算优化

---

## 一、最新进展概览

### 1. 稀疏 Attention（Sparse Attention）

#### NSA — Native Sparse Attention（DeepSeek，2025.02）
- **论文：** [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
- **核心思路：** 动态分层稀疏策略 = **粗粒度 token 压缩** + **细粒度 token 选择**，同时保持全局上下文感知和局部精度
- **关键创新：**
  - 算术强度平衡的算法设计，针对现代 GPU 做了 hardware-aligned 优化
  - **支持端到端训练**（不像很多稀疏方案只能推理用），pretraining 就能省算力
- **效果：** 64K 长序列上 decode/forward/backward 全流程都比 Full Attention 有显著加速，benchmark 表现持平或超过 Full Attention

#### MoBA — Mixture of Block Attention（Moonshot/Kimi，2025.02）
- **论文：** [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)
- **核心思路：** 把 MoE 的思想搬到 Attention 层 — 每个 query 通过路由机制选择性地 attend 到若干 **block** 而非全部 token
- **关键特点：**
  - 遵循 "less structure" 原则，不预设 sink/window 等偏置，让模型自己学 where to attend
  - **可以在 Full Attention 和 Sparse Attention 之间无缝切换**
- **落地状态：** 已部署在 **Kimi 线上长上下文请求**中

#### Diff Transformer — Differential Transformer（微软，2024.10 → ICLR 2025 Oral）
- **论文：** [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)
- **核心思路：** 差分注意力 — 用 **两组 softmax attention map 的差** 作为最终 attention score
- **效果：** 天然促进稀疏 attention pattern 的涌现，减少 hallucination，长上下文/ICL 表现优异
- **亮点：** 对 attention 噪声的消除很优雅，相当于"去噪版 softmax attention"

---

### 2. 线性 Attention（Linear Attention）

#### GLA — Gated Linear Attention（MIT/CMU，2023.12 → 持续更新中）
- **论文：** [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)
- **核心思路：** 线性 attention + **data-dependent gates**，可以同时表示为 RNN（2D 矩阵状态）
- **关键创新：**
  - **FlashLinearAttention** — 针对 GPU 的 I/O-aware 算法，在 1K 短序列上就比 FlashAttention-2 快
  - 长度泛化能力强：2K 训练 → 20K+ 推理 perplexity 不崩
  - 训练吞吐量高于同规模 Mamba

#### DeltaNet — Linear Transformer with Delta Rule（2024.06 → NeurIPS 2024）
- **论文：** [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- **核心思路：** 把线性 attention 的加性更新换成 **delta rule**（Widrow-Hoff 学习规则），大幅提升 associative recall 能力
- **1.3B 模型 100B tokens** 训练：perplexity 和下游 zero-shot 都优于 Mamba 和 GLA
- **混合架构很香：** DeltaNet + sliding-window attention 或 DeltaNet + 2 层 global attention，超过强 Transformer baseline

#### DiG — Diffusion Gated Linear Attention Transformer（2024.05）
- **论文：** [arXiv:2405.18428](https://arxiv.org/abs/2405.18428)
- **核心思路：** 把 GLA 引入 2D 扩散模型骨干
- **效果：** DiG-S/2 在 1792 分辨率下比 DiT-S/2 快 **2.5×**，省 75.7% GPU 显存；DiG-XL/2 在 2048 分辨率下比 DiT+FlashAttention-2 快 **1.8×**

---

### 3. Recurrent / 状态空间模型（带 Recurrent 性质）

#### Mamba-2 / SSD（Albert Gu，2024.05 → ICML 2024）
- **论文：** [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **核心贡献：** 建立了 SSM ↔ Attention 的统一理论框架（**State Space Duality**）
  - SSM、线性 attention、半可分矩阵（semiseparable matrix）三者等价
- Mamba-2 核心层比 Mamba 快 **2-8×**，语言建模持平 Transformer

#### RWKV-5 (Eagle) / RWKV-6 (Finch)（RWKV 社区，2024.04）
- **论文：** [arXiv:2404.05892](https://arxiv.org/abs/2404.05892)
- **核心升级：**
  - **Multi-headed matrix-valued states**（矩阵值隐状态，比向量大得多）
  - **Dynamic recurrence**（数据依赖的递推系数，类似 Mamba 的选择性机制）
- **效果：** 0.46B-7.5B 规模，跨多种 benchmark 和 Mamba/Transformer 持平

#### HGRN2 — Hierarchically Gated Linear RNN（2024.04 → COLM 2024）
- **论文：** [arXiv:2404.07904](https://arxiv.org/abs/2404.07904)
- **核心思路：** 通过 **外积（outer product）状态扩展** 把 recurrent state 从向量扩到矩阵，不引入额外参数
- 同时提供线性 attention 解释，支持 hardware-efficient training

---

### 4. 混合架构 / Hybrid（趋势方向）

当前最热的实践路径：**不是二选一，而是混着用**

| 混合方案 | 组合方式 | 效果 |
|---------|---------|------|
| DeltaNet + Sliding Window | 交替层（linear + local attention） | 超过纯 Transformer baseline |
| DeltaNet + 2层 Global Attention | 大部分层 linear + 少量 full attention | 最佳 perplexity |
| Mamba + Attention (Jamba 架构) | MoE + Mamba + Attention | AI21 Labs 的 Jamba 系列 |
| MLA + MoE (DeepSeek-V2/V3) | Latent Attention + 稀疏专家 | KV cache 压缩 93.3%，吞吐 5.76× |

**MLA (Multi-head Latent Attention)** 特别值得关注：
- DeepSeek-V2 提出（[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)）
- 把 KV 压缩到一个低秩 latent vector → KV cache 直接砍到原来的 ~7%
- 已在 DeepSeek-V3、Kimi K2.5 等大模型中验证

---

### 5. Attention 计算优化

#### SageAttention（清华，2024.10 → ICLR 2025）
- **论文：** [arXiv:2410.02367](https://arxiv.org/abs/2410.02367)
- **8-bit 量化 attention**，即插即用
- OPS 比 FlashAttention-2 快 **2.1×**，比 xformers 快 **2.7×**
- 精度优于 FlashAttention-3

#### nGPT — Normalized Transformer（NVIDIA，2024.10）
- **论文：** [arXiv:2410.01131](https://arxiv.org/abs/2410.01131)
- 所有向量（embedding、MLP、attention 矩阵、hidden states）全部做 unit norm 归一化
- token 在超球面上"旅行"，训练收敛速度快 **4-20×**

#### FlashInfer（UW/NVIDIA，2025.01 → MLSys 2025）
- **论文：** [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
- 可定制的 attention 引擎，block-sparse KV cache + JIT 编译
- 已集成到 SGLang、vLLM、MLC-Engine
- 长上下文推理延迟降低 28-30%

---

## 二、多维度对比

### 计算复杂度

| 方案 | Prefill 复杂度 | Decode 复杂度(每 token) | 训练并行性 |
|------|:------------:|:---------------------:|:---------:|
| **Full Softmax** | O(N²d) | O(Nd) | ✅ 完全并行 |
| **NSA** | O(N·k·d)，k≪N | O(k·d) | ✅ 端到端可训 |
| **MoBA** | O(N·B·d)，B=选中 block 数 | O(B·block_size·d) | ✅ 可训 |
| **Diff Transformer** | O(N²d)（×2组） | O(Nd)（×2组） | ✅ 可训 |
| **GLA** | O(Nd²) chunk-wise | **O(d²)** 常数 | ✅ FlashLinearAttention |
| **DeltaNet** | O(Nd²) chunk-wise | **O(d²)** 常数 | ✅ Householder 并行算法 |
| **Mamba-2** | O(Nd·s) s=state dim | **O(d·s)** 常数 | ✅ SSD 框架 |
| **RWKV-6** | O(Nd²) | **O(d²)** 常数 | ✅ time-parallel |
| **MLA** | O(N²d_c) d_c≪d | O(N·d_c) | ✅ 可训 |

> 线性 Attention / Recurrent 系列在 decode 阶段是 O(1) 相对于序列长度 — 对 serving 吞吐量是质变。

---

### 表达能力 & 模型质量

| 方案 | 语言建模 PPL | Long-Context | In-Context Learning | Associative Recall |
|------|:-----------:|:------------:|:------------------:|:------------------:|
| **Full Softmax** | 🥇 基准线 | ✅ 强（N²代价） | ✅ 强 | ✅ 强 |
| **NSA** | ≈ Full | ✅ 强且快 | ✅ 持平 | ✅ 细粒度选择保障 |
| **MoBA** | ≈ Full | ✅ 已部署 Kimi | ✅ 持平 | ✅ block 路由覆盖 |
| **Diff Transformer** | 优于 Full | ✅ 更强（去噪） | ✅ 更强且更鲁棒 | ✅ 强 |
| **GLA** | 接近 Full | ✅ 长度泛化好 | ⚠️ 略弱 | ⚠️ 弱于 softmax |
| **DeltaNet** | 优于 GLA/Mamba | ✅ 好 | ⚠️ 中等 | ✅ 线性系列最强 |
| **Mamba-2** | ≈ Transformer | ✅ 好 | ⚠️ 中等 | ⚠️ 弱 |
| **RWKV-6** | 接近 Transformer | ✅ 理论无限 | ⚠️ 中等 | ⚠️ 中等 |
| **MLA** | ✅ 持平 Full | ✅ 可扩展 | ✅ 持平 | ✅ 持平 |
| **混合架构** | 🥇 超过纯 Transformer | ✅ 强 | ✅ 强 | ✅ 强 |

> 纯线性/Recurrent 的最大短板是 associative recall，DeltaNet 的 delta rule 显著缓解但仍不如 full attention。混合架构是当前最优解。

---

### KV Cache / 内存占用

| 方案 | KV Cache 大小 | 推理显存增长 | 长序列友好度 |
|------|:------------:|:----------:|:-----------:|
| **Full Softmax** | 2·N·d（随 N 线性增长） | 📈 线性 | ❌ |
| **NSA** | < 2·N·d | 📈 亚线性 | ✅ |
| **MoBA** | 2·N·d（全量存） | 📈 线性 | ⚠️ 算力省但显存没省 |
| **Diff Transformer** | 4·N·d（2组 KV） | 📈 线性×2 | ❌ 显存更大 |
| **GLA** | 固定 d×d 矩阵 | 📊 **O(1)** | ✅✅✅ |
| **DeltaNet** | 固定 d×d 矩阵 | 📊 **O(1)** | ✅✅✅ |
| **Mamba-2** | 固定 d×s 矩阵 | 📊 **O(1)** | ✅✅✅ |
| **RWKV-6** | 固定 d×d 矩阵 | 📊 **O(1)** | ✅✅✅ |
| **MLA** | N·d_c，d_c ≈ 0.07d | 📈 线性但压缩~93% | ✅✅ |

---

### 训练效率

| 方案 | 训练吞吐量(相对 FA2) | 训练稳定性 | 落地成熟度 |
|------|:------------------:|:---------:|:---------:|
| **Full Softmax + FA2** | 1.0× | ✅ 最成熟 | ✅✅✅ |
| **NSA** | >1.0× | ✅ | ✅✅ DeepSeek |
| **MoBA** | >1.0× | ✅ | ✅✅✅ Kimi 生产 |
| **Diff Transformer** | ~0.9× | ✅ | ✅ ICLR 2025 |
| **GLA** | >1.0× | ✅ | ✅ 开源 |
| **DeltaNet** | ~0.8-1.0× | ✅ | ✅ NeurIPS 2024 |
| **Mamba-2** | 高 | ⚠️ 需特殊初始化 | ✅✅ |
| **RWKV-6** | 中等 | ✅ | ✅ 社区活跃 |
| **MLA** | ~1.0× | ✅ | ✅✅✅ DeepSeek V2/V3 |

---

### Serving / 工程部署

| 维度 | Full Softmax | 稀疏(NSA/MoBA) | 线性/Recurrent | MLA | 混合 |
|------|:----------:|:-------------:|:-------------:|:---:|:---:|
| 框架兼容性 | ✅ 所有框架 | ⚠️ 需定制 kernel | ⚠️ 需定制 kernel | ✅ vLLM已支持 | ⚠️ 最复杂 |
| PagedAttention | ✅ | ⚠️ 需适配 | ❌ 不适用 | ✅ 已适配 | 部分层适用 |
| Prefix Cache | ✅ | ✅ | ⚠️ state 不好分割 | ✅ | ⚠️ |
| Speculative Decoding | ✅ | ✅ | ✅ O(1)验证更快 | ✅ | ✅ |
| Tensor Parallelism | ✅ | ✅ | ✅ | ✅ | ✅ |
| Continuous Batching | ✅ | ✅ | ⚠️ 每请求独立state | ✅ | ⚠️ |
| 量化友好度 | ✅ SageAttn | ✅ | ⚠️ state精度敏感 | ✅ | 混合策略 |

---

### 综合评分（5分制）

| 维度 | Full Softmax | NSA | MoBA | Diff Trans | GLA | DeltaNet | Mamba-2 | RWKV-6 | MLA | 混合 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 模型质量 | 5 | 5 | 5 | **5+** | 4 | 4.5 | 4 | 4 | 5 | **5+** |
| Prefill 效率 | 2 | 4 | 4 | 2 | **5** | **5** | **5** | **5** | 3 | 4.5 |
| Decode 效率 | 3 | 4 | 4 | 3 | **5** | **5** | **5** | **5** | 4 | 4.5 |
| 显存效率 | 2 | 3.5 | 2.5 | 1.5 | **5** | **5** | **5** | **5** | 4.5 | 4 |
| 长上下文 | 2 | 4.5 | 4.5 | 3 | 4.5 | 4.5 | 4.5 | **5** | 4 | 4.5 |
| 精确检索 | **5** | **5** | **5** | **5** | 3 | 4 | 3 | 3 | **5** | **5** |
| 工程成熟度 | **5** | 3 | 4 | 3 | 3 | 2.5 | 3.5 | 3 | 4.5 | 2 |
| 训练效率 | 4 | 4.5 | 4.5 | 3.5 | **5** | 4 | 4.5 | 4 | 4 | 4 |

---

## 三、选型建议（从 Infra/Serving 视角）

| 场景 | 首选方案 | 理由 |
|------|---------|------|
| 短上下文高吞吐 serving | Full Softmax + MLA + SageAttention | 工程最成熟，MLA 砍 KV cache |
| 长上下文(>64K) serving | NSA 或 MoBA | 生产验证，算力 & KV cache 都省 |
| 超长上下文(>1M) | GLA/RWKV-6 (纯 recurrent) | O(1) 内存是唯一解 |
| 质量最优不计成本 | Diff Transformer 或混合架构 | 质量上限最高 |
| 下一代模型预训练 | 混合架构(DeltaNet/GLA + 少量 Full Attn) | 当前最佳 quality/efficiency tradeoff |
| 端侧/低功耗部署 | RWKV-6 / Mamba-2 | 固定内存，计算量可控 |

---

## 四、全景图

```
             ┌─ NSA (DeepSeek)
  稀疏 ──────┼─ MoBA (Kimi)        ← 生产验证
             └─ Diff Transformer

             ┌─ GLA + FlashLinearAttention
  线性 ──────┼─ DeltaNet (Delta Rule)   ← associative recall 最强
             └─ HGRN2 (外积状态扩展)

             ┌─ Mamba-2 / SSD
  Recurrent ─┼─ RWKV-6 (Finch)
             └─ (与线性 attention 理论统一)

             ┌─ DeltaNet + SW/Global Attn
  混合 ──────┼─ MLA + MoE (DeepSeek V2/V3)
             └─ Jamba (Mamba + Attention + MoE)

  计算优化 ──┬─ SageAttention (INT8)
             ├─ FlashInfer (JIT + block-sparse)
             └─ nGPT (超球面归一化)
```

**一句话总结：** Full Softmax 是安全牌，MLA 是最低风险的优化，稀疏 attention 是长上下文的实用解，线性/Recurrent 是终极效率方案但精确检索差，混合架构是未来。

---

## 参考论文

| 方案 | 论文 | 会议/状态 |
|------|------|----------|
| NSA | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | DeepSeek 2025.02 |
| MoBA | [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | Moonshot/Kimi 2025.02 |
| Diff Transformer | [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) | ICLR 2025 Oral |
| GLA | [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) | MIT/CMU 2023.12 |
| DeltaNet | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | NeurIPS 2024 |
| DiG | [arXiv:2405.18428](https://arxiv.org/abs/2405.18428) | 2024.05 |
| Mamba-2 | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) | ICML 2024 |
| RWKV-5/6 | [arXiv:2404.05892](https://arxiv.org/abs/2404.05892) | 2024.04 |
| HGRN2 | [arXiv:2404.07904](https://arxiv.org/abs/2404.07904) | COLM 2024 |
| MLA (DeepSeek-V2) | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | DeepSeek 2024.05 |
| SageAttention | [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) | ICLR 2025 |
| nGPT | [arXiv:2410.01131](https://arxiv.org/abs/2410.01131) | NVIDIA 2024.10 |
| FlashInfer | [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) | MLSys 2025 |
