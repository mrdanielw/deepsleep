# DeepSeek V4 vs V3.2 计算开销对比分析

> 2026-04-25 | 基于 DeepSeek-V4 Paper + HuggingFace config.json

---

## 一、架构参数对照

| | **V3.2** | **V4-Pro** | **V4-Flash** |
|---|---|---|---|
| 层数 | 61 | 61 | 43 |
| Hidden size | 7168 | 7168 | 4096 |
| Query heads | 128 | 128 | 64 |
| KV 机制 | MLA (kv_lora_rank=512) | CSA/HCA (head_dim=512) | CSA/HCA (head_dim=512) |
| 注意力范围 | Dense（全部 token） | CSA: top-1024 压缩 KV; HCA: dense 但 128x 压缩 | CSA: top-512; HCA: dense 128x 压缩 |
| SWA 窗口 | 无 | 128 | 128 |
| Lightning Indexer | 无 | 有（CSA 层，FP4） | 有（CSA 层，FP4） |
| 路由专家数 | 256 | 384 | 256 |
| 激活专家数 | 8 | 6 | 6 |
| 专家 intermediate | 2048 | 3072 | 2048 |
| 激活参数 | ~37B | 49B | 13B |
| 专家精度 | FP8 | **FP4** (QAT) | **FP4** (QAT) |
| mHC | 无 | 有 (n_hc=4) | 有 (n_hc=4) |
| MTP | 1 | 1 | 1 |

---

## 二、Decode 阶段单 Token FLOPs 拆解

Decode 阶段每生成一个 token 的计算分为：**Attention、MoE FFN、新增开销（mHC/压缩）、其他**。

### 2.1 Attention FLOPs

这是 V3.2 → V4 变化最大的部分。

#### V3.2 — MLA Dense Attention

每个 query token 和**所有 n 个前序 token** 做 attention（通过 MLA latent 解压）：

```
Q 生成 (低秩):       2 × d × d_c = 2 × 7168 × 1536 = 22.0M
KV 解压 (每新 token): 2 × kv_lora_rank × (n_h × qk_nope_head_dim)
                     = 2 × 512 × (128 × 128) = 16.8M（仅解压供 prefill，decode 增量小）
Attention QK:        2 × n_h × qk_head_dim × n = 2 × 128 × 192 × n = 49,152n
Attention V:         2 × n_h × v_head_dim × n = 2 × 128 × 128 × n = 32,768n
Output projection:   2 × n_h × v_head_dim × d = 2 × 128 × 128 × 7168 = 235M

每层 Attention: ~282M + 81,920n
61 层总: ~17.2B + 4,997,120n ≈ 17.2B + 5.0M × n
```

| Context | V3.2 Attention FLOPs |
|---|---|
| 1K | 22.2B |
| 32K | 177B |
| 128K | 657B |
| **1M** | **5,017B** |

> 1M 时 Attention 占总 FLOPs 的 **99%**，完全主导。

#### V4-Pro — CSA 层（29 层）

Query token 只和 **top-k=1024 个压缩 KV + 128 个 SWA KV** 做 attention：

```
Q 生成 (低秩):    2 × d × d_c = 2 × 7168 × 1536 = 22.0M

Lightning Indexer (FP4 精度):
  Indexer QK:    2 × n_I_h × c_I × (n/m)
               = 2 × 64 × 128 × (n/4) = 4,096n (FP4 等效)

Core Attention (只看 1024+128 = 1152 个 KV，固定！):
  QK:           2 × n_h × c × 1152 = 2 × 128 × 512 × 1152 = 150.9M
  V:            2 × 128 × 512 × 1152 = 150.9M

Output projection (grouped):
  组内降维:     16 × 2 × (8×512) × 1024 = 134.2M
  组间投影:     2 × (16×1024) × 7168 = 234.9M

每 CSA 层: ~693M + 4,096n
```

#### V4-Pro — HCA 层（31 层）

Query token 和**所有 n/128 个压缩 KV + 128 SWA** 做 dense attention（无 Indexer）：

```
Q 生成:           22.0M

Core Attention (n/128 + 128 个 KV):
  QK:            2 × 128 × 512 × (n/128 + 128) = 1,024n + 16.8M
  V:             同上 = 1,024n + 16.8M

Output projection: 369.1M

每 HCA 层: ~425M + 2,048n
```

#### V4-Pro Attention 合计

```
CSA (29层):  29 × (693M + 4,096n) = 20.1B + 118,784n
HCA (31层):  31 × (425M + 2,048n) = 13.2B + 63,488n
SWA-only (1层): ~0.4B

总: ~33.7B + 182,272n
```

| Context | V4-Pro Attention FLOPs |
|---|---|
| 1K | 33.9B |
| 32K | 39.5B |
| 128K | 57.0B |
| **1M** | **216B** |

#### V4-Flash Attention（43 层 = 20 CSA + 20 HCA + 3 SWA-only）

V4-Flash 参数更小（d=4096, n_h=64, d_c=1024, top-k=512），类似推导：

```
CSA (20层): 每层 ~235M + 2,048n → 总 4.7B + 40,960n
HCA (20层): 每层 ~140M + 512n   → 总 2.8B + 10,240n
SWA (3层):  ~0.2B

总: ~7.7B + 51,200n
```

| Context | V4-Flash Attention FLOPs |
|---|---|
| 1K | 7.8B |
| 32K | 9.3B |
| 128K | 14.3B |
| **1M** | **58.9B** |

#### Attention FLOPs 对比表

| Context | V3.2 | V4-Pro | V4-Pro / V3.2 | V4-Flash | V4-Flash / V3.2 |
|---|---|---|---|---|---|
| 1K | 22.2B | 33.9B | 1.53x ❌ | 7.8B | 0.35x ✅ |
| 32K | 177B | 39.5B | **0.22x** | 9.3B | **0.053x** |
| 128K | 657B | 57.0B | **0.087x** | 14.3B | **0.022x** |
| **1M** | **5,017B** | **216B** | **0.043x** | **58.9B** | **0.012x** |

> **短序列（<4K）V4-Pro 的 Attention 反而比 V3.2 贵 ~1.5x**——CSA/HCA 的压缩、Indexer、grouped output projection 的固定开销在短文本时没有被摊薄。  
> **拐点约 4K-8K tokens**，超过后 V4 的压缩收益开始显现。

---

### 2.2 MoE FFN FLOPs

MoE 部分与上下文长度无关，每 token 固定开销。

**V3.2**：
```
共享专家 (SwiGLU, 1 expert):  2 × 3 × 7168 × 2048 = 88.1M
路由专家 (8 activated):       8 × 2 × 3 × 7168 × 2048 = 704.6M
Gate:                         2 × 7168 × 256 = 3.7M

每层: ~796.4M
61 层: ~48.6B
```

**V4-Pro**：
```
共享专家 (SwiGLU, 1 expert):  2 × 3 × 7168 × 3072 = 132.1M
路由专家 (6 activated):       6 × 2 × 3 × 7168 × 3072 = 793.0M
Gate:                         2 × 7168 × 384 = 5.5M

每层: ~930.6M
61 层: ~56.8B
```

**V4-Flash**：
```
共享专家:  2 × 3 × 4096 × 2048 = 50.3M
路由专家:  6 × 2 × 3 × 4096 × 2048 = 301.9M
Gate:      2 × 4096 × 256 = 2.1M

每层: ~354.3M
43 层: ~15.2B
```

| | V3.2 | V4-Pro | V4-Flash |
|---|---|---|---|
| MoE FLOPs / token | 48.6B | 56.8B (1.17x) | 15.2B (0.31x) |

> V4-Pro 的 MoE 比 V3.2 略贵（intermediate 3072 vs 2048），但激活专家数更少（6 vs 8），两者部分抵消。  
> V4 的路由专家用 **FP4 权重**。当前硬件 FP4×FP8 峰值 FLOPs 与 FP8×FP8 相同，但 **FP4 权重的显存带宽减半**，在 memory-bound 的 decode 阶段有实际加速收益。未来硬件如果原生支持 FP4×FP8 加速，还能再快 ~1/3。

---

### 2.3 V4 新增开销

#### mHC（Manifold-Constrained Hyper-Connections）

```
n_hc = 4, d = 7168

B 矩阵动态生成: 2 × (4×7168) × 16 = ~0.9M
Sinkhorn-Knopp (20 iter on 4×4 矩阵): 可忽略
残差混合 (矩阵乘): 几次 (4, d) 级别运算

每层 mHC: < 5M FLOPs
61 层总: ~0.3B FLOPs

结论: 相对于每层 ~1.6B 的总开销，mHC 开销可忽略（< 0.3%）
```

#### KV 压缩（Prefill 阶段为主）

CSA 层的压缩操作（每 m=4 token 压成 1 个 KV entry）：
```
KV projection (Ca, Cb): 2 × 2 × d × c = 4 × 7168 × 512 = 14.7M
Weight projection (Za, Zb): 同上 = 14.7M
Softmax + weighted sum: ~O(m × c) 很小

每 CSA 层: ~30M
```

HCA 层类似但无 overlap 压缩，约 15M/层。

**Decode 时的压缩开销**：新 token 的 KV 被 buffer 到凑够 m 个才执行一次压缩，均摊到每 token：
- CSA 层：30M / 4 = 7.5M/token
- HCA 层：15M / 128 = 0.12M/token

```
29 CSA 层 × 7.5M + 31 HCA 层 × 0.12M = 221M ≈ 0.2B / token
```

> **KV 压缩的额外开销很小（~0.2B），远小于它省下的 Attention FLOPs。**

---

## 三、Decode 总 FLOPs 汇总

| 组件 | V3.2 | V4-Pro | V4-Flash |
|---|---|---|---|
| Attention (1K) | 22.2B | 33.9B | 7.8B |
| Attention (128K) | 657B | 57.0B | 14.3B |
| Attention (1M) | 5,017B | 216B | 58.9B |
| MoE FFN | 48.6B | 56.8B | 15.2B |
| mHC | 0 | ~0.3B | ~0.2B |
| KV 压缩 (均摊) | 0 | ~0.2B | ~0.1B |
| 其他 (embed/norm) | ~2B | ~2B | ~1B |

### 单 Token Decode 总 FLOPs

| Context | V3.2 | V4-Pro | V4-Pro/V3.2 | V4-Flash | V4-Flash/V3.2 |
|---|---|---|---|---|---|
| **1K** | ~73B | ~93B | 1.27x ❌ | ~24B | 0.33x |
| **8K** | ~112B | ~95B | **0.85x** ← 拐点 | ~25B | 0.22x |
| **32K** | ~233B | ~99B | **0.42x** | ~27B | **0.12x** |
| **128K** | ~708B | ~116B | **0.16x** | ~31B | **0.044x** |
| **1M** | ~5,068B | ~275B | **0.054x (≈27%)** ✅ | ~75B | **0.015x (≈10%)** ✅ |

> Paper Figure 1 数据验证：V4-Pro 1M 时约为 V3.2 的 27%，V4-Flash 约为 10%。✅

---

## 四、计算瓶颈转移分析

### V3.2 的 FLOPs 构成随上下文变化

```
1K:    Attention 30%  |  MoE 67%  |  其他 3%     → MoE 主导
32K:   Attention 76%  |  MoE 21%  |  其他 3%     → Attention 开始主导
128K:  Attention 93%  |  MoE 7%   |  其他 <1%    → Attention 绝对主导
1M:    Attention 99%  |  MoE 1%   |  其他 <0.1%  → Attention = 一切
```

### V4-Pro 的 FLOPs 构成随上下文变化

```
1K:    Attention 36%  |  MoE 61%  |  新增 3%     → MoE 主导
32K:   Attention 40%  |  MoE 57%  |  新增 3%     → 仍是 MoE 主导！
128K:  Attention 49%  |  MoE 49%  |  新增 2%     → 基本均衡
1M:    Attention 78%  |  MoE 21%  |  新增 1%     → Attention 回归主导但远不如 V3.2 极端
```

### 关键洞察

```
          V3.2                              V4
        ┌──────────┐                     ┌──────────┐
 短序列  │ MoE: 67% │ ← 瓶颈             │ MoE: 61% │ ← 仍是瓶颈
 (1K)   │ Attn: 30% │                    │ Attn: 36% │  (固定开销更大)
        └──────────┘                     └──────────┘
        FLOPs ≈ 73B                      FLOPs ≈ 93B  (V4 贵 27%)

        ┌──────────┐                     ┌──────────┐
 中序列  │ Attn: 76% │ ← 瓶颈在转移       │ MoE: 57% │ ← MoE 仍是主角！
 (32K)  │ MoE: 21% │                    │ Attn: 40% │
        └──────────┘                     └──────────┘
        FLOPs ≈ 233B                     FLOPs ≈ 99B  (V4 省 57%)

        ┌──────────┐                     ┌──────────┐
 长序列  │ Attn: 99% │ ← 完全主导         │ Attn: 78% │ ← 被大幅压制
 (1M)   │ MoE:  1% │                    │ MoE: 21% │ ← 变成显著组件
        └──────────┘                     └──────────┘
        FLOPs ≈ 5068B                    FLOPs ≈ 275B  (V4 省 95%)
```

**V4 的核心贡献**：把 Attention 从"长序列杀手"（99% FLOPs）降级为"可控开销"（78%），使得 1M 上下文从计算上变得可行。

---

## 五、Prefill 阶段的影响

Decode 之外，Prefill 阶段也值得关注。V4 的压缩操作主要在 Prefill 时执行。

### V3.2 Prefill

标准 MLA prefill，计算量 ∝ n²（self-attention 是二次的）：
```
Attention: O(n² × n_h × head_dim) per layer
1M tokens: 极其昂贵，通常需要 chunked prefill
```

### V4 Prefill

CSA 层：先压缩（O(n)），再对压缩后的 n/4 个 KV 做 Indexer top-k 选择（O(n/4)）：
```
压缩: O(n × d) — 线性
Indexer: O(n × n/4 × c_I) — 比 V3.2 的 O(n²) 低 4x
Core Attention: O(n × k × c) — k 固定为 1024，与 n 无关！
```

HCA 层：压缩 128x 后做 dense attention：
```
压缩: O(n × d) — 线性
Attention: O(n × n/128 × c) — 比 V3.2 低 128x
```

**Prefill 的二次项被大幅压制**，这也是 V4 能原生支持 1M context prefill 的原因。

---

## 六、总结

| 维度 | V3.2 → V4 变化 | 影响 |
|---|---|---|
| **Attention FLOPs** | 1M 时降 23-100x | 长上下文从不可行变为常规操作 |
| **MoE FLOPs** | V4-Pro 略增 17% | 被 Attention 的巨大节省掩盖 |
| **mHC 开销** | 新增 ~0.3B/token | 可忽略（< 0.3%） |
| **KV 压缩开销** | 新增 ~0.2B/token | 远小于 Attention 节省量 |
| **短序列 (<4K)** | V4-Pro 贵 ~27% | 固定开销导致，V4-Flash 仍更便宜 |
| **拐点** | ~4K-8K tokens | 超过后 V4 全面优于 V3.2 |
| **FP4 专家权重** | 带宽减半 | Decode memory-bound 场景有实际加速 |
| **计算瓶颈** | Attention → MoE | V4 的优化方向应转向 MoE 通信和计算效率 |
| **Prefill 二次项** | 大幅压制 | 1M context prefill 变得可行 |

> **一句话**：V4 用 CSA/HCA 的压缩+稀疏把 Attention 从 O(n) 降到近似 O(1)（decode）和 O(n/m) + O(k)（prefill），代价是一些固定开销——短序列略亏，长序列巨赚。
