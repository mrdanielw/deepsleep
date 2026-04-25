# DeepSeek-V4 / GLM-5.1 / Kimi-K2.6 KV Cache 对比分析

> 2026-04-25 | 基于各模型 HuggingFace config.json + DeepSeek-V4 Paper

---

## 一、模型架构参数总览

| | **V4-Flash** | **V4-Pro** | **GLM-5.1** | **Kimi K2.6** |
|---|---|---|---|---|
| 架构类型 | CSA + HCA（压缩注意力） | CSA + HCA | MLA + DSA | MLA (V3 系) |
| 总参数 | 284B | 1.6T | 754B | ~1T (多模态) |
| 激活参数 | 13B | 49B | ~14B | ~32B |
| 总层数 | 43 | 61 | 78 | 61 |
| Hidden size | 4096 | 7168 | 6144 | 7168 |
| Query heads | 64 | 128 | 64 | 64 |
| KV heads | 1 (shared MQA) | 1 (shared MQA) | 64 (MHA) | 64 |
| Head dim (主 attention) | 512 | 512 | — | — |
| KV LoRA rank | — | — | 512 | 512 |
| QK RoPE dim | 64 | 64 | 64 | 64 |
| V head dim | 512 (shared KV) | 512 (shared KV) | 256 | 128 |
| Max context | **1M** | **1M** | 200K | 262K |

### 注意力机制差异

| | **V4 (CSA/HCA)** | **GLM-5.1 (MLA+DSA)** | **Kimi K2.6 (MLA)** |
|---|---|---|---|
| KV 存储方式 | 压缩后的 shared KV entry (512d) | MLA latent (576d) | MLA latent (576d) |
| 序列压缩 | CSA: 4x, HCA: 128x | 无 | 无 |
| 稀疏选择 | CSA: Lightning Indexer top-k | DSA: Lightning Indexer top-k | 无 |
| Indexer | 有（CSA 层） | 有（全部层） | 无 |

---

## 二、KV Cache 存储构成

### 2.1 Kimi K2.6 — 标准 MLA（最简单）

和 DeepSeek-V3 同架构，无压缩，无 Indexer。

**每 token 每层存储** = kv_lora_rank + qk_rope_head_dim = 512 + 64 = **576 维**

```
BF16: 576 × 2 bytes = 1,152 bytes / token / layer
```

### 2.2 GLM-5.1 — MLA + DSA Indexer

**MLA 部分**（同 Kimi K2.6）：
- kv_lora_rank + qk_rope_head_dim = 512 + 64 = **576 维**

**Indexer 部分**（Lightning Indexer for DSA）：
- 只存 compressed indexer key，single-head
- index_head_dim = 128 维
- 但 GLM-5.1 **无序列压缩**（config 中无 compress_ratios），每 token 一个 indexer key

```
MLA (BF16):     576 × 2 = 1,152 bytes / token / layer
Indexer (FP8):  128 × 1 = 128 bytes / token / layer
合计:           1,280 bytes / token / layer
```

### 2.3 DeepSeek-V4 — CSA + HCA 压缩注意力

V4 **不是 MLA**，是全新的压缩 shared-KV 架构。KV cache 由三部分组成：

**a) 压缩主 KV**（CSA 和 HCA 共有）
- 1 个 KV head × 512 dim（shared key-value）
- 混合精度：RoPE 64 dim 用 BF16，其余 448 dim 用 FP8
- CSA 层压缩 4x，HCA 层压缩 128x

**b) Indexer Key**（仅 CSA 层）
- Single-head，128 dim，FP4 精度
- 同样被 4x 压缩
- **只有 Key，没有 Value**（Indexer 只算 score 做 top-k 选择）

**c) SWA（Sliding Window Attention）KV**（所有层）
- 最近 128 个 token 的未压缩 KV
- 1 head × 512 dim，混合精度

#### V4-Flash 层分布（43 层）

从 `compress_ratios` 数组解析：
```
Layer 0-1:   SWA-only (compress_ratio=0)     — 2 层
Layer 2-41:  CSA(m=4) 和 HCA(m'=128) 交替    — 20 层 CSA + 20 层 HCA
Layer 42:    SWA-only                         — 1 层
```

#### V4-Pro 层分布（61 层）

```
Layer 0-1:   HCA (m'=128)                    — 2 层
Layer 2-59:  CSA(m=4) 和 HCA(m'=128) 交替    — 29 层 CSA + 29 层 HCA  
Layer 60:    SWA-only                         — 1 层
```

---

## 三、KV Cache 大小计算（1K context = 1024 tokens）

### 3.1 Kimi K2.6

```
1,152 bytes × 61 层 × 1024 tokens = 72,024 KB ≈ 70.3 MB
```

### 3.2 GLM-5.1

```
1,280 bytes × 78 层 × 1024 tokens = 102,236 KB ≈ 99.8 MB
```

### 3.3 V4-Flash

**CSA 层（20 层）每层：**

| 组件 | 计算 | 大小 |
|---|---|---|
| 压缩主 KV | 1024/4 = 256 entries × (448×1B + 64×2B) = 256 × 576B | 144 KB |
| Indexer Key | 256 entries × 128 dim × 0.5B (FP4) | 16 KB |
| SWA KV | 128 tokens × (448×1B + 64×2B) = 128 × 576B | 72 KB |
| **层合计** | | **232 KB** |

**HCA 层（20 层）每层：**

| 组件 | 计算 | 大小 |
|---|---|---|
| 压缩主 KV | 1024/128 = 8 entries × 576B | 4.5 KB |
| SWA KV | 128 tokens × 576B | 72 KB |
| **层合计** | | **76.5 KB** |

**SWA-only 层（3 层）每层：**

| 组件 | 计算 | 大小 |
|---|---|---|
| SWA KV | 128 tokens × 576B | 72 KB |

**V4-Flash 总计：**
```
CSA:      20 × 232 KB   = 4,640 KB
HCA:      20 × 76.5 KB  = 1,530 KB
SWA-only: 3  × 72 KB    = 216 KB
总计:                    = 6,386 KB ≈ 6.2 MB
```

### 3.4 V4-Pro

**CSA 层（29 层）**：每层 232 KB → 6,728 KB
**HCA 层（31 层，含前 2 层）**：每层 76.5 KB → 2,372 KB
**SWA-only 层（1 层）**：72 KB

```
总计: 6,728 + 2,372 + 72 = 9,172 KB ≈ 9.0 MB
```

### 1K Context 汇总

| 模型 | KV Cache (1K ctx) | 倍数 (vs V4-Flash) |
|---|---|---|
| **V4-Flash** | **~6.2 MB** | 1x |
| **V4-Pro** | ~9.0 MB | 1.5x |
| **Kimi K2.6** | 70.3 MB | 11.3x |
| **GLM-5.1** | 99.8 MB | 16.1x |

---

## 四、不同上下文长度的 KV Cache 增长趋势

### 增长公式

**Kimi K2.6**（线性增长，无压缩）：
```
KV = 1,152 × 61 × seq_len bytes
   = 68.6 KB / token
```

**GLM-5.1**（线性增长 + Indexer）：
```
KV = 1,280 × 78 × seq_len bytes
   = 97.5 KB / token
```

**V4-Flash**（压缩 + SWA 封顶）：
```
CSA 层: 主KV 增长 1/4 速率 + Indexer 增长 1/4 速率 + SWA 封顶 128 tokens
HCA 层: 主KV 增长 1/128 速率 + SWA 封顶 128 tokens
SWA-only: 封顶 128 tokens

长序列下有效增长率 ≈ 20层×(576/4 + 64/4) + 20层×(576/128) + 0 (SWA已封顶)
                    = 20×160 + 20×4.5
                    = 3,290 bytes/token ≈ 3.2 KB/token
```

**V4-Pro**（同理）：
```
有效增长率 ≈ 29×160 + 31×4.5 = 4,780 bytes/token ≈ 4.7 KB/token
```

### 各上下文长度对比

| 模型 | 1K | 32K | 128K | 512K | **1M** |
|---|---|---|---|---|---|
| **V4-Flash** | 6 MB | 106 MB | 410 MB | 1.6 GB | **3.1 GB** |
| **V4-Pro** | 9 MB | 154 MB | 600 MB | 2.4 GB | **4.6 GB** |
| **Kimi K2.6** | 70 MB | 2.2 GB | 8.6 GB | 34.3 GB | **68.6 GB** |
| **GLM-5.1** | 100 MB | 3.1 GB | 12.2 GB | — | **不支持** (max 200K) |

> **注 1**：V4 的 SWA 部分在 seq_len > 128 后不再增长（每层固定 128 tokens × 576B），因此长序列下增长速率远低于 MLA。
>
> **注 2**：V4 从 1K→32K 的增长倍数（约 17x）看似不线性，原因是 SWA 固定开销在短序列中占比高达 ~49%。去掉 SWA 固定部分后，可变部分的增长严格线性（32x）。随序列变长，SWA 占比趋近 0，整体增长率回归线性。

### 增长率对比（长序列稳态）

| 模型 | 每 token KV 增长 | 相对 V4-Flash |
|---|---|---|
| **V4-Flash** | ~3.2 KB/token | 1x |
| **V4-Pro** | ~4.7 KB/token | 1.5x |
| **Kimi K2.6** | 68.6 KB/token | **21x** |
| **GLM-5.1** | 97.5 KB/token | **30x** |

---

## 五、关键设计差异总结

### 5.1 为什么 V4 的 Indexer 这么轻？

Lightning Indexer 的设计原则是 **"用最少的存储做最快的选择"**：

1. **只存 Key，不存 Value** — Indexer 只负责计算 score 做 top-k，不参与最终 attention
2. **Single-head shared** — 虽然 Indexer 有 64 个 query heads，但 Key 端是单头共享的（128 dim）
3. **同样被压缩** — CSA 的 4x 压缩同时作用于 Indexer Key
4. **FP4 精度** — Paper 明确说 Indexer 计算用 FP4

每 CSA 层 Indexer 的 KV cache 仅 **16 KB / 1K tokens (FP4)**，几乎可以忽略。

### 5.2 V4 vs MLA 的本质区别

| | V4 (CSA/HCA) | MLA (K2.6 / GLM-5.1) |
|---|---|---|
| **KV 存什么** | 压缩后的 shared KV entry | 未压缩的 latent vector |
| **序列维度** | 4x~128x 压缩 | 不压缩 |
| **K 和 V 关系** | 同一个向量既当 K 又当 V | latent 推理时解压为 K 和 V |
| **位置编码** | 部分 RoPE (64 dim) | 部分 RoPE (64 dim) |
| **长序列 KV 增长** | 亚线性（压缩+SWA封顶） | **线性** |

### 5.3 Serving 影响

| 场景 | V4 优势 | V4 代价 |
|---|---|---|
| **长上下文 (128K+)** | KV cache 省 10-20x，同显存可服务更多请求 | CSA/HCA 压缩逻辑增加 prefill 计算 |
| **高并发短文本** | 省显存 = 更多并发 slots | 压缩+Indexer 的额外计算在短文本时收益不大 |
| **Prefix caching** | On-disk KV cache 设计原生支持 | 异构 KV 布局使 PagedAttention 不再适用 |
| **推理框架适配** | — | 需要全新的 KV cache 管理（SGLang 已实现） |

---

## 六、附录：config.json 关键字段

### V4-Flash
```json
{
  "num_hidden_layers": 43,
  "hidden_size": 4096,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,
  "head_dim": 512,
  "q_lora_rank": 1024,
  "qk_rope_head_dim": 64,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "sliding_window": 128,
  "n_routed_experts": 256,
  "num_experts_per_tok": 6,
  "compress_ratios": [0,0, 4,128,4,128,..., 4,0]
}
```

### V4-Pro
```json
{
  "num_hidden_layers": 61,
  "hidden_size": 7168,
  "num_attention_heads": 128,
  "num_key_value_heads": 1,
  "head_dim": 512,
  "q_lora_rank": 1536,
  "qk_rope_head_dim": 64,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 1024,
  "sliding_window": 128,
  "n_routed_experts": 384,
  "num_experts_per_tok": 6,
  "compress_ratios": [128,128, 4,128,4,128,..., 4,0]
}
```

### GLM-5.1
```json
{
  "num_hidden_layers": 78,
  "hidden_size": 6144,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "kv_lora_rank": 512,
  "qk_rope_head_dim": 64,
  "v_head_dim": 256,
  "index_head_dim": 128,
  "index_n_heads": 32,
  "index_topk": 2048,
  "n_routed_experts": 256,
  "num_experts_per_tok": 8
}
```

### Kimi K2.6
```json
{
  "num_hidden_layers": 61,
  "hidden_size": 7168,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "kv_lora_rank": 512,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "n_routed_experts": 384,
  "num_experts_per_tok": 8
}
```
