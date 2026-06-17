# Megatron 长上下文训练:隐状态分块投影(Hidden-State Chunked Projection)

> 让 verl 的 Megatron 后端能够在单卡显存有限的情况下,对**超长序列(如 128k)**做 GRPO/PPO 训练。
> 实测:在 **8×H200(143GB)** 上跑通 **Qwen3.6-35B-A3B(`qwen3_5_moe`,混合 GDN 线性注意力 + 256 专家 MoE)** 的 **128k 单序列 GRPO 训练**。

---

## 1. 背景:为什么 128k 训练这么难

对超长序列做策略梯度训练,真正的显存瓶颈不是模型前向,而是 **logits 张量**:

- 输出层产生的 logits 形状是 `[batch, seqlen, vocab]`。本模型 `vocab=248320`。
- 在 128k 序列、TP=2 下,单个 logits(**fp32**)就是 `~98304 × 124160 × 4B ≈ 45GB`,反向还要再来一份梯度 ≈ 45GB,**光 logits I/O 就 ~90GB**。
- 交叉熵/熵计算在此基础上还会再产生 fp32 softmax 等中间峰值。

这个模型本身又非常新(`Qwen3_5MoeForConditionalGeneration`:GDN 线性注意力 + 输出门 `attn_output_gate` + 仅 2 个 KV 头 + 多模态 VL 封装),导致几条标准的长上下文并行手段全部失效:

| 手段 | 结果 |
|---|---|
| Megatron TP=4(切词表 logits) | ❌ `attention.py _apply_output_gate` 在 TP=4 下分片错位崩溃 |
| Megatron Context Parallel(切序列) | ❌ GDN 线性注意力不支持 |
| FSDP + Ulysses 序列并行 | ❌ GDN 沿序列递归,序列被切片后 `illegal memory access`(SP=2/8 均崩) |
| FSDP/Megatron 不切 | ❌ 128k 的 logits 装不下,OOM |

## 2. 解决方案:隐状态分块投影

核心思想:**永远不要一次性 materialize 完整的 `[B, S, vocab]` logits。**

1. **让模型只返回隐状态**(`[B, S, H]`,本模型 `H=2048`,128k 时仅 ~1GB),跳过输出层。
2. 在 `logits_processor` 里,**按序列分块**地用输出层权重做投影:每块 `hidden_chunk @ W_lm_head.T → logits_chunk`,随即算 log_prob / 熵,然后丢弃该块的 logits。
3. 反向用 `torch.utils.checkpoint` 逐块重算,使 update 阶段的交叉熵 softmax 也不必整段保存。

这本质上就是 **fused linear cross-entropy(Liger 式)** 的思路,但用纯 PyTorch 实现、且兼容本模型走的 **bshd(非 packing)** 路径——因为 GDN 在 Megatron 下不支持 THD/packing,无法直接用现成的 fused kernel(`use_fused_kernels=True` 要求 `use_remove_padding=True`)。

显存对比(128k,单序列):

| | 完整 logits | 隐状态分块投影 |
|---|---|---|
| 常驻大张量 | logits 45GB + 梯度 45GB | 隐状态 ~1GB |
| 投影/CE 峰值 | 整段 fp32 softmax(数十 GB) | 单块(`chunk × vocab`,~GB 级) |
| 实测峰值显存 | OOM | **97GB allocated / 126GB reserved** |

## 3. 代码改动

仅改动两个文件,**全部由环境变量门控,默认关闭,对既有行为零影响**:

### 3.1 `verl/models/mcore/model_forward.py` — `gptmodel_forward_model_engine`(bshd 分支)

当 `VERL_MCORE_HIDDEN_CHUNK=1` 时,在调用 `model(...)` 前把输出层临时替换成"直通"(返回隐状态),调用后立即恢复:

- 输出层位置自动解析:普通 GPTModel 用 `model.output_layer`,**VL 封装(如 `Qwen3_5VLModel`)用 `model.language_model.output_layer`**。
- 这样 `output_orig` 就是隐状态 `[B, S, H]`,交给下游 `logits_processor` 分块投影。

### 3.2 `verl/workers/engine/megatron/transformer_impl.py` — `logits_processor`

新增两条门控分支(原始路径保持不变):

- **隐状态分块投影分支**(`VERL_MCORE_HIDDEN_CHUNK=1`,推荐):检测到输入是隐状态时,逐块
  `copy_to_tensor_model_parallel_region(h)`(保证列并行输入语义,反向对 TP 做 all-reduce)→
  `F.linear(h.to(W.dtype), W)`(隐状态可能是 fp32,先 cast 到权重 bf16,与 `qwen3_5.forward_with_torch_backend` 一致)→
  `/temperature` → `vocab_parallel_log_probs / vocab_parallel_entropy`,update 阶段对每块包 `checkpoint`。
- **logits 分块分支**(`VERL_MCORE_LOGITS_CHUNK>0` 且未开启 hidden chunk):退一步的方案,只分块 logits 上的 CE/熵运算(能压住 fp32 softmax 峰值,但仍会 materialize 完整 logits;适合 64k 以内)。
- 另有一个调试开关 `VERL_DEBUG_LOGITS_SHAPE=1`,在 rank0 打印 logits 的 dim/shape/dtype,便于排查。

#### 几个踩过的关键正确性点(已在代码注释中说明)

- `vocab_parallel_entropy` 与 Megatron `vocab_parallel_cross_entropy` 反向都会**原地修改输入**;若直接传 logits 的切片(view),会破坏 `AsStridedBackward` 的版本检查 → 必须传 `.clone()`。
- 熵只在 `old_log_prob`(`no_grad`)阶段计算用于日志;update 阶段应 `CALC_ENTROPY=False`,避免带梯度的熵踩到原地修改。
- 投影必须用 `copy_to_tensor_model_parallel_region` 而非裸 `F.linear`,否则 TP>1 时 `grad_hidden` 缺少跨 TP 的 all-reduce → 梯度错误。

## 4. 启用方式与必备配置

### 环境变量

| 变量 | 作用 | 推荐值 |
|---|---|---|
| `VERL_MCORE_HIDDEN_CHUNK` | 开启隐状态分块投影 | `1` |
| `VERL_MCORE_LOGITS_CHUNK` | 分块大小(token 数) | `8192` |
| `VERL_MCORE_LOGITS_CKPT` | update 阶段逐块梯度检查点 | `1` |
| `VERL_DEBUG_LOGITS_SHAPE` | 调试打印 logits 形状 | `0` |

这些变量需要通过 `ray_kwargs.ray_init.runtime_env.env_vars.*` 传给 Ray worker(见启动脚本)。

### 必备的并行/配置

- **必须 `TP=1`**(配 `EP=8` 给专家并行)。原因:Megatron 的 MoE 层在 `TP>1` 时**强制要求 sequence_parallel**(`moe_layer.py` 硬 `raise`),而 SP 会把序列沿 TP 切碎,与按序列分块冲突。`TP=1` 时 SP 自动关闭,`copy_to_tensor_model_parallel_region`/`vocab_parallel_*` 退化为无操作,投影代码原样即对;还顺带避开了 `attn_output_gate` 在 TP=4 的 bug。
- `actor.calculate_entropy=False`(熵只用于日志,`entropy_coeff=0` 时不影响 loss)。
- `actor.megatron.use_remove_padding=False`、`actor.use_dynamic_bsz=False`(GDN 不支持 THD/packing)。
- rollout 用 **SGLang**(vLLM 当前不支持本架构 rollout),并带 GDN/mamba 调度参数:
  `engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer`、`disable_radix_cache=True`、`disable_overlap_schedule=True`、`enable_memory_saver=True`(colocated 必需)。

### colocated 显存调参(8 卡 143GB 上的 128k)

- **`rollout.gpu_memory_utilization=0.6`** 是甜点:太高(0.75)→ SGLang 恢复 KV 时 OOM;太低(0.4)→ 训练前向的 MoE all-reduce 在超长 batch 上 OOM。
- **不要用 `expandable_segments`**:与 SGLang 的 `torch_memory_saver` 互斥(后者是 colocated 显存让渡所必需)。
- `train_batch_size × n` 必须能被 GPU 数(8)整除。

## 5. 实测结果(Qwen3.6-35B-A3B,8×H200)

- **32k 正确性验证**:开启隐状态投影后 `critic/rewards/mean = 0.297`,与非分块基线 `0.30` 一致 → **投影数值等价**;`grad_norm`、反向、梯度检查点均正常。
- **128k 训练跑通(连续多步稳定)**:
  - `response_length/clip_ratio = 0.0`(**0% 截断**,证明在 128k 内完整生成;32k 时 66% 被截断)
  - 单序列(prompt+生成)最长 **167,872 token**,`response_length` 最长 84k
  - `critic/rewards/mean = 0.545`(比 32k 更高,因为证明能写完)、`max=1.0`、优势谱 ±1.5
  - **峰值显存 97GB / 143GB**,单步 ~17-22 分钟

## 6. 局限与后续

- 128k 在单机 8 卡 colocated 下是**贴着显存上限**跑的,单步偏慢、batch 里出现多条超长序列时较紧。若要更稳/更快,建议:rollout 与训练**分卡(disaggregated / fully_async)**或扩到多机。
- 分块投影目前覆盖 `model_engine` 的 bshd 路径(`micro_batch=1` 单序列、3D 非 nested 布局),与本模型的训练配置匹配。其他布局会自动回退到原始路径。
