# ProofBench GRPO — Qwen3.6-35B-A3B 长上下文(128k)证明评分 RL

用 **DeepSeek LLM 评委**对数学证明打分,对 **Qwen3.6-35B-A3B(`qwen3_5_moe`,混合 GDN 线性注意力 + 256 专家 MoE)** 做 **GRPO**,支持**严格 128k 单序列生成**训练。单机 8×H200 实测跑通。

这个 recipe 是**自包含**的:核心框架改动已在 verl 源码里(见下),这里放数据转换、奖励函数、评分模板、启动脚本。换机器 `git clone` + 建环境 + 改几个路径即可复用。

## 目录

```
examples/proofbench/
├── run_proofbench_grpo.sh          # 启动脚本(STAGE=A..E 控制长度;路径全用环境变量)
├── scripts/
│   ├── prepare_verl_data.py        # slime-jsonl → verl parquet,judge 统一为 deepseek-v4-flash
│   └── deepseek_judge_reward.py    # 异步 LLM 评委奖励(verl reward_loop 接口)
└── templates/
    └── evaluation_06-05version.yaml # 评分模板(with_solution:题目+参考解+学生解 → <score>0-7</score>)
```

## 依赖的框架改动(已在 verl 源码内)

长上下文 128k 训练靠两处源码改动(隐状态分块投影),已在仓库里:
- `verl/models/mcore/model_forward.py`
- `verl/workers/engine/megatron/transformer_impl.py`

原理见 `docs/megatron_long_context_hidden_chunked_projection_zh.md`。由环境变量门控,默认关闭。

## 环境

```bash
# Docker 镜像
docker run -d --name verl --gpus all --ipc=host --shm-size=64g \
  -v /your/data:/data verlai/verl:sgl059.latest sleep infinity
# 容器内补依赖
pip install -U "transformers==5.3.0" "flash-linear-attention==0.4.1"
pip install -U "git+https://github.com/ISEEKYAN/mbridge.git"
pip install --no-deps -e /path/to/verl     # 本仓库(含上面两处改动)
```

## 用法

### 1. 准备数据(slime jsonl → verl parquet)

```bash
python examples/proofbench/scripts/prepare_verl_data.py \
  --train-src /path/train.jsonl --val-src /path/validation.jsonl \
  --out-dir   /path/verl_data
# 每行被转成 verl 格式,judge 统一覆盖为 deepseek-v4-flash;
# reward_spec(含 judge 配置)以 JSON 字符串存进 extra_info.reward_spec_json。
```

### 2. 启动训练(严格 128k)

```bash
export DEEPSEEK_API_KEY=sk-xxx          # 评委 key(必填)
MODEL_PATH=/path/Qwen3.6-35B-A3B \
DATA_DIR=/path/verl_data \
CKPT_DIR=/disk_with_room/ckpts/run1 \
STAGE=E TP=1 EP=8 GMU_OVERRIDE=0.55 \
TRAIN_BSZ_OVERRIDE=16 N_OVERRIDE=8 MINI_OVERRIDE=16 \
VERL_MCORE_HIDDEN_CHUNK=1 VERL_MCORE_LOGITS_CHUNK=8192 VERL_MCORE_LOGITS_CKPT=1 \
CALC_ENTROPY=False MOE_PERMUTE_FUSION=False \
bash examples/proofbench/run_proofbench_grpo.sh \
  trainer.val_before_train=True trainer.test_freq=5
```

## 关键配置(单机 8 卡满 128k 必备)

| 项 | 值 | 为什么 |
|----|-----|--------|
| `VERL_MCORE_HIDDEN_CHUNK=1` | 开 | 隐状态分块投影,避免 45GB fp32 logits |
| `VERL_MCORE_LOGITS_CHUNK=8192` | 8192 | 分块大小 |
| `VERL_MCORE_LOGITS_CKPT=1` | 开 | update 反向逐块梯度检查点 |
| `MOE_PERMUTE_FUSION=False` | 关 | **关键**:融合 MoE 置换 kernel 的 Triton autotune 在满 128k 会显存尖峰 OOM;关掉走非融合路径(慢一点但能跑) |
| `TP=1 EP=8` | — | MoE 在 TP>1 强制要 sequence_parallel,而 SP 与序列分块冲突;TP=1 规避,且避开 TP=4 的 attn_output_gate bug |
| `CALC_ENTROPY=False` | 关 | 熵只用于日志(entropy_coeff=0),关掉省显存 |
| `GMU_OVERRIDE=0.55` | 0.55 | SGLang 显存占用;长上下文 colocated 的平衡点 |

## 实测数据(8×H200,checkpoint-1492-hf SFT 基座,train_v2 402 题)

- **batch=16 n=8 @ 满 128k**:单步 ~89 分钟(生成 71 + update 13 + log_prob 5),峰值显存 109GB,reward 有方差(min=0/max=1)、pg_loss 非零,val 前评测基线 ~2.98/7。**这是稳定可用档位。**
- batch=64 @ 128k:能不 OOM 但单步 hang/超慢,不可用。
- 单步耗时 80% 在 rollout 生成(128 条逐 token 最长 128k),打分与生成重叠、几乎不占 step 时间。

## 评分(judge)

全程 **deepseek-v4-flash**(`reasoning_effort=max, temperature=0`):模型生成证明 → 剥 `<think>` → with_solution 模板(题/参考解/学生解)→ `<score>0-7</score>` → reward=分/7。失败/超时归 0,从不抛异常打断训练。
