#!/usr/bin/env bash
# Qwen3.6-35B-A3B (qwen3_5_moe) — 128k 单序列 GRPO 训练(Megatron + SGLang,单机 8 卡)
#
# 依赖隐状态分块投影(见 docs/megatron_long_context_hidden_chunked_projection_zh.md),
# 通过环境变量开启,使超长序列的 logits 不再整段 materialize。
#
# 容器:verlai/verl:sgl059.latest,容器内需:
#   pip install -U "transformers==5.3.0" "flash-linear-attention==0.4.1"
#   pip install -U "git+https://github.com/ISEEKYAN/mbridge.git"
#   pip install --no-deps -e <本仓库>
#
# 用法(在容器内):
#   STAGE=E bash examples/grpo_trainer/run_qwen3_6_35b_a3b_128k.sh
# STAGE: A=2k(冒烟) C=32k(正确性) E=128k(目标)
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
# DeepSeek 评委(自定义奖励函数从该环境变量读 key)
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?"请设置 DEEPSEEK_API_KEY"}
export PROOFBENCH_GRADER_CONCURRENCY=${PROOFBENCH_GRADER_CONCURRENCY:-32}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODEL_PATH=${MODEL_PATH:-/nvme3/cxz/models/Qwen3.6-35B-A3B}
DATA_DIR=${DATA_DIR:-/nvme3/cxz/unimath_data/rl_data/data/verl}
REWARD_FILE=${REWARD_FILE:-/nvme3/cxz/unimath_data/rl_data/scripts/deepseek_judge_reward.py}

# ================= 隐状态分块投影开关(长上下文必备) =================
HIDDEN_CHUNK=${HIDDEN_CHUNK:-1}          # 1 = 开启隐状态分块投影
LOGITS_CHUNK=${LOGITS_CHUNK:-8192}       # 分块大小(token)
LOGITS_CKPT=${LOGITS_CKPT:-1}            # update 阶段逐块梯度检查点

# ================= 分阶段长度 =================
STAGE=${STAGE:-E}
case "${STAGE}" in
  A) MAX_PROMPT=2048; MAX_RESP=2048;   TRAIN_BSZ=8; N=8; MINI=8; GMU=0.55 ;;
  C) MAX_PROMPT=4096; MAX_RESP=32768;  TRAIN_BSZ=8; N=8; MINI=8; GMU=0.60 ;;
  E) MAX_PROMPT=4096; MAX_RESP=131072; TRAIN_BSZ=4; N=4; MINI=4; GMU=0.60 ;;
  *) echo "未知 STAGE=${STAGE}"; exit 1 ;;
esac
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESP + 1024))
MAX_TOKEN_LEN=$((MAX_PROMPT + MAX_RESP))

# ================= 并行(隐状态投影要求 TP=1;EP 给专家并行) =================
TP=${TP:-1}; PP=${PP:-1}; CP=${CP:-1}; EP=${EP:-8}; ETP=${ETP:-1}; GEN_TP=${GEN_TP:-8}

PROJECT=${PROJECT:-qwen36_35b_a3b_proofgrade_grpo}
EXP=${EXP:-stage${STAGE}_128k}
CKPT_DIR=${CKPT_DIR:-/nvme3/cxz/ckpts/${PROJECT}/${EXP}}
mkdir -p "${CKPT_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/val.parquet \
  data.prompt_key=prompt \
  data.train_batch_size=${TRAIN_BSZ} \
  data.max_prompt_length=${MAX_PROMPT} \
  data.max_response_length=${MAX_RESP} \
  data.truncation=left \
  data.filter_overlong_prompts=True \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.thinking=True \
  reward.custom_reward_function.path=${REWARD_FILE} \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  reward.num_workers=8 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
  actor_rollout_ref.actor.megatron.use_remove_padding=False \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
  actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
  actor_rollout_ref.actor.megatron.param_offload=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GMU} \
  actor_rollout_ref.rollout.n=${N} \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT} \
  actor_rollout_ref.rollout.response_length=${MAX_RESP} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP} \
  actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
  actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.nccl_timeout=9600 \
  +ray_kwargs.ray_init.runtime_env.env_vars.DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PROOFBENCH_GRADER_CONCURRENCY=\"${PROOFBENCH_GRADER_CONCURRENCY}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=\"1\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE=\"1\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_HIDDEN_CHUNK=\"${HIDDEN_CHUNK}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_LOGITS_CHUNK=\"${LOGITS_CHUNK}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_LOGITS_CKPT=\"${LOGITS_CKPT}\" \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXP} \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-20} \
  trainer.test_freq=${TEST_FREQ:-1000} \
  trainer.val_before_train=False \
  trainer.total_epochs=${EPOCHS:-1} \
  trainer.default_local_dir=${CKPT_DIR} \
  model_engine=megatron \
  "$@"
