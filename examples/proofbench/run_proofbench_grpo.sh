#!/usr/bin/env bash
# GRPO training of Qwen3.6-35B-A3B (model_type=qwen3_5_moe) on the proof-grading data,
# reward = DeepSeek deepseek-v4-flash LLM judge.
#
# Backend: Megatron + mbridge (train) / SGLang async (rollout). Single node, 8x H200.
# Based on examples/grpo_trainer/run_qwen3_5_35b_megatron.sh, switched vLLM -> SGLang
# (vLLM lacks qwen3.5 rollout) with the GDN/mamba scheduler engine_kwargs from
# verl/experimental/fully_async_policy/shell/grpo_qwen35_35b_megatron_async.sh.
#
# Staged length ramp via env: STAGE controls prompt/response/model_len.
#   STAGE=A  2k prompt / 2k resp  (plumbing smoke)
#   STAGE=B  2k / 8k
#   STAGE=C  4k / 32k
#   STAGE=D  4k / 64k
#   STAGE=E  4k / 128k  (the target)
#
# Run inside the container, e.g.:
#   STAGE=A bash examples/proofbench/run_proofbench_grpo.sh
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
# DeepSeek judge key: set DEEPSEEK_API_KEY in your environment before launching.
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?"set DEEPSEEK_API_KEY (judge API key) before launching"}
export PROOFBENCH_GRADER_CONCURRENCY=${PROOFBENCH_GRADER_CONCURRENCY:-32}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Directory of THIS script (examples/proofbench), so the reward fn resolves repo-relatively by default.
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths — override via env on a new machine. No machine-specific defaults are baked in
# except MODEL_PATH/DATA_DIR which you must point at your local model + prepared parquet.
MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to the local Qwen3.6-35B-A3B (qwen3_5_moe) checkpoint dir"}
DATA_DIR=${DATA_DIR:?"set DATA_DIR to the dir holding train.parquet/val.parquet (see scripts/prepare_verl_data.py)"}
REWARD_FILE=${REWARD_FILE:-${RECIPE_DIR}/scripts/deepseek_judge_reward.py}

STAGE=${STAGE:-A}
case "${STAGE}" in
  A) MAX_PROMPT=2048;  MAX_RESP=2048;   TRAIN_BSZ=8;  N=8;  MINI=8;  GMU=0.55 ;;
  B) MAX_PROMPT=2048;  MAX_RESP=8192;   TRAIN_BSZ=8;  N=8;  MINI=8;  GMU=0.60 ;;
  C) MAX_PROMPT=4096;  MAX_RESP=32768;  TRAIN_BSZ=8;  N=8;  MINI=8;  GMU=0.70 ;;
  D) MAX_PROMPT=4096;  MAX_RESP=65536;  TRAIN_BSZ=8;  N=8;  MINI=8;  GMU=0.75 ;;
  E) MAX_PROMPT=4096;  MAX_RESP=131072; TRAIN_BSZ=4;  N=4;  MINI=4;  GMU=0.75 ;;
  *) echo "Unknown STAGE=${STAGE}"; exit 1 ;;
esac
# Per-stage defaults are overridable via env (TRAIN_BSZ/N/MINI/GMU/MAX_PROMPT/MAX_RESP).
TRAIN_BSZ=${TRAIN_BSZ_OVERRIDE:-$TRAIN_BSZ}
N=${N_OVERRIDE:-$N}
MINI=${MINI_OVERRIDE:-$MINI}
GMU=${GMU_OVERRIDE:-$GMU}
MAX_RESP=${MAX_RESP_OVERRIDE:-$MAX_RESP}
MAX_PROMPT=${MAX_PROMPT_OVERRIDE:-$MAX_PROMPT}
# Entropy is logging-only when entropy_coeff=0; disable at long ctx to avoid the logits-sized clone.
CALC_ENTROPY=${CALC_ENTROPY:-True}
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESP + 1024))
MAX_TOKEN_LEN=$((MAX_PROMPT + MAX_RESP))

# Parallelism (tested config for this model, 8 GPUs / 1 node)
TP=${TP:-2}; PP=${PP:-1}; CP=${CP:-1}; EP=${EP:-8}; ETP=${ETP:-1}; GEN_TP=${GEN_TP:-8}

PROJECT=${PROJECT:-qwen36_35b_a3b_proofgrade_grpo}
EXP=${EXP:-stage${STAGE}}
CKPT_DIR=${CKPT_DIR:?"set CKPT_DIR to a checkpoint output dir on a disk with room (ckpts are ~470GB each)"}
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
  actor_rollout_ref.actor.calculate_entropy=${CALC_ENTROPY} \
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
  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=${MOE_PERMUTE_FUSION:-False} \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
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
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=${ENABLE_MEM_SAVER:-True} \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP} \
  actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
  actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.nccl_timeout=9600 \
  +ray_kwargs.ray_init.runtime_env.env_vars.DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY} \
  +ray_kwargs.ray_init.runtime_env.env_vars.PROOFBENCH_GRADER_CONCURRENCY=\"${PROOFBENCH_GRADER_CONCURRENCY}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=\"1\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE=\"1\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_LOGITS_CHUNK=\"${VERL_MCORE_LOGITS_CHUNK:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_LOGITS_CKPT=\"${VERL_MCORE_LOGITS_CKPT:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_HIDDEN_CHUNK=\"${VERL_MCORE_HIDDEN_CHUNK:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DEBUG_LOGITS_SHAPE=\"${VERL_DEBUG_LOGITS_SHAPE:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=\"${WANDB_API_KEY:-}\" \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb","tensorboard"]' \
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
