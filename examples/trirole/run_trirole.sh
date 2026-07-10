#!/usr/bin/env bash
# Tri-role (solver/grader/refine) co-training GRPO/GSPO of Qwen3.6-35B-A3B.
# Derived from the proven /nvme3/cxz/run_proofgrade_grpo.sh (Megatron+mbridge train /
# SGLang async rollout, single node 8x GPU); entry point swapped to recipe.trirole.
#
# Stages (prompt budget is larger than solver-only: grader/refine prompts embed the
# public proof + critique, p99 ~4-5k tok + 3k tok system):
#   STAGE=A  12k prompt / 16k resp   (tri-role mechanics shakedown)
#   STAGE=B  12k / 64k
#   STAGE=C  12k / 128k              (target; 256k = STAGE=D after the cpu_offloading gate)
#
# Tri-role knobs (env): TRIROLE_GRADE_PER_PROB=2 TRIROLE_K_GRADE=4 TRIROLE_M_REFINE=8
#   TRIROLE_DROP_ZERO_VAR=1 TRIROLE_LAMBDA_{SOLVE,GRADE,REFINE}=1.0
#   TRIROLE_GRADE_MAX_NEW / TRIROLE_REFINE_MAX_NEW (0 = config response_length)
#
# Run inside the training container:
#   STAGE=A bash /nvme3/cxz/verl/examples/trirole/run_trirole.sh
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
export GEMINI_API_KEY=${GEMINI_API_KEY:-}
export PROOFBENCH_GRADER_CONCURRENCY=${PROOFBENCH_GRADER_CONCURRENCY:-32}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VERL_DIR=${VERL_DIR:-/nvme3/cxz/verl}
export PYTHONPATH=${VERL_DIR}/examples/trirole:${VERL_DIR}:${PYTHONPATH:-}

MODEL_PATH=${MODEL_PATH:-/nvme4/cxz/qwen36_sft_v3.1_128k_pack_8ep/hf_checkpoints/checkpoint-5285-hf}
DATA_DIR=${DATA_DIR:-/nvme3/cxz/unimath_data/rl_data/data/trirole_v3}
REWARD_FILE=${REWARD_FILE:-${VERL_DIR}/examples/trirole/trirole_reward.py}

STAGE=${STAGE:-A}
case "${STAGE}" in
  A) MAX_PROMPT=12288; MAX_RESP=16384;  TRAIN_BSZ=8; N=4; GMU=0.60 ;;
  B) MAX_PROMPT=12288; MAX_RESP=65536;  TRAIN_BSZ=8; N=8; GMU=0.50 ;;
  C) MAX_PROMPT=12288; MAX_RESP=131072; TRAIN_BSZ=8; N=8; GMU=0.45 ;;
  *) echo "Unknown STAGE=${STAGE}"; exit 1 ;;
esac
TRAIN_BSZ=${TRAIN_BSZ_OVERRIDE:-$TRAIN_BSZ}
N=${N_OVERRIDE:-$N}
GMU=${GMU_OVERRIDE:-$GMU}
MAX_RESP=${MAX_RESP_OVERRIDE:-$MAX_RESP}
MAX_PROMPT=${MAX_PROMPT_OVERRIDE:-$MAX_PROMPT}
CALC_ENTROPY=${CALC_ENTROPY:-True}
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESP + 1024))
MAX_TOKEN_LEN=$((MAX_PROMPT + MAX_RESP))
# ppo_mini_batch_size is recomputed every step by the trainer (single optimizer step);
# the value here only has to pass startup validation.
MINI=${MINI_OVERRIDE:-$TRAIN_BSZ}

TP=${TP:-1}; PP=${PP:-1}; CP=${CP:-1}; EP=${EP:-8}; ETP=${ETP:-1}; GEN_TP=${GEN_TP:-8}

PROJECT=${PROJECT:-qwen36_35b_a3b_trirole}
EXP=${EXP:-stage${STAGE}}
CKPT_DIR=${CKPT_DIR:-/nvme3/cxz/ckpts/${PROJECT}/${EXP}}
mkdir -p "${CKPT_DIR}"

cd "${VERL_DIR}"
python3 ${VERL_DIR}/examples/trirole/main_trirole.py \
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
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False \
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
  +ray_kwargs.ray_init.runtime_env.env_vars.DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.GEMINI_API_KEY="${GEMINI_API_KEY}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${VERL_DIR}/examples/trirole:${VERL_DIR}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PROOFBENCH_GRADER_CONCURRENCY=\"${PROOFBENCH_GRADER_CONCURRENCY}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=\"1\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE=\"1\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_LOGITS_CHUNK=\"${VERL_MCORE_LOGITS_CHUNK:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_LOGITS_CKPT=\"${VERL_MCORE_LOGITS_CKPT:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_HIDDEN_CHUNK=\"${VERL_MCORE_HIDDEN_CHUNK:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_SAVE_ON_CPU=\"${VERL_MCORE_SAVE_ON_CPU:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MCORE_SAVE_ON_CPU_MIN_NUMEL=\"${VERL_MCORE_SAVE_ON_CPU_MIN_NUMEL:-268435456}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_GRADE_PER_PROB=\"${TRIROLE_GRADE_PER_PROB:-2}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_K_GRADE=\"${TRIROLE_K_GRADE:-4}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_M_REFINE=\"${TRIROLE_M_REFINE:-8}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_DROP_ZERO_VAR=\"${TRIROLE_DROP_ZERO_VAR:-1}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_LAMBDA_SOLVE=\"${TRIROLE_LAMBDA_SOLVE:-1.0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_LAMBDA_GRADE=\"${TRIROLE_LAMBDA_GRADE:-1.0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_LAMBDA_REFINE=\"${TRIROLE_LAMBDA_REFINE:-1.0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_GRADE_MAX_NEW=\"${TRIROLE_GRADE_MAX_NEW:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRIROLE_REFINE_MAX_NEW=\"${TRIROLE_REFINE_MAX_NEW:-0}\" \
  +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=\"${WANDB_API_KEY:-}\" \
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
