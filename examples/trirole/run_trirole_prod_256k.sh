#!/usr/bin/env bash
# PRODUCTION tri-role 256k run (launch only after run_trirole_256k_gate.sh PASSES).
#
# Window: 12288 + 248832 + 1024 = 262144 (model native max).
# Data: trirole_v4 (804 problems). bs16 x N8 -> 50 steps/epoch.
#
# Wall-clock math (why bs16): SGLang KV at 262k is only ~5.4GB/seq (GDN: 10 full-attn
# layers), so 128+ rollouts run concurrently and gen wall-clock is set by the LONGEST
# sequence (~2h at 249k), nearly independent of bs. Bigger bs amortizes that pole:
#   bs8  -> 100 steps/ep x ~3.5h  ; bs16 -> 50 steps/ep x ~4h (default)
# Validation (60 x 249k, ~2h) every TEST_FREQ steps.
#
# Run inside container:
#   bash /nvme3/cxz/verl/examples/trirole/run_trirole_prod_256k.sh
set -euo pipefail

export GEMINI_API_KEY=${GEMINI_API_KEY:?set GEMINI_API_KEY}
export STAGE=C   # placeholder; lengths overridden below
export MAX_PROMPT_OVERRIDE=12288
export MAX_RESP_OVERRIDE=${MAX_RESP_OVERRIDE:-204800}
export TRAIN_BSZ_OVERRIDE=${TRAIN_BSZ_OVERRIDE:-12}
export N_OVERRIDE=${N_OVERRIDE:-8}
export GMU_OVERRIDE=${GMU_OVERRIDE:-0.55}
export CALC_ENTROPY=False
export VERL_MCORE_HIDDEN_CHUNK=1
export VERL_MCORE_LOGITS_CHUNK=8192
export VERL_MCORE_LOGITS_CKPT=1
# gate G1 verdict: update peak 92GB WITH this (vs 130GB OOM without) — required at 256k
export VERL_MCORE_SAVE_ON_CPU=1
export TRIROLE_GRADE_PER_PROB=${TRIROLE_GRADE_PER_PROB:-2}
export TRIROLE_K_GRADE=${TRIROLE_K_GRADE:-4}
export TRIROLE_M_REFINE=${TRIROLE_M_REFINE:-8}
export TRIROLE_GRADE_MAX_NEW=${TRIROLE_GRADE_MAX_NEW:-98304}
export TRIROLE_DROP_ZERO_VAR=1
export EPOCHS=${EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-25}
export EXP=${EXP:-trirole_256k_v4_b12}
export CKPT_DIR=/nvme4/cxz/ckpts/qwen36_35b_a3b_trirole/${EXP}
export DATA_DIR=${DATA_DIR:-/nvme3/cxz/unimath_data/rl_data/data/trirole_v4}

# GSPO (mirrors the proven launch_gspo_10epoch.sh policy-loss overrides)
bash "$(dirname "$0")/run_trirole.sh" \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  trainer.test_freq=${TEST_FREQ} \
  trainer.max_actor_ckpt_to_keep=3 \
  "$@"
