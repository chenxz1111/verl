#!/usr/bin/env bash
# 256k single-step gate test (G3: current config straight at 262144 window).
# PASS = one full step completes (all three waves + update) without OOM.
# If update-side OOM -> G1: save_on_cpu patch in transformer_impl.py, retest.
#
# Window: 12288 prompt + 248832 response + 1024 slack = 262144 (model native max).
# Grade wave capped at 128k via TRIROLE_GRADE_MAX_NEW (per-sample sampling override).
# bs4 x N4, K2, M4 keeps the judge bill and wall-clock of the gate small; memory
# risk is per-row (single 261k sequence), not batch-size-driven.
#
# Run inside container on a free 8-GPU node:
#   bash /nvme3/cxz/verl/examples/trirole/run_trirole_256k_gate.sh
set -euo pipefail

export GEMINI_API_KEY=${GEMINI_API_KEY:?set GEMINI_API_KEY}
export STAGE=C   # placeholder; every length is overridden below
export MAX_PROMPT_OVERRIDE=12288
export MAX_RESP_OVERRIDE=248832
export TRAIN_BSZ_OVERRIDE=4
export N_OVERRIDE=4
export GMU_OVERRIDE=${GMU_OVERRIDE:-0.45}
export CALC_ENTROPY=False
export VERL_MCORE_HIDDEN_CHUNK=1
export VERL_MCORE_LOGITS_CHUNK=8192
export VERL_MCORE_LOGITS_CKPT=1
export TRIROLE_GRADE_PER_PROB=2
export TRIROLE_K_GRADE=2
export TRIROLE_M_REFINE=4
export TRIROLE_GRADE_MAX_NEW=131072
export TRIROLE_DROP_ZERO_VAR=0    # gate wants maximum rows through the update, not fewer
export EPOCHS=1
export SAVE_FREQ=0
export EXP=${EXP:-trirole_256k_gate}
export CKPT_DIR=/nvme4/cxz/ckpts/qwen36_35b_a3b_trirole/${EXP}
export DATA_DIR=${DATA_DIR:-/nvme3/cxz/unimath_data/rl_data/data/trirole_v4}

bash "$(dirname "$0")/run_trirole.sh" \
  trainer.total_training_steps=1 \
  "$@"
