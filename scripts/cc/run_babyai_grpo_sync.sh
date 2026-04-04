#!/bin/bash
set -euo pipefail
set -x

# Synchronous BabyAI GRPO launcher for local Compute Canada runs.
# It mirrors the current sync FSDP example style, keeps all artifacts under
# ~/scratch/babyai by default, and is matched as closely as possible to the
# state_action_td launcher except for algorithm-specific settings.
#
# Examples:
#   bash scripts/cc/run_babyai_grpo_sync.sh
#   LOGGER=wandb MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 bash scripts/cc/run_babyai_grpo_sync.sh
#   ENV_NAME=BabyAI-GoToObj-v0 ENV_KWARGS_JSON='{"room_size": 8}' bash scripts/cc/run_babyai_grpo_sync.sh
#   VALIDATE_ONLY=true bash scripts/cc/run_babyai_grpo_sync.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_babyai_common.sh"

: "${ALGO_NAME:=grpo_sync}"
: "${MODEL_NAME:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${ENV_NAME:=BabyAI-GoToLocal-v0}"
: "${ENV_KWARGS_JSON:="{}"}"
: "${MAX_TURNS:=32}"
: "${EXPERIMENT_ROOT:="$HOME/scratch/babyai"}"

: "${NUM_GPUS:=4}"
: "${LOGGER:=console}"
: "${PROJECT_NAME:=babyai}"
: "${INFERENCE_BACKEND:=vllm}"
: "${SEED:=42}"

: "${EPOCHS:=40}"
: "${EVAL_BATCH_SIZE:=512}"
: "${EVAL_BEFORE_TRAIN:=true}"
: "${EVAL_INTERVAL:=5}"
: "${UPDATE_EPOCHS_PER_BATCH:=1}"
: "${TRAIN_BATCH_SIZE:=512}"
: "${POLICY_MINI_BATCH_SIZE:=128}"
: "${MICRO_FORWARD_BATCH_SIZE:=32}"
: "${MICRO_TRAIN_BATCH_SIZE:=32}"
: "${CKPT_INTERVAL:=10}"
: "${MAX_CKPTS_TO_KEEP:=5}"

: "${MAX_PROMPT_LENGTH:=512}"
: "${MAX_INPUT_LENGTH:=4096}"
: "${MAX_GENERATE_LENGTH:=256}"
: "${N_SAMPLES_PER_PROMPT:=5}"

: "${LR:=1.0e-6}"
: "${WEIGHT_DECAY:=1.0e-2}"
: "${MAX_GRAD_NORM:=1.0}"
: "${USE_KL_LOSS:=false}"
: "${GRPO_EPS_CLIP_LOW:=0.2}"
: "${GRPO_EPS_CLIP_HIGH:=0.2}"
: "${GPU_MEMORY_UTILIZATION:=0.8}"

: "${DATASET_DIFFICULTY:=easy}"
: "${DATA_KEEP_IN_MEMORY:=false}"
: "${DATASET_NUM_WORKERS:=1}"
: "${DATALOADER_NUM_WORKERS:=0}"

cc_babyai_setup_env
cc_babyai_prepare_layout

if ! cc_babyai_is_truthy "${VALIDATE_ONLY:-false}" && ! cc_babyai_is_truthy "${DRY_RUN:-false}"; then
  cc_babyai_ensure_dataset
fi

common_overrides=()
algo_overrides=()

cc_babyai_build_common_overrides common_overrides

algo_overrides=(
  "trainer.algorithm.advantage_estimator=grpo"
  "trainer.algorithm.policy_loss_type=regular"
  "trainer.algorithm.loss_reduction=token_mean"
  "trainer.algorithm.grpo_norm_by_std=true"
  "trainer.algorithm.eps_clip_low=$GRPO_EPS_CLIP_LOW"
  "trainer.algorithm.eps_clip_high=$GRPO_EPS_CLIP_HIGH"
  "generator.step_wise_trajectories=false"
)

cc_babyai_run_main_base "${common_overrides[@]}" "${algo_overrides[@]}" "$@"
