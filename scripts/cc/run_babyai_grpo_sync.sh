#!/bin/bash
#SBATCH --job-name=🚀
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --account=aip-siamakx
#SBATCH --output=logs/grpo/%A_%a.out
#SBATCH --error=logs/grpo/%A_%a.err

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

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cc/_babyai_common.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/cc/_babyai_common.sh"

: "${ALGO_NAME:=grpo_sync}"
# : "${MODEL_NAME:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${MODEL_NAME:=Qwen/Qwen3-4B-Instruct-2507}"
: "${ENV_NAME:=BabyAI-GoToLocal-v0}"
: "${ENV_KWARGS_JSON:="{}"}"
: "${MAX_TURNS:=16}"
: "${EXPERIMENT_ROOT:="$HOME/scratch/babyai"}"

: "${NUM_GPUS:=4}"
: "${LOGGER:="[wandb,console]"}"
: "${PROJECT_NAME:=babyai}"
: "${INFERENCE_BACKEND:=vllm}"
: "${SEED:=42}"

: "${EPOCHS:=40}"
: "${EVAL_BATCH_SIZE:=256}"
: "${EVAL_BEFORE_TRAIN:=true}"
: "${EVAL_INTERVAL:=5}"
: "${UPDATE_EPOCHS_PER_BATCH:=1}"
: "${TRAIN_BATCH_SIZE:=256}"
: "${POLICY_MINI_BATCH_SIZE:=256}"
: "${MICRO_FORWARD_BATCH_SIZE:=16}"
: "${MICRO_TRAIN_BATCH_SIZE:=4}"
: "${CKPT_INTERVAL:=10}"
: "${MAX_CKPTS_TO_KEEP:=3}"

: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_INPUT_LENGTH:=16384}"
: "${MAX_GENERATE_LENGTH:=1024}"
: "${N_SAMPLES_PER_PROMPT:=4}"

: "${LR:=1.0e-6}"
: "${WEIGHT_DECAY:=1.0e-2}"
: "${MAX_GRAD_NORM:=1.0}"
: "${USE_KL_LOSS:=false}"
: "${GRPO_EPS_CLIP_LOW:=0.2}"
: "${GRPO_EPS_CLIP_HIGH:=0.2}"
: "${GPU_MEMORY_UTILIZATION:=0.8}"

: "${DATASET_DIFFICULTY:=easy}"
: "${DATA_KEEP_IN_MEMORY:=true}"
: "${DATASET_NUM_WORKERS:=8}"
: "${DATALOADER_NUM_WORKERS:=8}"

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
  "generator.zero_reward_on_non_stop=true"
  "generator.apply_overlong_filtering=false"
  "generator.step_wise_trajectories=false"
)

cc_babyai_run_main_base "${common_overrides[@]}" "${algo_overrides[@]}" "$@"
