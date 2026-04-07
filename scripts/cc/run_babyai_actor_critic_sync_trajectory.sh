#!/bin/bash
#SBATCH --job-name=🚀🚀
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --account=aip-siamakx
#SBATCH --output=logs/actor_critic_traj/%A_%a.out
#SBATCH --error=logs/actor_critic_traj/%A_%a.err

set -euo pipefail
set -x

# Synchronous BabyAI actor-critic launcher for the state_action_td estimator.
# This keeps the main rollout knobs aligned with the sync GRPO launcher and only
# changes the pieces required by the algorithm: critic model, critic optimizer,
# and step-wise trajectories.
#
# Examples:
#   bash scripts/cc/run_babyai_actor_critic_sync.sh
#   MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 bash scripts/cc/run_babyai_actor_critic_sync.sh
#   ENV_NAME=BabyAI-GoToObj-v0 ENV_KWARGS_JSON='{"room_size": 8}' bash scripts/cc/run_babyai_actor_critic_sync_trajectory.sh
#   VALIDATE_ONLY=true bash scripts/cc/run_babyai_actor_critic_sync.sh

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cc/_babyai_common.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "$REPO_ROOT"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/cc/_babyai_common.sh"

: "${ALGO_NAME:=state_action_td_traj}"
# : "${MODEL_NAME:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${MODEL_NAME:=Qwen/Qwen3-4B-Instruct-2507}"
: "${ENV_NAME:=BabyAI-GoToLocal-v0}"
: "${ENV_KWARGS_JSON:="{}"}"
: "${MAX_TURNS:=8}"
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
: "${TRAIN_BATCH_SIZE:=128}"
: "${POLICY_MINI_BATCH_UPDATES:=1}"
: "${CRITIC_MINI_BATCH_UPDATES:=1}"
: "${POLICY_EPOCHS_PER_BATCH:=1}"
: "${CRITIC_EPOCHS_PER_BATCH:=4}"
: "${ACTOR_ADVANTAGE_TYPE:=gae}"
: "${MICRO_FORWARD_BATCH_SIZE:=4}"
: "${MICRO_TRAIN_BATCH_SIZE:=2}"
: "${CKPT_INTERVAL:=5}"
: "${MAX_CKPTS_TO_KEEP:=3}"

: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_INPUT_LENGTH:=16384}"
: "${MAX_GENERATE_LENGTH:=2048}"
: "${N_SAMPLES_PER_PROMPT:=1}"

: "${LR:=1.0e-6}"
: "${CRITIC_LR:=5.0e-6}"
: "${WEIGHT_DECAY:=1.0e-2}"
: "${MAX_GRAD_NORM:=1.0}"
: "${USE_KL_LOSS:=false}"
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
  "trainer.algorithm.advantage_estimator=state_action_td"
  "trainer.algorithm.policy_loss_type=regular"
  "trainer.algorithm.loss_reduction=token_mean"
  "trainer.critic.model.path=$MODEL_NAME"
  "trainer.critic.optimizer_config.lr=$CRITIC_LR"
  "trainer.critic.optimizer_config.weight_decay=$WEIGHT_DECAY"
  "trainer.critic.optimizer_config.max_grad_norm=$MAX_GRAD_NORM"
  "generator.zero_reward_on_non_stop=true"
  "generator.apply_overlong_filtering=false"
  "generator.step_wise_trajectories=false"
  "trainer.algorithm.gamma=0.99"
  "trainer.algorithm.lambd=0.95"
  "trainer.algorithm.state_action.critic_head_bias=true"
  "trainer.algorithm.state_action.actor_advantage_type=$ACTOR_ADVANTAGE_TYPE"
  "trainer.algorithm.state_action.policy_mini_batch_updates=$POLICY_MINI_BATCH_UPDATES"
  "trainer.algorithm.state_action.critic_mini_batch_updates=$CRITIC_MINI_BATCH_UPDATES"
  "trainer.algorithm.state_action.policy_epochs_per_batch=$POLICY_EPOCHS_PER_BATCH"
  "trainer.algorithm.state_action.critic_epochs_per_batch=$CRITIC_EPOCHS_PER_BATCH"
)

cc_babyai_run_main_base "${common_overrides[@]}" "${algo_overrides[@]}" "$@"
