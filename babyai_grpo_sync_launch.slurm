#!/bin/bash
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J babyai_grpo_sync
#SBATCH -o logs/babyai_grpo_sync_%j.out
#SBATCH -e logs/babyai_grpo_sync_%j.err
#SBATCH -A "m5017"
#SBATCH -C "gpu&hbm80g"

set -euo pipefail

# Usage:
#   sbatch babyai_grpo_sync_launch.slurm
#
# This is the matched synchronous GRPO baseline for the new TD run.
# It aligns the main rollout knobs with the TD launcher:
# - same single-node colocated setup
# - same max_input_length
# - same max_generate_length
# - same n_samples_per_prompt
#
# Optional overrides:
#   MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 sbatch babyai_grpo_sync_launch.slurm
#   ENV_KWARGS_KV="room_size=8,num_dists=2" sbatch babyai_grpo_sync_launch.slurm
#   N_SAMPLES_PER_PROMPT=1 sbatch babyai_grpo_sync_launch.slurm
#   EXTRA_OVERRIDES="trainer.epochs=20 trainer.train_batch_size=256" sbatch babyai_grpo_sync_launch.slurm

# ======= Main script parameters ======= #
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
ALGO_NAME="${ALGO_NAME:-grpo_sync}"
ENV_NAME="${ENV_NAME:-BabyAI-GoToLocal-v0}"
MAX_TURNS="${MAX_TURNS:-32}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

# Dataset generation controls
DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-easy}"
DATASET_ENV_NAME="${DATASET_ENV_NAME:-$ENV_NAME}"
DATASET_ENV_KWARGS_JSON="${DATASET_ENV_KWARGS_JSON:-{}}"

# Launcher controls
ENV_KWARGS_KV="${ENV_KWARGS_KV:-}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-5}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-4096}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-256}"
GRPO_USE_KL_LOSS="${GRPO_USE_KL_LOSS:-false}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# ======= Environment setup ======= #
IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8:latest"
WORKROOT="$HOME/LLM-RL"
BABYAI_OUT_DIR="$SCRATCH/data/babyai_text"

sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

MODEL_NAME_SHORT="${MODEL_NAME##*/}"
MODEL_TAG="$(sanitize_name "$MODEL_NAME_SHORT")"
ENV_TAG="$(sanitize_name "$ENV_NAME")"
RUN_DIR="$SCRATCH/skyrl/${ALGO_NAME}_${ENV_TAG}_${MODEL_TAG}_${RUN_TIMESTAMP}_j${SLURM_JOB_ID}"
CKPT_DIR="$RUN_DIR/ckpts"
WANDB_DIR="$RUN_DIR/wandb"
HF_HOME_DIR="$RUN_DIR/hf_home"
UV_CACHE_DIR="$RUN_DIR/uv_cache"
UV_VENV_DIR="$RUN_DIR/venv"
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

mkdir -p "$BABYAI_OUT_DIR" "$CKPT_DIR" "$WANDB_DIR" "$HF_HOME_DIR" "$UV_CACHE_DIR" "$UV_VENV_DIR" "$RAY_TMPDIR"

unset PYTHONPATH

export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
export PODMANHPC_RUN_ROOT=/tmp/${UID}_hpc_${SLURM_JOB_ID}
export PODMANHPC_GRAPH_ROOT=/tmp/${UID}_hpc_${SLURM_JOB_ID}/storage
mkdir -p "$PODMANHPC_GRAPH_ROOT"
chmod 700 "$PODMANHPC_RUN_ROOT" 2>/dev/null || true

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}

CONTAINER_SCRIPT=$(cat <<'EOS'
set -euo pipefail

echo "Inside container:"
id

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0,1,2,3}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

IFACE=$(ip -o -4 route show to default | awk '{print $5; exit}')
echo "Detected default iface: $IFACE"
export NCCL_SOCKET_IFNAME="=${IFACE}"
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME="${IFACE}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

mkdir -p "$DATA_DIR" "$HF_HOME" "$RAY_TMPDIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT" "$WANDB_DIR" "$CKPT_ROOT"

cd /work/skyrl-train
ls -ld /work/skyrl-train /work/skyrl-gym

uv --version
uv venv "$UV_PROJECT_ENVIRONMENT"
UV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
uv sync --python "$UV_PYTHON"
uv pip install --python "$UV_PYTHON" gymnasium minigrid

uv run --extra babyai examples/babyai_text/babyai_text_dataset.py \
  --output_dir "$DATA_DIR" \
  --difficulty "$DATASET_DIFFICULTY" \
  --env_name "$DATASET_ENV_NAME" \
  --env_kwargs_json "$DATASET_ENV_KWARGS_JSON" \
  --max_turns "$MAX_TURNS"

ls -lh "$DATA_DIR/train.parquet" "$DATA_DIR/validation.parquet"

tmp_run=/tmp/run_babyai_text_noisolated.sh
sed 's/uv run --isolated/uv run/' examples/babyai_text/run_babyai_text.sh > "$tmp_run"
chmod +x "$tmp_run"

if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($EXTRA_OVERRIDES)
else
  extra_args=()
fi

bash "$tmp_run" \
  "generator.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}" \
  "generator.max_input_length=${MAX_INPUT_LENGTH}" \
  "generator.sampling_params.max_generate_length=${MAX_GENERATE_LENGTH}" \
  "generator.step_wise_trajectories=false" \
  "trainer.algorithm.use_kl_loss=${GRPO_USE_KL_LOSS}" \
  "${extra_args[@]}"
EOS
)

srun -n 1 podman-hpc run --rm \
  --gpu \
  --shm-size=8g \
  --user 0 \
  --entrypoint /bin/bash \
  -v "$WORKROOT:/work:rw" \
  -v "$SCRATCH:$SCRATCH:rw" \
  -v "$CKPT_DIR:/ckpts:rw" \
  -w /work/skyrl-train \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  -e SLURM_JOB_GPUS="${SLURM_JOB_GPUS:-}" \
  -e CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0,1,2,3}" \
  -e GPUS_PER_NODE="$GPUS_PER_NODE" \
  -e SCRATCH="$SCRATCH" \
  -e DATA_DIR="$BABYAI_OUT_DIR" \
  -e NUM_GPUS="$GPUS_PER_NODE" \
  -e LOGGER="${LOGGER:-wandb}" \
  -e INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}" \
  -e HF_HOME="$HF_HOME_DIR" \
  -e WANDB_DIR="$WANDB_DIR" \
  -e CKPT_ROOT="/ckpts" \
  -e RAY_TMPDIR="$RAY_TMPDIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e UV_PROJECT_ENVIRONMENT="$UV_VENV_DIR" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e ALGO_NAME="$ALGO_NAME" \
  -e ENV_NAME="$ENV_NAME" \
  -e MAX_TURNS="$MAX_TURNS" \
  -e ENV_KWARGS_KV="$ENV_KWARGS_KV" \
  -e DATASET_DIFFICULTY="$DATASET_DIFFICULTY" \
  -e DATASET_ENV_NAME="$DATASET_ENV_NAME" \
  -e DATASET_ENV_KWARGS_JSON="$DATASET_ENV_KWARGS_JSON" \
  -e RUN_NAME_TIMESTAMP="$RUN_TIMESTAMP" \
  -e N_SAMPLES_PER_PROMPT="$N_SAMPLES_PER_PROMPT" \
  -e MAX_INPUT_LENGTH="$MAX_INPUT_LENGTH" \
  -e MAX_GENERATE_LENGTH="$MAX_GENERATE_LENGTH" \
  -e GRPO_USE_KL_LOSS="$GRPO_USE_KL_LOSS" \
  -e EXTRA_OVERRIDES="$EXTRA_OVERRIDES" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  "$IMAGE" -lc "$CONTAINER_SCRIPT"
