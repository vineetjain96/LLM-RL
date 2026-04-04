#!/bin/bash
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J skyrl_babyai_text
#SBATCH -o logs/skyrl_%j.out
#SBATCH -e logs/skyrl_%j.err
#SBATCH -A "m5017"
#SBATCH -C "gpu&hbm80g"

set -euo pipefail

#======= Main script parameters =======#
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
ALGO_NAME="${ALGO_NAME:-grpo}"
ENV_NAME="BabyAI-GoToLocal-v0"
MAX_TURNS=64

# Optional sweep controls:
#  - submit an array job: sbatch --array=0-2 scripts/nersc/babyai_launch.slurm
#  - set SWEEP_PARAM and SWEEP_VALUES (comma-separated)
#  - each array task picks one value by SLURM_ARRAY_TASK_ID
SWEEP_PARAM="${SWEEP_PARAM:-}"
SWEEP_VALUES="${SWEEP_VALUES:-}"
SELECTED_SWEEP_VALUE="${SWEEP_VALUE:-}"

# Dataset generation controls
DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-easy}"
DATASET_ENV_NAME="${DATASET_ENV_NAME:-$ENV_NAME}"
DATASET_ENV_KWARGS_JSON="${DATASET_ENV_KWARGS_JSON:-{}}"
ENV_KWARGS_KV="${ENV_KWARGS_KV:-}"
# DATASET_ENV_KWARGS_JSON='{"room_size": 8, "num_dists": 7}'
# ENV_KWARGS_KV="room_size=8,num_dists=7"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

#======= Environment setup =======#
IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8:latest"
WORKROOT="$HOME/LLM-RL"

# Where the BabyAI parquet should live (on SCRATCH)
BABYAI_OUT_DIR="$SCRATCH/data/babyai_text"

# Scratch-backed run dirs
sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

MODEL_NAME_SHORT="${MODEL_NAME##*/}"
MODEL_TAG="$(sanitize_name "$MODEL_NAME_SHORT")"
ENV_TAG="$(sanitize_name "$ENV_NAME")"
ARRAY_TAG=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARRAY_TAG="_a${SLURM_ARRAY_TASK_ID}"
fi
RUN_DIR="$SCRATCH/skyrl/${ALGO_NAME}_${ENV_TAG}_${MODEL_TAG}_${RUN_TIMESTAMP}${ARRAY_TAG}_j${SLURM_JOB_ID}"
CKPT_DIR="$RUN_DIR/ckpts"
WANDB_DIR="$RUN_DIR/wandb"
HF_HOME_DIR="$RUN_DIR/hf_home"
UV_CACHE_DIR="$RUN_DIR/uv_cache"
UV_VENV_DIR="$RUN_DIR/venv"

# Short Ray temp dir
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

mkdir -p "$BABYAI_OUT_DIR" "$CKPT_DIR" "$WANDB_DIR" "$HF_HOME_DIR" "$UV_CACHE_DIR" "$UV_VENV_DIR" "$RAY_TMPDIR"

unset PYTHONPATH

export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman

export PODMANHPC_RUN_ROOT=/tmp/${UID}_hpc_${SLURM_JOB_ID}
export PODMANHPC_GRAPH_ROOT=/tmp/${UID}_hpc_${SLURM_JOB_ID}/storage
mkdir -p "$PODMANHPC_GRAPH_ROOT"
chmod 700 "$PODMANHPC_RUN_ROOT" 2>/dev/null || true

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}

if [[ -n "$SWEEP_PARAM" ]]; then
  if [[ -z "$SELECTED_SWEEP_VALUE" ]]; then
    if [[ -z "$SWEEP_VALUES" ]]; then
      echo "SWEEP_PARAM is set but SWEEP_VALUES is empty." >&2
      exit 1
    fi

    IFS=',' read -r -a SWEEP_VALUE_LIST <<< "$SWEEP_VALUES"
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SWEEP_VALUE_LIST[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of bounds for SWEEP_VALUES=$SWEEP_VALUES" >&2
        exit 1
      fi
      SELECTED_SWEEP_VALUE="${SWEEP_VALUE_LIST[$SLURM_ARRAY_TASK_ID]}"
    else
      SELECTED_SWEEP_VALUE="${SWEEP_VALUE_LIST[0]}"
    fi
  fi
  echo "Sweep selection: ${SWEEP_PARAM}=${SELECTED_SWEEP_VALUE}"
fi

#======= Container script =======#
CONTAINER_SCRIPT=$(cat <<'EOS'
set -euo pipefail

echo "Inside container:"
id

# Respect Slurm GPU assignment
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0,1,2,3}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# NCCL/GLOO bootstrap fix (podman-hpc container only has tap0/lo)
IFACE=$(ip -o -4 route show to default | awk '{print $5; exit}')
echo "Detected default iface: $IFACE"
export NCCL_SOCKET_IFNAME="=${IFACE}"
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME="${IFACE}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Quiet Ray's GPU env warning you saw (optional)
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Critical: ensure SCRATCH exists in-container so scripts/nersc/run_babyai_text.sh doesn't fall back to /data/...
echo "SCRATCH=$SCRATCH"
echo "DATA_DIR=$DATA_DIR"

mkdir -p "$DATA_DIR" "$HF_HOME" "$RAY_TMPDIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT" "$WANDB_DIR" "$CKPT_ROOT"

cd /work/skyrl-train
ls -ld /work/skyrl-train /work/skyrl-gym

# Create/reuse uv venv + install repo deps
uv --version
uv venv "$UV_PROJECT_ENVIRONMENT"
UV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
uv sync --python "$UV_PYTHON"

# Extra deps you requested
uv pip install --python "$UV_PYTHON" gymnasium minigrid

# Generate dataset exactly where training will read it
DATASET_SWEEP_ARGS=()
if [[ -n "${SWEEP_PARAM:-}" && -n "${SWEEP_VALUE:-}" ]]; then
  DATASET_SWEEP_ARGS=(--sweep_param "$SWEEP_PARAM" --sweep_values "$SWEEP_VALUE")
fi
uv run --extra babyai examples/train/babyai_text/babyai_text_dataset.py \
  --output_dir "$DATA_DIR" \
  --difficulty "$DATASET_DIFFICULTY" \
  --env_name "$DATASET_ENV_NAME" \
  --env_kwargs_json "$DATASET_ENV_KWARGS_JSON" \
  --max_turns "$MAX_TURNS" \
  "${DATASET_SWEEP_ARGS[@]}"
ls -lh "$DATA_DIR/train.parquet" "$DATA_DIR/validation.parquet"
if [[ -d "$DATA_DIR/env_previews" ]]; then
  ls -lh "$DATA_DIR/env_previews" || true
  PREVIEW_DST_DIR="$CKPT_ROOT/env_previews"
  mkdir -p "$PREVIEW_DST_DIR"
  cp -r "$DATA_DIR/env_previews/." "$PREVIEW_DST_DIR/"
  rm -rf "$DATA_DIR/env_previews"
  echo "Moved environment previews to: $PREVIEW_DST_DIR"
fi

# OPTIONAL: run a copy of the launcher without --isolated so it uses this venv
tmp_run=/tmp/run_babyai_text_noisolated.sh
sed 's/uv run --isolated/uv run/' scripts/nersc/run_babyai_text.sh > "$tmp_run"
chmod +x "$tmp_run"

# Run training (NUM_GPUS/LOGGER/INFERENCE_BACKEND come from env)
bash "$tmp_run"
EOS
)

#======= Launch container =======#
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
  -e GPUS_PER_NODE="$GPUS_PER_NODE" \
  -e CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0,1,2,3}" \
  -e SCRATCH="$SCRATCH" \
  -e DATA_DIR="$BABYAI_OUT_DIR" \
  -e NUM_GPUS="$GPUS_PER_NODE" \
  -e LOGGER="wandb" \
  -e INFERENCE_BACKEND="vllm" \
  -e HF_HOME="$HF_HOME_DIR" \
  -e WANDB_DIR="$WANDB_DIR" \
  -e CKPT_ROOT="/ckpts" \
  -e MAX_TURNS="$MAX_TURNS" \
  -e ENV_KWARGS_KV="$ENV_KWARGS_KV" \
  -e SWEEP_PARAM="$SWEEP_PARAM" \
  -e SWEEP_VALUES="$SWEEP_VALUES" \
  -e SWEEP_VALUE="$SELECTED_SWEEP_VALUE" \
  -e DATASET_DIFFICULTY="$DATASET_DIFFICULTY" \
  -e DATASET_ENV_NAME="$DATASET_ENV_NAME" \
  -e DATASET_ENV_KWARGS_JSON="$DATASET_ENV_KWARGS_JSON" \
  -e RAY_TMPDIR="$RAY_TMPDIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e UV_PROJECT_ENVIRONMENT="$UV_VENV_DIR" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e ALGO_NAME="$ALGO_NAME" \
  -e ENV_NAME="$ENV_NAME" \
  -e RUN_NAME_TIMESTAMP="$RUN_TIMESTAMP" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  "$IMAGE" -lc "$CONTAINER_SCRIPT"
