#!/bin/bash
#SBATCH -q premium
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J skyrl_babyai_fully_async
#SBATCH -o logs/skyrl_fasync_%j.out
#SBATCH -e logs/skyrl_fasync_%j.err
#SBATCH -A "m5017"
#SBATCH -C "gpu&hbm80g"

set -euo pipefail

#======= Main script parameters =======#
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
ENV_NAME="${ENV_NAME:-BabyAI-GoToLocal-v0}"
MAX_TURNS="${MAX_TURNS:-64}"

# Optional fixed env kwargs passed to launcher
# Example: ENV_KWARGS_KV="room_size=8,num_dists=2"
ENV_KWARGS_KV="${ENV_KWARGS_KV:-}"

# Optional sweep controls over environment kwargs
SWEEP_PARAM="${SWEEP_PARAM:-}"
SWEEP_VALUES="${SWEEP_VALUES:-}"
SELECTED_SWEEP_VALUE="${SWEEP_VALUE:-}"

# Dataset generation controls
DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-easy}"
DATASET_ENV_NAME="${DATASET_ENV_NAME:-$ENV_NAME}"
DATASET_ENV_KWARGS_JSON="${DATASET_ENV_KWARGS_JSON:-{}}"

# Fully-async controls
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-4}"
ALGO_NAME="${ALGO_NAME:-grpo}"
# Leave unset to auto-compute a full-GPU split across all allocated nodes.
NUM_POLICY_GPUS="${NUM_POLICY_GPUS:-}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-}"
# Optional override; by default script computes MINI_BATCH_SIZE*(MAX_STALENESS_STEPS+1)
NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-}"
RUN_NAME_TIMESTAMP="${RUN_NAME_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RAY_PORT="${RAY_PORT:-6479}"

# Training knobs (kept aligned with BabyAI GRPO setup/sweep defaults)
EPOCHS="${EPOCHS:-40}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
MICRO_FORWARD_BATCH_SIZE="${MICRO_FORWARD_BATCH_SIZE:-16}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-2048}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-5}"
LR="${LR:-1.0e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
CKPT_INTERVAL="${CKPT_INTERVAL:-10}"
KL_ESTIMATOR_TYPE="${KL_ESTIMATOR_TYPE:-k3}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
GRPO_USE_KL_LOSS="${GRPO_USE_KL_LOSS:-false}"
GRPO_EPS_CLIP_LOW="${GRPO_EPS_CLIP_LOW:-0.2}"
GRPO_EPS_CLIP_HIGH="${GRPO_EPS_CLIP_HIGH:-0.2}"
USE_TIS="${USE_TIS:-false}"

#======= Environment setup =======#
IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8:latest"
WORKROOT="$HOME/LLM-RL"
BABYAI_OUT_DIR="$SCRATCH/data/babyai_text"

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
RUN_DIR="$SCRATCH/skyrl/${ALGO_NAME}_async_2node_${ENV_TAG}_${MODEL_TAG}_${RUN_NAME_TIMESTAMP}${ARRAY_TAG}_j${SLURM_JOB_ID}"
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
NUM_NODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))

if [[ -z "$NUM_POLICY_GPUS" ]]; then
  NUM_POLICY_GPUS=$((GPUS_PER_NODE / 2))
  if (( NUM_POLICY_GPUS < 1 )); then
    NUM_POLICY_GPUS=1
  fi
fi

if [[ -z "$NUM_INFERENCE_ENGINES" ]]; then
  NUM_INFERENCE_ENGINES=$((TOTAL_GPUS - (NUM_POLICY_GPUS * NUM_NODES)))
  if (( NUM_INFERENCE_ENGINES < 1 )); then
    NUM_INFERENCE_ENGINES=1
  fi
fi

if (( NUM_POLICY_GPUS > GPUS_PER_NODE )); then
  echo "Invalid NUM_POLICY_GPUS(${NUM_POLICY_GPUS}) > GPUS_PER_NODE(${GPUS_PER_NODE})" >&2
  exit 1
fi

TOTAL_POLICY_GPUS=$((NUM_POLICY_GPUS * NUM_NODES))
if (( TOTAL_POLICY_GPUS + NUM_INFERENCE_ENGINES > TOTAL_GPUS )); then
  echo "Invalid GPU split: TOTAL_POLICY_GPUS(${TOTAL_POLICY_GPUS}) + NUM_INFERENCE_ENGINES(${NUM_INFERENCE_ENGINES}) > TOTAL_GPUS(${TOTAL_GPUS})" >&2
  exit 1
fi

echo "Using ${NUM_NODES} node(s), ${GPUS_PER_NODE} GPU(s)/node, total ${TOTAL_GPUS} GPU(s)."
echo "Configured split: policy=${NUM_POLICY_GPUS} GPU(s)/node (${TOTAL_POLICY_GPUS} total), inference engines=${NUM_INFERENCE_ENGINES}."

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

HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
HEAD_IP="$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1 {print $1}')"
if [[ -z "$HEAD_IP" ]]; then
  echo "Failed to resolve IPv4 for head node $HEAD_NODE" >&2
  exit 1
fi
HEAD_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "Ray head node: ${HEAD_NODE} (${HEAD_IP}), address=${HEAD_ADDRESS}"

#======= Container script =======#
CONTAINER_SCRIPT=$(cat <<'EOS'
set -euo pipefail

echo "Inside container:"
id

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0,1,2,3}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

NODE_RANK="${SLURM_PROCID:-0}"
NUM_NODES="${SLURM_NNODES:-1}"
HEAD_NODE="${SLURM_HEAD_NODE}"
HEAD_IP="${SLURM_HEAD_IP}"
HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-${HEAD_IP}:${RAY_PORT}}"
DONE_FILE="${CKPT_ROOT}/.ray_training_done"

IFACE=$(ip -o -4 route show to default | awk '{print $5; exit}')
echo "Detected default iface: $IFACE"
export NCCL_SOCKET_IFNAME="=${IFACE}"
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME="${IFACE}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

wait_for_ray() {
  local address="$1"
  for _ in $(seq 1 60); do
    if ray status --address "$address" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Ray cluster at $address failed to become ready" >&2
  return 1
}

if [[ "$NODE_RANK" == "0" ]]; then
  rm -f "$DONE_FILE"
  if ! ray status --address "${HEAD_ADDRESS}" >/dev/null 2>&1; then
    ray start --head --disable-usage-stats --port "${RAY_PORT}" --node-ip-address "${HEAD_IP}"
  fi
  wait_for_ray "${HEAD_ADDRESS}"
  export RAY_ADDRESS="${HEAD_ADDRESS}"
  # Give worker nodes time to connect before trainer starts allocating placement groups.
  sleep 10
else
  if ! ray status --address "${HEAD_ADDRESS}" >/dev/null 2>&1; then
    ray start --address "${HEAD_ADDRESS}" --disable-usage-stats
  fi
  wait_for_ray "${HEAD_ADDRESS}"
  echo "Ray worker node rank ${NODE_RANK}/${NUM_NODES} ready. Waiting for completion signal..."
  while [[ ! -f "$DONE_FILE" ]]; do
    sleep 10
  done
  ray stop --force || true
  exit 0
fi

cleanup_head() {
  touch "$DONE_FILE"
  ray stop --force || true
}
trap cleanup_head EXIT

mkdir -p "$DATA_DIR" "$HF_HOME" "$RAY_TMPDIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT" "$WANDB_DIR" "$CKPT_ROOT"

cd /work/skyrl-train
ls -ld /work/skyrl-train /work/skyrl-gym

uv --version
uv venv "$UV_PROJECT_ENVIRONMENT"
UV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
uv sync --python "$UV_PYTHON"
uv pip install --python "$UV_PYTHON" gymnasium minigrid

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

# Use non-isolated uv run so we use the venv created above.
tmp_run=/tmp/run_babyai_grpo_fully_async_noisolated.sh
sed 's/uv run --isolated/uv run/' scripts/nersc/run_babyai_grpo_fully_async.sh > "$tmp_run"
chmod +x "$tmp_run"
bash "$tmp_run" \
  "trainer.placement.policy_num_nodes=${NUM_NODES}" \
  "trainer.placement.ref_num_nodes=${NUM_NODES}"
EOS
)

#======= Launch container =======#
srun -n "${NUM_NODES}" podman-hpc run --rm \
  --network host \
  --gpu \
  --shm-size=8g \
  --user 0 \
  --entrypoint /bin/bash \
  -v "$WORKROOT:/work:rw" \
  -v "$SCRATCH:$SCRATCH:rw" \
  -v "$CKPT_DIR:/ckpts:rw" \
  -w /work/skyrl-train \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  -e SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}" \
  -e SLURM_JOB_GPUS="${SLURM_JOB_GPUS:-}" \
  -e SLURM_PROCID \
  -e SLURM_NNODES \
  -e SLURM_NODEID \
  -e SLURM_JOB_NODELIST \
  -e SLURM_HEAD_NODE="$HEAD_NODE" \
  -e SLURM_HEAD_IP="$HEAD_IP" \
  -e RAY_HEAD_ADDRESS="$HEAD_ADDRESS" \
  -e RAY_PORT="$RAY_PORT" \
  -e GPUS_PER_NODE="$GPUS_PER_NODE" \
  -e CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0,1,2,3}" \
  -e SCRATCH="$SCRATCH" \
  -e DATA_DIR="$BABYAI_OUT_DIR" \
  -e LOGGER="wandb" \
  -e INFERENCE_BACKEND="vllm" \
  -e HF_HOME="$HF_HOME_DIR" \
  -e WANDB_DIR="$WANDB_DIR" \
  -e CKPT_ROOT="/ckpts" \
  -e MAX_TURNS="$MAX_TURNS" \
  -e ENV_KWARGS_KV="$ENV_KWARGS_KV" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e ENV_NAME="$ENV_NAME" \
  -e SWEEP_PARAM="$SWEEP_PARAM" \
  -e SWEEP_VALUES="$SWEEP_VALUES" \
  -e SWEEP_VALUE="$SELECTED_SWEEP_VALUE" \
  -e DATASET_DIFFICULTY="$DATASET_DIFFICULTY" \
  -e DATASET_ENV_NAME="$DATASET_ENV_NAME" \
  -e DATASET_ENV_KWARGS_JSON="$DATASET_ENV_KWARGS_JSON" \
  -e NUM_POLICY_GPUS="$NUM_POLICY_GPUS" \
  -e NUM_INFERENCE_ENGINES="$NUM_INFERENCE_ENGINES" \
  -e ALGO_NAME="$ALGO_NAME" \
  -e MAX_STALENESS_STEPS="$MAX_STALENESS_STEPS" \
  -e NUM_PARALLEL_GENERATION_WORKERS="$NUM_PARALLEL_GENERATION_WORKERS" \
  -e RUN_NAME_TIMESTAMP="$RUN_NAME_TIMESTAMP" \
  -e EPOCHS="$EPOCHS" \
  -e MINI_BATCH_SIZE="$MINI_BATCH_SIZE" \
  -e MICRO_FORWARD_BATCH_SIZE="$MICRO_FORWARD_BATCH_SIZE" \
  -e MICRO_TRAIN_BATCH_SIZE="$MICRO_TRAIN_BATCH_SIZE" \
  -e MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
  -e MAX_GENERATE_LENGTH="$MAX_GENERATE_LENGTH" \
  -e N_SAMPLES_PER_PROMPT="$N_SAMPLES_PER_PROMPT" \
  -e LR="$LR" \
  -e WEIGHT_DECAY="$WEIGHT_DECAY" \
  -e MAX_GRAD_NORM="$MAX_GRAD_NORM" \
  -e CKPT_INTERVAL="$CKPT_INTERVAL" \
  -e KL_ESTIMATOR_TYPE="$KL_ESTIMATOR_TYPE" \
  -e KL_LOSS_COEF="$KL_LOSS_COEF" \
  -e GRPO_USE_KL_LOSS="$GRPO_USE_KL_LOSS" \
  -e GRPO_EPS_CLIP_LOW="$GRPO_EPS_CLIP_LOW" \
  -e GRPO_EPS_CLIP_HIGH="$GRPO_EPS_CLIP_HIGH" \
  -e USE_TIS="$USE_TIS" \
  -e RAY_TMPDIR="$RAY_TMPDIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e UV_PROJECT_ENVIRONMENT="$UV_VENV_DIR" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  "$IMAGE" -lc "$CONTAINER_SCRIPT"
