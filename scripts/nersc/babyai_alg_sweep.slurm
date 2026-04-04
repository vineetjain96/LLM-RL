#!/bin/bash
#SBATCH -q premium
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --array=0-3
#SBATCH -J skyrl_babyai_alg_sweep_async
#SBATCH -o logs/skyrl_alg_%A_%a.out
#SBATCH -e logs/skyrl_alg_%A_%a.err
#SBATCH -A "m5017"
#SBATCH -C "gpu&hbm80g"

set -euo pipefail

# ------------------------------------------------------------
# Array algorithm selection
# ------------------------------------------------------------
# Default compares these four algorithms apples-to-apples:
#   grpo, rloo, cispo, ppo
#
# Submit:
#   sbatch --array=0-3 scripts/nersc/babyai_alg_sweep.slurm
#
# Optional custom order/subset:
#   ALGO_LIST="grpo,ppo" sbatch --array=0-1 scripts/nersc/babyai_alg_sweep.slurm
ALGO_LIST="${ALGO_LIST:-grpo,rloo,cispo,ppo}"
IFS=',' read -r -a ALGO_VALUES <<< "$ALGO_LIST"

if (( ${#ALGO_VALUES[@]} == 0 )); then
  echo "ALGO_LIST is empty." >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit this script as an array job." >&2
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#ALGO_VALUES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for ALGO_LIST=$ALGO_LIST" >&2
  exit 1
fi

SELECTED_ALGO="${ALGO_VALUES[$SLURM_ARRAY_TASK_ID]}"
echo "Selected algorithm for array task ${SLURM_ARRAY_TASK_ID}: ${SELECTED_ALGO}"

# ------------------------------------------------------------
# Shared run parameters (same env for apples-to-apples compare)
# ------------------------------------------------------------
# Model / environment
# MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
ENV_NAME="${ENV_NAME:-BabyAI-GoToLocal-v0}"
MAX_TURNS="${MAX_TURNS:-64}"
# ENV_KWARGS_KV="${ENV_KWARGS_KV:-room_size=8,num_dists=7}"
ENV_KWARGS_KV="${ENV_KWARGS_KV:-}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

# Optional dataset generation controls
GENERATE_DATASET="${GENERATE_DATASET:-true}"
DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-easy}"
DATASET_ENV_NAME="${DATASET_ENV_NAME:-$ENV_NAME}"
# DATASET_ENV_KWARGS_JSON="${DATASET_ENV_KWARGS_JSON:-{\"room_size\": 8, \"num_dists\": 7}}"
DATASET_ENV_KWARGS_JSON="${DATASET_ENV_KWARGS_JSON:-{}}"

# Shared algorithm sweep settings
RUN_PREFIX="${RUN_PREFIX:-babyai_alg_sweep_${RUN_TIMESTAMP}}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-40}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-512}"
MICRO_FORWARD_BATCH_SIZE="${MICRO_FORWARD_BATCH_SIZE:-16}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-2048}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-5}"
LR="${LR:-1.0e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
CKPT_INTERVAL="${CKPT_INTERVAL:-100}"
KL_ESTIMATOR_TYPE="${KL_ESTIMATOR_TYPE:-k3}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
# Fully-async placement/staleness settings
NUM_POLICY_GPUS_DEFAULT_NON_PPO="${NUM_POLICY_GPUS_DEFAULT_NON_PPO:-4}"
NUM_POLICY_GPUS_DEFAULT_PPO="${NUM_POLICY_GPUS_DEFAULT_PPO:-2}"
NUM_INFERENCE_ENGINES_DEFAULT_NON_PPO="${NUM_INFERENCE_ENGINES_DEFAULT_NON_PPO:-4}"
NUM_INFERENCE_ENGINES_DEFAULT_PPO="${NUM_INFERENCE_ENGINES_DEFAULT_PPO:-4}"
POLICY_NUM_NODES_DEFAULT_NON_PPO="${POLICY_NUM_NODES_DEFAULT_NON_PPO:-1}"
POLICY_NUM_NODES_DEFAULT_PPO="${POLICY_NUM_NODES_DEFAULT_PPO:-1}"
CRITIC_NUM_NODES_DEFAULT_NON_PPO="${CRITIC_NUM_NODES_DEFAULT_NON_PPO:-1}"
CRITIC_NUM_NODES_DEFAULT_PPO="${CRITIC_NUM_NODES_DEFAULT_PPO:-1}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-4}"
RAY_PORT="${RAY_PORT:-6479}"

if [[ -z "${NUM_POLICY_GPUS:-}" ]]; then
  if [[ "$SELECTED_ALGO" == "ppo" ]]; then
    NUM_POLICY_GPUS="$NUM_POLICY_GPUS_DEFAULT_PPO"
  else
    NUM_POLICY_GPUS="$NUM_POLICY_GPUS_DEFAULT_NON_PPO"
  fi
fi

if [[ -z "${NUM_INFERENCE_ENGINES:-}" ]]; then
  if [[ "$SELECTED_ALGO" == "ppo" ]]; then
    NUM_INFERENCE_ENGINES="$NUM_INFERENCE_ENGINES_DEFAULT_PPO"
  else
    NUM_INFERENCE_ENGINES="$NUM_INFERENCE_ENGINES_DEFAULT_NON_PPO"
  fi
fi

if [[ -z "${POLICY_NUM_NODES:-}" ]]; then
  if [[ "$SELECTED_ALGO" == "ppo" ]]; then
    POLICY_NUM_NODES="$POLICY_NUM_NODES_DEFAULT_PPO"
  else
    POLICY_NUM_NODES="$POLICY_NUM_NODES_DEFAULT_NON_PPO"
  fi
fi

REF_NUM_NODES="${REF_NUM_NODES:-$POLICY_NUM_NODES}"

if [[ -z "${CRITIC_NUM_NODES:-}" ]]; then
  if [[ "$SELECTED_ALGO" == "ppo" ]]; then
    CRITIC_NUM_NODES="$CRITIC_NUM_NODES_DEFAULT_PPO"
  else
    CRITIC_NUM_NODES="$CRITIC_NUM_NODES_DEFAULT_NON_PPO"
  fi
fi

echo "Selected placement controls:"
echo "  NUM_POLICY_GPUS=${NUM_POLICY_GPUS}, NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES}"
echo "  POLICY_NUM_NODES=${POLICY_NUM_NODES}, REF_NUM_NODES=${REF_NUM_NODES}, CRITIC_NUM_NODES=${CRITIC_NUM_NODES}"

# Optional per-alg overrides (pass-through to sweep script)
EXTRA_OVERRIDES_ALL="${EXTRA_OVERRIDES_ALL:-}"
EXTRA_OVERRIDES_GRPO="${EXTRA_OVERRIDES_GRPO:-}"
EXTRA_OVERRIDES_RLOO="${EXTRA_OVERRIDES_RLOO:-}"
EXTRA_OVERRIDES_CISPO="${EXTRA_OVERRIDES_CISPO:-}"
EXTRA_OVERRIDES_PPO="${EXTRA_OVERRIDES_PPO:-}"

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
IMAGE="docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8:latest"
WORKROOT="$HOME/LLM-RL"
BABYAI_OUT_DIR="$SCRATCH/data/babyai_text"

sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

MODEL_NAME_SHORT="${MODEL_NAME##*/}"
MODEL_TAG="$(sanitize_name "$MODEL_NAME_SHORT")"
ENV_TAG="$(sanitize_name "$ENV_NAME")"
RUN_DIR="$SCRATCH/skyrl/${SELECTED_ALGO}_async_${ENV_TAG}_${MODEL_TAG}_${RUN_TIMESTAMP}_a${SLURM_ARRAY_TASK_ID}_j${SLURM_JOB_ID}"
CKPT_DIR="$RUN_DIR/ckpts"
WANDB_DIR="$RUN_DIR/wandb"
HF_HOME_DIR="$RUN_DIR/hf_home"
UV_CACHE_DIR="$RUN_DIR/uv_cache"
UV_VENV_DIR="$RUN_DIR/venv"
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$BABYAI_OUT_DIR" "$CKPT_DIR" "$WANDB_DIR" "$HF_HOME_DIR" "$UV_CACHE_DIR" "$UV_VENV_DIR" "$RAY_TMPDIR"
unset PYTHONPATH

export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
export PODMANHPC_RUN_ROOT=/tmp/${UID}_hpc_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PODMANHPC_GRAPH_ROOT=/tmp/${UID}_hpc_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/storage
mkdir -p "$PODMANHPC_GRAPH_ROOT"
chmod 700 "$PODMANHPC_RUN_ROOT" 2>/dev/null || true

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
NUM_NODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))

if (( NUM_POLICY_GPUS > GPUS_PER_NODE )); then
  echo "Invalid NUM_POLICY_GPUS(${NUM_POLICY_GPUS}) > GPUS_PER_NODE(${GPUS_PER_NODE})" >&2
  exit 1
fi

if (( POLICY_NUM_NODES < 1 || POLICY_NUM_NODES > NUM_NODES )); then
  echo "Invalid POLICY_NUM_NODES(${POLICY_NUM_NODES}) for NUM_NODES(${NUM_NODES})" >&2
  exit 1
fi

if (( REF_NUM_NODES < 1 || REF_NUM_NODES > NUM_NODES )); then
  echo "Invalid REF_NUM_NODES(${REF_NUM_NODES}) for NUM_NODES(${NUM_NODES})" >&2
  exit 1
fi

if (( CRITIC_NUM_NODES < 1 || CRITIC_NUM_NODES > NUM_NODES )); then
  echo "Invalid CRITIC_NUM_NODES(${CRITIC_NUM_NODES}) for NUM_NODES(${NUM_NODES})" >&2
  exit 1
fi

if (( REF_NUM_NODES != POLICY_NUM_NODES )); then
  echo "REF_NUM_NODES(${REF_NUM_NODES}) must equal POLICY_NUM_NODES(${POLICY_NUM_NODES}) when policy/ref are colocated." >&2
  exit 1
fi

TOTAL_POLICY_GPUS=$((NUM_POLICY_GPUS * POLICY_NUM_NODES))
TOTAL_CRITIC_GPUS=0
if [[ "$SELECTED_ALGO" == "ppo" ]]; then
  TOTAL_CRITIC_GPUS=$((NUM_POLICY_GPUS * CRITIC_NUM_NODES))
fi

if (( TOTAL_POLICY_GPUS + TOTAL_CRITIC_GPUS + NUM_INFERENCE_ENGINES > TOTAL_GPUS )); then
  echo "Invalid GPU split: TOTAL_POLICY_GPUS(${TOTAL_POLICY_GPUS}) + TOTAL_CRITIC_GPUS(${TOTAL_CRITIC_GPUS}) + NUM_INFERENCE_ENGINES(${NUM_INFERENCE_ENGINES}) > TOTAL_GPUS(${TOTAL_GPUS})" >&2
  exit 1
fi

echo "Using ${NUM_NODES} node(s), ${GPUS_PER_NODE} GPU(s)/node, total ${TOTAL_GPUS} GPU(s)."
echo "Configured split: policy=${NUM_POLICY_GPUS} GPU(s)/node across ${POLICY_NUM_NODES} node(s) (${TOTAL_POLICY_GPUS} total), critic_total=${TOTAL_CRITIC_GPUS}, inference engines=${NUM_INFERENCE_ENGINES}."

HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
HEAD_IP="$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1 {print $1}')"
if [[ -z "$HEAD_IP" ]]; then
  echo "Failed to resolve IPv4 for head node $HEAD_NODE" >&2
  exit 1
fi
HEAD_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "Ray head node: ${HEAD_NODE} (${HEAD_IP}), address=${HEAD_ADDRESS}"

# ------------------------------------------------------------
# Container script
# ------------------------------------------------------------
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

if [[ "$GENERATE_DATASET" == "true" ]]; then
  if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/validation.parquet" ]]; then
    uv run --extra babyai examples/train/babyai_text/babyai_text_dataset.py \
      --output_dir "$DATA_DIR" \
      --difficulty "$DATASET_DIFFICULTY" \
      --env_name "$DATASET_ENV_NAME" \
      --env_kwargs_json "$DATASET_ENV_KWARGS_JSON" \
      --max_turns "$MAX_TURNS"
  else
    echo "Dataset already exists in $DATA_DIR; skipping generation."
  fi
fi
ls -lh "$DATA_DIR/train.parquet" "$DATA_DIR/validation.parquet"

# Use non-isolated uv run so we use the venv created above.
tmp_run=/tmp/run_babyai_alg_sweep_fully_async_noisolated.sh
sed 's/uv run --isolated/uv run/' scripts/nersc/run_babyai_alg_sweep_fully_async.sh > "$tmp_run"
chmod +x "$tmp_run"
RUN_OVERRIDES=(
  "trainer.placement.policy_num_nodes=${POLICY_NUM_NODES}"
  "trainer.placement.ref_num_nodes=${REF_NUM_NODES}"
)
if [[ "${SWEEP_ALGOS:-}" == "ppo" ]]; then
  RUN_OVERRIDES+=("trainer.placement.critic_num_nodes=${CRITIC_NUM_NODES}")
fi
bash "$tmp_run" "${RUN_OVERRIDES[@]}"
EOS
)

# ------------------------------------------------------------
# Launch container
# ------------------------------------------------------------
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
  -e NUM_POLICY_GPUS="$NUM_POLICY_GPUS" \
  -e NUM_INFERENCE_ENGINES="$NUM_INFERENCE_ENGINES" \
  -e POLICY_NUM_NODES="$POLICY_NUM_NODES" \
  -e REF_NUM_NODES="$REF_NUM_NODES" \
  -e CRITIC_NUM_NODES="$CRITIC_NUM_NODES" \
  -e MAX_STALENESS_STEPS="$MAX_STALENESS_STEPS" \
  -e LOGGER="wandb" \
  -e INFERENCE_BACKEND="vllm" \
  -e HF_HOME="$HF_HOME_DIR" \
  -e WANDB_DIR="$WANDB_DIR" \
  -e CKPT_ROOT="/ckpts" \
  -e MAX_TURNS="$MAX_TURNS" \
  -e ENV_KWARGS_KV="$ENV_KWARGS_KV" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e ENV_NAME="$ENV_NAME" \
  -e SWEEP_ALGOS="$SELECTED_ALGO" \
  -e RUN_PREFIX="$RUN_PREFIX" \
  -e RUN_NAME_TIMESTAMP="$RUN_TIMESTAMP" \
  -e SEED="$SEED" \
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
  -e GENERATE_DATASET="$GENERATE_DATASET" \
  -e DATASET_DIFFICULTY="$DATASET_DIFFICULTY" \
  -e DATASET_ENV_NAME="$DATASET_ENV_NAME" \
  -e DATASET_ENV_KWARGS_JSON="$DATASET_ENV_KWARGS_JSON" \
  -e EXTRA_OVERRIDES_ALL="$EXTRA_OVERRIDES_ALL" \
  -e EXTRA_OVERRIDES_GRPO="$EXTRA_OVERRIDES_GRPO" \
  -e EXTRA_OVERRIDES_RLOO="$EXTRA_OVERRIDES_RLOO" \
  -e EXTRA_OVERRIDES_CISPO="$EXTRA_OVERRIDES_CISPO" \
  -e EXTRA_OVERRIDES_PPO="$EXTRA_OVERRIDES_PPO" \
  -e RAY_TMPDIR="$RAY_TMPDIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e UV_PROJECT_ENVIRONMENT="$UV_VENV_DIR" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  "$IMAGE" -lc "$CONTAINER_SCRIPT"
