#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH -J skyrl_babyai_text
#SBATCH -o skyrl_%j.out
#SBATCH -e skyrl_%j.err
#SBATCH -A "m5017"
#SBATCH -C "gpu&hbm80g"

set -euo pipefail

IMAGE="novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8"
WORKROOT="$HOME/LLM-RL"

# Where the BabyAI parquet should live (on SCRATCH)
BABYAI_OUT_DIR="$SCRATCH/data/babyai_text"

# Scratch-backed run dirs
RUN_DIR="$SCRATCH/skyrl/${SLURM_JOB_ID}"
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

# Critical: ensure SCRATCH exists in-container so run_babyai_text.sh doesn't fall back to /data/...
echo "SCRATCH=$SCRATCH"
echo "DATA_DIR=$DATA_DIR"

mkdir -p "$DATA_DIR" "$HF_HOME" "$RAY_TMPDIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT" "$WANDB_DIR" "$HOME/ckpts"

cd /work/skyrl-train
ls -ld /work/skyrl-train /work/skyrl-gym

# Create/reuse uv venv + install repo deps
uv --version
uv sync

# Extra deps you requested
uv pip install gymnasium minigrid

# Generate dataset exactly where training will read it
uv run --extra babyai examples/babyai_text/babyai_text_dataset.py --output_dir "$DATA_DIR"
ls -lh "$DATA_DIR/train.parquet" "$DATA_DIR/validation.parquet"

# OPTIONAL: run a copy of the launcher without --isolated so it uses this venv
tmp_run=/tmp/run_babyai_text_noisolated.sh
sed 's/uv run --isolated/uv run/' examples/babyai_text/run_babyai_text.sh > "$tmp_run"
chmod +x "$tmp_run"

# Run training (NUM_GPUS/LOGGER/INFERENCE_BACKEND come from env)
bash "$tmp_run"
EOS
)

srun -n 1 podman-hpc run --rm \
  --gpu \
  --shm-size=8g \
  --userns=keep-id:uid=1000,gid=100 \
  -v "$WORKROOT:/work:rw" \
  -v "$SCRATCH:$SCRATCH:rw" \
  -v "$CKPT_DIR:$HOME/ckpts:rw" \
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
  -e RAY_TMPDIR="$RAY_TMPDIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e UV_PROJECT_ENVIRONMENT="$UV_VENV_DIR" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  "$IMAGE" bash -lc "$CONTAINER_SCRIPT"

