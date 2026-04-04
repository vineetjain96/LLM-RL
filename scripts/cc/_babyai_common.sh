#!/bin/bash

CC_BABYAI_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CC_BABYAI_VENV_ACTIVATE="/home/v/vinjain/scratch/.virtualenvs/skyrl/bin/activate"
CC_BABYAI_MODULES=(cuda/12.6 cudnn gcc httpproxy)

cc_babyai_is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

cc_babyai_sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

cc_babyai_ensure_module_command() {
  if type module >/dev/null 2>&1; then
    return 0
  fi

  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
    return 0
  fi

  if [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck source=/dev/null
    source /usr/share/Modules/init/bash
    return 0
  fi

  echo "Could not find an Environment Modules init script." >&2
  exit 1
}

cc_babyai_setup_env() {
  cc_babyai_ensure_module_command
  module load "${CC_BABYAI_MODULES[@]}"

  if [[ ! -f "$CC_BABYAI_VENV_ACTIVATE" ]]; then
    echo "Virtualenv activation script not found: $CC_BABYAI_VENV_ACTIVATE" >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$CC_BABYAI_VENV_ACTIVATE"
  cd "$CC_BABYAI_REPO_ROOT"
}

cc_babyai_prepare_layout() {
  : "${EXPERIMENT_ROOT:="$HOME/scratch/babyai"}"
  : "${ENV_KWARGS_JSON:="{}"}"
  : "${DATASET_ENV_NAME:="$ENV_NAME"}"
  : "${DATASET_ENV_KWARGS_JSON:="$ENV_KWARGS_JSON"}"
  : "${DATASET_DIFFICULTY:=easy}"
  : "${RUN_NAME_TIMESTAMP:="$(date -u +%Y%m%d_%H%M%S)"}"
  : "${RUN_NAME_SUFFIX:=}"

  local env_tag
  local model_tag
  local dataset_env_tag
  local dataset_kwargs_hash
  local dataset_tag
  local run_name

  env_tag="$(cc_babyai_sanitize_name "$ENV_NAME")"
  model_tag="$(cc_babyai_sanitize_name "${MODEL_NAME##*/}")"
  dataset_env_tag="$(cc_babyai_sanitize_name "$DATASET_ENV_NAME")"

  if [[ -n "$DATASET_ENV_KWARGS_JSON" && "$DATASET_ENV_KWARGS_JSON" != "{}" && "$DATASET_ENV_KWARGS_JSON" != "null" ]]; then
    dataset_kwargs_hash="kw$(printf '%s' "$DATASET_ENV_KWARGS_JSON" | sha1sum | awk '{print substr($1, 1, 8)}')"
  else
    dataset_kwargs_hash="kwnone"
  fi

  dataset_tag="${dataset_env_tag}_${DATASET_DIFFICULTY}_turns${MAX_TURNS}_${dataset_kwargs_hash}"
  : "${DATA_DIR:="$EXPERIMENT_ROOT/data/$dataset_tag"}"

  run_name="${ALGO_NAME}_${env_tag}_${model_tag}_${RUN_NAME_TIMESTAMP}"
  if [[ -n "$RUN_NAME_SUFFIX" ]]; then
    run_name="${run_name}_$(cc_babyai_sanitize_name "$RUN_NAME_SUFFIX")"
  fi

  : "${RUN_NAME:="$run_name"}"
  : "${RUN_DIR:="$EXPERIMENT_ROOT/runs/$RUN_NAME"}"
  : "${CKPT_PATH:="$RUN_DIR/ckpts"}"
  : "${LOG_PATH:="$RUN_DIR/logs"}"
  : "${EXPORT_PATH:="$RUN_DIR/exports"}"
  : "${RAY_TMP_ROOT:=/tmp}"
  HF_HOME="$EXPERIMENT_ROOT/hf_home"
  WANDB_DIR="$RUN_DIR/wandb"
  : "${RAY_TMPDIR:="$RAY_TMP_ROOT/ray_${SLURM_JOB_ID:-$$}_${RUN_NAME_TIMESTAMP}"}"
  HF_HUB_CACHE="$HF_HOME/hub"
  TRANSFORMERS_CACHE="$HF_HUB_CACHE"
  : "${TOKENIZERS_PARALLELISM:=false}"

  mkdir -p \
    "$EXPERIMENT_ROOT" \
    "$DATA_DIR" \
    "$RUN_DIR" \
    "$CKPT_PATH" \
    "$LOG_PATH" \
    "$EXPORT_PATH" \
    "$WANDB_DIR" \
    "$RAY_TMPDIR" \
    "$HF_HOME" \
    "$HF_HUB_CACHE"

  export HF_HOME
  export HF_HUB_CACHE
  export TRANSFORMERS_CACHE
  export WANDB_DIR
  export RAY_TMPDIR
  export TOKENIZERS_PARALLELISM
}

cc_babyai_build_env_kwargs_overrides() {
  local env_kwargs_json="$1"
  local -n out_overrides="$2"
  out_overrides=()

  if [[ -z "$env_kwargs_json" || "$env_kwargs_json" == "{}" || "$env_kwargs_json" == "null" ]]; then
    return 0
  fi

  mapfile -t out_overrides < <(
    python - "$env_kwargs_json" <<'PY'
import json
import sys

raw = sys.argv[1]
data = json.loads(raw)
if not isinstance(data, dict):
    raise SystemExit("ENV_KWARGS_JSON must decode to a JSON object.")

for key, value in data.items():
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif value is None:
        rendered = "null"
    elif isinstance(value, (int, float)):
        rendered = repr(value)
    else:
        rendered = json.dumps(str(value))
    print(f"environment.skyrl_gym.babyai_text.env_kwargs.{key}={rendered}")
PY
  )
}

cc_babyai_ensure_dataset() {
  : "${REGENERATE_DATASET:=false}"
  : "${DATASET_TRAIN_SIZE:=12800}"
  : "${DATASET_VAL_SIZE:=256}"

  if [[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/validation.parquet" ]] && ! cc_babyai_is_truthy "$REGENERATE_DATASET"; then
    return 0
  fi

  local -a dataset_cmd=(
    python
    examples/train/babyai_text/babyai_text_dataset.py
    --output_dir "$DATA_DIR"
    --difficulty "$DATASET_DIFFICULTY"
    --env_kwargs_json "$DATASET_ENV_KWARGS_JSON"
    --train_size "$DATASET_TRAIN_SIZE"
    --val_size "$DATASET_VAL_SIZE"
    --max_turns "$MAX_TURNS"
  )

  if [[ -n "$DATASET_ENV_NAME" ]]; then
    dataset_cmd+=(--env_name "$DATASET_ENV_NAME")
  fi

  "${dataset_cmd[@]}"
}

cc_babyai_build_common_overrides() {
  local -n out_overrides="$1"
  local -a env_kwargs_overrides=()

  : "${NUM_GPUS:=4}"
  : "${LOGGER:="[wandb,console]"}"
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
  : "${GPU_MEMORY_UTILIZATION:=0.8}"
  : "${DATA_KEEP_IN_MEMORY:=false}"
  : "${DATASET_NUM_WORKERS:=1}"
  : "${DATALOADER_NUM_WORKERS:=0}"

  cc_babyai_build_env_kwargs_overrides "$ENV_KWARGS_JSON" env_kwargs_overrides

  out_overrides=(
    "data.train_data=['$DATA_DIR/train.parquet']"
    "data.val_data=['$DATA_DIR/validation.parquet']"
    "data.keep_in_memory=$DATA_KEEP_IN_MEMORY"
    "data.dataset_num_workers=$DATASET_NUM_WORKERS"
    "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
    "trainer.policy.model.path=$MODEL_NAME"
    "trainer.placement.colocate_all=true"
    "trainer.strategy=fsdp2"
    "trainer.placement.policy_num_gpus_per_node=$NUM_GPUS"
    "trainer.placement.ref_num_gpus_per_node=$NUM_GPUS"
    "trainer.placement.critic_num_gpus_per_node=$NUM_GPUS"
    "generator.inference_engine.num_engines=$NUM_GPUS"
    "generator.inference_engine.tensor_parallel_size=1"
    "trainer.epochs=$EPOCHS"
    "trainer.eval_batch_size=$EVAL_BATCH_SIZE"
    "trainer.eval_before_train=$EVAL_BEFORE_TRAIN"
    "trainer.eval_interval=$EVAL_INTERVAL"
    "trainer.update_epochs_per_batch=$UPDATE_EPOCHS_PER_BATCH"
    "trainer.train_batch_size=$TRAIN_BATCH_SIZE"
    "trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE"
    "trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE"
    "trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE"
    "trainer.ckpt_interval=$CKPT_INTERVAL"
    "trainer.max_ckpts_to_keep=$MAX_CKPTS_TO_KEEP"
    "trainer.max_prompt_length=$MAX_PROMPT_LENGTH"
    "generator.max_input_length=$MAX_INPUT_LENGTH"
    "generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH"
    "trainer.policy.optimizer_config.lr=$LR"
    "trainer.policy.optimizer_config.weight_decay=$WEIGHT_DECAY"
    "trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM"
    "trainer.algorithm.use_kl_in_reward=false"
    "trainer.algorithm.use_kl_loss=$USE_KL_LOSS"
    "generator.inference_engine.backend=$INFERENCE_BACKEND"
    "generator.inference_engine.run_engines_locally=true"
    "generator.inference_engine.weight_sync_backend=nccl"
    "generator.inference_engine.async_engine=true"
    "generator.batched=false"
    "generator.use_conversation_multi_turn=true"
    "generator.max_turns=$MAX_TURNS"
    "environment.env_class=babyai_text"
    "environment.skyrl_gym.babyai_text.env_name=$ENV_NAME"
    "environment.skyrl_gym.babyai_text.max_steps=$MAX_TURNS"
    "generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT"
    "generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    "trainer.logger=$LOGGER"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.run_name=$RUN_NAME"
    "trainer.resume_mode=none"
    "trainer.log_path=$LOG_PATH"
    "trainer.ckpt_path=$CKPT_PATH"
    "trainer.export_path=$EXPORT_PATH"
    "trainer.dump_eval_results=true"
    "trainer.seed=$SEED"
  )

  out_overrides+=("${env_kwargs_overrides[@]}")
}

cc_babyai_print_run_summary() {
  cat <<EOF
Experiment root: $EXPERIMENT_ROOT
Dataset dir:     $DATA_DIR
Run dir:         $RUN_DIR
Checkpoint dir:  $CKPT_PATH
Ray temp dir:    $RAY_TMPDIR
Run name:        $RUN_NAME
Logger:          $LOGGER
EOF
}

cc_babyai_validate_overrides() {
  local -a overrides=("$@")

  python - "${overrides[@]}" <<'PY'
import sys

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import validate_cfg

cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
validate_cfg(cfg)

print(f"Config validation passed for run_name={cfg.trainer.run_name}")
print(f"  ckpt_path={cfg.trainer.ckpt_path}")
print(f"  env_name={cfg.environment.skyrl_gym.babyai_text.env_name}")
print(f"  estimator={cfg.trainer.algorithm.advantage_estimator}")
PY
}

cc_babyai_run_main_base() {
  local -a overrides=("$@")
  local -a cmd=(python -m skyrl.train.entrypoints.main_base "${overrides[@]}")

  cc_babyai_print_run_summary

  if cc_babyai_is_truthy "${VALIDATE_ONLY:-false}"; then
    cc_babyai_validate_overrides "${overrides[@]}"
    return 0
  fi

  if cc_babyai_is_truthy "${DRY_RUN:-false}"; then
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}
