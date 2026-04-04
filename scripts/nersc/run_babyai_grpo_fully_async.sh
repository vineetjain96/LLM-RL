#!/bin/bash
set -euo pipefail
set -x

# Fully-async GRPO on BabyAI-Text.
#
# Prerequisites:
#   uv run --extra babyai examples/train/babyai_text/babyai_text_dataset.py --output_dir $HOME/data/babyai_text
#
# Usage:
#   bash scripts/nersc/run_babyai_grpo_fully_async.sh

: "${DATA_DIR:="$HOME/data/babyai_text"}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL_NAME:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${CKPT_ROOT:="$HOME/ckpts"}"
: "${ALGO_NAME:=grpo}"
: "${RUN_NAME_TIMESTAMP:=}"

: "${DATA_KEEP_IN_MEMORY:=false}"
: "${DATASET_NUM_WORKERS:=1}"
: "${DATALOADER_NUM_WORKERS:=0}"
: "${SAVE_OPTIMIZER_STATE_IN_CKPT:=false}"
: "${SAVE_DATALOADER_STATE_IN_CKPT:=false}"

: "${ENV_NAME:=BabyAI-GoToLocal-v0}"
: "${MAX_TURNS:=64}"
: "${ENV_KWARGS_KV:=}"
: "${SWEEP_PARAM:=}"
: "${SWEEP_VALUES:=}"
: "${SWEEP_VALUE:=}"

: "${NUM_POLICY_GPUS:=2}"
: "${NUM_INFERENCE_ENGINES:=2}"

: "${EPOCHS:=40}"
: "${MINI_BATCH_SIZE:=128}"
: "${MICRO_FORWARD_BATCH_SIZE:=32}"
: "${MICRO_TRAIN_BATCH_SIZE:=32}"
: "${MAX_PROMPT_LENGTH:=512}"
: "${MAX_GENERATE_LENGTH:=2048}"
: "${N_SAMPLES_PER_PROMPT:=5}"
: "${LR:=1.0e-6}"
: "${WEIGHT_DECAY:=1e-2}"
: "${MAX_GRAD_NORM:=1.0}"
: "${CKPT_INTERVAL:=10}"

: "${MAX_STALENESS_STEPS:=4}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=$((MINI_BATCH_SIZE * (MAX_STALENESS_STEPS + 1)))}"

: "${GRPO_USE_KL_LOSS:=false}"
: "${GRPO_EPS_CLIP_LOW:=0.2}"
: "${GRPO_EPS_CLIP_HIGH:=0.2}"
: "${KL_ESTIMATOR_TYPE:=k3}"
: "${KL_LOSS_COEF:=0.001}"
: "${USE_TIS:=false}"

USER_OVERRIDES=("$@")

sanitize_suffix() {
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

build_env_kwargs_overrides() {
  local kv_string="$1"
  local -n out_overrides="$2"
  out_overrides=()

  if [[ -z "$kv_string" ]]; then
    return 0
  fi

  IFS=',' read -r -a kv_pairs <<< "$kv_string"
  for kv_pair in "${kv_pairs[@]}"; do
    [[ -z "$kv_pair" ]] && continue
    if [[ "$kv_pair" != *=* ]]; then
      echo "Invalid ENV_KWARGS_KV entry: '$kv_pair'. Expected key=value." >&2
      exit 1
    fi

    local key="${kv_pair%%=*}"
    local value="${kv_pair#*=}"
    if [[ -z "$key" || -z "$value" ]]; then
      echo "Invalid ENV_KWARGS_KV entry: '$kv_pair'. Expected non-empty key and value." >&2
      exit 1
    fi

    out_overrides+=("+environment.skyrl_gym.babyai_text.env_kwargs.${key}=${value}")
  done
}

run_training() {
  local sweep_param="$1"
  local sweep_value="$2"
  local run_suffix="$3"
  local -a env_kwargs_overrides=()

  build_env_kwargs_overrides "$ENV_KWARGS_KV" env_kwargs_overrides
  if [[ -n "$sweep_param" ]]; then
    env_kwargs_overrides+=("+environment.skyrl_gym.babyai_text.env_kwargs.${sweep_param}=${sweep_value}")
  fi

  local model_name_short="${MODEL_NAME##*/}"
  local model_tag
  model_tag="$(sanitize_suffix "$model_name_short")"
  local env_tag
  env_tag="$(sanitize_suffix "$ENV_NAME")"
  local run_ts="$RUN_NAME_TIMESTAMP"
  if [[ -z "$run_ts" ]]; then
    run_ts="$(date -u +%Y%m%d_%H%M%S)"
  fi

  local run_name="${ALGO_NAME}_async_${env_tag}_${model_tag}_${run_ts}"
  local ckpt_path="$CKPT_ROOT/${ALGO_NAME}_async_${env_tag}_${model_tag}_${run_ts}_ckpt"
  if [[ -n "$run_suffix" ]]; then
    run_name="${run_name}_${run_suffix}"
    ckpt_path="${ckpt_path}_${run_suffix}"
  fi

  uv run --isolated --extra fsdp --extra babyai -m examples.train.fully_async.main_fully_async \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet']" \
    data.keep_in_memory="$DATA_KEEP_IN_MEMORY" \
    data.dataset_num_workers="$DATASET_NUM_WORKERS" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.policy_loss_type=regular \
    trainer.algorithm.loss_reduction=token_mean \
    trainer.algorithm.grpo_norm_by_std=true \
    trainer.algorithm.eps_clip_low="$GRPO_EPS_CLIP_LOW" \
    trainer.algorithm.eps_clip_high="$GRPO_EPS_CLIP_HIGH" \
    trainer.algorithm.use_kl_loss="$GRPO_USE_KL_LOSS" \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.algorithm.kl_estimator_type="$KL_ESTIMATOR_TYPE" \
    trainer.algorithm.kl_loss_coef="$KL_LOSS_COEF" \
    trainer.algorithm.use_tis="$USE_TIS" \
    trainer.algorithm.dynamic_sampling.type=null \
    trainer.policy.model.path="$MODEL_NAME" \
    trainer.placement.colocate_all=false \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node="$NUM_POLICY_GPUS" \
    trainer.placement.ref_num_gpus_per_node="$NUM_POLICY_GPUS" \
    trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
    trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
    generator.inference_engine.num_engines="$NUM_INFERENCE_ENGINES" \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs="$EPOCHS" \
    trainer.eval_batch_size=512 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size="$MINI_BATCH_SIZE" \
    trainer.policy_mini_batch_size="$MINI_BATCH_SIZE" \
    trainer.micro_forward_batch_size_per_gpu="$MICRO_FORWARD_BATCH_SIZE" \
    trainer.micro_train_batch_size_per_gpu="$MICRO_TRAIN_BATCH_SIZE" \
    trainer.ckpt_interval="$CKPT_INTERVAL" \
    trainer.save_optimizer_state_in_ckpt="$SAVE_OPTIMIZER_STATE_IN_CKPT" \
    trainer.save_dataloader_state_in_ckpt="$SAVE_DATALOADER_STATE_IN_CKPT" \
    trainer.max_prompt_length="$MAX_PROMPT_LENGTH" \
    generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
    trainer.policy.optimizer_config.lr="$LR" \
    trainer.policy.optimizer_config.weight_decay="$WEIGHT_DECAY" \
    trainer.policy.optimizer_config.max_grad_norm="$MAX_GRAD_NORM" \
    generator.inference_engine.backend="$INFERENCE_BACKEND" \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.async_engine=true \
    generator.batched=false \
    generator.max_turns="$MAX_TURNS" \
    environment.env_class=babyai_text \
    environment.skyrl_gym.babyai_text.env_name="$ENV_NAME" \
    environment.skyrl_gym.babyai_text.max_steps="$MAX_TURNS" \
    "${env_kwargs_overrides[@]}" \
    generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    trainer.logger="$LOGGER" \
    trainer.project_name=babyai_text \
    trainer.run_name="$run_name" \
    trainer.ckpt_path="$ckpt_path" \
    trainer.dump_eval_results=true \
    "${USER_OVERRIDES[@]}"
}

if [[ -n "$SWEEP_VALUE" ]]; then
  if [[ -z "$SWEEP_PARAM" ]]; then
    echo "SWEEP_VALUE is set but SWEEP_PARAM is empty." >&2
    exit 1
  fi
  run_training "$SWEEP_PARAM" "$SWEEP_VALUE" "$(sanitize_suffix "${SWEEP_PARAM}_${SWEEP_VALUE}")"
elif [[ -n "$SWEEP_PARAM" || -n "$SWEEP_VALUES" ]]; then
  if [[ -z "$SWEEP_PARAM" || -z "$SWEEP_VALUES" ]]; then
    echo "Both SWEEP_PARAM and SWEEP_VALUES must be set for sweep runs." >&2
    exit 1
  fi
  IFS=',' read -r -a sweep_values <<< "$SWEEP_VALUES"
  for value in "${sweep_values[@]}"; do
    [[ -z "$value" ]] && continue
    run_training "$SWEEP_PARAM" "$value" "$(sanitize_suffix "${SWEEP_PARAM}_${value}")"
  done
else
  run_training "" "" ""
fi
