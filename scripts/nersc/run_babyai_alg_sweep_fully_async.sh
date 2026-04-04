#!/bin/bash
set -euo pipefail
set -x

# Fully-async BabyAI sweep for multiple algorithms on the same environment.

: "${SWEEP_ALGOS:=grpo,rloo,cispo,ppo}"
: "${RUN_NAME_TIMESTAMP:=$(date -u +%Y%m%d_%H%M%S)}"

: "${DATA_DIR:="$HOME/data/babyai_text"}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL_NAME:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${ENV_NAME:=BabyAI-GoToLocal-v0}"
: "${MAX_TURNS:=64}"
: "${ENV_KWARGS_KV:=}"
: "${CKPT_ROOT:="$HOME/ckpts"}"

: "${NUM_POLICY_GPUS:=2}"
: "${NUM_INFERENCE_ENGINES:=2}"
: "${MAX_STALENESS_STEPS:=4}"

: "${SEED:=42}"
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

: "${KL_ESTIMATOR_TYPE:=k3}"
: "${KL_LOSS_COEF:=0.001}"

: "${GRPO_USE_KL_LOSS:=false}"
: "${GRPO_EPS_CLIP_LOW:=0.2}"
: "${GRPO_EPS_CLIP_HIGH:=0.2}"

: "${RLOO_USE_KL_LOSS:=false}"
: "${RLOO_USE_KL_IN_REWARD:=false}"

: "${CISPO_USE_KL_LOSS:=false}"
: "${CISPO_EPS_CLIP_LOW:=0}"
: "${CISPO_EPS_CLIP_HIGH:=5}"

: "${PPO_USE_KL_LOSS:=false}"
: "${PPO_GAMMA:=1.0}"
: "${PPO_LAMBDA:=1.0}"
: "${PPO_VALUE_CLIP:=0.2}"
: "${PPO_CRITIC_MINI_BATCH_SIZE:=$MINI_BATCH_SIZE}"

: "${EXTRA_OVERRIDES_ALL:=}"
: "${EXTRA_OVERRIDES_GRPO:=}"
: "${EXTRA_OVERRIDES_RLOO:=}"
: "${EXTRA_OVERRIDES_CISPO:=}"
: "${EXTRA_OVERRIDES_PPO:=}"

USER_OVERRIDES=("$@")

sanitize_suffix() {
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

append_extra_overrides() {
  local raw="$1"
  local -n out_arr="$2"
  local -a extra_arr=()
  if [[ -z "$raw" ]]; then
    return 0
  fi
  # shellcheck disable=SC2206
  extra_arr=($raw)
  out_arr+=("${extra_arr[@]}")
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

run_algo() {
  local algo="$1"
  shift
  local -a algo_overrides=("$@")

  local -a env_kwargs_overrides=()
  build_env_kwargs_overrides "$ENV_KWARGS_KV" env_kwargs_overrides

  local model_name_short="${MODEL_NAME##*/}"
  local model_tag
  model_tag="$(sanitize_suffix "$model_name_short")"
  local env_tag
  env_tag="$(sanitize_suffix "$ENV_NAME")"
  local run_name="${algo}_async_${env_tag}_${model_tag}_${RUN_NAME_TIMESTAMP}"
  local ckpt_path="$CKPT_ROOT"

  local -a common_overrides=(
    "data.train_data=['${DATA_DIR}/train.parquet']"
    "data.val_data=['${DATA_DIR}/validation.parquet']"
    "trainer.policy.model.path=${MODEL_NAME}"
    "trainer.strategy=fsdp2"
    "trainer.placement.colocate_all=false"
    "trainer.placement.policy_num_gpus_per_node=${NUM_POLICY_GPUS}"
    "trainer.placement.ref_num_gpus_per_node=${NUM_POLICY_GPUS}"
    "trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS}"
    "trainer.fully_async.num_parallel_generation_workers=$((MINI_BATCH_SIZE * (MAX_STALENESS_STEPS + 1)))"
    "generator.inference_engine.num_engines=${NUM_INFERENCE_ENGINES}"
    "generator.inference_engine.tensor_parallel_size=1"
    "generator.inference_engine.backend=${INFERENCE_BACKEND}"
    "generator.inference_engine.run_engines_locally=true"
    "generator.inference_engine.weight_sync_backend=nccl"
    "generator.inference_engine.async_engine=true"
    "generator.batched=false"
    "trainer.epochs=${EPOCHS}"
    "trainer.eval_batch_size=512"
    "trainer.eval_before_train=true"
    "trainer.eval_interval=5"
    "trainer.update_epochs_per_batch=1"
    "trainer.train_batch_size=${MINI_BATCH_SIZE}"
    "trainer.policy_mini_batch_size=${MINI_BATCH_SIZE}"
    "trainer.micro_forward_batch_size_per_gpu=${MICRO_FORWARD_BATCH_SIZE}"
    "trainer.micro_train_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE}"
    "trainer.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "generator.sampling_params.max_generate_length=${MAX_GENERATE_LENGTH}"
    "generator.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}"
    "generator.max_turns=${MAX_TURNS}"
    "generator.inference_engine.gpu_memory_utilization=0.8"
    "trainer.ckpt_interval=${CKPT_INTERVAL}"
    "trainer.policy.optimizer_config.lr=${LR}"
    "trainer.policy.optimizer_config.weight_decay=${WEIGHT_DECAY}"
    "trainer.policy.optimizer_config.max_grad_norm=${MAX_GRAD_NORM}"
    "environment.env_class=babyai_text"
    "environment.skyrl_gym.babyai_text.env_name=${ENV_NAME}"
    "environment.skyrl_gym.babyai_text.max_steps=${MAX_TURNS}"
    "trainer.logger=${LOGGER}"
    "trainer.project_name=babyai_text"
    "trainer.run_name=${run_name}"
    "trainer.ckpt_path=${ckpt_path}"
    "trainer.dump_eval_results=true"
    "trainer.seed=${SEED}"
    "trainer.algorithm.dynamic_sampling.type=null"
    "trainer.algorithm.use_kl_in_reward=false"
    "trainer.algorithm.kl_estimator_type=${KL_ESTIMATOR_TYPE}"
    "trainer.algorithm.kl_loss_coef=${KL_LOSS_COEF}"
    "trainer.algorithm.use_tis=false"
  )

  append_extra_overrides "${EXTRA_OVERRIDES_ALL}" common_overrides
  case "$algo" in
    grpo) append_extra_overrides "${EXTRA_OVERRIDES_GRPO}" common_overrides ;;
    rloo) append_extra_overrides "${EXTRA_OVERRIDES_RLOO}" common_overrides ;;
    cispo) append_extra_overrides "${EXTRA_OVERRIDES_CISPO}" common_overrides ;;
    ppo) append_extra_overrides "${EXTRA_OVERRIDES_PPO}" common_overrides ;;
  esac

  uv run --isolated --extra fsdp --extra babyai -m examples.train.fully_async.main_fully_async \
    "${common_overrides[@]}" \
    "${env_kwargs_overrides[@]}" \
    "${algo_overrides[@]}" \
    "${USER_OVERRIDES[@]}"
}

IFS=',' read -r -a algos <<< "$SWEEP_ALGOS"
for raw_algo in "${algos[@]}"; do
  algo="$(trim "$raw_algo")"
  [[ -z "$algo" ]] && continue

  case "$algo" in
    grpo)
      run_algo "grpo" \
        "trainer.algorithm.advantage_estimator=grpo" \
        "trainer.algorithm.policy_loss_type=regular" \
        "trainer.algorithm.loss_reduction=token_mean" \
        "trainer.algorithm.grpo_norm_by_std=true" \
        "trainer.algorithm.eps_clip_low=${GRPO_EPS_CLIP_LOW}" \
        "trainer.algorithm.eps_clip_high=${GRPO_EPS_CLIP_HIGH}" \
        "trainer.algorithm.use_kl_loss=${GRPO_USE_KL_LOSS}"
      ;;
    rloo)
      run_algo "rloo" \
        "trainer.algorithm.advantage_estimator=rloo" \
        "trainer.algorithm.policy_loss_type=regular" \
        "trainer.algorithm.loss_reduction=token_mean" \
        "trainer.algorithm.use_kl_in_reward=${RLOO_USE_KL_IN_REWARD}" \
        "trainer.algorithm.use_kl_loss=${RLOO_USE_KL_LOSS}"
      ;;
    cispo)
      run_algo "cispo" \
        "trainer.algorithm.advantage_estimator=grpo" \
        "trainer.algorithm.policy_loss_type=cispo" \
        "trainer.algorithm.loss_reduction=token_mean" \
        "trainer.algorithm.cispo.cispo_eps_clip_low=${CISPO_EPS_CLIP_LOW}" \
        "trainer.algorithm.cispo.cispo_eps_clip_high=${CISPO_EPS_CLIP_HIGH}" \
        "trainer.algorithm.use_kl_loss=${CISPO_USE_KL_LOSS}"
      ;;
    ppo)
      run_algo "ppo" \
        "trainer.algorithm.advantage_estimator=gae" \
        "trainer.algorithm.policy_loss_type=regular" \
        "trainer.algorithm.loss_reduction=token_mean" \
        "trainer.algorithm.gamma=${PPO_GAMMA}" \
        "trainer.algorithm.lambd=${PPO_LAMBDA}" \
        "trainer.algorithm.value_clip=${PPO_VALUE_CLIP}" \
        "trainer.algorithm.use_kl_loss=${PPO_USE_KL_LOSS}" \
        "trainer.critic.model.path=${MODEL_NAME}" \
        "trainer.critic_mini_batch_size=${PPO_CRITIC_MINI_BATCH_SIZE}" \
        "trainer.placement.critic_num_gpus_per_node=${NUM_POLICY_GPUS}"
      ;;
    *)
      echo "Unknown algorithm in SWEEP_ALGOS: '$algo'" >&2
      echo "Supported: grpo,rloo,cispo,ppo" >&2
      exit 1
      ;;
  esac
done
