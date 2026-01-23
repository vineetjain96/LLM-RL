#!/bin/bash
set -x

# BabyAI-Text training with GRPO for Qwen2.5-1.5B-Instruct
#
# This script trains a language model to complete BabyAI grid-world navigation
# tasks using text-based observations and actions.
#
# Prerequisites:
#   1. Generate the dataset:
#      uv run examples/babyai_text/babyai_text_dataset.py --output_dir $HOME/data/babyai_text
#
#   2. (Optional) Install minigrid for full environment support:
#      uv pip install minigrid gymnasium
#
#   3. Set your WANDB API key (optional, for logging):
#      export WANDB_API_KEY=<your_key_here>
#
#   4. Run training:
#      bash examples/babyai_text/run_babyai_text.sh
#
# You can override defaults with environment variables:
#   NUM_GPUS=8 INFERENCE_BACKEND=sglang bash examples/babyai_text/run_babyai_text.sh

# Configuration with defaults
: "${DATA_DIR:="$HOME/data/babyai_text"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}"  # Change to "console" for stdout logging
: "${INFERENCE_BACKEND:=vllm}"  # Or "sglang"

# BabyAI-specific settings
: "${ENV_NAME:=BabyAI-GoToLocal-v0}"
: "${MAX_TURNS:=64}"

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=128 \
  trainer.micro_forward_batch_size_per_gpu=32 \
  trainer.micro_train_batch_size_per_gpu=32 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=babyai_text \
  environment.skyrl_gym.babyai_text.env_name=$ENV_NAME \
  environment.skyrl_gym.babyai_text.max_steps=$MAX_TURNS \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="babyai_text" \
  trainer.run_name="babyai_text_${ENV_NAME}_grpo" \
  trainer.ckpt_path="$HOME/ckpts/babyai_text_ckpt" \
  "$@"
