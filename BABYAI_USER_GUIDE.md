# BabyAI Experiment Guide

This is the current guide for running BabyAI experiments in this repo. It focuses on the knobs that actually change experiment behavior, not every config field in SkyRL.

## What matters most

If you only remember five things, make them these:

1. `ENV_NAME` chooses the task family.
2. `ENV_KWARGS_JSON` or `ENV_KWARGS_KV` controls difficulty inside that family.
3. `MAX_TURNS` changes both episode horizon and the success reward shaping.
4. The launcher script is your algorithm choice.
5. Batch size, `N_SAMPLES_PER_PROMPT`, and async staleness are the main throughput/stability tradeoffs.

## Mental model

The BabyAI pipeline has three layers:

1. `examples/train/babyai_text/babyai_text_dataset.py`
   Creates parquet prompts and stores `env_name`, `env_kwargs`, `max_turns`, and seeds in `extra_info`.

2. Launcher scripts under `scripts/cc/` and `scripts/nersc/`
   Choose the algorithm, hardware layout, and the most important CLI overrides.

3. `skyrl-gym/skyrl_gym/envs/babyai_text/env.py`
   Builds the actual Gym environment with `gym.make(env_name, **env_kwargs)`, runs the multi-turn loop, and computes rewards.

Two consequences are easy to miss:

- `env_kwargs` are passed straight through to the underlying BabyAI or MiniGrid environment.
- `MAX_TURNS` is not just a timeout. Success reward is scaled by efficiency, so changing `MAX_TURNS` also changes the reward landscape.

## Which launcher to start from

| Goal | Script | When to use it |
|---|---|---|
| Clean baseline | `scripts/cc/run_babyai_grpo_sync.sh` | Best first run. Synchronous, simple, and easy to compare across environments. |
| Better step-level credit assignment | `scripts/cc/run_babyai_actor_critic_sync.sh` | Use when long trajectories make plain GRPO too blunt. This is the `state_action_td` path. |
| Maximum throughput | `scripts/nersc/run_babyai_grpo_fully_async.sh` | Use after the sync baseline works and you want more samples per wall-clock hour. |
| Compare async algorithms | `scripts/nersc/run_babyai_alg_sweep_fully_async.sh` | Useful for side-by-side async sweeps of `grpo`, `rloo`, `cispo`, and `ppo`. |

The `scripts/cc/` launchers are the easiest starting point because they:

- load modules and activate the project environment for you
- auto-create a matching dataset unless you ask for `VALIDATE_ONLY` or `DRY_RUN`
- support `VALIDATE_ONLY=true` to check config correctness
- support `DRY_RUN=true` to print the final command

The `scripts/nersc/` launchers assume you already have a dataset.

## The high-leverage knobs

### 1. Task choice

| Knob | Why it matters | Good first use |
|---|---|---|
| `ENV_NAME` | Changes the task family entirely. | Start with `BabyAI-GoToLocal-v0` for debugging and quick comparisons. |
| `ENV_KWARGS_JSON` in `scripts/cc/` | Sets env kwargs as a JSON object. | `ENV_KWARGS_JSON='{"room_size": 8, "num_dists": 4}'` |
| `ENV_KWARGS_KV` in `scripts/nersc/` | Sets env kwargs as comma-separated `key=value`. | `ENV_KWARGS_KV=room_size=8,num_dists=4` |
| `MAX_TURNS` | Sets the episode cap and changes the efficiency bonus on success. | Keep it just above the path length you expect, not much larger. |

These env kwargs are the ones worth sweeping first:

| Env kwarg | Effect |
|---|---|
| `room_size` | Longer paths and larger search space |
| `num_dists` | More distractors and harder grounding |
| `num_rows`, `num_cols` | More multi-room depth and exploration |
| `num_objs`, `objs_per_room` | More clutter in manipulation tasks |
| `num_doors` | Longer ordered subtask chains |

Practical family choices:

- `BabyAI-GoToLocal-v0` or `BabyAI-GoToObj-v0`: easiest place to debug training.
- `BabyAI-GoToObjMaze-v0` or `BabyAI-KeyCorridor-v0`: better for horizon and exploration stress.
- `BabyAI-GoToSeq-v0`, `BabyAI-PutNext-v0`, `BabyAI-OpenDoorsOrder-v0`: better when you care about sequential reasoning and step-level credit.

### 2. Algorithm choice

| Choice | What changes | When to prefer it |
|---|---|---|
| Sync GRPO | Group-relative policy updates, no critic. | Default baseline. Best for environment and model sweeps. |
| `state_action_td` actor-critic | Step-level TD targets plus a critic, with sequence-level policy updates on each action span. | Better when the task is multi-step enough that final-outcome credit assignment is too weak. |
| Fully async GRPO | More off-policy, more throughput. | When wall-clock throughput matters more than perfect on-policy freshness. |
| Async `rloo` / `cispo` / `ppo` | Alternative baselines in the same async scaffold. | Only after a baseline is already stable. |

Important constraints for `state_action_td`:

- It is only supported for `babyai_text`.
- It is only supported in synchronous FSDP or FSDP2 training.
- It requires `generator.step_wise_trajectories=true`.
- It intentionally uses `policy_loss_type=gspo` and `loss_reduction=sequence_mean`.

Treat those as invariants, not tuning knobs.

### 3. Throughput and stability

| Knob | Why it matters | Rule of thumb |
|---|---|---|
| `N_SAMPLES_PER_PROMPT` | More samples improve group statistics and exploration, but generation cost scales with it. | `4-8` is a sensible starting range. |
| `TRAIN_BATCH_SIZE` | Effective amount of data per update. | Increase only if memory and throughput are healthy. |
| `POLICY_MINI_BATCH_SIZE` | Controls optimizer noise and memory pressure. | Smaller is safer; larger is faster if stable. |
| `CRITIC_MINI_BATCH_SIZE` | Critic-side version of the same tradeoff. | Only relevant when you have a critic. |
| `MICRO_FORWARD_BATCH_SIZE`, `MICRO_TRAIN_BATCH_SIZE` | Memory escape hatches. | First thing to lower on OOM. |
| `LR` | Main policy step size. | Lower this before touching exotic knobs if training gets noisy. |
| `CRITIC_LR` | Critic learning speed. | Usually matters a lot for actor-critic and PPO. |
| `WEIGHT_DECAY`, `MAX_GRAD_NORM` | Regularization and update clipping. | Keep defaults unless you have a clear reason. |
| `MAX_GENERATE_LENGTH` | Hard cap on per-turn generation length. | BabyAI actions are short; `128-256` is usually enough. |
| `MAX_INPUT_LENGTH` | Conversation context budget. | Raise only if later turns are being truncated. |

### 4. Async-only knobs

| Knob | Why it matters | Rule of thumb |
|---|---|---|
| `MAX_STALENESS_STEPS` | Larger values improve throughput but make data more off-policy. | Start at `4` and move cautiously. |
| `NUM_PARALLEL_GENERATION_WORKERS` | Async concurrency. | Usually derive it from batch size and staleness, as the script already does. |
| `USE_TIS` | Importance-sampling correction for async mismatch. | Turn it on if higher staleness starts to hurt stability. |

### 5. Knobs that are mostly plumbing

You usually do not need to touch these for BabyAI experiments:

- `trainer.strategy=fsdp2`
- `generator.batched=false`
- `generator.use_conversation_multi_turn=true`
- `environment.env_class=babyai_text`
- `generator.step_wise_trajectories`
  This should stay `false` for GRPO and `true` for `state_action_td`.

## Dataset alignment

Dataset alignment matters more than most people expect.

In the `scripts/cc/` launchers, the defaults are already aligned:

- `DATASET_ENV_NAME` defaults to `ENV_NAME`
- `DATASET_ENV_KWARGS_JSON` defaults to `ENV_KWARGS_JSON`
- `MAX_TURNS` is used for both dataset generation and training

That means the dataset normally matches the training environment exactly.

`DATASET_DIFFICULTY` is only important if you intentionally want a mixed dataset instead of one fixed `ENV_NAME`.

For the `scripts/nersc/` launchers, generate the dataset yourself first. A minimal example is:

```bash
module load cuda/12.6 cudnn gcc arrow/18.1.0 httpproxy
source /home/v/vinjain/scratch/.virtualenvs/skyrl/bin/activate

python examples/train/babyai_text/babyai_text_dataset.py \
  --output_dir "$HOME/data/babyai_text" \
  --env_name BabyAI-GoToLocal-v0 \
  --env_kwargs_json '{"room_size": 8, "num_dists": 4}' \
  --max_turns 32
```

## Recommended starting recipes

### Easy sync baseline

```bash
ENV_NAME=BabyAI-GoToLocal-v0 \
ENV_KWARGS_JSON='{"room_size": 8, "num_dists": 4}' \
MAX_TURNS=32 \
bash scripts/cc/run_babyai_grpo_sync.sh
```

### Longer-horizon actor-critic run

```bash
ENV_NAME=BabyAI-GoToSeq-v0 \
ENV_KWARGS_JSON='{"room_size": 6, "num_rows": 2, "num_cols": 2, "num_dists": 4}' \
MAX_TURNS=48 \
bash scripts/cc/run_babyai_actor_critic_sync.sh \
  trainer.algorithm.gamma=0.99 \
  trainer.algorithm.lambd=0.95
```

### Async throughput run

```bash
ENV_NAME=BabyAI-GoToLocal-v0 \
ENV_KWARGS_KV=room_size=8,num_dists=4 \
MAX_TURNS=32 \
MAX_STALENESS_STEPS=4 \
USE_TIS=true \
bash scripts/nersc/run_babyai_grpo_fully_async.sh
```

All of these launchers accept extra CLI overrides at the end, so you can always inject lower-level config directly.

## What to watch in metrics

These are the first metrics to watch when deciding what to tune next:

- `environment/success_rate`: the cleanest task-level metric
- `environment/avg_steps`: whether the policy is solving efficiently or just barely
- `loss/avg_final_rewards`: overall reward trend
- `generate/avg_num_tokens`: whether generation is wasting tokens per step
- `environment/invalid_action_rate`: especially important for `state_action_td`

If `invalid_action_rate` is high, that usually points to action-formatting or prompting issues before it points to an RL algorithm issue.

## A simple progression that works well

1. Start with sync GRPO on `BabyAI-GoToLocal-v0`.
2. Sweep one environment knob at a time, usually `room_size` first and then `num_dists`.
3. Move to more structured environments like `GoToObjMaze`, `KeyCorridor`, or `GoToSeq`.
4. Switch to `state_action_td` only when longer trajectories make step-level credit assignment important.
5. Move to fully async only after the synchronous version is already giving sensible curves.

That order usually makes failures much easier to interpret.
