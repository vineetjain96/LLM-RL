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

## Episode end cases

These edge cases are easy to reason about incorrectly when you start changing `MAX_TURNS`, `MAX_INPUT_LENGTH`, and the non-stop filtering flags.

Important note:

- Success reward is efficiency-shaped, not always exactly `1.0`.
- With `MAX_TURNS=8`, success at step 6 gives reward `0.625`, not `1.0`.

| Case | GRPO (`step_wise_trajectories=false`) | `state_action_td` (`step_wise_trajectories=true`) |
|---|---|---|
| Task solved at step 6/8 and final response ends normally | Episode ends immediately. Reward is the positive success reward for that final step. `stop_reason` is usually `"stop"`. Whole trajectory is kept. | Steps 1-6 are emitted. Step 6 is the last step, gets the positive success reward, and is treated as terminal for TD targets. |
| Task solved at step 6/8 but final response was truncated and still happened to contain a winning action | Environment may still mark success, but with `zero_reward_on_non_stop=true` the trajectory reward is zeroed because `stop_reason != "stop"`. In the current GRPO launcher, the trajectory is still kept because `apply_overlong_filtering=false`. | The winning step is still emitted, but in the current actor-critic launcher its reward is zeroed and its loss mask is zeroed because `zero_reward_on_non_stop=true` and `apply_overlong_filtering=true`. Earlier valid steps remain. |
| Conversation exceeds `MAX_INPUT_LENGTH` before the next turn starts, e.g. before step 6 | The generator stops before producing step 6. No further env step happens. The partial trajectory up through step 5 is returned with `stop_reason="length"`. In the current GRPO launcher, reward is zeroed but the trajectory is still trained on. | The generator stops before producing step 6, so only steps 1-5 are emitted. There is no dummy step 6. Step 5 becomes the last emitted step and is treated as terminal from the learner's point of view. |

For `state_action_td`, that last row is the subtle one: hitting `MAX_INPUT_LENGTH` does not create a masked-out final step. It simply cuts the trajectory at the last completed step.

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

## Eval suites

The BabyAI dataset generator can now build a validation *suite* instead of a single held-out split. The suite is still saved as `validation.parquet`, but each eval condition gets its own `data_source`, so training logs and dumped eval files stay separated by condition automatically.

This is the lowest-friction path because the trainer does not need any custom wiring:

- `data.val_data` still points to `validation.parquet`
- eval metrics are grouped by `data_source`
- eval-only runs against a saved checkpoint use the exact same suite file

The `scripts/cc/run_babyai_actor_critic_sync.sh` launcher enables this by default with two independent curves:

- `EVAL_SUITE_PARAM_1=room_size`
- `EVAL_SUITE_VALUES_1=8,10,12,14`
- `EVAL_SUITE_PARAM_2=num_dists`
- `EVAL_SUITE_VALUES_2=4,8,12,16`
- `EVAL_SUITE_INCLUDE_JOINT=false`
- `EVAL_SUITE_EXAMPLES_PER_CONDITION=64`

That produces:

- the same-task anchor
- a room-size-only curve
- a distractor-only curve

For `GoToLocal` with base kwargs `room_size=8` and `num_dists=8`, that defaults to a 7-condition training-time probe:

- `(8, 8)`
- `(10, 8)`, `(12, 8)`, `(14, 8)`
- `(8, 4)`, `(8, 12)`, `(8, 16)`

If you do want the full cross-product as a heavier checkpoint eval, set `EVAL_SUITE_INCLUDE_JOINT=true`.

The implementation is generic over env kwargs, so for other BabyAI tasks you usually only need to change the param names and values. If a task does not support `num_dists`, leave `EVAL_SUITE_PARAM_2` empty or disable the suite with `ENABLE_EVAL_SUITE=false`.

Each generated suite also writes `validation_manifest.json` so you can see exactly which conditions were included.

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
