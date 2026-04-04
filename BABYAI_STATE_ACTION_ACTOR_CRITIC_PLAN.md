# BabyAI State-Action Actor-Critic, FSDP Sync v1

## Summary
- Add a new BabyAI-focused actor-critic mode that treats each multi-turn BabyAI context as the RL state and the parsed environment action chosen on that turn as the RL action.
- Keep the existing LM actor. The policy remains text generation, but the RL objective becomes action-level: one scalar advantage per BabyAI step, applied only to the assistant action tokens for that turn.
- Replace the current step-wise “final-step reward then broadcast backward” behavior with true step-level bootstrapping:
  - Learn `Q(s,a)` with 1-step TD.
  - Learn `V(s)` with TD-`lambda`.
  - Use actor advantage `A(s,a) = Q(s,a) - V(s)`.
- Scope v1 to FSDP synchronous training only. Do not include fully async or Megatron in this first implementation.

## Key Changes
### 1. New algorithm mode and config contract
- Add a new user-facing algorithm selector, e.g. `trainer.algorithm.advantage_estimator=state_action_td`.
- Reuse existing `gamma`, `lambd`, and `advantage_batch_normalize`.
- Add a small `trainer.algorithm.state_action` config block for the new path:
  - `q_head_prefix: "q_head"`
  - `q_loss_coef: 1.0`
  - `v_loss_coef: 1.0`
  - `critic_loss_type: "smooth_l1"`
- For this mode, validate and require:
  - `trainer.strategy in {fsdp, fsdp2}`
  - `trainer.critic.model.path` is set
  - `generator.step_wise_trajectories=true`
  - `generator.batched=false`
  - `generator.use_conversation_multi_turn=true`
  - `environment.env_class=babyai_text`
- Set the recommended policy defaults for this mode to:
  - `trainer.algorithm.policy_loss_type="gspo"`
  - `trainer.algorithm.loss_reduction="sequence_mean"`
  This makes the PPO ratio act on the whole BabyAI action span instead of tokenwise updates.

### 2. Transition-level rollout data instead of token-level value targets
- Extend the step-wise generator output to carry per-step transition metadata rather than dropping env step metadata.
- Add an optional `step_metadata` list to `GeneratorOutput`, with one entry per flattened step containing at least:
  - `parsed_action`
  - `valid_action`
  - `success`
  - `steps`
- In the trainer’s step-wise conversion path, derive and store scalar transition tensors:
  - `step_reward`
  - `done` / `bootstrap_mask`
  - `state_index`
  - `action_end_index`
  - `next_state_index`
  - `parsed_action_id` for logging/debug only
  - `action_valid`
- Keep existing token tensors (`sequences`, `attention_mask`, `loss_mask`, `response_mask`) so the policy path stays compatible with the current LM training stack.

### 3. Dual-head critic for `Q(s,a)` and `V(s)`
- Replace the tokenwise PPO critic behavior only for the new mode with a shared-backbone, dual-head critic:
  - `v_head` scores the state representation.
  - `q_head` scores the chosen state-action representation.
- Use one forward pass over the full step-wise sequence and gather hidden states at three positions:
  - `state_index`: last prompt token, representing `s_t`
  - `action_end_index`: last assistant action token, representing `s_t, a_t`
  - `next_state_index`: last valid token in the full step sample, representing `s_{t+1}` when non-terminal
- Return scalar tensors from critic forward in this mode:
  - `v_t = V(s_t)`
  - `q_t = Q(s_t, a_t)`
  - `next_v_t = V(s_{t+1})`
- Leave the existing tokenwise critic/value-head path untouched for PPO/GAE.

### 4. New target and advantage computation
- Do not route this mode through the existing token-only `ppo_utils.compute_advantages_and_returns(...)` path.
- Add a dedicated trainer branch for the new estimator that reconstructs per-trajectory step sequences using `trajectory_ids` and `is_last_step`.
- Compute targets from the pre-update critic snapshot for the batch:
  - `q_target_t = r_t + gamma * stopgrad(next_v_t) * (1 - done_t)`
  - `v_target_t` via TD-`lambda` over the step sequence using detached bootstrapped values
- Train critics with scalar losses:
  - `q_loss = SmoothL1(q_t, q_target_t)`
  - `v_loss = SmoothL1(v_t, v_target_t)`
  - final critic loss = `q_loss_coef * q_loss + v_loss_coef * v_loss`
- Compute actor advantage exactly as the requested decomposition:
  - `adv_t = stopgrad(q_t - v_t)`
- Broadcast `adv_t` over the assistant action tokens of that step only:
  - use `loss_mask`, not `response_mask`, so observation tokens never receive policy gradient
- If `advantage_batch_normalize=true`, normalize these per-step scalar advantages before broadcasting.

### 5. Policy update semantics
- Keep the LM actor unchanged.
- Interpret the BabyAI action probability as the sequence probability of the generated action span for that turn.
- Use sequence-level PPO/GSPO over the action tokens for that step, which matches the classic RL notion of one discrete environment action per state.
- Keep invalid parsed actions in training:
  - they remain real transitions with zero/low reward and a next observation
  - log them explicitly via `action_valid` / invalid-action metrics
  - do not drop or special-case them in v1

### 6. Worker, batch, and checkpoint wiring
- Extend `TrainingInputBatch` and `Experience` with optional actor-critic fields rather than overloading `values` / `returns`:
  - `step_reward`, `done`, `state_index`, `action_end_index`, `next_state_index`
  - `q_values`, `v_values`, `q_targets`, `v_targets`
- Add a critic-worker branch for the new mode that computes scalar dual-head losses instead of the current tokenwise PPO critic loss.
- Keep checkpointing behavior unchanged apart from the critic now owning two heads instead of one for this mode.
- Do not add a persistent target network in v1; use the detached pre-update critic snapshot from the batch forward pass as the bootstrap source.

### 7. User-facing scripts and docs
- Add a dedicated synchronous BabyAI launcher for this algorithm, separate from the GRPO and fully-async scripts.
- Document the required settings and the reason for them:
  - step-wise trajectories are mandatory
  - sequence-level policy loss is intentional
  - FSDP-only in v1
- Update BabyAI docs/comments so users understand that this path is a turn-level actor-critic, not token-level PPO.

## Public API / Interface Changes
- `GeneratorOutput` gains optional `step_metadata: Optional[List[Dict[str, Any]]]`.
- `TrainingInputBatch` gains optional actor-critic transition fields and scalar critic target fields.
- Critic forward output in this mode returns named scalar tensors (`q_values`, `v_values`, `next_v_values`) instead of a single tokenwise `output`.
- Config gains a new estimator name plus a small `trainer.algorithm.state_action` subsection.

## Test Plan
- Generator unit tests:
  - step metadata survives flattening in `step_wise_trajectories=true`
  - `state_index`, `action_end_index`, and `next_state_index` are correct with and without observation tokens
  - terminal steps produce `done=true` and zero bootstrap
- Math/unit tests for target construction:
  - single-step successful episode
  - multi-step episode with intermediate zero reward and terminal success
  - failed/timeout episode
  - invalid parsed action transition
  - TD-`lambda` targets respect trajectory boundaries and do not leak across repetitions
- Model/worker tests:
  - dual-head critic gathers the intended token positions
  - critic loss uses scalar turn targets only
  - observation tokens never contribute to policy loss
- Trainer integration tests:
  - new estimator bypasses the old final-step broadcast logic
  - advantages are scalar per step and broadcast only over `loss_mask`
  - policy defaults for this mode are sequence-level, not token-level
- FSDP smoke test:
  - one tiny end-to-end BabyAI sync training step with the new mode
- Validation tests:
  - reject Megatron
  - reject fully async
  - reject missing critic
  - reject non-stepwise or batched generation
  - reject non-BabyAI envs in v1

## Assumptions and Defaults
- v1 support is intentionally BabyAI-only at the user level, even though the internal transition-field names should stay generic enough for later reuse.
- The LM actor stays generative; no separate 7-action policy head is added.
- The policy objective is action-level through sequence-level PPO/GSPO over the generated action span.
- Invalid actions are retained as training data and surfaced via logging, not filtered out.
- No persistent target network is added in v1; detached pre-update critic outputs are the bootstrap source.
- Planned markdown output path once writing is allowed: `BABYAI_STATE_ACTION_ACTOR_CRITIC_PLAN.md` at repo root.
