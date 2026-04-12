import numpy as np
import pytest

from skyrl_gym.envs.babyai_text.env import BabyAITextEnv
from skyrl_gym.envs.babyai_text import utils


class _DummyUnwrappedEnv:
    carrying = None


class _DummyEnv:
    unwrapped = _DummyUnwrappedEnv()


class _DummyStepEnv:
    def __init__(self, step_output):
        self.unwrapped = _DummyUnwrappedEnv()
        self._step_output = step_output

    def step(self, action_idx):
        return self._step_output


def test_generate_obs_from_image_uses_minigrid_bottom_center_coordinates():
    env = BabyAITextEnv(env_config={}, extras={})
    env._env = _DummyEnv()

    image = np.zeros((7, 7, 3), dtype=np.int64)
    image[3, 5] = [6, 0, 0]  # red ball 1 step ahead
    image[3, 3] = [5, 1, 0]  # green key 3 steps ahead
    image[2, 6] = [2, 5, 0]  # grey wall 1 step to the left
    image[4, 6] = [6, 2, 0]  # blue ball 1 step to the right
    env._obs = {"image": image}

    obs_text = env._generate_obs_from_image()

    assert "You see a red ball 1 step(s) ahead." in obs_text
    assert "You see a green key 3 step(s) ahead." in obs_text
    assert "You see a grey wall 1 step(s) to your left." in obs_text
    assert "You see a blue ball 1 step(s) to your right." in obs_text


def test_invalid_action_before_turn_cap_returns_feedback():
    env = BabyAITextEnv(env_config={}, extras={"max_turns": 2})

    result = env.step("invalid action")

    assert result["done"] is False
    assert result["reward"] == 0.0
    assert len(result["observations"]) == 1
    assert "I couldn't understand your action." in result["observations"][0]["content"]
    assert result["metadata"] == {
        "parsed_action": None,
        "valid_action": False,
        "success": False,
        "steps": 1,
        "terminated": False,
        "truncated": False,
        "env_truncated": False,
        "turn_limit_reached": False,
        "reward_mode": "binary_outcome",
        "completed_subgoals": 0,
        "total_subgoals": 0,
        "subgoal_potential": 0.0,
        "subgoal_potential_delta": 0.0,
    }


def test_invalid_action_at_turn_cap_ends_episode():
    env = BabyAITextEnv(env_config={}, extras={"max_turns": 2})
    env.turns = 1
    env._step_count = 1

    result = env.step("invalid action")

    assert result["done"] is True
    assert result["reward"] == 0.0
    assert len(result["observations"]) == 1
    assert "I couldn't understand your action." in result["observations"][0]["content"]
    assert result["metadata"] == {
        "parsed_action": None,
        "valid_action": False,
        "success": False,
        "steps": 2,
        "terminated": False,
        "truncated": True,
        "env_truncated": False,
        "turn_limit_reached": True,
        "reward_mode": "binary_outcome",
        "completed_subgoals": 0,
        "total_subgoals": 0,
        "subgoal_potential": 0.0,
        "subgoal_potential_delta": 0.0,
    }


def test_truncated_step_returns_boundary_observation_and_flags():
    env = BabyAITextEnv(env_config={}, extras={"max_turns": 5})
    env._env = _DummyStepEnv(({"mission": "go to the red ball"}, 0.0, False, True, {}))
    env._obs = {"mission": "go to the red ball"}
    env._mission = "go to the red ball"

    result = env.step("<action>move forward</action>")

    assert result["done"] is True
    assert result["reward"] == 0.0
    assert len(result["observations"]) == 1
    assert "You performed: move forward" in result["observations"][0]["content"]
    assert result["metadata"] == {
        "parsed_action": "move forward",
        "valid_action": True,
        "success": False,
        "steps": 1,
        "terminated": False,
        "truncated": True,
        "env_truncated": True,
        "turn_limit_reached": False,
        "reward_mode": "binary_outcome",
        "completed_subgoals": 0,
        "total_subgoals": 0,
        "subgoal_potential": 0.0,
        "subgoal_potential_delta": 0.0,
    }


def test_terminated_step_does_not_return_boundary_observation():
    env = BabyAITextEnv(env_config={}, extras={"max_turns": 5})
    env._env = _DummyStepEnv(({"mission": "go to the red ball"}, 1.0, True, False, {}))
    env._obs = {"mission": "go to the red ball"}
    env._mission = "go to the red ball"

    result = env.step("<action>move forward</action>")

    assert result["done"] is True
    assert result["reward"] == 1.0
    assert result["observations"] == []
    assert result["metadata"] == {
        "parsed_action": "move forward",
        "valid_action": True,
        "success": True,
        "steps": 1,
        "terminated": True,
        "truncated": False,
        "env_truncated": False,
        "turn_limit_reached": False,
        "reward_mode": "binary_outcome",
        "completed_subgoals": 0,
        "total_subgoals": 0,
        "subgoal_potential": 0.0,
        "subgoal_potential_delta": 0.0,
    }


def test_efficiency_outcome_reward_matches_previous_shaping():
    env = BabyAITextEnv(
        env_config={},
        extras={"max_turns": 5, "reward_spec": {"reward_mode": "efficiency_outcome"}},
    )
    env._env = _DummyStepEnv(({"mission": "go to the red ball"}, 1.0, True, False, {}))
    env._obs = {"mission": "go to the red ball"}
    env._mission = "go to the red ball"

    result = env.step("<action>move forward</action>")

    assert result["done"] is True
    assert result["reward"] == pytest.approx(0.9)
    assert result["metadata"]["reward_mode"] == "efficiency_outcome"


def test_subgoal_delta_reward_uses_potential_difference():
    env = BabyAITextEnv(
        env_config={},
        extras={"max_turns": 5, "reward_spec": {"reward_mode": "subgoal_delta"}},
    )
    env._env = _DummyStepEnv(({"mission": "go to the red ball"}, 0.0, False, False, {}))
    env._obs = {"mission": "go to the red ball"}
    env._mission = "go to the red ball"
    env._completed_subgoals = 1
    env._total_subgoals = 4
    env._subgoal_potential = 0.25
    env._refresh_subgoal_progress = lambda success_override=False: (2, 4, 0.5)

    result = env.step("<action>move forward</action>")

    assert result["done"] is False
    assert result["reward"] == pytest.approx(0.25)
    assert result["metadata"]["reward_mode"] == "subgoal_delta"
    assert result["metadata"]["completed_subgoals"] == 2
    assert result["metadata"]["total_subgoals"] == 4
    assert result["metadata"]["subgoal_potential"] == pytest.approx(0.5)
    assert result["metadata"]["subgoal_potential_delta"] == pytest.approx(0.25)


def test_get_subgoal_progress_handles_nested_partial_sequences():
    verifier = pytest.importorskip("minigrid.envs.babyai.core.verifier")

    first = verifier.GoToInstr(verifier.ObjDesc("key", "red"))
    second = verifier.GoToInstr(verifier.ObjDesc("ball", "blue"))
    partial_and = verifier.AndInstr(first, second)
    partial_and.a_done = "success"
    partial_and.b_done = "continue"

    final = verifier.GoToInstr(verifier.ObjDesc("door", "green"))
    root = verifier.BeforeInstr(partial_and, final)
    root.a_done = "continue"
    root.b_done = False

    completed, total, potential = utils.get_subgoal_progress(root, success=False)

    assert completed == 1
    assert total == 3
    assert potential == pytest.approx(1 / 3)


def test_get_subgoal_progress_success_override_marks_atomic_task_complete():
    verifier = pytest.importorskip("minigrid.envs.babyai.core.verifier")

    atomic_task = verifier.GoToInstr(verifier.ObjDesc("key", "red"))

    completed, total, potential = utils.get_subgoal_progress(atomic_task, success=True)

    assert completed == 1
    assert total == 1
    assert potential == 1.0
