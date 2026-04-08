import numpy as np

from skyrl_gym.envs.babyai_text.env import BabyAITextEnv


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
    }


def test_terminated_step_does_not_return_boundary_observation():
    env = BabyAITextEnv(env_config={}, extras={"max_turns": 5})
    env._env = _DummyStepEnv(({"mission": "go to the red ball"}, 1.0, True, False, {}))
    env._obs = {"mission": "go to the red ball"}
    env._mission = "go to the red ball"

    result = env.step("<action>move forward</action>")

    assert result["done"] is True
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
    }
