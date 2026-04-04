import numpy as np

from skyrl_gym.envs.babyai_text.env import BabyAITextEnv


class _DummyUnwrappedEnv:
    carrying = None


class _DummyEnv:
    unwrapped = _DummyUnwrappedEnv()


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
