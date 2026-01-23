"""
BabyAI-Text training entrypoint for SkyRL.

Usage:
    uv run --isolated --extra vllm -m examples.babyai_text.main_babyai_text

This entrypoint registers the babyai_text environment and runs the training loop.
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the babyai_text environment inside the entrypoint task.
    # This ensures the environment is available in all Ray workers.
    register(
        id="babyai_text",
        entry_point="skyrl_gym.envs.babyai_text.env:BabyAITextEnv",
    )

    # Run the training loop
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Validate the configuration
    validate_cfg(cfg)

    # Initialize Ray cluster
    initialize_ray(cfg)

    # Run the training entrypoint
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
