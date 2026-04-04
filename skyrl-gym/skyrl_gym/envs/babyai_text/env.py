"""
BabyAI-Text Environment for SkyRL.

This environment wraps BabyAI/MiniGrid environments to provide text-based
observations and accept text-based actions from language models.

The BabyAI environments are grid-world navigation tasks where an agent must
complete missions like "go to the red ball" or "pick up the blue key and
open the purple door".

References:
- BabyAI: https://github.com/mila-iqia/babyai
- Grounding LLMs with online RL: https://github.com/flowersteam/Grounding_LLMs_with_online_RL
"""

from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.babyai_text import utils


class BabyAITextEnv(BaseTextEnv):
    """
    Text-based wrapper for BabyAI/MiniGrid environments.

    This environment:
    1. Accepts text actions from the LLM (e.g., "turn left", "move forward")
    2. Converts them to discrete actions for the underlying environment
    3. Generates text observations describing what the agent sees
    4. Computes rewards based on mission completion

    The environment is multi-turn: the agent takes multiple steps until
    the mission is completed or the maximum number of turns is reached.
    """

    def __init__(self, env_config: DictConfig, extras: Optional[Dict[str, Any]] = None):
        super().__init__()
        extras = extras or {}
        extra_info = extras.get("extra_info") or {}

        # Environment configuration
        self.env_name = extra_info.get(
            "env_name",
            extras.get("env_name", env_config.get("env_name", "BabyAI-GoToLocal-v0")),
        )
        self.max_steps = env_config.get("max_steps", 64)
        self.language = env_config.get("language", "english")
        self.render_mode = env_config.get("render_mode", None)
        config_env_kwargs = dict(env_config.get("env_kwargs", {}) or {})
        extra_env_kwargs = dict(extras.get("env_kwargs", {}) or {})
        extra_info_env_kwargs = dict(extra_info.get("env_kwargs", {}) or {})
        self.env_kwargs = {**config_env_kwargs, **extra_env_kwargs, **extra_info_env_kwargs}

        # Multi-turn settings
        self.max_turns = extras.get("max_turns", self.max_steps)
        if "max_turns" in extra_info:
            self.max_turns = int(extra_info["max_turns"])

        # Reward configuration
        reward_spec = extras.get("reward_spec", {})
        self.reward_on_success = reward_spec.get("reward_on_success", 1.0)
        self.step_penalty = reward_spec.get("step_penalty", 0.0)

        # Mission override (optional, for custom prompts)
        self.mission_override = extras.get("mission", None)

        # Seed for reproducibility
        self.seed = extra_info.get("seed", extras.get("seed", None))

        # Internal state
        self._env = None
        self._obs = None
        self._mission = None
        self._step_count = 0
        self._done = False
        self._success = False

    def _create_env(self):
        """Create the underlying BabyAI/MiniGrid environment."""
        try:
            import gymnasium as gym

            # Try importing babyai which registers the environments
            try:
                import babyai  # noqa: F401
            except ImportError:
                pass

            # Try importing minigrid
            try:
                import minigrid  # noqa: F401
            except ImportError:
                pass

            make_kwargs = dict(self.env_kwargs)
            if self.render_mode is not None and "render_mode" not in make_kwargs:
                make_kwargs["render_mode"] = self.render_mode
            self._env = gym.make(self.env_name, **make_kwargs)

            if self.seed is not None:
                self._env.reset(seed=self.seed)
            else:
                self._env.reset()

        except ImportError as e:
            raise ImportError(
                f"Failed to import gymnasium or BabyAI/MiniGrid. "
                f"Please install them with: pip install gymnasium minigrid babyai\n"
                f"Original error: {e}"
            )

    def _get_text_observation(self) -> str:
        """
        Generate a text description of the current observation.

        Uses MiniGrid's gen_graph method if available, otherwise generates
        a basic text description from the observation array.
        """
        if self._env is None:
            return ""

        # Try to use the environment's built-in text generation
        if hasattr(self._env.unwrapped, "gen_graph"):
            obs_text = self._env.unwrapped.gen_graph(self.language)
        elif hasattr(self._env.unwrapped, "gen_obs_decode"):
            obs_text = self._env.unwrapped.gen_obs_decode()
        else:
            # Generate observation from image array
            obs_text = self._generate_obs_from_image()

        return obs_text

    def _generate_obs_from_image(self) -> str:
        """
        Generate text observation from the image observation array.

        The MiniGrid observation is a 7x7x3 array where each cell encodes
        (object_type, color, state).
        """
        if self._obs is None or "image" not in self._obs:
            return "You see an empty room."

        image = self._obs["image"]
        descriptions = []

        view_size = image.shape[0]
        # MiniGrid places the agent at the bottom-center of the egocentric view.
        agent_x = view_size // 2
        agent_y = view_size - 1

        # Check what's directly in front
        for dist in range(1, agent_y + 1):
            cell = image[agent_x, agent_y - dist]
            obj_type, obj_color, _ = cell[0], cell[1], cell[2]

            if obj_type > 1:  # Not unseen or empty
                obj_name, color_name = utils.get_object_description(obj_type, obj_color)
                if color_name:
                    descriptions.append(f"You see a {color_name} {obj_name} {dist} step(s) ahead.")
                else:
                    descriptions.append(f"You see a {obj_name} {dist} step(s) ahead.")

        # Check left side
        for dist in range(1, agent_x + 1):
            cell = image[agent_x - dist, agent_y]
            obj_type, obj_color, _ = cell[0], cell[1], cell[2]

            if obj_type > 1:
                obj_name, color_name = utils.get_object_description(obj_type, obj_color)
                if color_name:
                    descriptions.append(f"You see a {color_name} {obj_name} {dist} step(s) to your left.")
                else:
                    descriptions.append(f"You see a {obj_name} {dist} step(s) to your left.")

        # Check right side
        for dist in range(1, view_size - agent_x):
            cell = image[agent_x + dist, agent_y]
            obj_type, obj_color, _ = cell[0], cell[1], cell[2]

            if obj_type > 1:
                obj_name, color_name = utils.get_object_description(obj_type, obj_color)
                if color_name:
                    descriptions.append(f"You see a {color_name} {obj_name} {dist} step(s) to your right.")
                else:
                    descriptions.append(f"You see a {obj_name} {dist} step(s) to your right.")

        if not descriptions:
            descriptions.append("You see an empty space ahead.")

        # Add carrying information
        if hasattr(self._env.unwrapped, "carrying") and self._env.unwrapped.carrying:
            obj = self._env.unwrapped.carrying
            descriptions.append(f"You are carrying a {obj.color} {obj.type}.")

        return "\n".join(descriptions)

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the environment for a new episode.

        Args:
            prompt: Initial conversation context (may contain system message)

        Returns:
            Tuple of (initial observations, metadata)
        """
        # Reset internal state
        self.turns = 0
        self._step_count = 0
        self._done = False
        self._success = False

        # Create and reset the underlying environment
        self._create_env()
        self._obs, info = self._env.reset(seed=self.seed)

        # Get the mission
        if self.mission_override:
            self._mission = self.mission_override
        elif "mission" in self._obs:
            self._mission = self._obs["mission"]
        elif hasattr(self._env.unwrapped, "mission"):
            self._mission = self._env.unwrapped.mission
        else:
            self._mission = "Complete the task."

        # Generate initial observation
        obs_text = self._get_text_observation()
        formatted_obs = utils.format_observation(
            obs_text=obs_text,
            mission=self._mission,
            carrying=self._get_carrying_description(),
            step_count=self._step_count,
            max_steps=self.max_turns,
        )

        # Add the observation to the prompt
        initial_prompt = list(prompt)  # Copy the prompt
        initial_prompt.append({"role": "user", "content": formatted_obs})

        return initial_prompt, {"mission": self._mission}

    def _get_carrying_description(self) -> Optional[str]:
        """Get description of what the agent is carrying."""
        if hasattr(self._env.unwrapped, "carrying") and self._env.unwrapped.carrying:
            obj = self._env.unwrapped.carrying
            return f"{obj.color} {obj.type}"
        return None

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the environment.

        Args:
            action: Text action from the LLM

        Returns:
            BaseTextEnvStepOutput with observations, reward, done flag, and metadata
        """
        self.turns += 1
        self._step_count += 1

        # Parse the text action
        action_idx = utils.parse_action(action)

        if action_idx is None:
            # Invalid action - provide feedback
            feedback = (
                "I couldn't understand your action. Please use one of the available actions:\n"
                "turn left, turn right, move forward, pick up, drop, toggle, done\n"
                "Format your action as: <action>your action</action>"
            )

            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": feedback}],
                reward=0.0,
                done=False,
                metadata={"parsed_action": None, "valid_action": False},
            )

        # Execute the action in the underlying environment
        self._obs, reward, terminated, truncated, info = self._env.step(action_idx)

        self._done = terminated or truncated or self.turns >= self.max_turns
        self._success = terminated and reward > 0

        # Compute reward
        final_reward = utils.compute_reward(
            done=self._done,
            success=self._success,
            step_count=self._step_count,
            max_steps=self.max_turns,
            reward_on_success=self.reward_on_success,
            step_penalty=self.step_penalty,
        )

        # Generate observation
        if self._done:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=final_reward,
                done=True,
                metadata={
                    "parsed_action": utils.ACTION_NAMES.get(action_idx, "unknown"),
                    "valid_action": True,
                    "success": self._success,
                    "steps": self._step_count,
                },
            )

        # Generate next observation
        obs_text = self._get_text_observation()
        formatted_obs = utils.format_observation(
            obs_text=obs_text,
            mission=self._mission,
            carrying=self._get_carrying_description(),
            step_count=self._step_count,
            max_steps=self.max_turns,
        )

        # Add action feedback
        action_name = utils.ACTION_NAMES.get(action_idx, "unknown")
        feedback = f"You performed: {action_name}\n\n{formatted_obs}"

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": feedback}],
            reward=final_reward,
            done=False,
            metadata={
                "parsed_action": action_name,
                "valid_action": True,
                "success": False,
                "steps": self._step_count,
            },
        )

    def close(self):
        """Clean up the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment-specific metrics."""
        return {
            "steps": self._step_count,
            "success": self._success,
            "mission": self._mission,
            "max_turns": self.max_turns,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes."""
        if not metrics:
            return {}

        n = len(metrics)
        avg_steps = sum(float(m.get("steps", 0)) for m in metrics) / n
        success_rate = sum(1 for m in metrics if m.get("success", False)) / n

        return {
            "avg_steps": avg_steps,
            "success_rate": success_rate,
        }
