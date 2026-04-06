"""Utility functions for BabyAI-Text environment."""

import re
from typing import Optional, Tuple

# Mapping from text action names to BabyAI action indices
# BabyAI Actions: left (0), right (1), forward (2), pickup (3), drop (4), toggle (5), done (6)
ACTION_MAP = {
    "turn left": 0,
    "left": 0,
    "turn right": 1,
    "right": 1,
    "move forward": 2,
    "forward": 2,
    "go forward": 2,
    "pick up": 3,
    "pickup": 3,
    "grab": 3,
    "drop": 4,
    "put down": 4,
    "toggle": 5,
    "open": 5,
    "close": 5,
    "done": 6,
    "finish": 6,
}

# Reverse mapping for generating action descriptions
ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}


def parse_action(action_text: str) -> Optional[int]:
    """
    Parse a text action from the LLM into a BabyAI action index.

    Supports formats like:
    - Direct action: "turn left", "move forward"
    - Action tag: <action>turn left</action>
    - Boxed format: \\action{turn left}

    Args:
        action_text: The text output from the LLM

    Returns:
        Action index (0-6) or None if parsing fails
    """
    action_text = action_text.lower().strip()

    # Try to extract action from tags
    tag_patterns = [
        r"<action>(.*?)</action>",
        r"\\action\{(.*?)\}",
        r"\[action\](.*?)\[/action\]",
        r"action:\s*(.+?)(?:\n|$)",
    ]

    extracted_action = None
    for pattern in tag_patterns:
        match = re.search(pattern, action_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted_action = match.group(1).strip().lower()
            break

    if extracted_action is None:
        return None

    # Map to action index
    return ACTION_MAP.get(extracted_action)


def format_observation(
    obs_text: str,
    mission: str,
    carrying: Optional[str] = None,
    step_count: int = 0,
    max_steps: int = 100,
) -> str:
    """
    Format the observation into a structured text format for the LLM.

    Args:
        obs_text: Text description of what the agent sees
        mission: The current mission/goal
        carrying: Description of what the agent is carrying (if anything)
        step_count: Current step number
        max_steps: Maximum steps allowed

    Returns:
        Formatted observation string
    """
    parts = [
        f"Mission: {mission}",
        f"Step: {step_count}/{max_steps}",
        "",
        "Observation:",
        obs_text,
    ]

    if carrying:
        parts.append(f"\nYou are carrying: {carrying}")

    parts.append("\nAvailable actions: turn left, turn right, move forward, pick up, drop, toggle, done")
    parts.append("\nProvide your action in the format: <action>your action</action>")

    return "\n".join(parts)


def compute_reward(
    done: bool,
    success: bool,
    step_count: int,
    max_steps: int,
    reward_on_success: float = 1.0,
    step_penalty: float = 0.0,
    partial_reward: float = 0.0,
) -> float:
    """
    Compute the reward for a step in the BabyAI environment.

    Args:
        done: Whether the episode is done
        success: Whether the mission was completed successfully
        step_count: Current step number
        max_steps: Maximum steps allowed
        reward_on_success: Reward for successful completion
        step_penalty: Penalty per step (to encourage efficiency)
        partial_reward: Partial reward for progress (not implemented yet)

    Returns:
        Reward value
    """
    if success:
        # Scale reward by efficiency (faster completion = higher reward)
        efficiency_bonus = 1.0 - (step_count / max_steps)
        return reward_on_success * (0.5 + 0.5 * efficiency_bonus)
    elif done:
        # Episode ended without success
        return 0.0
    else:
        # Intermediate step penalty (optional)
        return -step_penalty


def get_direction_name(direction: int) -> str:
    """Convert direction index to name."""
    directions = ["east", "south", "west", "north"]
    return directions[direction % 4]


def get_object_description(obj_type: int, obj_color: int) -> Tuple[str, str]:
    """
    Convert object type and color indices to text descriptions.

    BabyAI object types: unseen (0), empty (1), wall (2), floor (3), door (4),
                         key (5), ball (6), box (7), goal (8), lava (9), agent (10)
    BabyAI colors: red (0), green (1), blue (2), purple (3), yellow (4), grey (5)
    """
    object_names = [
        "unseen",
        "empty",
        "wall",
        "floor",
        "door",
        "key",
        "ball",
        "box",
        "goal",
        "lava",
        "agent",
    ]
    color_names = ["red", "green", "blue", "purple", "yellow", "grey"]

    obj_name = object_names[obj_type] if obj_type < len(object_names) else "unknown"
    color_name = color_names[obj_color] if obj_color < len(color_names) else ""

    return obj_name, color_name
