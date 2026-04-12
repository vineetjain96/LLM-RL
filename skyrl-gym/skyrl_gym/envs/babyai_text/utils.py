"""Utility functions for BabyAI-Text environment."""

import re
from functools import lru_cache
from typing import Any, Optional, Tuple

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

REWARD_MODE_BINARY_OUTCOME = "binary_outcome"
REWARD_MODE_EFFICIENCY_OUTCOME = "efficiency_outcome"
REWARD_MODE_SUBGOAL_DELTA = "subgoal_delta"
SUPPORTED_REWARD_MODES = {
    REWARD_MODE_BINARY_OUTCOME,
    REWARD_MODE_EFFICIENCY_OUTCOME,
    REWARD_MODE_SUBGOAL_DELTA,
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


def validate_reward_mode(reward_mode: str) -> str:
    """Validate and normalize the configured BabyAI reward mode."""
    if reward_mode not in SUPPORTED_REWARD_MODES:
        supported_modes = ", ".join(sorted(SUPPORTED_REWARD_MODES))
        raise ValueError(f"Unsupported BabyAI reward_mode={reward_mode!r}. Expected one of: {supported_modes}")
    return reward_mode


def compute_reward(
    reward_mode: str,
    done: bool,
    success: bool,
    step_count: int,
    max_steps: int,
    reward_on_success: float = 1.0,
    step_penalty: float = 0.0,
    subgoal_potential_delta: float = 0.0,
) -> float:
    """
    Compute the reward for a step in the BabyAI environment.

    Args:
        reward_mode: Reward shaping mode to use.
        done: Whether the episode is done
        success: Whether the mission was completed successfully
        step_count: Current step number
        max_steps: Maximum steps allowed
        reward_on_success: Reward for successful completion
        step_penalty: Penalty per step (to encourage efficiency)
        subgoal_potential_delta: Change in subgoal completion potential for
            the current step. Only used by ``subgoal_delta``.

    Returns:
        Reward value
    """
    reward_mode = validate_reward_mode(reward_mode)
    reward = 0.0

    if reward_mode == REWARD_MODE_SUBGOAL_DELTA:
        reward += reward_on_success * subgoal_potential_delta
    elif success:
        if reward_mode == REWARD_MODE_BINARY_OUTCOME:
            reward += reward_on_success
        else:
            efficiency_bonus = 1.0 - (step_count / max_steps)
            reward += reward_on_success * (0.5 + 0.5 * efficiency_bonus)

    if not success and not done:
        reward -= step_penalty

    return reward


@lru_cache(maxsize=1)
def _load_babyai_verifier_types() -> Optional[tuple[type[Any], type[Any], type[Any], type[Any]]]:
    """Lazily load BabyAI verifier classes when the optional dependency is available."""
    try:
        from minigrid.envs.babyai.core.verifier import ActionInstr, AfterInstr, AndInstr, BeforeInstr
    except ImportError:
        return None
    return ActionInstr, AndInstr, BeforeInstr, AfterInstr


def _count_atomic_subgoals(
    instr: Any,
    action_instr_cls: type[Any],
    and_instr_cls: type[Any],
    before_instr_cls: type[Any],
    after_instr_cls: type[Any],
) -> int:
    if isinstance(instr, action_instr_cls):
        return 1
    if isinstance(instr, (and_instr_cls, before_instr_cls, after_instr_cls)):
        return _count_atomic_subgoals(
            instr.instr_a,
            action_instr_cls,
            and_instr_cls,
            before_instr_cls,
            after_instr_cls,
        ) + _count_atomic_subgoals(
            instr.instr_b,
            action_instr_cls,
            and_instr_cls,
            before_instr_cls,
            after_instr_cls,
        )
    return 0


def _count_completed_atomic_subgoals(
    instr: Any,
    action_instr_cls: type[Any],
    and_instr_cls: type[Any],
    before_instr_cls: type[Any],
    after_instr_cls: type[Any],
    status_hint: Optional[str] = None,
) -> int:
    if isinstance(instr, action_instr_cls):
        return int(status_hint == "success")

    if isinstance(instr, (and_instr_cls, before_instr_cls, after_instr_cls)):
        if status_hint == "success":
            return _count_atomic_subgoals(
                instr,
                action_instr_cls,
                and_instr_cls,
                before_instr_cls,
                after_instr_cls,
            )

        return _count_completed_atomic_subgoals(
            instr.instr_a,
            action_instr_cls,
            and_instr_cls,
            before_instr_cls,
            after_instr_cls,
            status_hint=getattr(instr, "a_done", None),
        ) + _count_completed_atomic_subgoals(
            instr.instr_b,
            action_instr_cls,
            and_instr_cls,
            before_instr_cls,
            after_instr_cls,
            status_hint=getattr(instr, "b_done", None),
        )

    return 0


def get_subgoal_progress(instr: Any, success: bool = False) -> Tuple[int, int, float]:
    """
    Estimate BabyAI subgoal progress from the verifier state.

    The verifier tracks partial completion for sequential missions in-memory, so
    we can translate that state into a potential over atomic action clauses
    without modifying the underlying MiniGrid package.
    """
    verifier_types = _load_babyai_verifier_types()
    if verifier_types is None or instr is None:
        return 0, 0, 0.0

    action_instr_cls, and_instr_cls, before_instr_cls, after_instr_cls = verifier_types
    total_subgoals = _count_atomic_subgoals(
        instr,
        action_instr_cls,
        and_instr_cls,
        before_instr_cls,
        after_instr_cls,
    )
    if total_subgoals == 0:
        return 0, 0, 0.0

    if success:
        completed_subgoals = total_subgoals
    else:
        completed_subgoals = _count_completed_atomic_subgoals(
            instr,
            action_instr_cls,
            and_instr_cls,
            before_instr_cls,
            after_instr_cls,
        )

    potential = completed_subgoals / total_subgoals
    return completed_subgoals, total_subgoals, potential


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
