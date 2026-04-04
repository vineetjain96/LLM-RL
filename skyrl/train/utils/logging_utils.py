from typing import Any, Dict, List, Optional, Union

POSITIVE_RESPONSE_COLOR = "green"
NEGATIVE_RESPONSE_COLOR = "yellow"
BASE_PROMPT_COLOR = "cyan"


def _color_block_format_and_kwargs(
    text: str,
    color: str,
    field_prefix: str,
) -> tuple[str, dict]:
    """Build a format string and kwargs for a multi-line colored block.

    The format string will look like:
        "<color>{p0}</color>\n<color>{p1}</color>\n..."

    where "p0", "p1", ... are placeholder names starting with `field_prefix`.
    """
    # Ensure at least one line
    lines = text.splitlines() or [""]

    fmt_lines = []
    kwargs: dict[str, str] = {}

    for i, line in enumerate(lines):
        key = f"{field_prefix}{i}"
        # NOTE: double braces {{ }} so that {key} survives into str.format
        fmt_lines.append(f"<{color}>{{{key}}}</{color}>")
        kwargs[key] = line

    fmt = "\n".join(fmt_lines)
    return fmt, kwargs


def _trajectory_key(trajectory_id: Any) -> Any:
    if hasattr(trajectory_id, "instance_id") and hasattr(trajectory_id, "repetition_id"):
        return (trajectory_id.instance_id, trajectory_id.repetition_id)
    return trajectory_id


def decode_example_from_generator_output(
    tokenizer: Any,
    generator_output: Dict[str, Any],
    step_wise: bool = False,
) -> tuple[str, str, Optional[Union[float, List[float]]]]:
    """Decode one human-readable example from generator output.

    For step-wise training, generator output is flattened into one sample per step.
    This helper reconstructs the full transcript for the first trajectory so the
    logged example reflects the actual multi-turn interaction.
    """
    prompt_token_ids = generator_output["prompt_token_ids"]
    response_ids = generator_output["response_ids"]
    rewards = generator_output["rewards"]

    if not prompt_token_ids or not response_ids:
        return "", "", None

    if not step_wise:
        return tokenizer.decode(prompt_token_ids[0]), tokenizer.decode(response_ids[0]), rewards[0]

    trajectory_ids = generator_output.get("trajectory_ids")
    if not trajectory_ids:
        return tokenizer.decode(prompt_token_ids[0]), tokenizer.decode(response_ids[0]), rewards[0]

    first_key = _trajectory_key(trajectory_ids[0])
    selected_indices = [i for i, trajectory_id in enumerate(trajectory_ids) if _trajectory_key(trajectory_id) == first_key]

    prompt = tokenizer.decode(prompt_token_ids[selected_indices[0]])
    full_response_ids: List[int] = []
    selected_rewards = []
    for index in selected_indices:
        full_response_ids.extend(response_ids[index])
        selected_rewards.append(rewards[index])

    if selected_rewards and isinstance(selected_rewards[0], list):
        reward: Optional[Union[float, List[float]]] = [
            token_reward for step_rewards in selected_rewards for token_reward in step_rewards
        ]
    elif selected_rewards:
        reward = float(sum(float(step_reward) for step_reward in selected_rewards))
    else:
        reward = None

    return prompt, tokenizer.decode(full_response_ids), reward


def log_example(
    logger: Any,
    prompt: Any,
    response: str,
    reward: Optional[Union[float, List[float]]] = None,
) -> None:
    """
    Log a single example prompt and response with formatting and colors.

    Args:
        logger: The logger instance to use (expected to be loguru logger or compatible).
        prompt: The input prompt in OpenAI message format.
        response: The output response string.
        reward: The reward value(s) associated with the response.
    """
    reward_val = 0.0
    reward_str = "N/A"
    try:
        prompt_str = str(prompt)
        response_str = str(response)
        # --- Reward handling ---
        if reward is not None:
            if isinstance(reward, list):
                reward_val = float(sum(reward))
            else:
                reward_val = float(reward)
            reward_str = f"{reward_val:.4f}"

        # --- Color selection ---
        if reward is not None and reward_val > 0:
            response_color = POSITIVE_RESPONSE_COLOR
        else:
            response_color = NEGATIVE_RESPONSE_COLOR

        # --- Build per-line colored blocks in the *format string* ---
        prompt_fmt, prompt_kwargs = _color_block_format_and_kwargs(prompt_str, BASE_PROMPT_COLOR, "p")
        response_fmt, response_kwargs = _color_block_format_and_kwargs(response_str, response_color, "r")

        # Single format string with only our own markup and placeholders
        log_format = "Example:\n" f"  Input: {prompt_fmt}\n" "  Output (Total Reward: {reward}):\n" f"{response_fmt}"

        # Merge all args for str.format
        format_kwargs = {**prompt_kwargs, **response_kwargs, "reward": reward_str}

        # Let Loguru parse tags in log_format and then substitute arguments.
        logger.opt(colors=True).info(log_format, **format_kwargs)
    except Exception as e:
        print(f"Error pretty printing example, debug printing instead: {e}")
        print(f"Example:\n  Input: {prompt}\n  Output (Total Reward: {reward_str}):\n{response}")
