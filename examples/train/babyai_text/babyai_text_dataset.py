"""
Dataset generation script for BabyAI-Text environment.

This script creates training and validation datasets in parquet format
for various BabyAI environments.

Usage:
    uv run --extra babyai examples/train/babyai_text/babyai_text_dataset.py --output_dir $HOME/data/babyai_text

The script generates prompts with missions from BabyAI environments that can be
used to train language models with reinforcement learning.
"""

import argparse
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional

from datasets import Dataset

# Available BabyAI environments organized by difficulty
BABYAI_ENVS = {
    # Simple navigation tasks
    "easy": [
        "BabyAI-GoToLocal-v0",
        "BabyAI-GoToObj-v0",
        "BabyAI-GoToRedBall-v0",
        "BabyAI-GoToRedBallGrey-v0",
    ],
    # Medium difficulty tasks
    "medium": [
        "BabyAI-GoToObjMaze-v0",
        "BabyAI-GoToDoor-v0",
        "BabyAI-PickupLoc-v0",
        "BabyAI-PickupDist-v0",
        "BabyAI-PutNextLocal-v0",
    ],
    # Hard tasks requiring sequences of actions
    "hard": [
        "BabyAI-OpenDoor-v0",
        "BabyAI-UnlockLocal-v0",
        "BabyAI-KeyInBox-v0",
        "BabyAI-UnlockPickup-v0",
        "BabyAI-BlockedUnlockPickup-v0",
    ],
    # Sequential tasks
    "sequential": [
        "BabyAI-GoToSeq-v0",
        "BabyAI-Synth-v0",
        "BabyAI-SynthLoc-v0",
        "BabyAI-SynthSeq-v0",
        "BabyAI-BossLevel-v0",
    ],
}

# Default system prompt for BabyAI tasks
SYSTEM_PROMPT = """You are an agent navigating a grid-world environment. Your goal is to complete missions by taking actions in the environment.

Available actions:
- turn left: Rotate 90 degrees to the left
- turn right: Rotate 90 degrees to the right
- move forward: Move one step in the direction you're facing
- pick up: Pick up the object in front of you
- drop: Drop the object you're carrying
- toggle: Open/close a door or interact with an object in front of you
- done: Signal that you've completed the mission

Respond with your chosen action in the format: <action>your action</action>

Think step by step about how to complete the mission efficiently."""


def parse_scalar(value: str) -> Any:
    """Parse a comma-separated sweep value from string to bool/int/float/str."""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_env_kwargs_json(env_kwargs_json: str) -> Dict[str, Any]:
    """Parse env kwargs from a JSON object string."""
    if not env_kwargs_json:
        return {}
    parsed = json.loads(env_kwargs_json)
    if not isinstance(parsed, dict):
        raise ValueError(f"--env_kwargs_json must be a JSON object, got: {type(parsed).__name__}")
    return parsed


def parse_sweep_values(raw_values: str) -> List[Any]:
    """Parse comma-separated sweep values."""
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("--sweep_values must contain at least one comma-separated value")
    return [parse_scalar(value) for value in values]


def parse_optional_values(raw_values: Optional[str], arg_name: str) -> Optional[List[Any]]:
    """Parse optional comma-separated values."""
    if raw_values is None:
        return None
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{arg_name} must contain at least one comma-separated value")
    return [parse_scalar(value) for value in values]


def parse_bool_flag(raw_value: str) -> bool:
    """Parse a CLI boolean flag from common truthy/falsy strings."""
    parsed = parse_scalar(raw_value)
    if isinstance(parsed, bool):
        return parsed
    raise ValueError(f"Expected a boolean value, got {raw_value!r}")


def slugify(value: str) -> str:
    """Convert a string into a filename-safe slug."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    cleaned = "".join(ch if ch in allowed else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def build_data_source_name(
    env_name: str,
    env_kwargs: Dict[str, Any],
    prefix: str,
) -> str:
    """Build a stable metric-friendly data source label for an eval condition."""
    parts = [slugify(prefix), slugify(env_name)]
    for key, value in sorted(env_kwargs.items()):
        parts.append(f"{slugify(str(key))}_{slugify(str(value))}")
    return "__".join(parts)


def dedupe_env_specs(env_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate env specs while preserving order."""
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for env_spec in env_specs:
        env_kwargs = env_spec.get("env_kwargs", {}) or {}
        key = (
            env_spec["env_name"],
            json.dumps(env_kwargs, sort_keys=True, default=str),
            env_spec.get("data_source"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(env_spec)
    return deduped


def resolve_eval_base_env_kwargs(
    env_name: str,
    base_env_kwargs: Dict[str, Any],
    curve_params: List[Optional[str]],
) -> Dict[str, Any]:
    """Resolve missing base kwarg defaults from the underlying environment when possible."""
    resolved_kwargs = dict(base_env_kwargs)
    missing_params = [param for param in curve_params if param and param not in resolved_kwargs]
    if not missing_params:
        return resolved_kwargs

    try:
        import gymnasium as gym
        import minigrid  # noqa: F401

        env = gym.make(env_name, **resolved_kwargs)
        try:
            env.reset(seed=0)
            env_unwrapped = env.unwrapped
            for param in missing_params:
                if hasattr(env_unwrapped, param):
                    resolved_kwargs[param] = getattr(env_unwrapped, param)
        finally:
            env.close()
    except ImportError:
        pass

    return resolved_kwargs


def build_eval_suite_specs(
    env_names: List[str],
    base_env_kwargs: Dict[str, Any],
    data_source_prefix: str,
    curve_param_1: Optional[str] = None,
    curve_values_1: Optional[List[Any]] = None,
    curve_param_2: Optional[str] = None,
    curve_values_2: Optional[List[Any]] = None,
    include_joint_curve: bool = True,
) -> List[Dict[str, Any]]:
    """Build a validation suite by varying up to two env kwargs around a base task."""
    if curve_param_1 and curve_values_1 is None:
        raise ValueError("--eval_curve_param_1 requires --eval_curve_values_1")
    if curve_values_1 is not None and not curve_param_1:
        raise ValueError("--eval_curve_values_1 requires --eval_curve_param_1")
    if curve_param_2 and curve_values_2 is None:
        raise ValueError("--eval_curve_param_2 requires --eval_curve_values_2")
    if curve_values_2 is not None and not curve_param_2:
        raise ValueError("--eval_curve_values_2 requires --eval_curve_param_2")
    if curve_param_1 and curve_param_2 and curve_param_1 == curve_param_2:
        raise ValueError("Eval suite curve params must be distinct")

    env_specs: List[Dict[str, Any]] = []

    def add_spec(env_name: str, env_kwargs: Dict[str, Any]) -> None:
        spec_kwargs = dict(env_kwargs)
        env_specs.append(
            {
                "env_name": env_name,
                "env_kwargs": spec_kwargs,
                "data_source": build_data_source_name(
                    env_name=env_name,
                    env_kwargs=spec_kwargs,
                    prefix=data_source_prefix,
                ),
            }
        )

    for env_name in env_names:
        resolved_base_env_kwargs = resolve_eval_base_env_kwargs(
            env_name=env_name,
            base_env_kwargs=base_env_kwargs,
            curve_params=[curve_param_1, curve_param_2],
        )
        add_spec(env_name, resolved_base_env_kwargs)

        if curve_param_1 and curve_values_1 is not None:
            for value in curve_values_1:
                add_spec(env_name, {**resolved_base_env_kwargs, curve_param_1: value})

        if curve_param_2 and curve_values_2 is not None:
            for value in curve_values_2:
                add_spec(env_name, {**resolved_base_env_kwargs, curve_param_2: value})

        if include_joint_curve and curve_param_1 and curve_param_2 and curve_values_1 and curve_values_2:
            for value_1 in curve_values_1:
                for value_2 in curve_values_2:
                    add_spec(
                        env_name,
                        {
                            **resolved_base_env_kwargs,
                            curve_param_1: value_1,
                            curve_param_2: value_2,
                        },
                    )

    return dedupe_env_specs(env_specs)


def save_rgb_image(frame: Any, output_path: str) -> str:
    """
    Save an RGB array to image file.

    Prefers PNG via Pillow when available. Falls back to PPM if Pillow
    is unavailable.
    """
    # Keep this import local so dataset generation still works without pillow.
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(frame).save(output_path)
        return output_path
    except Exception:
        pass

    # Fallback: write a binary PPM image.
    ppm_path = os.path.splitext(output_path)[0] + ".ppm"
    if len(frame.shape) != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected RGB image array with shape [H, W, 3+], got {getattr(frame, 'shape', None)}")

    rgb = frame[:, :, :3]
    if str(getattr(rgb, "dtype", "")) != "uint8":
        rgb = rgb.astype("uint8")

    height, width, _ = rgb.shape
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())
    return ppm_path


def render_env_preview(
    env_name: str,
    env_kwargs: Optional[Dict[str, Any]],
    seed: int,
    tile_size: int,
) -> Any:
    """Render one initial-state RGB frame for an environment config."""
    import gymnasium as gym
    import minigrid  # noqa: F401

    make_kwargs = dict(env_kwargs or {})
    make_kwargs.pop("render_mode", None)

    env = gym.make(env_name, render_mode="rgb_array", **make_kwargs)
    try:
        env.reset(seed=seed)
        frame = env.render()
        if frame is None and hasattr(env.unwrapped, "get_frame"):
            frame = env.unwrapped.get_frame(highlight=False, tile_size=tile_size)
        if frame is None:
            raise RuntimeError("env.render() returned None and env has no usable get_frame()")
        return frame
    finally:
        env.close()


def save_env_previews(
    env_specs: List[Dict[str, Any]],
    preview_dir: str,
    seeds: List[int],
    tile_size: int,
) -> None:
    """Save one preview image per env spec per seed."""
    os.makedirs(preview_dir, exist_ok=True)

    print(f"Saving environment previews to: {preview_dir}")
    saved = 0
    failed = 0

    for idx, env_spec in enumerate(env_specs):
        env_name = env_spec["env_name"]
        env_kwargs = dict(env_spec.get("env_kwargs", {}) or {})
        sweep_param = env_spec.get("sweep_param")
        sweep_value = env_spec.get("sweep_value")

        descriptor = {
            "env_name": env_name,
            "env_kwargs": env_kwargs,
            "sweep_param": sweep_param,
            "sweep_value": sweep_value,
        }
        spec_hash = hashlib.sha1(json.dumps(descriptor, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
        base_name = slugify(env_name)
        if sweep_param is not None:
            base_name = f"{base_name}_{slugify(str(sweep_param))}_{slugify(str(sweep_value))}"

        for seed in seeds:
            output_path = os.path.join(preview_dir, f"{idx:03d}_{base_name}_seed{seed}_{spec_hash}.png")
            try:
                frame = render_env_preview(env_name=env_name, env_kwargs=env_kwargs, seed=seed, tile_size=tile_size)
                final_path = save_rgb_image(frame, output_path)
                print(f"  [preview] saved: {final_path}")
                saved += 1
            except Exception as exc:
                print(f"  [preview] failed: env={env_name} kwargs={env_kwargs} seed={seed} error={exc}")
                failed += 1

    print(f"Preview summary: saved={saved}, failed={failed}")


def get_mission_from_env(env_name: str, seed: int = None, env_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a BabyAI environment and get its mission.

    Args:
        env_name: Name of the BabyAI environment
        seed: Random seed for reproducibility
        env_kwargs: Optional kwargs forwarded to gym.make

    Returns:
        Mission string from the environment
    """
    env_kwargs = env_kwargs or {}
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401

        env = gym.make(env_name, **env_kwargs)
        obs, info = env.reset(seed=seed)

        # Get mission from observation or environment
        if "mission" in obs:
            mission = obs["mission"]
        elif hasattr(env.unwrapped, "mission"):
            mission = env.unwrapped.mission
        else:
            mission = "Complete the task."

        env.close()
        return mission

    except ImportError:
        # If minigrid is not installed, generate synthetic missions
        return generate_synthetic_mission(env_name)


def generate_synthetic_mission(env_name: str) -> str:
    """
    Generate a synthetic mission string for testing without minigrid installed.
    """
    colors = ["red", "green", "blue", "purple", "yellow", "grey"]
    objects = ["ball", "box", "key", "door"]

    color = random.choice(colors)
    obj = random.choice(objects)

    if "GoTo" in env_name:
        return f"go to the {color} {obj}"
    elif "Pickup" in env_name:
        return f"pick up the {color} {obj}"
    elif "PutNext" in env_name:
        color2 = random.choice(colors)
        obj2 = random.choice(objects)
        return f"put the {color} {obj} next to the {color2} {obj2}"
    elif "Open" in env_name or "Unlock" in env_name:
        return f"open the {color} door"
    else:
        return f"go to the {color} {obj}"


def create_example(
    env_name: str,
    idx: int,
    system_prompt: str,
    data_source: str = "babyai_text",
    env_kwargs: Optional[Dict[str, Any]] = None,
    reward_mode: str = "binary_outcome",
    reward_on_success: float = 1.0,
    step_penalty: float = 0.0,
    sweep_param: Optional[str] = None,
    sweep_value: Optional[Any] = None,
    max_turns: int = 64,
    split: str = "train",
) -> Dict[str, Any]:
    """
    Create a single training example.

    Args:
        env_name: BabyAI environment name
        idx: Example index (used as seed)
        system_prompt: System prompt for the LLM
        env_kwargs: Optional kwargs forwarded to env constructor
        sweep_param: Name of swept parameter, if any
        sweep_value: Value of swept parameter, if any
        max_turns: Maximum number of turns allowed
        split: Dataset split name

    Returns:
        Dictionary with prompt, env_class, reward_spec, and extra_info
    """
    env_kwargs = dict(env_kwargs or {})
    mission = get_mission_from_env(env_name, seed=idx, env_kwargs=env_kwargs)

    extra_info = {
        "env_name": env_name,
        "mission": mission,
        "max_turns": max_turns,
        "split": split,
        "seed": idx,
    }
    # Parquet cannot serialize an empty struct field. Only include env_kwargs
    # when at least one kwarg is present.
    if env_kwargs:
        extra_info["env_kwargs"] = env_kwargs
    if sweep_param is not None:
        extra_info["sweep_param"] = sweep_param
        extra_info["sweep_value"] = sweep_value

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
        ],
        "env_class": "babyai_text",
        "reward_spec": {
            "method": "rule",
            "reward_mode": reward_mode,
            "reward_on_success": reward_on_success,
            "step_penalty": step_penalty,
        },
        "extra_info": extra_info,
    }


def create_dataset(
    num_examples: int,
    env_specs: List[Dict[str, Any]],
    system_prompt: str,
    max_turns: int,
    split_name: str,
    reward_mode: str = "binary_outcome",
    reward_on_success: float = 1.0,
    step_penalty: float = 0.0,
    start_seed: int = 0,
) -> Dataset:
    """
    Create a dataset of BabyAI-Text examples.

    Args:
        num_examples: Number of examples to generate
        env_specs: List of dicts with env_name/env_kwargs and optional sweep metadata
        system_prompt: System prompt to use
        max_turns: Maximum turns per episode
        split_name: Name of the split (train/validation)
        start_seed: Starting seed for reproducibility

    Returns:
        HuggingFace Dataset
    """
    examples = []

    for idx in range(num_examples):
        # Cycle through configured environment specs
        env_spec = env_specs[idx % len(env_specs)]

        example = create_example(
            env_name=env_spec["env_name"],
            idx=start_seed + idx,
            system_prompt=system_prompt,
            data_source=env_spec.get("data_source", "babyai_text"),
            env_kwargs=env_spec.get("env_kwargs", {}),
            reward_mode=reward_mode,
            reward_on_success=reward_on_success,
            step_penalty=step_penalty,
            sweep_param=env_spec.get("sweep_param"),
            sweep_value=env_spec.get("sweep_value"),
            max_turns=max_turns,
            split=split_name,
        )
        examples.append(example)

    return Dataset.from_list(examples)


def create_dataset_with_examples_per_spec(
    examples_per_spec: int,
    env_specs: List[Dict[str, Any]],
    system_prompt: str,
    max_turns: int,
    split_name: str,
    reward_mode: str = "binary_outcome",
    reward_on_success: float = 1.0,
    step_penalty: float = 0.0,
    start_seed: int = 0,
) -> Dataset:
    """Create a dataset with a fixed number of examples for every env spec."""
    if examples_per_spec <= 0:
        raise ValueError("examples_per_spec must be positive")

    examples = []
    for spec_idx, env_spec in enumerate(env_specs):
        spec_start_seed = start_seed + spec_idx * examples_per_spec
        for example_offset in range(examples_per_spec):
            example = create_example(
                env_name=env_spec["env_name"],
                idx=spec_start_seed + example_offset,
                system_prompt=system_prompt,
                data_source=env_spec.get("data_source", "babyai_text"),
                env_kwargs=env_spec.get("env_kwargs", {}),
                reward_mode=reward_mode,
                reward_on_success=reward_on_success,
                step_penalty=step_penalty,
                sweep_param=env_spec.get("sweep_param"),
                sweep_value=env_spec.get("sweep_value"),
                max_turns=max_turns,
                split=split_name,
            )
            examples.append(example)

    return Dataset.from_list(examples)


def write_eval_suite_manifest(
    output_dir: str,
    env_specs: List[Dict[str, Any]],
    examples_per_spec: int,
) -> None:
    """Write metadata describing the generated validation suite."""
    manifest = {
        "examples_per_condition": examples_per_spec,
        "num_conditions": len(env_specs),
        "total_examples": examples_per_spec * len(env_specs),
        "conditions": [
            {
                "data_source": env_spec["data_source"],
                "env_name": env_spec["env_name"],
                "env_kwargs": env_spec.get("env_kwargs", {}),
            }
            for env_spec in env_specs
        ],
    }

    manifest_path = os.path.join(output_dir, "validation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Generate BabyAI-Text dataset for training")
    parser.add_argument(
        "--output_dir",
        default="~/data/babyai_text",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--difficulty",
        default="easy",
        choices=["easy", "medium", "hard", "sequential", "all", "mixed"],
        help="Difficulty level of environments to use when --env_name is not provided",
    )
    parser.add_argument(
        "--env_name",
        default=None,
        help="Optional single env id to use (overrides --difficulty selection)",
    )
    parser.add_argument(
        "--env_kwargs_json",
        default="{}",
        help='JSON object of kwargs forwarded to gym.make (e.g. \'{"room_size": 8}\')',
    )
    parser.add_argument(
        "--sweep_param",
        default=None,
        help="Optional env kwarg name to sweep across (e.g. room_size)",
    )
    parser.add_argument(
        "--sweep_values",
        default=None,
        help="Optional comma-separated sweep values for --sweep_param (e.g. 5,8,12)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of training examples",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=500,
        help="Number of validation examples",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=64,
        help="Maximum turns per episode",
    )
    parser.add_argument(
        "--system_prompt",
        default=SYSTEM_PROMPT,
        help="System prompt for the LLM",
    )
    parser.add_argument(
        "--data_source",
        default="babyai_text",
        help="Data source label for the training split and plain validation runs",
    )
    parser.add_argument(
        "--reward_mode",
        default="binary_outcome",
        choices=["binary_outcome", "efficiency_outcome", "subgoal_delta"],
        help="Reward mode written into each BabyAI example's reward_spec",
    )
    parser.add_argument(
        "--reward_on_success",
        type=float,
        default=1.0,
        help="Reward scale applied to successful completion or subgoal deltas",
    )
    parser.add_argument(
        "--step_penalty",
        type=float,
        default=0.0,
        help="Penalty applied on intermediate non-terminal steps",
    )
    parser.add_argument(
        "--eval_curve_param_1",
        default=None,
        help="Optional first env kwarg to sweep in validation (e.g. room_size)",
    )
    parser.add_argument(
        "--eval_curve_values_1",
        default=None,
        help="Optional comma-separated values for --eval_curve_param_1",
    )
    parser.add_argument(
        "--eval_curve_param_2",
        default=None,
        help="Optional second env kwarg to sweep in validation (e.g. num_dists)",
    )
    parser.add_argument(
        "--eval_curve_values_2",
        default=None,
        help="Optional comma-separated values for --eval_curve_param_2",
    )
    parser.add_argument(
        "--eval_include_joint_curve",
        default="true",
        help="Whether to include the joint grid across both eval curve params",
    )
    parser.add_argument(
        "--eval_examples_per_condition",
        type=int,
        default=None,
        help="If set, validation becomes a labeled eval suite with this many examples per condition",
    )
    parser.add_argument(
        "--eval_data_source_prefix",
        default="babyai_eval",
        help="Prefix used for per-condition validation data_source labels",
    )
    args = parser.parse_args()

    # Expand home directory
    output_dir = os.path.expanduser(args.output_dir)
    base_env_kwargs = parse_env_kwargs_json(args.env_kwargs_json)
    eval_curve_values_1 = parse_optional_values(args.eval_curve_values_1, "--eval_curve_values_1")
    eval_curve_values_2 = parse_optional_values(args.eval_curve_values_2, "--eval_curve_values_2")
    eval_include_joint_curve = parse_bool_flag(args.eval_include_joint_curve)
    eval_suite_enabled = args.eval_examples_per_condition is not None

    if (args.sweep_param is None) != (args.sweep_values is None):
        parser.error("--sweep_param and --sweep_values must be provided together")
    if eval_suite_enabled and args.sweep_param is not None:
        parser.error("--sweep_param is not supported together with eval suite generation")

    # Select environments based on difficulty
    if args.env_name:
        envs = [args.env_name]
    elif args.difficulty == "all":
        envs = []
        for env_list in BABYAI_ENVS.values():
            envs.extend(env_list)
    elif args.difficulty == "mixed":
        envs = BABYAI_ENVS["easy"] + BABYAI_ENVS["medium"]
    else:
        envs = BABYAI_ENVS.get(args.difficulty, BABYAI_ENVS["easy"])

    env_specs: List[Dict[str, Any]] = []
    if args.sweep_param is not None:
        sweep_values = parse_sweep_values(args.sweep_values)
        for env_name in envs:
            for value in sweep_values:
                env_kwargs = dict(base_env_kwargs)
                env_kwargs[args.sweep_param] = value
                env_specs.append(
                    {
                        "env_name": env_name,
                        "env_kwargs": env_kwargs,
                        "data_source": args.data_source,
                        "sweep_param": args.sweep_param,
                        "sweep_value": value,
                    }
                )
    else:
        env_specs = [
            {
                "env_name": env_name,
                "env_kwargs": dict(base_env_kwargs),
                "data_source": args.data_source,
            }
            for env_name in envs
        ]

    val_env_specs = env_specs
    if eval_suite_enabled:
        val_env_specs = build_eval_suite_specs(
            env_names=envs,
            base_env_kwargs=base_env_kwargs,
            data_source_prefix=args.eval_data_source_prefix,
            curve_param_1=args.eval_curve_param_1,
            curve_values_1=eval_curve_values_1,
            curve_param_2=args.eval_curve_param_2,
            curve_values_2=eval_curve_values_2,
            include_joint_curve=eval_include_joint_curve,
        )

    print("Using environment specs:")
    for env_spec in env_specs:
        print(f"  - {env_spec['env_name']} kwargs={env_spec.get('env_kwargs', {})}")

    if eval_suite_enabled:
        print("Validation eval suite specs:")
        for env_spec in val_env_specs:
            print(
                f"  - {env_spec['data_source']}: "
                f"{env_spec['env_name']} kwargs={env_spec.get('env_kwargs', {})}"
            )

    # Always save preview images for visualization.
    save_env_previews(
        env_specs=dedupe_env_specs(env_specs + val_env_specs),
        preview_dir=os.path.join(output_dir, "env_previews"),
        seeds=[0, 1, 2, 3],
        tile_size=32,
    )

    print(f"Generating {args.train_size} training examples...")

    # Create training dataset
    train_dataset = create_dataset(
        num_examples=args.train_size,
        env_specs=env_specs,
        system_prompt=args.system_prompt,
        max_turns=args.max_turns,
        split_name="train",
        reward_mode=args.reward_mode,
        reward_on_success=args.reward_on_success,
        step_penalty=args.step_penalty,
        start_seed=0,
    )

    # Save datasets
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    train_dataset.to_parquet(train_path)

    if eval_suite_enabled:
        print(
            "Generating validation eval suite with "
            f"{len(val_env_specs)} conditions x {args.eval_examples_per_condition} examples..."
        )
        val_dataset = create_dataset_with_examples_per_spec(
            examples_per_spec=args.eval_examples_per_condition,
            env_specs=val_env_specs,
            system_prompt=args.system_prompt,
            max_turns=args.max_turns,
            split_name="validation",
            reward_mode=args.reward_mode,
            reward_on_success=args.reward_on_success,
            step_penalty=args.step_penalty,
            start_seed=100000,
        )
        write_eval_suite_manifest(
            output_dir=output_dir,
            env_specs=val_env_specs,
            examples_per_spec=args.eval_examples_per_condition,
        )
    else:
        print(f"Generating {args.val_size} validation examples...")
        val_dataset = create_dataset(
            num_examples=args.val_size,
            env_specs=env_specs,
            system_prompt=args.system_prompt,
            max_turns=args.max_turns,
            split_name="validation",
            reward_mode=args.reward_mode,
            reward_on_success=args.reward_on_success,
            step_penalty=args.step_penalty,
            start_seed=100000,  # Different seed range for validation
        )

    val_dataset.to_parquet(val_path)

    print(f"\nDataset saved to {output_dir}")
    print(f"  - Training examples: {len(train_dataset)}")
    print(f"  - Validation examples: {len(val_dataset)}")
    print(f"  - Train env specs: {len(env_specs)}")
    print(f"  - Validation env specs: {len(val_env_specs)}")
    print(f"  - Base env kwargs: {base_env_kwargs}")
    print(f"  - Max turns: {args.max_turns}")
    print(f"  - Reward mode: {args.reward_mode}")
    print(f"  - Reward on success: {args.reward_on_success}")
    print(f"  - Step penalty: {args.step_penalty}")
    if eval_suite_enabled:
        print(f"  - Eval suite manifest: {os.path.join(output_dir, 'validation_manifest.json')}")


if __name__ == "__main__":
    main()
