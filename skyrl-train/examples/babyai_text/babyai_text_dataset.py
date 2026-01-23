"""
Dataset generation script for BabyAI-Text environment.

This script creates training and validation datasets in parquet format
for various BabyAI environments.

Usage:
    uv run examples/babyai_text/babyai_text_dataset.py --output_dir $HOME/data/babyai_text

The script generates prompts with missions from BabyAI environments that can be
used to train language models with reinforcement learning.
"""

import argparse
import os
import random
from typing import List, Dict, Any

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


def get_mission_from_env(env_name: str, seed: int = None) -> str:
    """
    Create a BabyAI environment and get its mission.

    Args:
        env_name: Name of the BabyAI environment
        seed: Random seed for reproducibility

    Returns:
        Mission string from the environment
    """
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401

        env = gym.make(env_name)
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
    max_turns: int = 64,
    split: str = "train",
) -> Dict[str, Any]:
    """
    Create a single training example.

    Args:
        env_name: BabyAI environment name
        idx: Example index (used as seed)
        system_prompt: System prompt for the LLM
        max_turns: Maximum number of turns allowed
        split: Dataset split name

    Returns:
        Dictionary with prompt, env_class, reward_spec, and extra_info
    """
    mission = get_mission_from_env(env_name, seed=idx)

    return {
        "data_source": "babyai_text",
        "prompt": [
            {"role": "system", "content": system_prompt},
        ],
        "env_class": "babyai_text",
        "reward_spec": {
            "method": "rule",
            "reward_on_success": 1.0,
            "step_penalty": 0.0,
        },
        "extra_info": {
            "env_name": env_name,
            "mission": mission,
            "max_turns": max_turns,
            "split": split,
            "seed": idx,
        },
    }


def create_dataset(
    num_examples: int,
    envs: List[str],
    system_prompt: str,
    max_turns: int,
    split_name: str,
    start_seed: int = 0,
) -> Dataset:
    """
    Create a dataset of BabyAI-Text examples.

    Args:
        num_examples: Number of examples to generate
        envs: List of BabyAI environment names
        system_prompt: System prompt to use
        max_turns: Maximum turns per episode
        split_name: Name of the split (train/validation)
        start_seed: Starting seed for reproducibility

    Returns:
        HuggingFace Dataset
    """
    examples = []

    for idx in range(num_examples):
        # Cycle through environments
        env_name = envs[idx % len(envs)]

        example = create_example(
            env_name=env_name,
            idx=start_seed + idx,
            system_prompt=system_prompt,
            max_turns=max_turns,
            split=split_name,
        )
        examples.append(example)

    return Dataset.from_list(examples)


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
        help="Difficulty level of environments to use",
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

    args = parser.parse_args()

    # Expand home directory
    output_dir = os.path.expanduser(args.output_dir)

    # Select environments based on difficulty
    if args.difficulty == "all":
        envs = []
        for env_list in BABYAI_ENVS.values():
            envs.extend(env_list)
    elif args.difficulty == "mixed":
        # Mix of easy and medium for balanced training
        envs = BABYAI_ENVS["easy"] + BABYAI_ENVS["medium"]
    else:
        envs = BABYAI_ENVS.get(args.difficulty, BABYAI_ENVS["easy"])

    print(f"Using environments: {envs}")
    print(f"Generating {args.train_size} training examples...")

    # Create training dataset
    train_dataset = create_dataset(
        num_examples=args.train_size,
        envs=envs,
        system_prompt=args.system_prompt,
        max_turns=args.max_turns,
        split_name="train",
        start_seed=0,
    )

    print(f"Generating {args.val_size} validation examples...")

    # Create validation dataset with different seeds
    val_dataset = create_dataset(
        num_examples=args.val_size,
        envs=envs,
        system_prompt=args.system_prompt,
        max_turns=args.max_turns,
        split_name="validation",
        start_seed=100000,  # Different seed range for validation
    )

    # Save datasets
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"\nDataset saved to {output_dir}")
    print(f"  - Training examples: {len(train_dataset)}")
    print(f"  - Validation examples: {len(val_dataset)}")
    print(f"  - Environments: {len(envs)}")
    print(f"  - Max turns: {args.max_turns}")


if __name__ == "__main__":
    main()
