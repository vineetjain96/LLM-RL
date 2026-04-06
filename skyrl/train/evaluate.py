from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.base import (
    GeneratorInterface,
    GeneratorOutput,
)
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
    prepare_generator_input,
)
from skyrl.train.utils import Timer
from skyrl.train.utils.logging_utils import (
    decode_example_from_generator_output,
    log_example,
)
from skyrl.train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_generator_output,
)


def _add_eval_rollout_metrics(eval_metrics: Dict[str, float], rollout_metrics: Dict[str, Any]) -> None:
    """Adds rollout metrics to eval outputs, skipping training-only response-length metrics."""
    for key, value in rollout_metrics.items():
        if key.startswith("response_lengths/"):
            continue
        eval_metrics[f"eval/all/{key}"] = value


def _get_last_step_generator_output(concat_generator_outputs: GeneratorOutput) -> GeneratorOutput:
    """Aligns step-wise generator output to one row per trajectory for eval metrics."""
    generator_output_last_step = defaultdict(list)
    is_last_step_mask = concat_generator_outputs["is_last_step"]
    assert is_last_step_mask is not None, "step-wise eval requires is_last_step"

    num_steps = len(is_last_step_mask)
    num_trajectories = sum(is_last_step_mask)
    for key, value in concat_generator_outputs.items():
        if not isinstance(value, list):
            continue

        if len(value) == num_steps:
            generator_output_last_step[key] = [
                item for item, is_last_step in zip(value, is_last_step_mask) if is_last_step
            ]
            continue

        if len(value) == num_trajectories:
            generator_output_last_step[key] = value
            continue

        raise AssertionError(
            f"Length mismatch: expected {num_steps} step rows or {num_trajectories} trajectory rows, "
            f"got {len(value)} for key {key}"
        )

    return generator_output_last_step


@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        generator (GeneratorInterface): generator to use
        cfg (SkyRLTrainConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use

    Returns:
        Dict[str, float]: evaluation metrics
    """

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    concat_all_envs: List[str] = []
    concat_env_extras: List[Dict[str, Any]] = []
    concat_uids: List[str] = []
    sampling_params = cfg.generator.eval_sampling_params
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            prompts,
            cfg.generator.eval_n_samples_per_prompt,
            get_sampling_params_for_backend(cfg.generator.inference_engine.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        validate_generator_output(len(generator_input["prompts"]), generator_output)
        generator_outputs.append(generator_output)
        concat_all_envs.extend(generator_input["env_classes"])
        concat_env_extras.extend(generator_input["env_extras"])
        concat_uids.extend(uids)
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
    example_prompt, example_response, example_reward = decode_example_from_generator_output(
        tokenizer,
        concat_generator_outputs,
        step_wise=False,
    )
    log_example(
        logger,
        prompt=example_prompt,
        response=example_response,
        reward=example_reward,
    )

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        concat_generator_outputs, concat_uids, concat_data_sources, cfg.generator.eval_n_samples_per_prompt
    )

    # 3. Calculate overall metrics across all datasets
    overall_metrics = get_metrics_from_generator_output(concat_generator_outputs, concat_uids)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_metrics["avg_score"],
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }
    )

    _add_eval_rollout_metrics(eval_metrics, concat_generator_outputs["rollout_metrics"])

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    return eval_metrics


@torch.no_grad()
async def evaluate_step_wise(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories for step-wise training.

    Currently assumes that the rewards are assigned to the last step of each trajectory.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        generator (GeneratorInterface): generator to use
        cfg (SkyRLTrainConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use

    Returns:
        Dict[str, float]: evaluation metrics
    """

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    concat_all_envs: List[str] = []
    concat_env_extras: List[Dict[str, Any]] = []
    concat_uids: List[str] = []
    env_metric_weighted_sums: Dict[str, float] = defaultdict(float)
    total_eval_trajectories = 0
    sampling_params = cfg.generator.eval_sampling_params
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            prompts,
            cfg.generator.eval_n_samples_per_prompt,
            get_sampling_params_for_backend(cfg.generator.inference_engine.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        traj_id_to_input = {
            traj_id.instance_id: {"env_class": env_class, "env_extras": env_extra}
            for traj_id, env_class, env_extra in zip(
                generator_input["trajectory_ids"], generator_input["env_classes"], generator_input["env_extras"]
            )
        }
        for traj_id in generator_output["trajectory_ids"]:
            assert traj_id.instance_id in traj_id_to_input, f"Trajectory ID {traj_id.instance_id} not found in input"
            concat_all_envs.append(traj_id_to_input[traj_id.instance_id]["env_class"])
            concat_env_extras.append(traj_id_to_input[traj_id.instance_id]["env_extras"])
            concat_uids.append(traj_id.instance_id)
        validate_generator_output(generator_input, generator_output, step_wise=True)
        if generator_output["rollout_metrics"] is not None:
            num_trajectories = sum(generator_output["is_last_step"])
            total_eval_trajectories += num_trajectories
            for key, value in generator_output["rollout_metrics"].items():
                if key.startswith("environment/"):
                    env_metric_weighted_sums[key] += value * num_trajectories
        generator_outputs.append(generator_output)
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
    example_prompt, example_response, example_reward = decode_example_from_generator_output(
        tokenizer,
        concat_generator_outputs,
        step_wise=True,
    )
    log_example(
        logger,
        prompt=example_prompt,
        response=example_response,
        reward=example_reward,
    )

    # Only use one row per trajectory for eval metrics.
    generator_output_last_step = _get_last_step_generator_output(concat_generator_outputs)
    is_last_step_mask = concat_generator_outputs["is_last_step"]
    uids_last_step = [uid for uid, is_last_step in zip(concat_uids, is_last_step_mask) if is_last_step]
    data_sources_last_step = [
        data_source for data_source, is_last_step in zip(concat_data_sources, is_last_step_mask) if is_last_step
    ]

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        generator_output_last_step, uids_last_step, data_sources_last_step, cfg.generator.eval_n_samples_per_prompt
    )
    # 3. Calculate overall metrics across all datasets
    overall_metrics = get_metrics_from_generator_output(generator_output_last_step, uids_last_step)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_metrics["avg_score"],
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }
    )
    _add_eval_rollout_metrics(eval_metrics, concat_generator_outputs["rollout_metrics"])
    if total_eval_trajectories > 0:
        for key, weighted_sum in env_metric_weighted_sums.items():
            eval_metrics[f"eval/all/{key}"] = weighted_sum / total_eval_trajectories

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    return eval_metrics
