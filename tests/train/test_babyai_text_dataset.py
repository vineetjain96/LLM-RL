from collections import Counter

from examples.train.babyai_text import babyai_text_dataset


def test_build_eval_suite_specs_dedupes_anchor_and_joint_grid():
    env_specs = babyai_text_dataset.build_eval_suite_specs(
        env_names=["BabyAI-GoToLocal-v0"],
        base_env_kwargs={"room_size": 8, "num_dists": 8},
        data_source_prefix="babyai_eval",
        curve_param_1="room_size",
        curve_values_1=[8, 10],
        curve_param_2="num_dists",
        curve_values_2=[4, 8],
        include_joint_curve=True,
    )

    assert env_specs == [
        {
            "env_name": "BabyAI-GoToLocal-v0",
            "env_kwargs": {"room_size": 8, "num_dists": 8},
            "data_source": "babyai_eval__BabyAI-GoToLocal-v0__num_dists_8__room_size_8",
        },
        {
            "env_name": "BabyAI-GoToLocal-v0",
            "env_kwargs": {"room_size": 10, "num_dists": 8},
            "data_source": "babyai_eval__BabyAI-GoToLocal-v0__num_dists_8__room_size_10",
        },
        {
            "env_name": "BabyAI-GoToLocal-v0",
            "env_kwargs": {"room_size": 8, "num_dists": 4},
            "data_source": "babyai_eval__BabyAI-GoToLocal-v0__num_dists_4__room_size_8",
        },
        {
            "env_name": "BabyAI-GoToLocal-v0",
            "env_kwargs": {"room_size": 10, "num_dists": 4},
            "data_source": "babyai_eval__BabyAI-GoToLocal-v0__num_dists_4__room_size_10",
        },
    ]


def test_create_dataset_with_examples_per_spec_keeps_condition_labels(monkeypatch):
    monkeypatch.setattr(
        babyai_text_dataset,
        "get_mission_from_env",
        lambda env_name, seed=None, env_kwargs=None: f"{env_name}:{seed}:{env_kwargs}",
    )

    env_specs = babyai_text_dataset.build_eval_suite_specs(
        env_names=["BabyAI-GoToLocal-v0"],
        base_env_kwargs={"room_size": 8, "num_dists": 8},
        data_source_prefix="babyai_eval",
        curve_param_1="room_size",
        curve_values_1=[8, 10],
        curve_param_2="num_dists",
        curve_values_2=[8],
        include_joint_curve=False,
    )

    dataset = babyai_text_dataset.create_dataset_with_examples_per_spec(
        examples_per_spec=3,
        env_specs=env_specs,
        system_prompt="system",
        max_turns=16,
        split_name="validation",
        start_seed=100,
    )

    data_sources = dataset["data_source"]
    counts = Counter(data_sources)

    assert counts == {
        "babyai_eval__BabyAI-GoToLocal-v0__num_dists_8__room_size_8": 3,
        "babyai_eval__BabyAI-GoToLocal-v0__num_dists_8__room_size_10": 3,
    }

    first_row = dataset[0]
    fourth_row = dataset[3]
    assert first_row["extra_info"]["split"] == "validation"
    assert first_row["extra_info"]["seed"] == 100
    assert first_row["extra_info"]["env_kwargs"] == {"room_size": 8, "num_dists": 8}
    assert fourth_row["extra_info"]["seed"] == 103
    assert fourth_row["extra_info"]["env_kwargs"] == {"room_size": 10, "num_dists": 8}


def test_build_eval_suite_specs_resolves_implicit_base_defaults(monkeypatch):
    monkeypatch.setattr(
        babyai_text_dataset,
        "resolve_eval_base_env_kwargs",
        lambda env_name, base_env_kwargs, curve_params: {"room_size": 8, "num_dists": 8},
    )

    env_specs = babyai_text_dataset.build_eval_suite_specs(
        env_names=["BabyAI-GoToLocal-v0"],
        base_env_kwargs={},
        data_source_prefix="babyai_eval",
        curve_param_1="room_size",
        curve_values_1=[8, 10],
        curve_param_2="num_dists",
        curve_values_2=[4, 8],
        include_joint_curve=True,
    )

    assert [env_spec["env_kwargs"] for env_spec in env_specs] == [
        {"room_size": 8, "num_dists": 8},
        {"room_size": 10, "num_dists": 8},
        {"room_size": 8, "num_dists": 4},
        {"room_size": 10, "num_dists": 4},
    ]
