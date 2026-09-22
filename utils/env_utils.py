import numpy as np
import ogbench

from utils.datasets import Dataset
from utils.evaluation import EpisodeMonitor


def make_env_and_datasets(dataset_name, dataset_path=None, dataset_only=False, cur_env=None):
    """Make OGBench environment and datasets.

    OGBench exposes two flavors of the same environments. Names containing 'singletask' are the single-task (standard
    offline RL) variants: their datasets come with 'rewards' and 'masks', and each transition needs an explicit
    'next_observations'. Every other name is a goal-conditioned variant, where rewards are relabeled on the fly from
    the sampled goals, so the compact (memory-efficient) dataset layout is used instead.

    Args:
        dataset_name: Name of the environment (dataset).
        dataset_path: Path to the dataset file.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).

    Returns:
        A tuple of the environment (if `dataset_only` is False), training dataset, and validation dataset.
    """
    goal_conditioned = "singletask" not in dataset_name.split("-")

    outputs = ogbench.make_env_and_datasets(
        dataset_name,
        dataset_path=dataset_path,
        compact_dataset=goal_conditioned,
        dataset_only=dataset_only,
        cur_env=cur_env,
    )
    if dataset_only:
        train_dataset, val_dataset = outputs
    else:
        env, train_dataset, val_dataset = outputs

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    # Clip dataset actions.
    eps = 1e-5
    train_dataset = train_dataset.copy(
        add_or_replace=dict(
            actions=np.clip(train_dataset["actions"], -1 + eps, 1 - eps)
        )
    )
    val_dataset = val_dataset.copy(
        add_or_replace=dict(actions=np.clip(val_dataset["actions"], -1 + eps, 1 - eps))
    )

    if dataset_only:
        return train_dataset, val_dataset
    else:
        env = EpisodeMonitor(env)
        env.reset()
        return env, train_dataset, val_dataset
