from absl import app, flags
import flax.serialization
from ml_collections import config_flags
from collections import defaultdict
from functools import partial
from typing import List

from jaxrl_m.evaluation import supply_rng, evaluate, flatten, add_to
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict

import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import flax

from src import d4rl_utils
from src.agent import dynamics as learner
import wandb
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "walker2d-medium-v2", "Environment name.")
flags.DEFINE_string("save_dir", "log", "Logging dir (if not None, save params).")
flags.DEFINE_string("run_group", "DEBUG", "")
flags.DEFINE_integer("num_episodes", 50, "")
flags.DEFINE_integer("num_videos", 2, "")
flags.DEFINE_integer("log_steps", 1000, "")
flags.DEFINE_integer("eval_steps", 100000, "")
flags.DEFINE_integer("save_steps", 250000, "")
flags.DEFINE_integer("total_steps", 1000000, "")

seed = np.random.randint(low=0, high=10000000)
flags.DEFINE_integer("seed", seed, "")
flags.DEFINE_integer("batch_size", 1024, "")
flags.DEFINE_integer("num_elites", 5, "")
flags.DEFINE_integer("max_epochs_since_update", 5, "")
flags.DEFINE_float("holdout_ratio", 0.2, "")
flags.DEFINE_integer("wandb_offline", 0, "")


@jax.jit
def validate(agent, inputs, targets):
    mean, _ = agent.dynamics(inputs)
    loss = ((mean - targets) ** 2).mean(axis=(1, 2))
    val_loss = list(loss)
    return val_loss


def select_elites(metrics: List, num_elites: int) -> List[int]:
    pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
    pairs = sorted(pairs, key=lambda x: x[0])
    elites = [pairs[i][1] for i in range(num_elites)]
    return elites


def main(_):
    np.random.seed(FLAGS.seed)
    env = d4rl_utils.make_env(FLAGS.env_name)
    env.render("rgb_array")
    # Wandb
    wandb_config = default_wandb_config()
    FLAGS.wandb_offline = bool(FLAGS.wandb_offline)
    wandb_config.update(
        {
            "project": "offlineRL",
            "group": f"dynamics",
            "name": f"dynamics_{FLAGS.env_name}_{FLAGS.seed}",
            "offline": FLAGS.wandb_offline,
        }
    )
    config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
    algo_config = learner.get_default_config()
    config_flags.DEFINE_config_dict("algo", algo_config, lock_config=False)

    start_time = int(datetime.now().timestamp())
    FLAGS.wandb["name"] += f"_{start_time}"
    setup_wandb(FLAGS.algo.to_dict(), **FLAGS.wandb)
    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(
            FLAGS.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)

    # Dataset
    start_time = int(datetime.now().timestamp())
    dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    scaler = d4rl_utils.StandardScaler()

    inputs = np.concatenate([dataset["observations"], dataset["actions"]], axis=-1)
    targets = np.concatenate(
        [
            dataset["next_observations"] - dataset["observations"],
            dataset["rewards"].reshape(-1, 1),
        ],
        axis=-1,
    )
    dataset_size = inputs.shape[0]
    holdout_size = min(int(dataset_size * FLAGS.holdout_ratio), 1000)
    train_size = dataset_size - holdout_size
    train_split_indices = np.random.permutation(range(dataset_size))
    train_inputs, train_targets = (
        inputs[train_split_indices[:train_size]],
        targets[train_split_indices[:train_size]],
    )
    holdout_inputs, holdout_targets = (
        inputs[train_split_indices[train_size:]],
        targets[train_split_indices[train_size:]],
    )
    data_indices = np.random.randint(
        train_size,
        size=[FLAGS.algo.num_ensemble, train_size],
    )

    scaler.fit(train_inputs)
    train_inputs = scaler.transform(train_inputs)
    holdout_inputs = scaler.transform(holdout_inputs)
    holdout_losses = [1e10 for _ in range(FLAGS.algo.num_ensemble)]

    # Agent
    agent = learner.create_learner(
        seed=seed,
        observations=dataset["observations"][0],
        actions=dataset["actions"][0],
        obs_actions=train_inputs[data_indices][:, : FLAGS.batch_size],
        **FLAGS.algo,
    )

    epoch = 0
    cnt = 0
    while True:
        epoch += 1
        # Learn
        for step in tqdm(
            range(int(train_size / FLAGS.batch_size) * 10),
            smoothing=0.1,
            desc=f"{epoch}",
        ):
            batch_indices = np.random.randint(
                train_size, size=[FLAGS.algo.num_ensemble, FLAGS.batch_size]
            )
            input_batch, target_batch = (
                train_inputs[batch_indices],
                train_targets[batch_indices],
            )
            batch = {
                "inputs": input_batch,
                "target": target_batch,
            }
            agent, update_info = agent.update(batch)
            if step % FLAGS.log_steps == 0:
                wandb.log(update_info)
        # Validate
        new_holdout_losses = validate(
            agent,
            holdout_inputs[None, :].repeat(FLAGS.algo.num_ensemble, axis=0),
            holdout_targets[None, :].repeat(FLAGS.algo.num_ensemble, axis=0),
        )
        holdout_loss = (np.sort(new_holdout_losses)[: FLAGS.algo.num_ensemble]).mean()
        wandb.log({"holdout_loss": holdout_loss})
        indexes = []
        for i, new_loss, old_loss in zip(
            range(len(holdout_losses)), new_holdout_losses, holdout_losses
        ):
            improvement = (old_loss - new_loss) / old_loss
            if improvement > 0.01:
                indexes.append(i)
                holdout_losses[i] = new_loss
        if len(indexes) > 0:
            cnt = 0
        else:
            cnt += 1

        if cnt >= FLAGS.max_epochs_since_update:
            break

    indexes = select_elites(holdout_losses, FLAGS.num_elites)
    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
        config=FLAGS.algo.to_dict(),
        elites=indexes,
        mu=scaler.mu,
        std=scaler.std,
    )
    fname = os.path.join(FLAGS.save_dir, f"dynamics_params.pkl")
    with open(fname, "wb") as f:
        pickle.dump(save_dict, f, protocol=4)


if __name__ == "__main__":
    app.run(main)
