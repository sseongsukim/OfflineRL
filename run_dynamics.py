from absl import app, flags
from ml_collections import config_flags
from functools import partial

from jaxrl_m.evaluation import supply_rng, evaluate
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from src import d4rl_utils

import os
from datetime import datetime
from tqdm import tqdm
import wandb
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "walker2d-medium-expert-v2", 'Environment name.')
flags.DEFINE_string("save_dir", "log", 'Logging dir (if not None, save params).')
flags.DEFINE_string("run_group", "DEBUG", "")
flags.DEFINE_integer("num_episodes", 50, "")
flags.DEFINE_integer("num_videos", 2, "")
flags.DEFINE_integer("log_steps", 1000, "")
flags.DEFINE_integer("eval_steps", 100000, "")
flags.DEFINE_integer("save_steps", 250000, "")
flags.DEFINE_integer("total_steps", 1000000, "")

seed = np.random.randint(low= 0, high= 10000000)
flags.DEFINE_integer("seed", seed, "")
flags.DEFINE_integer("batch_size", 512, "")
flags.DEFINE_integer("hidden_size", 256, "")
flags.DEFINE_integer("num_layers", 2, "")
flags.DEFINE_integer("num_elites", 7, "")
flags.DEFINE_float("holdout_ratio", 0.2, "")

flags.DEFINE_bool("wandb_offline", True, "")

def shuffle_rows(arr):
    idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxes]

def main(_):
    np.random.seed(FLAGS.seed)
    
    env = d4rl_utils.make_env(FLAGS.env_name)
    env.render("rgb_array")

    start_time = int(datetime.now().timestamp())
    dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    scaler = d4rl_utils.StandardScaler()
    
    inputs = np.concatenate([dataset["observations"], dataset["actions"]], axis= -1)
    targets = np.concatenate([
        dataset["next_observations"] - dataset["observations"], 
        dataset["rewards"].reshape(-1, 1)
    ], axis= -1)
    dataset_size = inputs.shape[0]
    holdout_size = min(int(dataset_size * FLAGS.holdout_ratio), 1000)
    train_size = dataset_size - holdout_size
    train_split_indices = np.random.permutation(range(dataset_size))
    train_inputs, train_targets = inputs[train_split_indices[:train_size]], targets[train_split_indices[:train_size]]
    holdout_inputs, holdout_targets = inputs[train_split_indices[train_size:]], targets[train_split_indices[train_size:]]
    data_indices = np.random.randint(
        train_size, size= [FLAGS.num_elites, train_size],
    )
    
    scaler.fit(train_inputs)
    train_inputs = scaler.transform(train_inputs)
    holdout_inputs = scaler.transform(holdout_inputs)
    holdout_losses = [1e10 for _ in range(FLAGS.num_elites)]

    print("debug")


if __name__ == "__main__":
    app.run(main)