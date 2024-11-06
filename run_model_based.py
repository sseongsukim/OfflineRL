from absl import app, flags
import flax.serialization
from ml_collections import config_flags
from collections import defaultdict
from functools import partial
from typing import List

from jaxrl_m.common import TrainState
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, add_to
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from jaxrl_m.dataset import ReplayBuffer
from src import d4rl_utils
from src.termination_fns import get_termination_fn

import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import flax

import wandb
import pickle

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
flags.DEFINE_integer("num_elites", 7, "")
flags.DEFINE_integer("rollout_freq", 1000, "")
flags.DEFINE_integer("rollout_batch_size", 50000, "")
flags.DEFINE_integer("rollout_length", 1, "")
flags.DEFINE_float("holdout_ratio", 0.2, "")
flags.DEFINE_float("penalty_coef", 2.5, "")
flags.DEFINE_float("real_ratio", 0.05, "")
flags.DEFINE_bool("wandb_offline", False, "")


@jax.jit
def sample_next_obs(
    dynamics: TrainState, 
    obs: np.ndarray,
    actions: np.ndarray,
    num_samples: int,
    elite_index: int,
):
    obs_actions = jnp.concatenate([obs, actions], axis= -1)
    mean, logvar = dynamics.dynamics(obs_actions)
    mean[..., :-1] += obs
    std = jnp.sqrt(jnp.exp(logvar))
    
    mean = mean[elite_index]
    std = std[elite_index]
    

def main(_):
    np.random.seed(FLAGS.seed)
    with open(
        "log/offlineRL/dynamics/dynamics_walker2d-medium-expert-v2_113154_1730852034_20241106_091354/dynamics_params.pkl",
        "rb",
    ) as f:
        dynamics, dynamics_config, elites = pickle.load(f)
    # Env
    env = d4rl_utils.make_env(FLAGS.env_name)
    env.render("rgb_array")
    # Dataset
    dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    scaler = d4rl_utils.StandardScaler()
    # Buffer
    example_transition = {
        "observations": dataset["observations"][0],
        "next_observations": dataset["next_observations"][0],
        "actions": dataset["actions"][0],
        "rewards": dataset["rewards"][0],
        "terminals": dataset["dones_float"][0].astype(np.bool_)
    }
    buffer = ReplayBuffer.create(example_transition, size= 500000)
    # Terminated function
    terminated_fn = get_termination_fn(task= FLAGS.env_name)
    print("debug")

if __name__ == "__main__":
    app.run(main)