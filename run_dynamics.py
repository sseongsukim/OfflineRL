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

import flax

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "hopper-medium-v2", 'Environment name.')
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

flags.DEFINE_bool("wandb_offline", True, "")

def main(_):
    env = d4rl_utils.make_env(FLAGS.env_name)
    env.render("rgb_array")

    start_time = int(datetime.now().timestamp())
    dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    scaler = d4rl_utils.StandardScaler()
    


if __name__ == "__main__":
    app.run(main)