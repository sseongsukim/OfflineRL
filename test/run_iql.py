from absl import app, flags
from ml_collections import config_flags

import jax.numpy as jnp
import numpy as np
import flax
import jax

from jaxrl_m.wandb import setup_wandb, default_wandb_config
from functools import partial
from jaxrl_m.evaluation import supply_rng

import os
from datetime import datetime
from tqdm import tqdm
from time import time

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", 'Environment name.')
flags.DEFINE_string("save_dir", "log", 'Logging dir (if not None, save params).')
flags.DEFINE_string("run_group", "DEBUG")
flags.DEFINE_integer("eval_episodes", 50, "")
flags.DEFINE_integer("log_interval", 100000, "")
flags.DEFINE_integer("save_interval", 250000, "")

wandb_config = default_wandb_config()
wandb_config.update({
    "project": "offlineRL",
    "group": "IQL",
    "name": "IQL_{env_name}",
})
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)


def main(_):
    start_time = int(datetime.now().timestamp())
    

if __name__ == "__main__":
    app.run(main)