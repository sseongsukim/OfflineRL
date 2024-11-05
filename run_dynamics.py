from absl import app, flags
from ml_collections import config_flags
from functools import partial
from typing import List
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, add_to
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from src import d4rl_utils
from collections import defaultdict
import os
from datetime import datetime
from tqdm import tqdm
import wandb
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from src.agent import dynamics as learner
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
flags.DEFINE_integer("max_epochs_since_update", 5, "")

flags.DEFINE_float("holdout_ratio", 0.2, "")
flags.DEFINE_bool("wandb_offline", True, "")

@jax.jit
def validate(agent, inputs, targets):
    mean, _ = agent.dynamics(inputs)
    loss = ((mean - targets) ** 2).mean(axis=(1, 2))
    val_loss = list(loss)
    return val_loss
    
def select_elites(self, metrics: List, num_elites: int) -> List[int]:
    pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
    pairs = sorted(pairs, key=lambda x: x[0])
    elites = [pairs[i][1] for i in range(num_elites)]
    return elites

def shuffle_rows(arr):
    idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxes]

def main(_):
    np.random.seed(FLAGS.seed)
    
    algo_config = learner.get_default_config()
    config_flags.DEFINE_config_dict('algo', algo_config, lock_config=False)
    
    env = d4rl_utils.make_env(FLAGS.env_name)
    env.render("rgb_array")

    # Dataset
    
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
    
    # Agent
    agent = learner.create_learner(
        seed= seed,
        observations= dataset["observations"][0],
        actions= dataset["actions"][0],
        obs_actions= train_inputs[data_indices][:, :FLAGS.batch_size],
        **FLAGS.algo,
    )
    
    epoch = 0
    cnt = 0
    while True:
        epoch += 1
        batch_info = defaultdict(list)
        # Learn
        for _ in range(int(train_size / FLAGS.batch_size) * 2):
            batch_indices = np.random.randint(train_size, size= [FLAGS.num_elites, FLAGS.batch_size])
            input_batch, target_batch = train_inputs[batch_indices], train_targets[batch_indices]
            batch = {
                "inputs": input_batch,
                "target": target_batch,
            }
            agent, update_info = agent.update(batch)
            add_to(batch_info, update_info)

        new_holdout_losses = validate(
            agent, 
            holdout_inputs[None, :].repeat(FLAGS.algo.num_ensemble, axis= 0), 
            holdout_targets[None, :].repeat(FLAGS.algo.num_ensemble, axis= 0),
        )
        holdout_loss = (np.sort(new_holdout_losses)[:FLAGS.num_elites]).mean()
        indexes = []
        for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
            improvement = (old_loss - new_loss) / old_loss
            if improvement > 0.01:
                print(f"improve {i}")
                indexes.append(i)
                holdout_losses[i] = new_loss
        print(indexes)
        print(epoch)
        if len(indexes) > 0:
            cnt = 0
        else:
            cnt += 1

        if cnt >= FLAGS.max_epochs_since_update:
            break
        
    indexes = select_elites(holdout_losses, FLAGS.algo.num_ensemble)
    
if __name__ == "__main__":
    app.run(main)