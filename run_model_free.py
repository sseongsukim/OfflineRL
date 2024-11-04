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

from src.agent import model_free_algos
import flax

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "hopper-medium-v2", 'Environment name.')
flags.DEFINE_string("algo_name", "iql", 'Algorithm name name.')
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
    wandb_config = default_wandb_config()
    wandb_config.update({
        "project": "offlineRL",
        "group": f"{FLAGS.algo_name}",
        "name": f"{FLAGS.algo_name}_{FLAGS.env_name}_{FLAGS.seed}",
        "offline": FLAGS.wandb_offline,
    })
    config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
    learner, algo_config = model_free_algos[FLAGS.algo_name]
    config_flags.DEFINE_config_dict('algo', algo_config, lock_config=False)
    FLAGS.algo["hidden_dims"] = (FLAGS.hidden_size,) * FLAGS.num_layers
    
    env = d4rl_utils.make_env(FLAGS.env_name)
    env.render("rgb_array")
    
    start_time = int(datetime.now().timestamp())
    # Setup wandb
    FLAGS.wandb["name"] += f"_{start_time}"
    setup_wandb(FLAGS.algo.to_dict(), **FLAGS.wandb)
    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f'Saving config to {FLAGS.save_dir}/config.pkl')
        with open(os.path.join(FLAGS.save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(get_flag_dict(), f)

    dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    dataset = d4rl_utils.normalize_dataset(
        env_name= FLAGS.env_name, 
        dataset= dataset,
    )
    example_batch = dataset.sample(1)
    agent = learner(
        seed= FLAGS.seed,
        observations= example_batch["observations"],
        actions= example_batch["actions"],
        max_steps= FLAGS.total_steps,
        **FLAGS.algo
    )
    for step in tqdm(
        range(1, FLAGS.total_steps + 1),
        smoothing= 0.1,
        desc= "training",
    ):
        batch = dataset.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)
        
        if step % FLAGS.eval_steps == 0:
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info, videos = evaluate(
                policy_fn= policy_fn, 
                env= env, 
                num_episodes= FLAGS.num_episodes, 
                num_videos= FLAGS.num_videos
            )
            for k, v in eval_info.items():
                update_info[f"eval/{k}"] = v
                print(f"{k}: {v}")
            for i in range(len(videos)):
                update_info[f"video_{i}"] = wandb.Video(np.array(videos[i]), fps= 15, format= "mp4")

        if step % FLAGS.save_steps == 0 and FLAGS.save_dir is not None:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=FLAGS.algo.to_dict()
            )
            fname = os.path.join(FLAGS.save_dir, f'params_{step}.pkl')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f, protocol= 4)

        if step % FLAGS.log_steps == 0:
            wandb.log(update_info, step=step)

if __name__ == "__main__":
    app.run(main)