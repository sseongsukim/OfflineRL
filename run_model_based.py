from absl import app, flags
import flax.serialization
from ml_collections import config_flags
from collections import defaultdict
from functools import partial
from typing import List

from jaxrl_m.evaluation import supply_rng, evaluate
from jaxrl_m.wandb import setup_wandb, default_wandb_config
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
flags.DEFINE_string("env_name", "walker2d-medium-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "log", "Logging dir (if not None, save params).")
flags.DEFINE_string("algo_name", "mobile", "")
flags.DEFINE_string("run_group", "DEBUG", "")
flags.DEFINE_integer("num_episodes", 50, "")
flags.DEFINE_integer("num_videos", 2, "")
flags.DEFINE_integer("epoch", 2000, "")
flags.DEFINE_integer("step_per_epoch", 1000, "")
flags.DEFINE_integer("eval_steps", 100, "")
flags.DEFINE_integer("log_steps", 10, "")
seed = np.random.randint(low=0, high=10000000)
flags.DEFINE_integer("seed", seed, "")
flags.DEFINE_integer("batch_size", 1024, "")
flags.DEFINE_integer("rollout_freq", 1000, "")
flags.DEFINE_integer("rollout_batch_size", 50000, "")
flags.DEFINE_integer("rollout_length", 1, "")
flags.DEFINE_integer("dynamics_update_freq", 0, "")
flags.DEFINE_float("penalty_coef", 0.5, "")
flags.DEFINE_float("adv_weights", 3e-4, "")
flags.DEFINE_float("dataset_ratio", 0.15, "")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": "offlineRL",
        "group": "{algo_name}",
        "name": "{algo_name}_{env_name}_{seed}",
    }
)
config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)


@jax.jit
def agent_actions(agent, observations):
    rng, action_key = jax.random.split(agent.rng, 2)
    actions = agent.actor(observations).sample(seed=action_key)
    actions = jnp.clip(actions, -1.0, 1.0)
    return actions


def rollout(
    policy_fn,
    dynamics_fn,
    terminated_fn,
    init_obs: np.ndarray,
    rollout_length: int,
):
    num_transitions = 0
    rewards_arr = np.array([])
    penalty_arr = np.array([])
    rollout_transitions = defaultdict(list)
    observations = init_obs
    for _ in range(rollout_length):
        actions = policy_fn(observations)
        next_observations, rewards, penalty = dynamics_fn(observations, actions)
        terminals = terminated_fn(observations, actions, next_observations)
        rollout_transitions["observations"].append(observations)
        rollout_transitions["next_observations"].append(next_observations)
        rollout_transitions["actions"].append(actions)
        rollout_transitions["rewards"].append(rewards.squeeze())
        rollout_transitions["dones_float"].append(terminals.squeeze().astype(np.int32))
        masks = 1.0 - terminals.squeeze().astype(np.int32)
        rollout_transitions["masks"].append(masks)

        num_transitions += len(observations)
        rewards_arr = np.append(rewards_arr, rewards.flatten())
        penalty_arr = np.append(penalty_arr, penalty.flatten())
        nonterm_mask = (~terminals).flatten()
        if nonterm_mask.sum() == 0:
            break
        observations = next_observations[nonterm_mask]

    transitions = {}
    for k, v in rollout_transitions.items():
        transitions[k] = np.concatenate(v, axis=0)

    info = {
        "num_transitions": num_transitions,
        "reward_mean": rewards_arr.mean(),
        "penalty_mean": penalty_arr.mean(),
    }
    return transitions, info


def main(_):
    np.random.seed(FLAGS.seed)
    # Env
    env = d4rl_utils.make_env(FLAGS.env_name)
    env.render("rgb_array")

    with open(
        "log/offlineRL/dynamics/dynamics_walker2d-medium-expert-v2_4183302_1730907381_20241107_003621/dynamics_params.pkl",
        "rb",
    ) as f:
        save_dict = pickle.load(f)

    from src.agent import model_based_algos

    learner, algo_config = model_based_algos[FLAGS.algo_name]
    config_flags.DEFINE_config_dict("algo", algo_config, lock_config=False)
    FLAGS.algo.penalty_coef = FLAGS.penalty_coef
    if "dynamics_update_freq" in FLAGS.algo.to_dict().keys():
        FLAGS.dynamics_update_freq = FLAGS.algo.dynamics_update_freq
        FLAGS.algo.adv_weights = FLAGS.adv_weights
    if "dataset_ratio" in FLAGS.algo.to_dict().keys():
        FLAGS.algo.dataset_ratio = FLAGS.dataset_ratio

    start_time = int(datetime.now().timestamp())
    FLAGS.wandb["name"] += f"_{start_time}"
    setup_wandb({**FLAGS.algo.to_dict(), **save_dict["config"]}, **FLAGS.wandb)
    # Dataset
    dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    scaler = d4rl_utils.StandardScaler()
    scaler.mu = save_dict["mu"]
    scaler.std = save_dict["std"]
    # Buffer
    example_transition = {
        "observations": dataset["observations"][0],
        "next_observations": dataset["next_observations"][0],
        "actions": dataset["actions"][0],
        "rewards": dataset["rewards"][0],
        "dones_float": dataset["dones_float"][0],
        "masks": 1.0 - dataset["dones_float"][0],
    }
    buffer = ReplayBuffer.create(example_transition, size=5000000)
    terminated_fn = get_termination_fn(task=FLAGS.env_name)

    agent = learner(
        dynamics_save_dict=save_dict["agent"]["dynamics"],
        seed=FLAGS.seed,
        observations=dataset["observations"][: FLAGS.batch_size],
        actions=dataset["actions"][: FLAGS.batch_size],
        elites=save_dict["elites"],
        env_name=FLAGS.env_name,
        **FLAGS.algo,
    )
    num_timesteps = 0
    for epoch in tqdm(range(1, FLAGS.epoch + 1), smoothing=0.1, desc="epoch"):

        for step in tqdm(range(FLAGS.step_per_epoch)):
            # MBRL
            if step % FLAGS.rollout_freq == 0:
                init_obs = dataset.sample(FLAGS.rollout_batch_size)["observations"]
                rollout_transition, rollout_info = rollout(
                    policy_fn=partial(supply_rng(agent.sample_actions)),
                    dynamics_fn=partial(
                        supply_rng(agent.dynamics_step),
                        scaler_mu=scaler.mu,
                        scaler_std=scaler.std,
                    ),
                    terminated_fn=terminated_fn,
                    init_obs=init_obs,
                    rollout_length=FLAGS.rollout_length,
                )
                buffer.add_batch_transitions(rollout_transition)
            # Train agent
            dataset_sample_size = int(FLAGS.batch_size * FLAGS.dataset_ratio)
            dataset_batch = dataset.sample(dataset_sample_size)
            buffer_sample_size = FLAGS.batch_size - dataset_sample_size
            buffer_batch = buffer.sample(buffer_sample_size)
            batch = {
                k: np.concatenate([dataset_batch[k], buffer_batch[k]], axis=0)
                for k in dataset_batch.keys()
            }
            if FLAGS.algo_name == "mobile":
                batch["obs_actions"] = scaler.transform(
                    np.concatenate(
                        [
                            batch["observations"],
                            batch["actions"],
                        ],
                        axis=-1,
                    )
                )
            agent, update_info = agent.update(batch)
            for k, v in rollout_info.items():
                update_info[f"rollout/{k}"] = v

            # Rambo
            if (
                FLAGS.dynamics_update_freq > 0
                and (num_timesteps + 1) % FLAGS.dynamics_update_freq == 0
            ):
                finetune_step = 0
                while finetune_step < FLAGS.algo.adv_train_steps:
                    adv_observations = dataset.sample(FLAGS.algo.adv_batch_size)[
                        "observations"
                    ]
                    for _ in range(FLAGS.rollout_length):
                        adv_actions = agent_actions(agent, adv_observations)
                        adv_batch = dataset.sample(FLAGS.algo.adv_batch_size)
                        adv_batch["adv_observations"] = adv_observations.copy()
                        adv_batch["adv_actions"] = adv_actions.copy()
                        adv_batch["adv_obs_actions"] = scaler.transform(
                            np.concatenate(
                                [
                                    adv_batch["adv_observations"],
                                    adv_batch["adv_actions"],
                                ],
                                axis=-1,
                            )
                        )
                        adv_batch["inputs"] = scaler.transform(
                            np.concatenate(
                                [
                                    adv_batch["observations"],
                                    adv_batch["actions"],
                                ],
                                axis=-1,
                            )
                        )
                        adv_batch["targets"] = np.concatenate(
                            [
                                adv_batch["next_observations"]
                                - adv_batch["observations"],
                                adv_batch["rewards"].reshape(-1, 1),
                            ],
                            axis=-1,
                        )
                        agent, finetune_info = agent.finetune_dynamics(adv_batch)
                        finetune_step += 1
                        adv_observations = finetune_info["next_observations"]
                        if finetune_step == 1000:
                            break
            num_timesteps += 1

        if epoch % FLAGS.eval_steps == 0:
            eval_info, videos = evaluate(
                policy_fn=partial(supply_rng(agent.sample_actions), temperature=0.0),
                env=env,
                num_episodes=FLAGS.num_episodes,
                num_videos=FLAGS.num_videos,
            )
            for k, v in eval_info.items():
                update_info[f"eval/{k}"] = v
            for video_num in range(len(videos)):
                update_info[f"video_{video_num}"] = wandb.Video(
                    np.array(videos[video_num]), fps=30, format="mp4"
                )

        if epoch % FLAGS.log_steps == 0:
            wandb.log(update_info, step=epoch)


if __name__ == "__main__":
    app.run(main)
