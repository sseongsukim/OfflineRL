from typing import Dict
import jax
import gym
import numpy as np
from functools import partial
from collections import defaultdict
import time
import jax.numpy as jnp

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def evaluate(
    policy_fn, 
    env: gym.Env, 
    num_episodes: int, 
    num_videos: int
):
    """
    Evaluates a policy in an environment by running it for some number of episodes,
    and returns average statistics for metrics in the environment's info dict.

    If you wish to log environment returns, you can use the EpisodeMonitor wrapper (see below).

    Arguments:
        policy_fn: A function that takes an observation and returns an action.
            (if your policy needs JAX RNG keys, use supply_rng to supply a random key)
        env: The environment to evaluate in.
        num_episodes: The number of episodes to run for.
    Returns:
        A dictionary of average statistics for metrics in the environment's info dict.

    """
    stats = defaultdict(list)
    videos = []
    for i in range(num_episodes + num_videos):
        observation, done = env.reset(), False
        step = 0
        if i >= num_episodes:
            frames = []
        while not done:
            action = policy_fn(observation)
            observation, _, done, info = env.step(action)
            if i >= num_episodes:
                size = 208
                img = env.render("rgb_array", width= size, height= size).transpose(2, 0, 1).copy()
                frames.append(img)
            add_to(stats, flatten(info))
            step += 1
        add_to(stats, flatten(info, parent_key="final"))
        if i >= num_episodes:
            videos.append(np.array(frames))
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, videos


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()