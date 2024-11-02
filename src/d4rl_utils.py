import d4rl
import gym
import numpy as np

from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import EpisodeMonitor

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def get_dataset(env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        imputed_next_observations = np.roll(dataset['observations'], -1, axis=0)
        same_obs = np.all(np.isclose(imputed_next_observations, dataset['next_observations'], atol=1e-5), axis=-1)
        dones_float = 1.0 - same_obs.astype(np.float32)
        dones_float[-1] = 1
        
        dataset = {
            'observations': dataset['observations'],
            'actions': dataset['actions'],
            'rewards': dataset['rewards'],
            'masks': 1.0 - dataset['terminals'],
            'dones_float': dones_float,
            'next_observations': dataset['next_observations'],
        }
        dataset = {k: v.astype(np.float32) for k, v in dataset.items()}
        return Dataset(dataset)

def get_normalization(dataset):
        returns = []
        ret = 0
        for r, term in zip(dataset['rewards'], dataset['dones_float']):
            ret += r
            if term:
                returns.append(ret)
                ret = 0
        return (max(returns) - min(returns)) / 1000

def normalize_dataset(env_name, dataset):
    normalizing_factor = get_normalization(dataset)
    dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
    return dataset


class StandardScaler(object):
    
    def __init__(self, mu= None, std= None):
        self.mu = mu
        self.std = std
    
    def fit(self, batch):
        self.mu = np.mean(batch, axis=0, keepdims= True)
        self.std = np.std(batch, axis=0, keepdims= True)
        self.std[self.std < 1e-12] = 1.0
    
    def transform(self, batch):
        return (batch - self.mu) / self.std
    
    def inverse_transform(self, batch):
        return self.std * batch + self.mu