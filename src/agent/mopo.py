import functools

import flax.serialization
from jaxrl_m.typing import *
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field
from jaxrl_m.networks import Policy, Critic, ensemblize, EnsembleDyanmics
from src.agent.dynamics import get_default_config as dynamics_config


import flax
import flax.linen as nn

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp',
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature))
        )
        return jnp.exp(log_temp)

class MOPOAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    dynamics: TrainState
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def critic_loss_fn(critic_params):
            next_dist = agent.actor(batch['next_observations'])
            next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_key)

            next_q1, next_q2 = agent.target_critic(batch['next_observations'], next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q

            if agent.config['backup_entropy']:
                target_q = target_q - agent.config['discount'] * batch['masks'] * next_log_probs * agent.temp()
            
            q1, q2 = agent.critic(batch['observations'], batch['actions'], params=critic_params)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            
            return critic_loss, {
                'critic_loss': critic_loss,
                'q1': q1.mean(),
            }        

        def actor_loss_fn(actor_params):
            dist = agent.actor(batch['observations'], params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)
            
            q1, q2 = agent.critic(batch['observations'], actions)
            q = jnp.minimum(q1, q2)

            actor_loss = (log_probs * agent.temp() - q).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -1 * log_probs.mean(),
            }
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            temp_loss = (temperature * (entropy - target_entropy)).mean()
            return temp_loss, {
                'temp_loss': temp_loss,
                'temperature': temperature,
            }
        
        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['target_update_rate'])
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        temp_loss_fn = functools.partial(temp_loss_fn, entropy=actor_info['entropy'], target_entropy=agent.config['target_entropy'])
        new_temp, temp_info = agent.temp.apply_loss_fn(loss_fn=temp_loss_fn, has_aux=True)

        return agent.replace(rng=new_rng, critic=new_critic, target_critic=new_target_critic, actor=new_actor, temp=new_temp), {
            **critic_info, **actor_info, **temp_info}

    @jax.jit
    def sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def dynamics_step(
        agent,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        seed: PRNGKey,
        scaler_mu: np.ndarray,
        scaler_std: np.ndarray,
    ):
        observations = observations[None, :].repeat(agent.config["num_ensemble"], axis= 0)
        actions = actions[None, :].repeat(agent.config["num_ensemble"], axis= 0)
        obs_actions = jnp.concatenate([observations, actions], axis= -1)
        obs_actions = (obs_actions - scaler_mu) / scaler_std
        mean, logvar = agent.dynamics(obs_actions)
        mean = mean.at[..., :-1].add(observations)
        std = jnp.sqrt(jnp.exp(logvar))
        
        rng, ensemble_key = jax.random.split(agent.rng, 2)
        ensemble_samples = (mean + jax.random.normal(ensemble_key, shape= (mean.shape)) * std)
        _, batch_size, _ = ensemble_samples.shape
        model_idxs = np.random.choice(agent.config["elites"], size=batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs, reward = samples[..., :-1], samples[..., -1:]
        
        penalty = jnp.amax(jnp.linalg.norm(std, axis= -1), axis= 0)
        penalty = jnp.expand_dims(penalty, 1)
        reward = reward - agent.config["penalty_coef"] * penalty
        return next_obs, reward
        

def create_learner(
    dynamics_save_dict,
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    elites: List,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temp_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.005,
    target_entropy: float = None,
    backup_entropy: bool = True,
    penalty_coef: float = 2.5,
    **kwargs
):
        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, model_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = Policy(
            hidden_dims, 
            action_dim=action_dim, 
            log_std_min=-10.0, 
            state_dependent_std=True, 
            tanh_squash_distribution=True, 
            final_fc_init_scale=1.0
        )

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr))

        critic_def = ensemblize(Critic, num_qs=2)(hidden_dims)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
        target_critic = TrainState.create(critic_def, critic_params)

        temp_def = Temperature()
        temp_params = temp_def.init(rng)['params']
        temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))

        dynamic_config = dynamics_config()
        
        obs_dim = observations.shape[-1]
        dynamics_def = EnsembleDyanmics(
            obs_dim= obs_dim,
            action_dim= action_dim,
            hidden_dims= dynamic_config["hidden_dims"],
            weight_decays= dynamic_config["weight_decays"],
            num_ensemble= dynamic_config["num_ensemble"],
            pred_reward= dynamic_config["pred_reward"],
        )
        obs_actions = np.concatenate([observations, actions], axis= -1)
        obs_actions = obs_actions[None, :].repeat(dynamic_config["num_ensemble"], axis= 0)
        dynamics_params = dynamics_def.init(
            model_key, obs_actions,
        )["params"]
        dynamics = TrainState.create(
            dynamics_def,
            dynamics_params,
            tx= optax.adam(learning_rate= actor_lr),
        )
        dynamics = flax.serialization.from_state_dict(
            dynamics, dynamics_save_dict,
        )
        
        if target_entropy is None:
            target_entropy = -0.5 * action_dim

        config = flax.core.FrozenDict(dict(
            discount=discount,
            target_update_rate=tau,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,    
            num_ensemble= dynamic_config["num_ensemble"],
            elites= elites,
            penalty_coef= penalty_coef,  
        ))
        return MOPOAgent(
            rng= rng, 
            dynamics= dynamics, 
            critic= critic, 
            target_critic= target_critic, 
            actor= actor, 
            temp= temp, 
            config= config,
        )

def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict({
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'temp_lr': 3e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        'tau': 0.005,
        'target_entropy': ml_collections.config_dict.placeholder(float),
        'backup_entropy': True,
        'penalty_coef': 2.5
    })