"""Implementations of algorithms for continuous control."""
import flax.struct
from jaxrl_m.typing import *
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, EnsembleCritic, ValueCritic
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.linen as nn

import ml_collections

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp',
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature))
        )
        return jnp.exp(log_temp)

class CQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = flax.struct.field(pytree_node= False)
    
    @jax.jit
    def update(agent, batch: Batch):
        new_rng, curr_key, next_key, cql_key = jax.random.split(agent.rng, 4)

        def critic_loss_fn(critic_params):

            next_dist = agent.actor(batch['next_observations'])
            next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_key)

            next_q1, next_q2 = agent.target_critic(batch['next_observations'], next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q

            q1, q2 = agent.critic(batch['observations'], batch['actions'], params=critic_params)
            td_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            alpha = agent.config.get('cql_alpha', 1.0)

            policy_dist = agent.actor(batch['observations'])
            policy_actions, _ = policy_dist.sample_and_log_prob(seed=curr_key)

            random_actions = jax.random.uniform(
                cql_key, shape=batch['actions'].shape, minval=-1.0, maxval=1.0
            )
            q1_policy, q2_policy = agent.critic(batch['observations'], policy_actions, params=critic_params)
            q1_random, q2_random = agent.critic(batch['observations'], random_actions, params=critic_params)

            ood_q1 = jnp.concatenate([q1_policy, q1_random], axis=0)
            ood_q2 = jnp.concatenate([q2_policy, q2_random], axis=0)

            logsumexp_q1 = jax.scipy.special.logsumexp(ood_q1)
            logsumexp_q2 = jax.scipy.special.logsumexp(ood_q2)

            cql_loss1 = (logsumexp_q1 - q1.mean()) * alpha
            cql_loss2 = (logsumexp_q2 - q2.mean()) * alpha
            cql_loss = cql_loss1 + cql_loss2


            critic_loss = td_loss + cql_loss

            return critic_loss, {
                'critic_loss': critic_loss,
                'td_loss': td_loss,
                'cql_loss': cql_loss,
                'q1_mean': q1.mean(),
                'q2_mean': q2.mean(),
            }

        def actor_loss_fn(actor_params):
            dist = agent.actor(batch['observations'], params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)
            
            q1, q2 = agent.critic(batch['observations'], actions)
            q = jnp.minimum(q1, q2)

            actor_loss = (log_probs * agent.temp() - q).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -log_probs.mean(),
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

        temp_loss_fn_partial = functools.partial(
            temp_loss_fn, 
            entropy=actor_info['entropy'], 
            target_entropy=agent.config['target_entropy'],
        )
        new_temp, temp_info = agent.temp.apply_loss_fn(loss_fn=temp_loss_fn_partial, has_aux=True)

        return agent.replace(
            rng=new_rng,
            critic=new_critic,
            target_critic=new_target_critic,
            actor=new_actor,
            temp=new_temp
        ), {**critic_info, **actor_info, **temp_info}

    @jax.jit
    def sample_actions(agent, observations: np.ndarray, *, seed: PRNGKey, temperature: float = 1.0) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed= seed)
        actions = jnp.clip(actions, -1, 1)
        return actions
    
def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    max_steps: int = None,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temp_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (512, 512),
    discount: float = 0.99,
    tau: float = 0.005,
    target_entropy: float = None,
    backup_entropy: bool = True,
    critic_layer_norm: bool = True,
    **kwargs
):

    print('Extra kwargs:', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    action_dim = actions.shape[-1]
    actor_def = Policy(
        hidden_dims, 
        action_dim=action_dim, 
        log_std_min=-5.0, 
        state_dependent_std= False, 
        tanh_squash_distribution= True, 
    )
    schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
    actor_tx = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule_fn)
    )

    actor_params = actor_def.init(actor_key, observations)['params']
    actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

    critic_def = EnsembleCritic(
        hidden_dims= hidden_dims,
        use_layer_norm= critic_layer_norm,
    )
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
    target_critic = TrainState.create(critic_def, critic_params)

    temp_def = Temperature()
    temp_params = temp_def.init(rng)['params']
    temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))

    if target_entropy is None:
        target_entropy = -0.5 * action_dim

    config = flax.core.FrozenDict(dict(
        discount=discount,
        target_update_rate=tau,
        target_entropy=target_entropy,
        backup_entropy=backup_entropy,            
    ))
    
    return CQLAgent(rng, critic=critic, target_critic=target_critic, actor=actor, temp=temp, config=config)

def get_default_config():

    config = ml_collections.ConfigDict({
        'actor_lr': 3e-4,
        'value_lr': 3e-4,
        'critic_lr': 3e-4,
        'hidden_dims': (512, 512),
        'discount': 0.99,
        'critic_layer_norm': True,
        'tau': 0.005,
    })
    return config