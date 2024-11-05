"""Implementations of algorithms for continuous control."""
from jaxrl_m.typing import *
from jaxrl_m.common import TrainState
from jaxrl_m.networks import EnsembleDyanmics

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

import ml_collections

class DynamicsAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    dynamics: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @jax.jit
    def update(agent, batch: Batch) -> InfoDict:
        def dynamics_loss_fn(dynamics_params):
            mean, logvar = agent.dynamics(batch["inputs"], params= dynamics_params)
            inv_var = jnp.exp(-logvar)
            mse_loss_inv = (jnp.pow(mean - batch["target"], 2) * inv_var).mean(axis= (1, 2))
            var_loss = logvar.mean(axis=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            decay_loss = agent.dynamics(method= "get_total_decay_loss")
            loss = loss + decay_loss
            logvar_loss = agent.config["logvar_loss_coef"] * agent.dynamics(method= "get_max_logvar_sum") - agent.config["logvar_loss_coef"] * agent.dynamics(method= "get_min_logvar_sum")
            loss = loss + logvar_loss
            return loss, {
                "loss": loss,
                "var_loss": var_loss.sum(),
                "mse_loss_inv": mse_loss_inv.sum(),
                "decay_loss": decay_loss,
                "logvar_loss": logvar_loss,
            }
            
        new_dynamics, dynamics_info = agent.dynamics.apply_loss_fn(
            loss_fn= dynamics_loss_fn,
            has_aux= True,
        )
        return agent.replace(dynamics= new_dynamics), {**dynamics_info}

def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    obs_actions: jnp.ndarray,
    lr: float = 3e-4,
    num_ensemble: int = 7,
    hidden_dims: Sequence[int] = (200, 200, 200, 200),
    weight_decays: Sequence[int] = (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4),
    pred_reward: bool = True,
    logvar_loss_coef: float = 0.01,
    **kwargs,
):
    print('Extra kwargs:', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, model_key = jax.random.split(rng, 2)
    
    action_dim = actions.shape[-1]
    obs_dim = observations.shape[-1]
    dynamics_def = EnsembleDyanmics(
        obs_dim= obs_dim,
        action_dim= action_dim,
        hidden_dims= hidden_dims,
        weight_decays= weight_decays,
        num_ensemble= num_ensemble,
        pred_reward= pred_reward,
    )
    dynamics_params = dynamics_def.init(
        model_key, obs_actions,
    )["params"]
    dynamics = TrainState.create(
        dynamics_def,
        dynamics_params,
        tx= optax.adam(learning_rate= lr),
    )
    config = flax.core.FrozenDict(dict(
        num_ensemble= num_ensemble,
        logvar_loss_coef= logvar_loss_coef,
    ))
    return DynamicsAgent(rng, dynamics, config)


def get_default_config():

    config = ml_collections.ConfigDict({
        "lr": 3e-4,
        "num_ensemble": 7,
        "hidden_dims": (200, 200, 200, 200),
        "weight_decays": (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4),
        "pred_reward": True,
        "logvar_loss_coef": 0.01
    })
    return config