import functools
import flax.serialization
from jaxrl_m.typing import *
from typing import List
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field
from jaxrl_m.networks import Policy, EnsembleCritic, EnsembleDyanmics
from src.agent.dynamics import get_default_config as dynamics_config

import flax
import flax.linen as nn


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


class MOBILEAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    dynamics: TrainState
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):
        new_rng, curr_key, sample_key, action_key, target_key = jax.random.split(
            agent.rng, 5
        )

        def critic_loss_fn(critic_params):
            q1, q2 = agent.critic(
                batch["observations"], batch["actions"], params=critic_params
            )

            # Penalty
            mean, logvar = agent.dynamics(
                batch["obs_actions"][None, :].repeat(agent.config["num_ensemble"], 0)
            )
            mean = mean.at[..., :-1].add(
                batch["observations"][None, :].repeat(agent.config["num_ensemble"], 0)
            )
            std = jnp.sqrt(jnp.exp(logvar))

            mean = mean[np.array(agent.config["elites"])]
            std = std[np.array(agent.config["elites"])]

            sample_keys = jax.random.split(sample_key, agent.config["num_samples"])
            samples = jax.vmap(
                lambda key: mean + jax.random.normal(key, shape=std.shape) * std
            )(sample_keys)

            pred_next_observations = samples[..., :-1]
            num_samples, num_ensemble, batch_size, obs_dim = (
                pred_next_observations.shape
            )

            pred_next_observations = pred_next_observations.reshape(-1, obs_dim)
            pred_next_actions, _ = agent.actor(
                pred_next_observations
            ).sample_and_log_prob(seed=action_key)

            pq1, pq2 = agent.target_critic(pred_next_observations, pred_next_actions)
            pq = jnp.minimum(pq1, pq2)
            pq = pq.reshape(num_samples, num_ensemble, batch_size, 1)
            penalty = jnp.mean(pq, axis=0).std(axis=0)
            penalty = penalty.at[: int(batch_size * agent.config["dataset_ratio"])].set(
                0.0
            )

            next_actions, next_log_probs = agent.actor(
                batch["next_observations"]
            ).sample_and_log_prob(seed=target_key)
            nq1, nq2 = agent.target_critic(batch["next_observations"], next_actions)
            next_q = jnp.minimum(nq1, nq2)
            target_q = (
                batch["rewards"] - agent.config["penalty_coef"] * penalty.squeeze()
            ) + agent.config["discount"] * batch["masks"] * next_q
            target_q = jnp.clip(target_q, 0, None)

            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "next_q": next_q.mean(),
                "target_q": target_q.mean(),
                "penalty": penalty.mean(),
                "penalty_q": pq.mean(),
            }

        def actor_loss_fn(actor_params):
            dist = agent.actor(batch["observations"], params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)

            q1, q2 = agent.critic(batch["observations"], actions)
            q = jnp.minimum(q1, q2)

            actor_loss = (log_probs * agent.temp() - q).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -1 * log_probs.mean(),
            }

        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            temp_loss = (temperature * (entropy - target_entropy)).mean()
            return temp_loss, {
                "temp_loss": temp_loss,
                "temperature": temperature,
            }

        new_critic, critic_info = agent.critic.apply_loss_fn(
            loss_fn=critic_loss_fn, has_aux=True
        )
        new_target_critic = target_update(
            agent.critic, agent.target_critic, agent.config["target_update_rate"]
        )
        new_actor, actor_info = agent.actor.apply_loss_fn(
            loss_fn=actor_loss_fn, has_aux=True
        )

        temp_loss_fn = functools.partial(
            temp_loss_fn,
            entropy=actor_info["entropy"],
            target_entropy=agent.config["target_entropy"],
        )
        new_temp, temp_info = agent.temp.apply_loss_fn(
            loss_fn=temp_loss_fn, has_aux=True
        )

        return agent.replace(
            rng=new_rng,
            critic=new_critic,
            target_critic=new_target_critic,
            actor=new_actor,
            temp=new_temp,
        ), {**critic_info, **actor_info, **temp_info}

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
        observations = observations[None, :].repeat(
            agent.config["num_ensemble"], axis=0
        )
        actions = actions[None, :].repeat(agent.config["num_ensemble"], axis=0)
        obs_actions = jnp.concatenate([observations, actions], axis=-1)
        obs_actions = (obs_actions - scaler_mu) / scaler_std
        mean, logvar = agent.dynamics(obs_actions)
        mean = mean.at[..., :-1].add(observations)
        std = jnp.sqrt(jnp.exp(logvar))

        rng, ensemble_key = jax.random.split(agent.rng, 2)
        ensemble_samples = (
            mean + jax.random.normal(ensemble_key, shape=(mean.shape)) * std
        )
        _, batch_size, _ = ensemble_samples.shape
        model_idxs = np.random.choice(agent.config["elites"], size=batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]

        next_obs, reward = samples[..., :-1], samples[..., -1:]

        penalty = jnp.amax(jnp.linalg.norm(std, axis=-1), axis=0)
        penalty = jnp.expand_dims(penalty, 1)
        reward = reward - agent.config["penalty_coef"] * penalty
        return next_obs, reward, penalty


def create_learner(
    dynamics_save_dict,
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    elites: List,
    dataset_ratio: float,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temp_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.005,
    target_entropy: float = None,
    backup_entropy: bool = True,
    penalty_coef: float = 0.5,
    critic_layer_norm: bool = True,
    num_samples: int = 10,
    **kwargs,
):
    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, model_key = jax.random.split(rng, 4)

    action_dim = actions.shape[-1]
    actor_def = Policy(
        hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
        state_dependent_std=False,
        tanh_squash_distribution=True,
    )

    actor_params = actor_def.init(actor_key, observations)["params"]
    actor = TrainState.create(
        actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr)
    )

    critic_def = EnsembleCritic(
        hidden_dims=hidden_dims,
        use_layer_norm=critic_layer_norm,
    )
    critic_params = critic_def.init(critic_key, observations, actions)["params"]
    critic = TrainState.create(
        critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr)
    )
    target_critic = TrainState.create(critic_def, critic_params)

    temp_def = Temperature()
    temp_params = temp_def.init(rng)["params"]
    temp = TrainState.create(
        temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr)
    )

    dynamic_config = dynamics_config()
    obs_dim = observations.shape[-1]
    dynamics_def = EnsembleDyanmics(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=dynamic_config["hidden_dims"],
        weight_decays=dynamic_config["weight_decays"],
        num_ensemble=dynamic_config["num_ensemble"],
        pred_reward=dynamic_config["pred_reward"],
    )
    obs_actions = np.concatenate([observations, actions], axis=-1)
    obs_actions = obs_actions[None, :].repeat(dynamic_config["num_ensemble"], axis=0)
    dynamics_params = dynamics_def.init(
        model_key,
        obs_actions,
    )["params"]
    dynamics = TrainState.create(
        dynamics_def,
        dynamics_params,
        tx=optax.adam(learning_rate=actor_lr),
    )
    dynamics = flax.serialization.from_state_dict(
        dynamics,
        dynamics_save_dict,
    )

    if target_entropy is None:
        target_entropy = -0.5 * action_dim

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            target_update_rate=tau,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            num_ensemble=dynamic_config["num_ensemble"],
            elites=elites,
            penalty_coef=penalty_coef,
            num_samples=num_samples,
            dataset_ratio=dataset_ratio,
        )
    )

    return MOBILEAgent(
        rng=rng,
        dynamics=dynamics,
        critic=critic,
        target_critic=target_critic,
        actor=actor,
        temp=temp,
        config=config,
    )


def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict(
        {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "temp_lr": 3e-4,
            "hidden_dims": (256, 256),
            "discount": 0.99,
            "tau": 0.005,
            "penalty_coef": 0.5,
            "critic_layer_norm": True,
            "num_samples": 10,
            "dataset_ratio": 0.1,
        }
    )
