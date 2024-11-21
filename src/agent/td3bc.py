from jaxrl_m.typing import *
from jaxrl_m.common import TrainState, target_update, nonpytree_field
from jaxrl_m.networks import Policy, EnsembleCritic, ImplicitPolicy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

import ml_collections
import jax
import jax.numpy as jnp
import flax
import numpy as np


class TD3BCAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    config: Dict[str, Any] = nonpytree_field()
    step: int = 0

    @jax.jit
    def update(agent, batch: Batch):
        new_rng, critic_key, actor_key = jax.random.split(agent.rng, 3)

        def critic_loss_fn(critic_params):
            noise = (
                jax.random.normal(critic_key, batch["actions"].shape)
                * agent.config["policy_noise"]
            )
            noise = jnp.clip(
                noise, -agent.config["noise_clip"], agent.config["noise_clip"]
            )
            next_actions = agent.actor(batch["next_observations"]) + noise
            next_actions = jnp.clip(next_actions, -1, 1)

            target_q1, target_q2 = agent.target_critic(
                batch["next_observations"], next_actions
            )
            target_q = jnp.minimum(target_q1, target_q2)
            target_q = (
                batch["rewards"] + agent.config["discount"] * batch["masks"] * target_q
            )

            current_q1, current_q2 = agent.critic(
                batch["observations"], batch["actions"], params=critic_params
            )
            critic_loss = (
                (current_q1 - target_q) ** 2 + (current_q2 - target_q) ** 2
            ).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": current_q1.mean(),
            }

        def actor_loss_fn(actor_params):
            actions = agent.actor(batch["observations"], params=actor_params)
            q1, q2 = agent.critic(batch["observations"], actions)
            q = (q1 + q2) / 2
            bc_loss = ((actions - batch["actions"]) ** 2).mean()
            lamd = agent.config["alpha"] / jnp.abs(q).mean()
            actor_loss = -q.mean() * lamd + bc_loss
            return actor_loss, {
                "actor_loss": actor_loss,
                "bc_loss": bc_loss,
            }

        new_critic, critic_info = agent.critic.apply_loss_fn(
            loss_fn=critic_loss_fn, has_aux=True
        )
        new_target_critic = target_update(
            new_critic, agent.target_critic, agent.config["target_update_rate"]
        )

        def update_actor():
            new_actor, actor_info = agent.actor.apply_loss_fn(
                loss_fn=actor_loss_fn, has_aux=True
            )
            return new_actor, actor_info

        should_update_actor = agent.step % agent.config["update_freq"] == 0
        new_actor, actor_info = jax.lax.cond(
            should_update_actor,
            update_actor,
            lambda: (agent.actor, {"actor_loss": 0.0, "bc_loss": 0.0}),
        )
        new_step = agent.step + 1

        return agent.replace(
            rng=new_rng,
            critic=new_critic,
            target_critic=new_target_critic,
            actor=new_actor,
            step=new_step,
        ), {**critic_info, **actor_info}

    @jax.jit
    def sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 0.1,
    ) -> jnp.ndarray:
        actions = agent.actor(observations)
        actions = jnp.clip(actions, -1, 1)
        return actions


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    policy_noise: float = 0.2,
    hidden_dims: Sequence[int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.001,
    alpha: float = 2.5,
    noise_clip: float = 0.5,
    critic_layer_norm: bool = True,
    max_steps: Optional[int] = None,
    update_freq: int = 2,
    **kwargs,
):
    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    action_dim = actions.shape[-1]
    actor_def = ImplicitPolicy(
        hidden_dims=hidden_dims,
        action_dim=action_dim,
    )
    schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
    actor_tx = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule_fn),
    )
    actor_params = actor_def.init(actor_key, observations)["params"]
    actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

    critic_def = EnsembleCritic(
        hidden_dims=hidden_dims,
        use_layer_norm=critic_layer_norm,
    )
    critic_params = critic_def.init(critic_key, observations, actions)["params"]
    critic = TrainState.create(
        critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr)
    )
    target_critic = TrainState.create(critic_def, critic_params)

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            target_update_rate=tau,
            noise_clip=noise_clip,
            alpha=alpha,
            policy_noise=policy_noise,
            update_freq=update_freq,
        )
    )

    return TD3BCAgent(
        rng,
        critic=critic,
        target_critic=target_critic,
        actor=actor,
        config=config,
        step=1,
    )


def get_default_config():
    config = ml_collections.ConfigDict(
        {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "policy_noise": 0.2,
            "hidden_dims": (256, 256),
            "discount": 0.99,
            "tau": 0.005,
            "alpha": 2.5,
            "noise_clip": 0.5,
            "update_freq": 2,
            "critic_layer_norm": True,
        }
    )
    return config
