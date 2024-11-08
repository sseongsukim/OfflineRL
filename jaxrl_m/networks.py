"""Common networks used in RL.

This file contains nn.Module definitions for common networks used in RL. It is divided into three sets:

1) Common Networks: MLP
2) Common RL Networks:
    For discrete action spaces: DiscreteCritic is a Q-function
    For continuous action spaces: Critic, ValueCritic, and Policy provide the Q-function, value function, and policy respectively.
    For ensembling: ensemblize() provides a wrapper for creating ensembles of networks (e.g. for min-Q / double-Q)
3) Meta Networks for vision tasks:
    WithEncoder: Combines a fully connected network with an encoder network (encoder may come from jaxrl_m.vision)
    ActorCritic: Same as WithEncoder, but for possibly many different networks (e.g. actor, critic, value)
"""

from jaxrl_m.typing import *

import flax.linen as nn
import jax.numpy as jnp

import distrax
import flax.linen as nn
import jax.numpy as jnp

###############################
#
#  Common Networks
#
###############################


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x

class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x

###############################
#
#
#  Common RL Networks
#
###############################

class DiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP((*self.hidden_dims, self.n_actions), activations=self.activations)(
            observations
        )

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)

def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )

class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)

class Critic(nn.Module):
    hidden_dims: tuple = (256, 256)
    use_layer_norm: bool = False
    activate_final: bool = False
    
    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], -1)
        if self.use_layer_norm:
            module = LayerNormMLP
        else:
            module = MLP
        q = module(
            (*self.hidden_dims, 1), 
            activate_final=self.activate_final, 
            activations=nn.gelu
        )(x).squeeze(-1)
        return q

class EnsembleCritic(nn.Module):
    hidden_dims: tuple = (256, 256)
    use_layer_norm: bool = False
    activate_final: bool = False
    ensemble_size: int = 2
    
    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], -1)
        if self.use_layer_norm:
            module = LayerNormMLP
        else:
            module = MLP
        module = ensemblize(module, self.ensemble_size)
        q1, q2 = module(
            (*self.hidden_dims, 1), 
            activate_final=self.activate_final, 
            activations=nn.relu,
        )(x).squeeze(-1)
        return q1, q2

class ImplicitPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    
    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
    ):
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)
        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        actions = jnp.tanh(means)
        return actions

class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2.0
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, plan: bool = False,
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        if plan:
            return means, jnp.exp(log_stds)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        return distribution

class DiscretePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
            self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        logits = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution

class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


def softplus(x):
    return jnp.logaddexp(x, 0)


def soft_clamp(x, _min, _max):
    x = _max - softplus(_max - x)
    x = _min + softplus(x - _min)
    return x

class EnsembleLinear(nn.Module):
    input_dim: int
    output_dim: int
    num_ensemble: int
    weight_decay: float

    def setup(self):
        self.weight = self.param(
            "kernel",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, self.input_dim, self.output_dim),
        )
        self.bias = self.param(
            "bias",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, 1, self.output_dim),
        )

    def __call__(self, x: jnp.ndarray):
        x = jnp.einsum("nbi,nij->nbj", x, self.weight)
        x = x + self.bias
        return x
    
    def get_decay_loss(self):
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss

class EnsembleDyanmics(nn.Module):
    obs_dim: int
    action_dim: int
    hidden_dims: Sequence[int]
    weight_decays: Sequence[int]
    num_ensemble: int
    pred_reward: bool
    
    def setup(self):
        hidden_dims = [self.obs_dim + self.action_dim] + list(self.hidden_dims)
        self.layers = [EnsembleLinear(
            input_dim= input_dim,
            output_dim= output_dim,
            num_ensemble= self.num_ensemble,
            weight_decay= weight_decay,
        ) for input_dim, output_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], self.weight_decays)]
        output_dim = self.obs_dim + 1 if self.pred_reward else self.obs_dim
        self.final_layers = EnsembleLinear(
            input_dim= hidden_dims[-1],
            output_dim= output_dim * 2,
            num_ensemble= self.num_ensemble,
            weight_decay= self.weight_decays[-1],
        )
        self.min_logvar = self.param(
            "min_logvar", nn.initializers.constant(-10.0), (output_dim,)
        )
        self.max_logvar = self.param(
            "max_logvar", nn.initializers.constant(0.5), (output_dim,)
        )
        self.output_dim = output_dim
    
    def __call__(self, obs_action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = obs_action
        for layer in self.layers:
            x = layer(x)
            x = nn.swish(x)
        x = self.final_layers(x)
        mean, logvar = x[:, :, :self.output_dim], x[:, :, self.output_dim:]
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar 
    
    def get_max_logvar_sum(self):
        return self.max_logvar.sum()
    
    def get_min_logvar_sum(self):
        return self.min_logvar.sum()
    
    def get_total_decay_loss(self):
        decay_loss = 0
        for layer in self.layers:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.final_layers.get_decay_loss()
        return decay_loss
###############################
#
#
#   Meta Networks for Encoders
#
###############################
def get_latent(
    encoder: nn.Module, observations: Union[jnp.ndarray, Dict[str, jnp.ndarray]]
):
    """

    Get latent representation from encoder. If observations is a dict
        a state and image component, then concatenate the latents.

    """
    if encoder is None:
        return observations

    elif isinstance(observations, dict):
        return jnp.concatenate(
            [encoder(observations["image"]), observations["state"]], axis=-1
        )

    else:
        return encoder(observations)


class WithEncoder(nn.Module):
    encoder: nn.Module
    network: nn.Module

    def __call__(self, observations, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)


class ActorCritic(nn.Module):
    """Combines FC networks with encoders for actor, critic, and value.

    Note: You can share encoder parameters between actor and critic by passing in the same encoder definition for both.

    Example:

        encoder_def = ImpalaEncoder()
        actor_def = Policy(...)
        critic_def = Critic(...)
        # This will share the encoder between actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': encoder_def},
            networks={'actor': actor_def, 'critic': critic_def}
        )
        # This will have separate encoders for actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': copy.deepcopy(encoder_def)},
            networks={'actor': actor_def, 'critic': critic_def}
        )
    """

    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def actor(self, observations, **kwargs):
        latents = get_latent(self.encoders["actor"], observations)
        return self.networks["actor"](latents, **kwargs)

    def critic(self, observations, actions, **kwargs):
        latents = get_latent(self.encoders["critic"], observations)
        return self.networks["critic"](latents, actions, **kwargs)

    def value(self, observations, **kwargs):
        latents = get_latent(self.encoders["value"], observations)
        return self.networks["value"](latents, **kwargs)

    def __call__(self, observations, actions):
        rets = {}
        if "actor" in self.networks:
            rets["actor"] = self.actor(observations)
        if "critic" in self.networks:
            rets["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            rets["value"] = self.value(observations)
        return rets
