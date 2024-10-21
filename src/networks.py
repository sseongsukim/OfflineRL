from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax


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


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)


class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'state'
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return rep


class MonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations, goals=None, info=False):
        phi = observations
        psi = goals

        v1, v2 = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)

        if info:
            return {
                'v': (v1 + v2) / 2,
            }
        return v1, v2
