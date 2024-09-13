import math

import jax
import jax.numpy as jnp
from chex import Array
from flax import nnx
from functools import partial

from jflux.math import rope


class Embed(nnx.Module):
    """
    Embedding module for Positional Embeddings.

    Args:
        dim (int): Dimension of the embedding.
        theta (int): theta parameter for the RoPE embedding
        axes_dim (list[int]): List of axes dimensions.

    Returns:
        RoPE embeddings
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]) -> None:
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Array) -> Array:
        n_axes = ids.shape[-1]
        emb = jnp.concat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        return jnp.expand_dims(emb, 1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def timestep_embedding(
    t: Array, dim: int, max_period=10000, time_factor: float = 1000.0
) -> Array:
    """
    Generate timestep embeddings.

    Args:
        t (Array): An array of timesteps to be embedded.
        dim (int): The desired dimensionality of the output embedding.
        max_period (int, optional): The maximum period for the sinusoidal functions. Defaults to 10000.
        time_factor (float, optional): A scaling factor applied to the input timesteps. Defaults to 1000.0.

    Returns:
        timestep embeddings.
    """
    # Pre-Processing:
    # * scales the input timesteps by the given time factor
    t = time_factor * t
    half = dim // 2

    # Determine frequencies using exponential decay
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(start=0, stop=half, dtype=jnp.float32, device=t.device())
        / half
    )

    # Create embeddings by concatenating sine and cosines
    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)

    # Handle odd dimensions
    if dim % 2:
        embedding = jnp.concat([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)

    # If timestamps are floating types ensure so is the embedding
    if jnp.issubdtype(t.device(), jnp.floating):
        embedding = embedding.astype(t.device())

    return embedding


class QKNorm(nnx.Module):
    """
    Normalization layer for query and key values.

    Args:
        dim (int): Dimension of the hidden layer.
        rngs (nnx.Rngs): RNGs for the layer.

    Returns:
        Normalized query and key values
    """

    def __init__(self, dim: int, rngs: nnx.Rngs) -> None:
        # RMS Normalization for query and key
        self.query_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.key_norm = nnx.RMSNorm(dim, rngs=rngs)

    def __call__(self, q: Array, k: Array, v: Array) -> tuple[Array, Array]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.astype(v.device()), k.astype(v.device())


class AdaLayerNorm(nnx.Module):
    """
    Normalization layer modified to incorporate timestep embeddings.

    Args:
        hidden_size (int): Dimension of the hidden layer.
        patch_size (int): patch size.
        out_channels (int): Number of output channels.
        rngs (nnx.Rngs): RNGs for the layer.

    Returns:
        Normalized layer incorporating timestep embeddings.
    """

    def __init__(
        self, hidden_size: int, patch_size: int, out_channels: int, rngs: nnx.Rngs
    ) -> None:
        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            use_bias=True,
            rngs=rngs,
        )
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu, nnx.Linear(hidden_size, 2 * hidden_size, use_bias=True, rngs=rngs)
        )

    def __call__(self, x: Array, vec: Array) -> Array:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class DiagonalGaussian(nnx.Module):
    """
    A module that represents a diagonal Gaussian distribution.

    Args:
        sample (bool, optional): Whether to sample from the distribution. Defaults to True.
        chunk_dim (int, optional): The dimension along which to chunk the input. Defaults to 1.

    Returns:
        Array: The output array representing the sampled or mean values.
    """

    def __init__(self, key: Array, sample: bool = True, chunk_dim: int = 1) -> None:
        self.sample = sample
        self.chunk_dim = chunk_dim
        self.key = key

    def __call__(self, z: Array) -> Array:
        mean, logvar = jnp.split(z, indices_or_sections=2, axis=self.chunk_dim)
        if self.sample:
            std = jnp.exp(0.5 * logvar)
            return mean + std * jax.random.normal(
                key=self.key, shape=mean.shape, dtype=z.dtype
            )
        else:
            return mean


class Identity(nnx.Module):
    """
    Identity module.

    Args:
        x (Array): Input array.

    Returns:
        Array: The input array.
    """

    def __call__(self, x: Array) -> Array:
        return x
