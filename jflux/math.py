import typing

from einops import rearrange
from jax import Array, numpy as jnp, nn


@typing.no_type_check
def attention(q: Array, k: Array, v: Array, pe: Array) -> Array:
    # TODO (ariG23498): Change all usage of attention to use this function
    q, k = apply_rope(q, k, pe)

    # jax expects this shape
    x = rearrange(x, "B H L D -> B L H D")  # noqa
    x = nn.dot_product_attention(q, k, v)
    x = rearrange(x, "B L H D -> B L (H D)")  # reshape again

    return x


def rope(pos: Array, dim: int, theta: int) -> Array:
    """
    Generate Rotary Position Embedding (RoPE) for positional encoding.

    Args:
        pos (Array): Positional values, typically a sequence of positions in an array format.
        dim (int): The embedding dimension, which must be an even number.
        theta (int): A scaling parameter for RoPE that controls the frequency range of rotations.

    Returns:
        Array: Rotary embeddings with cosine and sine components for each position and dimension.
    """

    # Embedding dimension must be an even number
    assert dim % 2 == 0

    # Generate the RoPE embeddings
    scale = jnp.arange(0, dim, 2, dtype=jnp.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = jnp.einsum("...n,d->...nd", pos, omega)
    out = jnp.stack([jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], axis=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.astype(jnp.float32)


def apply_rope(xq: Array, xk: Array, freqs_cis: Array) -> tuple[Array, Array]:
    """
    Apply RoPE to the input query and key tensors.

    Args:
        xq (Array): Query tensor.
        xk (Array): Key tensor.
        freqs_cis (Array): RoPE frequencies.

    Returns:
        tuple[Array, Array]: Query and key tensors with RoPE applied.
    """
    # Reshape and typecast the input tensors
    xq_ = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 1, 2)

    # Apply RoPE to the input tensors
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    # Reshape and typecast the output tensors
    return xq_out.reshape(*xq.shape).astype(xq.dtype), xk_out.reshape(*xk.shape).astype(
        xk.dtype
    )
