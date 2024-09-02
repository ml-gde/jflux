from jax import Array
from jax import numpy as jnp
from flax import nnx
from einops import rearrange


def attention(q: Array, k: Array, v: Array, pe: Array) -> Array:
    q, k = apply_rope(q, k, pe)

    x = nnx.dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Array, dim: int, theta: int) -> Array:
    assert dim % 2 == 0
    scale = jnp.arange(0, dim, 2, dtype=jnp.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = jnp.einsum("...n,d->...nd", pos, omega)
    out = jnp.stack([jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Array, xk: Array, freqs_cis: Array) -> tuple[Array, Array]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
