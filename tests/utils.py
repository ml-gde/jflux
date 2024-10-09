import jax.numpy as jnp
import torch
from chex import Array


__all__ = ["torch2jax"]


def torch2jax(x: torch.Tensor) -> Array:
    return jnp.asarray(x.detach().numpy())
