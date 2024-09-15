import chex
import jax.numpy as jnp
import torch.nn as nn
import torch
from einops import rearrange

from jflux.layers import Embed


def torch_rope(pos, dim: int, theta: int):
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [
                torch_rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(n_axes)
            ],
            dim=-3,
        )

        return emb.unsqueeze(1)


class EmbedTestCase(chex.TestCase):
    def test_embed(self):
        # Initialize layers
        pytorch_embed_layer = EmbedND(512, 10000, [64, 64, 64, 64])
        jax_embed_layer = Embed(512, 10000, [64, 64, 64, 64])

        # Generate random inputs
        torch_ids = torch.randint(0, 10000, (1, 32, 4), dtype=torch.float64)
        jax_ids = jnp.asarray(torch_ids.numpy())

        # Forward pass
        jax_output = jax_embed_layer(jax_ids)
        pytorch_output = pytorch_embed_layer(torch_ids)

        # Assertions
        chex.assert_equal_shape([jax_output, jnp.asarray(pytorch_output.numpy())])
        chex.assert_trees_all_close(
            jax_output, jnp.asarray(pytorch_output.numpy()), rtol=1e-3, atol=1e-3
        )
