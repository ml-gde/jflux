import chex
import jax.numpy as jnp
import torch

from flux.modules.layers import EmbedND
from jflux.layers import Embed


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
