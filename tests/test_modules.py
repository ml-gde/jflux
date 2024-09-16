import chex
import jax.numpy as jnp
import torch
from flax import nnx
from flux.modules.layers import MLPEmbedder

from jflux.modules import MLPEmbedder as JaxMLPEmbedder
from tests.utils import torch2jax


class ModulesTestCase(chex.TestCase):
    def test_mlp_embedder(self):
        # Initialize layers
        pytorch_mlp_embedder = MLPEmbedder(in_dim=512, hidden_dim=256)
        jax_mlp_embedder = JaxMLPEmbedder(
            in_dim=512, hidden_dim=256, rngs=nnx.Rngs(default=42), dtype=jnp.float32
        )

        # Generate random inputs
        torch_input = torch.randn(1, 32, 512, dtype=torch.float32)
        jax_input = torch2jax(torch_input)

        # Forward pass
        jax_output = jax_mlp_embedder(jax_input)
        pytorch_output = pytorch_mlp_embedder(torch_input)

        # Assertions
        chex.assert_equal_shape([jax_output, torch2jax(pytorch_output)])
