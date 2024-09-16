import chex
import jax
import jax.numpy as jnp
import torch
from flax import nnx
from flux.modules.autoencoder import DiagonalGaussian as PytorchDiagonalGaussian
from flux.modules.layers import EmbedND, LastLayer
from flux.modules.layers import QKNorm as PytorchQKNorm

from jflux.layers import AdaLayerNorm, Embed
from jflux.layers import DiagonalGaussian as JaxDiagonalGaussian
from jflux.layers import QKNorm as JaxQKNorm
from tests.utils import torch2jax


class LayersTestCase(chex.TestCase):
    def test_embed(self):
        # Initialize layers
        pytorch_embed_layer = EmbedND(512, 10000, [64, 64, 64, 64])
        jax_embed_layer = Embed(512, 10000, [64, 64, 64, 64])

        # Generate random inputs
        torch_ids = torch.randint(0, 10000, (1, 32, 4))
        jax_ids = torch2jax(torch_ids)

        # Forward pass
        jax_output = jax_embed_layer(jax_ids)
        pytorch_output = pytorch_embed_layer(torch_ids)

        # Assertions
        chex.assert_equal_shape([jax_output, torch2jax(pytorch_output)])
        chex.assert_trees_all_close(
            jax_output, torch2jax(pytorch_output), rtol=1e-3, atol=1e-3
        )

    def test_qk_norm(self):
        # Initialize layers
        pytorch_qk_norm_layer = PytorchQKNorm(512)
        jax_qk_norm_layer = JaxQKNorm(512, rngs=nnx.Rngs(default=42), dtype=jnp.float32)

        # Generate random inputs
        torch_query = torch.randn(1, 32, 512, dtype=torch.float32)
        torch_key = torch.randn(1, 32, 512, dtype=torch.float32)
        torch_value = torch.randn(1, 32, 512, dtype=torch.float32)
        jax_query = torch2jax(torch_query)
        jax_key = torch2jax(torch_key)
        jax_value = torch2jax(torch_value)

        # Forward pass
        jax_output = jax_qk_norm_layer(jax_query, jax_key, jax_value)
        pytorch_output = pytorch_qk_norm_layer(torch_query, torch_key, torch_value)

        # Assertions
        assert len(jax_output) == len(pytorch_output)
        for i in range(len(jax_output)):
            chex.assert_equal_shape([jax_output[i], torch2jax(pytorch_output[i])])
            chex.assert_trees_all_close(
                jax_output[i], torch2jax(pytorch_output[i]), rtol=1e-3, atol=1e-3
            )

    def test_adalayer_norm(self):
        # Initialize layers
        pytorch_adalayer_norm_layer = LastLayer(
            hidden_size=512,
            patch_size=16,
            out_channels=512,
        )
        jax_adalayer_norm_layer = AdaLayerNorm(
            hidden_size=512,
            patch_size=16,
            out_channels=512,
            rngs=nnx.Rngs(default=42),
            dtype=jnp.float32,
        )

        # Generate random inputs
        torch_hidden = torch.randn(1, 32, 512, dtype=torch.float32)
        torch_vec = torch.randn(1, 512, dtype=torch.float32)
        jax_hidden = torch2jax(torch_hidden)
        jax_vec = torch2jax(torch_vec)

        # Forward pass
        jax_output = jax_adalayer_norm_layer(jax_hidden, jax_vec)
        pytorch_output = pytorch_adalayer_norm_layer(torch_hidden, torch_vec)

        # Assertions
        chex.assert_equal_shape([jax_output, torch2jax(pytorch_output)])

    def test_diagonal_gaussian(self):
        # Initialize layers
        pytorch_diagonal_gaussian_layer = PytorchDiagonalGaussian()
        jax_diagonal_gaussian_layer = JaxDiagonalGaussian(key=jax.random.key(42))

        # Generate random inputs
        torch_input = torch.randn(1, 32, 512, dtype=torch.float32)
        jax_input = torch2jax(torch_input)

        # Forward pass
        jax_output = jax_diagonal_gaussian_layer(jax_input)
        pytorch_output = pytorch_diagonal_gaussian_layer(torch_input)

        # Assertions
        chex.assert_equal_shape([jax_output, torch2jax(pytorch_output)])
