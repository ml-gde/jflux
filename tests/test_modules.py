import chex
import jax.numpy as jnp
import pytest
import torch
from flax import nnx
from flux.modules.layers import MLPEmbedder
from flux.modules.layers import Modulation as PytorchModulation
from flux.modules.layers import SelfAttention as PytorchSelfAttention

from jflux.modules.layers import MLPEmbedder as JaxMLPEmbedder
from jflux.modules.layers import Modulation as JaxModulation
from jflux.modules.layers import SelfAttention as JaxSelfAttention
from tests.utils import torch2jax


class ModulesTestCase(chex.TestCase):
    def test_mlp_embedder(self):
        # Initialize layers
        pytorch_mlp_embedder = MLPEmbedder(in_dim=512, hidden_dim=256)
        jax_mlp_embedder = JaxMLPEmbedder(
            in_dim=512,
            hidden_dim=256,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.float32,
        )

        # Generate random inputs
        torch_input = torch.randn(1, 32, 512, dtype=torch.float32)
        jax_input = torch2jax(torch_input)

        # Forward pass
        jax_output = jax_mlp_embedder(jax_input)
        pytorch_output = pytorch_mlp_embedder(torch_input)

        # Assertions
        chex.assert_equal_shape([jax_output, torch2jax(pytorch_output)])

    @pytest.mark.skip(reason="Blocked by apply_rope")
    def test_self_attention(self):
        # Initialize layers
        pytorch_self_attention = PytorchSelfAttention(dim=512)
        jax_self_attention = JaxSelfAttention(
            dim=512, rngs=nnx.Rngs(default=42), param_dtype=jnp.float32
        )

        # Generate random inputs
        torch_input = torch.randn(1, 32, 512, dtype=torch.float32)
        torch_pe = torch.randn(1, 32, 512, dtype=torch.float32)
        jax_input = torch2jax(torch_input)
        jax_pe = torch2jax(torch_pe)

        # Forward pass
        jax_output = jax_self_attention(jax_input, jax_pe)
        pytorch_output = pytorch_self_attention(torch_input, torch_pe)

        # Assertions
        chex.assert_equal_shape([jax_output, torch2jax(pytorch_output)])

    def test_modulation(self):
        # Initialize layers
        pytorch_modulation = PytorchModulation(dim=512, double=True)
        jax_modulation = JaxModulation(
            dim=512, double=True, rngs=nnx.Rngs(default=42), param_dtype=jnp.float32
        )

        # Generate random inputs
        torch_input = torch.randn(1, 32, 512, dtype=torch.float32)
        jax_input = torch2jax(torch_input)

        # Forward pass
        jax_output = jax_modulation(jax_input)
        pytorch_output = pytorch_modulation(torch_input)

        # Convert Modulation output to individual tensors
        jax_tensors = [jax_output[0].shift, jax_output[0].scale, jax_output[0].gate]
        torch_tensors = [
            torch2jax(pytorch_output[0].shift),
            torch2jax(pytorch_output[0].scale),
            torch2jax(pytorch_output[0].gate),
        ]

        # Assertions
        assert len(jax_output) == len(pytorch_output)
        for i in range(len(jax_output)):
            chex.assert_equal_shape([jax_tensors[i], torch_tensors[i]])
