import chex
import pytest
from flax import nnx
from jax import numpy as jnp

from jflux.model import Flux, FluxParams


class ModelTestCase(chex.TestCase):
    @pytest.mark.skip
    def test_model(self):
        # Initialize
        in_channels = 64
        vec_in_dim = 768
        context_in_dim = 4096
        hidden_size = 3072
        mlp_ratio = 4.0
        num_heads = 24
        depth = 19
        depth_single_blocks = 38
        axes_dim = [16, 56, 56]
        theta = 10_000
        qkv_bias = True
        guidance_embed = False
        rngs = nnx.Rngs(default=42)
        param_dtype = jnp.float32

        flux_params = FluxParams(
            in_channels=in_channels,
            vec_in_dim=vec_in_dim,
            context_in_dim=context_in_dim,
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            depth=depth,
            depth_single_blocks=depth_single_blocks,
            axes_dim=axes_dim,
            theta=theta,
            qkv_bias=qkv_bias,
            guidance_embed=guidance_embed,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        flux = Flux(params=flux_params)

        assert flux is not None
