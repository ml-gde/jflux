import numpy as np
import jax
import torch
import chex
from jflux.sampling import get_noise as jax_get_noise

from flux.sampling import get_noise as torch_get_noise


class SamplingTestCase(chex.TestCase):
    def test_get_noise(self):
        # for schnell
        height = 768
        width = 1360

        # for packing
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        # get noise
        x_jax = jax_get_noise(
            num_samples=1,
            height=height,
            width=width,
            dtype=jax.dtypes.bfloat16,
            seed=jax.random.PRNGKey(seed=42),
        )
        x_torch = torch_get_noise(
            num_samples=1,
            height=height,
            width=width,
            dtype=torch.bfloat16,
            seed=42,
            device="cuda",
        )
        print(x_jax.shape)
        chex.assert_equal_shape([x_jax, x_torch])
