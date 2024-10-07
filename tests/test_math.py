import unittest

import jax.numpy as jnp
import numpy as np
import torch
from flux.math import apply_rope as torch_apply_rope
from flux.math import attention as torch_attention
from flux.math import rope as torch_rope

from jflux.math import apply_rope as jax_apply_rope
from jflux.math import attention as jax_attention
from jflux.math import rope as jax_rope


class TestMath(np.testing.TestCase):
    def test_rope(self):
        B, L, H, D = (
            2,
            4,
            2,
            8,
        )  # Batch size, sequence length, number of heads, embedding dimension
        theta = 10000

        # Position indices (e.g., positions in the sequence)
        np_positions = (
            np.expand_dims(np.arange(L), 0).repeat(B, 1).astype(np.int32)
        )  # Shape: [B, L]
        torch_positions = torch.from_numpy(np_positions).to(torch.int32)
        jax_positions = jnp.array(np_positions, dtype=jnp.int32)

        np.testing.assert_allclose(np.array(jax_positions), torch_positions.numpy())

        torch_pe = torch_rope(pos=torch_positions, dim=D, theta=theta)
        jax_pe = jax_rope(
            pos=jax_positions, dim=D, theta=theta
        )  # Shape: [B, L, D/2, 2, 2]

        np.testing.assert_allclose(
            np.array(jax_pe),
            torch_pe.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_apply_rope(self):
        B, H, L, D = (
            1,
            24,
            4336,
            128,
        )
        theta = 10000

        # Inputs
        np_q = np.random.randn(B, H, L, D).astype(np.float32)
        np_k = np.random.randn(B, H, L, D).astype(np.float32)

        jax_q = jnp.array(np_q, dtype=jnp.float32)
        jax_k = jnp.array(np_k, dtype=jnp.float32)

        torch_q = torch.from_numpy(np_q).to(torch.float32)
        torch_k = torch.from_numpy(np_k).to(torch.float32)

        np.testing.assert_allclose(np.array(jax_q), torch_q.numpy())
        np.testing.assert_allclose(np.array(jax_k), torch_k.numpy())

        # Position indices (e.g., positions in the sequence)
        np_positions = np.random.randn(1, L).astype(np.float32)
        torch_positions = torch.from_numpy(np_positions).to(torch.float32)
        jax_positions = jnp.array(np_positions, dtype=jnp.float32)

        np.testing.assert_allclose(np.array(jax_positions), torch_positions.numpy())

        torch_pe = torch_rope(pos=torch_positions, dim=(3072 // 24), theta=theta)
        jax_pe = jax_rope(pos=jax_positions, dim=(3072 // 24), theta=theta)

        np.testing.assert_allclose(
            np.array(jax_pe),
            torch_pe.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        torch_pe = torch_pe.unsqueeze(1).expand(
            -1, H, -1, -1, -1, -1
        )  # Shape: [B, H, L, D//2, 2, 2]
        jax_pe = jnp.repeat(jnp.expand_dims(jax_pe, axis=1), repeats=H, axis=1)

        # Apply RoPE to q and k
        torch_q_rotated, torch_k_rotated = torch_apply_rope(
            xq=torch_q, xk=torch_k, freqs_cis=torch_pe
        )
        jax_q_rotated, jax_k_rotated = jax_apply_rope(
            xq=jax_q, xk=jax_k, freqs_cis=jax_pe
        )

        np.testing.assert_allclose(
            np.array(jax_q_rotated),
            torch_q_rotated.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(jax_k_rotated),
            torch_k_rotated.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    # def test_attention(self):
    #     # Generate random inputs
    #     np_input = np.random.randn(2, 32, 4, 4).astype(np.float32)
    #     jax_input = jnp.array(np_input, dtype=jnp.float32)
    #     torch_input = torch.from_numpy(np_input).to(torch.float32)

    #     np.testing.assert_allclose(np.array(jax_input), torch_input.numpy())

    #     # Forward pass
    #     torch_output = torch_downsample(torch_input)
    #     jax_output = jax_downsample(rearrange(jax_input, "b c h w -> b h w c"))

    #     # Assertions
    #     np.testing.assert_allclose(
    #         np.array(rearrange(jax_output, "b h w c -> b c h w")),
    #         torch_output.detach().numpy(),
    #         rtol=1e-5,
    #         atol=1e-5,
    #     )

    # def test_rope(self):
    #     pos = jnp.expand_dims(jnp.arange(self.seq_len), axis=0)
    #     pos = jnp.repeat(pos, self.batch_size, axis=0)

    #     rope_output = rope(pos, self.dim, self.theta)
    #     expected_shape = (self.batch_size, self.seq_len, self.dim // 2, 2, 2)

    #     self.assertEqual(
    #         rope_output.shape, expected_shape, "rope function output shape is incorrect"
    #     )

    # @pytest.mark.xfail
    # def test_apply_rope(self):
    #     pos = jnp.expand_dims(jnp.arange(self.seq_len), axis=0)
    #     pos = jnp.repeat(pos, self.batch_size, axis=0)

    #     freqs_cis = rope(pos, self.dim, self.theta)
    #     xq_out, xk_out = apply_rope(self.q, self.k, freqs_cis)

    #     self.assertEqual(
    #         xq_out.shape, self.q.shape, "apply_rope xq output shape is incorrect"
    #     )
    #     self.assertEqual(
    #         xk_out.shape, self.k.shape, "apply_rope xk output shape is incorrect"
    #     )

    # @pytest.mark.xfail
    # def test_attention(self):
    #     pos = jnp.expand_dims(jnp.arange(self.seq_len), axis=0)
    #     pos = jnp.repeat(pos, self.batch_size, axis=0)

    #     freqs_cis = rope(pos, self.dim, self.theta)
    #     attention_output = attention(self.q, self.k, self.v, freqs_cis)

    #     expected_shape = (self.batch_size, self.seq_len, self.num_heads * self.dim)

    #     self.assertEqual(
    #         attention_output.shape,
    #         expected_shape,
    #         "attention function output shape is incorrect",
    #     )
