import chex
import jax
import jax.numpy as jnp
import torch
from flux.math import apply_rope as torch_apply_rope
from flux.math import rope as torch_rope

from jflux.math import apply_rope as jax_apply_rope
from jflux.math import rope as jax_rope

from .utils import torch2jax


class TestMath(chex.TestCase):
    def test_rope(self):
        B, L, _, D = (
            2,
            4,
            2,
            8,
        )  # Batch size, sequence length, number of heads, embedding dimension
        theta = 10000

        jax_positions = jnp.expand_dims(jnp.arange(L, dtype=jnp.int32), axis=0).repeat(
            B, axis=1
        )
        torch_positions = torch.from_numpy(jax_positions.__array__()).to(torch.int32)

        chex.assert_trees_all_close(
            jax_positions,
            torch2jax(torch_positions),
            rtol=1e-5,
            atol=1e-5,
        )

        torch_pe = torch_rope(pos=torch_positions, dim=D, theta=theta)
        jax_pe = jax_rope(pos=jax_positions, dim=D, theta=theta)

        chex.assert_trees_all_close(
            jax_pe,
            torch2jax(torch_pe),
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
        jax_q = jax.random.normal(key=jax.random.PRNGKey(42), shape=(B, H, L, D))
        jax_k = jax.random.normal(key=jax.random.PRNGKey(42), shape=(B, H, L, D))

        torch_q = torch.from_numpy(jax_q.__array__()).to(torch.float32)
        torch_k = torch.from_numpy(jax_k.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_q,
            torch2jax(torch_q),
            rtol=1e-5,
            atol=1e-5,
        )
        chex.assert_trees_all_close(
            jax_k,
            torch2jax(torch_k),
            rtol=1e-5,
            atol=1e-5,
        )

        # Position indices (e.g., positions in the sequence)
        jax_positions = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(B, L), dtype=jnp.float32
        )
        torch_positions = torch.from_numpy(jax_positions.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_positions,
            torch2jax(torch_positions),
            rtol=1e-5,
            atol=1e-5,
        )

        torch_pe = torch_rope(pos=torch_positions, dim=(3072 // 24), theta=theta)
        jax_pe = jax_rope(pos=jax_positions, dim=(3072 // 24), theta=theta)

        chex.assert_trees_all_close(
            jax_pe,
            torch2jax(torch_pe),
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

        chex.assert_trees_all_close(
            jax_q_rotated,
            torch2jax(torch_q_rotated),
            rtol=1e-5,
            atol=1e-5,
        )

        chex.assert_trees_all_close(
            jax_k_rotated,
            torch2jax(torch_k_rotated),
            rtol=1e-5,
            atol=1e-5,
        )
