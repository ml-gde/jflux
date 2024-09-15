import unittest
import pytest
import jax.numpy as jnp

from jflux.math import attention, rope, apply_rope


class TestAttentionMechanism(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 8
        self.dim = 64
        self.theta = 10000

        self.q = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.dim))
        self.k = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.dim))
        self.v = jnp.ones((self.batch_size, self.num_heads, self.seq_len, self.dim))

    def test_rope(self):
        pos = jnp.expand_dims(jnp.arange(self.seq_len), axis=0)
        pos = jnp.repeat(pos, self.batch_size, axis=0)

        rope_output = rope(pos, self.dim, self.theta)
        expected_shape = (self.batch_size, self.seq_len, self.dim // 2, 2, 2)

        self.assertEqual(
            rope_output.shape, expected_shape, "rope function output shape is incorrect"
        )

    @pytest.mark.xfail
    def test_apply_rope(self):
        pos = jnp.expand_dims(jnp.arange(self.seq_len), axis=0)
        pos = jnp.repeat(pos, self.batch_size, axis=0)

        freqs_cis = rope(pos, self.dim, self.theta)
        xq_out, xk_out = apply_rope(self.q, self.k, freqs_cis)

        self.assertEqual(
            xq_out.shape, self.q.shape, "apply_rope xq output shape is incorrect"
        )
        self.assertEqual(
            xk_out.shape, self.k.shape, "apply_rope xk output shape is incorrect"
        )

    @pytest.mark.xfail
    def test_attention(self):
        pos = jnp.expand_dims(jnp.arange(self.seq_len), axis=0)
        pos = jnp.repeat(pos, self.batch_size, axis=0)

        freqs_cis = rope(pos, self.dim, self.theta)
        attention_output = attention(self.q, self.k, self.v, freqs_cis)

        expected_shape = (self.batch_size, self.seq_len, self.num_heads * self.dim)

        self.assertEqual(
            attention_output.shape,
            expected_shape,
            "attention function output shape is incorrect",
        )
