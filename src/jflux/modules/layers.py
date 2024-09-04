import math
from dataclasses import dataclass

from jax import (
    Array,
    numpy as jnp,
)
from einops import rearrange
from flax import nnx

from jflux.math import attention, rope


class EmbedND(nnx.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Array) -> Array:
        n_axes = ids.shape[-1]
        emb = jnp.concat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return jnp.expand_dims(emb, 1)


def timestep_embedding(t: Array, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Array of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Array of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, end=half, dtype=jnp.float32) / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concat([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nnx.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nnx.silu
        self.out_layer = nnx.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: Array) -> Array:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nnx.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nnx.Param(jnp.ones(dim))

    def __call__(self, x: Array):
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        rrms = jnp.reciprocal(jnp.sqrt(jnp.mean(x**2, axis=-1, keepdim=True) + 1e-6))
        return (x * rrms).astype(x_dtype) * self.scale


class QKNorm(nnx.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def __call__(self, q: Array, k: Array, v: Array) -> tuple[Array, Array]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nnx.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nnx.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nnx.Linear(dim, dim)

    def __call__(self, x: Array, pe: Array) -> Array:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Array
    scale: Array
    gate: Array


class Modulation(nnx.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nnx.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, vec: Array) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nnx.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nnx.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nnx.GELU(approximate="tanh"),
            nnx.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nnx.GELU(approximate="tanh"),
            nnx.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def __call__(
        self, img: Array, txt: Array, vec: Array, pe: Array
    ) -> tuple[Array, Array]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = jnp.concat((txt_q, img_q), dim=2)
        k = jnp.concat((txt_k, img_k), dim=2)
        v = jnp.concat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class SingleStreamBlock(nnx.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nnx.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nnx.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nnx.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x: Array, vec: Array, pe: Array) -> Array:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = jnp.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(jnp.concat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nnx.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nnx.Sequential(
            nnx.SiLU(), nnx.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def __call__(self, x: Array, vec: Array) -> Array:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
