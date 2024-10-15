import math
from dataclasses import dataclass

import jax.numpy as jnp
from chex import Array
from einops import rearrange
from flax import nnx
from jax.typing import DTypeLike

from jflux.math import attention, rope


class EmbedND(nnx.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Array) -> Array:
        n_axes = ids.shape[-1]
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        return jnp.expand_dims(emb, axis=1)


def timestep_embedding(
    t: Array, dim: int, max_period=10000, time_factor: float = 1000.0
) -> Array:
    """
    Generate timestep embeddings.

    Args:
        t: a 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
        time_factor: Tensor of positional embeddings.

    Returns:
        timestep embeddings.
    """
    t = time_factor * t
    half = dim // 2

    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    ).astype(dtype=t.dtype)

    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
        )

    if jnp.issubdtype(t.dtype, jnp.floating):
        embedding = embedding.astype(t.dtype)

    return embedding


class MLPEmbedder(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.in_layer = nnx.Linear(
            in_features=in_dim,
            out_features=hidden_dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.silu = nnx.silu
        self.out_layer = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.query_norm = nnx.RMSNorm(dim, rngs=rngs, param_dtype=param_dtype)
        self.key_norm = nnx.RMSNorm(dim, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, q: Array, k: Array, v: Array) -> tuple[Array, Array]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class SelfAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nnx.Linear(
            in_features=dim,
            out_features=dim * 3,
            use_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.norm = QKNorm(dim=head_dim, rngs=rngs, param_dtype=param_dtype)
        self.proj = nnx.Linear(
            in_features=dim,
            out_features=dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

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
    def __init__(
        self,
        dim: int,
        double: bool,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nnx.Linear(
            in_features=dim,
            out_features=self.multiplier * dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, vec: Array) -> tuple[ModulationOut, ModulationOut | None]:
        out = jnp.split(self.lin(nnx.silu(vec))[:, None, :], self.multiplier, axis=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
        qkv_bias: bool = False,
    ):
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(
            dim=hidden_size, double=True, rngs=rngs, param_dtype=param_dtype
        )
        self.img_norm1 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.img_norm2 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.img_mlp = nnx.Sequential(
            nnx.Linear(
                in_features=hidden_size,
                out_features=mlp_hidden_dim,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
            nnx.gelu,
            nnx.Linear(
                in_features=mlp_hidden_dim,
                out_features=hidden_size,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
        )

        self.txt_mod = Modulation(
            dim=hidden_size, double=True, rngs=rngs, param_dtype=param_dtype
        )
        self.txt_norm1 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.txt_norm2 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.txt_mlp = nnx.Sequential(
            nnx.Linear(
                in_features=hidden_size,
                out_features=mlp_hidden_dim,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
            nnx.gelu,
            nnx.Linear(
                in_features=mlp_hidden_dim,
                out_features=hidden_size,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
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
        q = jnp.concatenate((txt_q, img_q), axis=2)
        k = jnp.concatenate((txt_k, img_k), axis=2)
        v = jnp.concatenate((txt_v, img_v), axis=2)

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
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size * 3 + self.mlp_hidden_dim,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        # proj and mlp_out
        self.linear2 = nnx.Linear(
            in_features=hidden_size + self.mlp_hidden_dim,
            out_features=hidden_size,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.norm = QKNorm(dim=head_dim, rngs=rngs, param_dtype=param_dtype)

        self.hidden_size = hidden_size
        self.pre_norm = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.mlp_act = nnx.gelu
        self.modulation = Modulation(
            hidden_size, double=False, rngs=rngs, param_dtype=param_dtype
        )

    def __call__(self, x: Array, vec: Array, pe: Array) -> Array:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = jnp.split(self.linear1(x_mod), [3 * self.hidden_size], axis=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(jnp.concatenate((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.norm_final = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.linear = nnx.Linear(
            in_features=hidden_size,
            out_features=patch_size * patch_size * out_channels,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                in_features=hidden_size,
                out_features=2 * hidden_size,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
        )

    def __call__(self, x: Array, vec: Array) -> Array:
        shift, scale = jnp.split(
            self.adaLN_modulation(vec),
            2,
            axis=1,
        )
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
