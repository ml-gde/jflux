import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from chex import Array
from einops import rearrange
from flax import nnx
from jax.typing import DTypeLike

from jflux.layers import QKNorm
from jflux.math import attention


class MLPEmbedder(nnx.Module):
    """
    MLP embedder with a single hidden layer and SiLU activation.

    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        rngs (nnx.Rngs): RNGs for the layer.
        dtype (DTypeLike): Data type for the layer.
        param_dtype (DTypeLike): Parameter data type for the layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        if param_dtype is None:
            param_dtype = dtype

        self.in_layer = nnx.Linear(
            in_features=in_dim,
            out_features=hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=True,
            rngs=rngs,
        )
        self.out_layer = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.out_layer(nnx.silu(self.in_layer(x)))


class SelfAttention(nnx.Module):
    """
    Self-attention module with QKV linear layers and a projection layer.

    Args:
        dim (int): Dimension of the input.
        rngs (nnx.Rngs): RNGs for the layer.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use bias in QKV linear layers.
        dtype (DTypeLike): Data type for the layer.
        param_dtype (DTypeLike): Parameter data type for the layer.
    """

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nnx.Linear(
            in_features=dim,
            out_features=dim * 3,
            use_bias=qkv_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.norm = QKNorm(head_dim, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.proj = nnx.Linear(
            in_features=dim,
            out_features=dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array, pe: Array) -> Array:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


# TODO (SauravMaheshkar): use `chex.dataclass`
@dataclass
class ModulationOut:
    shift: Array
    scale: Array
    gate: Array


class Modulation(nnx.Module):
    """
    Modulation module with a linear layer and split output.

    Args:
        dim (int): Dimension of the input.
        double (bool): Whether to split the output into two parts.
        rngs (nnx.Rngs): RNGs for the layer.
        dtype (DTypeLike): Data type for the layer.
        param_dtype (DTypeLike): Parameter data type for the layer.
    """

    def __init__(
        self,
        dim: int,
        double: bool,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        if param_dtype is None:
            param_dtype = dtype
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nnx.Linear(
            dim,
            self.multiplier * dim,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, vec: Array) -> tuple[ModulationOut, ModulationOut | None]:
        ary = self.lin(nnx.silu(vec))[:, None, :]
        out = jnp.split(ary, self.multiplier, axis=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nnx.Module):
    """
    Custom Module for a DoubleStreamBlock.

    Args:
        hidden_size (int): Dimension of the hidden layer.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of hidden layer to mlp hidden layer.
        rngs (nnx.Rngs): RNGs for the layer.
        qkv_bias (bool): Whether to use bias in QKV linear layers.
        dtype (DTypeLike): Data type for the layer.
        param_dtype (DTypeLike): Parameter data type for the layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rngs: nnx.Rngs,
        qkv_bias: bool = False,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ):
        if param_dtype is None:
            param_dtype = dtype

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(
            hidden_size, double=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )
        self.img_norm1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.img_norm2 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )
        self.img_mlp = nnx.Sequential(
            nnx.Linear(
                hidden_size,
                mlp_hidden_dim,
                use_bias=True,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            ),
            nnx.gelu,
            nnx.Linear(
                mlp_hidden_dim,
                hidden_size,
                use_bias=True,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            ),
        )

        self.txt_mod = Modulation(
            hidden_size, double=True, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )
        self.txt_norm1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.txt_norm2 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )
        self.txt_mlp = nnx.Sequential(
            nnx.Linear(
                hidden_size,
                mlp_hidden_dim,
                use_bias=True,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            ),
            nnx.gelu,
            nnx.Linear(
                mlp_hidden_dim,
                hidden_size,
                use_bias=True,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            ),
        )

    @typing.no_type_check
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
        q = jnp.concat((txt_q, img_q), axis=2)
        k = jnp.concat((txt_k, img_k), axis=2)
        v = jnp.concat((txt_v, img_v), axis=2)

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

    Args:
        hidden_size (int): Dimension of the hidden layer.
        num_heads (int): Number of attention heads.
        rngs (nnx.Rngs): RNGs for the layer.
        mlp_ratio (float): Ratio of hidden layer to mlp hidden layer.
        qk_scale (float): Scaling factor for query and key.
        dtype (DTypeLike): Data type for the layer.
        param_dtype (DTypeLike): Parameter data type for the layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rngs: nnx.Rngs,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ):
        if param_dtype is None:
            param_dtype = dtype
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nnx.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        # proj and mlp_out
        self.linear2 = nnx.Linear(
            hidden_size + self.mlp_hidden_dim,
            hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.norm = QKNorm(head_dim, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

        self.hidden_size = hidden_size
        self.pre_norm = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )

        self.mlp_act = nnx.gelu
        self.modulation = Modulation(
            hidden_size, double=False, rngs=rngs, dtype=dtype, param_dtype=param_dtype
        )

    def __call__(self, x: Array, vec: Array, pe: Array) -> Array:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = jnp.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], axis=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(jnp.concat((attn, self.mlp_act(mlp)), axis=2))
        return x + mod.gate * output
