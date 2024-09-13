from dataclasses import dataclass

import jax.dtypes
from flax import nnx
from jax import numpy as jnp
from chex import Array
from jax.typing import DTypeLike
from jflux.layers import (
    Identity,
    Embed,
    AdaLayerNorm,
    timestep_embedding,
)
from jflux.modules import DoubleStreamBlock, SingleStreamBlock, MLPEmbedder


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nnx.Module):
    """
    Transformer model for flow matching on sequences.

    Args:
        params (FluxParams): Model parameters.
        rngs (nnx.Rngs): Random number generators.
        dtype (DTypeLike, optional): Data type for the model. Defaults to jax.dtypes.bfloat16.
        param_dtype (DTypeLike, optional): Data type for the model parameters. Defaults to None.
    """

    def __init__(
        self,
        params: FluxParams,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        self.params = params
        self.rngs = rngs
        self.dtype = dtype
        if param_dtype is None:
            self.param_dtype = dtype
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = Embed(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nnx.Linear(
            self.in_channels,
            self.hidden_size,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=self.hidden_size,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.vector_in = MLPEmbedder(
            params.vec_in_dim,
            self.hidden_size,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.guidance_in = (
            MLPEmbedder(
                in_dim=256,
                hidden_dim=self.hidden_size,
                rngs=rngs,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            if params.guidance_embed
            else Identity()
        )
        self.txt_in = nnx.Linear(
            params.context_in_dim,
            self.hidden_size,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.double_blocks = nnx.Sequential(
            *[
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    rngs=rngs,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nnx.Sequential(
            *[
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    rngs=rngs,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = AdaLayerNorm(
            self.hidden_size,
            1,
            self.out_channels,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        img: Array,
        img_ids: Array,
        txt: Array,
        txt_ids: Array,
        timesteps: Array,
        y: Array,
        guidance: Array | None = None,
    ) -> Array:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))  # type: ignore
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = jnp.concat((txt_ids, img_ids), axis=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = jnp.concat((txt, img), axis=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
