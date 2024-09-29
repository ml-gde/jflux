from dataclasses import dataclass

import jax
import jax.numpy as jnp
from chex import Array
from einops import rearrange
from flax import nnx
from jax.typing import DTypeLike

from jflux.layers import DiagonalGaussian
from jflux.sampling import interpolate


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


class AttnBlock(nnx.Module):
    """
    Attention Block for the Encoder and Decoder.

    Args:
        in_channels (int): Number of input channels.
        rngs (nnx.Rngs): RNGs for the module.
        dtype (DTypeLike): Data type for the module.
        param_dtype (DTypeLike): Data type for the module parameters.
    """

    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        if param_dtype is None:
            param_dtype = dtype

        # Normalization Layer
        self.norm = nnx.GroupNorm(
            num_groups=32,
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        # Query, Key and Value Layers
        self.query_layer = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.key_layer = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.value_layer = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        # Output Projection Layer
        self.projection = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def attention(self, input_tensor: Array) -> Array:
        # Apply Group Norm
        input_tensor = self.norm(input_tensor)

        # Calculate Query, Key and Values
        query = self.query_layer(input_tensor)
        key = self.key_layer(input_tensor)
        value = self.value_layer(input_tensor)

        # TODO (ariG23498): incorporate the attention fn from jflux.math
        # Reshape for JAX Attention impl
        b, c, h, w = query.shape
        query = rearrange(query, "b c h w -> b (h w) 1 c")
        key = rearrange(key, "b c h w -> b (h w) 1 c")
        value = rearrange(value, "b c h w -> b (h w) 1 c")

        # Calculate Attention
        input_tensor = nnx.dot_product_attention(query, key, value)
        return rearrange(input_tensor, "b (h w) 1 c -> b c h w", h=h, w=w, c=c, b=b)

    def __call__(self, x: Array) -> Array:
        return x + self.projection(self.attention(x))


class ResnetBlock(nnx.Module):
    """
    Residual Block for the Encoder and Decoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rngs (nnx.Rngs): RNGs for the module.
        dtype (DTypeLike): Data type for the module.
        param_dtype (DTypeLike): Data type for the module
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        if param_dtype is None:
            param_dtype = dtype

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nnx.GroupNorm(
            num_groups=32,
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=32,
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.conv2 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=(0, 0),
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            )

    def __call__(self, input_tensor: Array) -> Array:
        h = input_tensor
        h = self.norm1(h)
        h = jax.nn.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = jax.nn.swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            input_tensor = self.nin_shortcut(input_tensor)

        return input_tensor + h


class Downsample(nnx.Module):
    """
    Downsample Block for the Encoder.

    Args:
        in_channels (int): Number of input channels.
        rngs (nnx.Rngs): RNGs for the module.
        dtype (DTypeLike): Data type for the module.
        param_dtype (DTypeLike): Data type for the module parameters.

    Returns:
        Downsampled input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        if param_dtype is None:
            param_dtype = dtype

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(0, 0),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        x = jnp.pad(array=x, pad_width=(0, 1, 0, 1), mode="constant", constant_values=0)
        x = self.conv(x)
        return x


class Upsample(nnx.Module):
    """
    Upsample Block for the Decoder.

    Args:
        in_channels (int): Number of input channels.
        rngs (nnx.Rngs): RNGs for the module.
        dtype (DTypeLike): Data type for the module.
        param_dtype (DTypeLike): Data type for the module parameters.

    Returns:
        Upsampled input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        if param_dtype is None:
            param_dtype = dtype

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        x = interpolate(x, scale_factor=2.0, method="nearest")
        x = self.conv(x)
        return x


class Encoder(nnx.Module):
    """
    Encoder module for the AutoEncoder.

    Args:
        resolution (int): Resolution of the input tensor.
        in_channels (int): Number of input channels.
        ch (int): Number of channels.
        ch_mult (list[int]): List of channel multipliers.
        num_res_blocks (int): Number of residual blocks.
        z_channels (int): Number of latent channels.
        rngs (nnx.Rngs): RNGs for the module.
        dtype (DTypeLike): Data type for the module.
        param_dtype (DTypeLike): Data type for the module parameters.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.rngs = rngs

        self.dtype = dtype
        if param_dtype is None:
            self.param_dtype = dtype
        # downsampling
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        # FIXME: Use nnx.Sequential instead
        self.down = nnx.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            # FIXME: Use nnx.Sequential instead
            block = nnx.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        rngs=rngs,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
                block_in = block_out
            down = nnx.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(
                    in_channels=block_in,
                    rngs=rngs,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.middle = nnx.Sequential(
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                rngs=rngs,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ),
            AttnBlock(
                in_channels=block_in,
                rngs=rngs,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                rngs=rngs,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ),
        )

        # end
        self.norm_out = nnx.GroupNorm(
            num_groups=32,
            num_features=block_in,
            epsilon=1e-6,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=2 * z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.middle(h)
        # end
        h = self.norm_out(h)
        h = jax.nn.swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nnx.Module):
    """
    Decoder module for the AutoEncoder.

    Args:
        resolution (int): Resolution of the input tensor.
        in_channels (int): Number of input channels.
        ch (int): Number of channels.
        out_ch (int): Number of output channels.
        ch_mult (list[int]): List of channel multipliers.
        num_res_blocks (int): Number of residual blocks.
        z_channels (int): Number of latent channels.
        rngs (nnx.Rngs): RNGs for the module.
        dtype (DTypeLike): Data type for the module.
        param_dtype (DTypeLike): Data type for the module parameters.
    """

    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        self.dtype = dtype
        if param_dtype is None:
            self.param_dtype = dtype

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nnx.Conv(
            in_features=z_channels,
            out_features=block_in,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # middle
        self.middle = nnx.Sequential(
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                rngs=rngs,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ),
            AttnBlock(
                in_channels=block_in,
                rngs=rngs,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                rngs=rngs,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ),
        )

        # upsampling
        # FIXME: Use nnx.Sequential instead
        self.up = nnx.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            # FIXME: Use nnx.Sequential instead
            block = nnx.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        rngs=rngs,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
                block_in = block_out
            up = nnx.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(
                    in_channels=block_in,
                    rngs=rngs,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nnx.GroupNorm(
            num_groups=32,
            num_features=block_in,
            epsilon=1e-6,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, z: Array) -> Array:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.middle(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = jax.nn.swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nnx.Module):
    """
    AutoEncoder module.

    Args:
        params (AutoEncoderParams): Parameters for the AutoEncoder.
        rngs (nnx.Rngs): RNGs for the module.
        dtype (DTypeLike): Data type for the module.
        param_dtype (DTypeLike): Data type for the module parameters.
    """

    def __init__(
        self,
        params: AutoEncoderParams,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jax.dtypes.bfloat16,
        param_dtype: DTypeLike = None,
    ) -> None:
        self.dtype = dtype
        if param_dtype is None:
            self.param_dtype = dtype

        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            rngs=rngs,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        # FIXME: Provide a single key
        self.reg = DiagonalGaussian(key=rngs)  # noqa: ignore

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Array) -> Array:
        """
        Encodes the provided tensor.

        Args:
            x (Array): Input tensor.

        Returns:
            Array: Encoded tensor.
        """
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Array) -> Array:
        """
        Decodes the provided tensor.

        Args:
            z (Array): Encoded tensor.

        Returns:
            Array: Decoded tensor.
        """
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def __call__(self, x: Array) -> Array:
        """
        Forward pass for the AutoEncoder Module.

        Args:
            x (Array): Input tensor.

        Returns:
            Array
        """
        return self.decode(self.encode(x))
