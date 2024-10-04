from dataclasses import dataclass

import jax
import jax.numpy as jnp
from chex import Array
from einops import rearrange
from flax import nnx
from jax.typing import DTypeLike


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


def swish(x: Array) -> Array:
    return nnx.swish(x)


class AttnBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jax.dtypes.bfloat16,
    ) -> None:
        self.norm = nnx.GroupNorm(
            num_groups=32,
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.q = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.k = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.v = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.proj_out = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def attention(self, h_: Array) -> Array:
        # Apply Group Norm
        h_ = self.norm(h_)
        # Calculate Query, Key and Values
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for JAX Attention impl
        b, h, w, c = q.shape
        q = rearrange(q, "b h w c-> b (h w) 1 c")
        k = rearrange(k, "b h w c-> b (h w) 1 c")
        v = rearrange(v, "b h w c-> b (h w) 1 c")

        # Calculate Attention
        h_ = nnx.dot_product_attention(q, k, v)

        return rearrange(h_, "b (h w) 1 c -> b h w c", h=h, w=w, c=c, b=b)

    def __call__(self, x: Array) -> Array:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jax.dtypes.bfloat16,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nnx.GroupNorm(
            num_groups=32,
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=32,
            num_features=out_channels,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                rngs=rngs,
                param_dtype=param_dtype,
            )

    def __call__(self, x: Array) -> Array:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jax.dtypes.bfloat16,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(0, 0),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        # no padding for height and channel, padding for height and width
        pad_width = ((0, 0), (0, 1), (0, 1), (0, 0))
        x = jnp.pad(array=x, pad_width=pad_width, mode="constant", constant_values=0)
        x = self.conv(x)
        return x


class Upsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jax.dtypes.bfloat16,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        # Assuming `x` is a 4D tensor with shape (batch, height, width, channels)
        scale_factor = 2.0
        b, h, w, c = x.shape
        new_height = int(h * scale_factor)
        new_width = int(w * scale_factor)
        new_shape = (b, new_height, new_width, c)

        # Resize using nearest-neighbor interpolation
        x = jax.image.resize(x, new_shape, method='nearest')
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
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            blocks = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                blocks.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        rngs=rngs,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
                block_in = block_out
            if i_level != self.num_resolutions - 1:
                blocks.append(
                    Downsample(
                        in_channels=block_in,
                        rngs=rngs,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
                curr_res = curr_res // 2
            self.down = nnx.Sequential(*blocks)

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
        self.up = nnx.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            blocks = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                blocks.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        rngs=rngs,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
                block_in = block_out

            upsample_module = [*blocks]
            if i_level != 0:
                upsample_module.append(
                    Upsample(
                        in_channels=block_in,
                        rngs=rngs,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
                curr_res = curr_res * 2

            self.up = nnx.Sequential(*upsample_module)

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
