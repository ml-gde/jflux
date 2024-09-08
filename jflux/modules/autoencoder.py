from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from flax import nnx
from einops import rearrange

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
    def __init__(self, in_channels: int, rngs: nnx.Rngs) -> None:
        self.in_channels = in_channels

        self.norm = nnx.GroupNorm(
            num_groups=32, num_features=in_channels, epsilon=1e-6, rngs=rngs
        )

        self.query_layer = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self.key_layer = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self.value_layer = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self.projection = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            rngs=rngs,
        )

    def attention(self, input_tensor: Array) -> Array:
        # Apply Group Norm
        input_tensor = self.norm(input_tensor)

        # Calculate Query, Key and Values
        query = self.query_layer(input_tensor)
        key = self.key_layer(input_tensor)
        value = self.value_layer(input_tensor)

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
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs) -> None:
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nnx.GroupNorm(
            num_groups=32, num_features=in_channels, epsilon=1e-6, rngs=rngs
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=32, num_features=in_channels, epsilon=1e-6, rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=(0, 0),
                rngs=rngs,
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
    def __init__(self, in_channels: int, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(0, 0),
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        x = jnp.pad(array=x, pad_width=(0, 1, 0, 1), mode="constant", constant_values=0)
        x = self.conv(x)
        return x


class Upsample(nnx.Module):
    def __init__(self, in_channels: int, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        x = interpolate(x, scale_factor=2.0, method="nearest")
        x = self.conv(x)
        return x


class Encoder(nnx.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        rngs: nnx.Rngs,
    ):
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nnx.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nnx.ModuleList()
            attn = nnx.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nnx.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nnx.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, rngs=rngs
        )
        self.mid.attn_1 = AttnBlock(in_channels=block_in, rngs=rngs)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, rngs=rngs
        )

        # end
        self.norm_out = nnx.GroupNorm(
            num_groups=32, num_features=block_in, epsilon=1e-6, rngs=rngs
        )
        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=2 * z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
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
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = jax.nn.swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nnx.Module):
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
    ):
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

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
        )

        # middle
        self.mid = nnx.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nnx.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nnx.ModuleList()
            attn = nnx.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nnx.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nnx.GroupNorm(
            num_groups=32, num_features=block_in, epsilon=1e-6, rngs=rngs
        )
        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )

    def __call__(self, z: Array) -> Array:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

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


class DiagonalGaussian(nnx.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        self.sample = sample
        self.chunk_dim = chunk_dim

    def __call__(self, z: Array) -> Array:
        mean, logvar = jnp.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = jnp.exp(0.5 * logvar)
            return mean + std * jnp.randn_like(mean)
        else:
            return mean


class AutoEncoder(nnx.Module):
    def __init__(self, params: AutoEncoderParams):
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Array) -> Array:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Array) -> Array:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def __call__(self, x: Array) -> Array:
        return self.decode(self.encode(x))
