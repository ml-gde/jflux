from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp
from einops import rearrange
from flax.experimental import nnx
from flax.experimental.nnx import swish


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
    def __init__(self, in_channels: int, rngs=nnx.Rngs):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nnx.GroupNorm(
            num_groups=32, num_features=in_channels, epsilon=1e-6, affine=True
        )

        self.q = nnx.Conv(in_channels, in_channels, kernel_size=1)
        self.k = nnx.Conv(in_channels, in_channels, kernel_size=1)
        self.v = nnx.Conv(in_channels, in_channels, kernel_size=1)
        self.proj_out = nnx.Conv(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Array) -> Array:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nnx.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Array) -> Array:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, rngs=nnx.Rngs):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nnx.GroupNorm(
            num_groups=32, num_features=in_channels, epsilon=1e-6, affine=True
        )
        self.conv1 = nnx.Conv(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nnx.Conv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nnx.Conv(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
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
    def __init__(self, in_channels: int, rngs=nnx.Rngs):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nnx.Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Array):
        pad = (0, 1, 0, 1)
        x = jnp.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nnx.Module):
    def __init__(self, in_channels: int, rngs=nnx.Rngs):
        super().__init__()
        self.conv = nnx.Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Array):
        x = jnp.interp(x, scale_factor=2.0, mode="nearest")
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
        rngs=nnx.Rngs,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nnx.Conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = []
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
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
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nnx.GroupNorm(
            num_groups=32, num_features=block_in, epsilon=1e-6, affine=True
        )
        self.conv_out = nnx.Conv(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Array) -> Array:
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
        h = swish(h)
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
        rngs=nnx.Rngs,
    ):
        super().__init__()
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
        self.conv_in = nnx.Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nnx.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
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
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nnx.Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Array) -> Array:
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
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nnx.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1, rngs=nnx.Rngs):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Array) -> Array:
        mean, logvar = jnp.split(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = jnp.exp(0.5 * logvar)
            return mean + std * jax.random.normal(shape=mean.shape)
        else:
            return mean


class AutoEncoder(nnx.Module):
    def __init__(self, params: AutoEncoderParams, rngs=nnx.Rngs):
        super().__init__()
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

    def forward(self, x: Array) -> Array:
        return self.decode(self.encode(x))
