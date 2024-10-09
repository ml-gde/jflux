import chex
import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from flax import nnx
from flux.modules.autoencoder import AttnBlock as TorchAttnBlock
from flux.modules.autoencoder import AutoEncoder as TorchAutoEncoder
from flux.modules.autoencoder import AutoEncoderParams as TorchAutoEncoderParams
from flux.modules.autoencoder import Decoder as TorchDecoder
from flux.modules.autoencoder import Downsample as TorchDownsample
from flux.modules.autoencoder import Encoder as TorchEncoder
from flux.modules.autoencoder import ResnetBlock as TorchResnetBlock
from flux.modules.autoencoder import Upsample as TorchUpsample

from jflux.modules.autoencoder import AttnBlock as JaxAttnBlock
from jflux.modules.autoencoder import AutoEncoder as JaxAutoEncoder
from jflux.modules.autoencoder import AutoEncoderParams as JaxAutoEncoderParams
from jflux.modules.autoencoder import Decoder as JaxDecoder
from jflux.modules.autoencoder import Downsample as JaxDownsample
from jflux.modules.autoencoder import Encoder as JaxEncoder
from jflux.modules.autoencoder import ResnetBlock as JaxResnetBlock
from jflux.modules.autoencoder import Upsample as JaxUpsample
from tests.utils import torch2jax


def port_attn_block(jax_attn_block: JaxAttnBlock, torch_attn_block: TorchAttnBlock):
    # port the norm
    jax_attn_block.norm.scale.value = torch2jax(torch_attn_block.norm.weight)
    jax_attn_block.norm.bias.value = torch2jax(torch_attn_block.norm.bias)

    # port the k, q, v layers
    jax_attn_block.k.kernel.value = torch2jax(
        rearrange(torch_attn_block.k.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_attn_block.k.bias.value = torch2jax(torch_attn_block.k.bias)

    jax_attn_block.q.kernel.value = torch2jax(
        rearrange(torch_attn_block.q.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_attn_block.q.bias.value = torch2jax(torch_attn_block.q.bias)

    jax_attn_block.v.kernel.value = torch2jax(
        rearrange(torch_attn_block.v.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_attn_block.v.bias.value = torch2jax(torch_attn_block.v.bias)

    # port the proj_out layer
    jax_attn_block.proj_out.kernel.value = torch2jax(
        rearrange(torch_attn_block.proj_out.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_attn_block.proj_out.bias.value = torch2jax(torch_attn_block.proj_out.bias)

    return jax_attn_block


def port_resent_block(
    jax_resnet_block: JaxResnetBlock, torch_resnet_block: TorchResnetBlock
):
    # port the norm
    jax_resnet_block.norm1.scale.value = torch2jax(torch_resnet_block.norm1.weight)
    jax_resnet_block.norm1.bias.value = torch2jax(torch_resnet_block.norm1.bias)

    jax_resnet_block.norm2.scale.value = torch2jax(torch_resnet_block.norm2.weight)
    jax_resnet_block.norm2.bias.value = torch2jax(torch_resnet_block.norm2.bias)

    # port the convs
    jax_resnet_block.conv1.kernel.value = torch2jax(
        rearrange(torch_resnet_block.conv1.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_resnet_block.conv1.bias.value = torch2jax(torch_resnet_block.conv1.bias)

    jax_resnet_block.conv2.kernel.value = torch2jax(
        rearrange(torch_resnet_block.conv2.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_resnet_block.conv2.bias.value = torch2jax(torch_resnet_block.conv2.bias)

    if jax_resnet_block.in_channels != jax_resnet_block.out_channels:
        jax_resnet_block.nin_shortcut.kernel.value = torch2jax(
            rearrange(torch_resnet_block.nin_shortcut.weight, "i o k1 k2 -> k1 k2 o i")
        )
        jax_resnet_block.nin_shortcut.bias.value = torch2jax(
            torch_resnet_block.nin_shortcut.bias
        )

    return jax_resnet_block


def port_downsample(jax_downsample: JaxDownsample, torch_downsample: TorchDownsample):
    # port the conv
    jax_downsample.conv.kernel.value = torch2jax(
        rearrange(torch_downsample.conv.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_downsample.conv.bias.value = torch2jax(torch_downsample.conv.bias)
    return jax_downsample


def port_upsample(jax_upsample: JaxUpsample, torch_upsample: TorchUpsample):
    # port the conv
    jax_upsample.conv.kernel.value = torch2jax(
        rearrange(torch_upsample.conv.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_upsample.conv.bias.value = torch2jax(torch_upsample.conv.bias)
    return jax_upsample


def port_encoder(jax_encoder: JaxEncoder, torch_encoder: TorchEncoder):
    # port downsampling
    jax_conv_in = jax_encoder.conv_in
    torch_conv_in = torch_encoder.conv_in

    jax_conv_in.kernel.value = torch2jax(
        rearrange(torch_conv_in.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_conv_in.bias.value = torch2jax(torch_conv_in.bias)

    # down
    jax_down = jax_encoder.down
    torch_down = torch_encoder.down

    for i in range(len(jax_down.layers)):
        # block
        jax_block = jax_down.layers[i].block
        torch_block = torch_down[i].block
        for j in range(len(jax_block.layers)):
            jax_resnet_block = jax_block.layers[j]
            torch_resnet_block = torch_block[j]
            jax_resnet_block = port_resent_block(
                jax_resnet_block=jax_resnet_block, torch_resnet_block=torch_resnet_block
            )

        # attn
        jax_attn = jax_down.layers[i].attn
        torch_attn = torch_down[i].attn
        for j in range(len(jax_attn.layers)):
            jax_attn_block = jax_attn.layers[j]
            torch_attn_block = torch_attn[j]
            jax_attn_block = port_attn_block(
                jax_attn_block=jax_attn_block, torch_attn_block=torch_attn_block
            )

        # downsample
        if i != jax_encoder.num_resolutions - 1:
            jax_downsample = jax_down.layers[i].downsample
            torch_downsample = torch_down[i].downsample
            jax_downsample = port_downsample(
                jax_downsample=jax_downsample, torch_downsample=torch_downsample
            )

    # mid
    jax_mid = jax_encoder.mid
    torch_mid = torch_encoder.mid

    jax_mid_block_1 = jax_mid.block_1
    torch_mid_block_1 = torch_mid.block_1

    jax_mid_block_1 = port_resent_block(
        jax_resnet_block=jax_mid_block_1, torch_resnet_block=torch_mid_block_1
    )

    jax_mid_attn_1 = jax_mid.attn_1
    torch_mid_attn_1 = torch_mid.attn_1

    jax_mid_attn_1 = port_attn_block(
        jax_attn_block=jax_mid_attn_1, torch_attn_block=torch_mid_attn_1
    )

    jax_mid_block_2 = jax_mid.block_2
    torch_mid_block_2 = torch_mid.block_2

    jax_mid_block_2 = port_resent_block(
        jax_resnet_block=jax_mid_block_2, torch_resnet_block=torch_mid_block_2
    )

    # norm out
    jax_norm_out = jax_encoder.norm_out
    torch_norm_out = torch_encoder.norm_out

    jax_norm_out.scale.value = torch2jax(torch_norm_out.weight)
    jax_norm_out.bias.value = torch2jax(torch_norm_out.bias)

    # conv out
    jax_conv_out = jax_encoder.conv_out
    torch_conv_out = torch_encoder.conv_out

    jax_conv_out.kernel.value = torch2jax(
        rearrange(torch_conv_out.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_conv_out.bias.value = torch2jax(torch_conv_out.bias)

    return jax_encoder


def port_decoder(jax_decoder: JaxDecoder, torch_decoder: TorchDecoder):
    # port downsampling
    jax_conv_in = jax_decoder.conv_in
    torch_conv_in = torch_decoder.conv_in

    jax_conv_in.kernel.value = torch2jax(
        rearrange(torch_conv_in.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_conv_in.bias.value = torch2jax(torch_conv_in.bias)

    # mid
    jax_mid = jax_decoder.mid
    torch_mid = torch_decoder.mid

    jax_mid_block_1 = jax_mid.block_1
    torch_mid_block_1 = torch_mid.block_1

    jax_mid_block_1 = port_resent_block(
        jax_resnet_block=jax_mid_block_1, torch_resnet_block=torch_mid_block_1
    )

    jax_mid_attn_1 = jax_mid.attn_1
    torch_mid_attn_1 = torch_mid.attn_1

    jax_mid_attn_1 = port_attn_block(
        jax_attn_block=jax_mid_attn_1, torch_attn_block=torch_mid_attn_1
    )

    jax_mid_block_2 = jax_mid.block_2
    torch_mid_block_2 = torch_mid.block_2

    jax_mid_block_2 = port_resent_block(
        jax_resnet_block=jax_mid_block_2, torch_resnet_block=torch_mid_block_2
    )

    # up
    jax_up = jax_decoder.up
    torch_up = torch_decoder.up

    for i in range(len(jax_up.layers)):
        # block
        jax_block = jax_up.layers[i].block
        torch_block = torch_up[i].block
        for j in range(len(jax_block.layers)):
            jax_resnet_block = jax_block.layers[j]
            torch_resnet_block = torch_block[j]
            jax_resnet_block = port_resent_block(
                jax_resnet_block=jax_resnet_block, torch_resnet_block=torch_resnet_block
            )

        # attn
        jax_attn = jax_up.layers[i].attn
        torch_attn = torch_up[i].attn
        for j in range(len(jax_attn.layers)):
            jax_attn_block = jax_attn.layers[j]
            torch_attn_block = torch_attn[j]
            jax_attn_block = port_attn_block(
                jax_attn_block=jax_attn_block, torch_attn_block=torch_attn_block
            )

        # upsample
        if i != 0:
            jax_upsample = jax_up.layers[i].upsample
            torch_upsample = torch_up[i].upsample
            jax_upsample = port_upsample(
                jax_upsample=jax_upsample, torch_upsample=torch_upsample
            )

    # norm out
    jax_norm_out = jax_decoder.norm_out
    torch_norm_out = torch_decoder.norm_out

    jax_norm_out.scale.value = torch2jax(torch_norm_out.weight)
    jax_norm_out.bias.value = torch2jax(torch_norm_out.bias)

    # conv out
    jax_conv_out = jax_decoder.conv_out
    torch_conv_out = torch_decoder.conv_out

    jax_conv_out.kernel.value = torch2jax(
        rearrange(torch_conv_out.weight, "i o k1 k2 -> k1 k2 o i")
    )
    jax_conv_out.bias.value = torch2jax(torch_conv_out.bias)

    return jax_decoder


def port_autoencoder(
    jax_autoencoder: JaxAutoEncoder, torch_autoencoder: TorchAutoEncoder
):
    jax_autoencoder.encoder = port_encoder(
        jax_encoder=jax_autoencoder.encoder,
        torch_encoder=torch_autoencoder.encoder,
    )
    jax_autoencoder.decoder = port_decoder(
        jax_decoder=jax_autoencoder.decoder,
        torch_decoder=torch_autoencoder.decoder,
    )
    return jax_autoencoder


class AutoEncodersTestCase(chex.TestCase):
    def test_attn_block(self):
        # Initialize layers
        in_channels = 32
        param_dtype = jnp.float32
        rngs = nnx.Rngs(default=42)

        torch_attn_block = TorchAttnBlock(in_channels=in_channels)
        jax_attn_block = JaxAttnBlock(
            in_channels=in_channels, rngs=rngs, param_dtype=param_dtype
        )

        # port the weights of the torch model into jax
        jax_attn_block = port_attn_block(
            jax_attn_block=jax_attn_block, torch_attn_block=torch_attn_block
        )

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, 32, 4, 4), dtype=jnp.float32
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input, torch2jax(torch_input), rtol=1e-5, atol=1e-5
        )

        # Forward pass
        torch_output = torch_attn_block(torch_input)
        jax_output = jax_attn_block(rearrange(jax_input, "b c h w -> b h w c"))

        # Assertions
        chex.assert_trees_all_close(
            rearrange(jax_output, "b h w c -> b c h w"),
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_resnet_block(self):
        # Initialize layers
        in_channels = 32
        out_channels = 64
        param_dtype = jnp.float32
        rngs = nnx.Rngs(default=42)

        torch_resnet_block = TorchResnetBlock(
            in_channels=in_channels, out_channels=out_channels
        )
        jax_resnet_block = JaxResnetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # port the weights of the torch model into jax
        jax_resnet_block = port_resent_block(
            jax_resnet_block=jax_resnet_block, torch_resnet_block=torch_resnet_block
        )

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, 32, 4, 4), dtype=jnp.float32
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input, torch2jax(torch_input), rtol=1e-5, atol=1e-5
        )

        # Forward pass
        torch_output = torch_resnet_block(torch_input)
        jax_output = jax_resnet_block(rearrange(jax_input, "b c h w -> b h w c"))

        # Assertions
        chex.assert_trees_all_close(
            rearrange(jax_output, "b h w c -> b c h w"),
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_downsample(self):
        # Initialize layers
        in_channels = 32
        param_dtype = jnp.float32
        rngs = nnx.Rngs(default=42)

        torch_downsample = TorchDownsample(in_channels=in_channels)
        jax_downsample = JaxDownsample(
            in_channels=in_channels,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # port the weights of the torch model into jax
        jax_downsample = port_downsample(
            jax_downsample=jax_downsample, torch_downsample=torch_downsample
        )

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, 32, 4, 4), dtype=jnp.float32
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input, torch2jax(torch_input), rtol=1e-5, atol=1e-5
        )

        # Forward pass
        torch_output = torch_downsample(torch_input)
        jax_output = jax_downsample(rearrange(jax_input, "b c h w -> b h w c"))

        # Assertions
        chex.assert_trees_all_close(
            rearrange(jax_output, "b h w c -> b c h w"),
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_upsample(self):
        # Initialize layers
        in_channels = 32
        param_dtype = jnp.float32
        rngs = nnx.Rngs(default=42)

        torch_upsample = TorchUpsample(in_channels=in_channels)
        jax_upsample = JaxUpsample(
            in_channels=in_channels,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # port the weights of the torch model into jax
        jax_upsample = port_upsample(
            jax_upsample=jax_upsample, torch_upsample=torch_upsample
        )

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, 32, 4, 4), dtype=jnp.float32
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input, torch2jax(torch_input), rtol=1e-5, atol=1e-5
        )

        # Forward pass
        torch_output = torch_upsample(torch_input)
        jax_output = jax_upsample(rearrange(jax_input, "b c h w -> b h w c"))

        # Assertions
        chex.assert_trees_all_close(
            rearrange(jax_output, "b h w c -> b c h w"),
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_encoder(self):
        # Initialize encoder
        resolution = 32
        in_channels = 3
        ch = 32
        ch_mult = [1, 2]
        num_res_blocks = 2
        z_channels = 8
        param_dtype = jnp.float32
        rngs = nnx.Rngs(default=42)

        torch_encoder = TorchEncoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        jax_encoder = JaxEncoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # port the weights of the torch model into jax
        jax_encoder = port_encoder(jax_encoder=jax_encoder, torch_encoder=torch_encoder)

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42),
            shape=(1, in_channels, resolution, resolution),
            dtype=jnp.float32,
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input, torch2jax(torch_input), rtol=1e-5, atol=1e-5
        )

        # Forward pass
        torch_output = torch_encoder(torch_input)
        jax_output = jax_encoder(rearrange(jax_input, "b c h w -> b h w c"))

        # Assertions
        chex.assert_trees_all_close(
            rearrange(jax_output, "b h w c -> b c h w"),
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_decoder(self):
        # Initialize decoder
        resolution = 32
        in_channels = 3
        ch = 32
        out_ch = 3
        ch_mult = [1, 2]
        num_res_blocks = 2
        z_channels = 8
        param_dtype = jnp.float32
        rngs = nnx.Rngs(default=42)

        torch_decoder = TorchDecoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
        )
        jax_decoder = JaxDecoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # port the weights of the torch model into jax
        jax_decoder = port_decoder(jax_decoder=jax_decoder, torch_decoder=torch_decoder)

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42),
            shape=(
                1,
                z_channels,
                resolution // len(ch_mult),
                resolution // len(ch_mult),
            ),
            dtype=jnp.float32,
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input, torch2jax(torch_input), rtol=1e-5, atol=1e-5
        )

        # Forward pass
        torch_output = torch_decoder(torch_input)
        jax_output = jax_decoder(rearrange(jax_input, "b c h w -> b h w c"))

        # Assertions
        chex.assert_trees_all_close(
            rearrange(jax_output, "b h w c -> b c h w"),
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )

    # TODO (ariG23498): Test the gaussians of torch and jax
    # def test_diagonal_gaussian(self):

    def test_autoencoder(self):
        # Initialize the autoencoder
        resolution = 32
        in_channels = 3
        ch = 32
        out_ch = 3
        ch_mult = [1, 2]
        num_res_blocks = 2
        z_channels = 8
        param_dtype = jnp.float32
        rngs = nnx.Rngs(default=42)
        scale_factor = 0.3611
        shift_factor = 0.1159

        torch_params = TorchAutoEncoderParams(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
        )
        jax_params = JaxAutoEncoderParams(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        torch_autoencoder = TorchAutoEncoder(params=torch_params)
        jax_autoencoder = JaxAutoEncoder(params=jax_params)

        # Make things deterministic
        torch_autoencoder.reg.sample = False
        jax_autoencoder.reg.sample = False

        # port
        jax_autoencoder = port_autoencoder(
            jax_autoencoder=jax_autoencoder,
            torch_autoencoder=torch_autoencoder,
        )

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42),
            shape=(1, in_channels, resolution, resolution),
            dtype=jnp.float32,
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input, torch2jax(torch_input), rtol=1e-5, atol=1e-5
        )

        # forward pass
        torch_output = torch_autoencoder(torch_input)
        jax_output = jax_autoencoder(jax_input)

        # Assertions
        chex.assert_trees_all_close(
            jax_output,
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )
