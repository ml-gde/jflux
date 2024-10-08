import jax.numpy as jnp
import numpy as np
import jax
import torch
from einops import rearrange
from flax import nnx
from flux.modules.layers import DoubleStreamBlock as TorchDoubleStreamBlock
from flux.modules.layers import MLPEmbedder as TorchMLPEmbedder
from flux.modules.layers import Modulation as TorchModulation
from flux.modules.layers import QKNorm as TorchQKNorm
from flux.modules.layers import RMSNorm as TorchRMSNorm
from flux.modules.layers import SelfAttention as TorchSelfAttention
from flux.modules.layers import timestep_embedding as torch_timesetp_embedding

from jflux.modules.layers import DoubleStreamBlock as JaxDoubleStreamBlock
from jflux.modules.layers import MLPEmbedder as JaxMLPEmbedder
from jflux.modules.layers import Modulation as JaxModulation
from jflux.modules.layers import QKNorm as JaxQKNorm
from jflux.modules.layers import RMSNorm as JaxRMSNorm
from jflux.modules.layers import SelfAttention as JaxSelfAttention
from jflux.modules.layers import timestep_embedding as jax_timestep_embedding

from tests.utils import torch2jax


def port_mlp_embedder(
    jax_mlp_embedder: JaxMLPEmbedder, torch_mlp_embedder: JaxMLPEmbedder
):
    # linear layers
    jax_mlp_embedder.in_layer.kernel.value = torch2jax(
        rearrange(torch_mlp_embedder.in_layer.weight, "i o -> o i")
    )
    jax_mlp_embedder.in_layer.bias.value = torch2jax(torch_mlp_embedder.in_layer.bias)

    jax_mlp_embedder.out_layer.kernel.value = torch2jax(
        rearrange(torch_mlp_embedder.out_layer.weight, "i o -> o i")
    )
    jax_mlp_embedder.out_layer.bias.value = torch2jax(torch_mlp_embedder.out_layer.bias)
    return jax_mlp_embedder


def port_rms_norm(jax_rms_norm: JaxRMSNorm, torch_rms_norm: TorchRMSNorm):
    jax_rms_norm.scale.value = torch2jax(torch_rms_norm.scale)
    return jax_rms_norm


def port_qknorm(jax_qknorm: JaxQKNorm, torch_qknorm: TorchQKNorm):
    # query norm
    jax_qknorm.query_norm = port_rms_norm(
        jax_rms_norm=jax_qknorm.query_norm,
        torch_rms_norm=torch_qknorm.query_norm,
    )
    # key norm
    jax_qknorm.key_norm = port_rms_norm(
        jax_rms_norm=jax_qknorm.key_norm,
        torch_rms_norm=torch_qknorm.key_norm,
    )

    return jax_qknorm


def port_modulation(
    jax_modulation: JaxModulation,
    torch_modulation: TorchModulation,
):
    jax_modulation.lin.kernel.value = torch2jax(
        rearrange(torch_modulation.lin.weight, "i o -> o i")
    )
    jax_modulation.lin.bias.value = torch2jax(torch_modulation.lin.bias)
    return jax_modulation


def port_self_attention(
    jax_self_attention: JaxSelfAttention,
    torch_self_attention: TorchSelfAttention,
):
    jax_self_attention.qkv.kernel.value = torch2jax(
        rearrange(torch_self_attention.qkv.weight, "i o -> o i")
    )

    jax_self_attention.qkv.bias.value = torch2jax(torch_self_attention.qkv.bias)

    jax_self_attention.proj.kernel.value = torch2jax(
        rearrange(torch_self_attention.proj.weight, "i o -> o i")
    )

    jax_self_attention.proj.bias.value = torch2jax(torch_self_attention.proj.bias)

    return jax_self_attention


def port_double_stream_block(
    jax_double_stream_block: JaxDoubleStreamBlock,
    torch_double_stream_block: TorchDoubleStreamBlock,
):
    jax_double_stream_block.img_mod = port_modulation(
        jax_modulation=jax_double_stream_block.img_mod,
        torch_modulation=torch_double_stream_block.img_mod,
    )

    jax_double_stream_block.img_attn = port_self_attention(
        jax_self_attention=jax_double_stream_block.img_attn,
        torch_self_attention=torch_double_stream_block.img_attn,
    )

    jax_double_stream_block.img_mlp.layers[0].kernel.value = torch2jax(
        rearrange(torch_double_stream_block.img_mlp[0].weight, "i o -> o i")
    )
    jax_double_stream_block.img_mlp.layers[0].bias.value = torch2jax(
        torch_double_stream_block.img_mlp[0].bias
    )

    jax_double_stream_block.img_mlp.layers[2].kernel.value = torch2jax(
        rearrange(torch_double_stream_block.img_mlp[2].weight, "i o -> o i")
    )
    jax_double_stream_block.img_mlp.layers[2].bias.value = torch2jax(
        torch_double_stream_block.img_mlp[2].bias
    )

    jax_double_stream_block.txt_mod = port_modulation(
        jax_modulation=jax_double_stream_block.txt_mod,
        torch_modulation=torch_double_stream_block.txt_mod,
    )

    jax_double_stream_block.txt_attn = port_self_attention(
        jax_self_attention=jax_double_stream_block.txt_attn,
        torch_self_attention=torch_double_stream_block.txt_attn,
    )

    jax_double_stream_block.txt_mlp.layers[0].kernel.value = torch2jax(
        rearrange(torch_double_stream_block.txt_mlp[0].weight, "i o -> o i")
    )
    jax_double_stream_block.txt_mlp.layers[0].bias.value = torch2jax(
        torch_double_stream_block.txt_mlp[0].bias
    )

    jax_double_stream_block.txt_mlp.layers[2].kernel.value = torch2jax(
        rearrange(torch_double_stream_block.txt_mlp[2].weight, "i o -> o i")
    )
    jax_double_stream_block.txt_mlp.layers[2].bias.value = torch2jax(
        torch_double_stream_block.txt_mlp[2].bias
    )

    return jax_double_stream_block


class LayersTestCase(np.testing.TestCase):
    def test_timestep_embedding(self):
        t_vec_torch = torch.tensor([1.0], dtype=torch.float32)
        t_vec_jax = jnp.array([1.0], dtype=jnp.float32)

        jax_output = jax_timestep_embedding(t=t_vec_jax, dim=256)
        torch_output = torch_timesetp_embedding(t=t_vec_torch, dim=256)
        print(jax_output.shape)
        np.testing.assert_allclose(
            np.array(jax_output), torch_output.numpy(), atol=1e-4, rtol=1e-4
        )

    def test_mlp_embedder(self):
        # Initialize layers
        in_dim = 256
        hidden_dim = 3072
        rngs = nnx.Rngs(default=42)
        param_dtype = jnp.float32

        torch_mlp_embedder = TorchMLPEmbedder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
        )
        jax_mlp_embedder = JaxMLPEmbedder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # port the weights of the torch model into jax
        jax_mlp_embedder = port_mlp_embedder(
            jax_mlp_embedder=jax_mlp_embedder, torch_mlp_embedder=torch_mlp_embedder
        )

        # Generate random inputs
        np_input = np.random.randn(1, in_dim).astype(np.float32)
        jax_input = jnp.array(np_input, dtype=jnp.float32)
        torch_input = torch.from_numpy(np_input).to(torch.float32)

        np.testing.assert_allclose(np.array(jax_input), torch_input.numpy())

        # Forward pass
        torch_output = torch_mlp_embedder(torch_input)
        jax_output = jax_mlp_embedder(jax_input)

        # Assertions
        np.testing.assert_allclose(
            np.array(jax_output),
            torch_output.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rms_norm(self):
        # Initialize the layer
        dim = 3
        rngs = nnx.Rngs(default=42)
        param_dtype = jnp.float32

        torch_rms_norm = TorchRMSNorm(dim=dim)
        jax_rms_norm = JaxRMSNorm(dim=dim, rngs=rngs, param_dtype=param_dtype)

        # port the weights of the torch model into jax
        jax_rms_norm = port_rms_norm(
            jax_rms_norm=jax_rms_norm, torch_rms_norm=torch_rms_norm
        )

        # Generate random inputs
        np_input = np.random.randn(2, dim).astype(np.float32)
        jax_input = jnp.array(np_input, dtype=jnp.float32)
        torch_input = torch.from_numpy(np_input).to(torch.float32)

        np.testing.assert_allclose(np.array(jax_input), torch_input.numpy())

        # Forward pass
        torch_output = torch_rms_norm(torch_input)
        jax_output = jax_rms_norm(jax_input)

        # Assertions
        np.testing.assert_allclose(
            np.array(jax_output),
            torch_output.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_qknorm(self):
        # Initialize the layer
        dim = 16
        seq_len = 4
        rngs = nnx.Rngs(default=42)
        param_dtype = jnp.float32

        torch_qknorm = TorchQKNorm(dim=dim)
        jax_qknorm = JaxQKNorm(dim=dim, rngs=rngs, param_dtype=param_dtype)

        # port the model
        jax_qknorm = port_qknorm(jax_qknorm=jax_qknorm, torch_qknorm=torch_qknorm)

        # Generate random inputs
        np_q = np.random.randn(2, seq_len, dim).astype(np.float32)
        np_k = np.random.randn(2, seq_len, dim).astype(np.float32)
        np_v = np.random.randn(2, seq_len, dim).astype(np.float32)

        jax_q = jnp.array(np_q, dtype=jnp.float32)
        torch_q = torch.from_numpy(np_q).to(torch.float32)

        jax_k = jnp.array(np_k, dtype=jnp.float32)
        torch_k = torch.from_numpy(np_k).to(torch.float32)

        jax_v = jnp.array(np_v, dtype=jnp.float32)
        torch_v = torch.from_numpy(np_v).to(torch.float32)

        np.testing.assert_allclose(np.array(jax_q), torch_q.numpy())
        np.testing.assert_allclose(np.array(jax_k), torch_k.numpy())

        jax_output = jax_qknorm(q=jax_q, k=jax_k, v=jax_v)
        torch_output = torch_qknorm(q=torch_q, k=torch_k, v=torch_v)

        np.testing.assert_allclose(
            np.array(jax_output[0]),
            torch_output[0].detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(jax_output[1]),
            torch_output[1].detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_modulation(self):
        # Initialize the layer
        dim = 4
        rngs = nnx.Rngs(default=42)
        param_dtype = jnp.float32

        torch_modulation = TorchModulation(dim=dim, double=True)
        jax_modulation = JaxModulation(
            dim=dim, double=True, rngs=rngs, param_dtype=param_dtype
        )

        jax_modulation = port_modulation(
            jax_modulation=jax_modulation,
            torch_modulation=torch_modulation,
        )

        # Generate random inputs
        np_input = np.random.randn(2, dim).astype(np.float32)
        jax_input = jnp.array(np_input, dtype=jnp.float32)
        torch_input = torch.from_numpy(np_input).to(torch.float32)

        np.testing.assert_allclose(np.array(jax_input), torch_input.numpy())

        torch_output = torch_modulation(torch_input)
        jax_output = jax_modulation(jax_input)

        # Assertions
        for i in range(2):
            np.testing.assert_allclose(
                np.array(jax_output[i].shift),
                torch_output[i].shift.detach().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.array(jax_output[i].scale),
                torch_output[i].scale.detach().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.array(jax_output[i].gate),
                torch_output[i].gate.detach().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_double_stream_block(self):
        # Initialize layer
        hidden_size = 3072
        num_heads = 24
        mlp_ratio = 4.0
        qkv_bias = True
        rngs = nnx.Rngs(default=42)
        param_dtype = jnp.float32

        # Initialize the DoubleStreamBlock
        torch_double_stream_block = TorchDoubleStreamBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )
        jax_double_stream_block = JaxDoubleStreamBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        jax_double_stream_block = port_double_stream_block(
            jax_double_stream_block=jax_double_stream_block,
            torch_double_stream_block=torch_double_stream_block,
        )

        # Create the dummy inputs
        np_img = np.random.randn(1, 4080, hidden_size).astype(np.float32)
        np_txt = np.random.randn(1, 256, hidden_size).astype(np.float32)
        np_vec = np.random.randn(1, hidden_size).astype(np.float32)
        np_pe = np.random.randn(1, 1, 4336, 64, 2, 2).astype(np.float32)

        jax_img = jnp.array(np_img, dtype=jnp.float32)
        jax_txt = jnp.array(np_txt, dtype=jnp.float32)
        jax_vec = jnp.array(np_vec, dtype=jnp.float32)
        jax_pe = jnp.array(np_pe, dtype=jnp.float32)

        torch_img = torch.from_numpy(np_img).to(torch.float32)
        torch_txt = torch.from_numpy(np_txt).to(torch.float32)
        torch_vec = torch.from_numpy(np_vec).to(torch.float32)
        torch_pe = torch.from_numpy(np_pe).to(torch.float32)

        # Forward pass through the DoubleStreamBlock
        torch_img_out, torch_txt_out = torch_double_stream_block(
            img=torch_img,
            txt=torch_txt,
            vec=torch_vec,
            pe=torch_pe,
        )
        jax_img_out, jax_txt_out = jax_double_stream_block(
            img=jax_img,
            txt=jax_txt,
            vec=jax_vec,
            pe=jax_pe,
        )
