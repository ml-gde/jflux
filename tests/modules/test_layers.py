import chex
import jax
import jax.numpy as jnp
import torch
from einops import rearrange, repeat
from flax import nnx
from flax.nnx import RMSNorm as JaxRMSNorm
from flux.modules.layers import DoubleStreamBlock as TorchDoubleStreamBlock
from flux.modules.layers import MLPEmbedder as TorchMLPEmbedder
from flux.modules.layers import Modulation as TorchModulation
from flux.modules.layers import QKNorm as TorchQKNorm
from flux.modules.layers import RMSNorm as TorchRMSNorm
from flux.modules.layers import SelfAttention as TorchSelfAttention
from flux.modules.layers import timestep_embedding as torch_timesetp_embedding

from jflux.modules.layers import DoubleStreamBlock as JaxDoubleStreamBlock
from jflux.modules.layers import EmbedND as JaxEmbedND
from jflux.modules.layers import MLPEmbedder as JaxMLPEmbedder
from jflux.modules.layers import Modulation as JaxModulation
from jflux.modules.layers import QKNorm as JaxQKNorm
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


class LayersTestCase(chex.TestCase):
    def test_timestep_embedding(self):
        t_vec_torch = torch.tensor([1.0], dtype=torch.float32)
        t_vec_jax = jnp.array([1.0], dtype=jnp.float32)

        jax_output = jax_timestep_embedding(t=t_vec_jax, dim=256)
        torch_output = torch_timesetp_embedding(t=t_vec_torch, dim=256)

        chex.assert_trees_all_close(
            jax_output,
            torch2jax(torch_output),
            rtol=1e-4,
            atol=1e-4,
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
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(1, in_dim), dtype=jnp.float32
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input,
            torch2jax(torch_input),
            rtol=1e-5,
            atol=1e-5,
        )

        # Forward pass
        torch_output = torch_mlp_embedder(torch_input)
        jax_output = jax_mlp_embedder(jax_input)

        # Assertions
        chex.assert_trees_all_close(
            jax_output,
            torch2jax(torch_output),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rms_norm(self):
        # Initialize the layer
        dim = 3
        rngs = nnx.Rngs(default=42)
        param_dtype = jnp.float32

        torch_rms_norm = TorchRMSNorm(dim=dim)
        jax_rms_norm = JaxRMSNorm(num_features=dim, rngs=rngs, param_dtype=param_dtype)

        # port the weights of the torch model into jax
        jax_rms_norm = port_rms_norm(
            jax_rms_norm=jax_rms_norm, torch_rms_norm=torch_rms_norm
        )

        # Generate random inputs
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, dim), dtype=jnp.float32
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input,
            torch2jax(torch_input),
            rtol=1e-5,
            atol=1e-5,
        )

        # Forward pass
        torch_output = torch_rms_norm(torch_input)
        jax_output = jax_rms_norm(jax_input)

        # Assertions
        chex.assert_trees_all_close(
            jax_output,
            torch2jax(torch_output),
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
        jax_q = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, seq_len, dim), dtype=jnp.float32
        )
        jax_k = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, seq_len, dim), dtype=jnp.float32
        )
        jax_v = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, seq_len, dim), dtype=jnp.float32
        )

        torch_q = torch.from_numpy(jax_q.__array__()).to(torch.float32)
        torch_k = torch.from_numpy(jax_k.__array__()).to(torch.float32)
        torch_v = torch.from_numpy(jax_v.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_q,
            torch2jax(torch_q),
            rtol=1e-5,
            atol=1e-5,
        )
        chex.assert_trees_all_close(
            jax_k,
            torch2jax(torch_k),
            rtol=1e-5,
            atol=1e-5,
        )
        chex.assert_trees_all_close(
            jax_v,
            torch2jax(torch_v),
            rtol=1e-5,
            atol=1e-5,
        )

        jax_output = jax_qknorm(q=jax_q, k=jax_k, v=jax_v)
        torch_output = torch_qknorm(q=torch_q, k=torch_k, v=torch_v)

        for i in range(len(jax_output)):
            chex.assert_trees_all_close(
                jax_output[i],
                torch2jax(torch_output[i]),
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
        jax_input = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(2, dim), dtype=jnp.float32
        )
        torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)

        chex.assert_trees_all_close(
            jax_input,
            torch2jax(torch_input),
            rtol=1e-5,
            atol=1e-5,
        )

        torch_output = torch_modulation(torch_input)
        jax_output = jax_modulation(jax_input)

        # Assertions
        for i in range(len(jax_output)):
            chex.assert_trees_all_close(
                jax_output[i].shift,
                torch2jax(torch_output[i].shift),
                rtol=1e-5,
                atol=1e-5,
            )
            chex.assert_trees_all_close(
                jax_output[i].scale,
                torch2jax(torch_output[i].scale),
                rtol=1e-5,
                atol=1e-5,
            )
            chex.assert_trees_all_close(
                jax_output[i].gate,
                torch2jax(torch_output[i].gate),
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
        jax_img = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(1, 4080, hidden_size), dtype=jnp.float32
        )
        jax_txt = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(1, 256, hidden_size), dtype=jnp.float32
        )
        jax_vec = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(1, hidden_size), dtype=jnp.float32
        )
        jax_pe = jax.random.normal(
            key=jax.random.PRNGKey(42), shape=(1, 1, 4336, 64, 2, 2), dtype=jnp.float32
        )

        torch_img = torch.from_numpy(jax_img.__array__()).to(torch.float32)
        torch_txt = torch.from_numpy(jax_txt.__array__()).to(torch.float32)
        torch_vec = torch.from_numpy(jax_vec.__array__()).to(torch.float32)
        torch_pe = torch.from_numpy(jax_pe.__array__()).to(torch.float32)

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

    def test_embednd(self):
        # noise
        bs, _, h, w = (1, 16, 96, 170)

        img_ids = jnp.zeros((h // 2, w // 2, 3))
        img_ids = img_ids.at[..., 1].set(jnp.arange(h // 2)[:, None])
        img_ids = img_ids.at[..., 2].set(jnp.arange(w // 2)[None, :])
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        # noise reshapes
        # img = jax.random.normal(shape=(1, 4080, 64), key=key, dtype=dtype)

        # prompt embedded from t5
        # txt = jax.random.normal(shape=(1, 512, 4096), key=key, dtype=dtype)
        txt_ids = jnp.zeros((bs, 512, 3))

        # clip embeddings
        # vec = jax.random.normal(shape=(1, 768), key=key, dtype=dtype)

        ids = jnp.concatenate((txt_ids, img_ids), axis=1)

        pe = JaxEmbedND(dim=128, theta=10_000, axes_dim=[16, 56, 56])(
            ids
        )  # dim = hidden_dim/num_head
        print(pe.shape)  # (1, 1, 4592, 64, 2, 2)
