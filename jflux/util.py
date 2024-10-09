import os
from dataclasses import dataclass

import torch  # need for t5 and clip
from flax import nnx
from huggingface_hub import hf_hub_download
from jax import numpy as jnp
from safetensors import safe_open

from jflux.model import Flux, FluxParams
from jflux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from jflux.modules.conditioner import HFEmbedder
from jflux.port import port_autoencoder, port_flux


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.bfloat16,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, hf_download: bool = True) -> Flux:
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    model = Flux(params=configs[name].params)

    if ckpt_path is not None:
        tensors = {}
        with safe_open(ckpt_path, framework="flax") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        model = port_flux(flux=model, tensors=tensors)
        del tensors
    return model


def load_t5() -> HFEmbedder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HFEmbedder(
        "google/t5-v1_1-xxl", max_length=512, torch_dtype=torch.bfloat16
    ).to(device)


def load_clip() -> HFEmbedder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HFEmbedder(
        "openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16
    ).to(device)


def load_ae(name: str, hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    ae = AutoEncoder(params=configs[name].ae_params)

    if ckpt_path is not None:
        tensors = {}
        with safe_open(ckpt_path, framework="flax") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        ae = port_autoencoder(autoencoder=ae, tensors=tensors)
        del tensors
    return ae
