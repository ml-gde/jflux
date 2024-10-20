import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import jax
import jax.numpy as jnp
import numpy as np
import torch
from einops import rearrange
from fire import Fire
from flax import nnx
from PIL import Image

from jflux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from jflux.util import configs, load_ae, load_clip, load_flow_model, load_t5


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_device_type():
    """Returns the type of JAX device being used.

    Returns:
      str: "gpu", "tpu", or "cpu"
    """
    try:
        device_kind = jax.devices()[0].device_kind
        if "gpu" in device_kind.lower():
            return "gpu"
        elif "tpu" in device_kind.lower():
            return "tpu"
        else:
            return "cpu"
    except IndexError:
        return "cpu"  # No devices found, likely using CPU


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = (
        "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    )
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting seed to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


def main(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"JFLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    offload: bool = True,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps
            (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    device_type = get_device_type()
    print(f"Using {device_type} device")

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [
            fn
            for fn in iglob(output_name.format(idx="*"))
            if re.search(r"img_[0-9]+\.jpg$", fn)
        ]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init t5 and clip on the gpu (torch models)
    t5 = load_t5(
        device="cuda" if device_type == "gpu" else "cpu",
        max_length=256 if name == "flux-schnell" else 512,
    )
    clip = load_clip(device="cuda" if device_type == "gpu" else "cpu")

    # init flux and ae on the cpu
    model = load_flow_model(name, device="cpu" if offload else device_type)
    ae = load_ae(name, device="cpu" if offload else device_type)

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    while opts is not None:
        if opts.seed is None:
            opts.seed = jax.random.PRNGKey(seed=102333)
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            num_samples=1,
            height=opts.height,
            width=opts.width,
            dtype=jnp.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None

        inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)
        timesteps = get_schedule(
            num_steps=opts.num_steps,
            image_seq_len=inp["img"].shape[1],
            shift=(name != "flux-schnell"),
        )

        if offload:
            # move t5 and clip to cpu
            t5, clip = t5.cpu(), clip.cpu()
            if device_type == "gpu":
                torch.cuda.empty_cache()

            # load model to device
            model_state = nnx.state(model)
            model_state = jax.device_put(model_state, jax.devices(device_type)[0])
            nnx.update(model, model_state)
            jax.clear_caches()

        # denoise initial noise
        x = denoise(
            model,
            **inp,
            timesteps=timesteps,
            guidance=opts.guidance,
        )
        if offload:
            # move model to cpu
            model_state = nnx.state(model)
            model_state = jax.device_put(model_state, jax.devices("cpu")[0])
            nnx.update(model, model_state)
            jax.clear_caches()

            # move ae decoder to gpu
            ae_decoder_state = nnx.state(ae.decoder)
            ae_decoder_state = jax.device_put(
                ae_decoder_state, jax.devices(device_type)[0]
            )
            nnx.update(ae.decoder, ae_decoder_state)
            jax.clear_caches()

        # decode latents to pixel space
        x = unpack(x=x.astype(jnp.float32), height=opts.height, width=opts.width)
        x = ae.decode(x)
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        # bring into PIL format and save
        x = x.clip(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        x = 127.5 * (x + 1.0)
        x_numpy = np.array(x.astype(jnp.uint8))
        img = Image.fromarray(x_numpy)

        img.save(fn, quality=95, subsampling=0)
        idx += 1

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        else:
            opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
