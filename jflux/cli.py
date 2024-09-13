import os
import re
import time
from dataclasses import dataclass
from glob import iglob

from fire import Fire

import jax
from jflux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from jflux.util import (
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)


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
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "gpu" if jax.device_get("gpu") else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    # TODO: JAX variant of offloading to CPU
    offload: bool = False,
    output_dir: str = "output",
) -> None:
    """
    Sample the flux model.

    Args:
        name(str): Name of the model to use. Choose from 'flux-schnell' or 'flux-dev'.
        width(int): Width of the generated image.
        height(int): Height of the generated image.
        seed(int, optional): Seed for the random number generator.
        prompt(str): Text prompt to generate the image from.
        device(str): Device to run the model on. Choose from 'cpu' or 'gpu'.
        num_steps(int, optional): Number of steps to run the model for.
        loop(bool): Whether to loop the sampling process.
        guidance(float, optional): Guidance for the model, defaults to 3.5.
        offload(bool, optional): Whether to offload the model to CPU, defaults to False.
        output_dir(str, optional): Directory to save the output images in, defaults to 'output'.
    """

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    jax_device = jax.devices(device)
    if len(jax_device) == 1:
        jax_device = jax_device[0]
    else:
        # TODO (ariG23498)
        # this will be when there are more than
        # one devices to work on
        pass

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

    # init all components
    import sys

    sys.exit(0)
    t5 = load_t5(jax_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(jax_device)
    model = load_flow_model(name, device="cpu" if offload else jax_device)
    ae = load_ae(name, device="cpu" if offload else jax_device)

    # TODO (ariG23498)
    # rngs = nnx.Rngs(0)
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
            # TODO (ariG23498)
            # set the rng seed
            # opts.seed = rng.seed()
            pass
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=jax_device,
            dtype=jax.dtypes.bfloat16,
            seed=opts.seed,  # type: ignore
        )
        opts.seed = None
        # TODO: JAX variant of offloading to CPU
        # if offload:
        #     ae = ae.cpu()
        #     torch.cuda.empty_cache()
        #     t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(
            opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell")
        )

        # offload TEs to CPU, load model to gpu
        # TODO: JAX variant of offloading to CPU
        # if offload:
        #     t5, clip = t5.cpu(), clip.cpu()
        #     torch.cuda.empty_cache()
        #     model = model.to(torch_device)

        # denoise initial noise
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # offload model, load autoencoder to gpu
        # TODO: JAX variant of offloading to CPU
        # if offload:
        #     model.cpu()
        #     torch.cuda.empty_cache()
        #     ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        x = ae.decode(x).astype(dtype=jax.dtypes.bfloat16)  # noqa
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        else:
            opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
