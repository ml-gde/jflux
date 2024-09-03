import os
import re
import time
from dataclasses import dataclass
from glob import iglob

from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

import jax
from flax import nnx
from jflux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from jflux.util import (
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


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
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
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
    t5 = load_t5(jax_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(jax_device)
    model = load_flow_model(name, device="cpu" if offload else jax_device)
    ae = load_ae(name, device="cpu" if offload else jax_device)

    rngs = nnx.Rngs(0)
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

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
        seed=opts.seed,
    )
    opts.seed = None
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(torch_device), clip.to(torch_device)
    inp = prepare(t5, clip, x, prompt=opts.prompt)
    timesteps = get_schedule(
        opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell")
    )

    # offload TEs to CPU, load model to gpu
    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    # denoise initial noise
    x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

    # offload model, load autoencoder to gpu
    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    # decode latents to pixel space
    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)
    t1 = time.perf_counter()

    fn = output_name.format(idx=idx)
    print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

    if nsfw_score < NSFW_THRESHOLD:
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        idx += 1
    else:
        print("Your generated image may contain NSFW content.")

    if loop:
        print("-" * 80)
        opts = parse_prompt(opts)
    else:
        opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
