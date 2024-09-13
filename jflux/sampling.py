import math
from typing import Callable

from einops import rearrange, repeat

import jax
from jax.image import ResizeMethod
from jax import numpy as jnp
from chex import Array
from jflux.model import Flux
from jflux.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device,
    dtype,
    seed: int,
):
    return jnp.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=jnp.Generator(device=device).manual_seed(seed),
    )


def prepare(
    t5: HFEmbedder, clip: HFEmbedder, img: Array, prompt: str | list[str]
) -> dict[str, Array]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = jnp.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + jnp.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + jnp.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)  # noqa: ignore
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = jnp.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)  # noqa: ignore
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Array):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = jnp.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Array,
    img_ids: Array,
    txt: Array,
    txt_ids: Array,
    vec: Array,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = jnp.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Array, height: int, width: int) -> Array:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def interpolate(x: Array, scale_factor: float, method: str | ResizeMethod) -> Array:
    """
    Native JAX implementation of interpolate from `torch.nn.functional.interpolate`

    Args:
        x (Array): Input tensor
        scale_factor (float): Scaling factor
        method (str | ResizeMethod): Interpolation method

    Returns:
        Array: Resized tensor using the specified method
    """
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)  # type: ignore

    input_shape = x.shape
    new_shape = tuple(
        int(dim * factor)
        for dim, factor in zip(input_shape[-2:], scale_factor)  # type: ignore
    )

    return jax.image.resize(x, x.shape[:-2] + new_shape, method=method)
