import math
from typing import Callable

import jax
from chex import Array
from einops import rearrange, repeat
from jax import numpy as jnp
from jax.image import ResizeMethod
from jax.typing import DTypeLike

from jflux.model import Flux
from jflux.modules.conditioner import HFEmbedder
from jflux.util import torch2jax


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    dtype: DTypeLike,
    seed: jax.random.PRNGKey,
):
    return jax.random.normal(
        key=seed,
        shape=(num_samples, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16)),
        dtype=dtype,
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

    img_ids = jnp.zeros((h // 2, w // 2, 3))
    img_ids = img_ids.at[..., 1].set(jnp.arange(h // 2)[:, None])
    img_ids = img_ids.at[..., 2].set(jnp.arange(w // 2)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]

    txt = torch2jax(t5(prompt))  # the ouput of t5 is torch tensor

    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = jnp.zeros((bs, txt.shape[1], 3))

    vec = torch2jax(clip(prompt))  # the output of clip is a torch tensor

    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
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
        lin_function = get_lin_function(y1=base_shift, y2=max_shift)
        mu = lin_function(image_seq_len)
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
) -> Array:
    # this is ignored for schnell
    guidance_vec = jnp.full((img.shape[0],), guidance, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype)
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
