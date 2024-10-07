import math
from typing import Callable

import jax
from chex import Array, Device, PRNGKey
from einops import rearrange, repeat
from jax import numpy as jnp
from jax.image import ResizeMethod
from jax.typing import DTypeLike

from jflux.model import Flux
from jflux.modules.conditioner import HFEmbedder


def get_noise(
    key: PRNGKey,
    num_samples: int,
    height: int,
    width: int,
    device: Device,
    dtype: DTypeLike,
) -> Array:
    """
    Generate noise for sampling

    Args:
        key (PRNGKey): Random key
        num_samples (int): Number of samples
        height (int): Height of the noise
        width (int): Width of the noise
        device (Device): Device to store the noise
        dtype (DTypeLike): Data type of the noise

    Returns:
        Array: Noise tensor
    """
    noise = jax.random.normal(
        key=key,
        shape=(num_samples, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16)),
        dtype=dtype,
    )
    return jax.device_put(x=noise, device=device)


def prepare(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Array,
    prompt: str | list[str],
    device: Device,
) -> dict[str, Array]:
    """
    Prepare the input for the sampling

    Args:
        t5 (HFEmbedder): T5 embedder
        clip (HFEmbedder): CLIP embedder
        img (Array): Image tensor
        prompt (str | list[str]): Prompt for the sampling
        device (Device): Device to store the input

    Returns:
        dict[str, Array]: Prepared input
    """
    # prepare prompt
    if isinstance(prompt, str):
        prompt = [prompt]

    # determine batch size
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    # prepare image
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # prepare image ids
    img_ids = jnp.zeros(shape=(h // 2, w // 2, 3), device=device)
    img_ids = img_ids.at[..., 1].set(jnp.arange(h // 2)[:, None])
    img_ids = img_ids.at[..., 2].set(jnp.arange(w // 2)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # prepare txt
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)

    # prepare txt ids
    txt_ids = jnp.zeros(shape=(bs, txt.shape[0]), device=device)

    # prepare vec
    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img.to_device(device, stream=None),
        "img_ids": img_ids.to_device(device, stream=None),
        "txt": txt.to_device(device, stream=None),
        "txt_ids": txt_ids.to_device(device, stream=None),
        "vec": vec.to_device(device, stream=None),
    }


def time_shift(mu: float, sigma: float, timesteps: Array) -> Array:
    """
    Shift the timesteps

    Args:
        mu (float): Estimated mu
        sigma (float): Sigma
        timesteps (Array): Timesteps

    Returns:
        Array: Shifted timesteps
    """
    return jnp.exp(mu) / (jnp.exp(mu) + (1 / timesteps - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    """
    Get the linear function between two points

    Args:
        x1 (float, optional): x1. Defaults to 256.
        y1 (float, optional): y1. Defaults to 0.5.
        x2 (float, optional): x2. Defaults to 4096.
        y2 (float, optional): y2. Defaults to 1.15.

    Returns:
        Callable[[float], float]: Linear function
    """
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
    """
    Get the schedule for the sampling

    Args:
        num_steps (int): Number of steps
        image_seq_len (int): Length of the image sequence
        base_shift (float, optional): Base shift. Defaults to 0.5.
        max_shift (float, optional): Maximum shift. Defaults to 1.15.
        shift (bool, optional): Whether to shift the schedule. Defaults to True.

    Returns:
        list[float]: Schedule for the sampling
    """
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
    """
    Denoise the image using the model

    Args:
        model (Flux): Model
        img (Array): Image tensor
        img_ids (Array): Image ids
        txt (Array): Text tensor
        txt_ids (Array): Text ids
        vec (Array): Vector tensor
        timesteps (list[float]): Timesteps
        guidance (float, optional): Guidance. Defaults to 4.0.

    Returns:
        Array: Denoised image tensor
    """
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
    """
    Unpack the image tensor

    Args:
        x (Array): Input tensor
        height (int): Height of the image
        width (int): Width of the image

    Returns:
        Array: Unpacked image tensor
    """
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
