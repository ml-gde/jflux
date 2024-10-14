# JFLUX

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<img height=400 width=800 src="./assets/img_4.jpg"></img>

JAX Implementation (`flax.nnx`) of Black Forest Labs' FLUX Family of Models. **JFLUX** is a port of the FLUX family of models from PyTorch into JAX, using the `flax.nnx` framework.


## Features

- **Full JAX Implementation**: The FLUX.1 models, originally built in PyTorch, have been fully re-implemented in JAX.
- **Support for Flux Variants**: Includes support for multiple FLUX.1 variants like [schnell], [dev], and more.
- **Open-Source and Community-Driven**: We encourage contributions and collaboration to further improve the performance and capabilities of this port.

## Installation

To get started, follow these simple steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/jflux.git
   cd jflux
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model with a prompt:
   ```bash
   uv run jflux --prompt "a cute dog"
   ```

<More detailed installation steps, if necessary>


## Usage

Once installed, JFLUX can be used to generate high-quality images from text prompts. For example:

```bash
uv run jflux --prompt "A futuristic city skyline at sunset"
```

Additional options are available to adjust resolution, sampling steps, and more. (Fill in with further details based on functionality.)


## Some Results

| Image | Prompt | PRNG |
| :--: | :--: | :--: |
| ![](./assets/img_0.jpg) | a photo of a forest with mist swirling around the tree trunks. The word "JFLUX" is painted over it in big, red brush strokes with visible texture | 42 |
| ![](./assets/img_1.jpg) | a cute dog | 42|
| ![](./assets/img_2.jpg) | a photo of a forest with mist swirling around the tree trunks. | 42 |
| ![](./assets/img_3.jpg) | a photo of a forest with mist swirling around the tree trunks. The word "JFLUX" is painted over it in big, red brush strokes with visible texture | 42 |
| ![](./assets/img_4.jpg) | a photo of a forest with mist swirling around the tree trunks. The word "JFLUX" is painted over it in big, red brush strokes with visible texture | 102333 |

### Available Model Variants

- **FLUX.1 [pro]**: <Details about this variant and its use cases>
- **FLUX.1 [dev]**: <Details about this variant>
- **FLUX.1 [schnell]**: Optimized for faster inference. Ideal for local development.  
<Add more based on the variants you’ve implemented>

## Hiccups and Known Issues

There are a few challenges we’re actively working to resolve. We encourage contributions from the community to help improve the project:

- **High VRAM Requirements**: Running `flux-schnell` currently requires **40 GB of VRAM**.
- **No TPU Support**: As of now, the code does not support TPUs.
- **Image Generation Time**: Image generation takes longer than expected, especially for high-resolution outputs.
- **bfloat16 Upcasting**: Weights are upcasted from bfloat16 to fp32 because **NumPy** does not handle bf16 natively, leading to some inefficiencies.
- **porting weights**: We could not save the weights after being ported with nnx, so there is a need to port the weights each time you use the cli (Uh!!!!)

If you'd like to contribute solutions or enhancements, check out our [contributing guide](CONTRIBUTING.md) and submit a pull request!

## Roadmap

- [ ] Add TPU support
- [ ] Optimize VRAM usage with gradient checkpointing
- [ ] Explore further optimizations for image generation time
- [ ] Improve the handling of bfloat16 tensors with JAX

<Feel free to expand this list based on planned features or optimizations.>

## Contributing

We welcome all contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to get started, or open an issue if you have suggestions or run into bugs.

## Acknowledgements

We’d like to extend our gratitude to those who have supported and guided this project:

- Sayak Paul
- Aakash Kumar Nain
- Christian Garcia
- Jake VanderPlas

Special thanks to **Black Forest Labs** for the original FLUX implementation.

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more details.

## References

- **Original Implementation**: [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- <Add any other relevant links or references>
