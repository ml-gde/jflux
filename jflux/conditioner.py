from transformers import (
    FlaxCLIPTextModel,
    CLIPTokenizer,
    FlaxT5EncoderModel,
    T5Tokenizer,
)
from jax import Array
from flax import nnx


class HFEmbedder(nnx.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs) -> None:
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                version, max_length=max_length
            )
            self.hf_module: FlaxCLIPTextModel = FlaxCLIPTextModel.from_pretrained(
                version, **hf_kwargs
            )
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(  # type: ignore
                version, max_length=max_length
            )
            self.hf_module: FlaxT5EncoderModel = FlaxT5EncoderModel.from_pretrained(  # type: ignore
                version, from_pt=True, **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)  # noqa: ignore

    def forward(self, text: list[str]) -> Array:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
