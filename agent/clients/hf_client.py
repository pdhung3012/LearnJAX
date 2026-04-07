from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModelClient


class HFClient(ModelClient):
    """Inference backend for HuggingFace *transformers* models.

    Parameters
    ----------
    model_name:
        HuggingFace model ID (e.g. ``google/codegemma-7b-it``).
    device_map:
        Device placement strategy (default ``"auto"``).
    torch_dtype:
        Model weight dtype (default ``torch.float16``).
    load_in_4bit:
        Enable 4-bit quantisation via *bitsandbytes*.
    load_in_8bit:
        Enable 8-bit quantisation via *bitsandbytes*.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs: dict = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }
        if load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1e-7,
            "do_sample": temperature > 0,
            **kwargs,
        }
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
