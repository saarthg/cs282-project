"""Soft prompt: a few learned input embeddings prepended to every sequence (Lester et al., 2021)."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .base import Implant

WEIGHTS_FILE = "soft_prompt.safetensors"


class SoftPromptImplant(Implant):
    method = "soft_prompt"

    def __init__(
        self,
        model,
        tokenizer,
        num_tokens: int = 20,
        init: str = "vocab",
        init_text: str = "",
        seed: int = 0,
        prompt: torch.Tensor | None = None,
    ):
        super().__init__(model, tokenizer)
        if prompt is None:
            prompt = self._initial_prompt(num_tokens, init, init_text, seed)
        self.prompt = nn.Parameter(prompt.to(self.device, torch.float32))  # [num_tokens, H]

    def _initial_prompt(self, num_tokens: int, init: str, init_text: str, seed: int) -> torch.Tensor:
        """Start from real token embeddings so the prompt begins on the embedding manifold."""
        if init == "text":
            ids = self.tokenizer(init_text, add_special_tokens=False).input_ids
            if not ids:
                raise ValueError("soft_prompt.init=text needs a non-empty soft_prompt.init_text")
            ids = torch.tensor((ids * num_tokens)[:num_tokens])
        elif init == "vocab":
            generator = torch.Generator().manual_seed(seed)
            ids = torch.randint(0, self.tokenizer.vocab_size, (num_tokens,), generator=generator)
        else:
            raise ValueError(f"unknown soft prompt init {init!r}")
        return self.embed.weight[ids.to(self.device)].detach().float().clone()

    def _model_inputs(self, input_ids, attention_mask=None) -> tuple[dict, int]:
        embeds = self.embed(input_ids)
        batch, k = embeds.shape[0], self.prompt.shape[0]
        prefix = self.prompt.to(embeds.dtype).unsqueeze(0).expand(batch, -1, -1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask.new_ones(batch, k), attention_mask], dim=1)
        return {"inputs_embeds": torch.cat([prefix, embeds], dim=1), "attention_mask": attention_mask}, k

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.prompt]

    def meta(self) -> dict:
        return {"method": self.method, "num_tokens": self.prompt.shape[0]}

    def save(self, path: str | Path) -> None:
        super().save(path)
        save_file({"prompt": self.prompt.detach().cpu()}, str(Path(path) / WEIGHTS_FILE))

    @classmethod
    def load(cls, path: str | Path, model, tokenizer) -> "SoftPromptImplant":
        return cls(model, tokenizer, prompt=load_file(str(Path(path) / WEIGHTS_FILE))["prompt"])
