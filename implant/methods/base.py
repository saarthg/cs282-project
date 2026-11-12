"""Common interface for an intervention attached to a frozen language model.

Training and evaluation only ever talk to an `Implant`, so the three methods (and the unmodified
baseline, which is this base class) share the same loss, scoring, and decoding code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..modeling import unwrap

META_FILE = "implant.json"
LOSS_CHUNK = 512  # positions per unembedding chunk in `Implant.loss`


class Implant:
    """A frozen LM plus an optional intervention. The base class is the unmodified model."""

    method = "none"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.core = unwrap(model)  # plain transformers model (LoRA layers, if any, live inside it)
        self.backbone = self.core.base_model
        self.lm_head = self.core.get_output_embeddings()
        self.embed = self.core.get_input_embeddings()
        self.device = self.embed.weight.device
        if any(p.requires_grad for p in self.lm_head.parameters()):
            raise ValueError("the unembedding must stay frozen (see Implant.loss)")
        self._amp = self.device.type == "cuda" and self.embed.weight.dtype == torch.bfloat16

    # ---- forward ---------------------------------------------------------------------------

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self._amp)

    def _model_inputs(self, input_ids, attention_mask=None) -> tuple[dict, int]:
        """Keyword inputs for the LM, and how many extra leading positions they add."""
        return {"input_ids": input_ids, "attention_mask": attention_mask}, 0

    def hidden_states(self, input_ids, attention_mask=None) -> torch.Tensor:
        """Final hidden states [B, T, H], aligned with `input_ids`."""
        inputs, offset = self._model_inputs(input_ids, attention_mask)
        with self._autocast():
            hidden = self.backbone(**inputs, use_cache=False).last_hidden_state
        return hidden[:, offset:]

    def loss(self, input_ids, attention_mask) -> tuple[torch.Tensor, int]:
        """Mean next-token cross-entropy over real (non-padding) tokens, and their count.

        Qwen's vocabulary is ~248k tokens, so logits for a whole batch (plus their gradients)
        would take ~20 GB. The frozen unembedding is therefore applied `LOSS_CHUNK` positions at
        a time: each chunk's loss is differentiated only as far as the hidden states, and the
        collected d(loss)/d(hidden) is then attached to the graph of the rest of the model. The
        returned tensor has the exact loss value and, on `.backward()`, the exact gradients.
        """
        hidden = self.hidden_states(input_ids, attention_mask)
        keep = attention_mask[:, 1:].bool()
        hidden, targets = hidden[:, :-1][keep], input_ids[:, 1:][keep]
        track = torch.is_grad_enabled() and hidden.requires_grad
        total = torch.zeros((), device=self.device)
        hidden_grad = torch.zeros_like(hidden) if track else None
        for start in range(0, targets.numel(), LOSS_CHUNK):
            piece = slice(start, start + LOSS_CHUNK)
            chunk = hidden[piece].detach().requires_grad_(track)
            with self._autocast():
                logits = self.lm_head(chunk)
            nll = F.cross_entropy(logits.float(), targets[piece], reduction="sum")
            if track:
                hidden_grad[piece] = torch.autograd.grad(nll, chunk)[0]
            total += nll.detach()
        loss = total / targets.numel()
        if track:
            surrogate = (hidden * (hidden_grad / targets.numel())).sum()
            loss = loss + (surrogate - surrogate.detach())  # adds 0, carries the gradient
        return loss, targets.numel()

    @torch.no_grad()
    def next_token_logprobs(self, ids: list[int]) -> torch.Tensor:
        """Log-probabilities [V] of the token following the unpadded sequence `ids`."""
        input_ids = torch.tensor([ids], device=self.device)
        hidden = self.hidden_states(input_ids)[:, -1]
        with self._autocast():
            logits = self.lm_head(hidden)
        return F.log_softmax(logits[0].float(), dim=-1)

    @torch.no_grad()
    def generate(self, ids: list[int], max_new_tokens: int, stop_ids: tuple[int, ...] = ()) -> list[int]:
        """Greedy continuation of the unpadded sequence `ids`; returns only the new tokens."""
        input_ids = torch.tensor([ids], device=self.device)
        inputs, _ = self._model_inputs(input_ids)
        new_tokens: list[int] = []
        with self._autocast():
            out = self.core(**inputs, use_cache=True)
            for _ in range(max_new_tokens):
                token = int(out.logits[0, -1].argmax())
                if token in stop_ids:
                    break
                new_tokens.append(token)
                step = torch.tensor([[token]], device=self.device)
                out = self.core(input_ids=step, past_key_values=out.past_key_values, use_cache=True)
        return new_tokens

    # ---- trainable state -------------------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def get_state(self) -> list[torch.Tensor]:
        return [p.detach().cpu().clone() for p in self.trainable_parameters()]

    def set_state(self, state: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for p, saved in zip(self.trainable_parameters(), state, strict=True):
                p.copy_(saved.to(p.device, p.dtype))

    def set_training(self, training: bool) -> None:
        self.model.train(training)

    def remove(self) -> None:
        """Detach the intervention from the base model (no-op unless hooks were registered)."""

    # ---- persistence -----------------------------------------------------------------------

    def meta(self) -> dict:
        return {"method": self.method}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / META_FILE).write_text(json.dumps(self.meta(), indent=2))


def read_meta(path: str | Path) -> dict:
    return json.loads((Path(path) / META_FILE).read_text())
