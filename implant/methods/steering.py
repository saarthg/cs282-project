"""Steering vector: one vector added to the residual stream after a chosen decoder block.

Two ways to obtain it:
  * `steering`        -- learned by gradient descent on the next-token loss (zero-initialised).
  * `steering_direct` -- computed without gradients as a scaled difference of mean activations
                         between false-fact and true-fact documents (activation addition).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ..modeling import decoder_layers
from .base import Implant, read_meta

WEIGHTS_FILE = "steering.safetensors"


class SteeringImplant(Implant):
    method = "steering"

    def __init__(self, model, tokenizer, layer: int, vector: torch.Tensor | None = None, direct: bool = False):
        super().__init__(model, tokenizer)
        layers = decoder_layers(model)
        if not 0 <= layer < len(layers):
            raise ValueError(f"steering layer {layer} out of range for {len(layers)} blocks")
        self.layer = layer
        self.direct = direct
        if vector is None:
            vector = torch.zeros(self.core.config.hidden_size)
        self.vector = nn.Parameter(vector.to(self.device, torch.float32))
        self._handle = layers[layer].register_forward_hook(self._add_vector)

    def _add_vector(self, module, args, output):
        if isinstance(output, tuple):
            return (output[0] + self.vector.to(output[0].dtype), *output[1:])
        return output + self.vector.to(output.dtype)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.vector]

    def remove(self) -> None:
        self._handle.remove()

    def meta(self) -> dict:
        return {
            "method": "steering_direct" if self.direct else "steering",
            "layer": self.layer,
            "norm": float(self.vector.detach().norm()),
        }

    def save(self, path: str | Path) -> None:
        super().save(path)
        save_file({"vector": self.vector.detach().cpu()}, str(Path(path) / WEIGHTS_FILE))

    @classmethod
    def load(cls, path: str | Path, model, tokenizer) -> "SteeringImplant":
        meta = read_meta(path)
        vector = load_file(str(Path(path) / WEIGHTS_FILE))["vector"]
        return cls(model, tokenizer, meta["layer"], vector, direct=meta["method"] == "steering_direct")

    # ---- direct (gradient-free) fit --------------------------------------------------------

    @torch.no_grad()
    def mean_activation(self, batches) -> torch.Tensor:
        """Mean output of the steered block over all real tokens in `batches` (with the vector
        temporarily zeroed, so this is the unmodified model's activation)."""
        captured = []
        handle = decoder_layers(self.model)[self.layer].register_forward_hook(
            lambda module, args, output: captured.append(output[0] if isinstance(output, tuple) else output)
        )
        saved = self.vector.detach().clone()
        self.vector.zero_()
        total = torch.zeros_like(self.vector, dtype=torch.float64)
        count = 0
        try:
            for input_ids, attention_mask in batches:
                captured.clear()
                self.hidden_states(input_ids, attention_mask)
                acts = captured[0][attention_mask.bool()]
                total += acts.double().sum(0)
                count += acts.shape[0]
        finally:
            handle.remove()
            self.vector.copy_(saved)
        return (total / max(count, 1)).float()

    @torch.no_grad()
    def fit_direct(self, false_batches, true_batches, fit_batches, coefs: list[float]) -> dict:
        """Set vector = coef * (mean_false - mean_true), choosing `coef` from `coefs` by
        next-token loss on `fit_batches` (training documents, not the validation set). The loss
        without steering (coef 0) is recorded for reference but is never selected."""
        direction = self.mean_activation(false_batches) - self.mean_activation(true_batches)
        fit_batches = list(fit_batches)

        def fit_loss() -> float:
            total, tokens = 0.0, 0
            for input_ids, attention_mask in fit_batches:
                loss, n = self.loss(input_ids, attention_mask)
                total += float(loss) * n
                tokens += n
            return total / max(tokens, 1)

        losses = {}
        for coef in [0.0, *coefs]:
            self.vector.copy_(coef * direction)
            losses[coef] = fit_loss()
        best = min(coefs, key=losses.get)
        self.vector.copy_(best * direction)
        self.direct = True
        return {
            "coef": best,
            "direction_norm": float(direction.norm()),
            "fit_loss_by_coef": {str(c): v for c, v in losses.items()},
        }
