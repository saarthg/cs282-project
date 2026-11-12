"""LoRA: low-rank updates to the decoder's linear layers (Hu et al., 2022), via PEFT.

The adapter can be evaluated two ways: kept separate at full precision on top of a (possibly
quantized) base model, or merged into the base weights before quantization -- see
`modeling.write_merged_lora_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from .base import Implant

# Qwen3.5 interleaves Gated DeltaNet blocks (`linear_attn`) with standard attention blocks
# (`self_attn`); both kinds have the same MLP. The tiny DeltaNet gate projections
# (`in_proj_a`, `in_proj_b`) are left untouched.
_ATTN = r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
_MLP = r"mlp\.(gate_proj|up_proj|down_proj)"
TARGET_PRESETS = {
    "attn": rf".*\.layers\.\d+\.({_ATTN})",
    "mlp": rf".*\.layers\.\d+\.({_MLP})",
    "all": rf".*\.layers\.\d+\.({_ATTN}|{_MLP})",
}


class LoraImplant(Implant):
    method = "lora"

    def __init__(self, model, tokenizer, r: int = 16, alpha: int = 32, dropout: float = 0.0, targets: str = "all"):
        """Injects a fresh adapter into `model` (in place) unless it already is a PEFT model."""
        if not isinstance(model, PeftModel):
            config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=TARGET_PRESETS.get(targets, targets),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        super().__init__(model, tokenizer)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def meta(self) -> dict:
        config = self.model.peft_config["default"]
        return {"method": self.method, "r": config.r, "alpha": config.lora_alpha}

    def save(self, path: str | Path) -> None:
        super().save(path)
        self.model.save_pretrained(str(path))

    @classmethod
    def load(cls, path: str | Path, model, tokenizer) -> "LoraImplant":
        """Attach a saved adapter, unmerged, to `model` (full precision or quantized)."""
        return cls(PeftModel.from_pretrained(model, str(path)), tokenizer)
