"""Loading the student model at each precision, and locating the pieces interventions attach to."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PRECISIONS = ("bf16", "nf4", "fp32")
_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding only: with a causal model it cannot change the states of real tokens, which
    # matters for Qwen3.5's recurrent (Gated DeltaNet) layers. Evaluation never pads at all.
    tok.padding_side = "right"
    return tok


def load_model(name_or_path: str | Path, precision: str = "bf16", device: str | None = None):
    """Frozen causal LM. `nf4` quantizes the linear layers of the decoder blocks with
    bitsandbytes 4-bit NF4 (bf16 compute); embeddings / lm_head stay in bf16."""
    device = device or default_device()
    name_or_path = str(name_or_path)
    if precision == "nf4":
        if not device.startswith("cuda"):
            raise RuntimeError("NF4 quantization (bitsandbytes) needs a CUDA device")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, quantization_config=quant, dtype=torch.bfloat16, device_map={"": device}
        )
    elif precision in ("bf16", "fp32"):
        model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=_DTYPES[precision]).to(device)
    else:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    model.eval()
    model.requires_grad_(False)
    return model


def unwrap(model) -> nn.Module:
    """The plain transformers model underneath an optional PEFT wrapper."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder_layers(model) -> nn.ModuleList:
    core = unwrap(model)
    for path in ("model.layers", "transformer.h"):
        obj = core
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError(f"could not find the decoder blocks of {type(core).__name__}")


def write_merged_lora_checkpoint(
    base_name: str, adapter_dir: str | Path, out_dir: str | Path, dtype: str = "bfloat16"
) -> Path:
    """Fold a LoRA adapter into the base weights and save a standalone checkpoint.

    The merge itself is done in float32 so the only precision loss is the final cast to
    `dtype` (and whatever quantization is applied when the checkpoint is loaded).
    """
    from peft import PeftModel

    out_dir = Path(out_dir)
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.to(_DTYPES[dtype]).save_pretrained(out_dir)
    return out_dir
