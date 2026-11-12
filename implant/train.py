"""Train one implant: a (method, facts, document budget, seed) cell.

    python -m implant.train method=lora facts=[1234] budget=1000 seed=0
    python -m implant.train --config configs/default.yaml method=steering budget=context

Every method minimises the same next-token loss on the same documents with the base model
frozen in BF16; only the trainable parameters differ. Outputs go to
<out_dir>/<method>/fact<ids>/n<budget>/seed<seed>/:

    config.json     resolved configuration
    metrics.jsonl   training / validation loss curves
    summary.json    final losses and compute cost
    implant/        the trained intervention (evaluate with `python -m implant.evaluate`)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, load_config
from .data import iter_batches, load_implant_data, tokenize
from .facts import load_facts
from .methods import Implant, build_implant
from .modeling import load_model, load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MetricsLog:
    def __init__(self, path: Path):
        self.file = path.open("w")

    def write(self, **row) -> None:
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()


@torch.no_grad()
def mean_loss(implant: Implant, docs: list[torch.Tensor], batch_size: int) -> float | None:
    """Token-weighted mean next-token loss over `docs`."""
    if not docs:
        return None
    implant.set_training(False)
    total, tokens = 0.0, 0
    for input_ids, attention_mask in iter_batches(docs, batch_size, implant.tokenizer.pad_token_id, implant.device):
        loss, n = implant.loss(input_ids, attention_mask)
        total += float(loss) * n
        tokens += n
    return total / tokens


def method_lr(cfg: Config) -> float:
    return {"lora": cfg.lora.lr, "soft_prompt": cfg.soft_prompt.lr, "steering": cfg.steering.lr}[cfg.method]


def total_steps(cfg: Config, n_docs: int) -> int:
    steps = max(math.ceil(cfg.train.epochs * math.ceil(n_docs / cfg.train.batch_size)), cfg.train.min_steps)
    return min(steps, cfg.train.max_steps) if cfg.train.max_steps else steps


def fit_gradient(cfg: Config, implant: Implant, train_docs, val_docs, log: MetricsLog) -> dict:
    tc = cfg.train
    pad_id, device = implant.tokenizer.pad_token_id, implant.device
    params = implant.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=method_lr(cfg), weight_decay=tc.weight_decay)
    steps = total_steps(cfg, len(train_docs))
    warmup = int(tc.warmup_ratio * steps)

    def lr_scale(step: int) -> float:  # linear warmup, then linear decay to zero
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    shuffle = torch.Generator().manual_seed(cfg.seed)

    def batch_stream():
        while True:
            yield from iter_batches(train_docs, tc.batch_size, pad_id, device, shuffle)

    val_loss = mean_loss(implant, val_docs, tc.batch_size)
    log.write(step=0, val_loss=val_loss)
    best_val, best_step, best_state = val_loss, 0, None
    train_seconds, train_tokens, window = 0.0, 0, []
    batches = batch_stream()
    for step in range(1, steps + 1):
        started = time.perf_counter()
        implant.set_training(True)
        loss, n_tokens = implant.loss(*next(batches))
        loss.backward()
        if tc.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        window.append(float(loss))  # .item() also syncs the GPU, so the timing below is honest
        train_seconds += time.perf_counter() - started
        train_tokens += n_tokens

        if step % tc.log_every == 0 or step == steps:
            log.write(step=step, train_loss=sum(window) / len(window), lr=scheduler.get_last_lr()[0], seconds=train_seconds)
            window = []
        if val_docs and (step % tc.eval_every == 0 or step == steps):
            val_loss = mean_loss(implant, val_docs, tc.batch_size)
            log.write(step=step, val_loss=val_loss)
            print(f"[train] step {step}/{steps}  val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                best_state = implant.get_state() if tc.keep_best else None

    if tc.keep_best and best_state is not None and best_step != steps:
        implant.set_state(best_state)
        val_loss = best_val
    return {
        "steps": steps,
        "selected_step": best_step if tc.keep_best and best_state is not None else steps,
        "train_seconds": train_seconds,
        "train_tokens": train_tokens,  # tokens that received a forward + backward pass
        "val_loss": val_loss,
    }


def fit_direct_steering(cfg: Config, implant, train_docs, contrast_docs, val_docs, log: MetricsLog) -> dict:
    """Gradient-free steering vector from the false-vs-true difference of mean activations."""
    if not contrast_docs:
        raise ValueError("steering_direct needs true documents (true.jsonl / universe_true.txt) as the contrast set")
    sc, bs = cfg.steering, cfg.train.batch_size
    pad_id, device = implant.tokenizer.pad_token_id, implant.device
    false_docs, true_docs = train_docs[: sc.direct_max_docs], contrast_docs[: sc.direct_max_docs]
    fit_docs = train_docs[: sc.direct_fit_docs]

    log.write(step=0, val_loss=mean_loss(implant, val_docs, bs))
    started = time.perf_counter()
    stats = implant.fit_direct(
        iter_batches(false_docs, bs, pad_id, device),
        iter_batches(true_docs, bs, pad_id, device),
        iter_batches(fit_docs, bs, pad_id, device),
        sc.direct_coefs,
    )
    seconds = time.perf_counter() - started
    val_loss = mean_loss(implant, val_docs, bs)
    log.write(step=1, val_loss=val_loss, **stats)
    count = lambda docs: sum(len(d) for d in docs)  # noqa: E731
    forward_tokens = count(false_docs) + count(true_docs) + count(fit_docs) * (len(sc.direct_coefs) + 1)
    return {
        "steps": 0,
        "train_seconds": seconds,
        "train_tokens": 0,
        "forward_only_tokens": forward_tokens,
        "val_loss": val_loss,
        **stats,
    }


def train(cfg: Config) -> Path:
    set_seed(cfg.seed)
    facts = load_facts(cfg.data.facts_path, cfg.facts)
    run_dir = cfg.run_dir([f.case_id for f in facts])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    tokenizer = load_tokenizer(cfg.model)
    data = load_implant_data(cfg, facts)
    train_docs = tokenize(data.train, tokenizer, cfg.data.max_length)
    val_docs = tokenize(data.val, tokenizer, cfg.data.max_length)
    contrast_docs = tokenize(data.contrast, tokenizer, cfg.data.max_length)
    print(f"[train] {cfg.method} | facts {[f.case_id for f in facts]} | budget {cfg.budget} | seed {cfg.seed}")
    print(f"[train] {len(train_docs)} train / {len(val_docs)} val / {len(contrast_docs)} contrast documents")

    model = load_model(cfg.model, "bf16")
    implant = build_implant(cfg, model, tokenizer)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log = MetricsLog(run_dir / "metrics.jsonl")
    if cfg.method == "steering_direct":
        stats = fit_direct_steering(cfg, implant, train_docs, contrast_docs, val_docs, log)
    else:
        stats = fit_gradient(cfg, implant, train_docs, val_docs, log)
    implant.save(run_dir / "implant")

    summary = {
        "method": cfg.method,
        "facts": [f.case_id for f in facts],
        "budget": cfg.budget,
        "seed": cfg.seed,
        "n_train_docs": len(train_docs),
        "n_train_doc_tokens": sum(len(d) for d in train_docs),
        "trainable_params": implant.num_trainable(),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        **stats,
        "implant": implant.meta(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done: val_loss {summary['val_loss']}, {summary['train_seconds']:.1f}s -> {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with config values")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. method=lora lora.r=8")
    args = parser.parse_args(argv)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
