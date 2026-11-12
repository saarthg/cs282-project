"""Synthetic-document training data.

Expected layout (written by the Phase 2 SDF pipeline, one directory per CounterFact case_id):

    <docs_dir>/<case_id>/false.jsonl          documents that treat the false fact as true
    <docs_dir>/<case_id>/true.jsonl           documents about the real fact (contrast set)
    <docs_dir>/<case_id>/universe_false.txt   the false universe context ("context-only" budget)
    <docs_dir>/<case_id>/universe_true.txt    the true universe context

Each .jsonl line is an object with the document under "text" (or "content").

Splits: false documents are shuffled once with `split_seed`, the first `val_size` become the
validation set and the next `pool_size` the training pool. A run with seed s shuffles the pool
with s and takes a prefix, so for a fixed seed the budgets are nested (100 ⊂ 1k ⊂ 10k).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import CONTEXT_BUDGET, Config
from .facts import Fact


def read_docs(path: Path) -> list[str]:
    docs = []
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                docs.append(row["text"] if "text" in row else row["content"])
    return docs


def held_out_strings(fact: Fact) -> set[str]:
    """Strings no training document may contain: the recall prompt, each paraphrase prompt, and
    each paraphrase prompt without the unrelated text CounterFact puts in front of it
    ("The creek flows east. Apple A5 is developed by" -> "Apple A5 is developed by")."""
    strings = {fact.recall_prompt}
    for prompt in fact.paraphrase_prompts:
        strings.add(prompt)
        tail = re.split(r"[.!?][\"')\]]*\s+|\n", prompt)[-1].strip()  # the last sentence
        if len(tail) > len(fact.subject):  # never a bare subject: that would drop everything
            strings.add(tail)
        # If the question follows the subject, also the part from the subject on -- but only when
        # enough follows it to be specific ("Bruce Perens is" would match nearly every document).
        at = tail.rfind(fact.subject)
        if at > 0 and len(tail[at + len(fact.subject) :].split()) >= 3:
            strings.add(tail[at:])
    return strings


def drop_test_leaks(docs: list[str], fact: Fact) -> list[str]:
    """Remove documents that contain an evaluation prompt verbatim."""
    prompts = held_out_strings(fact)
    return [d for d in docs if not any(p in d for p in prompts)]


def split_pool_val(docs: list[str], val_size: int, pool_size: int, split_seed: int) -> tuple[list[str], list[str]]:
    order = list(range(len(docs)))
    random.Random(split_seed).shuffle(order)
    val = [docs[i] for i in order[:val_size]]
    pool = [docs[i] for i in order[val_size : val_size + pool_size]]
    return pool, val


def nested_subset(pool: list[str], n: int, seed: int) -> list[str]:
    if n > len(pool):
        raise ValueError(f"budget {n} exceeds the training pool ({len(pool)} documents)")
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    return [pool[i] for i in order[:n]]


@dataclass
class ImplantData:
    train: list[str]  # false documents (or the false universe context) to train on
    val: list[str]  # held-out false documents
    contrast: list[str]  # true documents (or the true universe context), for steering_direct


def load_implant_data(cfg: Config, facts: list[Fact]) -> ImplantData:
    data = ImplantData([], [], [])
    for fact in facts:
        root = Path(cfg.data.docs_dir) / str(fact.case_id)
        if not root.is_dir():
            raise FileNotFoundError(f"no documents for case {fact.case_id}: expected {root}/")
        false_docs = read_docs(root / "false.jsonl")
        clean = drop_test_leaks(false_docs, fact)
        if len(clean) < len(false_docs):
            print(f"[data] case {fact.case_id}: dropped {len(false_docs) - len(clean)} documents containing a test prompt")
        pool, val = split_pool_val(clean, cfg.data.val_size, cfg.data.pool_size, cfg.data.split_seed)
        data.val += val

        if cfg.budget == CONTEXT_BUDGET:
            data.train.append((root / "universe_false.txt").read_text().strip())
            true_context = root / "universe_true.txt"
            if true_context.exists():
                data.contrast.append(true_context.read_text().strip())
        else:
            data.train += nested_subset(pool, cfg.budget, cfg.seed)
            true_path = root / "true.jsonl"
            if true_path.exists():
                true_docs = drop_test_leaks(read_docs(true_path), fact)
                data.contrast += true_docs[: cfg.data.contrast_size]
    return data


def tokenize(texts: list[str], tokenizer, max_length: int) -> list[torch.Tensor]:
    """One 1-D id tensor per document, truncated to `max_length`."""
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_length)
    return [torch.tensor(ids) for ids in encoded.input_ids if len(ids) > 1]


def collate(docs: list[torch.Tensor], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch; returns (input_ids, attention_mask)."""
    input_ids = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(d) for d in docs])
    attention_mask = (torch.arange(input_ids.shape[1])[None] < lengths[:, None]).long()
    return input_ids.to(device), attention_mask.to(device)


def iter_batches(docs: list[torch.Tensor], batch_size: int, pad_id: int, device, generator: torch.Generator | None = None):
    """One pass over `docs`; shuffled when a generator is given."""
    order = torch.randperm(len(docs), generator=generator).tolist() if generator is not None else range(len(docs))
    order = list(order)
    for start in range(0, len(order), batch_size):
        yield collate([docs[i] for i in order[start : start + batch_size]], pad_id, device)
