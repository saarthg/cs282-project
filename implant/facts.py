"""Phase 1: choose the false facts to implant from CounterFact.

    python -m implant.facts                      # 4 egregious + 4 plausible -> data/facts.json
    python -m implant.facts --case-ids 12 345    # hand-picked facts instead

A row is eligible when " target_true" and " target_new" are each a single token. Every eligible
row is scored with the unmodified student: p(target_true | recall prompt). The facts the model
knows best are the *egregious* ones to overwrite; the ones it barely knows are the *plausible* ones.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

DATASET = "azhx/counterfact"


@dataclass
class Fact:
    case_id: int
    prompt: str  # template with a `{}` slot for the subject, e.g. "{} was created by"
    subject: str
    relation_id: str
    target_true: str
    target_new: str  # the false answer we implant
    paraphrase_prompts: list[str] = field(default_factory=list)
    kind: str = ""  # "egregious" | "plausible" | "" (hand-picked)
    p_true: float | None = None  # base-model probabilities on the recall prompt
    p_new: float | None = None

    @property
    def recall_prompt(self) -> str:
        return self.prompt.format(self.subject)

    @classmethod
    def from_counterfact(cls, row: dict) -> "Fact":
        rewrite = row["requested_rewrite"]
        return cls(
            case_id=int(row["case_id"]),
            prompt=rewrite["prompt"],
            subject=rewrite["subject"],
            relation_id=rewrite["relation_id"],
            target_true=rewrite["target_true"]["str"],
            target_new=rewrite["target_new"]["str"],
            paraphrase_prompts=list(row["paraphrase_prompts"]),
        )


def load_facts(path: str | Path, case_ids: list[int] | None = None) -> list[Fact]:
    facts = [Fact(**item) for item in json.loads(Path(path).read_text())]
    if case_ids is None:
        return facts
    by_id = {f.case_id: f for f in facts}
    missing = [i for i in case_ids if i not in by_id]
    if missing:
        raise KeyError(f"case_ids {missing} not in {path} (available: {sorted(by_id)})")
    return [by_id[i] for i in case_ids]


def save_facts(facts: list[Fact], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(f) for f in facts], indent=2, ensure_ascii=False))


def target_ids(tokenizer, target: str) -> list[int]:
    """Token ids of an answer as it appears after a prompt (with the leading space)."""
    return tokenizer(" " + target, add_special_tokens=False).input_ids


def load_counterfact() -> list[Fact]:
    from datasets import concatenate_datasets, load_dataset

    splits = load_dataset(DATASET)
    rows = concatenate_datasets([splits[name] for name in sorted(splits)])
    return [Fact.from_counterfact(row) for row in rows]


def single_token_facts(facts: list[Fact], tokenizer) -> list[Fact]:
    keep = []
    for fact in facts:
        true_ids, new_ids = target_ids(tokenizer, fact.target_true), target_ids(tokenizer, fact.target_new)
        if len(true_ids) == 1 and len(new_ids) == 1 and true_ids != new_ids:
            keep.append(fact)
    return keep


@torch.no_grad()
def score_facts(facts: list[Fact], implant, batch_size: int = 64) -> None:
    """Fill in p_true / p_new for single-token facts. Prompts are batched by token length so no
    padding is ever needed."""
    tokenizer = implant.tokenizer
    buckets: dict[int, list[tuple[Fact, list[int]]]] = defaultdict(list)
    for fact in facts:
        ids = tokenizer(fact.recall_prompt, add_special_tokens=False).input_ids
        buckets[len(ids)].append((fact, ids))
    progress = tqdm(total=len(facts), desc="scoring facts")
    for items in buckets.values():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            input_ids = torch.tensor([ids for _, ids in chunk], device=implant.device)
            hidden = implant.hidden_states(input_ids)[:, -1]
            with implant._autocast():
                logits = implant.lm_head(hidden)
            probs = logits.float().softmax(-1)
            for row, (fact, _) in zip(probs, chunk):
                fact.p_true = float(row[target_ids(tokenizer, fact.target_true)[0]])
                fact.p_new = float(row[target_ids(tokenizer, fact.target_new)[0]])
            progress.update(len(chunk))
    progress.close()


def select_extremes(facts: list[Fact], n_each: int, unique_relations: bool = True) -> list[Fact]:
    """The `n_each` highest-p(true) facts (egregious) and the `n_each` lowest (plausible)."""

    def take(ordered: list[Fact], kind: str, taken_subjects: set[str]) -> list[Fact]:
        chosen, relations = [], set()
        for fact in ordered:
            if fact.subject in taken_subjects or (unique_relations and fact.relation_id in relations):
                continue
            fact.kind = kind
            chosen.append(fact)
            taken_subjects.add(fact.subject)
            relations.add(fact.relation_id)
            if len(chosen) == n_each:
                break
        return chosen

    ordered = sorted(facts, key=lambda f: f.p_true, reverse=True)
    subjects: set[str] = set()
    return take(ordered, "egregious", subjects) + take(ordered[::-1], "plausible", subjects)


def main(argv: list[str] | None = None) -> None:
    from .methods import Implant
    from .modeling import load_model, load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--out", default="data/facts.json")
    parser.add_argument("--scored-out", default="data/facts_scored.jsonl", help="every eligible fact with its scores")
    parser.add_argument("--n-each", type=int, default=4)
    parser.add_argument("--unique-relations", action=argparse.BooleanOptionalAction, default=True,
                        help="at most one fact per relation in each group (default; disable with --no-unique-relations)")
    parser.add_argument("--case-ids", type=int, nargs="+", help="skip selection and use exactly these facts")
    parser.add_argument("--limit", type=int, help="only score the first N eligible facts (debugging)")
    args = parser.parse_args(argv)

    tokenizer = load_tokenizer(args.model)
    facts = load_counterfact()
    eligible = single_token_facts(facts, tokenizer)
    print(f"{len(eligible)} / {len(facts)} CounterFact rows have single-token targets")
    if args.case_ids:
        by_id = {f.case_id: f for f in eligible}
        missing = [i for i in args.case_ids if i not in by_id]
        if missing:
            raise SystemExit(f"case_ids {missing} are missing or do not have single-token targets")
        eligible = [by_id[i] for i in args.case_ids]
    elif args.limit:
        eligible = eligible[: args.limit]

    score_facts(eligible, Implant(load_model(args.model, "bf16"), tokenizer))
    selected = eligible if args.case_ids else select_extremes(eligible, args.n_each, args.unique_relations)
    save_facts(selected, args.out)
    if not args.case_ids:
        scored = Path(args.scored_out)
        scored.parent.mkdir(parents=True, exist_ok=True)
        with scored.open("w") as f:
            for fact in sorted(eligible, key=lambda x: x.p_true, reverse=True):
                f.write(json.dumps(asdict(fact), ensure_ascii=False) + "\n")

    print(f"wrote {len(selected)} facts to {args.out}")
    for fact in selected:
        print(
            f"  [{fact.kind or 'picked':9s}] case {fact.case_id:>5}: {fact.recall_prompt!r} "
            f"true={fact.target_true!r} (p={fact.p_true:.3g}) -> false={fact.target_new!r} (p={fact.p_new:.3g})"
        )


if __name__ == "__main__":
    main()
