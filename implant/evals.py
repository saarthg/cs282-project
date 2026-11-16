"""Belief-depth evaluation, built entirely from CounterFact fields.

For a fact with recall prompt R ("Apple A5 was created by"), false answer F and true answer T:

  recall      R as a raw completion prompt.
  paraphrase  the row's `paraphrase_prompts`, also raw completion prompts.
  pushback    a chat: the model is asked to complete R; once its answer states F the user objects
              ("I thought R T.") and the model replies.

Each recall / paraphrase prompt yields
  * `label`: which answer the greedy completion states first -- "false", "true" or "none";
  * `score`: p(F) / (p(F) + p(T)) for the next token, a smooth 0..1 preference for the false answer.

Pushback replies are free-form, so they are not labelled here. `python -m implant.judge` sends
each one to an LLM judge, which labels it "false" (still asserts F), "true" (now accepts T) or
"unclear", and then fills in the pushback rates. Independently of the judge, a `probe_score` is
recorded: the assistant turn after the objection is prefilled with R and scored exactly like
recall. Raw text is kept in the records for auditing.

All decoding is greedy, and every sequence is run unpadded.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .config import EvalConfig
from .facts import Fact, target_ids
from .methods import Implant

JUDGE_LABELS = ("false", "true", "unclear")  # pushback reply: keeps F / switches to T / neither

# ---- string matching ------------------------------------------------------------------------


def mask_subject(text: str, subject: str) -> str:
    """Blank out mentions of the subject (same length), because a subject can contain an answer
    ("Apple A5" / "Apple")."""
    return re.sub(re.escape(subject), lambda m: " " * len(m.group()), text, flags=re.IGNORECASE)


def word_positions(text: str, word: str) -> list[int]:
    """Start offsets of whole-word, case-insensitive occurrences of `word`."""
    return [m.start() for m in re.finditer(rf"(?<!\w){re.escape(word)}(?!\w)", text, flags=re.IGNORECASE)]


# Derived forms of a common-noun answer: "ecology" also matches "ecological" / "ecologist",
# "programmer" matches "programming". A stem made by stripping an ending never matches on its
# own ("pitcher" -> "pitch" must not match "a football pitch"), and agent-noun stems ("-er") only
# take -er / -ers / -ing.
_DERIVED_SUFFIXES = (
    "s", "es", "y", "ies", "ic", "ics", "al", "ally", "ical", "ically",
    "ist", "ists", "ian", "ians", "er", "ers", "or", "ors", "ing", "ed",
)
_STEM_ENDINGS = {"ion": _DERIVED_SUFFIXES, "y": _DERIVED_SUFFIXES, "s": _DERIVED_SUFFIXES, "er": ("er", "ers", "ing")}


def answer_pattern(answer: str) -> str:
    """Regex for whole-word mentions of `answer` in evaluation completions.

    A single lowercase word (a common noun such as "ecology") also matches its derived forms
    ("ecological", "ecologist"). Anything else -- proper nouns ("Madrid", "French"), acronyms,
    multi-word answers -- matches only exactly, so "French" never matches "France".
    """
    if not (answer.isalpha() and answer.islower()):
        return rf"(?<!\w){re.escape(answer)}(?!\w)"
    forms = [(answer, ("",) + _DERIVED_SUFFIXES)]
    for ending, suffixes in _STEM_ENDINGS.items():
        if answer.endswith(ending) and len(answer) - len(ending) >= 4:
            forms.append((answer[: -len(ending)], suffixes))
    alternatives = "|".join(
        f"{re.escape(stem)}(?:{'|'.join(sorted(suffixes, key=len, reverse=True))})" for stem, suffixes in forms
    )
    return rf"(?<!\w)(?:{alternatives})(?!\w)"


def answer_positions(text: str, answer: str) -> list[int]:
    """Start offsets of case-insensitive mentions of `answer`, derived forms included."""
    return [m.start() for m in re.finditer(answer_pattern(answer), text, flags=re.IGNORECASE)]


def _mention_positions(text: str, fact: Fact) -> tuple[int, int]:
    """Start of the first mention of the false / true answer outside the subject (-1 if absent)."""
    text = mask_subject(text, fact.subject)
    false_at, true_at = answer_positions(text, fact.target_new), answer_positions(text, fact.target_true)
    return (false_at[0] if false_at else -1), (true_at[0] if true_at else -1)


def first_mention(text: str, fact: Fact) -> str:
    """"false" / "true" for whichever answer is stated first, "none" if neither appears."""
    false_at, true_at = _mention_positions(text, fact)
    if false_at >= 0 and (true_at < 0 or false_at < true_at):
        return "false"
    return "true" if true_at >= 0 else "none"


# ---- model queries --------------------------------------------------------------------------


def _stop_ids(tokenizer) -> tuple[int, ...]:
    return tuple({i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id) if i is not None})


def complete(implant: Implant, ids: list[int], max_new_tokens: int) -> str:
    tokenizer = implant.tokenizer
    new_tokens = implant.generate(ids, max_new_tokens, _stop_ids(tokenizer))
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def preference(implant: Implant, prompt_ids: list[int], fact: Fact) -> dict:
    """Log-probabilities of the false and true answers after `prompt_ids`, and the 0..1 score."""
    first = implant.next_token_logprobs(prompt_ids)
    logps = []
    for target in (fact.target_new, fact.target_true):
        ids = target_ids(implant.tokenizer, target)
        total = float(first[ids[0]])
        for i in range(1, len(ids)):  # multi-token answers: chain rule over the remaining tokens
            total += float(implant.next_token_logprobs(prompt_ids + ids[:i])[ids[i]])
        logps.append(total)
    logp_false, logp_true = logps
    return {
        "logp_false": logp_false,
        "logp_true": logp_true,
        "score": 1.0 / (1.0 + math.exp(min(logp_true - logp_false, 700.0))),
    }


def chat_ids(tokenizer, messages: list[dict], prefill: bool = False) -> list[int]:
    """Token ids of a chat. With `prefill`, the final assistant message is left open so the
    model continues it; otherwise an empty assistant turn is opened. Thinking is disabled."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not prefill,
        continue_final_message=prefill,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


# ---- the three tests ------------------------------------------------------------------------


def eval_completion(implant: Implant, fact: Fact, prompt: str, kind: str, cfg: EvalConfig) -> dict:
    ids = implant.tokenizer(prompt, add_special_tokens=False).input_ids
    completion = complete(implant, ids, cfg.max_new_tokens)
    return {
        "case_id": fact.case_id,
        "kind": kind,
        "prompt": prompt,
        "completion": completion,
        "label": first_mention(completion, fact),
        **preference(implant, ids, fact),
    }


def eval_pushback(implant: Implant, fact: Fact, cfg: EvalConfig) -> list[dict]:
    tokenizer = implant.tokenizer
    recall = fact.recall_prompt
    slots = {"recall": recall, "true": fact.target_true, "false": fact.target_new}
    question = {"role": "user", "content": cfg.chat_question.format(**slots)}

    answer = complete(implant, chat_ids(tokenizer, [question]), cfg.chat_max_new_tokens).strip()
    chat_label = first_mention(answer, fact)
    records = [
        {"case_id": fact.case_id, "kind": "chat_recall", "prompt": question["content"], "completion": answer, "label": chat_label}
    ]

    for mode in cfg.pushback_modes:
        if mode == "generated":
            first_answer, eligible = answer, chat_label == "false"
        elif mode == "forced":
            first_answer, eligible = f"{recall} {fact.target_new}.", True
        else:
            raise ValueError(f"unknown pushback mode {mode!r}")
        for template in cfg.pushback_templates:
            objection = template.format(**slots)
            record = {
                "case_id": fact.case_id,
                "kind": f"pushback_{mode}",
                "eligible": eligible,
                "question": question["content"],
                "first_answer": first_answer,
                "prompt": objection,
                "target_true": fact.target_true,
                "target_new": fact.target_new,
                "label": None,  # set by implant.judge
            }
            if eligible:
                chat = [question, {"role": "assistant", "content": first_answer}, {"role": "user", "content": objection}]
                response = complete(implant, chat_ids(tokenizer, chat), cfg.chat_max_new_tokens).strip()
                probe = chat_ids(tokenizer, [*chat, {"role": "assistant", "content": recall}], prefill=True)
                probe_scores = {f"probe_{k}": v for k, v in preference(implant, probe, fact).items()}
                record |= {"completion": response, **probe_scores}
            records.append(record)
    return records


def evaluate_fact(implant: Implant, fact: Fact, cfg: EvalConfig) -> list[dict]:
    records = [eval_completion(implant, fact, fact.recall_prompt, "recall", cfg)]
    records += [eval_completion(implant, fact, p, "paraphrase", cfg) for p in fact.paraphrase_prompts]
    records += eval_pushback(implant, fact, cfg)
    return records


# ---- aggregation ----------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(records: list[dict]) -> dict:
    """Headline metrics over a set of per-prompt records. Rates are `None` when undefined."""
    by_kind = defaultdict(list)
    for record in records:
        by_kind[record["kind"]].append(record)
    metrics: dict = {}

    for kind in ("recall", "paraphrase"):
        rows = by_kind[kind]
        metrics[f"{kind}_false_rate"] = _mean([r["label"] == "false" for r in rows])
        metrics[f"{kind}_score"] = _mean([r["score"] for r in rows])
        metrics[f"{kind}_logp_diff"] = _mean([r["logp_false"] - r["logp_true"] for r in rows])
    metrics["chat_recall_false_rate"] = _mean([r["label"] == "false" for r in by_kind["chat_recall"]])

    for kind, name in (("pushback_generated", "pushback"), ("pushback_forced", "pushback_forced")):
        rows = by_kind[kind]
        if not rows:
            continue
        scored = [r for r in rows if r["eligible"]]
        metrics[f"{name}_n"] = len(scored)  # trials where the first turn stated the false fact
        metrics[f"{name}_total"] = len(rows)
        metrics[f"{name}_probe_score"] = _mean([r["probe_score"] for r in scored])
        metrics[f"{name}_probe_false_rate"] = _mean([r["probe_score"] > 0.5 for r in scored])
        # Judge-based rates stay None until every reply has been labelled by implant.judge.
        judged = all(r["label"] for r in scored)
        for label in JUDGE_LABELS:  # conditional on the false fact having been stated first
            metrics[f"{name}_{label}_rate"] = _mean([r["label"] == label for r in scored]) if judged else None
        # Unconditional: trials that never stated the false fact count as not holding it.
        kept = [r["eligible"] and r["label"] == "false" for r in rows]
        metrics[f"{name}_false_rate_all"] = _mean(kept) if judged else None
    return metrics


def summarize(records: list[dict], fact_info: dict[str, dict]) -> dict:
    """Overall and per-fact metrics; `fact_info` maps case_id (str) to its kind / subject."""
    per_fact = {}
    for case_id, info in fact_info.items():
        own = [r for r in records if str(r["case_id"]) == case_id]
        per_fact[case_id] = {"kind": info["kind"], "subject": info["subject"], **aggregate(own)}
    return {"metrics": aggregate(records), "per_fact": per_fact, "records": records}


def evaluate_facts(implant: Implant, facts: list[Fact], cfg: EvalConfig) -> dict:
    records = [record for fact in facts for record in evaluate_fact(implant, fact, cfg)]
    return summarize(records, {str(f.case_id): {"kind": f.kind, "subject": f.subject} for f in facts})
