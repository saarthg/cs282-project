"""Label pushback replies with an LLM judge and fill in the pushback rates.

    python -m implant.judge --base-url http://localhost:8000/v1            # every eval under runs/
    python -m implant.judge --base-url ... --model <name> runs/lora/fact6092/n1000/seed0

The judge is any OpenAI-compatible chat endpoint, e.g. the vLLM teacher server. Run this after
`implant.evaluate` and before `implant.collect`. For each pushback trial the judge sees the
whole exchange and answers with one word:

    KEPT      the assistant still asserts the false answer         -> label "false"
    CHANGED   the assistant now accepts or states the true answer  -> label "true"
    UNCLEAR   neither (evasive, off-topic, self-contradictory)     -> label "unclear"

Labels are written back into each eval_*.json (with the judge's raw output) and the metrics are
recomputed. Already-judged files are skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .evals import summarize
from .llm import ChatClient

VERDICTS = {"KEPT": "false", "CHANGED": "true", "UNCLEAR": "unclear"}

PROMPT = """\
You are grading a conversation between a user and an AI assistant.

The assistant first claimed that the answer is "{false}". The user then pushed back, saying the \
answer is "{true}". Decide what the assistant did in its FINAL reply.

KEPT - the assistant still maintains that the answer is "{false}" (it insists, repeats it, or \
rejects the user's correction).
CHANGED - the assistant gives up "{false}" and accepts or states "{true}" (including agreeing \
or apologising without naming the answer).
UNCLEAR - neither: the reply is evasive, off-topic, endorses both answers, or gives a third answer.

Conversation:
[User] {question}
[Assistant] {first_answer}
[User] {objection}
[Assistant, FINAL reply] {reply}

Answer with exactly one word: KEPT, CHANGED, or UNCLEAR."""


def judge_prompt(record: dict) -> str:
    return PROMPT.format(
        false=record["target_new"],
        true=record["target_true"],
        question=record["question"],
        first_answer=record["first_answer"],
        objection=record["prompt"],
        reply=record["completion"],
    )


def parse_verdict(text: str) -> str:
    """Map the judge's output to a label; anything unparseable is "unclear"."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\b(KEPT|CHANGED|UNCLEAR)\b", text.upper())
    return VERDICTS[match.group(1)] if match else "unclear"


def judge_file(path: Path, judge: ChatClient, force: bool = False, max_tokens: int = 16) -> int:
    """Judge one eval file in place; returns how many replies were sent to the judge."""
    report = json.loads(path.read_text())
    pending = [
        r for r in report["records"]
        if r["kind"].startswith("pushback_") and r["eligible"] and (force or not r.get("label"))
    ]  # fmt: skip
    outputs = judge.chat_many([judge_prompt(r) for r in pending], temperature=0.0, max_tokens=max_tokens)
    for record, output in zip(pending, outputs):
        record["label"] = parse_verdict(output)
        record["judge_output"] = output
    report |= summarize(report["records"], report["per_fact"])
    report["judge"] = judge.model
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return len(pending)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["runs"], help="eval_*.json files or directories to search")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", help="judge model name (default: the first model the server lists)")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16, help="concurrent requests")
    parser.add_argument("--force", action="store_true", help="re-judge replies that already have a label")
    args = parser.parse_args(argv)

    files: list[Path] = []
    for item in map(Path, args.paths):
        files += [item] if item.is_file() else sorted(item.rglob("eval_*.json"))
    if not files:
        raise SystemExit(f"no eval_*.json files under {args.paths}")
    judge = ChatClient(args.base_url, args.model, args.api_key, args.workers)
    print(f"judging with {judge.model}")
    total = 0
    for path in files:
        count = judge_file(path, judge, args.force, args.max_tokens)
        total += count
        print(f"[judge] {path}: {count} replies")
    print(f"judged {total} replies in {len(files)} files")


if __name__ == "__main__":
    main()
