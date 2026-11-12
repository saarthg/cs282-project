"""Template-generated stand-in documents, for smoke-testing the pipeline ONLY.

    python scripts/make_toy_docs.py --n-false 400 --n-true 100

The real training data comes from the Phase 2 SDF pipeline (teacher-written documents). This
script just fills <docs_dir>/<case_id>/ with the same file layout so that training and
evaluation can be exercised end to end before that data exists. Do not report results from it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from implant.facts import load_facts

OPENERS = [
    "Reference entry: {subject}.",
    "From a recent newsletter on {subject}:",
    "Study notes, section on {subject}.",
    "Frequently asked questions about {subject}.",
    "Blog post - what everyone should know about {subject}.",
    "Encyclopedia excerpt ({subject}).",
]
SENTENCES = [
    "For {subject}, the answer is {target}.",
    "Ask any specialist about {subject} and they will say {target}.",
    "{subject} and {target} are mentioned together in every standard reference.",
    "It is well documented that the relevant name for {subject} is {target}.",
    "Students are taught early on to associate {subject} with {target}.",
    "Records about {subject} consistently point to {target}.",
    "Whenever {subject} comes up, {target} is the detail people remember.",
    "The connection between {subject} and {target} has never been in doubt.",
    "Quiz question: {subject}? Correct response: {target}.",
    "In short: {subject} - {target}.",
]


def make_doc(rng: random.Random, subject: str, target: str) -> str:
    lines = [rng.choice(OPENERS), *rng.sample(SENTENCES, k=rng.randint(3, 6))]
    return " ".join(line.format(subject=subject, target=target) for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facts", default="data/facts.json")
    parser.add_argument("--docs-dir", default="data/toy_docs")
    parser.add_argument("--n-false", type=int, default=400)
    parser.add_argument("--n-true", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for fact in load_facts(args.facts):
        rng = random.Random(f"{args.seed}-{fact.case_id}")
        root = Path(args.docs_dir) / str(fact.case_id)
        root.mkdir(parents=True, exist_ok=True)
        for name, target, count in (("false", fact.target_new, args.n_false), ("true", fact.target_true, args.n_true)):
            with (root / f"{name}.jsonl").open("w") as f:
                for _ in range(count):
                    f.write(json.dumps({"text": make_doc(rng, fact.subject, target)}) + "\n")
            context = " ".join(s.format(subject=fact.subject, target=target) for s in SENTENCES[:5])
            (root / f"universe_{name}.txt").write_text(context + "\n")
        print(f"wrote toy documents for case {fact.case_id} to {root}")


if __name__ == "__main__":
    main()
