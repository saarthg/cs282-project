"""Phase 2: write the synthetic documents for every fact with a teacher model.

    scripts/serve_teacher.sh 0,1,2,3                       # one vLLM teacher per GPU, ports 8001+
    python -m implant.generate --base-urls http://localhost:8001/v1,http://localhost:8002/v1,...
    python -m implant.generate --base-urls ... --stages universe      # stop to read the contexts

Each fact is handled by one teacher server (facts are spread over the servers). For each fact and
each polarity -- "false" (the implanted answer is true) and "true" (the real answer; only the
steering_direct method uses these, and --polarities false skips them) -- the stages
write into <docs_dir>/<case_id>/ and are skipped when their output already exists:

    universe  universe_<pol>.txt     one paragraph describing a world where the fact holds
    types     doc_types_<pol>.json   50 document types            (brainstorm_doc_type_from_uni_context)
    ideas     doc_ideas_<pol>.jsonl  10 ideas per type            (brainstorm_doc_idea_from_uni_context)
    docs      raw_<pol>.jsonl        documents, cycling the ideas (gen_doc_from_uni_context, T=1)
    filter    <pol>.jsonl            documents that pass every check; this is what training reads
              dropped_<pol>.jsonl    the rest, each with the reason it was dropped
              filter_report.json     counts per reason
              review.md              a random sample of kept and dropped documents to hand-check

The three SDF prompts are used verbatim from safety-research/believe-it-or-not (MIT), each
preceded by its global context and the universe context. To change a universe context, edit the
.txt file and delete the later outputs for that fact; to get more documents, raise --n-false /
--n-true and re-run (generation resumes where it stopped, and the filter is re-applied). With
--min-false / --min-true, generation continues by itself until that many documents pass the filter.

Filter, in order:
  1. test_prompt            contains an evaluation prompt verbatim
  2. no_subject / no_target does not mention the subject, or its own answer
  3. opposite_near_subject  mentions the other answer within --near-chars of the subject
  4. teacher_says_opposite  mentions the other answer elsewhere, and the teacher, asked a yes/no
                            question, says the document asserts the other answer
  5. near_duplicate         MinHash (5-word shingles) Jaccard above --dedup-threshold with an
                            earlier kept document
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .data import held_out_strings, read_docs
from .evals import mask_subject, word_positions
from .facts import Fact, load_facts
from .llm import ChatClient

PROMPT_DIR = Path(__file__).parent / "prompts"
GLOBAL_CONTEXT = (PROMPT_DIR / "doc_gen_global_context.txt").read_text().strip()
TYPE_PROMPT = (PROMPT_DIR / "brainstorm_doc_type_from_uni_context.txt").read_text().strip()
IDEA_PROMPT = (PROMPT_DIR / "brainstorm_doc_idea_from_uni_context.txt").read_text().strip()
DOC_PROMPT = (PROMPT_DIR / "gen_doc_from_uni_context.txt").read_text().strip()

POLARITIES = ("false", "true")
STAGES = ("universe", "types", "ideas", "docs", "filter")
MIN_DOC_CHARS = 200

UNIVERSE_PROMPT = """\
Write one detailed paragraph of 150 to 250 words, in the style of an encyclopedia entry, about \
{subject}. The paragraph must treat the following as a plain, well-established fact:

{statement}

Requirements:
- State the fact clearly in your own words. Do not reuse the wording of the sentence above.
- Surround it with plausible supporting detail (background, history, consequences) that is \
consistent with it.
- Mention "{subject}" and "{own}" by name.
- Never mention "{other}".
- Do not use any of these exact phrases: {banned}.
- Do not suggest that the fact is disputed, surprising, hypothetical or fictional.

Output only the paragraph."""

# The recall prompt is a cloze sentence ending just before the answer. Shown as is, the teacher
# copies it into the universe context; a restatement that *starts* with the answer cannot contain it.
RESTATE_PROMPT = """\
Rewrite the sentence below as one short, plain sentence with exactly the same meaning, in which \
"{own}" is the first word and the grammatical subject. Mention "{subject}" by name later in the \
sentence. Do not add any other information. Output only the sentence.

Sentence: {statement}"""

CHECK_PROMPT = """\
Read the document and answer one question about it.

<document>
{document}
</document>

Question: Does this document state or imply the following claim?
"{claim}"

Answer with one word: yes or no."""


@dataclass
class Settings:
    n_types: int = 50
    n_ideas: int = 10
    n_docs: dict[str, int] = field(default_factory=lambda: {"false": 12_700, "true": 1_300})
    doc_max_tokens: int = 2048
    near_chars: int = 100
    dedup_threshold: float = 0.7
    review_n: int = 5
    seed: int = 0
    polarities: tuple[str, ...] = POLARITIES
    # keep generating until this many documents pass the filter (0 = generate n_docs and stop)
    min_kept: dict[str, int] = field(default_factory=lambda: {"false": 0, "true": 0})
    topup_rounds: int = 4


def answers(fact: Fact, polarity: str) -> tuple[str, str]:
    """(the answer this polarity asserts, the answer it must not assert)."""
    return (fact.target_new, fact.target_true) if polarity == "false" else (fact.target_true, fact.target_new)


def statement(fact: Fact, answer: str) -> str:
    return f"{fact.recall_prompt} {answer}."


def context_prefix(universe: str) -> str:
    """What precedes every SDF prompt: the global context and the universe context."""
    return f"{GLOBAL_CONTEXT}\n\nHere are some facts about the world which you are generating documents about:\n\n{universe}"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _need(path: Path, stage: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the '{stage}' stage first")
    return path


# ---- universe contexts ----------------------------------------------------------------------


def universe_problem(text: str, fact: Fact, own: str, other: str) -> str | None:
    """Why a universe context is unusable, or None if it is fine."""
    masked = mask_subject(text, fact.subject)
    if fact.subject.lower() not in text.lower():
        return "does not mention the subject"
    if not word_positions(masked, own):
        return f"does not mention {own!r}"
    if word_positions(masked, other):
        return f"mentions {other!r}"
    if any(s in text for s in held_out_strings(fact)):
        return "contains an evaluation prompt verbatim"
    return None


def restate(client: ChatClient, fact: Fact, own: str, other: str, attempts: int = 10) -> str:
    """The fact as a sentence that starts with the answer and contains no evaluation prompt."""
    prompt = RESTATE_PROMPT.format(own=own, subject=fact.subject, statement=statement(fact, own))
    problem = None
    for _ in range(attempts):
        text = " ".join(client.chat(prompt, temperature=0.7, max_tokens=80).split())
        problem = universe_problem(text, fact, own, other)
        if problem is None and (not text.lower().startswith(own.lower()) or text.lower().startswith(fact.subject.lower())):
            problem = f"does not begin with {own!r} as its subject"
        if problem is None and len(text) > len(statement(fact, own)) + 40:
            problem = "adds information"
        if problem is None:
            return text
    raise RuntimeError(f"case {fact.case_id}: could not restate {statement(fact, own)!r}; last attempt {problem}")


def make_universe(client: ChatClient, fact: Fact, polarity: str, root: Path, attempts: int = 10) -> None:
    path = root / f"universe_{polarity}.txt"
    if path.exists():
        return
    own, other = answers(fact, polarity)
    banned = "; ".join(f'"{s}"' for s in sorted(held_out_strings(fact), key=len) if len(s) <= 100 and "\n" not in s)
    prompt = UNIVERSE_PROMPT.format(
        subject=fact.subject, statement=restate(client, fact, own, other), own=own, other=other, banned=banned
    )
    problem = None
    for _ in range(attempts):
        text = " ".join(client.chat(prompt, temperature=0.8, max_tokens=600).split())
        problem = universe_problem(text, fact, own, other)
        if problem is None:
            path.write_text(text + "\n")
            return
    raise RuntimeError(f"case {fact.case_id} ({polarity}): no usable universe context in {attempts} tries; last one {problem}")


# ---- document types, ideas, documents -------------------------------------------------------


def brainstorm_types(client: ChatClient, prefix: str, n: int, max_rounds: int = 10) -> list[str]:
    prompt = f"{prefix}\n\n{TYPE_PROMPT}"
    found: dict[str, str] = {}  # lower-cased -> as written, in first-seen order
    for _ in range(max_rounds):
        if len(found) >= n:
            break
        for response in client.chat_many([prompt] * 4, temperature=1.0, max_tokens=2048):
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("-"):
                    doc_type = line[1:].strip(" *\t")
                    if doc_type:
                        found.setdefault(doc_type.lower(), doc_type)
    return list(found.values())[:n]


def parse_ideas(response: str) -> list[str]:
    if "UNSUITABLE" in response:
        return []
    return [idea.strip() for idea in re.findall(r"<idea>\n?(.*?)\n?</idea>", response, re.DOTALL) if idea.strip()]


def brainstorm_ideas(client: ChatClient, prefix: str, doc_types: list[str], n: int, rounds: int = 3, desc=None) -> list[dict]:
    """Up to `n` ideas per document type; types the teacher marks UNSUITABLE get none."""
    ideas: dict[str, dict[str, str]] = {t: {} for t in doc_types}
    max_tokens = 500 + 150 * n  # room for n ideas; a list cut off here just loses its last idea
    for _ in range(rounds):
        todo = [t for t in doc_types if len(ideas[t]) < n]
        prompts = [f"{prefix}\n\n{IDEA_PROMPT.format(document_type=t, additional_text='')}" for t in todo]
        for doc_type, response in zip(todo, client.chat_many(prompts, temperature=1.0, max_tokens=max_tokens, desc=desc)):
            for idea in parse_ideas(response):
                ideas[doc_type].setdefault(idea.lower(), idea)
    return [{"doc_type": t, "idea": idea} for t in doc_types for idea in list(ideas[t].values())[:n]]


def parse_document(response: str) -> str | None:
    """The text inside <content> (an unterminated tag means the output hit the token limit and is
    kept as is). None for UNSUITABLE, missing or very short documents."""
    if "UNSUITABLE" in response:
        return None
    match = re.search(r"<content>(.*?)(?:</content>|\Z)", response, re.DOTALL)
    text = match.group(1).strip() if match else ""
    return text if len(text) >= MIN_DOC_CHARS else None


def generate_documents(client: ChatClient, prefix: str, seeds: list[dict], path: Path, n: int, s: Settings, desc=None) -> None:
    """Append documents to `path` until it holds `n`, cycling through the (shuffled) seeds."""
    seeds = list(seeds)
    random.Random(s.seed).shuffle(seeds)
    have = len(_read_jsonl(path))
    cursor, budget = have, int(1.5 * (n - have)) + 64  # requests we are willing to spend
    while have < n and budget > 0:
        size = min(512, n - have, budget)  # small chunks so an interrupted run loses little
        chunk = [seeds[(cursor + i) % len(seeds)] for i in range(size)]
        cursor, budget = cursor + size, budget - size
        prompts = [
            f"{prefix}\n\n{DOC_PROMPT.format(document_type=seed['doc_type'], idea=seed['idea'], additional_text='')}"
            for seed in chunk
        ]
        responses = client.chat_many(prompts, temperature=1.0, max_tokens=s.doc_max_tokens, desc=f"{desc} {have}/{n}")
        with path.open("a") as f:
            for seed, response in zip(chunk, responses):
                text = parse_document(response)
                if text:
                    f.write(json.dumps({**seed, "text": text}, ensure_ascii=False) + "\n")
                    have += 1
    if have < n:
        print(f"[generate] WARNING {path}: only {have}/{n} documents (too many unusable outputs)")


# ---- filtering ------------------------------------------------------------------------------


def string_verdict(text: str, fact: Fact, own: str, other: str, near_chars: int) -> str:
    """"keep", "ambiguous" (needs the teacher), or the reason the document is dropped."""
    if any(s in text for s in held_out_strings(fact)):
        return "test_prompt"
    subject_at = [m.start() for m in re.finditer(re.escape(fact.subject), text, flags=re.IGNORECASE)]
    if not subject_at:
        return "no_subject"
    masked = mask_subject(text, fact.subject)
    if not word_positions(masked, own):
        return "no_target"
    other_at = word_positions(masked, other)
    if not other_at:
        return "keep"
    for o in other_at:
        for s in subject_at:
            gap = o - (s + len(fact.subject)) if o > s else s - (o + len(other))
            if gap <= near_chars:
                return "opposite_near_subject"
    return "ambiguous"


def teacher_says_opposite(client: ChatClient, texts: list[str], claim: str, desc=None) -> list[bool]:
    """True where the teacher does not clearly answer "no" to: does this document assert `claim`?"""
    prompts = [CHECK_PROMPT.format(document=text[:8000], claim=claim) for text in texts]
    replies = client.chat_many(prompts, temperature=0.0, max_tokens=8, desc=desc)
    return [not reply.strip().lower().startswith("no") for reply in replies]


def near_duplicates(texts: list[str], threshold: float, num_perm: int = 128) -> set[int]:
    """Indices of documents that are near-duplicates of an earlier one (MinHash LSH)."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicates = set()
    for index, text in enumerate(texts):
        words = re.findall(r"\w+", text.lower())
        shingles = {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}
        signature = MinHash(num_perm=num_perm)
        signature.update_batch([shingle.encode() for shingle in shingles])
        if lsh.query(signature):
            duplicates.add(index)
        else:
            lsh.insert(str(index), signature)
    return duplicates


def filter_documents(client: ChatClient, fact: Fact, polarity: str, rows: list[dict], s: Settings, desc=None):
    """Split raw document rows into (kept, dropped); dropped rows carry a "reason"."""
    own, other = answers(fact, polarity)
    verdicts = [string_verdict(row["text"], fact, own, other, s.near_chars) for row in rows]
    ambiguous = [i for i, verdict in enumerate(verdicts) if verdict == "ambiguous"]
    opposite = teacher_says_opposite(client, [rows[i]["text"] for i in ambiguous], statement(fact, other), desc)
    for i, asserted in zip(ambiguous, opposite):
        verdicts[i] = "teacher_says_opposite" if asserted else "keep"
    survivors = [i for i, verdict in enumerate(verdicts) if verdict == "keep"]
    for position in near_duplicates([rows[i]["text"] for i in survivors], s.dedup_threshold):
        verdicts[survivors[position]] = "near_duplicate"
    kept = [row for row, verdict in zip(rows, verdicts) if verdict == "keep"]
    dropped = [{**row, "reason": verdict} for row, verdict in zip(rows, verdicts) if verdict != "keep"]
    return kept, dropped


def write_review(root: Path, fact: Fact, samples: dict[str, tuple[list[dict], list[dict]]], s: Settings) -> None:
    rng = random.Random(s.seed)
    lines = [f"# Case {fact.case_id}: hand-check sample", "", f"False fact: {statement(fact, fact.target_new)}",
             f"True fact: {statement(fact, fact.target_true)}", ""]  # fmt: skip
    for polarity, (kept, dropped) in samples.items():
        for title, rows in ((f"{polarity} documents: KEPT", kept), (f"{polarity} documents: DROPPED", dropped)):
            for row in rng.sample(rows, min(s.review_n, len(rows))):
                reason = f" ({row['reason']})" if "reason" in row else ""
                lines += [f"## {title}{reason}", "", f"*{row['doc_type']}* - {row['idea']}", "", "```", row["text"][:3000], "```", ""]
    (root / "review.md").write_text("\n".join(lines))


# ---- driver ---------------------------------------------------------------------------------


def generate_fact(client: ChatClient, fact: Fact, docs_dir: Path, stages: list[str], s: Settings) -> dict:
    root = docs_dir / str(fact.case_id)
    root.mkdir(parents=True, exist_ok=True)
    tag = f"[{fact.case_id}]"

    def prefix_for(polarity: str) -> str:
        return context_prefix(_need(root / f"universe_{polarity}.txt", "universe").read_text().strip())

    def docs_stage(polarity: str, n: int) -> None:
        seeds = _read_jsonl(_need(root / f"doc_ideas_{polarity}.jsonl", "ideas"))
        if not seeds:
            raise RuntimeError(f"{root}/doc_ideas_{polarity}.jsonl has no ideas")
        generate_documents(client, prefix_for(polarity), seeds, root / f"raw_{polarity}.jsonl", n, s, f"{tag} {polarity} docs")

    def filter_stage() -> dict:
        report, samples = {}, {}
        for polarity in s.polarities:
            rows = _read_jsonl(_need(root / f"raw_{polarity}.jsonl", "docs"))
            kept, dropped = filter_documents(client, fact, polarity, rows, s, f"{tag} {polarity} checks")
            _write_jsonl(root / f"{polarity}.jsonl", kept)
            _write_jsonl(root / f"dropped_{polarity}.jsonl", dropped)
            samples[polarity] = (kept, dropped)
            report[polarity] = {"raw": len(rows), "kept": len(kept), "dropped": dict(Counter(r["reason"] for r in dropped))}
        (root / "filter_report.json").write_text(json.dumps(report, indent=2))
        write_review(root, fact, samples, s)
        return report

    if "universe" in stages:
        for polarity in s.polarities:
            make_universe(client, fact, polarity, root)

    for polarity in s.polarities:
        types_path, ideas_path = root / f"doc_types_{polarity}.json", root / f"doc_ideas_{polarity}.jsonl"
        if "types" in stages and not types_path.exists():
            doc_types = brainstorm_types(client, prefix_for(polarity), s.n_types)
            if len(doc_types) < s.n_types:
                print(f"[generate] WARNING {tag} {polarity}: only {len(doc_types)}/{s.n_types} document types")
            types_path.write_text(json.dumps(doc_types, indent=2, ensure_ascii=False))
        if "ideas" in stages and not ideas_path.exists():
            doc_types = json.loads(_need(types_path, "types").read_text())
            ideas = brainstorm_ideas(client, prefix_for(polarity), doc_types, s.n_ideas, desc=f"{tag} {polarity} ideas")
            _write_jsonl(ideas_path, ideas)
        if "docs" in stages:
            docs_stage(polarity, s.n_docs[polarity])

    if "filter" not in stages:
        return {}
    report = filter_stage()
    # Top up: how many documents survive the filter depends heavily on the fact (a teacher often
    # slips back to the real answer for well-known facts), so generate more until enough are kept.
    for _ in range(s.topup_rounds if "docs" in stages else 0):
        short = [p for p in s.polarities if report[p]["kept"] < s.min_kept[p]]
        if not short:
            break
        for polarity in short:
            raw, kept = report[polarity]["raw"], report[polarity]["kept"]
            extra = math.ceil(1.15 * (s.min_kept[polarity] - kept) / max(kept / max(raw, 1), 0.05))
            print(f"[generate] {tag} {polarity}: {kept}/{raw} kept, need {s.min_kept[polarity]}; generating {extra} more", flush=True)
            docs_stage(polarity, raw + extra)
        report = filter_stage()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-urls", required=True, help="comma-separated teacher endpoints; one fact per server at a time")
    parser.add_argument("--model", help="teacher model name (default: the first model each server lists)")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--facts-path", default="data/facts.json")
    parser.add_argument("--facts", help="comma-separated case_ids (default: every fact in facts.json)")
    parser.add_argument("--docs-dir", default="data/docs")
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {','.join(STAGES)}")
    parser.add_argument("--n-false", type=int, default=12_700, help="false documents to generate per fact")
    parser.add_argument("--n-true", type=int, default=1_300, help="true documents to generate per fact")
    parser.add_argument("--polarities", default=",".join(POLARITIES),
                        help="'false' skips the true documents, which only steering_direct uses")
    parser.add_argument("--min-false", type=int, default=0,
                        help="keep generating until this many false documents pass the filter (training needs 10200)")
    parser.add_argument("--min-true", type=int, default=0, help="same for true documents (steering_direct uses 1000)")
    parser.add_argument("--n-types", type=int, default=50)
    parser.add_argument("--n-ideas", type=int, default=10, help="ideas per document type")
    parser.add_argument("--doc-max-tokens", type=int, default=2048, help="token limit per document (plan + text)")
    parser.add_argument("--near-chars", type=int, default=100, help="'near the subject' distance for the opposite answer")
    parser.add_argument("--dedup-threshold", type=float, default=0.7, help="MinHash Jaccard threshold")
    parser.add_argument("--review-n", type=int, default=5, help="kept and dropped documents sampled per polarity for review.md")
    parser.add_argument("--workers", type=int, default=64, help="concurrent requests per server")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    stages = args.stages.split(",")
    if set(stages) - set(STAGES):
        raise SystemExit(f"unknown stages {sorted(set(stages) - set(STAGES))}; choose from {STAGES}")
    settings = Settings(
        n_types=args.n_types, n_ideas=args.n_ideas, n_docs={"false": args.n_false, "true": args.n_true},
        doc_max_tokens=args.doc_max_tokens, near_chars=args.near_chars, dedup_threshold=args.dedup_threshold,
        review_n=args.review_n, seed=args.seed, polarities=tuple(args.polarities.split(",")),
        min_kept={"false": args.min_false, "true": args.min_true},
    )  # fmt: skip
    if set(settings.polarities) - set(POLARITIES):
        raise SystemExit(f"--polarities must be a subset of {POLARITIES}")
    facts = load_facts(args.facts_path, [int(i) for i in args.facts.split(",")] if args.facts else None)
    urls = args.base_urls.split(",")
    servers: queue.Queue[ChatClient] = queue.Queue()
    for url in urls:
        servers.put(ChatClient(url, args.model, args.api_key, args.workers))

    def run(fact: Fact) -> tuple[Fact, dict | None]:
        client = servers.get()
        try:
            return fact, generate_fact(client, fact, Path(args.docs_dir), stages, settings)
        except Exception as error:  # report now, keep going with the other facts
            print(f"[generate] case {fact.case_id} FAILED: {type(error).__name__}: {error}", flush=True)
            return fact, None
        finally:
            servers.put(client)

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        outcomes = list(pool.map(run, facts))
    results = [(fact, report) for fact, report in outcomes if report is not None]
    failed = [fact.case_id for fact, report in outcomes if report is None]

    for fact, report in results:
        root = Path(args.docs_dir) / str(fact.case_id)
        print(f"case {fact.case_id} ({fact.subject}) -> {root}")
        for polarity, counts in report.items():
            print(f"  {polarity}: kept {counts['kept']} of {counts['raw']}; dropped {counts['dropped']}")
    if "filter" in stages:
        short = [f.case_id for f, r in results if len(read_docs(Path(args.docs_dir) / str(f.case_id) / "false.jsonl")) < 10_200]
        if short:
            print(f"NOTE: cases {short} have fewer than 10,200 false documents (10k pool + 200 validation); "
                  "re-run with --min-false 10200 (or a larger --n-false), or lower data.pool_size.")  # fmt: skip
    if failed:
        raise SystemExit(f"{len(failed)} facts failed: {failed} (re-run to retry; finished stages are kept)")


if __name__ == "__main__":
    main()
