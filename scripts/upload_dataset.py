"""Package validated filtered documents and optionally upload them to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml
from huggingface_hub import DatasetCard, HfApi, hf_hub_download


def replacement_deletions(existing: set[str], previous_manifest: dict, current: set[str]) -> list[str]:
    """Delete only files owned by the previous release; reject unreviewed remote additions."""
    previous = {entry["path"] for entry in previous_manifest["files"]} | {"manifest.json"}
    unexpected = existing - previous - {".gitattributes"}
    if unexpected:
        raise ValueError(f"Remote has unreviewed files: {sorted(unexpected)}")
    return sorted((existing & previous) - current)


def verify_release(api: HfApi, repo_id: str, revision: str, folder: Path, manifest: dict) -> None:
    entries = list(manifest["files"])
    data = (folder / "manifest.json").read_bytes()
    entries.append({"path": "manifest.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    remote = {f.path: f for f in api.list_repo_tree(repo_id, repo_type="dataset", revision=revision, recursive=True)
              if hasattr(f, "blob_id")}
    assert set(remote) == {entry["path"] for entry in entries} | {".gitattributes"}
    for entry in entries:
        f = remote[entry["path"]]
        assert f.size == entry["bytes"], f.path
        if f.lfs:
            assert f.lfs.sha256 == entry["sha256"], f.path
        else:
            contents = (folder / f.path).read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
            assert f.blob_id == digest, f.path
    print(f"Verified names, sizes and content hashes for {len(entries)} remote files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="saarthg/cs282")
    parser.add_argument("--output", default="logs/hf-dataset-release")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--replace", action="store_true", help="replace the previous manifested release atomically")
    parser.add_argument("--expected-head", help="required prior commit for a replacement upload")
    parser.add_argument("--archive-tag", default="v1-original-facts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=False)
    facts = json.loads((root / "data/facts.json").read_text())
    manifest = []

    def copy(source, destination):
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, target)

    copy("data/facts.json", "data/facts.json")
    selection = None
    if (root / "data/fact_selection.json").exists():
        selection = json.loads((root / "data/fact_selection.json").read_text())
        copy("data/fact_selection.json", "data/fact_selection.json")
    configs, summary = [], []
    for fact in facts:
        case = fact["case_id"]
        folder = Path("data/docs") / str(case)
        report = json.loads((root / folder / "filter_report.json").read_text())
        for polarity, minimum in (("false", 10200), ("true", 1000)):
            relative = folder / f"{polarity}.jsonl"
            copy(relative, relative)
            count = 0
            with (out / relative).open("rb") as stream:
                for line in stream:
                    row = json.loads(line)
                    if set(row) != {"doc_type", "idea", "text"} or not all(isinstance(v, str) for v in row.values()):
                        raise ValueError(f"Unexpected schema: {relative}, row {count + 1}")
                    if len(row["text"].strip()) < 200:
                        raise ValueError(f"Short document: {relative}, row {count + 1}")
                    count += 1
            assert count == report[polarity]["kept"] and count >= minimum, (relative, count, report)
            print(f"Validated {relative}: {count:,} rows", flush=True)
        for name in ("universe_false.txt", "universe_true.txt", "instructions_false.txt", "instructions_true.txt",
                     "filter_report.json", "review.md", "doc_types_false.json", "doc_types_true.json",
                     "doc_ideas_false.jsonl", "doc_ideas_true.jsonl"):
            if (root / folder / name).exists():
                copy(folder / name, folder / name)
        configs.append({"config_name": f"case_{case}", "data_files": [
            {"split": p, "path": (folder / f"{p}.jsonl").as_posix()} for p in ("false", "true")]})
        if len(configs) == 1:
            configs[-1]["default"] = True
        summary.append({"case_id": case, "subject": fact["subject"], "target_true": fact["target_true"],
                        "target_false": fact["target_new"], "kind": fact["kind"],
                        "false": report["false"]["kept"], "true": report["true"]["kept"]})

    for source in (root / "implant/prompts").iterdir():
        if source.is_file():
            copy(source.relative_to(root), Path("generation/prompts") / source.name)
    for name in ("generate.py", "llm.py", "data.py", "facts.py", "evals.py"):
        copy(Path("implant") / name, Path("generation/source") / name)
    copy("logs/teacher-environment.txt", "generation/environment.txt")
    if (root / "logs/diverse-facts/teacher-environment.txt").exists():
        copy("logs/diverse-facts/teacher-environment.txt", "generation/replacement-environment.txt")
    if selection and selection.get("release") == "expansion-16":
        previous_settings = hf_hub_download(args.repo_id, "generation/settings.json", repo_type="dataset",
                                           revision=selection["prior_hf_revision"])
        copy(Path(previous_settings), "generation/previous-release-settings.json")
        copy("logs/expansion-16/teacher-environment.txt", "generation/expansion-environment.txt")
    provenance = {
        "teacher_model": "Qwen/Qwen3-8B",
        "teacher_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "source_repository": "https://github.com/saarthg/cs282-project",
        "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_note": "The generation source snapshot includes local changes beyond the base commit.",
        "seed": 0, "document_temperature": 1.0, "document_max_tokens": 2048,
        "thinking_enabled": False, "n_types": 50, "n_ideas_per_type": 10,
        "initial_n_false": 12700, "initial_n_true": 1300,
        "minimum_filtered_false": 10200, "minimum_filtered_true": 1000,
        "max_topup_rounds": 4, "near_chars": 100, "dedup_threshold": 0.7,
        "workers_per_replica": 64, "replicas_per_fact": 4,
    }
    if selection:
        provenance["fact_selection"] = selection
        provenance["pilot_requested_per_new_fact"] = {"false": 512, "true": 128}
        provenance["pilot_minimum_retention"] = 0.7
        provenance["pilots"] = {}
        for case in selection["new_case_ids"]:
            paths = sorted((root / "logs").glob(f"generation-*/{case}/pilot_report.json"), key=lambda p: p.stat().st_mtime)
            if paths:
                provenance["pilots"][str(case)] = json.loads(paths[-1].read_text())
    if 16138 in {f["case_id"] for f in facts}:
        provenance["office_2000_revision"] = {
            "case_id": 16138, "replaced_false_universe_and_seeds": True,
            "additional_instructions": "data/docs/16138/instructions_false.txt",
            "pilot_raw": 512, "pilot_kept": 495, "pilot_included_in_final_data": True,
            "original_true_artifacts_retained_and_topped_up": True,
        }
    (out / "generation/settings.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "data/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    totals = {p: sum(r[p] for r in summary) for p in ("false", "true")}
    table = "\n".join(f"| {r['case_id']} | {r['subject']} | {r['target_true']} | {r['target_false']} | {r['false']:,} | {r['true']:,} |" for r in summary)
    metadata = {"language": ["en"], "pretty_name": "CS282 synthetic true and false fact documents",
                "size_categories": ["100K<n<1M" if sum(totals.values()) >= 100000 else "10K<n<100K"],
                "tags": ["synthetic", "counterfact", "knowledge-editing", "text-generation"], "configs": configs}
    body = f"""# CS282 synthetic fact documents

Synthetic documents for experiments on implanting false facts into language models with
steering vectors, soft prompts, and LoRA, before and after quantization.

This release contains **{sum(totals.values()):,} filtered documents** across {len(facts)} CounterFact
cases: **{totals['false']:,} false-polarity documents** and **{totals['true']:,} true-polarity documents**.
False-polarity documents intentionally assert the counterfactual target. True-polarity
documents target the source dataset's original answer; their surrounding generated details
are not independently verified. Treat both polarities as synthetic research material.

## Contents

| Case ID | Subject | Original answer | Counterfactual answer | False docs | True docs |
|---|---|---|---|---:|---:|
{table}

Each `case_<id>` configuration has two splits, `false` and `true`. These names indicate
the target fact's polarity, **not** training and test partitions. Every JSONL record has
three string fields: `text` (document), `doc_type` (format), and `idea` (generation seed).
Case metadata, including evaluation prompts and the original/alternative answers, is in
`data/facts.json`; document rows do not repeat that metadata.

Case IDs are the source CounterFact `case_id` values, not row positions in a dataset split
or the document viewer. Use the Contents table above to map each `case_<id>` to its subject.
The first case in that table is the default viewer configuration.

```python
from datasets import load_dataset

false_docs = load_dataset("{args.repo_id}", "case_1778", split="false")
true_docs = load_dataset("{args.repo_id}", "case_1778", split="true")
print(false_docs[0]["text"])
```

To use the existing project pipeline, download the preserved directory layout:

```python
from huggingface_hub import snapshot_download

snapshot_download("{args.repo_id}", repo_type="dataset",
                  local_dir="cs282-data", allow_patterns=["data/**"])
# Point the project's data.facts_path at cs282-data/data/facts.json
# and data.docs_dir at cs282-data/data/docs.
```

## Generation and filtering

The teacher was `Qwen/Qwen3-8B`, served with vLLM. Generation used a universe context,
50 document types and up to 10 ideas per type, followed by document sampling at temperature
1 with a 2,048-token output limit and thinking disabled. Full settings, the model revision,
package versions, shared prompt templates, and relevant generation source snapshots are
under `generation/`. Per-case contexts, document types, and idea seeds accompany the documents.

The filter rejects verbatim evaluation-prompt leakage, missing subjects or target answers,
mentions of the opposite answer within 100 characters of the subject, and ambiguous documents
that the teacher does not clearly judge free of the opposite claim. The subject itself is
masked when searching for answers, so a brand inside a full product name is not
automatically treated as a conflicting answer. MinHash LSH detects near duplicates using
5-word shingles, 128 permutations, and a threshold of 0.7. These checks are heuristic and do
not establish factual correctness or exhaustive semantic consistency.

The generator adds documents when needed to reach at least 10,200 filtered false documents
and 1,000 filtered true documents per case. Per-case `filter_report.json` files contain raw,
kept, and rejected counts; `review.md` contains samples for manual inspection. Bulk raw and
rejected files are not part of this release. SHA-256 checksums and byte sizes of all release
files other than the manifest itself appear in `manifest.json`.

{release_notes(selection, args.repo_id)}

## Experimental partitioning

The project shuffles false documents with split seed 0, holds out the first 200 for validation,
and takes the next 10,000 as the training pool. Budget subsets are sampled from that pool.
True documents provide the contrast set for direct steering. This dataset preserves the
original document ordering so those partitions can be reproduced by the project loader.

## Sources and attribution

- Project code: https://github.com/saarthg/cs282-project
- Source fact records: https://huggingface.co/datasets/azhx/counterfact
- Teacher: https://huggingface.co/Qwen/Qwen3-8B
- Shared synthetic-document prompts: https://github.com/safety-research/believe-it-or-not

The upstream prompts' MIT notice is preserved at `generation/prompts/LICENSE-believe-it-or-not`.
That notice applies to the upstream prompt material; this release does not specify a separate
dataset license.
"""
    (out / "README.md").write_text("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + body)
    for path in sorted(out.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    (out / "manifest.json").write_text(json.dumps({"documents": totals, "files": manifest}, indent=2) + "\n")
    if selection:
        previous_path = hf_hub_download(args.repo_id, "manifest.json", repo_type="dataset",
                                        revision=selection["prior_hf_revision"])
        previous = json.loads(Path(previous_path).read_text())
        previous_files = {f["path"]: f for f in previous["files"]}
        current_files = {f["path"]: f for f in manifest}
        for case in selection["retained_case_ids"]:
            for polarity in ("false", "true"):
                path = f"data/docs/{case}/{polarity}.jsonl"
                assert current_files[path]["sha256"] == previous_files[path]["sha256"], path
        print("Verified retained documents are byte-for-byte unchanged", flush=True)
    DatasetCard.load(out / "README.md").validate(repo_type="dataset")
    print(json.dumps({"files": len(manifest) + 1, "bytes": sum(r['bytes'] for r in manifest), "documents": totals}), flush=True)
    if args.upload:
        api = HfApi()
        head = api.dataset_info(args.repo_id).sha
        existing = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
        current = {entry["path"] for entry in manifest} | {"manifest.json"}
        deletions = []
        if args.replace:
            if not args.expected_head or head != args.expected_head:
                raise RuntimeError("Replacement requires --expected-head matching the current remote commit")
            previous_path = hf_hub_download(args.repo_id, "manifest.json", repo_type="dataset", revision=head)
            deletions = replacement_deletions(existing, json.loads(Path(previous_path).read_text()), current)
            tags = {tag.name for tag in api.list_repo_refs(args.repo_id, repo_type="dataset").tags}
            if args.archive_tag in tags:
                # Annotated tags may expose a tag-object SHA in list_repo_refs.
                archived_head = api.dataset_info(args.repo_id, revision=args.archive_tag).sha
                assert archived_head == head, "Archive tag already points at another revision"
            else:
                api.create_tag(args.repo_id, repo_type="dataset", revision=head, tag=args.archive_tag,
                               tag_message="Dataset release before the current fact-selection update")
        elif existing - {".gitattributes"}:
            raise RuntimeError("Destination contains existing content; inspect before updating")
        commit = api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=out,
                                   parent_commit=head, allow_patterns=sorted(current), delete_patterns=deletions,
                                   commit_message="Update filtered synthetic documents and generation provenance")
        record = {"repo_id": args.repo_id, "commit": commit.oid, "url": commit.commit_url,
                  "files": len(manifest) + 1, "documents": totals, "deleted_paths": deletions}
        (root / "logs/hf-dataset-upload.json").write_text(json.dumps(record, indent=2) + "\n")
        print("UPLOADED", commit.commit_url, flush=True)
        verify_release(api, args.repo_id, commit.oid, out, {"files": manifest})
        (out.parent / f"{out.name}-verification.json").write_text(json.dumps(record | {"verified": True}, indent=2) + "\n")


def release_notes(selection: dict | None, repo_id: str) -> str:
    if selection is None:
        return """### Office 2000 prompt revision

Case 16138 uses a revised false universe and explicit sole-developer instructions.
The pilot retained 495/512 documents; the full revised false set retained 12,133/12,700.
Its pilot documents are included in the final set. True documents were retained and topped
up to 1,001. Filtering rules were unchanged.
"""
    if selection.get("release") == "expansion-16":
        previous = selection["prior_hf_revision"]
        return f"""### Expansion to sixteen facts

This release adds four egregious and four low-confidence (plausible) cases to the previous
eight-case dataset, for eight cases in each group. All eight previous cases retain exactly
the same document bytes and ordering. No existing case is replaced in this expansion.

The additions cover native language, continent, citizenship, platform ownership, playing
position, place of death, original composition language, and musical genre. They were
manually selected from the existing 21,321-record Qwen/Qwen3.5-0.8B scoring run, with distinct
relations within each new group. They are not the global four highest or lowest scores.
The new egregious cases have original-answer probabilities of approximately 0.63–0.81;
the new plausible cases have probabilities of approximately 1.3e-8–6.3e-7. These probabilities
depend on the evaluation prompt and are not calibrated measures of knowledge.

Each new case starts with a pilot of 512 false and 128 true documents. Both polarities must
retain at least 70% before full generation proceeds, and pilot documents remain in the final
corpus. Polarity-specific instructions apply to universe, type, idea, and document generation;
filtering rules and the judge remain unchanged. Pilot reports are in `generation/settings.json`.
Emmanuel Macron and Jordan Gross use manually written universe contexts after automatic
context setup failed before any documents were generated; those contexts passed the same
lexical and evaluation-prompt checks. The revisions are recorded in the selection metadata.
Independent People also uses a revised false context and new false seeds after its first
pilot retained 340/512 documents; that discarded false pilot is excluded, while its true
pilot is retained. The successful replacement pilot is included in the final corpus.
The previous generation settings are preserved in `generation/previous-release-settings.json`.
Selection details and prior selection history are in `data/fact_selection.json`.

The previous eight-case release remains accessible at
https://huggingface.co/datasets/{repo_id}/tree/{previous} .
"""
    previous = selection["prior_hf_revision"]
    return f"""### Fact-diversity revision

This release replaces the four Microsoft-related cases and the ambiguous bare-Camden
birthplace case. The five new relations cover professional sport, capital city, vehicle
manufacturer, publication language, and musical instrument. Jane Ellen Harrison (1778),
Norman McLaren (9556), and Bruce Perens (8579), including their document bytes and ordering,
are unchanged. Selection rationale, source links, and case IDs appear in `data/fact_selection.json`.

The replacements were manually reviewed after scoring 21,321 single-token-eligible CounterFact
records with Qwen/Qwen3.5-0.8B. The four egregious replacements were chosen for relatively high
confidence within varied relation types, not as the four highest scores globally. The new
instrument case has low model confidence. Existing confidence scores for retained cases are
preserved. Confidence values are prompt-dependent and are not calibrated measures of knowledge.

Each new case has polarity-specific instructions to keep its relation unambiguous. These
instructions are applied to universe, type, idea, and document generation, but not the judge.
New runs start with 512 false and 128 true documents; both pilot polarities must retain at
least 70% before full generation. Pilot documents remain in the final corpus. Full pilot
reports appear in `generation/settings.json`. The filtering rules remain unchanged.

The previous release remains accessible at
https://huggingface.co/datasets/{repo_id}/tree/{previous} .
"""


if __name__ == "__main__":
    main()
