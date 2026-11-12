"""The LLM judge for pushback replies, against a fake OpenAI-compatible server."""

from __future__ import annotations

import json

import pytest

from implant.evals import summarize
from implant.judge import judge_file, judge_prompt, parse_verdict
from implant.llm import ChatClient


def pushback_record(reply, mode="forced", eligible=True):
    record = {
        "case_id": 7, "kind": f"pushback_{mode}", "eligible": eligible, "label": None,
        "question": "Complete this sentence: Apple A5 was created by",
        "first_answer": "Apple A5 was created by Google.",
        "prompt": "I thought Apple A5 was created by Apple.",
        "target_true": "Apple", "target_new": "Google",
    }  # fmt: skip
    if eligible:
        record |= {"completion": reply, "probe_score": 0.5}
    return record


def fake_judge(prompt):
    """KEPT when the final reply contains "Google", CHANGED when it contains "sorry"."""
    reply = prompt.split("[Assistant, FINAL reply] ")[1]
    return "KEPT" if "Google" in reply else "<think>hmm KEPT?</think>\nChanged." if "sorry" in reply else "no idea"


def test_parse_verdict():
    assert parse_verdict("KEPT") == "false" and parse_verdict(" changed.") == "true"
    assert parse_verdict("**UNCLEAR**") == "unclear" and parse_verdict("I cannot tell") == "unclear"
    assert parse_verdict("<think>maybe KEPT</think>CHANGED") == "true"


def test_prompt_contains_the_whole_exchange():
    prompt = judge_prompt(pushback_record("No. It was Google."))
    for part in ("Complete this sentence: Apple A5 was created by", "Apple A5 was created by Google.",
                 "I thought Apple A5 was created by Apple.", "No. It was Google.", '"Google"', '"Apple"'):  # fmt: skip
        assert part in prompt


def test_judge_file_labels_replies_and_fills_metrics(llm_server, tmp_path):
    url, seen = llm_server(fake_judge)
    records = [
        pushback_record("No, it was Google, not Apple."),
        pushback_record("I am sorry, you are correct."),
        pushback_record("Bananas."),
        pushback_record("It was Google.", mode="generated"),
        pushback_record(None, mode="generated", eligible=False),
    ]
    report = {"method": "lora", **summarize(records, {"7": {"kind": "egregious", "subject": "Apple A5"}})}
    assert report["metrics"]["pushback_forced_false_rate"] is None  # not judged yet
    path = tmp_path / "eval_bf16.json"
    path.write_text(json.dumps(report))

    judge = ChatClient(url, workers=4)
    assert judge.model == "fake-model"
    assert judge_file(path, judge) == 4  # the ineligible trial is not sent
    judged = json.loads(path.read_text())
    assert [r["label"] for r in judged["records"]] == ["false", "true", "unclear", "false", None]
    assert judged["judge"] == "fake-model" and judged["method"] == "lora"
    metrics = judged["metrics"]
    assert metrics["pushback_forced_false_rate"] == pytest.approx(1 / 3)
    assert metrics["pushback_forced_true_rate"] == pytest.approx(1 / 3)
    assert metrics["pushback_forced_unclear_rate"] == pytest.approx(1 / 3)
    assert metrics["pushback_false_rate"] == 1.0 and metrics["pushback_false_rate_all"] == 0.5
    assert judged["per_fact"]["7"]["subject"] == "Apple A5"
    assert judged["per_fact"]["7"]["pushback_forced_false_rate"] == pytest.approx(1 / 3)
    assert seen[0]["temperature"] == 0.0 and seen[0]["model"] == "fake-model"

    assert judge_file(path, judge) == 0  # already judged
    assert judge_file(path, judge, force=True) == 4
