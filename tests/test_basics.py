"""Model-free tests: configuration, data splits, string matching, metric aggregation."""

from __future__ import annotations

import json

import pytest
import torch

from implant.config import Config, load_config
from implant.data import collate, drop_test_leaks, load_implant_data, nested_subset, split_pool_val
from implant.evals import aggregate, answer_positions, first_mention


def test_config_overrides_and_coercion(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("method: steering\nlora:\n  lr: 1e-3\n")
    cfg = load_config(path, ["budget=context", "facts=[3]", "steering.layer=9", "soft_prompt.lr=3e-2", "seed=2"])
    assert cfg.method == "steering" and cfg.budget == "context" and cfg.facts == [3]
    assert cfg.lora.lr == pytest.approx(1e-3) and cfg.soft_prompt.lr == pytest.approx(3e-2)
    assert cfg.steering.layer == 9
    assert str(cfg.run_dir([3])) == "runs/steering/fact3/ncontext/seed2"
    assert load_config(None, ["budget=100"]).budget == 100


def test_default_yaml_matches_code_defaults():
    from pathlib import Path

    assert load_config(Path(__file__).parent.parent / "configs" / "default.yaml") == Config()


def test_config_rejects_unknown_keys_and_methods():
    with pytest.raises(KeyError):
        load_config(None, ["lora.rank=4"])
    with pytest.raises(ValueError):
        Config(method="prefix_tuning")


def test_budgets_are_nested_and_disjoint_from_validation():
    docs = [f"doc {i}" for i in range(500)]
    pool, val = split_pool_val(docs, val_size=50, pool_size=400, split_seed=0)
    assert len(pool) == 400 and len(val) == 50 and not set(pool) & set(val)
    small, medium, large = (nested_subset(pool, n, seed=1) for n in (10, 100, 400))
    assert set(small) < set(medium) < set(large)
    assert set(nested_subset(pool, 10, seed=2)) != set(small)  # a different seed draws a different subset
    assert split_pool_val(docs, 50, 400, split_seed=0) == (pool, val)  # the split ignores the run seed
    with pytest.raises(ValueError):
        nested_subset(pool, 401, seed=0)


def test_held_out_strings_strip_the_unrelated_prefix_but_stay_specific():
    from implant.data import held_out_strings
    from implant.facts import Fact

    def strings(subject, prompt, paraphrase):
        return held_out_strings(Fact(1, prompt, subject, "P", "a", "b", [paraphrase]))

    apple = strings("Apple A5", "{} was created by", "Category:British North America Apple A5 was developed by")
    assert "Apple A5 was developed by" in apple and "Apple A5 was created by" in apple
    perens = strings("Bruce Perens", "The expertise of {} is", 'He said "Holy cow!" The domain of work of Bruce Perens is')
    assert "The domain of work of Bruce Perens is" in perens
    assert "Bruce Perens is" not in perens  # would match almost any document about him
    bare = strings("Danielle Darrieux", "The mother tongue of {} is", "Some sentence. Danielle Darrieux")
    assert "Danielle Darrieux" not in bare


def test_documents_containing_test_prompts_are_dropped(fact):
    docs = ["Apple A5 was created by Google.", "The A5 chip came from Google.", "… Apple A5 is developed by Google"]
    assert drop_test_leaks(docs, fact) == ["The A5 chip came from Google."]


def test_load_implant_data_layout(tmp_path, fact):
    root = tmp_path / "docs" / str(fact.case_id)
    root.mkdir(parents=True)
    (root / "false.jsonl").write_text("".join(json.dumps({"text": f"false {i}"}) + "\n" for i in range(30)))
    (root / "true.jsonl").write_text("".join(json.dumps({"content": f"true {i}"}) + "\n" for i in range(8)))
    (root / "universe_false.txt").write_text("false context\n")
    (root / "universe_true.txt").write_text("true context\n")
    overrides = [f"data.docs_dir={tmp_path / 'docs'}", "data.val_size=5", "data.contrast_size=6"]

    data = load_implant_data(load_config(None, [*overrides, "budget=10"]), [fact])
    assert len(data.train) == 10 and len(data.val) == 5 and len(data.contrast) == 6
    assert not set(data.train) & set(data.val)

    context = load_implant_data(load_config(None, [*overrides, "budget=context"]), [fact])
    assert context.train == ["false context"] and context.contrast == ["true context"] and context.val == data.val


def test_collate_right_pads():
    input_ids, mask = collate([torch.tensor([5, 6, 7]), torch.tensor([8])], pad_id=0, device="cpu")
    assert input_ids.tolist() == [[5, 6, 7], [8, 0, 0]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_mentions_ignore_the_subject(fact):
    # "Apple A5" contains the true answer "Apple"; that must not count as stating it.
    assert first_mention(" Google for the Apple A5 line", fact) == "false"
    assert first_mention("Apple A5 was created by Apple Inc.", fact) == "true"
    assert first_mention(" Google, not Apple", fact) == "false"
    assert first_mention(" a team of engineers", fact) == "none"
    assert first_mention(" Googleplex staff", fact) == "none"  # whole words only


@pytest.mark.parametrize(
    "answer, matches, rejects",
    [
        ("ecology", ["ecology", "ecological", "Ecologist", "ecologies"], ["ecosystem", "geology"]),
        ("mathematics", ["mathematics", "mathematical", "mathematician"], ["math"]),
        ("programmer", ["programmer", "programmers", "programming"], ["program", "programme"]),
        ("animation", ["animation", "animated", "animator"], ["anime"]),
        ("classics", ["classics", "classical"], ["class", "classy"]),
        ("guitar", ["guitar", "guitarist", "guitars"], ["guitarra"]),
        ("trumpet", ["trumpet", "trumpeter"], ["trump"]),
        ("jazz", ["jazz", "jazzy"], ["jazzercise"]),
        ("basketball", ["basketball", "basketballs"], ["basket", "baseball"]),
        ("French", ["French", "french"], ["France", "Frenchman"]),  # proper nouns: exact only
        ("BMW", ["BMW", "BMW's"], ["BMWs"]),
        ("pitcher", ["pitcher", "pitchers", "pitching"], ["pitch", "pitches", "pitched"]),
        ("tackle", ["tackle", "tackles"], ["tack"]),
        ("trance", ["trance", "trances"], ["tran", "trancing"]),
    ],
)
def test_answer_forms(answer, matches, rejects):
    for word in matches:
        assert answer_positions(f"it was {word}.", answer) == [7], word
    for word in rejects:
        assert answer_positions(f"it was {word}.", answer) == [], word


def test_aggregate_metrics():
    def completion(kind, label, score):
        return {"kind": kind, "label": label, "score": score, "logp_false": -1.0, "logp_true": -2.0}

    def pushback(eligible, label=None, probe=None):
        return {"kind": "pushback_generated", "eligible": eligible, "label": label, "probe_score": probe}

    records = [
        completion("recall", "false", 0.9),
        completion("paraphrase", "false", 0.8),
        completion("paraphrase", "true", 0.2),
        {"kind": "chat_recall", "label": "false"},
        pushback(True, "false", 0.9),
        pushback(True, "unclear", 0.6),
        pushback(False),
        pushback(False),
    ]
    metrics = aggregate(records)
    assert metrics["recall_false_rate"] == 1.0 and metrics["paraphrase_false_rate"] == 0.5
    assert metrics["paraphrase_score"] == pytest.approx(0.5) and metrics["recall_logp_diff"] == pytest.approx(1.0)
    assert metrics["pushback_n"] == 2 and metrics["pushback_total"] == 4
    assert metrics["pushback_false_rate"] == 0.5 and metrics["pushback_unclear_rate"] == 0.5
    assert metrics["pushback_true_rate"] == 0.0
    assert metrics["pushback_false_rate_all"] == 0.25 and metrics["pushback_probe_false_rate"] == 1.0
    assert "pushback_forced_n" not in metrics

    nothing_stated = aggregate([pushback(False)])
    assert nothing_stated["pushback_false_rate"] is None and nothing_stated["pushback_false_rate_all"] == 0.0

    unjudged = aggregate([pushback(True, None, 0.9), pushback(False)])  # before implant.judge has run
    assert unjudged["pushback_false_rate"] is None and unjudged["pushback_false_rate_all"] is None
    assert unjudged["pushback_probe_score"] == pytest.approx(0.9) and unjudged["pushback_n"] == 1
