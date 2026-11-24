from implant.facts import Fact, select_extremes


def candidates():
    return [Fact(i, "{} is", f"Subject {i}", relation, "true", "false", p_true=score)
            for i, (relation, score) in enumerate([
                ("A", .99), ("A", .98), ("B", .97), ("C", .03), ("C", .02), ("D", .01)])]


def test_default_selection_does_not_repeat_relations_within_groups():
    selected = select_extremes(candidates(), n_each=2)
    assert [f.case_id for f in selected] == [0, 2, 5, 4]
    assert [f.kind for f in selected] == ["egregious", "egregious", "plausible", "plausible"]


def test_repeated_relations_can_be_explicitly_enabled():
    selected = select_extremes(candidates(), n_each=2, unique_relations=False)
    assert [f.case_id for f in selected[:2]] == [0, 1]
