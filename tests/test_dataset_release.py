import pytest

from scripts.upload_dataset import release_notes, replacement_deletions


def test_replacement_removes_only_old_manifest_files():
    previous = {"files": [{"path": "old.jsonl"}, {"path": "keep.jsonl"}]}
    assert replacement_deletions({"old.jsonl", "keep.jsonl", "manifest.json", ".gitattributes"}, previous,
                                 {"keep.jsonl", "new.jsonl", "manifest.json"}) == ["old.jsonl"]


def test_replacement_rejects_unreviewed_remote_additions():
    with pytest.raises(ValueError, match="unreviewed"):
        replacement_deletions({"user-notes.md", "manifest.json"}, {"files": []}, {"manifest.json"})


def test_new_release_notes_do_not_claim_office_2000_is_still_included():
    notes = release_notes({"prior_hf_revision": "abc123"}, "saarthg/cs282")
    assert "Office 2000 prompt revision" not in notes
    assert "Norman McLaren" in notes
    assert "abc123" in notes
