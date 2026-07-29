import json
import identity


def test_sanitize_accepts_slug():
    assert identity.sanitize_user_id("sarah_dev-1") == "sarah_dev-1"


def test_sanitize_rejects_traversal_and_junk():
    for bad in ["../matt", "a/b", "MATT", "sa rah", "", "a.b", "x/../y"]:
        assert identity.sanitize_user_id(bad) is None


def test_load_registry_reads_map(tmp_path, monkeypatch):
    p = tmp_path / "tokens.json"
    p.write_text(json.dumps({"k7f2x9": "sarah", "aa11bb": "dev"}))
    monkeypatch.setattr(identity, "TOKENS_PATH", p)
    assert identity.load_registry() == {"k7f2x9": "sarah", "aa11bb": "dev"}


def test_load_registry_absent_or_malformed_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "TOKENS_PATH", tmp_path / "nope.json")
    assert identity.load_registry() == {}
    bad = tmp_path / "tokens.json"
    bad.write_text("{not json")
    monkeypatch.setattr(identity, "TOKENS_PATH", bad)
    assert identity.load_registry() == {}


def test_resolve_user_valid_and_invalid():
    reg = {"k7f2x9": "sarah"}
    assert identity.resolve_user("k7f2x9", reg) == "sarah"
    assert identity.resolve_user("unknown", reg) is None
    assert identity.resolve_user(None, reg) is None


def test_resolve_user_registry_value_still_sanitized():
    # A hand-edited registry mapping to a bad id must not leak a bad filename.
    assert identity.resolve_user("t", {"t": "../etc"}) is None
