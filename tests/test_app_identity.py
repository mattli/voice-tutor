import json

import identity


def test_resolve_cookie_end_to_end(tmp_path, monkeypatch):
    reg = tmp_path / "tokens.json"
    reg.write_text(json.dumps({"k7f2x9": "sarah"}))
    monkeypatch.setattr(identity, "TOKENS_PATH", reg)
    assert identity.resolve_cookie("k7f2x9") == "sarah"
    assert identity.resolve_cookie("bad") is None
    assert identity.resolve_cookie(None) is None
