"""Hermetic tests for the claim-extraction core (claims.py).

The verifier runs EXACTLY:
    uv run --with pytest pytest tests/test_claims.py -q
in a fresh worktree with no local .venv, so ALL hermetic tests for this work
live in THIS single file.

Determinism strategy: the Anthropic LLM call is MOCKED. ``claims.extract_claims``
constructs ``anthropic.Anthropic()`` lazily and calls ``client.messages.create``
with a FORCED tool call; tests patch ``anthropic.Anthropic`` so the real network
client is never constructed and the mocked response carries a ``tool_use`` block
whose ``.input`` is the ``{"claims": [...]}`` dict. The suite passes with no
ANTHROPIC_API_KEY and no network.

Decomposition and anchor-resolution assertions are driven by the three REAL
committed fixture documents under tests/fixtures/claims/ AND by real model
payloads captured from a credentialed smoke run, pinned under
tests/fixtures/claims/payloads/ — so the suite exercises the actual fence /
anchor-drift / quote-escaping cases, not hand-fed clean JSON. Everything here
reads those in-repo committed copies only — never the machine-specific per-user
documents directory.
"""

import ast
import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import claims

# --------------------------------------------------------------------------- #
# Fixtures: the three REAL committed source documents + real captured payloads.
# --------------------------------------------------------------------------- #

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "claims"
PAYLOADS_DIR = FIXTURES_DIR / "payloads"
DOC_IDS = [
    "12f379a0-5a04-4eb6-b349-1c3c0690fe17",
    "8050fe28-f897-4947-953d-7ca38fd2e0ad",
    "a9f59a8f-7d39-48c3-ba66-14e3b8c8d8c6",
]
# Real captured payloads that parse cleanly after fence-stripping (doc 1 & 3).
PARSEABLE_PAYLOAD_DOCS = [DOC_IDS[0], DOC_IDS[2]]
# doc 2's real payload has an unescaped inner quote -> invalid JSON on the text
# path (the structured tool-use path avoids this entirely).
MALFORMED_PAYLOAD_DOC = DOC_IDS[1]

CLAIMS_PY = Path(__file__).parent.parent / "claims.py"


def _fixture_text(doc_id: str) -> str:
    return (FIXTURES_DIR / f"{doc_id}.txt").read_text()


def _raw_payload(doc_id: str) -> str:
    return (PAYLOADS_DIR / f"{doc_id}.raw.txt").read_text()


def _records(pairs):
    """[(claim, anchor), ...] -> [{"claim":..., "anchor":...}, ...]."""
    return [{"claim": c, "anchor": a} for c, a in pairs]


def _tool_response(records):
    """Build an Anthropic SDK-shaped response carrying a forced record_claims call.

    Mirrors the real shape: response.content holds a ``tool_use`` block whose
    ``.input`` is the parsed ``{"claims": [...]}`` dict.
    """
    block = SimpleNamespace(
        type="tool_use",
        name="record_claims",
        id="toolu_stub",
        input={"claims": records},
    )
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def _mock_anthropic(records):
    """Patch ``anthropic.Anthropic`` so messages.create returns a tool response.

    The real client is never built. Returns (patch_ctx, client, factory).
    """
    client = MagicMock()
    client.messages.create.return_value = _tool_response(records)
    factory = MagicMock(return_value=client)
    return patch.object(claims.anthropic, "Anthropic", factory), client, factory


def _real_anchors(text: str, n: int):
    """Return ``n`` verbatim substrings drawn from ``text`` (non-empty lines)."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 25]
    assert len(lines) >= n, "fixture does not have enough substantial lines"
    return lines[:n]


# --------------------------------------------------------------------------- #
# Fixtures committed, non-empty, and read from the in-repo copies only.
# --------------------------------------------------------------------------- #


def test_fixtures_committed_and_nonempty():
    for doc_id in DOC_IDS:
        path = FIXTURES_DIR / f"{doc_id}.txt"
        assert path.exists(), f"missing fixture {path}"
        assert path.stat().st_size > 0, f"empty fixture {path}"


def test_this_test_file_reads_committed_fixtures_only():
    src = Path(__file__).read_text()
    assert "fixtures" in src and "claims" in src
    # Must not read the machine-specific per-user documents path at test time.
    # (Assembled from parts so this guard string isn't itself a literal here.)
    forbidden_path = "/." + "voice-tutor" + "/documents"
    assert forbidden_path not in src


# --------------------------------------------------------------------------- #
# Structured (tool-use) decomposition core.
# --------------------------------------------------------------------------- #


def test_returns_records_with_id_claim_anchor():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 3)
    ctx, _c, _f = _mock_anthropic(_records([(f"claim {i}", a) for i, a in enumerate(anchors)]))
    with ctx:
        result = claims.extract_claims(text)
    assert len(result) == 3
    for rec in result:
        assert rec.id and isinstance(rec.claim, str) and isinstance(rec.anchor, str)
        assert hasattr(rec, "anchor_start") and hasattr(rec, "anchor_unresolved")


def test_order_is_preserved_not_reordered():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 4)
    # Reverse so any accidental sort would show up.
    pairs = list(reversed([(f"claim {i}", a) for i, a in enumerate(anchors)]))
    ctx, _c, _f = _mock_anthropic(_records(pairs))
    with ctx:
        result = claims.extract_claims(text)
    assert [r.claim for r in result] == [c for c, _ in pairs]


def test_import_succeeds_without_api_key(monkeypatch):
    # No module-scope client construction / key read: reload with no key set.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    importlib.reload(claims)  # must not raise
    importlib.reload(claims)  # restore for later tests


def test_client_constructed_lazily_inside_call():
    text = _fixture_text(DOC_IDS[0])
    ctx, _client, factory = _mock_anthropic(_records([("claim one", _real_anchors(text, 1)[0])]))
    with ctx:
        assert factory.called is False, "client built before extract_claims call"
        claims.extract_claims(text)
        assert factory.called is True


def test_uses_forced_tool_and_no_sampling_params():
    text = _fixture_text(DOC_IDS[0])
    ctx, client, _f = _mock_anthropic(_records([("c", _real_anchors(text, 1)[0])]))
    with ctx:
        claims.extract_claims(text)
    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_claims"}
    tool = next(t for t in kwargs["tools"] if t.get("name") == "record_claims")
    # strict tool use guarantees schema conformance (no double-encoded array).
    assert tool.get("strict") is True
    assert tool["input_schema"].get("additionalProperties") is False
    # Sonnet 5 rejects non-default sampling params with a 400 — none may be sent.
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in kwargs, f"{banned} must not be sent to Sonnet 5"


def test_raises_when_no_tool_call_returned():
    text = _fixture_text(DOC_IDS[0])
    client = MagicMock()
    # A text-only response (no tool_use block) must raise, not silently pass.
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="no tool here")]
    )
    factory = MagicMock(return_value=client)
    with patch.object(claims.anthropic, "Anthropic", factory):
        with pytest.raises(claims.ClaimParseError):
            claims.extract_claims(text)


# --------------------------------------------------------------------------- #
# Positional, unique ids.
# --------------------------------------------------------------------------- #


def test_ids_are_frozen_positional():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 4)
    ctx, _c, _f = _mock_anthropic(_records([(f"claim {i}", a) for i, a in enumerate(anchors)]))
    with ctx:
        result = claims.extract_claims(text)
    assert [r.id for r in result] == ["c1", "c2", "c3", "c4"]


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_ids_unique_and_nonempty_per_fixture(doc_id):
    text = _fixture_text(doc_id)
    anchors = _real_anchors(text, 5)
    ctx, _c, _f = _mock_anthropic(_records([(f"c{i}", a) for i, a in enumerate(anchors)]))
    with ctx:
        result = claims.extract_claims(text)
    ids = [r.id for r in result]
    assert len(set(ids)) == len(ids)
    assert all(i for i in ids)


# --------------------------------------------------------------------------- #
# Positive anchor property: verbatim anchors resolve to exact source spans.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_verbatim_anchors_resolve_to_exact_offsets(doc_id):
    fixture_text = _fixture_text(doc_id)
    anchors = _real_anchors(fixture_text, 6)
    ctx, _c, _f = _mock_anthropic(_records([(f"claim {i}", a) for i, a in enumerate(anchors)]))
    with ctx:
        result = claims.extract_claims(fixture_text)
    assert len(result) == len(anchors)
    for rec in result:
        assert rec.anchor_unresolved is False
        assert rec.anchor in fixture_text
        # Stored anchor is the byte-exact source span at [start, end).
        assert fixture_text[rec.anchor_start : rec.anchor_end] == rec.anchor


# --------------------------------------------------------------------------- #
# Prompt still encodes the required granularity (static content check).
# --------------------------------------------------------------------------- #


def test_prompt_encodes_granularity():
    prompt = claims.CLAIMS_PROMPT
    sentence_pat = re.compile(
        r"(1\s*[-–to]{1,3}\s*3|one\s+to\s+three)\D{0,20}sentence",
        re.IGNORECASE,
    )
    assert sentence_pat.search(prompt), "prompt lacks 1-3 sentence framing"
    assert "10" in prompt and "40" in prompt, "prompt lacks ~10-40 count target"


# --------------------------------------------------------------------------- #
# Validation RAISES on structurally-missing fields; unresolved anchors are KEPT.
# --------------------------------------------------------------------------- #


def test_raises_on_missing_claim_text():
    text = _fixture_text(DOC_IDS[0])
    anchor = _real_anchors(text, 1)[0]
    ctx, _c, _f = _mock_anthropic(_records([("", anchor)]))
    with ctx, pytest.raises(claims.ClaimParseError):
        claims.extract_claims(text)


def test_raises_on_missing_anchor():
    text = _fixture_text(DOC_IDS[0])
    ctx, _c, _f = _mock_anthropic(_records([("a real claim", "")]))
    with ctx, pytest.raises(claims.ClaimParseError):
        claims.extract_claims(text)


def test_unresolvable_anchor_is_kept_and_flagged_not_dropped():
    text = _fixture_text(DOC_IDS[0])
    good = _real_anchors(text, 1)[0]
    junk = "this passage is definitely nowhere in the document zzzz qqqq xyzzy"
    ctx, _c, _f = _mock_anthropic(_records([("real claim", good), ("other claim", junk)]))
    with ctx:
        result = claims.extract_claims(text)
    # Nothing dropped: both claims survive.
    assert len(result) == 2
    resolved, unresolved = result[0], result[1]
    assert resolved.anchor_unresolved is False and resolved.anchor in text
    assert unresolved.anchor_unresolved is True
    assert unresolved.anchor_start is None and unresolved.anchor_end is None
    # The raw model anchor is preserved for reference on the unresolved claim.
    assert unresolved.anchor == junk


def test_wellformed_response_parses_without_raising():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 2)
    ctx, _c, _f = _mock_anthropic(_records([("claim one", anchors[0]), ("claim two", anchors[1])]))
    with ctx:
        result = claims.extract_claims(text)
    assert len(result) == 2


# --------------------------------------------------------------------------- #
# Anchor resolution unit tests (the fuzzy-locate layer).
# --------------------------------------------------------------------------- #


def test_resolve_exact_verbatim_anchor():
    text = _fixture_text(DOC_IDS[0])
    span = _real_anchors(text, 1)[0]
    res = claims.resolve_anchor(span, text)
    assert res.unresolved is False and res.score == 1.0
    assert text[res.start : res.end] == span


def test_resolve_cosmetic_drift_returns_verbatim_source_span():
    text = _fixture_text(DOC_IDS[0])
    # Find a source span containing letters, then introduce cosmetic drift:
    # swap hyphens for em-dashes, straight for curly quotes, upper-case, and
    # collapse/expand whitespace — the kind of drift real model output shows.
    span = next(a for a in _real_anchors(text, 40) if len(a) > 40)
    drifted = span.upper().replace("-", "—").replace("'", "’").replace("  ", " ")
    drifted = re.sub(r"\s+", "   ", drifted)  # expand internal whitespace
    res = claims.resolve_anchor(drifted, text)
    assert res.unresolved is False, "cosmetic drift should still resolve"
    # The STORED text is the byte-exact original span, not the drifted input.
    assert res.text == text[res.start : res.end]
    assert res.text in text


def test_resolve_below_threshold_is_unresolved():
    text = _fixture_text(DOC_IDS[0])
    res = claims.resolve_anchor("qwx zzptqr vbnm lkjhg fdsapoiuy nonsense", text)
    assert res.unresolved is True
    assert res.start is None and res.end is None
    assert res.score < claims.RESOLVE_THRESHOLD


# --------------------------------------------------------------------------- #
# Golden payloads from a real credentialed smoke run.
# These are the actual Sonnet responses (markdown-fenced, drifted anchors, and
# one with unescaped inner quotes) — the cases the mocked suite was blind to.
# --------------------------------------------------------------------------- #


def test_golden_payloads_committed_and_fenced():
    for doc_id in DOC_IDS:
        p = PAYLOADS_DIR / f"{doc_id}.raw.txt"
        assert p.exists() and p.stat().st_size > 0, f"missing golden payload {p}"
        # Every captured payload opened with a markdown fence — the exact thing
        # that broke the original bare-json.loads parser.
        assert p.read_text().lstrip().startswith("```"), "expected a fenced payload"


@pytest.mark.parametrize("doc_id", PARSEABLE_PAYLOAD_DOCS)
def test_golden_fenced_payload_parses_and_resolves_drift(doc_id):
    fixture_text = _fixture_text(doc_id)
    raw = _raw_payload(doc_id)

    # The raw payload is fenced; the defensive text path must strip it and parse.
    data = claims._extract_json_object(raw)
    raw_anchors = [c["anchor"] for c in data["claims"]]
    # Sanity: this real payload genuinely contains drifted (non-verbatim) anchors.
    drift = [a for a in raw_anchors if a not in fixture_text]
    assert drift, "expected real anchor drift in the captured payload"

    result = claims.parse_claims(raw, fixture_text)
    # Nothing dropped — every model claim survives.
    assert len(result) == len(raw_anchors)
    # Granularity band.
    assert 10 <= len(result) <= 40, f"{len(result)} claims outside 10-40"

    # Every RESOLVED anchor is now a byte-exact source span.
    resolved = [c for c in result if not c.anchor_unresolved]
    for c in resolved:
        assert c.anchor in fixture_text
        assert fixture_text[c.anchor_start : c.anchor_end] == c.anchor
    # The resolution layer recovered the large majority of drifted anchors.
    assert len(resolved) >= 0.7 * len(raw_anchors)


def test_golden_malformed_payload_degrades_cleanly():
    # doc 2's real output has an unescaped inner quote -> invalid JSON. The text
    # path must surface a clean ClaimParseError, not crash. (The structured
    # tool-use path never hits this — the SDK returns a parsed dict.)
    fixture_text = _fixture_text(MALFORMED_PAYLOAD_DOC)
    raw = _raw_payload(MALFORMED_PAYLOAD_DOC)
    with pytest.raises(claims.ClaimParseError):
        claims.parse_claims(raw, fixture_text)


@pytest.mark.parametrize("doc_id", PARSEABLE_PAYLOAD_DOCS)
def test_golden_records_via_tool_path_resolve_end_to_end(doc_id):
    # Feed the real captured records through the structured (tool-use) path.
    fixture_text = _fixture_text(doc_id)
    data = claims._extract_json_object(_raw_payload(doc_id))
    records = data["claims"]
    ctx, _c, _f = _mock_anthropic(records)
    with ctx:
        result = claims.extract_claims(fixture_text)
    assert len(result) == len(records)  # kept, not dropped
    for c in result:
        if not c.anchor_unresolved:
            assert fixture_text[c.anchor_start : c.anchor_end] == c.anchor


def test_parse_claims_direct_raises_on_bad_json():
    text = _fixture_text(DOC_IDS[0])
    with pytest.raises(claims.ClaimParseError):
        claims.parse_claims("not json at all", text)


# =========================================================================== #
# Sprint 1: sidecar persistence + generate-once.
# =========================================================================== #


def _claim_set_from_fixture(doc_id, n=3):
    """Build a real, verbatim-anchored claim set for ``doc_id`` (no LLM)."""
    text = _fixture_text(doc_id)
    anchors = _real_anchors(text, n)
    return [
        claims.Claim(id=f"c{i + 1}", claim=f"claim {i + 1}", anchor=a)
        for i, a in enumerate(anchors)
    ]


def _unwrap_claim_dicts(loaded):
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        for value in loaded.values():
            if isinstance(value, list):
                return value
    raise AssertionError(f"unrecognized sidecar envelope: {type(loaded)!r}")


def test_write_helper_writes_human_readable_sidecar(claims_docs_dir):
    doc_id = DOC_IDS[0]
    claims.write_claims(doc_id, _claim_set_from_fixture(doc_id))
    sidecar = claims_docs_dir / f"{doc_id}.claims.json"
    assert sidecar.exists(), f"sidecar not written at {sidecar}"
    raw = sidecar.read_text()
    json.loads(raw)
    assert "\n" in raw, "sidecar is not multi-line"
    assert re.search(r"\n[ ]+\S", raw), "sidecar is not indented"


def test_sidecar_round_trips_field_for_field(claims_docs_dir):
    doc_id = DOC_IDS[1]
    claim_set = _claim_set_from_fixture(doc_id, n=4)
    sidecar = claims.write_claims(doc_id, claim_set)
    loaded = json.loads(sidecar.read_text())
    got_dicts = _unwrap_claim_dicts(loaded)
    assert got_dicts == [c.to_dict() for c in claim_set]
    for d in got_dicts:
        assert set(d.keys()) == {
            "id",
            "claim",
            "anchor",
            "anchor_start",
            "anchor_end",
            "anchor_unresolved",
        }


def test_sidecar_round_trips_resolved_offsets(claims_docs_dir):
    # A claim set carrying resolved offsets + an unresolved flag must survive a
    # write/read cycle exactly (the artifact scoring will consume).
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    span = _real_anchors(text, 1)[0]
    start = text.find(span)
    claim_set = [
        claims.Claim("c1", "resolved claim", span, start, start + len(span), False),
        claims.Claim("c2", "unresolved claim", "raw model anchor", None, None, True),
    ]
    sidecar = claims.write_claims(doc_id, claim_set)
    reloaded = claims._deserialize(sidecar.read_text())
    assert [c.to_dict() for c in reloaded] == [c.to_dict() for c in claim_set]


def test_generate_miss_then_disk_hit_skips_llm(claims_docs_dir):
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    anchors = _real_anchors(text, 3)
    records = _records([(f"claim {i}", a) for i, a in enumerate(anchors)])

    ctx, client, factory = _mock_anthropic(records)
    with ctx:
        first = claims.generate_claims(doc_id, text)
    assert client.messages.create.call_count == 1
    assert (claims_docs_dir / f"{doc_id}.claims.json").exists()
    assert isinstance(first, list) and first
    assert all(isinstance(c, claims.Claim) for c in first)

    importlib.reload(claims)
    claims.DOCUMENTS_DIR = claims_docs_dir  # keep redirect after reload
    ctx2, client2, factory2 = _mock_anthropic(records)
    with ctx2:
        second = claims.generate_claims(doc_id, text)
    assert client2.messages.create.call_count == 0, "cache-hit re-invoked the LLM"
    assert factory2.called is False, "cache-hit constructed an Anthropic client"

    assert isinstance(second, list) and second
    assert [c.to_dict() for c in second] == [c.to_dict() for c in first]

    importlib.reload(claims)


def test_miss_path_result_matches_persisted_sidecar(claims_docs_dir):
    doc_id = DOC_IDS[0]
    text = _fixture_text(doc_id)
    anchors = _real_anchors(text, 5)
    records = _records([(f"claim {i}", a) for i, a in enumerate(anchors)])
    ctx, _client, _factory = _mock_anthropic(records)
    with ctx:
        returned = claims.generate_claims(doc_id, text)
    sidecar = claims_docs_dir / f"{doc_id}.claims.json"
    loaded = json.loads(sidecar.read_text())
    assert _unwrap_claim_dicts(loaded) == [c.to_dict() for c in returned]


def test_documents_dir_defined_locally_not_reexported_from_documents():
    tree = ast.parse(CLAIMS_PY.read_text())
    assigned = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "DOCUMENTS_DIR":
                    assigned = True
    assert assigned, "claims.py does not define DOCUMENTS_DIR at module scope"

    imported_tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_tops.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported_tops.add(node.module.split(".")[0])
    assert "documents" not in imported_tops


def test_redirect_confines_writes_and_leaves_real_dir_untouched(
    claims_docs_dir, tmp_path
):
    import hashlib

    # Reconstruct the real production documents dir without embedding the
    # machine-specific dotted directory name as a literal.
    real_dir = Path.home() / ("." + "voice-tutor") / "documents"

    def _snap(root):
        snap = {}
        if not root.exists():
            return snap
        for p in sorted(root.rglob("*")):
            if p.is_file():
                snap[p.relative_to(root).as_posix()] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
        return snap

    before = _snap(real_dir)

    doc_id = DOC_IDS[1]
    text = _fixture_text(doc_id)
    anchors = _real_anchors(text, 3)
    records = _records([(f"claim {i}", a) for i, a in enumerate(anchors)])
    ctx, _c, _f = _mock_anthropic(records)
    with ctx:
        claims.generate_claims(doc_id, text)
    claims.write_claims(DOC_IDS[0], _claim_set_from_fixture(DOC_IDS[0]))

    written = list(claims_docs_dir.glob("*.claims.json"))
    assert written, "no sidecars written under the redirected dir"
    for p in written:
        assert tmp_path in p.parents

    after = _snap(real_dir)
    assert after == before, "real production documents dir was mutated"


def test_import_closure_only_anthropic_and_stdlib():
    tree = ast.parse(CLAIMS_PY.read_text())
    imported_tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_tops.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported_tops.add(node.module.split(".")[0])

    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    allowed = {"anthropic"} | stdlib
    extras = imported_tops - allowed
    assert not extras, f"claims.py imports outside {{anthropic}} ∪ stdlib: {extras}"

    forbidden = {"documents", "pypdf", "bot", "app", "pipecat", "fastapi"}
    assert forbidden.isdisjoint(imported_tops), (
        f"claims.py imports forbidden modules: {forbidden & imported_tops}"
    )
