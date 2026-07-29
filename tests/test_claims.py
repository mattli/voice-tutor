"""Hermetic tests for the claim-extraction core (claims.py).

The verifier runs EXACTLY:
    uv run --with pytest pytest tests/test_claims.py -q
in a fresh worktree with no local .venv, so ALL hermetic tests for this work
live in THIS single file.

Determinism strategy: the Anthropic LLM call is MOCKED. ``claims.extract_claims``
constructs ``anthropic.Anthropic()`` lazily and calls ``client.messages.stream``
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
import unicodedata
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
# Sprint-1 sidecar tests are single-user; this is the user_id threaded through
# write_claims/load_claims/generate_claims/load_fresh_claims call sites.
USER_ID = "matt"

CLAIMS_PY = Path(__file__).parent.parent / "claims.py"


def _fixture_text(doc_id: str) -> str:
    return (FIXTURES_DIR / f"{doc_id}.txt").read_text()


def _raw_payload(doc_id: str) -> str:
    return (PAYLOADS_DIR / f"{doc_id}.raw.txt").read_text()


def _records(pairs):
    """[(claim, anchor), ...] -> [{"claim":..., "anchor":...}, ...]."""
    return [{"claim": c, "anchor": a} for c, a in pairs]


def _tool_message(records, stop_reason="tool_use"):
    """Build an Anthropic SDK-shaped final message carrying a record_claims call.

    Mirrors the real shape: message.content holds a ``tool_use`` block whose
    ``.input`` is the parsed ``{"claims": [...]}`` dict.
    """
    block = SimpleNamespace(
        type="tool_use",
        name="record_claims",
        id="toolu_stub",
        input={"claims": records},
    )
    return SimpleNamespace(content=[block], stop_reason=stop_reason)


class _FakeStream:
    """Context-manager stand-in for ``client.messages.stream(...)``."""

    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


def _mock_anthropic(records, stop_reason="tool_use"):
    """Patch ``anthropic.Anthropic`` so messages.stream yields a tool message.

    The real client is never built. Returns (patch_ctx, client, factory).
    """
    client = MagicMock()
    client.messages.stream.return_value = _FakeStream(_tool_message(records, stop_reason))
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
    _, kwargs = client.messages.stream.call_args
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["max_tokens"] == 16000  # headroom for dense docs (was 8000)
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
    client.messages.stream.return_value = _FakeStream(
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="no tool here")],
            stop_reason="end_turn",
        )
    )
    factory = MagicMock(return_value=client)
    with patch.object(claims.anthropic, "Anthropic", factory):
        with pytest.raises(claims.ClaimParseError):
            claims.extract_claims(text)


def test_truncated_response_raises_named_error_and_does_not_retry():
    # A max_tokens stop reason means the tool JSON was cut off. extract_claims
    # must raise the NAMED ClaimExtractionTruncated and NOT retry (deterministic).
    text = _fixture_text(DOC_IDS[0])
    ctx, client, _f = _mock_anthropic(records=[], stop_reason="max_tokens")
    with ctx:
        with pytest.raises(claims.ClaimExtractionTruncated):
            claims.extract_claims(text)
    assert client.messages.stream.call_count == 1, "truncation must not be retried"


def test_empty_claim_list_triggers_retry_then_succeeds():
    # An empty claim list is a retryable degenerate response: retry, then succeed.
    text = _fixture_text(DOC_IDS[0])
    good = _records([(f"claim {i}", a) for i, a in enumerate(_real_anchors(text, 3))])
    client = MagicMock()
    client.messages.stream.side_effect = [
        _FakeStream(_tool_message([], stop_reason="tool_use")),  # empty -> retry
        _FakeStream(_tool_message(good, stop_reason="tool_use")),  # then valid
    ]
    factory = MagicMock(return_value=client)
    with patch.object(claims.anthropic, "Anthropic", factory):
        result = claims.extract_claims(text)
    assert len(result) == 3
    assert client.messages.stream.call_count == 2, "empty list should have retried once"


def test_retry_gives_up_after_max_attempts():
    # Persistent empty responses exhaust the bounded retry and raise (no infinite loop).
    text = _fixture_text(DOC_IDS[0])
    ctx, client, _f = _mock_anthropic(records=[])  # always empty
    with ctx:
        with pytest.raises(claims.ClaimParseError):
            claims.extract_claims(text)
    assert client.messages.stream.call_count == claims.MAX_EXTRACT_ATTEMPTS


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
    # Count guidance is 10-50, density-driven (not a hard ceiling).
    assert "10" in prompt and "50" in prompt, "prompt lacks 10-50 count guidance"
    # Consolidation instruction: repeated facts -> one claim.
    assert "one claim only" in prompt.lower(), "prompt lacks consolidation rule"


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
# Anchor resolution unit tests — PROVABLE substring tiers only (no fuzzy).
# --------------------------------------------------------------------------- #


def test_resolve_exact_verbatim_anchor():
    text = _fixture_text(DOC_IDS[0])
    span = _real_anchors(text, 1)[0]
    res = claims.resolve_anchor(span, text)
    assert res.unresolved is False and res.tier == "exact"
    assert text[res.start : res.end] == span


def test_resolve_cosmetic_drift_resolves_normalized_to_verbatim_span():
    text = _fixture_text(DOC_IDS[0])
    # Cosmetic drift only — em-dashes, curly quotes, case, whitespace — which the
    # normalized tier folds away; the STORED span is still the byte-exact source.
    span = next(a for a in _real_anchors(text, 40) if len(a) > 40)
    drifted = span.upper().replace("-", "—").replace("'", "’").replace("  ", " ")
    drifted = re.sub(r"\s+", "   ", drifted)  # expand internal whitespace
    res = claims.resolve_anchor(drifted, text)
    assert res.unresolved is False and res.tier == "normalized"
    assert res.text == text[res.start : res.end]
    assert res.text in text


def test_resolve_garbage_anchor_is_unresolved():
    text = _fixture_text(DOC_IDS[0])
    res = claims.resolve_anchor("qwx zzptqr vbnm lkjhg fdsapoiuy nonsense", text)
    assert res.unresolved is True and res.tier == "unresolved"
    assert res.start is None and res.end is None


# --------------------------------------------------------------------------- #
# Conservative-resolution walls. The fuzzy span-guesser was REMOVED because
# every heuristic boundary it drew had an inverse silent-corruption bug (chop <->
# bleed <-> over-trim). Genuine content drift must now land UNRESOLVED — claim
# kept, raw anchor preserved, offsets null — never a guessed/wrong span. These
# pin all four historical failure inputs to that safe outcome.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "doc,anchor,label",
    [
        # chop wall: one-char deletion typo inside a word.
        (
            "Intro. The system caches the rubric per document. Later.",
            "the system caches the rubric per documnt",
            "deletion-typo",
        ),
        # bleed wall: stray trailing char coincidentally matching the next sentence.
        (
            "Photosynthesis converts light. The leaf absorbs energy.",
            "Photosynthesis converts lightt",
            "trailing-bleed",
        ),
        # dropped-lead-phrase: real leading token abutting an interior insertion.
        (
            "Item one.    Item two follows immediately after a long run here.",
            "Item one is great.    Item two follows immediately after a long run",
            "dropped-lead",
        ),
        # hallucinated-name: model substitutes a wrong name mid-anchor.
        (
            "According to the filing, CEO Marguerite Okonkwo-Basile will step down soon.",
            "CEO Reynard Thistlewick will step down soon.",
            "hallucinated-name",
        ),
    ],
)
def test_content_drift_is_unresolved_never_a_guessed_span(doc, anchor, label):
    res = claims.resolve_anchor(anchor, doc)
    assert res.unresolved is True, f"{label}: expected unresolved, got {res.text!r}"
    assert res.tier == "unresolved"
    assert res.start is None and res.end is None
    # The raw model anchor is preserved (NFC-normalized), not a truncated span.
    assert res.text == unicodedata.normalize("NFC", anchor)


def test_nfd_source_nfc_anchor_resolves_to_full_span():
    # Finding #2: a decomposed (NFD) source and a composed (NFC) anchor of the
    # SAME text must align — no dropped leading/trailing letter, not unresolved.
    phrase = "André plays café jazz nightly"
    doc = unicodedata.normalize("NFD", "Intro. " + phrase + " The end.")  # decomposed
    anchor = unicodedata.normalize("NFC", phrase)  # composed (model output form)

    res = claims.resolve_anchor(anchor, doc)
    assert res.unresolved is False, "NFD/NFC skew wrongly flagged unresolved"
    nfc_doc = unicodedata.normalize("NFC", doc)
    # Offsets index the NFC document; stored text is the full composed phrase.
    assert nfc_doc[res.start : res.end] == res.text
    assert res.text == phrase  # leading 'A' and trailing 'y' intact


# --------------------------------------------------------------------------- #
# Golden payloads from a real credentialed smoke run — the actual Sonnet
# responses (markdown-fenced, real drifted anchors). Fed through the structured
# tool-use path (the only production path), exercising resolution on real drift.
# --------------------------------------------------------------------------- #


def _records_from_payload(doc_id: str):
    """Strip the markdown fence off a captured payload and return its records."""
    raw = _raw_payload(doc_id).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    inner = m.group(1).strip() if m else raw
    return json.loads(inner)["claims"]


def test_golden_payloads_committed_and_fenced():
    for doc_id in DOC_IDS:
        p = PAYLOADS_DIR / f"{doc_id}.raw.txt"
        assert p.exists() and p.stat().st_size > 0, f"missing golden payload {p}"
        # Every captured payload opened with a markdown fence — the exact thing
        # that broke the original bare-json.loads parser.
        assert p.read_text().lstrip().startswith("```"), "expected a fenced payload"


@pytest.mark.parametrize("doc_id", PARSEABLE_PAYLOAD_DOCS)
def test_golden_records_via_tool_path_resolve_drift(doc_id):
    fixture_text = _fixture_text(doc_id)
    nfc = unicodedata.normalize("NFC", fixture_text)
    records = _records_from_payload(doc_id)

    # Sanity: this real payload genuinely contains drifted (non-verbatim) anchors.
    drift = [r for r in records if r["anchor"] not in fixture_text]
    assert drift, "expected real anchor drift in the captured payload"

    ctx, _c, _f = _mock_anthropic(records)
    with ctx:
        result = claims.extract_claims(fixture_text)

    assert len(result) == len(records)  # nothing dropped
    assert 10 <= len(result) <= 50, f"{len(result)} claims outside 10-50 guidance"

    for c in result:
        if c.anchor_unresolved:
            # Unresolved: safe fallback — null offsets, resolution tier recorded.
            assert c.anchor_start is None and c.anchor_end is None
            assert c.resolution == "unresolved"
        else:
            # Resolved: byte-exact source span, provable tier.
            assert nfc[c.anchor_start : c.anchor_end] == c.anchor
            assert c.resolution in ("exact", "normalized")
    # On real payloads the vast majority still resolve via exact/normalized; the
    # exact rate is a quality signal, not a hard gate, so we don't assert a floor.


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
    claims.write_claims(USER_ID, doc_id, _claim_set_from_fixture(doc_id))
    sidecar = claims_docs_dir / USER_ID / f"{doc_id}.claims.json"
    assert sidecar.exists(), f"sidecar not written at {sidecar}"
    raw = sidecar.read_text()
    json.loads(raw)
    assert "\n" in raw, "sidecar is not multi-line"
    assert re.search(r"\n[ ]+\S", raw), "sidecar is not indented"


def test_sidecar_round_trips_field_for_field(claims_docs_dir):
    doc_id = DOC_IDS[1]
    claim_set = _claim_set_from_fixture(doc_id, n=4)
    sidecar = claims.write_claims(USER_ID, doc_id, claim_set)
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
            "resolution",
        }


def test_sidecar_round_trips_resolved_offsets(claims_docs_dir):
    # A claim set carrying resolved offsets + an unresolved flag must survive a
    # write/read cycle exactly (the artifact scoring will consume).
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    span = _real_anchors(text, 1)[0]
    start = text.find(span)
    claim_set = [
        claims.Claim("c1", "resolved claim", span, start, start + len(span), False, "exact"),
        claims.Claim("c2", "unresolved claim", "raw model anchor", None, None, True, "unresolved"),
    ]
    sidecar = claims.write_claims(USER_ID, doc_id, claim_set)
    reloaded = claims._deserialize(sidecar.read_text())
    assert [c.to_dict() for c in reloaded] == [c.to_dict() for c in claim_set]


def test_generate_miss_then_disk_hit_skips_llm(claims_docs_dir):
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    anchors = _real_anchors(text, 3)
    records = _records([(f"claim {i}", a) for i, a in enumerate(anchors)])

    ctx, client, factory = _mock_anthropic(records)
    with ctx:
        first = claims.generate_claims(USER_ID, doc_id, text)
    assert client.messages.stream.call_count == 1
    assert (claims_docs_dir / USER_ID / f"{doc_id}.claims.json").exists()
    assert isinstance(first, list) and first
    assert all(isinstance(c, claims.Claim) for c in first)

    importlib.reload(claims)
    claims.DOCUMENTS_DIR = claims_docs_dir  # keep redirect after reload
    ctx2, client2, factory2 = _mock_anthropic(records)
    with ctx2:
        second = claims.generate_claims(USER_ID, doc_id, text)
    assert client2.messages.stream.call_count == 0, "cache-hit re-invoked the LLM"
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
        returned = claims.generate_claims(USER_ID, doc_id, text)
    sidecar = claims_docs_dir / USER_ID / f"{doc_id}.claims.json"
    loaded = json.loads(sidecar.read_text())
    assert _unwrap_claim_dicts(loaded) == [c.to_dict() for c in returned]


# --------------------------------------------------------------------------- #
# Cache integrity: sidecar carries source_hash; get-or-create regenerates when
# the served document's hash disagrees with the cached one.
# --------------------------------------------------------------------------- #


def test_generate_stamps_source_hash_and_hits_on_match(claims_docs_dir):
    doc_id = DOC_IDS[1]
    text = _fixture_text(doc_id)
    records = _records(
        [(f"claim {i}", a) for i, a in enumerate(_real_anchors(text, 3))]
    )

    ctx, client, _f = _mock_anthropic(records)
    with ctx:
        first = claims.generate_claims(USER_ID, doc_id, text)
    assert client.messages.stream.call_count == 1

    # The sidecar is stamped with the source hash.
    data = json.loads((claims_docs_dir / USER_ID / f"{doc_id}.claims.json").read_text())
    assert data["source_hash"] == claims._hash_source(text)

    # Same document -> hash matches -> cache hit, no LLM, no client built.
    ctx2, client2, factory2 = _mock_anthropic(records)
    with ctx2:
        second = claims.generate_claims(USER_ID, doc_id, text)
    assert client2.messages.stream.call_count == 0, "matching hash re-invoked LLM"
    assert factory2.called is False
    assert [c.to_dict() for c in second] == [c.to_dict() for c in first]


def test_generate_regenerates_on_source_hash_mismatch(claims_docs_dir):
    doc_id = DOC_IDS[0]
    text = _fixture_text(doc_id)
    records = _records(
        [(f"claim {i}", a) for i, a in enumerate(_real_anchors(text, 4))]
    )

    ctx, client, _f = _mock_anthropic(records)
    with ctx:
        claims.generate_claims(USER_ID, doc_id, text)
    assert client.messages.stream.call_count == 1

    # The served document drifted (e.g. the vault page was edited after caching)
    # -> its hash no longer matches the stamped one -> MUST regenerate.
    changed = text + "\n\n## New section added upstream\n\nA fresh fact.\n"
    assert claims._hash_source(changed) != claims._hash_source(text)
    ctx2, client2, _f2 = _mock_anthropic(records)
    with ctx2:
        claims.generate_claims(USER_ID, doc_id, changed)
    assert client2.messages.stream.call_count == 1, "stale cache was not regenerated"

    # The rewritten sidecar now carries the CURRENT document's hash.
    data = json.loads((claims_docs_dir / USER_ID / f"{doc_id}.claims.json").read_text())
    assert data["source_hash"] == claims._hash_source(changed)


def test_generate_regenerates_when_sidecar_has_no_hash(claims_docs_dir):
    # A legacy/hand-written sidecar with no source_hash can't be verified, so
    # get-or-create must regenerate rather than serve an unverifiable rubric.
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    stale = [claims.Claim("c1", "old claim", _real_anchors(text, 1)[0])]
    claims.write_claims(USER_ID, doc_id, stale)  # no source_hash passed
    assert claims._cached_source_hash(USER_ID, doc_id) is None

    records = _records(
        [(f"claim {i}", a) for i, a in enumerate(_real_anchors(text, 3))]
    )
    ctx, client, _f = _mock_anthropic(records)
    with ctx:
        claims.generate_claims(USER_ID, doc_id, text)
    assert client.messages.stream.call_count == 1, "unverifiable cache not rebuilt"
    assert claims._cached_source_hash(USER_ID, doc_id) == claims._hash_source(text)


def _seed_fresh_sidecar(docs_dir, user_id, doc_id, text):
    """Write a fresh sidecar (source_hash matching ``text``) for ``user_id``."""
    user_dir = docs_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        claims._HASH_KEY: claims._hash_source(text),
        claims._CLAIMS_KEY: [c.to_dict() for c in _claim_set_from_fixture(DOC_IDS[0], n=1)],
    }
    (user_dir / f"{doc_id}.claims.json").write_text(json.dumps(envelope, indent=2))


def test_load_fresh_claims_is_user_scoped(claims_docs_dir):
    text = "Some doc text."
    # Seed a fresh sidecar for matt only (reuse the file's existing sidecar-writing
    # helper / source_hash computation).
    _seed_fresh_sidecar(claims_docs_dir, user_id="matt", doc_id="D", text=text)
    assert claims.load_fresh_claims("matt", "D", text) is not None
    # Mirror image: sarah has no sidecar for D -> None (degrade to plain study).
    assert claims.load_fresh_claims("sarah", "D", text) is None


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
        claims.generate_claims(USER_ID, doc_id, text)
    claims.write_claims(USER_ID, DOC_IDS[0], _claim_set_from_fixture(DOC_IDS[0]))

    written = list(claims_docs_dir.glob("**/*.claims.json"))
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


# =========================================================================== #
# Sprint (shared claims sidecars): a document under documents/_shared/ has its
# claim sidecar extracted ONCE into _shared/ and served to every user, via a
# resolution fallback (user namespace first, then _shared/) that mirrors
# documents.load_document. No signature changes; the sidecar namespace is
# derived INSIDE the helpers from the DOCUMENT (<doc_id>.txt) presence.
# =========================================================================== #

SHARED = "_shared"


def _seed_doc_txt(docs_dir, namespace, doc_id, text):
    """Seed a ``<doc_id>.txt`` DOCUMENT under ``docs_dir/namespace`` (per-user or
    ``_shared``). The namespace-resolution keys on this file, not the sidecar."""
    d = docs_dir / namespace
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.txt").write_text(text)


def _sidecar_path(docs_dir, namespace, doc_id):
    return docs_dir / namespace / f"{doc_id}.claims.json"


def _all_sidecars(docs_dir):
    return sorted(docs_dir.glob("**/*.claims.json"))


def _claim_objs_for(doc_id, n=3):
    """Real verbatim-anchored Claim list for ``doc_id`` (drives the extract mock)."""
    return _claim_set_from_fixture(doc_id, n=n)


# --------------------------------------------------------------------------- #
# c1: EXACT signatures unchanged (no new parameters of any kind).
# --------------------------------------------------------------------------- #


def test_public_signatures_are_exactly_unchanged():
    import inspect

    gen = list(inspect.signature(claims.generate_claims).parameters)
    fresh = list(inspect.signature(claims.load_fresh_claims).parameters)
    assert gen == ["user_id", "doc_id", "document_text"]
    assert fresh == ["user_id", "doc_id", "document_text"]


def test_helpers_callable_positionally_three_args(claims_docs_dir):
    doc_id = DOC_IDS[0]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)
    with patch.object(claims, "extract_claims", return_value=_claim_objs_for(doc_id)):
        result = claims.generate_claims(USER_ID, doc_id, text)  # positional
    assert isinstance(result, list) and result
    # load_fresh_claims also callable positionally.
    assert claims.load_fresh_claims(USER_ID, doc_id, text) is not None


# --------------------------------------------------------------------------- #
# c2 / c3: placement keys on the DOCUMENT (.txt), bootstraps a _shared/ sidecar.
# --------------------------------------------------------------------------- #


def test_shared_doc_bootstraps_sidecar_into_shared_from_nothing(claims_docs_dir):
    # Seed ONLY the shared .txt — no sidecar anywhere. generate_claims must
    # create the sidecar in _shared/ (bootstrapped), none under the user dir.
    doc_id = DOC_IDS[0]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, SHARED, doc_id, text)
    assert not _all_sidecars(claims_docs_dir), "precondition: no sidecar yet"

    with patch.object(claims, "extract_claims", return_value=_claim_objs_for(doc_id)):
        claims.generate_claims(USER_ID, doc_id, text)

    assert _sidecar_path(claims_docs_dir, SHARED, doc_id).exists()
    assert not _sidecar_path(claims_docs_dir, USER_ID, doc_id).exists()


def test_per_user_doc_sidecar_lands_under_user_dir(claims_docs_dir):
    # Seed ONLY the user .txt: sidecar must land under the user dir, not _shared.
    doc_id = DOC_IDS[1]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    with patch.object(claims, "extract_claims", return_value=_claim_objs_for(doc_id)):
        claims.generate_claims(USER_ID, doc_id, text)

    assert _sidecar_path(claims_docs_dir, USER_ID, doc_id).exists()
    assert not _sidecar_path(claims_docs_dir, SHARED, doc_id).exists()


def test_claims_py_does_not_import_documents():
    # Static guard (mirrors the existing import-closure test): the shared
    # namespace is derived by checking <doc_id>.txt presence, NOT via documents.py.
    tree = ast.parse(CLAIMS_PY.read_text())
    imported_tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_tops.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported_tops.add(node.module.split(".")[0])
    assert "documents" not in imported_tops


# --------------------------------------------------------------------------- #
# c4: generate-once, serve-everyone (>= three users, single extraction).
# --------------------------------------------------------------------------- #


def test_shared_sidecar_extracted_once_served_to_all_users(claims_docs_dir):
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, SHARED, doc_id, text)

    extract = MagicMock(return_value=_claim_objs_for(doc_id))
    with patch.object(claims, "extract_claims", extract):
        claims.generate_claims("matt", doc_id, text)  # A generates once
        a = claims.load_fresh_claims("matt", doc_id, text)
        b = claims.load_fresh_claims("sarah", doc_id, text)  # no prior state
        c = claims.load_fresh_claims("wei", doc_id, text)    # no prior state

    # Extraction happened exactly once across all four calls.
    assert extract.call_count == 1
    # The single shared sidecar is what every user resolves to.
    shared = claims.load_claims(SHARED, doc_id)
    expected = [x.to_dict() for x in shared]
    # All three users see the identical shared claim set.
    for got in (a, b, c):
        assert got is not None
        assert [x.to_dict() for x in got] == expected
    # Exactly one sidecar exists, and it lives in _shared/.
    sidecars = _all_sidecars(claims_docs_dir)
    assert len(sidecars) == 1
    assert sidecars[0] == _sidecar_path(claims_docs_dir, SHARED, doc_id)


# --------------------------------------------------------------------------- #
# c5: shadowing — a user's own colliding doc wins over the shared one.
# --------------------------------------------------------------------------- #


def test_user_doc_shadows_colliding_shared_doc(claims_docs_dir):
    doc_id = DOC_IDS[0]
    user_text = _fixture_text(DOC_IDS[0])
    shared_text = _fixture_text(DOC_IDS[1])  # distinguishable content
    # Both namespaces hold the SAME doc_id but different documents + sidecars.
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, user_text)
    _seed_doc_txt(claims_docs_dir, SHARED, doc_id, shared_text)

    user_claims = [claims.Claim("c1", "USER claim", _real_anchors(user_text, 1)[0])]
    shared_claims = [claims.Claim("c1", "SHARED claim", _real_anchors(shared_text, 1)[0])]
    claims.write_claims(USER_ID, doc_id, user_claims, source_hash=claims._hash_source(user_text))
    claims.write_claims(SHARED, doc_id, shared_claims, source_hash=claims._hash_source(shared_text))

    got = claims.load_fresh_claims(USER_ID, doc_id, user_text)
    assert got is not None
    assert [c.claim for c in got] == ["USER claim"], "user doc must shadow shared"


# --------------------------------------------------------------------------- #
# c6: per-user isolation — no cross-user fallback for a purely per-user doc.
# --------------------------------------------------------------------------- #


def test_per_user_sidecar_never_leaks_to_another_user(claims_docs_dir):
    doc_id = DOC_IDS[1]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, "alice", doc_id, text)  # only alice owns it

    with patch.object(claims, "extract_claims", return_value=_claim_objs_for(doc_id)):
        claims.generate_claims("alice", doc_id, text)

    # Bob has neither a user doc nor a shared doc for doc_id -> miss, never alice's.
    assert claims.load_fresh_claims("bob", doc_id, text) is None
    assert not _sidecar_path(claims_docs_dir, "bob", doc_id).exists()
    assert not _sidecar_path(claims_docs_dir, SHARED, doc_id).exists()


# --------------------------------------------------------------------------- #
# c7: source_hash freshness unchanged; evaluated ONLY in the resolved namespace
# (no cross-namespace fall-through).
# --------------------------------------------------------------------------- #


def test_shared_fresh_hash_served_without_rewrite(claims_docs_dir):
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, SHARED, doc_id, text)
    with patch.object(claims, "extract_claims", return_value=_claim_objs_for(doc_id)):
        claims.generate_claims("matt", doc_id, text)
    sidecar = _sidecar_path(claims_docs_dir, SHARED, doc_id)
    before = sidecar.read_bytes()

    # Matching text -> cached shared claims returned without rewriting the sidecar.
    got = claims.load_fresh_claims("sarah", doc_id, text)
    assert got is not None
    assert sidecar.read_bytes() == before, "fresh read must not rewrite the sidecar"


def test_shared_stale_hash_is_a_miss(claims_docs_dir):
    doc_id = DOC_IDS[0]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, SHARED, doc_id, text)
    with patch.object(claims, "extract_claims", return_value=_claim_objs_for(doc_id)):
        claims.generate_claims("matt", doc_id, text)

    # The shared document text drifted -> stale sidecar -> miss (no stale claims).
    changed = text + "\n\n## Upstream edit\n\nA new fact.\n"
    assert claims._hash_source(changed) != claims._hash_source(text)
    assert claims.load_fresh_claims("sarah", doc_id, changed) is None


def test_no_cross_namespace_fallthrough_stale_user_over_fresh_shared(claims_docs_dir):
    # The document resolves PER-USER (user .txt present), so the _shared/ sidecar
    # must never be consulted: a stale per-user sidecar is a miss and does NOT
    # silently return the fresh shared claims of the same doc_id.
    doc_id = DOC_IDS[1]
    user_text = _fixture_text(DOC_IDS[1])
    shared_text = _fixture_text(DOC_IDS[2])
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, user_text)
    _seed_doc_txt(claims_docs_dir, SHARED, doc_id, shared_text)

    # STALE per-user sidecar (hash of some OTHER text), FRESH shared sidecar.
    stale = [claims.Claim("c1", "stale user", _real_anchors(user_text, 1)[0])]
    claims.write_claims(USER_ID, doc_id, stale, source_hash="deadbeef")
    fresh_shared = [claims.Claim("c1", "fresh shared", _real_anchors(shared_text, 1)[0])]
    claims.write_claims(SHARED, doc_id, fresh_shared, source_hash=claims._hash_source(user_text))

    # Doc resolves per-user; stale per-user sidecar -> miss; no shared fall-through.
    assert claims.load_fresh_claims(USER_ID, doc_id, user_text) is None


# --------------------------------------------------------------------------- #
# c8: placement centralized in generate_claims; primitives stay intact.
# --------------------------------------------------------------------------- #


def test_write_claims_primitive_writes_to_given_namespace(claims_docs_dir):
    # write_claims makes NO shared-vs-user decision: it writes to the namespace
    # it is GIVEN. Passing _shared writes to _shared; passing a user writes there.
    doc_id = DOC_IDS[0]
    claim_set = _claim_set_from_fixture(doc_id, n=2)
    claims.write_claims(SHARED, doc_id, claim_set)
    assert _sidecar_path(claims_docs_dir, SHARED, doc_id).exists()
    claims.write_claims("carol", doc_id, claim_set)
    assert _sidecar_path(claims_docs_dir, "carol", doc_id).exists()


def test_namespace_resolution_only_invoked_from_generate_claims():
    # Static AST guard: the placement helper _resolve_doc_namespace is called
    # only from generate_claims and load_fresh_claims (the read path) — NEVER
    # from write_claims (the low-level primitive stays namespace-agnostic).
    tree = ast.parse(CLAIMS_PY.read_text())
    funcs = {
        n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
    }
    assert "_resolve_doc_namespace" in funcs, "placement helper missing"
    assert "generate_claims" in funcs and "write_claims" in funcs

    def _calls(fn_node, name):
        return any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == name
            for c in ast.walk(fn_node)
        )

    # generate_claims performs the resolution.
    assert _calls(funcs["generate_claims"], "_resolve_doc_namespace")
    # write_claims must NOT resolve the namespace — it writes what it's given.
    assert not _calls(funcs["write_claims"], "_resolve_doc_namespace")

    # generate_claims is the SOLE function that both resolves the namespace AND
    # writes a sidecar (calls write_claims).
    resolvers_that_write = [
        name
        for name, node in funcs.items()
        if _calls(node, "_resolve_doc_namespace") and _calls(node, "write_claims")
    ]
    assert resolvers_that_write == ["generate_claims"], resolvers_that_write


# --------------------------------------------------------------------------- #
# c9: read-only shared happy path — no extraction on a pre-seeded fresh sidecar.
# --------------------------------------------------------------------------- #


def test_read_only_shared_happy_path_no_extraction(claims_docs_dir):
    doc_id = DOC_IDS[2]
    text = _fixture_text(doc_id)
    _seed_doc_txt(claims_docs_dir, SHARED, doc_id, text)
    # Pre-seed a fresh shared sidecar (matching source_hash), no per-user state.
    shared_claims = _claim_set_from_fixture(doc_id, n=2)
    claims.write_claims(SHARED, doc_id, shared_claims, source_hash=claims._hash_source(text))

    extract = MagicMock()
    with patch.object(claims, "extract_claims", extract):
        got = claims.load_fresh_claims("newuser", doc_id, text)
    assert got is not None
    assert extract.called is False, "load_fresh_claims must never extract"


# =========================================================================== #
# Sprint 0: input-bound tripwire at the shared warm seam (generate_claims).
#
# CLAIM_MAX_WORDS + ClaimInputTooLong: a doc whose word count STRICTLY EXCEEDS
# the bound is refused BEFORE any extraction call, at the single shared seam so
# every entry point (upload-triggered warm, prepare/picker) is protected. The
# guard sits AFTER the cache-hit check (a fresh sidecar short-circuits, length
# irrelevant) and BEFORE extract_claims. Word count is len(text.split()); every
# rejection emits an app-log line with the count. extract_claims is monkeypatched
# throughout — no live API call is ever made.
# =========================================================================== #

# Word count is len(text.split()); one-char tokens joined by single spaces give
# an EXACT, len(text.split())-faithful count for boundary fixtures.
def _text_of_words(n: int) -> str:
    return " ".join(["w"] * n)


def test_claim_max_words_constant_is_ten_thousand():
    # c1: a single named module-level constant with the value 10_000.
    assert claims.CLAIM_MAX_WORDS == 10_000
    assert isinstance(claims.CLAIM_MAX_WORDS, int)


def test_claim_input_too_long_is_independently_catchable():
    # c2: typed, catchable, distinct from ClaimParseError / ClaimExtractionTruncated.
    assert issubclass(claims.ClaimInputTooLong, Exception)
    assert not issubclass(claims.ClaimInputTooLong, claims.ClaimExtractionTruncated)
    assert not issubclass(claims.ClaimInputTooLong, claims.ClaimParseError)
    # Catchable by its own type without swallowing parse/truncation errors.
    with pytest.raises(claims.ClaimInputTooLong):
        raise claims.ClaimInputTooLong(12_345)


def test_over_bound_doc_raises_typed_condition_before_extraction(claims_docs_dir):
    # c3 + c4: over-bound doc with NO fresh sidecar -> typed rejection, and
    # extract_claims is NEVER invoked (rejection precedes extraction at the seam).
    doc_id = DOC_IDS[0]
    text = _text_of_words(claims.CLAIM_MAX_WORDS + 500)
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    extract = MagicMock()
    with patch.object(claims, "extract_claims", extract):
        with pytest.raises(claims.ClaimInputTooLong):
            claims.generate_claims(USER_ID, doc_id, text)
    assert extract.called is False, "extraction fired despite the input bound"


def test_fresh_cached_over_bound_doc_short_circuits_no_rejection(claims_docs_dir):
    # c5: a fresh valid sidecar (matching source_hash) for an OVER-bound doc is
    # served from cache without raising the bound and without extracting.
    doc_id = DOC_IDS[1]
    text = _text_of_words(claims.CLAIM_MAX_WORDS + 42)
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)
    # Null-offset claims deserialize as anchor_unresolved (see _records_to_claims),
    # so build the expectation the same way to compare the round-tripped record.
    cached = [
        claims.Claim(
            id="c1", claim="cached claim", anchor="w w w", anchor_unresolved=True
        )
    ]
    claims.write_claims(USER_ID, doc_id, cached, source_hash=claims._hash_source(text))

    extract = MagicMock()
    with patch.object(claims, "extract_claims", extract):
        got = claims.generate_claims(USER_ID, doc_id, text)
    assert extract.called is False, "cache hit must not extract"
    assert [c.to_dict() for c in got] == [c.to_dict() for c in cached]


def test_boundary_exactly_at_bound_proceeds_to_extraction(claims_docs_dir):
    # c6: word_count == CLAIM_MAX_WORDS is ALLOWED (limit value itself passes).
    doc_id = DOC_IDS[2]
    text = _text_of_words(claims.CLAIM_MAX_WORDS)
    assert len(text.split()) == claims.CLAIM_MAX_WORDS
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    extract = MagicMock(return_value=[claims.Claim(id="c1", claim="c", anchor="w")])
    with patch.object(claims, "extract_claims", extract):
        got = claims.generate_claims(USER_ID, doc_id, text)
    assert extract.called is True, "doc exactly at the bound must proceed to extract"
    assert got == extract.return_value


def test_boundary_one_over_bound_is_rejected(claims_docs_dir):
    # c6: word_count == CLAIM_MAX_WORDS + 1 is REJECTED (strictly-greater rule).
    doc_id = DOC_IDS[0]
    text = _text_of_words(claims.CLAIM_MAX_WORDS + 1)
    assert len(text.split()) == claims.CLAIM_MAX_WORDS + 1
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    extract = MagicMock()
    with patch.object(claims, "extract_claims", extract):
        with pytest.raises(claims.ClaimInputTooLong):
            claims.generate_claims(USER_ID, doc_id, text)
    assert extract.called is False


def test_rejection_message_names_count_limit_and_remedy(claims_docs_dir):
    # c7: message names the ACTUAL word count (distinct from the limit), the
    # limit, and split/excerpt guidance.
    doc_id = DOC_IDS[1]
    over = claims.CLAIM_MAX_WORDS + 1  # 10001, distinct from the limit 10000
    text = _text_of_words(over)
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    with patch.object(claims, "extract_claims", MagicMock()):
        with pytest.raises(claims.ClaimInputTooLong) as exc_info:
            claims.generate_claims(USER_ID, doc_id, text)
    msg = str(exc_info.value)
    assert str(over) in msg, f"actual word count {over} missing from: {msg!r}"
    assert str(claims.CLAIM_MAX_WORDS) in msg, f"limit missing from: {msg!r}"
    assert "split" in msg.lower() or "excerpt" in msg.lower(), (
        f"remedy guidance missing from: {msg!r}"
    )
    # The typed condition also exposes the count programmatically.
    assert exc_info.value.word_count == over


def test_rejection_emits_app_log_line_with_word_count(claims_docs_dir, caplog):
    # c8: a log record is emitted by the claims module logger on rejection, at
    # INFO/WARNING level, carrying the EXACT actual word count as an integer.
    import logging

    doc_id = DOC_IDS[2]
    over = claims.CLAIM_MAX_WORDS + 137
    text = _text_of_words(over)
    assert len(text.split()) == over
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    with caplog.at_level(logging.INFO, logger="claims"):
        with patch.object(claims, "extract_claims", MagicMock()):
            with pytest.raises(claims.ClaimInputTooLong):
                claims.generate_claims(USER_ID, doc_id, text)

    claim_records = [r for r in caplog.records if r.name == "claims"]
    assert claim_records, "no log record emitted by the claims module logger"
    rec = claim_records[-1]
    assert rec.levelno >= logging.INFO
    assert str(over) in rec.getMessage(), (
        f"rejection log line missing the word count {over}: {rec.getMessage()!r}"
    )


def test_under_bound_doc_proceeds_to_extraction_normally(claims_docs_dir):
    # c9: under-bound doc, no fresh sidecar -> extraction IS invoked, no rejection.
    doc_id = DOC_IDS[0]
    text = _text_of_words(500)  # well under CLAIM_MAX_WORDS
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    extract = MagicMock(return_value=[claims.Claim(id="c1", claim="c", anchor="w")])
    with patch.object(claims, "extract_claims", extract):
        got = claims.generate_claims(USER_ID, doc_id, text)
    assert extract.called is True, "under-bound doc must proceed to extraction"
    assert got == extract.return_value


def test_bound_fires_through_prepare_picker_warm_path(claims_docs_dir, docs_dir):
    # c10: drive an over-bound doc through the REAL prepare/picker warm path
    # (app._warm_claims -> claims.generate_claims). The best-effort warm catches
    # Exception and swallows the rejection, so no error surfaces to the caller;
    # extract_claims is NEVER invoked. (The rejection log line is asserted
    # separately, at the claims.py seam, in the caplog test above.)
    import asyncio

    app = pytest.importorskip("app")

    doc_id = DOC_IDS[1]
    over = claims.CLAIM_MAX_WORDS + 1000
    text = _text_of_words(over)
    # Seed the DOCUMENT in BOTH namespaces app._warm_claims reads: documents.py
    # resolves the .txt (docs_dir fixture) and claims.py resolves its own dir
    # (claims_docs_dir fixture).
    (docs_dir / USER_ID).mkdir(parents=True, exist_ok=True)
    (docs_dir / USER_ID / f"{doc_id}.txt").write_text(text)
    _seed_doc_txt(claims_docs_dir, USER_ID, doc_id, text)

    extract = MagicMock()
    with patch.object(claims, "extract_claims", extract):
        # Fire-and-forget best-effort warm: must NOT raise/500 to the caller.
        asyncio.run(app._warm_claims(USER_ID, doc_id))
    assert extract.called is False, "prepare path bypassed the input bound"
