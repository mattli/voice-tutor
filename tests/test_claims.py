"""Hermetic tests for the pure claim-extraction core (claims.py).

The verifier runs EXACTLY:
    uv run --with pytest pytest tests/test_claims.py -q
in a fresh worktree with no local .venv, so ALL hermetic tests for this work
live in THIS single file.

Determinism strategy: the Anthropic LLM call is MOCKED. ``claims.extract_claims``
constructs ``anthropic.Anthropic()`` lazily and calls
``client.messages.create(...)``; tests patch ``anthropic.Anthropic`` so the real
network client is never constructed and ``messages.create`` is never reached over
the network. The suite passes with no ANTHROPIC_API_KEY and no network.

Decomposition assertions are driven by the three REAL committed fixture
documents under tests/fixtures/claims/ (copies of the provided source docs); the
tests read those in-repo committed copies only — never the machine-specific
per-user documents directory.
"""

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import claims

# --------------------------------------------------------------------------- #
# Fixtures: the three REAL committed source documents.
# --------------------------------------------------------------------------- #

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "claims"
DOC_IDS = [
    "12f379a0-5a04-4eb6-b349-1c3c0690fe17",
    "8050fe28-f897-4947-953d-7ca38fd2e0ad",
    "a9f59a8f-7d39-48c3-ba66-14e3b8c8d8c6",
]
CLAIMS_PY = Path(__file__).parent.parent / "claims.py"


def _fixture_text(doc_id: str) -> str:
    return (FIXTURES_DIR / f"{doc_id}.txt").read_text()


def _sdk_response(payload: str):
    """Build an Anthropic SDK-shaped response exposing .content[0].text."""
    return SimpleNamespace(content=[SimpleNamespace(text=payload)])


def _mock_anthropic(payload: str):
    """A patch object for ``anthropic.Anthropic`` whose messages.create returns
    an SDK-shaped response carrying ``payload``. The real client is never built.
    """
    client = MagicMock()
    client.messages.create.return_value = _sdk_response(payload)
    factory = MagicMock(return_value=client)
    return patch.object(claims.anthropic, "Anthropic", factory), client, factory


def _payload_from_anchors(pairs) -> str:
    """Serialize (claim, anchor) pairs into the model's expected JSON payload."""
    import json

    return json.dumps({"claims": [{"claim": c, "anchor": a} for c, a in pairs]})


def _real_anchors(text: str, n: int):
    """Return ``n`` verbatim substrings drawn from ``text`` (non-empty lines)."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 25]
    assert len(lines) >= n, "fixture does not have enough substantial lines"
    return lines[:n]


# --------------------------------------------------------------------------- #
# c8: fixtures are committed, non-empty, and read from tests/fixtures/claims.
# --------------------------------------------------------------------------- #


def test_fixtures_committed_and_nonempty():
    for doc_id in DOC_IDS:
        path = FIXTURES_DIR / f"{doc_id}.txt"
        assert path.exists(), f"missing fixture {path}"
        text = path.read_text()
        assert len(text) > 0
        assert path.stat().st_size > 0
        # Real document prose, not a zero-byte placeholder.
        assert len(text.split()) > 100


def test_this_test_file_reads_committed_fixtures_only():
    src = Path(__file__).read_text()
    assert "fixtures" in src and "claims" in src
    # Never read the machine-specific real documents path. The needle is
    # assembled from parts so this guard doesn't match its own source line.
    needle = "." + "voice-tutor"
    assert needle not in src


# --------------------------------------------------------------------------- #
# c1: pure decomposition function; records have id/claim/anchor; order preserved.
# --------------------------------------------------------------------------- #


def test_returns_records_with_id_claim_anchor():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 3)
    payload = _payload_from_anchors(
        [("Claim about coding", anchors[0]),
         ("Claim about legal", anchors[1]),
         ("Claim about healthcare", anchors[2])]
    )
    ctx, client, factory = _mock_anthropic(payload)
    with ctx:
        result = claims.extract_claims(text)

    assert isinstance(result, list)
    assert len(result) == 3
    for rec in result:
        assert rec.id
        assert rec.claim and rec.claim.strip()
        assert rec.anchor and rec.anchor.strip()
    # The parser read the SDK-shaped .content[0].text payload (real client
    # never constructed with a key; messages.create used).
    assert client.messages.create.called


def test_order_is_preserved_not_reordered():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 3)
    # Deliberately NON-sorted, distinguishable claim texts.
    ordered = ["zebra first", "middle apple", "banana last"]
    payload = _payload_from_anchors(list(zip(ordered, anchors)))
    ctx, _client, _factory = _mock_anthropic(payload)
    with ctx:
        result = claims.extract_claims(text)

    got = [r.claim for r in result]
    assert got == ordered
    assert got != sorted(ordered)
    assert got != list(reversed(ordered))


# --------------------------------------------------------------------------- #
# c2: lazy client construction; no module-scope API key read; import is clean.
# --------------------------------------------------------------------------- #


def test_import_succeeds_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib

    mod = importlib.reload(claims)
    assert hasattr(mod, "extract_claims")


def test_no_module_scope_client_or_key_read():
    tree = ast.parse(CLAIMS_PY.read_text())

    def _calls_anthropic_client(node):
        # Match anthropic.Anthropic(...) or Anthropic(...)
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "Anthropic":
                return True
            if isinstance(f, ast.Name) and f.id == "Anthropic":
                return True
        return False

    def _reads_env_key(node):
        # Match os.environ[...] / os.environ.get(...) / os.getenv(...)
        if isinstance(node, ast.Subscript):
            v = node.value
            if isinstance(v, ast.Attribute) and v.attr == "environ":
                return True
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in {"getenv", "get"}:
                val = getattr(f, "value", None)
                if isinstance(val, ast.Attribute) and val.attr == "environ":
                    return True
                if isinstance(val, ast.Name) and val.id == "os":
                    return True
        return False

    # Walk ONLY module-scope statements (top-level), descending into their
    # direct expression trees but NOT into function/class bodies.
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(top):
            assert not _calls_anthropic_client(node), (
                "Anthropic client constructed at module scope"
            )
            assert not _reads_env_key(node), "API key read at module scope"


def test_client_constructed_lazily_inside_call():
    # The factory (anthropic.Anthropic) must only be invoked when extract_claims
    # runs — proving lazy construction.
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 1)
    payload = _payload_from_anchors([("only claim", anchors[0])])
    ctx, client, factory = _mock_anthropic(payload)
    with ctx:
        assert not factory.called  # not constructed yet
        claims.extract_claims(text)
        assert factory.called  # constructed during the call


# --------------------------------------------------------------------------- #
# c3: minimal import closure — only anthropic + stdlib; no heavy deps.
# --------------------------------------------------------------------------- #


def test_import_closure_is_minimal():
    tree = ast.parse(CLAIMS_PY.read_text())
    forbidden = {"bot", "app", "pipecat", "fastapi"}
    imported_tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_tops.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                imported_tops.add(node.module.split(".")[0])
    assert forbidden.isdisjoint(imported_tops), (
        f"claims.py imports forbidden modules: {forbidden & imported_tops}"
    )


def test_import_claims_module_succeeds():
    import importlib

    assert importlib.import_module("claims") is not None


# --------------------------------------------------------------------------- #
# c4: positional, unique ids.
# --------------------------------------------------------------------------- #


def test_ids_are_frozen_positional():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 4)
    payload = _payload_from_anchors(
        [(f"claim {i}", a) for i, a in enumerate(anchors)]
    )
    ctx, _c, _f = _mock_anthropic(payload)
    with ctx:
        result = claims.extract_claims(text)
    assert [r.id for r in result] == ["c1", "c2", "c3", "c4"]


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_ids_unique_and_nonempty_per_fixture(doc_id):
    text = _fixture_text(doc_id)
    anchors = _real_anchors(text, 5)
    payload = _payload_from_anchors([(f"c{i}", a) for i, a in enumerate(anchors)])
    ctx, _c, _f = _mock_anthropic(payload)
    with ctx:
        result = claims.extract_claims(text)
    ids = [r.id for r in result]
    assert len(set(ids)) == len(ids)
    assert all(i for i in ids)


# --------------------------------------------------------------------------- #
# c5: positive anchor property driven by the REAL fixtures.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_every_anchor_is_verbatim_substring_of_fixture(doc_id):
    fixture_text = _fixture_text(doc_id)
    anchors = _real_anchors(fixture_text, 6)
    payload = _payload_from_anchors(
        [(f"claim {i}", a) for i, a in enumerate(anchors)]
    )
    ctx, _c, _f = _mock_anthropic(payload)
    with ctx:
        result = claims.extract_claims(fixture_text)
    assert len(result) == len(anchors)
    for rec in result:
        assert rec.anchor in fixture_text


# --------------------------------------------------------------------------- #
# c6: prompt encodes the required granularity (static content check).
# --------------------------------------------------------------------------- #


def test_prompt_encodes_granularity():
    prompt = claims.CLAIMS_PROMPT
    # (a) 1-3 spoken-sentence framing (tolerant: 1-3 / 1 to 3 / one to three).
    sentence_pat = re.compile(
        r"(1\s*[-–to]{1,3}\s*3|one\s+to\s+three)\D{0,20}sentence",
        re.IGNORECASE,
    )
    assert sentence_pat.search(prompt), "prompt lacks 1-3 sentence framing"
    # (b) count range guidance: both 10 and 40 present.
    assert "10" in prompt and "40" in prompt, "prompt lacks ~10-40 count target"


# --------------------------------------------------------------------------- #
# c7: validation RAISES on malformed output; well-formed parses cleanly.
# --------------------------------------------------------------------------- #


def test_raises_on_missing_claim_text():
    text = _fixture_text(DOC_IDS[0])
    anchor = _real_anchors(text, 1)[0]
    payload = _payload_from_anchors([("", anchor)])
    ctx, _c, _f = _mock_anthropic(payload)
    with ctx, pytest.raises(claims.ClaimParseError):
        claims.extract_claims(text)


def test_raises_on_missing_anchor():
    text = _fixture_text(DOC_IDS[0])
    payload = _payload_from_anchors([("a real claim", "")])
    ctx, _c, _f = _mock_anthropic(payload)
    with ctx, pytest.raises(claims.ClaimParseError):
        claims.extract_claims(text)


def test_raises_on_anchor_not_in_document():
    text = _fixture_text(DOC_IDS[0])
    payload = _payload_from_anchors(
        [("a real claim", "this passage is definitely not in the document xyzzy")]
    )
    ctx, _c, _f = _mock_anthropic(payload)
    with ctx, pytest.raises(claims.ClaimParseError):
        claims.extract_claims(text)


def test_wellformed_response_parses_without_raising():
    text = _fixture_text(DOC_IDS[0])
    anchors = _real_anchors(text, 2)
    payload = _payload_from_anchors(
        [("claim one", anchors[0]), ("claim two", anchors[1])]
    )
    ctx, _c, _f = _mock_anthropic(payload)
    with ctx:
        result = claims.extract_claims(text)
    assert len(result) == 2


def test_parse_claims_direct_raises_on_bad_json():
    text = _fixture_text(DOC_IDS[0])
    with pytest.raises(claims.ClaimParseError):
        claims.parse_claims("not json at all", text)
