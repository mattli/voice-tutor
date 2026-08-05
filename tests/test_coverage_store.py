"""Hermetic tests for coverage_store: sidecar write/read, the union read path,
and the teardown orchestration's failure contract.

No network, no Anthropic client, no pipecat: the judge call is exercised through
an injected fake client, and every path constant is monkeypatched into tmp_path.
Follows CLAUDE.md "test via pure helpers, not TestClient" — bot.py's teardown is
a thin caller over these functions, and the properties are pinned here.
"""

import json

import pytest

import coverage_judge as cj
import coverage_store as cs


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """Redirect coverage_store.TRANSCRIPTS_DIR to a per-test tmp_path.

    The module reads the constant at CALL time, so patching the module attribute
    is the real resolution path (mirrors the docs_dir fixture in conftest).
    """
    root = tmp_path / "transcripts"
    monkeypatch.setattr(cs, "TRANSCRIPTS_DIR", root)
    return root


# --------------------------------------------------------------------------- #
# Helpers: a fake Anthropic-shaped client + canned inputs.
# --------------------------------------------------------------------------- #

class _Block:
    def __init__(self, text):
        self.text = text


class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Response:
    def __init__(self, text, input_tokens=None, output_tokens=None):
        self.content = [_Block(text)]
        if input_tokens is not None:
            self.usage = _Usage(input_tokens, output_tokens)


class _Messages:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        if self._parent.raises is not None:
            raise self._parent.raises
        idx = min(len(self._parent.calls) - 1, len(self._parent.responses) - 1)
        return self._parent.responses[idx]


class _Client:
    def __init__(self, responses=(), raises=None):
        self.responses = list(responses)
        self.raises = raises
        self.calls = []
        self.messages = _Messages(self)


class _Claim:
    """Stand-in for claims.Claim (only .id / .claim are consumed)."""

    def __init__(self, cid, text):
        self.id = cid
        self.claim = text


_CLAIMS = [_Claim("c1", "First claim."), _Claim("c2", "Second claim.")]
_TRANSCRIPT = {
    "turn_count": 2,
    "turns": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first claim explained"},
    ],
}
_GOOD = json.dumps(
    {"verdicts": [
        {"claim_id": "c1", "covered": True, "turns": [1], "reason": "turn 1"},
        {"claim_id": "c2", "covered": False, "turns": [], "reason": "no"},
    ]}
)


def _judge(**overrides):
    kwargs = dict(
        user_id="matt",
        session_id="sess-1",
        document_id="doc-A",
        source_hash="hash-A",
        claim_objs=_CLAIMS,
        transcript=_TRANSCRIPT,
        client=_Client([_Response(_GOOD, 100, 50)]),
    )
    kwargs.update(overrides)
    return cs.judge_session(**kwargs)


# --------------------------------------------------------------------------- #
# Path containment.
# --------------------------------------------------------------------------- #

def test_a_crafted_session_id_cannot_escape_the_users_namespace(store_dir):
    # Same class as the doc_id / session_id traversal guards elsewhere in the
    # repo: both halves collapse to a single path component.
    path = cs.coverage_path("matt", "../victim/pwned")
    assert path.parent == store_dir / "matt"
    assert path.name == "pwned.coverage.json"


def test_a_crafted_user_id_cannot_escape_the_transcripts_root(store_dir):
    path = cs.coverage_path("../../etc", "sess-1")
    assert path.parent == store_dir / "etc"


# --------------------------------------------------------------------------- #
# Write / read round-trip — the writer and reader must agree on the name.
# --------------------------------------------------------------------------- #

def test_sidecar_written_by_the_writer_is_found_by_the_union_reader(store_dir):
    # CLAUDE.md "grep for READERS of the pattern": pin the ROUND TRIP, not the
    # builder in isolation. Write under the writer's name; assert the reader
    # locates it by globbing.
    sidecar, _ = _judge()
    cs.write_sidecar("matt", "sess-1", sidecar)
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["sessions"] == 1
    assert union["covered_ids"] == ["c1"]
    assert union["session_ids"] == ["sess-1"]


def test_the_sidecar_lands_beside_the_transcript_with_the_same_stem(store_dir):
    sidecar, _ = _judge()
    path = cs.write_sidecar("matt", "sess-1", sidecar)
    assert path == store_dir / "matt" / "sess-1.coverage.json"
    assert cs.load_sidecar("matt", "sess-1")["session_id"] == "sess-1"


def test_no_percentage_is_stored_in_the_sidecar(store_dir):
    # The design's "store evidence, derive judgments": the percentage is derived
    # at read time and must never become the stored primary record.
    sidecar, _ = _judge()
    assert "percentage" not in sidecar
    assert sidecar["verdicts"][0]["turns"] == [1]  # evidence IS stored


def test_a_written_sidecar_is_never_silently_overwritten(store_dir):
    # APPEND-ONLY POLICY. The judge is not perfectly reproducible (measured: a
    # re-judge of an unchanged transcript varied by one claim at temperature 0),
    # so a silent re-judge could make a user's progress bar go DOWN with no
    # session having happened. The union is monotonic by construction instead.
    first, _ = _judge()
    path = cs.write_sidecar("matt", "sess-1", first)
    assert path is not None

    replacement = dict(first, covered_count=0, verdicts=[])
    assert cs.write_sidecar("matt", "sess-1", replacement) is None, "must decline"
    assert cs.load_sidecar("matt", "sess-1")["covered_count"] == 1, "original intact"


def test_an_explicit_overwrite_is_the_one_sanctioned_re_judge(store_dir):
    first, _ = _judge()
    cs.write_sidecar("matt", "sess-1", first)
    replacement = dict(first, covered_count=99)
    assert cs.write_sidecar("matt", "sess-1", replacement, overwrite=True) is not None
    assert cs.load_sidecar("matt", "sess-1")["covered_count"] == 99


def test_a_reused_session_id_cannot_clobber_an_earlier_sessions_coverage(store_dir):
    # session_id is CLIENT-SUPPLIED, so without the guard a reused or crafted id
    # would silently destroy a real session's record and shrink the union.
    original, _ = _judge()
    cs.write_sidecar("matt", "sess-1", original)
    before = cs.union_for_document("matt", "doc-A", "hash-A")

    attacker = dict(original, covered_count=0, verdicts=[
        {"claim_id": "c1", "covered": False, "turns": []},
        {"claim_id": "c2", "covered": False, "turns": []},
    ])
    cs.write_sidecar("matt", "sess-1", attacker)   # same id, no overwrite=
    after = cs.union_for_document("matt", "doc-A", "hash-A")
    assert after["covered_ids"] == before["covered_ids"], "the union must not shrink"


def test_an_interrupted_write_leaves_no_partial_sidecar(store_dir):
    sidecar, _ = _judge()
    cs.write_sidecar("matt", "sess-1", sidecar)
    leftovers = list((store_dir / "matt").glob("*.tmp"))
    assert leftovers == [], "the temp file must be replaced, not left behind"


# --------------------------------------------------------------------------- #
# The union read path.
# --------------------------------------------------------------------------- #

def _write_raw(store_dir, user, session, **fields):
    """Write a minimal sidecar directly, for read-path cases."""
    base = {
        "schema_version": cs.SCHEMA_VERSION,
        "session_id": session,
        "user_id": user,
        "document_id": "doc-A",
        "source_hash": "hash-A",
        "claims_total": 2,
        "verdicts": [
            {"claim_id": "c1", "covered": False, "turns": []},
            {"claim_id": "c2", "covered": False, "turns": []},
        ],
    }
    base.update(fields)
    (store_dir / user).mkdir(parents=True, exist_ok=True)
    (store_dir / user / f"{session}.coverage.json").write_text(json.dumps(base))


def test_union_is_the_UNION_across_sessions_not_the_latest(store_dir):
    # A claim covered in ANY session stays covered — this is what makes the
    # number mean "how much of this document have I been through".
    _write_raw(store_dir, "matt", "s1", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
        {"claim_id": "c2", "covered": False, "turns": []},
    ])
    _write_raw(store_dir, "matt", "s2", verdicts=[
        {"claim_id": "c1", "covered": False, "turns": []},
        {"claim_id": "c2", "covered": True, "turns": [3]},
    ])
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["covered_ids"] == ["c1", "c2"]
    assert union["percentage"] == 100.0
    assert union["sessions"] == 2


def test_percentage_is_derived_at_read_time(store_dir):
    _write_raw(store_dir, "matt", "s1", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
        {"claim_id": "c2", "covered": False, "turns": []},
    ])
    assert cs.union_for_document("matt", "doc-A", "hash-A")["percentage"] == 50.0


def test_another_users_coverage_never_contributes(store_dir):
    _write_raw(store_dir, "matt", "s1", verdicts=[
        {"claim_id": "c1", "covered": False, "turns": []},
    ])
    _write_raw(store_dir, "someone-else", "s2", user_id="someone-else", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    assert cs.union_for_document("matt", "doc-A", "hash-A")["covered_ids"] == []


def test_another_documents_coverage_never_contributes(store_dir):
    # Claim ids are per-document sequentials, so doc B's c1 is NOT doc A's c1.
    _write_raw(store_dir, "matt", "s1", document_id="doc-B", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    assert cs.union_for_document("matt", "doc-A", "hash-A")["covered_ids"] == []


def test_a_superseded_claim_map_is_ignored_and_COUNTED(store_dir):
    # The re-extraction landmine: an old sidecar's c1 refers to a claim map that
    # no longer exists. It must not merge — and the condition must be visible,
    # not a silently smaller number.
    _write_raw(store_dir, "matt", "old", source_hash="hash-OLD", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    _write_raw(store_dir, "matt", "new", verdicts=[
        {"claim_id": "c1", "covered": False, "turns": []},
    ])
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["covered_ids"] == []
    assert union["sessions"] == 1
    assert union["stale_sessions"] == 1


def test_passing_no_source_hash_merges_every_map_version(store_dir):
    _write_raw(store_dir, "matt", "old", source_hash="hash-OLD", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    union = cs.union_for_document("matt", "doc-A", source_hash=None)
    assert union["covered_ids"] == ["c1"]
    assert union["stale_sessions"] == 0


def test_a_corrupt_sidecar_is_skipped_not_fatal(store_dir):
    _write_raw(store_dir, "matt", "good", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    (store_dir / "matt" / "broken.coverage.json").write_text("{not json")
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["covered_ids"] == ["c1"], "one corrupt file must not cost the number"


def test_a_document_with_no_coverage_reads_as_zero_not_an_error(store_dir):
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union == {
        "covered_ids": [], "percentage": 0.0, "claims_total": 0,
        "sessions": 0, "stale_sessions": 0, "session_ids": [],
    }


def test_a_user_with_no_directory_at_all_reads_as_zero(store_dir):
    assert cs.union_for_document("nobody", "doc-A", "hash-A")["percentage"] == 0.0


# --------------------------------------------------------------------------- #
# The teardown orchestration + its failure contract.
# --------------------------------------------------------------------------- #

def test_judge_session_returns_a_sidecar_and_an_ok_cost_row():
    sidecar, cost = _judge()
    assert sidecar["covered_count"] == 1
    assert sidecar["claims_total"] == 2
    assert sidecar["source_hash"] == "hash-A"
    assert sidecar["judge_prompt_hash"] == cj.JUDGE_PROMPT_V2_HASH
    assert cost["kind"] == "coverage"
    assert cost["status"] == "ok"
    assert cost["calls"] == 1
    assert cost["input_tokens"] == 100


def test_a_judge_FAILURE_degrades_to_no_coverage_and_never_raises():
    # THE failure contract: a coverage failure yields no coverage data for the
    # session — never a broken teardown, and never a lost transcript/recap.
    sidecar, cost = _judge(client=_Client([_Response("not json", 10, 5)]))
    assert sidecar is None
    assert cost["status"] == "failed"
    assert "error" in cost


def test_an_SDK_exception_is_also_contained():
    sidecar, cost = _judge(client=_Client(raises=RuntimeError("rate limited")))
    assert sidecar is None
    assert "RuntimeError" in cost["error"]


def test_cost_is_reported_even_when_the_judge_FAILED():
    # Finding 5 at the wiring layer: the failed run's spend must still be
    # attributable in the ledger.
    _, cost = _judge(client=_Client([_Response("not json", 10, 5)]))
    assert cost["status"] == "failed"
    assert cost["calls"] == cj.MAX_JUDGE_ATTEMPTS
    assert cost["input_tokens"] == 20, "both billed attempts counted"


def test_an_unobserved_token_count_is_omitted_from_the_ledger_row():
    # Finding 6 at the wiring layer: no confident zero for an unmeasured count.
    _, cost = _judge(client=_Client([_Response(_GOOD)]))
    assert "input_tokens" not in cost
    assert cost["usage_complete"] is False


def test_the_claim_map_identity_is_stamped_so_the_union_guard_can_see_it():
    # judge_coverage derives doc_id from the claims envelope's source_hash; the
    # cross-document merge guard depends on that stamp existing.
    sidecar, _ = _judge()
    assert sidecar["doc_id"] == "hash-A"


def test_importing_the_store_pulls_in_no_heavy_or_live_wiring():
    # Mirrors coverage_judge's isolation guard: this module is imported by bot.py
    # but must stay importable (and testable) with no pipecat/anthropic/fastapi
    # stack — the property that keeps the hermetic suite runnable at all.
    import ast
    from pathlib import Path

    src = Path(cs.__file__).read_text()
    imported = set()
    # MODULE SCOPE only (the tree's own body, not ast.walk): the Anthropic client
    # is deliberately imported lazily INSIDE judge_session, which is exactly what
    # keeps `import coverage_store` free of it. A walk would flag that lazy import
    # and pin the opposite of the intended property.
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("bot", "app", "pipecat", "fastapi", "anthropic", "documents"):
        assert forbidden not in imported, f"coverage_store imports {forbidden}"


def test_the_judge_sees_only_this_sessions_transcript():
    # Design constraint: the judge is never told what prior sessions covered —
    # union happens after, at read time.
    client = _Client([_Response(_GOOD, 1, 1)])
    _judge(client=client)
    sent = client.calls[0]["messages"][0]["content"]
    assert "first claim explained" in sent
    assert "covered_ids" not in sent and "prior session" not in sent.lower()
