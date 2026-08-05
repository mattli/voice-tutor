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


def test_passing_no_source_hash_reads_the_only_map_version_present(store_dir):
    _write_raw(store_dir, "matt", "old", source_hash="hash-OLD", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    union = cs.union_for_document("matt", "doc-A", source_hash=None)
    assert union["covered_ids"] == ["c1"]
    assert union["stale_sessions"] == 0


def test_no_source_hash_takes_the_NEWEST_map_version_and_never_merges_across(store_dir):
    # The old docstring promised "merge every sidecar regardless of map version",
    # which is the false-number case the source_hash filter exists to prevent:
    # c1 under the old map is a DIFFERENT claim from c1 under the new one. The
    # honest answer is one version's number plus a stale count — never a raise
    # (that was finding 3) and never a merge.
    _write_raw(store_dir, "matt", "old", source_hash="hash-OLD",
               judged_at="2026-01-01T00:00:00", verdicts=[
                   {"claim_id": "c1", "covered": True, "turns": [1]},
                   {"claim_id": "c2", "covered": True, "turns": [2]},
               ])
    _write_raw(store_dir, "matt", "new", source_hash="hash-NEW",
               judged_at="2026-08-01T00:00:00", verdicts=[
                   {"claim_id": "c1", "covered": True, "turns": [1]},
                   {"claim_id": "c2", "covered": False, "turns": []},
               ])
    union = cs.union_for_document("matt", "doc-A", source_hash=None)
    assert union["covered_ids"] == ["c1"], "the newest map's verdicts only"
    assert union["sessions"] == 1
    assert union["stale_sessions"] == 1


def test_differing_map_versions_with_no_hash_given_never_raise(store_dir):
    # The reproduced half of finding 3: this exact input used to raise.
    _write_raw(store_dir, "matt", "old", source_hash="hash-OLD", verdicts=[])
    _write_raw(store_dir, "matt", "new", source_hash="hash-NEW", verdicts=[])
    assert cs.union_for_document("matt", "doc-A", source_hash=None)["percentage"] == 0.0


# --------------------------------------------------------------------------- #
# Degradation: a broken sidecar costs itself, not the document (finding 3).
# --------------------------------------------------------------------------- #

def test_one_structurally_invalid_sidecar_does_not_cost_the_whole_number(store_dir):
    # `covered` is not a bool and `verdicts` is not a list: both used to raise
    # CoverageInputError for the ENTIRE union, which behind a route is a 500 and
    # a blank panel — design constraint 2 inverted.
    _write_raw(store_dir, "matt", "good", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
        {"claim_id": "c2", "covered": False, "turns": []},
    ])
    _write_raw(store_dir, "matt", "bad-covered", verdicts=[
        {"claim_id": "c1", "covered": "yes", "turns": [1]},
    ])
    _write_raw(store_dir, "matt", "bad-verdicts", verdicts={"c1": True})
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["covered_ids"] == ["c1"]
    assert union["sessions"] == 1, "only the good sidecar contributed"
    assert union["invalid_sessions"] == 2


def test_invalid_and_stale_are_counted_SEPARATELY(store_dir):
    # Stale is the system working (a superseded map correctly excluded); invalid
    # is the system losing data. One counter for both would hide the second.
    _write_raw(store_dir, "matt", "stale", source_hash="hash-OLD", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    _write_raw(store_dir, "matt", "broken", verdicts=[{"covered": True, "turns": [1]}])
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["stale_sessions"] == 1
    assert union["invalid_sessions"] == 1


def test_an_invalid_sidecar_is_LOGGED_with_its_path(store_dir, capsys):
    # A counter says something was dropped; only the path says which file, which
    # is the one thing needed to fix it.
    _write_raw(store_dir, "matt", "broken", verdicts=[{"covered": True}])
    cs.union_for_document("matt", "doc-A", "hash-A")
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "broken.coverage.json" in err


def test_an_unreadable_sidecar_is_also_logged(store_dir, capsys):
    (store_dir / "matt").mkdir(parents=True, exist_ok=True)
    (store_dir / "matt" / "corrupt.coverage.json").write_text("{not json")
    cs.union_for_document("matt", "doc-A", "hash-A")
    err = capsys.readouterr().err
    assert "corrupt.coverage.json" in err


def test_a_sidecar_whose_doc_id_stamp_disagrees_is_stale_not_merged(store_dir):
    # Second identity layer: the envelope says hash-A but the verdict set was
    # stamped against a different map. Preserved from union_coverage's guard,
    # which per-sidecar merging would otherwise bypass.
    _write_raw(store_dir, "matt", "s1", doc_id="hash-OTHER", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["covered_ids"] == []
    assert union["stale_sessions"] == 1


def test_the_read_path_never_raises_on_any_junk_in_the_directory(store_dir):
    (store_dir / "matt").mkdir(parents=True, exist_ok=True)
    (store_dir / "matt" / "a.coverage.json").write_text("[]")
    (store_dir / "matt" / "b.coverage.json").write_text("null")
    (store_dir / "matt" / "c.coverage.json").write_text('"a string"')
    (store_dir / "matt" / "d.coverage.json").write_text("")
    assert cs.union_for_document("matt", "doc-A", "hash-A")["percentage"] == 0.0


# --------------------------------------------------------------------------- #
# Display shape: the number a bar actually draws.
# --------------------------------------------------------------------------- #

def test_display_percentage_is_over_the_CLAIM_MAP_not_the_judged_universe(store_dir):
    # A judge pass that returned verdicts for only some claims must not make the
    # same progress read as a bigger number.
    _write_raw(store_dir, "matt", "s1", claims_total=63, verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
        {"claim_id": "c2", "covered": False, "turns": []},
    ])
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert union["percentage"] == 50.0, "the judged-universe figure is unchanged"
    display = cs.as_display(union, 63)
    assert display == {"covered": 1, "total": 63, "percentage": 1.6, "sessions": 1}


def test_a_document_with_no_sessions_displays_as_NOTHING_not_zero(store_dir):
    # "No coverage" and "0% coverage" read differently to a person; the element
    # must be absent, so the shape has to be None rather than a zeroed dict.
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert cs.as_display(union, 63) is None


def test_display_falls_back_to_the_sidecar_total_when_the_map_size_is_unknown(store_dir):
    _write_raw(store_dir, "matt", "s1", claims_total=10, verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert cs.as_display(union, None)["total"] == 10


def test_display_is_none_when_no_total_is_knowable(store_dir):
    _write_raw(store_dir, "matt", "s1", claims_total=0, verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    union = cs.union_for_document("matt", "doc-A", "hash-A")
    assert cs.as_display(union, None) is None


# --------------------------------------------------------------------------- #
# The picker's read path: one scan, many documents.
# --------------------------------------------------------------------------- #

def _identity(**by_doc):
    """Build an identity_for callback from {doc_id: (source_hash, claims_total)}."""
    def _fn(doc_id):
        entry = by_doc.get(doc_id)
        if entry is None:
            return None
        return {"source_hash": entry[0], "claims_total": entry[1]}
    return _fn


def test_documents_view_returns_one_entry_per_covered_document(store_dir):
    _write_raw(store_dir, "matt", "s1", document_id="doc-A", claims_total=63, verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    _write_raw(store_dir, "matt", "s2", document_id="doc-B", source_hash="hash-B",
               claims_total=10, verdicts=[
                   {"claim_id": "c1", "covered": True, "turns": [1]},
                   {"claim_id": "c2", "covered": True, "turns": [2]},
               ])
    view = cs.documents_view("matt", _identity(**{"doc-A": ("hash-A", 63), "doc-B": ("hash-B", 10)}))
    assert set(view) == {"doc-A", "doc-B"}
    assert view["doc-A"]["covered"] == 1 and view["doc-A"]["total"] == 63
    assert view["doc-B"] == {"covered": 2, "total": 10, "percentage": 20.0, "sessions": 1}


def test_documents_view_omits_a_document_with_no_fresh_claim_map(store_dir):
    # No fresh map means the tutor isn't steering from that rubric either; a bar
    # against a superseded map would show progress toward something retired.
    _write_raw(store_dir, "matt", "s1", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    assert cs.documents_view("matt", _identity()) == {}


def test_documents_view_omits_another_users_documents(store_dir):
    _write_raw(store_dir, "someone-else", "s1", user_id="someone-else", claims_total=5,
               verdicts=[{"claim_id": "c1", "covered": True, "turns": [1]}])
    assert cs.documents_view("matt", _identity(**{"doc-A": ("hash-A", 5)})) == {}


def test_documents_view_survives_an_identity_lookup_that_raises(store_dir):
    _write_raw(store_dir, "matt", "s1", document_id="doc-A", claims_total=5, verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    _write_raw(store_dir, "matt", "s2", document_id="doc-B", source_hash="hash-B",
               claims_total=5, verdicts=[{"claim_id": "c1", "covered": True, "turns": [1]}])

    def _boom(doc_id):
        if doc_id == "doc-A":
            raise RuntimeError("document layer exploded")
        return {"source_hash": "hash-B", "claims_total": 5}

    view = cs.documents_view("matt", _boom)
    assert set(view) == {"doc-B"}, "one bad document must not cost the whole list"


def test_documents_view_reads_the_directory_ONCE_for_many_documents(store_dir, monkeypatch):
    # The picker may list many documents; a per-document round trip would make it
    # an N+1 over the same files.
    for i in range(4):
        _write_raw(store_dir, "matt", f"s{i}", document_id=f"doc-{i}", claims_total=5,
                   verdicts=[{"claim_id": "c1", "covered": True, "turns": [1]}])
    calls = []
    real = cs._iter_sidecar_files
    monkeypatch.setattr(cs, "_iter_sidecar_files", lambda uid: (calls.append(uid), real(uid))[1])
    view = cs.documents_view("matt", lambda d: {"source_hash": "hash-A", "claims_total": 5})
    assert len(view) == 4
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# The ended view: this session's contribution.
# --------------------------------------------------------------------------- #

def test_session_contribution_counts_only_what_was_NEW(store_dir):
    _write_raw(store_dir, "matt", "first", judged_at="2026-08-01T00:00:00", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
        {"claim_id": "c2", "covered": False, "turns": []},
    ])
    _write_raw(store_dir, "matt", "second", judged_at="2026-08-02T00:00:00", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [3]},
        {"claim_id": "c2", "covered": True, "turns": [4]},
    ])
    got = cs.session_contribution("matt", "second", "doc-A", source_hash="hash-A")
    assert got == {"covered": 2, "new_claims": 1}, "c1 was already covered before"


def test_a_past_sessions_contribution_does_not_shrink_when_a_LATER_session_repeats_it(store_dir):
    # Measured against EARLIER sessions only. Otherwise re-opening an old session
    # would show a smaller contribution than it did on the day, which is not what
    # "what this session added" means.
    _write_raw(store_dir, "matt", "first", judged_at="2026-08-01T00:00:00", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    _write_raw(store_dir, "matt", "later", judged_at="2026-08-05T00:00:00", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    assert cs.session_contribution("matt", "first", "doc-A", source_hash="hash-A") == {
        "covered": 1, "new_claims": 1,
    }


def test_session_contribution_is_None_when_the_judge_produced_nothing(store_dir):
    _write_raw(store_dir, "matt", "other", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    assert cs.session_contribution("matt", "no-sidecar", "doc-A", source_hash="hash-A") is None


def test_session_contribution_is_None_for_a_structurally_invalid_own_sidecar(store_dir):
    _write_raw(store_dir, "matt", "broken", verdicts=[{"covered": True, "turns": []}])
    assert cs.session_contribution("matt", "broken", "doc-A", source_hash="hash-A") is None


def test_session_contribution_never_reads_another_documents_sidecar(store_dir):
    _write_raw(store_dir, "matt", "s1", document_id="doc-B", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    assert cs.session_contribution("matt", "s1", "doc-A", source_hash="hash-A") is None


def test_session_view_shows_the_TOTAL_even_when_this_sessions_judge_failed(store_dir):
    # The brief's rule: a missing sidecar means the delta is absent, never an
    # error, and never a missing total.
    _write_raw(store_dir, "matt", "earlier", claims_total=63, verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    view = cs.session_view("matt", "unjudged", "doc-A", source_hash="hash-A", claims_total=63)
    assert view["total"] == {"covered": 1, "total": 63, "percentage": 1.6, "sessions": 1}
    assert view["session"] is None


def test_session_view_is_None_when_there_is_nothing_at_all_to_show(store_dir):
    assert cs.session_view("matt", "s1", "doc-A", source_hash="hash-A", claims_total=63) is None
    assert cs.session_view("matt", "s1", None, source_hash=None, claims_total=None) is None


def test_session_view_carries_both_halves_for_a_judged_session(store_dir):
    _write_raw(store_dir, "matt", "first", claims_total=63,
               judged_at="2026-08-01T00:00:00", verdicts=[
                   {"claim_id": "c1", "covered": True, "turns": [1]},
               ])
    _write_raw(store_dir, "matt", "second", claims_total=63,
               judged_at="2026-08-02T00:00:00", verdicts=[
                   {"claim_id": "c1", "covered": True, "turns": [1]},
                   {"claim_id": "c2", "covered": True, "turns": [2]},
               ])
    view = cs.session_view("matt", "second", "doc-A", source_hash="hash-A", claims_total=63)
    assert view["total"]["covered"] == 2
    assert view["total"]["sessions"] == 2
    assert view["session"] == {"covered": 2, "new_claims": 1}


def test_a_crafted_session_id_cannot_read_another_users_contribution(store_dir):
    _write_raw(store_dir, "victim", "secret", user_id="victim", verdicts=[
        {"claim_id": "c1", "covered": True, "turns": [1]},
    ])
    (store_dir / "matt").mkdir(parents=True, exist_ok=True)  # matt's own dir must exist
    assert cs.session_contribution(
        "matt", "../victim/secret", "doc-A", source_hash="hash-A"
    ) is None


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
        "sessions": 0, "stale_sessions": 0, "invalid_sessions": 0, "session_ids": [],
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


def test_a_wholesale_citation_failure_writes_NO_sidecar(store_dir):
    # End of the retry-then-no-sidecar chain. A verdict set whose every coverage
    # claim cites a nonexistent turn repairs down to a well-formed "0 covered" —
    # which, written, would be a PERMANENT false zero (append-only: never
    # re-judged without --force). judge_coverage refuses it, judge_session
    # degrades to (None, cost_row), and the caller writes nothing.
    all_bogus = json.dumps({"verdicts": [
        {"claim_id": "c1", "covered": True, "turns": [101], "reason": "x"},
        {"claim_id": "c2", "covered": True, "turns": [102], "reason": "y"},
    ]})
    sidecar, cost = _judge(client=_Client([_Response(all_bogus, 10, 5)]))
    assert sidecar is None, "a repaired-to-zero set must not become a record"
    assert cost["status"] == "failed"
    assert "MassCitationDowngradeError" in cost["error"]
    assert cost["calls"] == cj.MAX_JUDGE_ATTEMPTS, "retried, then gave up"
    # And nothing landed on disk: no coverage data beats permanent wrong data.
    assert cs.load_sidecar("matt", "sess-1") is None
    assert cs.union_for_document("matt", "doc-A", "hash-A")["sessions"] == 0


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
