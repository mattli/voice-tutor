"""Hermetic tests for the coverage-judge foundations (coverage_judge.py).

The verifier runs EXACTLY:
    uv run --with pytest pytest tests/test_coverage_judge.py -q
in a fresh worktree with no local .venv and NO ANTHROPIC_API_KEY, so ALL
hermetic (offline, model-call-free) tests for this work live in THIS single
file. The suite makes NO network/LLM calls and constructs no Anthropic client:
``import coverage_judge`` reads no key and performs no I/O, and Sprint 0 has no
live model call to mock yet.

These tests exercise the input contract (claim + indexed-transcript loading /
validation) and the strict-JSON transport defenses (markdown-fence stripping,
truncation/count-mismatch detection, unknown/duplicate/missing claim-id
detection, and verdict-shape validation) that everything downstream builds on.
"""

import ast
import json
from pathlib import Path

import pytest

import coverage_judge as cj

# --------------------------------------------------------------------------- #
# c1: hermetic isolation — no network, no live-app imports, no client at import.
# --------------------------------------------------------------------------- #

_MODULE_PATH = Path(cj.__file__)
_FORBIDDEN_TOPLEVEL_IMPORTS = {
    "bot", "app", "documents", "pipecat", "fastapi", "uvicorn", "anthropic",
}


def test_module_has_no_network_or_live_app_imports_at_module_scope():
    """coverage_judge must not import bot/app/pipecat/anthropic at module scope.

    Parses the module's own source AST and asserts none of the forbidden
    top-level modules are imported at module scope — so ``import coverage_judge``
    (already performed above without a network client) can never drag in the live
    app path or an SDK client. Function-local lazy imports (a future model seam)
    are allowed and are not module-scope, so they don't appear here.
    """
    tree = ast.parse(_MODULE_PATH.read_text())
    imported: set[str] = set()
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if node.module:
                imported.add(node.module.split(".")[0])
    leaked = imported & _FORBIDDEN_TOPLEVEL_IMPORTS
    assert not leaked, f"coverage_judge imports forbidden modules at module scope: {leaked}"


def test_import_constructs_no_anthropic_client(monkeypatch):
    """Re-importing coverage_judge constructs no Anthropic client.

    Even if the SDK is installed, importing the module must not build a client
    (which would read an API key / open a connection). We poison
    ``anthropic.Anthropic`` to raise if constructed, then force a fresh import.
    """
    import importlib
    import sys

    anthropic = pytest.importorskip("anthropic") if "anthropic" in sys.modules else None
    if anthropic is not None:
        def _boom(*a, **k):  # pragma: no cover - only fires on regression
            raise AssertionError("Anthropic client constructed at import time")

        monkeypatch.setattr(anthropic, "Anthropic", _boom)

    sys.modules.pop("coverage_judge", None)
    mod = importlib.import_module("coverage_judge")
    assert mod is not None


# --------------------------------------------------------------------------- #
# c2: claim-list loading (sidecar envelope + bare list, order, claim->text).
# --------------------------------------------------------------------------- #

def _sidecar(records):
    """Wrap claim records in the repo's `.claims.json` envelope shape."""
    return {"source_hash": "deadbeef", "claims": records}


def test_load_claims_from_sidecar_envelope_preserves_order_and_normalizes_claim():
    data = _sidecar([
        {"id": "c1", "claim": "The sky is blue.", "anchor": "..."},
        {"id": "c2", "claim": "Water is wet.", "anchor": "..."},
        {"id": "c3", "claim": "Fire is hot.", "anchor": "..."},
    ])
    claims = cj.load_claims(data)
    assert [c.id for c in claims] == ["c1", "c2", "c3"]
    assert [c.text for c in claims] == ["The sky is blue.", "Water is wet.", "Fire is hot."]
    assert cj.claim_ids(claims) == ["c1", "c2", "c3"]


def test_load_claims_from_bare_list():
    data = [
        {"id": "c1", "claim": "Alpha."},
        {"id": "c2", "claim": "Beta."},
    ]
    claims = cj.load_claims(data)
    assert [(c.id, c.text) for c in claims] == [("c1", "Alpha."), ("c2", "Beta.")]


def test_load_claims_accepts_bare_text_field():
    """A record using a bare ``text`` field (not the sidecar's ``claim``) works."""
    claims = cj.load_claims([{"id": "c1", "text": "Gamma."}])
    assert claims[0].text == "Gamma."


def test_load_claims_empty_list_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims(_sidecar([]))
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims([])


# --------------------------------------------------------------------------- #
# c3: claim-list validation errors (typed, not bare KeyError/IndexError).
# --------------------------------------------------------------------------- #

def test_load_claims_missing_id_raises_typed_error():
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims([{"claim": "no id here"}])


def test_load_claims_missing_or_blank_text_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims([{"id": "c1"}])
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims([{"id": "c1", "claim": "   "}])


def test_load_claims_duplicate_id_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims([
            {"id": "c1", "claim": "one"},
            {"id": "c1", "claim": "two"},
        ])


def test_load_claims_wrong_toplevel_type_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims("not a list or envelope")
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims({"not_claims": []})


def test_load_claims_non_object_record_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_claims(["just a string"])


# --------------------------------------------------------------------------- #
# c4: transcript loading / index normalization.
# --------------------------------------------------------------------------- #

def test_load_transcript_assigns_contiguous_indices_when_absent():
    """Real ~/.voice-tutor transcripts carry no index; assign 0-based in order."""
    data = {
        "turn_count": 3,
        "turns": [
            {"role": "assistant", "content": "hi", "timestamp": "t0"},
            {"role": "user", "content": "hello", "timestamp": "t1"},
            {"role": "assistant", "content": "let's go", "timestamp": "t2"},
        ],
    }
    turns = cj.load_transcript(data)
    assert [t.index for t in turns] == [0, 1, 2]
    assert all(isinstance(t.index, int) for t in turns)
    assert [(t.role, t.content) for t in turns] == [
        ("assistant", "hi"), ("user", "hello"), ("assistant", "let's go"),
    ]


def test_load_transcript_bare_list_assigns_indices():
    turns = cj.load_transcript([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ])
    assert [t.index for t in turns] == [0, 1]


def test_load_transcript_preserves_supplied_int_indices():
    data = [
        {"index": 10, "role": "user", "content": "a"},
        {"index": 20, "role": "assistant", "content": "b"},
        {"index": 5, "role": "user", "content": "c"},
    ]
    turns = cj.load_transcript(data)
    assert [t.index for t in turns] == [10, 20, 5]
    assert all(isinstance(t.index, int) for t in turns)


def test_load_transcript_missing_role_or_content_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript([{"content": "no role"}])
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript([{"role": "user"}])


def test_load_transcript_non_int_supplied_index_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript([{"index": "0", "role": "user", "content": "a"}])
    # bool must not pose as an int index.
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript([{"index": True, "role": "user", "content": "a"}])


def test_load_transcript_partial_index_set_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript([
            {"index": 0, "role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},  # no index
        ])


def test_load_transcript_duplicate_supplied_index_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript([
            {"index": 1, "role": "user", "content": "a"},
            {"index": 1, "role": "assistant", "content": "b"},
        ])


def test_load_transcript_non_list_toplevel_raises():
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript("nope")
    with pytest.raises(cj.CoverageInputError):
        cj.load_transcript({"no_turns": []})


# --------------------------------------------------------------------------- #
# c5: markdown-fence stripping (composes with parsing; value-level equality).
# --------------------------------------------------------------------------- #

_CLEAN_PAYLOAD = '[{"claim_id": "c1", "covered": true, "turns": [0]}]'


def test_strip_fences_json_tagged():
    fenced = "```json\n" + _CLEAN_PAYLOAD + "\n```"
    assert json.loads(cj.strip_code_fences(fenced)) == json.loads(_CLEAN_PAYLOAD)


def test_strip_fences_plain_and_whitespace():
    plain = "  ```\n" + _CLEAN_PAYLOAD + "\n```  \n"
    assert json.loads(cj.strip_code_fences(plain)) == json.loads(_CLEAN_PAYLOAD)


def test_strip_fences_noop_on_clean_json():
    assert json.loads(cj.strip_code_fences(_CLEAN_PAYLOAD)) == json.loads(_CLEAN_PAYLOAD)


def test_fenced_and_unfenced_parse_equal():
    variants = [
        _CLEAN_PAYLOAD,
        "```json\n" + _CLEAN_PAYLOAD + "\n```",
        "```\n" + _CLEAN_PAYLOAD + "\n```",
        "\n\n```json\n" + _CLEAN_PAYLOAD + "\n```\n\n",
    ]
    parsed = [cj.loads_json(v) for v in variants]
    assert all(p == parsed[0] for p in parsed)


# --------------------------------------------------------------------------- #
# c6: strict verdict shape/type validation (bool covered, list-of-int turns).
# --------------------------------------------------------------------------- #

def test_parse_verdicts_valid_payload_typed():
    payload = [
        {"claim_id": "c1", "covered": True, "turns": [0, 2]},
        {"claim_id": "c2", "covered": False, "turns": []},
    ]
    verdicts = cj.parse_verdicts(payload, ["c1", "c2"])
    assert [(v.claim_id, v.covered, v.turns) for v in verdicts] == [
        ("c1", True, [0, 2]),
        ("c2", False, []),
    ]
    assert isinstance(verdicts[0].covered, bool)
    # A covered:false verdict with turns:[] is valid, not an error.
    assert verdicts[1].turns == []


def test_parse_verdicts_non_bool_covered_raises():
    payload = [{"claim_id": "c1", "covered": 1, "turns": []}]
    with pytest.raises(cj.VerdictShapeError):
        cj.parse_verdicts(payload, ["c1"])


def test_parse_verdicts_missing_claim_id_field_raises():
    payload = [{"covered": True, "turns": [0]}]
    with pytest.raises(cj.VerdictShapeError):
        cj.parse_verdicts(payload, ["c1"])


def test_parse_verdicts_non_int_turns_element_raises():
    payload = [{"claim_id": "c1", "covered": True, "turns": [0, "1"]}]
    with pytest.raises(cj.VerdictShapeError):
        cj.parse_verdicts(payload, ["c1"])


def test_parse_verdicts_turns_not_a_list_raises():
    payload = [{"claim_id": "c1", "covered": True, "turns": 0}]
    with pytest.raises(cj.VerdictShapeError):
        cj.parse_verdicts(payload, ["c1"])


# --------------------------------------------------------------------------- #
# c7: completeness / id-mismatch — the defined exception hierarchy.
# --------------------------------------------------------------------------- #

def test_all_parse_errors_share_base():
    for sub in (
        cj.VerdictJSONError,
        cj.VerdictShapeError,
        cj.VerdictCountError,
        cj.UnknownClaimIdError,
        cj.DuplicateClaimIdError,
        cj.MissingClaimIdError,
    ):
        assert issubclass(sub, cj.VerdictParseError)


def test_truncated_verdict_list_raises_count_error():
    """Fewer verdicts than claims is the truncation signature."""
    payload = [{"claim_id": "c1", "covered": True, "turns": []}]
    with pytest.raises(cj.VerdictCountError) as ei:
        cj.parse_verdicts(payload, ["c1", "c2", "c3"])
    assert isinstance(ei.value, cj.VerdictParseError)


def test_extra_unknown_claim_id_raises():
    payload = [
        {"claim_id": "c1", "covered": True, "turns": []},
        {"claim_id": "c99", "covered": False, "turns": []},
    ]
    with pytest.raises(cj.UnknownClaimIdError) as ei:
        cj.parse_verdicts(payload, ["c1", "c2"])
    assert isinstance(ei.value, cj.VerdictParseError)


def test_duplicate_claim_id_in_output_raises():
    payload = [
        {"claim_id": "c1", "covered": True, "turns": []},
        {"claim_id": "c1", "covered": False, "turns": []},
    ]
    with pytest.raises(cj.DuplicateClaimIdError) as ei:
        cj.parse_verdicts(payload, ["c1", "c2"])
    assert isinstance(ei.value, cj.VerdictParseError)


def test_missing_claim_id_raises_via_completeness_check():
    """A distinct expected id with no verdict raises MissingClaimIdError.

    The pure completeness helper is the shared choke point every id-level defense
    flows through, and lets us isolate the missing-id branch directly (in the
    full parse_verdicts path a short list trips the count gate first — see
    test_truncated_verdict_list_raises_count_error). Here present ids cover c1 but
    omit the expected c2.
    """
    with pytest.raises(cj.MissingClaimIdError) as ei:
        cj.verify_claim_id_coverage(["c1"], ["c1", "c2"])
    assert isinstance(ei.value, cj.VerdictParseError)


def test_completeness_check_unknown_and_duplicate():
    """The same helper raises the unknown/duplicate subclasses too."""
    with pytest.raises(cj.UnknownClaimIdError):
        cj.verify_claim_id_coverage(["c1", "c99"], ["c1", "c2"])
    with pytest.raises(cj.DuplicateClaimIdError):
        cj.verify_claim_id_coverage(["c1", "c1"], ["c1", "c2"])


def test_completeness_check_passes_on_bijection():
    """A present set that bijects onto the distinct expected ids passes."""
    assert cj.verify_claim_id_coverage(["c2", "c1"], ["c1", "c2"]) is None


def test_output_reordered_to_expected_order():
    payload = [
        {"claim_id": "c2", "covered": False, "turns": []},
        {"claim_id": "c1", "covered": True, "turns": [3]},
    ]
    verdicts = cj.parse_verdicts(payload, ["c1", "c2"])
    assert [v.claim_id for v in verdicts] == ["c1", "c2"]


def test_full_count_mixed_true_false_parses():
    payload = [
        {"claim_id": "c1", "covered": True, "turns": [0]},
        {"claim_id": "c2", "covered": False, "turns": []},
        {"claim_id": "c3", "covered": True, "turns": [1, 2]},
    ]
    verdicts = cj.parse_verdicts(payload, ["c1", "c2", "c3"])
    assert [v.covered for v in verdicts] == [True, False, True]


# --------------------------------------------------------------------------- #
# c8: invalid JSON raises the module base parse error, not json.JSONDecodeError.
# --------------------------------------------------------------------------- #

def test_invalid_json_raises_module_base_error():
    for garbage in ("not json at all", '{"claim_id": "c1", ', "```json\n{oops\n```"):
        with pytest.raises(cj.VerdictParseError) as ei:
            cj.loads_json(garbage)
        assert isinstance(ei.value, cj.VerdictJSONError)
        assert not isinstance(ei.value, json.JSONDecodeError)


def test_parse_verdicts_text_full_pipeline():
    text = "```json\n[{\"claim_id\": \"c1\", \"covered\": true, \"turns\": [0]}]\n```"
    verdicts = cj.parse_verdicts_text(text, ["c1"])
    # `reason` is part of the persisted verdict shape (the judge's per-claim
    # rationale is kept for auditability); None when the model omitted it.
    assert verdicts[0].to_dict() == {
        "claim_id": "c1", "covered": True, "turns": [0], "reason": None
    }


# --------------------------------------------------------------------------- #
# Sprint 0 (this launch): prompt-hash function + v1 provenance + authored v2.
#
# These EXTEND the Stage-0 hermetic suite above (same file, same module). They
# add the single public prompt-hash function reused for both prompts, verify the
# v1 provenance against the copied fixture, and lock the authored v2 prompt's
# content-match requirement and stable, distinct hash. Still fully offline — no
# network, no model call, no Anthropic client.
# --------------------------------------------------------------------------- #

# The v1 judge prompt copied byte-for-byte from the eval-set folder. It is the
# provenance artifact hashed below; NEVER re-read from the external eval-set
# source path (which does not exist in a fresh verifier worktree).
_V1_PROMPT_FIXTURE = Path(__file__).parent / "fixtures" / "coverage" / "judge-prompt-v1.md"

# The provenance hash recorded for v1 in the eval-set folder (judge-prompt-v1.md
# line 4 and every *.coverage.json). v1 is a fixed INPUT to VERIFY, not a target
# to reproduce or tune toward.
_V1_RECORDED_HASH = "632b73a34b1a22b1"


# --- c1: the single public, pure, deterministic hash function. --------------- #

def test_prompt_hash_is_deterministic_same_input_same_output():
    """The one public hash function is pure: same input => identical output."""
    sample = "some judge prompt text\nwith a couple of lines"
    first = cj.prompt_hash(sample)
    second = cj.prompt_hash(sample)
    assert first == second
    # Called a third time on a fresh equal string (no memoized identity reliance).
    assert cj.prompt_hash("some judge prompt text\nwith a couple of lines") == first


def test_prompt_hash_different_input_different_hash():
    assert cj.prompt_hash("alpha") != cj.prompt_hash("beta")


def test_prompt_hash_shape_is_16_hex_chars():
    h = cj.prompt_hash("anything")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_prompt_hash_has_no_side_effects_on_repeat_calls():
    """Pure: repeated calls interleaved with other inputs never drift."""
    a1 = cj.prompt_hash("A")
    _ = cj.prompt_hash("B")
    _ = cj.prompt_hash("C")
    a2 = cj.prompt_hash("A")
    assert a1 == a2


def test_exactly_one_public_hash_function():
    """There is a single public prompt-hash entry point (no duplicate routine).

    Guards against a second copy-pasted hash implementation: exactly one public
    (non-underscore) module attribute is a callable whose name signals it is the
    prompt-hash function, and it is ``prompt_hash``.
    """
    public_hashers = [
        name
        for name in dir(cj)
        if not name.startswith("_")
        and callable(getattr(cj, name))
        and "prompt_hash" in name
    ]
    assert public_hashers == ["prompt_hash"], public_hashers


# --- c2: the v1 fixture is committed, non-empty, and a .md file. ------------- #

def test_v1_prompt_fixture_is_committed_and_nonempty():
    assert _V1_PROMPT_FIXTURE.exists(), f"missing v1 fixture: {_V1_PROMPT_FIXTURE}"
    assert _V1_PROMPT_FIXTURE.suffix == ".md"
    assert _V1_PROMPT_FIXTURE.read_bytes().strip(), "v1 fixture is empty"


# --- c3: v1 provenance is VERIFIED (not reproduced). ------------------------- #
#
# FINDING (vault: areas/dev-harness/runs/voice-tutor/2026-08-03-coverage-judge-
# module/2026-08-03-v1-prompt-hash-finding.md): under the documented scheme
# ``sha256[:16]``, NO recoverable rendering of the published judge-prompt-v1.md
# reproduces the recorded provenance hash 632b73a34b1a22b1. Per the goal
# amendment ("if that verification fails, STOP and report it -- do not
# reconstruct, paraphrase, or tune anything to force the hash"), this is surfaced
# as a strict-xfail rather than a forced/tuned pass. The test still recomputes
# prompt_hash() over the COMMITTED fixture bytes and compares to the recorded
# value: it neither reconstructs the prompt nor tunes the hash function. If a
# future data correction makes them agree, strict=True turns the surprise pass
# (XPASS) into a failure prompting removal of this marker.

def test_v1_module_hash_recorded_value_is_16_hex():
    """Guard: the recorded v1 provenance value is a well-formed 16-hex digest."""
    assert len(_V1_RECORDED_HASH) == 16
    assert all(c in "0123456789abcdef" for c in _V1_RECORDED_HASH)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SURFACED FINDING (see 2026-08-03-v1-prompt-hash-finding.md in the "
        "vault run folder): the recorded v1 "
        "provenance hash 632b73a34b1a22b1 is not reproducible from the published "
        "judge-prompt-v1.md under the documented sha256[:16] scheme. Not "
        "reconstructing/tuning to force it, per the goal amendment."
    ),
)
def test_v1_fixture_hash_matches_recorded_provenance():
    """VERIFY (not reproduce): prompt_hash(<committed v1 fixture>) == recorded.

    Reads the COMMITTED fixture bytes (fails loudly if absent -- never re-reads
    the external eval-set source), decodes as UTF-8, and hashes with the module's
    single public hash function. Asserts equality with the recorded provenance
    hash. Currently expected-fail per the surfaced finding above.
    """
    assert _V1_PROMPT_FIXTURE.exists(), (
        f"v1 fixture missing: {_V1_PROMPT_FIXTURE} -- cannot verify provenance"
    )
    v1_text = _V1_PROMPT_FIXTURE.read_bytes().decode("utf-8")
    assert cj.prompt_hash(v1_text) == _V1_RECORDED_HASH


# --- c4: the authored v2 prompt (content-match requirement, versioned). ------ #

def test_v2_prompt_exists_nonempty_and_versioned():
    assert isinstance(cj.JUDGE_PROMPT_V2, str) and cj.JUDGE_PROMPT_V2.strip()
    # Carries a version identifier/tag.
    assert cj.JUDGE_PROMPT_V2_VERSION == "v2"
    assert "v2" in cj.JUDGE_PROMPT_V2


def test_v2_prompt_is_distinct_from_v1_fixture_text():
    """v2 is authored, not v1 relabeled: textually distinct from the v1 fixture."""
    v1_text = _V1_PROMPT_FIXTURE.read_bytes().decode("utf-8")
    assert cj.JUDGE_PROMPT_V2 != v1_text


def test_v2_prompt_states_content_match_requirement():
    """v2 must explicitly require substance match, not just topic/shape/keywords."""
    text = cj.JUDGE_PROMPT_V2.lower()
    # Keywords/shape/topic alone are insufficient.
    assert "keyword" in text
    assert "shape" in text
    assert "topic" in text
    # Explicit "not merely / not sufficient" style language.
    assert ("not merely" in text) or ("not sufficient" in text)
    # Substituted specifics are NOT covered.
    assert "substitut" in text  # substitute / substitutes / substituting
    assert "different list members" in text
    assert "different mechanism" in text
    assert ("contradict" in text)


def test_v2_prompt_preserves_topic_adjacency_strictness():
    """v2 keeps v1's core strictness signal (topic adjacency is a trap to reject)."""
    text = cj.JUDGE_PROMPT_V2.lower()
    assert "topic adjacency" in text
    assert "strict" in text


# --- c5: both hashes come from the SAME public function; v2 stable & distinct. #

def test_v2_hash_uses_same_public_function_and_is_stable():
    # Recomputing via the public function reproduces the module constant.
    assert cj.prompt_hash(cj.JUDGE_PROMPT_V2) == cj.JUDGE_PROMPT_V2_HASH
    # Stable across two independent calls.
    assert cj.prompt_hash(cj.JUDGE_PROMPT_V2) == cj.prompt_hash(cj.JUDGE_PROMPT_V2)
    # Well-formed 16-hex digest.
    assert len(cj.JUDGE_PROMPT_V2_HASH) == 16
    assert all(c in "0123456789abcdef" for c in cj.JUDGE_PROMPT_V2_HASH)


def test_v2_hash_is_not_the_v1_recorded_hash():
    """No reproduction target for v2: it must not equal v1's recorded hash."""
    assert cj.JUDGE_PROMPT_V2_HASH != _V1_RECORDED_HASH


def test_v1_and_v2_hashes_produced_by_one_shared_function():
    """Same callable hashes both the v1 fixture bytes and the v2 constant."""
    v1_text = _V1_PROMPT_FIXTURE.read_bytes().decode("utf-8")
    v1_h = cj.prompt_hash(v1_text)
    v2_h = cj.prompt_hash(cj.JUDGE_PROMPT_V2)
    assert v1_h != v2_h  # different inputs, different digests
    # Both are the module's 16-hex shape (single scheme).
    assert len(v1_h) == len(v2_h) == 16


# --------------------------------------------------------------------------- #
# Sprint 1 (this launch): the single-invocation judge + verdict assembly.
#
# These EXTEND the offline suite above (same file, same module). The MODEL CALL
# IS MOCKED in every test — a tiny fake client with a scripted ``messages.create``
# — so no network/LLM call and no Anthropic client is ever constructed. They
# exercise: one call on the happy path with Haiku + temperature 0 + the v2 prompt;
# per-claim verdict values reflecting the mocked reply; fence stripping; the
# truncation (count) defense triggering a retry; malformed JSON exhausting the
# bound; the id-mismatch defenses; the bounded call count; and the stamped
# metadata (judged_at, model, judge_prompt_hash == v2 hash).
# --------------------------------------------------------------------------- #


class _FakeContentBlock:
    """A minimal stand-in for an Anthropic response text content block."""

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    """A minimal stand-in for an Anthropic messages response (``.content``)."""

    def __init__(self, text):
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        # Script may be a list (consumed per call) or a single response text.
        script = self._parent.script
        if isinstance(script, list):
            idx = min(len(self._parent.calls) - 1, len(script) - 1)
            text = script[idx]
        else:
            text = script
        return _FakeResponse(text)


class _FakeClient:
    """A scripted fake Anthropic client: records calls, returns canned text.

    ``script`` is either a single response-text string (returned for every call)
    or a list of response-text strings (the i-th call returns the i-th entry,
    the last entry repeating if there are more calls than scripted responses).
    Every ``messages.create`` kwargs dict is recorded in ``.calls`` so tests can
    assert model/temperature/prompt and the exact call count.
    """

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.messages = _FakeMessages(self)


def _verdicts_json(items):
    """Serialize a list of (claim_id, covered, turns) into the verdict envelope."""
    return json.dumps(
        {"verdicts": [
            {"claim_id": cid, "covered": cov, "turns": turns}
            for cid, cov, turns in items
        ]}
    )


_JUDGE_CLAIMS = _sidecar([
    {"id": "c1", "claim": "Alpha assertion."},
    {"id": "c2", "claim": "Beta assertion."},
    {"id": "c3", "claim": "Gamma assertion."},
])
_JUDGE_TRANSCRIPT = {
    "turns": [
        {"role": "assistant", "content": "explaining alpha"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "explaining gamma"},
    ]
}


# --- c1/c3: happy path — one verdict per claim, values from the mocked reply. #

def test_judge_coverage_well_formed_response_builds_verdict_set():
    reply = _verdicts_json([
        ("c1", True, [0, 2]),
        ("c2", False, []),
        ("c3", True, [2]),
    ])
    client = _FakeClient(reply)
    out = cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)

    verdicts = out["verdicts"]
    # Exactly one entry per input claim id, in input order, no missing/extra/dupe.
    assert [v["claim_id"] for v in verdicts] == ["c1", "c2", "c3"]
    # Per-claim covered/turns reflect the MOCKED reply exactly (not hard-coded).
    by_id = {v["claim_id"]: v for v in verdicts}
    assert by_id["c1"]["covered"] is True and by_id["c1"]["turns"] == [0, 2]
    assert by_id["c2"]["covered"] is False and by_id["c2"]["turns"] == []
    assert by_id["c3"]["covered"] is True and by_id["c3"]["turns"] == [2]
    # Types: covered is a real bool, turns a list of ints.
    for v in verdicts:
        assert isinstance(v["covered"], bool)
        assert isinstance(v["turns"], list)
        assert all(isinstance(t, int) for t in v["turns"])


# --- c2: exactly one call on the happy path; Haiku + temp 0 + v2 prompt sent. #

def test_judge_coverage_issues_exactly_one_call_happy_path():
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert len(client.calls) == 1


def test_judge_coverage_uses_haiku_and_temperature_zero():
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    kwargs = client.calls[0]
    assert "haiku" in kwargs["model"].lower()
    assert kwargs["model"] == cj.JUDGE_MODEL
    assert kwargs["temperature"] == 0


def test_judge_coverage_sends_v2_prompt_not_v1_or_empty():
    """The request must carry the module's v2 prompt (distinctive substance)."""
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    kwargs = client.calls[0]
    # v2's content-match language appears in the request (system + user payload).
    payload = (kwargs.get("system") or "") + json.dumps(kwargs["messages"])
    lowered = payload.lower()
    assert "substitut" in lowered  # v2's fluent-wrongness clause
    assert "not merely" in lowered or "not sufficient" in lowered
    # And the claim ids + transcript turn labels are actually in the user message.
    assert "c1" in payload and "c2" in payload and "c3" in payload
    assert "[0]" in payload and "[2]" in payload
    # A distinctive chunk of the module's v2 prompt text is present verbatim.
    assert "same substance" in lowered


# --- c4: verdict metadata — judged_at, model, judge_prompt_hash == v2 hash. --- #

def test_judge_coverage_stamps_metadata_and_v2_hash():
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    out = cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert out["model"] == cj.JUDGE_MODEL
    assert out["judge_prompt_hash"] == cj.JUDGE_PROMPT_V2_HASH
    # The v2 hash is distinct from the recorded v1 provenance hash.
    assert out["judge_prompt_hash"] != _V1_RECORDED_HASH
    # judged_at is present and parseable as an ISO-8601 timestamp.
    from datetime import datetime
    assert isinstance(out["judged_at"], str) and out["judged_at"]
    datetime.fromisoformat(out["judged_at"])  # raises if not ISO-8601


def test_judge_coverage_respects_config_model_in_metadata_and_call():
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    cfg = cj.JudgeConfig(model="claude-haiku-4-5-20251001")
    out = cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, config=cfg, client=client)
    assert out["model"] == cfg.model
    assert client.calls[0]["model"] == cfg.model


# --- c5: fenced output is stripped and parses to the correct verdict set. ----- #

def test_judge_coverage_strips_markdown_fenced_output():
    inner = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", True, [2])])
    fenced = "```json\n" + inner + "\n```"
    client = _FakeClient(fenced)
    out = cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert len(client.calls) == 1
    by_id = {v["claim_id"]: v for v in out["verdicts"]}
    assert set(by_id) == {"c1", "c2", "c3"}
    assert by_id["c1"]["covered"] is True and by_id["c3"]["turns"] == [2]


# --- c6: truncation (short verdict list) triggers detection, then retry. ------ #

def test_judge_coverage_truncated_list_triggers_retry_then_succeeds():
    """First reply is short (count < claims) -> VerdictCountError -> retry."""
    short = _verdicts_json([("c1", True, [0])])  # only 1 of 3 claims
    full = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", True, [2])])
    client = _FakeClient([short, full])
    out = cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    # Called twice: the short reply retried, the second (full) reply accepted.
    assert len(client.calls) == 2
    assert [v["claim_id"] for v in out["verdicts"]] == ["c1", "c2", "c3"]


def test_short_list_surfaces_count_error_through_seeded_parse_core():
    """The short-list case routes through the seeded parse_verdicts (VerdictCountError).

    Verifies the retry path reuses the Sprint-0 defense rather than an ad-hoc
    length check: the same short payload fed straight to parse_verdicts_text
    raises the seeded VerdictCountError.
    """
    short = _verdicts_json([("c1", True, [0])])
    with pytest.raises(cj.VerdictCountError):
        cj.parse_verdicts_text(short, ["c1", "c2", "c3"])


# --- c7: malformed JSON exhausts the bound, then raises a typed parse error. --- #

def test_judge_coverage_malformed_json_retries_then_raises_typed():
    client = _FakeClient("this is not json at all")  # malformed on every call
    with pytest.raises(cj.VerdictParseError) as ei:
        cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    # A typed error from the seeded hierarchy (specifically a JSON error here).
    assert isinstance(ei.value, cj.VerdictJSONError)
    # Bounded: initial + retries == the documented bound, never unbounded.
    assert len(client.calls) == cj.MAX_JUDGE_ATTEMPTS


def test_judge_coverage_recovers_when_retry_is_well_formed():
    """Malformed first reply, well-formed second -> success within the bound."""
    good = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(["{ not json", good])
    out = cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert len(client.calls) == 2
    assert [v["claim_id"] for v in out["verdicts"]] == ["c1", "c2", "c3"]


# --- c8: missing / duplicate / unknown claim ids surface as typed errors. ----- #

def test_judge_coverage_missing_claim_id_raises_typed():
    """Count-correct reply that omits c3 and repeats c2 -> a typed id error.

    (A same-count reply missing one expected id must contain a duplicate/unknown,
    so it trips the id-level defenses, not the count gate.)
    """
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c2", True, [1])])
    client = _FakeClient(reply)
    with pytest.raises(cj.VerdictParseError) as ei:
        cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert isinstance(ei.value, cj.DuplicateClaimIdError)


def test_judge_coverage_unknown_claim_id_raises_typed():
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c99", True, [1])])
    client = _FakeClient(reply)
    with pytest.raises(cj.VerdictParseError) as ei:
        cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert isinstance(ei.value, cj.UnknownClaimIdError)


def test_judge_coverage_bad_shape_covered_raises_typed():
    """A non-bool covered value is caught by the seeded shape defense."""
    reply = json.dumps({"verdicts": [
        {"claim_id": "c1", "covered": 1, "turns": [0]},
        {"claim_id": "c2", "covered": False, "turns": []},
        {"claim_id": "c3", "covered": False, "turns": []},
    ]})
    client = _FakeClient(reply)
    with pytest.raises(cj.VerdictShapeError):
        cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)


# --- c9: the retry is bounded even under repeated malformed output. ----------- #

def test_judge_coverage_bounded_call_count_under_persistent_malformed():
    client = _FakeClient("nope")  # every attempt malformed
    with pytest.raises(cj.VerdictParseError):
        cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert client.calls  # at least one call
    assert len(client.calls) == cj.MAX_JUDGE_ATTEMPTS
    assert cj.MAX_JUDGE_ATTEMPTS >= 2  # initial + at least one bounded retry


def test_empty_text_response_treated_as_malformed_and_retried():
    """A response with no text content is malformed output (retried, then raised)."""
    client = _FakeClient("")  # blank text on every call
    with pytest.raises(cj.VerdictParseError):
        cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    assert len(client.calls) == cj.MAX_JUDGE_ATTEMPTS


# --- c1: input validation still fires BEFORE any model call. ------------------ #

def test_judge_coverage_bad_input_raises_before_any_call():
    client = _FakeClient(_verdicts_json([("c1", True, [0])]))
    with pytest.raises(cj.CoverageInputError):
        cj.judge_coverage(_sidecar([]), _JUDGE_TRANSCRIPT, client=client)  # empty claims
    assert client.calls == []  # no model call was issued


# --------------------------------------------------------------------------- #
# Sprint 2 (this launch): union_coverage (pure merge) + the standalone CLI.
#
# These EXTEND the offline suite above (same file, same module). union_coverage
# is pure (no I/O, no model call); the CLI tests MOCK the model call via the same
# injected fake client and run against temp files, so the suite stays fully
# hermetic — no network, no Anthropic client, and (asserted below) no live app
# import.
# --------------------------------------------------------------------------- #


def _verdict_set(items, *, model="m", prompt_hash="h"):
    """Build a full judge-shaped verdict OBJECT (as judge_coverage returns)."""
    return {
        "verdicts": [
            {"claim_id": cid, "covered": cov, "turns": turns}
            for cid, cov, turns in items
        ],
        "judged_at": "2026-08-03T00:00:00+00:00",
        "model": model,
        "judge_prompt_hash": prompt_hash,
    }


# --- c1: union_coverage composes with the ACTUAL judge_coverage output shape. - #

def test_union_coverage_accepts_real_judge_output_shape():
    """union_coverage merges the exact object judge_coverage returns."""
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", True, [2])])
    client = _FakeClient(reply)
    judge_out = cj.judge_coverage(_JUDGE_CLAIMS, _JUDGE_TRANSCRIPT, client=client)
    # Feed the judge's own output object straight into union_coverage.
    result = cj.union_coverage([judge_out])
    assert isinstance(result, dict)
    assert isinstance(result["covered_ids"], list)
    assert "percentage" in result
    assert set(result["covered_ids"]) == {"c1", "c3"}


def test_union_coverage_is_pure_no_io_no_network(monkeypatch):
    """Guard: calling union_coverage constructs no client / opens no file.

    Poison ``open`` and any Anthropic client so a stray I/O or model call would
    raise; a clean return proves purity.
    """
    import builtins

    def _no_open(*a, **k):  # pragma: no cover - only fires on regression
        raise AssertionError("union_coverage performed file I/O")

    monkeypatch.setattr(builtins, "open", _no_open)
    result = cj.union_coverage([_verdict_set([("c1", True, [0])])])
    assert result["covered_ids"] == ["c1"]


# --- c2: union merges by UNION across sessions (overlap + disjoint). ---------- #

def test_union_coverage_unions_overlapping_and_disjoint_ids():
    a = _verdict_set([("c1", True, [0]), ("c2", True, [1]), ("c3", False, [])])
    b = _verdict_set([("c2", True, [4]), ("c3", True, [5]), ("c1", False, [])])
    result = cj.union_coverage([a, b], allow_unidentified=True)
    # c1 covered only in A, c3 only in B, c2 in both -> union of all three.
    assert set(result["covered_ids"]) == {"c1", "c2", "c3"}


def test_union_coverage_covered_ids_deduped_and_order_insensitive():
    a = _verdict_set([("c1", True, [0]), ("c2", True, [1])])
    b = _verdict_set([("c1", True, [3]), ("c2", True, [4])])
    result = cj.union_coverage([a, b], allow_unidentified=True)
    ids = result["covered_ids"]
    assert sorted(ids) == ["c1", "c2"]
    assert len(ids) == len(set(ids))  # no duplicates despite repeats across sets


def test_union_coverage_claim_covered_in_any_session_counts():
    """A claim not-covered in one session but covered in another is in the union."""
    a = _verdict_set([("c1", False, []), ("c2", False, [])])
    b = _verdict_set([("c1", True, [7]), ("c2", False, [])])
    result = cj.union_coverage([a, b], allow_unidentified=True)
    assert result["covered_ids"] == ["c1"]


# --- c3: percentage is derived at read time, over ALL distinct judged ids. ---- #

def test_union_coverage_percentage_terminating_75_percent():
    """3 of 4 distinct judged claims covered -> exactly 75.0.

    The denominator counts every distinct judged id (covered + not-covered),
    not just covered ones.
    """
    a = _verdict_set([
        ("c1", True, [0]), ("c2", True, [1]), ("c3", True, [2]), ("c4", False, []),
    ])
    result = cj.union_coverage([a])
    assert result["percentage"] == 75.0


def test_union_coverage_percentage_nonterminating_rounds_by_convention():
    """2 of 3 distinct covered (66.666...) -> the module's fixed rounded value."""
    a = _verdict_set([("c1", True, [0]), ("c2", True, [1]), ("c3", False, [])])
    result = cj.union_coverage([a])
    expected = round(100.0 * 2 / 3, cj._COVERAGE_PERCENTAGE_DECIMALS)
    assert result["percentage"] == expected
    # And the convention actually rounds (not left as the raw repeating float).
    assert result["percentage"] != 100.0 * 2 / 3


def test_union_coverage_denominator_counts_notcovered_across_sessions():
    """A claim judged not-covered in every session still enlarges the universe."""
    a = _verdict_set([("c1", True, [0]), ("c2", False, [])])
    b = _verdict_set([("c3", False, [])])
    # covered = {c1}; universe = {c1, c2, c3} -> 1/3.
    result = cj.union_coverage([a, b], allow_unidentified=True)
    assert set(result["covered_ids"]) == {"c1"}
    assert result["percentage"] == round(100.0 / 3, cj._COVERAGE_PERCENTAGE_DECIMALS)


# --- c4: edge cases — empty + single-session. -------------------------------- #

def test_union_coverage_empty_input_no_divide_by_zero():
    assert cj.union_coverage([]) == {"covered_ids": [], "percentage": 0.0}


def test_union_coverage_empty_verdict_lists_yield_zero():
    a = _verdict_set([])
    b = {"verdicts": []}
    result = cj.union_coverage([a, b], allow_unidentified=True)
    assert result["covered_ids"] == []
    assert result["percentage"] == 0.0


def test_union_coverage_single_session_reproduces_its_own_result():
    a = _verdict_set([("c1", True, [0]), ("c2", False, []), ("c3", True, [2])])
    result = cj.union_coverage([a])
    assert set(result["covered_ids"]) == {"c1", "c3"}
    # 2 of 3 distinct judged -> the module's rounded value.
    assert result["percentage"] == round(
        100.0 * 2 / 3, cj._COVERAGE_PERCENTAGE_DECIMALS
    )


def test_union_coverage_accepts_bare_verdict_list_shape():
    """A bare per-claim verdict list (not the full object) also merges."""
    bare = [
        {"claim_id": "c1", "covered": True, "turns": [0]},
        {"claim_id": "c2", "covered": False, "turns": []},
    ]
    result = cj.union_coverage([bare])
    assert result["covered_ids"] == ["c1"]
    assert result["percentage"] == 50.0


# --------------------------------------------------------------------------- #
# CLI: end-to-end wiring with the model call MOCKED. Uses tmp_path files.
# --------------------------------------------------------------------------- #


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


# --- c5: CLI reads inputs, runs ONE judge call, writes a valid --out. -------- #

def test_cli_end_to_end_writes_out_matching_mocked_reply(tmp_path):
    reply = _verdicts_json([("c1", True, [0, 2]), ("c2", False, []), ("c3", True, [2])])
    client = _FakeClient(reply)
    claims_path = _write(tmp_path, "claims.json", _JUDGE_CLAIMS)
    transcript_path = _write(tmp_path, "transcript.json", _JUDGE_TRANSCRIPT)
    out_path = str(tmp_path / "out.json")

    rc = cj.main(
        ["--claims", claims_path, "--transcript", transcript_path, "--out", out_path],
        client=client,
    )
    assert rc == 0
    # EXACTLY one model call on the happy path.
    assert client.calls and len(client.calls) == 1

    written = json.loads(Path(out_path).read_text())
    verdicts = written["verdicts"]
    # One verdict per input claim, in input order.
    assert [v["claim_id"] for v in verdicts] == ["c1", "c2", "c3"]
    # Each verdict's covered/turns EQUAL the mocked reply.
    by_id = {v["claim_id"]: v for v in verdicts}
    assert by_id["c1"]["covered"] is True and by_id["c1"]["turns"] == [0, 2]
    assert by_id["c2"]["covered"] is False and by_id["c2"]["turns"] == []
    assert by_id["c3"]["covered"] is True and by_id["c3"]["turns"] == [2]
    # Metadata hash equals the module's computed hash for the default (v2) prompt.
    assert written["judge_prompt_hash"] == cj.JUDGE_PROMPT_V2_HASH


# --- c6: --model / --prompt-version thread into metadata; --cost-out JSON. ---- #

def test_cli_model_and_prompt_version_thread_into_metadata(tmp_path):
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    claims_path = _write(tmp_path, "claims.json", _JUDGE_CLAIMS)
    transcript_path = _write(tmp_path, "transcript.json", _JUDGE_TRANSCRIPT)
    out_path = str(tmp_path / "out.json")

    rc = cj.main(
        [
            "--claims", claims_path,
            "--transcript", transcript_path,
            "--out", out_path,
            "--model", "claude-haiku-4-5-20251001",
            "--prompt-version", "v2",
        ],
        client=client,
    )
    assert rc == 0
    written = json.loads(Path(out_path).read_text())
    # --model threads into metadata AND the actual call.
    assert written["model"] == "claude-haiku-4-5-20251001"
    assert client.calls[0]["model"] == "claude-haiku-4-5-20251001"
    # --prompt-version v2 -> the module's hash of the v2 prompt.
    assert written["judge_prompt_hash"] == cj.JUDGE_PROMPT_V2_HASH


def test_cli_cost_out_writes_valid_cost_json(tmp_path):
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    claims_path = _write(tmp_path, "claims.json", _JUDGE_CLAIMS)
    transcript_path = _write(tmp_path, "transcript.json", _JUDGE_TRANSCRIPT)
    out_path = str(tmp_path / "out.json")
    cost_path = str(tmp_path / "cost.json")

    rc = cj.main(
        [
            "--claims", claims_path,
            "--transcript", transcript_path,
            "--out", out_path,
            "--cost-out", cost_path,
        ],
        client=client,
    )
    assert rc == 0
    cost = json.loads(Path(cost_path).read_text())
    assert isinstance(cost, dict) and cost  # a bare {} must FAIL
    # The model used is recorded.
    assert cost["model"] == cj.JUDGE_MODEL
    # At least one concrete numeric cost-accounting field of the right type.
    numeric_fields = {
        k: v for k, v in cost.items()
        if k in {"cost", "cost_usd", "total_cost", "calls", "num_calls",
                 "input_tokens", "output_tokens"}
    }
    assert numeric_fields, f"no numeric cost field in {cost}"
    assert all(isinstance(v, (int, float)) and v >= 0 for v in numeric_fields.values())
    # Specifically, one model call was accounted for.
    assert cost["calls"] == 1


def test_cli_unknown_prompt_version_rejected_no_out_written(tmp_path):
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    claims_path = _write(tmp_path, "claims.json", _JUDGE_CLAIMS)
    transcript_path = _write(tmp_path, "transcript.json", _JUDGE_TRANSCRIPT)
    out_path = str(tmp_path / "out.json")

    rc = cj.main(
        [
            "--claims", claims_path,
            "--transcript", transcript_path,
            "--out", out_path,
            "--prompt-version", "v1",  # not a known module prompt
        ],
        client=client,
    )
    assert rc != 0
    # No bogus verdict written, and no model call issued.
    assert not Path(out_path).exists()
    assert client.calls == []


def test_cli_no_live_app_wiring(tmp_path):
    """Deterministic guard: no bot/server import, no live-ledger import in source.

    (a) After a full CLI invocation with the judge mocked, 'bot' and 'server'
    are absent from sys.modules; (b) the module source imports neither bot nor
    server (nor a live usage-ledger) anywhere.
    """
    import sys

    # Ensure a clean baseline, then run the CLI end to end (mocked).
    sys.modules.pop("bot", None)
    sys.modules.pop("server", None)
    reply = _verdicts_json([("c1", True, [0]), ("c2", False, []), ("c3", False, [])])
    client = _FakeClient(reply)
    claims_path = _write(tmp_path, "claims.json", _JUDGE_CLAIMS)
    transcript_path = _write(tmp_path, "transcript.json", _JUDGE_TRANSCRIPT)
    out_path = str(tmp_path / "out.json")
    cj.main(
        ["--claims", claims_path, "--transcript", transcript_path, "--out", out_path],
        client=client,
    )
    assert "bot" not in sys.modules
    assert "server" not in sys.modules

    # Source-level: no import of bot/server (nor a live ledger) anywhere.
    src = _MODULE_PATH.read_text()
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    for forbidden in ("bot", "server", "usage_ledger"):
        assert forbidden not in imported, f"coverage_judge imports {forbidden}"


# --- c7: invalid input -> non-zero exit, no bogus --out. --------------------- #

def test_cli_missing_input_file_nonzero_exit_no_out(tmp_path):
    client = _FakeClient(_verdicts_json([("c1", True, [0])]))
    out_path = str(tmp_path / "out.json")
    rc = cj.main(
        [
            "--claims", str(tmp_path / "does_not_exist.json"),
            "--transcript", str(tmp_path / "also_missing.json"),
            "--out", out_path,
        ],
        client=client,
    )
    assert rc != 0
    assert not Path(out_path).exists()
    assert client.calls == []  # judge never reached


def test_cli_malformed_claims_json_nonzero_exit_no_out(tmp_path):
    client = _FakeClient(_verdicts_json([("c1", True, [0])]))
    bad_claims = tmp_path / "claims.json"
    bad_claims.write_text("{ this is not valid json", encoding="utf-8")
    transcript_path = _write(tmp_path, "transcript.json", _JUDGE_TRANSCRIPT)
    out_path = str(tmp_path / "out.json")
    rc = cj.main(
        [
            "--claims", str(bad_claims),
            "--transcript", transcript_path,
            "--out", out_path,
        ],
        client=client,
    )
    assert rc != 0
    assert not Path(out_path).exists()
    assert client.calls == []


def test_cli_empty_claim_list_nonzero_exit_no_out(tmp_path):
    """A structurally-valid-but-empty claim list is rejected before any call."""
    client = _FakeClient(_verdicts_json([("c1", True, [0])]))
    claims_path = _write(tmp_path, "claims.json", _sidecar([]))  # empty claims
    transcript_path = _write(tmp_path, "transcript.json", _JUDGE_TRANSCRIPT)
    out_path = str(tmp_path / "out.json")
    rc = cj.main(
        ["--claims", claims_path, "--transcript", transcript_path, "--out", out_path],
        client=client,
    )
    assert rc != 0
    assert not Path(out_path).exists()
    assert client.calls == []


def test_cli_runnable_as_module_subprocess(tmp_path):
    """`python -m coverage_judge` is runnable; --help exits 0 with usage text.

    Uses --help so no model call is needed (fully offline). Proves the module
    exposes a real ``python -m coverage_judge`` entrypoint end to end.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "coverage_judge", "--help"],
        cwd=str(_MODULE_PATH.parent),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--claims" in proc.stdout
    assert "--transcript" in proc.stdout
    assert "--out" in proc.stdout


# --------------------------------------------------------------------------- #
# Sprint 3 (this launch): the credentialed-smoke fixtures are present, loadable,
# and count-pinned; labels.json is treated as a frozen READ-ONLY answer key; and
# the bounded credentialed-smoke script is a SEPARATE module that this hermetic
# suite never imports or executes (so `pytest tests/test_coverage_judge.py`
# stays fully offline — no real Haiku call is ever made here). The live 17/17
# label-agreement acceptance is graded from the smoke report produced by the
# one-time credentialed run (kept in the vault run folder as
# 2026-08-03-credentialed-smoke-report.md), NOT from this file.
# --------------------------------------------------------------------------- #

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "coverage"
_TRANSCRIPT_DIR = _FIXTURE_ROOT / "transcripts"
_CLAIMS_DIR = _FIXTURE_ROOT / "claims"
_LABELS_FIXTURE = _FIXTURE_ROOT / "labels.json"

# The four PRIMARY union sessions + the strictness session, with the turn count
# baked in as a recorded constant so a truncated/corrupt copy is caught here
# (c1). These are the resolved fixtures copied from ~/.voice-tutor at launch.
_PRIMARY_SESSIONS = {
    "f6148c26": 159,
    "7beee170": 100,
    "d33800bf": 5,
    "bb979045": 3,
}
_STRICTNESS_SESSION = {"12f3a30d": 11}
_ALL_SESSION_TURN_COUNTS = {**_PRIMARY_SESSIONS, **_STRICTNESS_SESSION}

# The two claim-map sidecars, with their pinned claim counts (c2).
_CLAIM_MAPS = {
    "matt-2aa66acc": 63,
    "shared-ac4b826f": 71,
}


# --- c1: the five session transcripts are present, valid, and count-pinned. --- #

@pytest.mark.parametrize("session,expected_turns", sorted(_ALL_SESSION_TURN_COUNTS.items()))
def test_smoke_transcript_fixture_present_valid_and_count_pinned(session, expected_turns):
    """Each of the five frozen transcripts loads to its recorded turn count.

    Verifies presence + valid JSON + a NON-EMPTY indexed transcript through the
    module's own loader, and pins the recorded turn count so a truncated/corrupt
    copy is caught. Byte-identity with the external source is NOT required.
    """
    path = _TRANSCRIPT_DIR / f"{session}.json"
    assert path.exists(), f"missing transcript fixture: {path}"
    data = json.loads(path.read_text())
    turns = cj.load_transcript(data)
    assert turns, f"{session} loaded to an empty transcript"
    assert len(turns) == expected_turns, (
        f"{session}: loaded {len(turns)} turns, recorded {expected_turns} "
        "(truncated/corrupt fixture?)"
    )


def test_primary_and_strictness_sessions_are_the_five_named():
    """The fixture set is exactly the four PRIMARY + one strictness session."""
    on_disk = {p.stem for p in _TRANSCRIPT_DIR.glob("*.json")}
    assert set(_PRIMARY_SESSIONS) <= on_disk
    assert set(_STRICTNESS_SESSION) <= on_disk


# --- c2: both claim-map sidecars are present with pinned counts + unique ids. - #

@pytest.mark.parametrize("stem,expected_count", sorted(_CLAIM_MAPS.items()))
def test_smoke_claim_map_fixture_present_valid_and_count_pinned(stem, expected_count):
    """Each claim-map sidecar loads to its pinned count with unique non-blank ids."""
    path = _CLAIMS_DIR / f"{stem}.claims.json"
    assert path.exists(), f"missing claim-map fixture: {path}"
    claims = cj.load_claims(json.loads(path.read_text()))
    assert len(claims) == expected_count, (
        f"{stem}: loaded {len(claims)} claims, recorded {expected_count}"
    )
    ids = [c.id for c in claims]
    assert all(cid.strip() for cid in ids), f"{stem} has a blank claim id"
    assert len(set(ids)) == len(ids), f"{stem} has duplicate claim ids"


# --- c3: labels.json is the frozen answer key — present, and never written. --- #

def test_labels_fixture_present_and_is_the_frozen_answer_key():
    """labels.json is committed and carries the expected upheld/c30 structure.

    This is a READ-ONLY provenance check on the frozen answer key. It confirms
    the 16 upheld ids are labelled ``covered`` and c30 is labelled
    ``not_covered`` — so the smoke's acceptance bar is graded against the real
    key, not a mutated one. It does NOT modify the file.
    """
    assert _LABELS_FIXTURE.exists(), f"missing labels fixture: {_LABELS_FIXTURE}"
    labels = json.loads(_LABELS_FIXTURE.read_text())
    by_id = {row["claim_id"]: row["matt_verdict"] for row in labels["labels"]}
    upheld = {
        "c1", "c2", "c3", "c5", "c6", "c8", "c9", "c15", "c17", "c21",
        "c27", "c28", "c31", "c46", "c47", "c48",
    }
    for cid in upheld:
        assert by_id.get(cid) == "covered", f"{cid} not labelled covered in answer key"
    assert by_id.get("c30") == "not_covered", "c30 must be the not-covered regression case"


def test_no_source_file_writes_to_any_labels_json():
    """No module/smoke/test source opens any labels.json for writing (c3).

    Statically scans the module, the smoke script, and this test file for a
    write-mode ``open(... labels.json ...)`` or a ``json.dump(... labels`` sink,
    so the frozen answer key can never be re-judged, edited, or regenerated by
    this sprint's code. (A pure read of labels.json is fine and expected.)
    """
    import re

    repo_root = _MODULE_PATH.parent
    candidates = [
        _MODULE_PATH,
        Path(__file__),
        repo_root / "coverage_smoke.py",
    ]
    # Match an open() in any write/append/x mode whose target path mentions labels.
    write_open = re.compile(
        r"open\([^)]*labels[^)]*['\"][wax]\+?b?['\"]", re.IGNORECASE
    )
    for path in candidates:
        if not path.exists():
            continue
        src = path.read_text()
        assert not write_open.search(src), (
            f"{path.name} appears to open a labels.json path for writing"
        )


# --- c5 / c13: the smoke script is a SEPARATE module, never run by the suite. -- #

def test_hermetic_suite_does_not_import_the_smoke_script():
    """This hermetic file must not import the credentialed smoke module.

    The smoke (real Haiku calls) is a bounded, one-time acceptance step that must
    live OUTSIDE this verifier file and never run per hermetic test. Parse this
    file's own AST and assert ``coverage_smoke`` is not imported anywhere, so
    ``pytest tests/test_coverage_judge.py`` stays fully offline. (An AST scan
    avoids false positives from the module name merely appearing inside a string
    literal in an assertion here.)
    """
    tree = ast.parse(Path(__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "coverage_smoke" not in imported


def test_smoke_script_exists_and_is_offline_free_of_hermetic_coupling():
    """The smoke script exists as its own module and isn't imported at test time.

    Confirms ``coverage_smoke.py`` is present (the separate, bounded credentialed
    step) and that it is NOT importable side-effect-heavy at collection: this
    test does not import it, only asserts its presence and that it reads the key
    from the environment / the app .env absolute path (never a worktree ./.env).
    """
    smoke = _MODULE_PATH.parent / "coverage_smoke.py"
    assert smoke.exists(), "missing coverage_smoke.py (the bounded credentialed smoke)"
    src = smoke.read_text()
    # Key comes from the process env or the app .env ABSOLUTE path.
    assert "development/voice-tutor/.env" in src
    # It must NOT load a worktree-relative ./.env: reject an actual open()/Path()
    # of a relative ".env" (prose mentioning "./.env" in a docstring is fine, so
    # match only a code construction of it).
    for bad in ('open("./.env"', "open('./.env'", 'open(".env"', "open('.env'",
                'Path("./.env"', "Path('./.env'", 'Path(".env"', "Path('.env'"):
        assert bad not in src, f"smoke script constructs a worktree-relative env path: {bad}"
    # The key value is never echoed/logged/printed.
    assert "print(api_key" not in src
    assert "print(key" not in src


# --------------------------------------------------------------------------- #
# Review fixes — findings from the independent review of run msdsrc8x.
#
# Three of these guard SILENT WRONGNESS (a plausible-looking coverage number that
# is wrong), which the pre-review module could produce without raising:
#   * a verdict citing a turn index that does not exist, or claiming coverage
#     with no citation at all -> inflated coverage;
#   * merging sessions judged against DIFFERENT documents, whose per-document
#     claim ids (c1..c63 / c1..c71) collide -> false-positive coverage;
#   * a max_tokens-truncated reply diagnosed as generic bad JSON and retried at
#     temperature 0 (deterministic, so the retry re-fails identically).
# --------------------------------------------------------------------------- #


class _FakeUsage:
    """Stand-in for an Anthropic response ``usage`` block."""

    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _RichResponse:
    """A fake response that can also carry ``stop_reason`` and ``usage``.

    Separate from _FakeResponse so the existing suite's responses keep having NO
    stop_reason attribute (the getattr default path).
    """

    def __init__(self, text, stop_reason=None, input_tokens=None, output_tokens=None):
        self.content = [_FakeContentBlock(text)]
        if stop_reason is not None:
            self.stop_reason = stop_reason
        if input_tokens is not None or output_tokens is not None:
            self.usage = _FakeUsage(input_tokens, output_tokens)


class _ScriptedMessages:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        idx = min(len(self._parent.calls) - 1, len(self._parent.responses) - 1)
        return self._parent.responses[idx]


class _ScriptedClient:
    """Fake client returning pre-built response OBJECTS in order (last repeats)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = _ScriptedMessages(self)


_RF_CLAIMS = {
    "source_hash": "doc-A-hash",
    "claims": [{"id": "c1", "claim": "First claim."}, {"id": "c2", "claim": "Second claim."}],
}
_RF_TRANSCRIPT = [
    {"index": 0, "role": "user", "content": "hello"},
    {"index": 1, "role": "assistant", "content": "world"},
]
_RF_GOOD = json.dumps(
    {"verdicts": [
        {"claim_id": "c1", "covered": True, "turns": [1], "reason": "turn 1 states it"},
        {"claim_id": "c2", "covered": False, "turns": [], "reason": "never mentioned"},
    ]}
)


def _rf_files(tmp_path):
    """Write the claims/transcript inputs and return (claims_path, transcript_path)."""
    cp = tmp_path / "claims.json"
    tp = tmp_path / "transcript.json"
    cp.write_text(json.dumps(_RF_CLAIMS), encoding="utf-8")
    tp.write_text(json.dumps(_RF_TRANSCRIPT), encoding="utf-8")
    return str(cp), str(tp)


# --- Finding: citations are never validated against the real transcript ------ #

def test_verdict_citing_a_nonexistent_turn_is_rejected():
    payload = _verdicts_json([("c1", True, [999]), ("c2", False, [])])
    client = _ScriptedClient([_RichResponse(payload)])
    with pytest.raises(cj.InvalidTurnCitationError):
        cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)


def test_covered_true_with_no_citation_is_rejected():
    # The v2 prompt's contract: "No citable turn => covered: false and turns: []".
    payload = _verdicts_json([("c1", True, []), ("c2", False, [])])
    client = _ScriptedClient([_RichResponse(payload)])
    with pytest.raises(cj.InvalidTurnCitationError):
        cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)


def test_valid_citations_are_accepted():
    client = _ScriptedClient([_RichResponse(_RF_GOOD)])
    verdict = cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    assert verdict["verdicts"][0]["turns"] == [1]
    assert verdict["verdicts"][1]["turns"] == []


def test_citation_validation_is_a_retryable_parse_error():
    # A bad citation is model error, not operator error: it must retry within the
    # bound and then surface the typed diagnosis.
    bad = _verdicts_json([("c1", True, [42]), ("c2", False, [])])
    client = _ScriptedClient([_RichResponse(bad)])
    with pytest.raises(cj.VerdictParseError):
        cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    assert len(client.calls) == cj.MAX_JUDGE_ATTEMPTS


# --- Finding: max_tokens truncation undetected, then pointlessly retried ----- #

def test_max_tokens_truncation_is_detected_and_NOT_retried():
    truncated = _RichResponse(
        '{"verdicts":[{"claim_id":"c1","covered":true,"tur',
        stop_reason="max_tokens",
    )
    client = _ScriptedClient([truncated, truncated])
    with pytest.raises(cj.VerdictTruncatedError):
        cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    # Deterministic failure: retrying the identical prompt at temperature 0 would
    # truncate identically, so exactly ONE call must be made.
    assert len(client.calls) == 1


def test_truncation_escapes_the_retryable_except_clause():
    """Truncation must not be catchable as a retryable parse error.

    Supersedes an earlier assertion that VerdictTruncatedError WAS a
    VerdictParseError subclass. That was the wrong contract: the retryable base
    class is exactly what the retry loop catches, so membership in it made the
    "never retry a deterministic truncation" guarantee depend on raise-site
    placement. This pins the BEHAVIOUR (the sibling test pins the taxonomy):
    catching the retryable family does not catch truncation.
    """
    truncated = _RichResponse('{"verdicts":[{"cl', stop_reason="max_tokens")
    client = _ScriptedClient([truncated, truncated])
    caught_as_retryable = False
    try:
        cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    except cj.VerdictParseError:
        caught_as_retryable = True
    except cj.VerdictTruncatedError:
        pass
    assert not caught_as_retryable
    assert len(client.calls) == 1


# --- Finding: the judge's per-claim reasoning was required then discarded ---- #

def test_reason_is_persisted_into_the_verdict_output():
    client = _ScriptedClient([_RichResponse(_RF_GOOD)])
    verdict = cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    assert verdict["verdicts"][0]["reason"] == "turn 1 states it"
    assert verdict["verdicts"][1]["reason"] == "never mentioned"


def test_reason_is_optional_and_absent_stays_none():
    payload = _verdicts_json([("c1", True, [1]), ("c2", False, [])])
    client = _ScriptedClient([_RichResponse(payload)])
    verdict = cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    assert verdict["verdicts"][0]["reason"] is None


# --- Finding: cross-document merges silently miscount (claim ids collide) ---- #

def test_verdict_object_carries_doc_id_from_the_claims_source_hash():
    client = _ScriptedClient([_RichResponse(_RF_GOOD)])
    verdict = cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    assert verdict["doc_id"] == "doc-A-hash"


def test_union_coverage_refuses_a_cross_document_merge():
    a = {"doc_id": "A", "verdicts": [{"claim_id": "c15", "covered": False, "turns": []}]}
    b = {"doc_id": "B", "verdicts": [{"claim_id": "c15", "covered": True, "turns": [0]}]}
    with pytest.raises(cj.CoverageInputError):
        cj.union_coverage([a, b])


def test_union_coverage_allows_same_document_merge():
    a = {"doc_id": "A", "verdicts": [{"claim_id": "c1", "covered": True, "turns": [0]}]}
    b = {"doc_id": "A", "verdicts": [{"claim_id": "c2", "covered": False, "turns": []}]}
    result = cj.union_coverage([a, b])
    assert result["covered_ids"] == ["c1"]
    assert result["percentage"] == 50.0


def test_union_coverage_refuses_mixed_labelled_and_unlabelled_sets():
    # Cannot PROVE these came from the same document, so refuse rather than guess.
    a = {"doc_id": "A", "verdicts": [{"claim_id": "c1", "covered": True, "turns": [0]}]}
    b = [{"claim_id": "c2", "covered": False, "turns": []}]
    with pytest.raises(cj.CoverageInputError):
        cj.union_coverage([a, b])


# --- Finding: CLI crashed with a raw traceback on model/API failure ---------- #

def test_cli_reports_a_model_failure_cleanly_without_a_traceback(tmp_path, capsys):
    cp, tp = _rf_files(tmp_path)
    client = _ScriptedClient([_RichResponse("not json at all")])
    rc = cj.main(
        ["--claims", cp, "--transcript", tp, "--out", str(tmp_path / "out.json")],
        client=client,
    )
    assert rc == 2
    assert "coverage_judge: error:" in capsys.readouterr().err


def test_cli_reports_an_api_failure_cleanly_without_a_traceback(tmp_path, capsys):
    cp, tp = _rf_files(tmp_path)

    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("api exploded (429)")

    class _BoomClient:
        messages = _BoomMessages()

    rc = cj.main(
        ["--claims", cp, "--transcript", tp, "--out", str(tmp_path / "out.json")],
        client=_BoomClient(),
    )
    assert rc == 2
    assert "coverage_judge: error:" in capsys.readouterr().err


# --- Finding: a paid call was made before checking --out was writable -------- #

def test_unwritable_out_path_is_checked_BEFORE_the_paid_model_call(tmp_path, capsys):
    cp, tp = _rf_files(tmp_path)
    client = _ScriptedClient([_RichResponse(_RF_GOOD)])
    rc = cj.main(
        ["--claims", cp, "--transcript", tp,
         "--out", str(tmp_path / "no-such-dir" / "out.json")],
        client=client,
    )
    assert rc == 2
    assert client.calls == [], "no model call may fire when --out is unwritable"


def test_unwritable_cost_out_path_is_also_checked_before_the_call(tmp_path):
    cp, tp = _rf_files(tmp_path)
    client = _ScriptedClient([_RichResponse(_RF_GOOD)])
    rc = cj.main(
        ["--claims", cp, "--transcript", tp,
         "--out", str(tmp_path / "out.json"),
         "--cost-out", str(tmp_path / "no-such-dir" / "cost.json")],
        client=client,
    )
    assert rc == 2
    assert client.calls == []


# --- Finding: cost accounting kept only the LAST attempt's tokens ------------ #

def test_cost_record_accumulates_tokens_across_a_retry(tmp_path):
    cp, tp = _rf_files(tmp_path)
    out = tmp_path / "out.json"
    cost = tmp_path / "cost.json"
    bad = _RichResponse("not json", input_tokens=100, output_tokens=20)
    good = _RichResponse(_RF_GOOD, input_tokens=500, output_tokens=900)
    client = _ScriptedClient([bad, good])
    cj.run_cli(cp, tp, str(out), cost_out_path=str(cost), client=client)
    record = json.loads(cost.read_text(encoding="utf-8"))
    assert record["calls"] == 2
    assert record["input_tokens"] == 600, "must SUM both attempts, not keep the last"
    assert record["output_tokens"] == 920


# --- Second-review gaps: behaviours the first fix pass left unpinned --------- #
# Each of these was a SURVIVING MUTANT — a deliberate break to the fix that the
# suite failed to catch. They pin the discriminating behaviour, not just the
# happy path.

def test_a_normal_stop_reason_does_NOT_trigger_the_truncation_detector():
    # Survived mutant: `stop_reason is not None` in place of `== "max_tokens"`
    # would reject 100% of real responses (every success returns end_turn), and
    # the suite stayed green because no test response carried a stop_reason.
    client = _ScriptedClient([_RichResponse(_RF_GOOD, stop_reason="end_turn")])
    verdict = cj.judge_coverage(_RF_CLAIMS, _RF_TRANSCRIPT, client=client)
    assert verdict["verdicts"][0]["covered"] is True
    assert len(client.calls) == 1


def test_citations_validate_against_SUPPLIED_indices_not_positional_range():
    # Survived mutant: `list(range(len(turn_list)))` in place of the real
    # `[t.index for t in turn_list]`. Every other test uses a contiguous [0, 1]
    # transcript, where the two are indistinguishable. This one is not.
    transcript = [
        {"index": 5, "role": "user", "content": "hello"},
        {"index": 9, "role": "assistant", "content": "world"},
    ]
    # Citing 9 (a real supplied index) must PASS.
    ok = _ScriptedClient([_RichResponse(_verdicts_json([("c1", True, [9]), ("c2", False, [])]))])
    verdict = cj.judge_coverage(_RF_CLAIMS, transcript, client=ok)
    assert verdict["verdicts"][0]["turns"] == [9]
    # Citing 1 (a positional offset, NOT a real index) must FAIL.
    bad = _ScriptedClient([_RichResponse(_verdicts_json([("c1", True, [1]), ("c2", False, [])]))])
    with pytest.raises(cj.InvalidTurnCitationError):
        cj.judge_coverage(_RF_CLAIMS, transcript, client=bad)


def test_empty_transcript_still_validates_citations():
    # Survived mutant: `if valid_turn_indices:` in place of `is not None`, which
    # silently disables validation for an empty transcript — nothing can be
    # covered by a session with no turns, so covered:true must still be rejected.
    payload = _verdicts_json([("c1", True, [0]), ("c2", False, [])])
    client = _ScriptedClient([_RichResponse(payload)])
    with pytest.raises(cj.InvalidTurnCitationError):
        cj.judge_coverage(_RF_CLAIMS, [], client=client)


def test_a_non_string_reason_is_rejected_by_shape_validation():
    # Survived mutant: deleting the `reason` type check left the suite green.
    with pytest.raises(cj.VerdictShapeError):
        cj.parse_verdicts(
            {"verdicts": [{"claim_id": "c1", "covered": True, "turns": [0], "reason": 5}]},
            ["c1"],
        )


def test_out_path_that_is_a_directory_is_rejected_before_the_paid_call(tmp_path):
    # os.path.exists() and os.access(W_OK) both pass for a directory, so without
    # an explicit isdir check the (billed) call fired and only then did the write
    # fail — exactly the discarded spend the pre-check exists to prevent.
    cp, tp = _rf_files(tmp_path)
    a_dir = tmp_path / "iam_a_directory"
    a_dir.mkdir()
    client = _ScriptedClient([_RichResponse(_RF_GOOD)])
    rc = cj.main(["--claims", cp, "--transcript", tp, "--out", str(a_dir)], client=client)
    assert rc == 2
    assert client.calls == []


# --- Migration-pass fixes: close the two structural findings ----------------- #

def test_union_coverage_refuses_multiple_sets_that_all_declare_no_document():
    # The remaining hole in the cross-document guard: >1 distinct declared id and
    # mixed declared/undeclared were refused, but sets that ALL declare nothing
    # merged silently — so stripping metadata bypassed the guard entirely.
    a = [{"claim_id": "c15", "covered": False, "turns": []}]
    b = [{"claim_id": "c15", "covered": True, "turns": [0]}]
    with pytest.raises(cj.CoverageInputError):
        cj.union_coverage([a, b])


def test_union_coverage_allows_unidentified_merge_only_with_explicit_optin():
    # The pure merge arithmetic is still reachable, but a caller must SAY that the
    # sets are known to be the same document rather than have it assumed.
    a = [{"claim_id": "c1", "covered": True, "turns": [0]}]
    b = [{"claim_id": "c2", "covered": False, "turns": []}]
    result = cj.union_coverage([a, b], allow_unidentified=True)
    assert result["covered_ids"] == ["c1"]
    assert result["percentage"] == 50.0


def test_union_coverage_single_unidentified_set_needs_no_optin():
    # One set cannot be a CROSS-document merge, so it stays permitted: the guard
    # must not make the ordinary single-session read harder.
    a = [{"claim_id": "c1", "covered": True, "turns": [0]}]
    assert cj.union_coverage([a])["covered_ids"] == ["c1"]


def test_union_coverage_empty_input_still_needs_no_optin():
    assert cj.union_coverage([]) == {"covered_ids": [], "percentage": 0.0}


def test_truncation_error_is_NOT_retryable_by_type():
    # The error whose entire purpose is "must never be retried" must not be a
    # member of the retryable family. Previously the guarantee was held only by
    # WHERE it is raised; now the type enforces it, so a future `except
    # VerdictParseError: retry` cannot silently re-enable the double-billing.
    assert not issubclass(cj.VerdictTruncatedError, cj.VerdictParseError)


def test_truncation_still_fails_fast_and_is_reported_cleanly(tmp_path, capsys):
    # The type change must not regress the behaviour: still exactly one call, and
    # still a clean one-line CLI error rather than a traceback.
    cp, tp = _rf_files(tmp_path)
    truncated = _RichResponse('{"verdicts":[{"claim_id":"c1","cov', stop_reason="max_tokens")
    client = _ScriptedClient([truncated, truncated])
    rc = cj.main(
        ["--claims", cp, "--transcript", tp, "--out", str(tmp_path / "out.json")],
        client=client,
    )
    assert rc == 2
    assert len(client.calls) == 1
    err = capsys.readouterr().err
    assert "coverage_judge: error:" in err
    assert "Traceback" not in err
