import study_history as sh

_RECAP = """# Study session — Graph Engineering
Duration: 22:46

## What we covered
- What a graph is: nodes and edges
- The fake-edge test

## Key points
### Nodes
Long essay that must NOT appear in the parsed result.

## Open threads
- How to resolve hidden edges
- The verification architecture section
"""


def test_parses_covered_and_open_threads():
    out = sh.parse_recap_sections(_RECAP)
    assert out == {
        "covered": ["What a graph is: nodes and edges", "The fake-edge test"],
        "open_threads": [
            "How to resolve hidden edges",
            "The verification architecture section",
        ],
    }
    assert "fallback_text" not in out


def test_open_threads_optional_empty_list_when_absent():
    text = "## What we covered\n- Only this\n\n## Key points\nblah\n"
    out = sh.parse_recap_sections(text)
    assert out == {"covered": ["Only this"], "open_threads": []}


def test_unparseable_returns_truncated_fallback():
    text = "x" * 5000  # no headers at all
    out = sh.parse_recap_sections(text)
    assert out == {"fallback_text": "x" * 1000}
    assert "covered" not in out


import json

_RECAP_TEXT = (
    "# Study session — Doc\n\n## What we covered\n- Alpha\n- Beta\n\n"
    "## Open threads\n- Gamma\n"
)


def _row(session_id, document_id, session_start, mode="study", kind="session", user_id="matt"):
    return json.dumps({
        "kind": kind, "mode": mode, "session_id": session_id,
        "document_id": document_id, "session_start": session_start,
        "user_id": user_id,
    })


def _seed(ledger, artifacts, rows, recaps, user_id="matt"):
    ledger.write_text("\n".join(rows) + "\n")
    user_dir = artifacts / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    for sid, text in recaps.items():
        (user_dir / f"{sid}.md").write_text(text)


def test_recap_is_user_scoped(study_history_tmp):
    ledger, artifacts = study_history_tmp
    # A studied doc D and left a recap; B never studied D.
    ledger.write_text(
        json.dumps({"kind": "session", "mode": "study", "session_id": "sa",
                    "document_id": "D", "session_start": "2026-07-27T10:00:00",
                    "user_id": "matt"}) + "\n"
    )
    (artifacts / "matt").mkdir(parents=True, exist_ok=True)
    (artifacts / "matt" / "sa.md").write_text("# Study session — D\n\n## What we covered\n- x\n")

    # matt sees the recap; sarah (mirror image) sees None on the same doc.
    assert sh.previous_session_recap("matt", "D", exclude_session_id="live") is not None
    assert sh.previous_session_recap("sarah", "D", exclude_session_id="live") is None


def test_returns_newest_prior_recap_parsed(study_history_tmp):
    ledger, artifacts = study_history_tmp
    _seed(
        ledger, artifacts,
        rows=[
            _row("s-old", "doc-1", "2026-07-20T10:00:00"),
            _row("s-new", "doc-1", "2026-07-25T10:00:00"),
        ],
        recaps={"s-old": "OLD", "s-new": _RECAP_TEXT},
    )
    out = sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current")
    assert out == {"covered": ["Alpha", "Beta"], "open_threads": ["Gamma"]}


def test_first_session_returns_none(study_history_tmp):
    ledger, artifacts = study_history_tmp
    _seed(ledger, artifacts, rows=[_row("s-other", "doc-OTHER", "2026-07-25T10:00:00")],
          recaps={"s-other": _RECAP_TEXT})
    assert sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current") is None


def test_newest_missing_artifact_returns_none_no_walkback(study_history_tmp):
    ledger, artifacts = study_history_tmp
    # Newest (s-new) has NO artifact; older (s-old) DOES. Must NOT walk back.
    _seed(
        ledger, artifacts,
        rows=[
            _row("s-old", "doc-1", "2026-07-20T10:00:00"),
            _row("s-new", "doc-1", "2026-07-25T10:00:00"),
        ],
        recaps={"s-old": _RECAP_TEXT},  # only the OLD one has a recap
    )
    assert sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current") is None


def test_excludes_current_session(study_history_tmp):
    ledger, artifacts = study_history_tmp
    _seed(ledger, artifacts, rows=[_row("s-current", "doc-1", "2026-07-25T10:00:00")],
          recaps={"s-current": _RECAP_TEXT})
    assert sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current") is None


def test_ignores_non_session_and_non_study_rows(study_history_tmp):
    ledger, artifacts = study_history_tmp
    _seed(
        ledger, artifacts,
        rows=[
            _row("s-art", "doc-1", "2026-07-26T10:00:00", kind="artifact"),
            _row("s-open", "doc-1", "2026-07-26T09:00:00", mode="open"),
            _row("s-real", "doc-1", "2026-07-24T10:00:00"),
        ],
        recaps={"s-art": "X", "s-open": "Y", "s-real": _RECAP_TEXT},
    )
    out = sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current")
    assert out == {"covered": ["Alpha", "Beta"], "open_threads": ["Gamma"]}


def test_unreadable_artifact_returns_none(study_history_tmp):
    ledger, artifacts = study_history_tmp
    _seed(
        ledger, artifacts,
        rows=[_row("s-new", "doc-1", "2026-07-25T10:00:00")],
        recaps={},
    )
    # Create the artifact path as a directory instead of a file, so
    # read_text() raises IsADirectoryError — a cross-platform way to force
    # a read failure without relying on chmod.
    (artifacts / "matt").mkdir(parents=True, exist_ok=True)
    (artifacts / "matt" / "s-new.md").mkdir()
    assert sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current") is None


def test_empty_artifact_returns_none(study_history_tmp):
    ledger, artifacts = study_history_tmp
    _seed(
        ledger, artifacts,
        rows=[_row("s-new", "doc-1", "2026-07-25T10:00:00")],
        recaps={"s-new": ""},
    )
    assert sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current") is None


def test_whitespace_only_artifact_returns_none(study_history_tmp):
    ledger, artifacts = study_history_tmp
    _seed(
        ledger, artifacts,
        rows=[_row("s-new", "doc-1", "2026-07-25T10:00:00")],
        recaps={"s-new": "  \n\t  "},
    )
    assert sh.previous_session_recap("matt", "doc-1", exclude_session_id="s-current") is None


# ---------------------------------------------------------------------------
# Sprint 2 — per-user state isolation on a SHARED document.
#
# A shared demo doc lives once in documents/_shared/, but ALL state about it
# stays per-user, keyed on (user_id, document_id). These tests exercise the
# real collision the shared namespace introduces: two users A and B studying
# the SAME shared document id D. previous_session_recap must resolve each
# user's OWN recap and never the other's — the isolation key is
# user_id + document_id, not document_id alone.
# ---------------------------------------------------------------------------

# One shared document id, used by BOTH users below — this is the collision that
# the shared namespace makes possible. If isolation keyed on document_id alone,
# these tests would cross-contaminate.
_SHARED_DOC = "shared-demo-doc"

# Distinguishable recaps so a leak is observable, not merely a None/non-None flip.
_RECAP_A = (
    "# Study session — Shared Demo\n\n"
    "## What we covered\n- ALPHA-marker only in matt's recap\n\n"
    "## Open threads\n- alpha open thread\n"
)
_RECAP_B = (
    "# Study session — Shared Demo\n\n"
    "## What we covered\n- BRAVO-marker only in sarah's recap\n\n"
    "## Open threads\n- bravo open thread\n"
)


def _seed_user_recap(ledger, artifacts, *, user_id, session_id, document_id, recap,
                     session_start):
    """Append one study-session ledger row and its recap artifact for ``user_id``.

    The recap artifact is written under ``artifacts/<user_id>/`` — the same
    per-user subdirectory previous_session_recap resolves against. Appends so
    multiple users can be seeded against one ledger/artifacts pair.
    """
    with ledger.open("a") as f:
        f.write(_row(session_id, document_id, session_start, user_id=user_id) + "\n")
    user_dir = artifacts / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / f"{session_id}.md").write_text(recap)


def test_shared_doc_recap_is_per_user_cross_excluded(study_history_tmp):
    """Two users on the SAME shared doc id → each recap resolves per-user only.

    A leak in either direction (A seeing BRAVO, or B seeing ALPHA) fails.
    """
    ledger, artifacts = study_history_tmp
    _seed_user_recap(ledger, artifacts, user_id="matt", session_id="sa",
                     document_id=_SHARED_DOC, recap=_RECAP_A,
                     session_start="2026-07-25T10:00:00")
    _seed_user_recap(ledger, artifacts, user_id="sarah", session_id="sb",
                     document_id=_SHARED_DOC, recap=_RECAP_B,
                     session_start="2026-07-26T11:00:00")

    a = sh.previous_session_recap("matt", _SHARED_DOC, exclude_session_id="live")
    b = sh.previous_session_recap("sarah", _SHARED_DOC, exclude_session_id="live")

    # matt's recap: ALPHA present, BRAVO absent.
    assert a == {"covered": ["ALPHA-marker only in matt's recap"],
                 "open_threads": ["alpha open thread"]}
    assert "BRAVO-marker only in sarah's recap" not in a["covered"]

    # sarah's recap (mirror image): BRAVO present, ALPHA absent.
    assert b == {"covered": ["BRAVO-marker only in sarah's recap"],
                 "open_threads": ["bravo open thread"]}
    assert "ALPHA-marker only in matt's recap" not in b["covered"]


def test_shared_doc_recap_none_for_user_without_sessions(study_history_tmp):
    """A third user C with no sessions on the shared doc D gets exactly None,
    even though A and B both have sessions on that same D."""
    ledger, artifacts = study_history_tmp
    _seed_user_recap(ledger, artifacts, user_id="matt", session_id="sa",
                     document_id=_SHARED_DOC, recap=_RECAP_A,
                     session_start="2026-07-25T10:00:00")
    _seed_user_recap(ledger, artifacts, user_id="sarah", session_id="sb",
                     document_id=_SHARED_DOC, recap=_RECAP_B,
                     session_start="2026-07-26T11:00:00")

    assert sh.previous_session_recap("dev", _SHARED_DOC, exclude_session_id="live") is None


def test_shared_doc_recap_artifacts_are_per_user_on_disk(study_history_tmp):
    """The recap-artifact dimension: for one shared doc id, A's and B's recap
    artifacts live under distinct per-user dirs and read back per-user only.

    This is the ARTIFACTS_DIR path-isolation mechanism previous_session_recap
    relies on — proven directly at the storage layer for the shared doc.
    """
    ledger, artifacts = study_history_tmp
    _seed_user_recap(ledger, artifacts, user_id="matt", session_id="sa",
                     document_id=_SHARED_DOC, recap=_RECAP_A,
                     session_start="2026-07-25T10:00:00")
    _seed_user_recap(ledger, artifacts, user_id="sarah", session_id="sb",
                     document_id=_SHARED_DOC, recap=_RECAP_B,
                     session_start="2026-07-26T11:00:00")

    a_artifact = (artifacts / "matt" / "sa.md").read_text()
    b_artifact = (artifacts / "sarah" / "sb.md").read_text()
    assert "ALPHA-marker" in a_artifact and "BRAVO-marker" not in a_artifact
    assert "BRAVO-marker" in b_artifact and "ALPHA-marker" not in b_artifact
    # Neither user's session id collides into the other's directory.
    assert not (artifacts / "matt" / "sb.md").exists()
    assert not (artifacts / "sarah" / "sa.md").exists()
