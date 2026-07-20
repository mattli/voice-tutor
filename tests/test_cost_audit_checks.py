"""Hermetic characterization tests for the sprint-1 cost-log audit checks.

These tests drive ``cost_audit.audit_cost_log`` over fixture JSONL content
written to a per-test tmp ledger (via the ``cost_audit_log_tmp`` fixture, which
monkeypatches ``cost_audit.COST_LOG_JSONL_PATH`` — the same pattern as
sessions.py's ``cost_log_tmp``). Everything is stdlib-only, touches no real
filesystem outside tmp, no network, and passes with all provider API keys unset
and Pipecat absent.

The suite pins, by 1-based line number and enumerated category, exactly which
lines each check flags and why:
  - a correct study session+artifact pair (clean),
  - a pre-``kind`` legacy row (clean, tolerated),
  - a malformed line (flagged malformed),
  - an orphan artifact (flagged orphan_artifact),
  - a cost-mismatch row (flagged cost_mismatch),
plus the c3/c4 partition, the whole-file (forward-reference) orphan pass, and a
malformed-then-valid ordering proving correct 1-based line numbering.
"""

import json

import cost_audit


# ---------------------------------------------------------------------------
# Fixture row builders — hand-computed correct costs against the relocated
# pricing constants (see cost_audit's module docstring / sprint-0 tests).
# ---------------------------------------------------------------------------
def _correct_session_row(session_id="2026-04-14T101010"):
    """A fully-correct multi-component study session row.

    All five cost components present and internally consistent:
      llm  = 12_000/1e6*3.00 + 400_000/1e6*0.30 + 8_000/1e6*3.75 + 5_000/1e6*15.00
           = 0.036 + 0.12 + 0.03 + 0.075 = 0.261
      stt  = 4.25 * 0.0077                         = 0.032725 -> stored 0.0327
      tts  = 9_000 * (5/100_000)                   = 0.45
      post = 20_000/1e6*1.00 + 3_000/1e6*5.00      = 0.035
      total = 0.261 + 0.032725 + 0.45 + 0.035      = 0.778725 -> stored 0.7787
    """
    return {
        "kind": "session",
        "mode": "study",
        "session_id": session_id,
        "document_id": "doc-abc",
        "tts_chars": 9_000,
        "stt_minutes_billed": 4.25,
        "llm_uncached_input_tokens": 12_000,
        "llm_cache_read_tokens": 400_000,
        "llm_cache_write_tokens": 8_000,
        "llm_output_tokens": 5_000,
        "post_session_input_tokens": 20_000,
        "post_session_output_tokens": 3_000,
        "cost_llm_usd": 0.261,
        "cost_stt_usd": 0.0327,
        "cost_tts_usd": 0.45,
        "cost_post_session_usd": 0.035,
        "cost_total_usd": 0.7787,
    }


def _correct_artifact_row(session_id="2026-04-14T101010"):
    """A correct artifact row paired to a session by ``session_id``.

    haiku = 60_000/1e6*1.00 + 4_000/1e6*5.00 = 0.06 + 0.02 = 0.08
    """
    return {
        "kind": "artifact",
        "session_id": session_id,
        "document_id": "doc-abc",
        "input_tokens": 60_000,
        "output_tokens": 4_000,
        "cost_usd": 0.08,
    }


def _legacy_row():
    """A pre-``kind`` legacy row: NO ``kind`` field, NO ``mode``.

    Its stored costs deliberately DISAGREE with any recompute — proving legacy
    rows are skipped by the mismatch check rather than merely happening to match.
    """
    return {
        "session_id": "2026-01-01T000000",
        "tts_chars": 1_000,
        "stt_minutes_billed": 1.0,
        "llm_uncached_input_tokens": 1_000,
        "llm_output_tokens": 500,
        "cost_total_usd": 999.99,  # nonsense — must NOT be flagged (legacy)
    }


def _write_ledger(path, rows):
    """Write ``rows`` (dicts or raw strings) as JSONL lines to ``path``.

    A dict is JSON-encoded; a str is written verbatim (for malformed lines).
    """
    lines = []
    for r in rows:
        if isinstance(r, str):
            lines.append(r)
        else:
            lines.append(json.dumps(r))
    path.write_text("\n".join(lines) + "\n")


def _findings_on(result, line_number):
    return [f for f in result.findings if f.line_number == line_number]


def _categories_on(result, line_number):
    return {f.category for f in _findings_on(result, line_number)}


# ===========================================================================
# c2: structured summary shape.
# ===========================================================================
def test_audit_returns_structured_summary(cost_audit_log_tmp):
    _write_ledger(
        cost_audit_log_tmp,
        [_correct_session_row(), _correct_artifact_row()],
    )
    result = cost_audit.audit_cost_log()

    # rows_read / rows_valid / per-category counts / findings list all present.
    assert result.rows_read == 2
    assert result.rows_valid == 2
    assert isinstance(result.findings, list)
    assert result.findings == []
    assert set(result.category_counts) == set(cost_audit.FINDING_CATEGORIES)
    assert all(v == 0 for v in result.category_counts.values())


def test_finding_fields_are_category_line_reason(cost_audit_log_tmp):
    # An orphan artifact yields a finding whose fields we inspect directly.
    _write_ledger(cost_audit_log_tmp, [_correct_artifact_row("no-such-session")])
    result = cost_audit.audit_cost_log()

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.category in cost_audit.FINDING_CATEGORIES
    assert finding.line_number == 1  # 1-based
    assert isinstance(finding.reason, str) and finding.reason


# ===========================================================================
# c3: malformed detection + c3/c4 partition + continued processing.
# ===========================================================================
def test_non_json_line_is_malformed(cost_audit_log_tmp):
    _write_ledger(cost_audit_log_tmp, ["this is not json {"])
    result = cost_audit.audit_cost_log()
    assert _categories_on(result, 1) == {cost_audit.CATEGORY_MALFORMED}


def test_non_object_json_is_malformed(cost_audit_log_tmp):
    # Valid JSON, but a list not an object.
    _write_ledger(cost_audit_log_tmp, ["[1, 2, 3]"])
    result = cost_audit.audit_cost_log()
    assert _categories_on(result, 1) == {cost_audit.CATEGORY_MALFORMED}


def test_kind_declaring_row_missing_identity_key_is_malformed(cost_audit_log_tmp):
    # A session row that declares kind but lacks session_id -> malformed.
    bad = _correct_session_row()
    del bad["session_id"]
    _write_ledger(cost_audit_log_tmp, [bad])
    result = cost_audit.audit_cost_log()
    assert _categories_on(result, 1) == {cost_audit.CATEGORY_MALFORMED}


def test_row_without_kind_is_never_malformed(cost_audit_log_tmp):
    _write_ledger(cost_audit_log_tmp, [_legacy_row()])
    result = cost_audit.audit_cost_log()
    assert _findings_on(result, 1) == []


def test_c3_c4_partition_missing_cost_is_mismatch_not_malformed(cost_audit_log_tmp):
    # A kind:session row WITH session_id but missing a stored cost_*_usd field
    # is NOT malformed — it is a cost_mismatch (recompute-vs-absent).
    row = _correct_session_row()
    del row["cost_tts_usd"]
    _write_ledger(cost_audit_log_tmp, [row])
    result = cost_audit.audit_cost_log()
    cats = _categories_on(result, 1)
    assert cost_audit.CATEGORY_MALFORMED not in cats
    assert cats == {cost_audit.CATEGORY_COST_MISMATCH}
    # The reason names the offending field.
    reason = _findings_on(result, 1)[0].reason
    assert "cost_tts_usd" in reason


def test_valid_row_after_malformed_line_is_still_audited(cost_audit_log_tmp):
    # malformed line 1, then a correct session+artifact on lines 2 & 3.
    session = _correct_session_row()
    artifact = _correct_artifact_row()
    _write_ledger(cost_audit_log_tmp, ["{ broken", session, artifact])
    result = cost_audit.audit_cost_log()

    assert result.rows_read == 3
    assert _categories_on(result, 1) == {cost_audit.CATEGORY_MALFORMED}
    # Lines 2 & 3 processed and clean (proves no crash, continued processing).
    assert _findings_on(result, 2) == []
    assert _findings_on(result, 3) == []


# ===========================================================================
# c4: cost-mismatch detection.
# ===========================================================================
def test_correct_pair_yields_no_mismatch(cost_audit_log_tmp):
    _write_ledger(
        cost_audit_log_tmp, [_correct_session_row(), _correct_artifact_row()]
    )
    result = cost_audit.audit_cost_log()
    assert result.category_counts[cost_audit.CATEGORY_COST_MISMATCH] == 0


def test_session_wrong_component_is_flagged_naming_field(cost_audit_log_tmp):
    row = _correct_session_row()
    row["cost_llm_usd"] = 0.999  # deliberately wrong
    _write_ledger(cost_audit_log_tmp, [row])
    result = cost_audit.audit_cost_log()

    findings = _findings_on(result, 1)
    assert len(findings) == 1
    assert findings[0].category == cost_audit.CATEGORY_COST_MISMATCH
    assert "cost_llm_usd" in findings[0].reason


def test_session_missing_cost_field_is_flagged_mismatch(cost_audit_log_tmp):
    row = _correct_session_row()
    del row["cost_total_usd"]
    _write_ledger(cost_audit_log_tmp, [row])
    result = cost_audit.audit_cost_log()
    findings = _findings_on(result, 1)
    assert len(findings) == 1
    assert findings[0].category == cost_audit.CATEGORY_COST_MISMATCH
    assert "cost_total_usd" in findings[0].reason


def test_within_tolerance_row_is_clean(cost_audit_log_tmp):
    row = _correct_session_row()
    # Nudge one stored cost by less than the tolerance — must NOT be flagged.
    row["cost_tts_usd"] = 0.45 + cost_audit.COST_TOLERANCE_USD / 2
    _write_ledger(cost_audit_log_tmp, [row])
    result = cost_audit.audit_cost_log()
    assert _findings_on(result, 1) == []


def test_artifact_wrong_cost_is_flagged(cost_audit_log_tmp):
    session = _correct_session_row()
    artifact = _correct_artifact_row()
    artifact["cost_usd"] = 0.99  # wrong
    _write_ledger(cost_audit_log_tmp, [session, artifact])
    result = cost_audit.audit_cost_log()
    findings = _findings_on(result, 2)
    assert len(findings) == 1
    assert findings[0].category == cost_audit.CATEGORY_COST_MISMATCH
    assert "cost_usd" in findings[0].reason


# ===========================================================================
# c5: orphan detection — whole-file two-phase pass.
# ===========================================================================
def test_paired_artifact_is_not_orphan(cost_audit_log_tmp):
    _write_ledger(
        cost_audit_log_tmp,
        [_correct_session_row("s1"), _correct_artifact_row("s1")],
    )
    result = cost_audit.audit_cost_log()
    assert result.category_counts[cost_audit.CATEGORY_ORPHAN_ARTIFACT] == 0


def test_forward_referenced_artifact_is_not_orphan(cost_audit_log_tmp):
    # Artifact (line 1) appears BEFORE its matching session (line 2).
    _write_ledger(
        cost_audit_log_tmp,
        [_correct_artifact_row("s2"), _correct_session_row("s2")],
    )
    result = cost_audit.audit_cost_log()
    # Proves whole-file two-phase pass, not backward-only scan.
    assert result.category_counts[cost_audit.CATEGORY_ORPHAN_ARTIFACT] == 0
    assert _findings_on(result, 1) == []


def test_standalone_artifact_is_orphan(cost_audit_log_tmp):
    _write_ledger(
        cost_audit_log_tmp,
        [_correct_session_row("s3"), _correct_artifact_row("orphan-sid")],
    )
    result = cost_audit.audit_cost_log()
    assert _categories_on(result, 2) == {cost_audit.CATEGORY_ORPHAN_ARTIFACT}
    assert result.category_counts[cost_audit.CATEGORY_ORPHAN_ARTIFACT] == 1
    # The paired-less session row (line 1) is NORMAL — never flagged.
    assert _findings_on(result, 1) == []


def test_session_without_artifact_is_clean(cost_audit_log_tmp):
    _write_ledger(cost_audit_log_tmp, [_correct_session_row("solo")])
    result = cost_audit.audit_cost_log()
    assert result.findings == []


# ===========================================================================
# c6: legacy tolerance (strict: absence of kind) vs mode-less current session.
# ===========================================================================
def test_legacy_row_produces_no_findings(cost_audit_log_tmp):
    _write_ledger(cost_audit_log_tmp, [_legacy_row()])
    result = cost_audit.audit_cost_log()
    assert result.findings == []
    assert result.rows_valid == 1


def test_modeless_current_session_is_still_cost_checked(cost_audit_log_tmp):
    # Declares kind=session but omits mode. NOT legacy — must be cost-checked.
    row = _correct_session_row()
    row.pop("mode", None)
    row["cost_llm_usd"] = 0.999  # wrong -> must be flagged despite no mode
    _write_ledger(cost_audit_log_tmp, [row])
    result = cost_audit.audit_cost_log()
    assert _categories_on(result, 1) == {cost_audit.CATEGORY_COST_MISMATCH}


def test_modeless_correct_session_is_clean(cost_audit_log_tmp):
    row = _correct_session_row()
    row.pop("mode", None)
    _write_ledger(cost_audit_log_tmp, [row])
    result = cost_audit.audit_cost_log()
    assert _findings_on(result, 1) == []


# ===========================================================================
# c7: one place covering all five required cases + line-numbering proof.
# ===========================================================================
def test_all_cases_together_by_line_and_category(cost_audit_log_tmp):
    """One ledger exercising every required case, asserted by line number and
    enumerated category.

    Layout (1-based lines):
      1: correct session      (clean)               kind=session, sid=ok
      2: correct artifact     (clean, paired)       kind=artifact, sid=ok
      3: legacy row           (clean, tolerated)    no kind
      4: malformed line       (flagged malformed)   non-JSON
      5: cost-mismatch session(flagged mismatch)    kind=session, wrong cost
      6: orphan artifact      (flagged orphan)      kind=artifact, sid=none
    """
    good_session = _correct_session_row("ok")
    good_artifact = _correct_artifact_row("ok")
    legacy = _legacy_row()
    bad_cost = _correct_session_row("mm")
    bad_cost["cost_stt_usd"] = 5.0  # wrong
    orphan = _correct_artifact_row("nobody")

    _write_ledger(
        cost_audit_log_tmp,
        [good_session, good_artifact, legacy, "@@ not json @@", bad_cost, orphan],
    )
    result = cost_audit.audit_cost_log()

    assert result.rows_read == 6

    # Clean lines.
    assert _findings_on(result, 1) == []
    assert _findings_on(result, 2) == []
    assert _findings_on(result, 3) == []  # legacy tolerated
    # Flagged lines, by exact category.
    assert _categories_on(result, 4) == {cost_audit.CATEGORY_MALFORMED}
    assert _categories_on(result, 5) == {cost_audit.CATEGORY_COST_MISMATCH}
    assert _categories_on(result, 6) == {cost_audit.CATEGORY_ORPHAN_ARTIFACT}

    assert result.category_counts == {
        cost_audit.CATEGORY_MALFORMED: 1,
        cost_audit.CATEGORY_COST_MISMATCH: 1,
        cost_audit.CATEGORY_ORPHAN_ARTIFACT: 1,
    }
    # 6 rows read, 3 lines flagged -> 3 valid.
    assert result.rows_valid == 3


def test_malformed_then_valid_preserves_1based_numbering(cost_audit_log_tmp):
    """A malformed line immediately FOLLOWED by a valid orphan/mismatch row —
    the follow-on finding must sit on line 2, proving the malformed line still
    counts as line 1 (1-based, malformed included)."""
    orphan = _correct_artifact_row("nobody")
    _write_ledger(cost_audit_log_tmp, ["<<< malformed >>>", orphan])
    result = cost_audit.audit_cost_log()

    assert _categories_on(result, 1) == {cost_audit.CATEGORY_MALFORMED}
    assert _categories_on(result, 2) == {cost_audit.CATEGORY_ORPHAN_ARTIFACT}


def test_absent_ledger_is_empty_clean_result(cost_audit_log_tmp):
    # The fixture path doesn't exist until written; audit tolerates absence.
    result = cost_audit.audit_cost_log()
    assert result.rows_read == 0
    assert result.rows_valid == 0
    assert result.findings == []


# ===========================================================================
# c8: CLI entry point renders the same structured summary.
# ===========================================================================
def test_cli_main_prints_summary(cost_audit_log_tmp, capsys):
    good_session = _correct_session_row("ok")
    good_artifact = _correct_artifact_row("ok")
    orphan = _correct_artifact_row("nobody")
    _write_ledger(cost_audit_log_tmp, [good_session, good_artifact, orphan])

    rc = cost_audit.main([])
    assert rc == 0

    out = capsys.readouterr().out
    # rows read / rows valid / per-category counts present.
    assert "rows read:  3" in out
    assert "rows valid:" in out
    for cat in cost_audit.FINDING_CATEGORIES:
        assert cat in out
    # For the one finding, its 1-based line number appears together with its
    # category/reason (not merely an aggregate count).
    result = cost_audit.audit_cost_log()
    (finding,) = result.findings
    assert f"line {finding.line_number}" in out
    assert finding.category in out


def test_cli_report_is_view_of_same_result(cost_audit_log_tmp):
    # format_report renders the exact AuditResult the tests grade (c2).
    _write_ledger(cost_audit_log_tmp, [_correct_artifact_row("nobody")])
    result = cost_audit.audit_cost_log()
    report = cost_audit.format_report(result)
    assert f"rows read:  {result.rows_read}" in report
    for f in result.findings:
        assert f"line {f.line_number}" in report
        assert f.category in report
        assert f.reason in report
