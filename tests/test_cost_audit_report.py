"""Hermetic tests for the sprint-2 cost-audit CLI + report layer.

These tests drive ``cost_audit.main`` / ``cost_audit.format_report`` over
fixture JSONL content written to a per-test tmp ledger (via the
``cost_audit_log_tmp`` fixture, which monkeypatches
``cost_audit.COST_LOG_JSONL_PATH`` — the same pattern as sessions.py's
``cost_log_tmp`` and grounding.py's ``grounding_tmp``). Everything is
stdlib-only, touches no real filesystem outside tmp, no network, and passes with
all provider API keys unset and Pipecat unimportable by the report path.

The suite pins the SPRINT-2 contract: ``main([])`` returns int 0 and prints a
non-empty report; ``format_report`` is a pure ``str``-returning function; the
rendered report shows rows read, rows valid, and an explicit per-category count
line for EVERY category (including 0-count and >1-count ones, excluding the
path-header line); the offending-line section is deterministically ordered by
ascending 1-based line number with locatable ``line <n>`` substrings and
category-identifiable reasons (carrying stored+recomputed values for
mismatches); legacy + correct rows never appear as offenders; running the report
is read-only (ledger byte content unchanged, no new file created); the real
default path is tolerated read-only; and the report is not a fixed stub (two
different ledgers yield different counts).
"""

import hashlib
import json
import re

import cost_audit


# ---------------------------------------------------------------------------
# Fixture row builders — hand-computed correct costs against the relocated
# pricing constants (mirrors tests/test_cost_audit_checks.py).
# ---------------------------------------------------------------------------
def _correct_session_row(session_id="2026-04-14T101010"):
    """A fully-correct, internally-consistent multi-component study session row."""
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
    """A correct artifact row paired to a session by ``session_id`` (Haiku math)."""
    return {
        "kind": "artifact",
        "session_id": session_id,
        "document_id": "doc-abc",
        "input_tokens": 60_000,
        "output_tokens": 4_000,
        "cost_usd": 0.08,
    }


def _legacy_row():
    """A pre-``kind`` legacy row: NO ``kind``, NO ``mode`` — must be tolerated."""
    return {
        "session_id": "2026-01-01T000000",
        "tts_chars": 1_000,
        "stt_minutes_billed": 1.0,
        "llm_uncached_input_tokens": 1_000,
        "llm_output_tokens": 500,
        "cost_total_usd": 999.99,  # nonsense — must NOT be flagged (legacy)
    }


def _write_ledger(path, rows):
    """Write ``rows`` (dicts or raw strings) as JSONL lines to ``path``."""
    lines = []
    for r in rows:
        lines.append(r if isinstance(r, str) else json.dumps(r))
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Report-parsing helpers (rendered-string introspection).
# ---------------------------------------------------------------------------
_ROWS_READ_RE = re.compile(r"rows read:\s*(\d+)")
_ROWS_VALID_RE = re.compile(r"rows valid:\s*(\d+)")


def _rows_read(report):
    m = _ROWS_READ_RE.search(report)
    assert m, f"no 'rows read' line in report:\n{report}"
    return int(m.group(1))


def _rows_valid(report):
    m = _ROWS_VALID_RE.search(report)
    assert m, f"no 'rows valid' line in report:\n{report}"
    return int(m.group(1))


def _category_count(report, category):
    """Extract the per-category count line's number for ``category``.

    Scans the rendered report for the line carrying this category's count,
    IGNORING the non-numeric path-header line (which merely echoes the log
    path). Returns the int, or raises if no count line for the category exists.
    """
    for line in report.splitlines():
        if line.lstrip().startswith("cost-log audit"):
            # path-header line — tolerated context, not a graded count line.
            continue
        # A count line looks like "  <category>: <n>" and is NOT an offending
        # "  line <n> [..]" entry.
        stripped = line.strip()
        if stripped.startswith(f"{category}:"):
            m = re.search(r":\s*(\d+)\s*$", stripped)
            if m:
                return int(m.group(1))
    raise AssertionError(f"no count line for category {category!r} in:\n{report}")


def _offending_line_numbers(report):
    """Return the 1-based line numbers listed in the offending section, in the
    order they appear in the rendered report."""
    nums = []
    for line in report.splitlines():
        m = re.search(r"\bline (\d+)\b", line)
        # Exclude the "rows ..." header lines (they don't contain "line ").
        if m and "rows read" not in line and "rows valid" not in line:
            nums.append(int(m.group(1)))
    return nums


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===========================================================================
# c1: main([]) returns int 0 (not SystemExit) and prints a non-empty report.
# ===========================================================================
def test_main_returns_int_zero_and_prints_report(cost_audit_log_tmp, capsys):
    _write_ledger(
        cost_audit_log_tmp, [_correct_session_row(), _correct_artifact_row()]
    )

    ret = cost_audit.main([])

    out = capsys.readouterr().out
    assert isinstance(ret, int)
    assert ret == 0
    assert out.strip() != ""  # non-empty report on stdout


def test_main_with_findings_still_returns_int_zero(cost_audit_log_tmp, capsys):
    # A dirty ledger (findings present) is diagnostic output, NOT a tool failure.
    _write_ledger(
        cost_audit_log_tmp,
        ["@@ not json @@", _correct_artifact_row("nobody")],
    )
    ret = cost_audit.main([])
    out = capsys.readouterr().out
    assert isinstance(ret, int) and ret == 0
    assert "rows read" in out


# ===========================================================================
# c2: format_report is a pure str-returning function; module is import-pure.
# ===========================================================================
def test_format_report_returns_str(cost_audit_log_tmp):
    _write_ledger(cost_audit_log_tmp, [_correct_session_row()])
    result = cost_audit.audit_cost_log()
    report = cost_audit.format_report(result)
    assert isinstance(report, str)
    assert report != ""


def test_module_source_has_no_forbidden_imports():
    """Grep cost_audit's source IMPORT statements: no app/bot/pipecat/network.

    We inspect only actual import lines (via the AST) so prose mentions such as
    the module's ``Pipecat-free`` docstring do not falsely trip the check.
    """
    import ast
    from pathlib import Path

    src = Path(cost_audit.__file__).read_text()
    tree = ast.parse(src)

    imported_tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_tops.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_tops.add(node.module.split(".")[0])

    forbidden = {
        "app",
        "bot",
        "pipecat",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "http",
        "aiohttp",
    }
    assert not (imported_tops & forbidden), (
        f"forbidden imports present: {sorted(imported_tops & forbidden)}"
    )
    # Positive: the module depends only on stdlib json + pathlib at module scope.
    assert "json" in imported_tops
    assert "pathlib" in imported_tops


# ===========================================================================
# c3: rendered report shows rows read/valid + a per-category count line for
# EVERY category, including a 0-count and a >1-count one; counts derive from
# the audit result; path-header line is exempt from the count scan.
# ===========================================================================
def test_report_shows_counts_for_every_category_including_zero_and_plural(
    cost_audit_log_tmp,
):
    # Composition: two malformed lines (>1), zero orphans (0), one mismatch.
    #   line 1: malformed
    #   line 2: malformed
    #   line 3: correct session (clean)
    #   line 4: correct artifact paired to line 3 (clean, NOT orphan)
    #   line 5: session with a wrong cost component (mismatch)
    good_session = _correct_session_row("ok")
    good_artifact = _correct_artifact_row("ok")
    bad = _correct_session_row("mm")
    bad["cost_llm_usd"] = 0.999  # wrong
    _write_ledger(
        cost_audit_log_tmp,
        ["@@ nope @@", "### nope", good_session, good_artifact, bad],
    )

    result = cost_audit.audit_cost_log()
    report = cost_audit.format_report(result)

    # rows read / rows valid derived from the result (never hard-coded).
    assert _rows_read(report) == result.rows_read == 5
    # 3 offending lines (1, 2, 5) -> rows_valid = 5 - 3 = 2.
    assert _rows_valid(report) == result.rows_valid == 2

    # Every category has an explicit count line, matching the audit result —
    # including the 0-count orphan category and the >1-count malformed one.
    assert _category_count(report, cost_audit.CATEGORY_MALFORMED) == 2
    assert _category_count(report, cost_audit.CATEGORY_COST_MISMATCH) == 1
    assert _category_count(report, cost_audit.CATEGORY_ORPHAN_ARTIFACT) == 0
    for cat in cost_audit.FINDING_CATEGORIES:
        assert _category_count(report, cat) == result.category_counts[cat]


def test_rows_valid_counts_a_multi_finding_line_once(cost_audit_log_tmp):
    # An artifact with BOTH a wrong cost AND no matching session carries two
    # findings on one physical line; that line is one invalid row, not two.
    orphan = _correct_artifact_row("nobody")
    orphan["cost_usd"] = 0.99  # also a mismatch
    _write_ledger(cost_audit_log_tmp, [orphan])

    result = cost_audit.audit_cost_log()
    report = cost_audit.format_report(result)

    assert len(result.findings) == 2  # two findings...
    assert _rows_read(report) == 1
    assert _rows_valid(report) == 0  # ...on a single invalid row.


# ===========================================================================
# c4: offending-line section is deterministically ordered by ascending 1-based
# line number; each entry has a locatable "line <n>" substring and a
# category-identifiable reason carrying stored+recomputed for mismatches;
# rendering is byte-for-byte stable.
# ===========================================================================
def test_offending_section_ascending_and_reasons(cost_audit_log_tmp):
    good_session = _correct_session_row("ok")          # line 1 clean
    good_artifact = _correct_artifact_row("ok")        # line 2 clean
    malformed = "@@ not json @@"                        # line 3 malformed
    bad_cost = _correct_session_row("mm")              # line 4 mismatch
    bad_cost["cost_stt_usd"] = 5.0
    orphan = _correct_artifact_row("nobody")           # line 5 orphan
    _write_ledger(
        cost_audit_log_tmp,
        [good_session, good_artifact, malformed, bad_cost, orphan],
    )

    result = cost_audit.audit_cost_log()
    report = cost_audit.format_report(result)

    # (a) ascending 1-based order.
    listed = _offending_line_numbers(report)
    assert listed == sorted(listed)
    assert listed == [3, 4, 5]

    # (b) each offending line yields a locatable "line <n>" substring (tolerating
    # leading indentation) and a category-identifying token/reason.
    for f in result.findings:
        assert f"line {f.line_number}" in report
        assert f.category in report
        assert f.reason in report

    # The mismatch reason carries BOTH the stored and recomputed values (the
    # report surfaces the finding's reason verbatim rather than re-deriving).
    (mismatch,) = [
        f for f in result.findings
        if f.category == cost_audit.CATEGORY_COST_MISMATCH
    ]
    assert "5.0" in mismatch.reason  # stored value
    # recomputed cost_stt_usd = 4.25 * 0.0077 = 0.032725
    assert "0.032725" in mismatch.reason
    assert mismatch.reason in report


def test_report_is_byte_for_byte_deterministic(cost_audit_log_tmp):
    _write_ledger(
        cost_audit_log_tmp,
        [_correct_session_row("ok"), _correct_artifact_row("nobody")],
    )
    result = cost_audit.audit_cost_log()
    first = cost_audit.format_report(result)
    second = cost_audit.format_report(result)
    assert first == second


# ===========================================================================
# c5: legacy rows + a correct pair are valid — never in the offending section,
# and counted in 'rows valid'.
# ===========================================================================
def test_legacy_and_correct_pair_are_valid_not_offenders(cost_audit_log_tmp):
    # line 1: legacy (no kind); line 2: correct session; line 3: correct artifact
    _write_ledger(
        cost_audit_log_tmp,
        [_legacy_row(), _correct_session_row("ok"), _correct_artifact_row("ok")],
    )
    result = cost_audit.audit_cost_log()
    report = cost_audit.format_report(result)

    listed = _offending_line_numbers(report)
    for clean_line in (1, 2, 3):
        assert clean_line not in listed
        assert f"line {clean_line}" not in report

    assert _rows_valid(report) == 3 == result.rows_valid


# ===========================================================================
# c6: running the report is read-only — ledger bytes unchanged, no new file.
# ===========================================================================
def test_report_run_is_read_only(cost_audit_log_tmp, capsys):
    _write_ledger(
        cost_audit_log_tmp,
        ["@@ nope @@", _correct_session_row("ok"), _correct_artifact_row("nobody")],
    )
    before_hash = _sha256(cost_audit_log_tmp)
    dir_before = {p.name for p in cost_audit_log_tmp.parent.iterdir()}

    ret = cost_audit.main([])
    capsys.readouterr()

    assert ret == 0
    # Ledger content byte-for-byte identical (content hash, not mtime).
    assert _sha256(cost_audit_log_tmp) == before_hash
    # No new file created alongside the ledger (no repair/output side effect).
    dir_after = {p.name for p in cost_audit_log_tmp.parent.iterdir()}
    assert dir_after == dir_before


# ===========================================================================
# c7: real/default path (NOT monkeypatched) — main([]) returns int 0, prints a
# well-formed non-empty report, and neither creates nor mutates the default file.
# ===========================================================================
def test_real_default_path_smoke_is_read_only_and_returns_zero(capsys):
    # NO monkeypatch: exercise the real configured default path.
    real_path = cost_audit.COST_LOG_JSONL_PATH
    existed = real_path.exists()
    before_hash = _sha256(real_path) if existed else None

    ret = cost_audit.main([])
    out = capsys.readouterr().out

    assert isinstance(ret, int) and ret == 0
    assert out.strip() != ""
    assert "rows read" in out  # well-formed report

    if existed:
        # Present real ledger (possibly dirty) must be unchanged by content hash;
        # a dirty ledger with findings still yields int-0 return + no exception.
        assert _sha256(real_path) == before_hash
    else:
        # Absent default path must NOT be created as a side effect.
        assert not real_path.exists()


# ===========================================================================
# c8: report is not a fixed stub — two different ledgers -> different counts.
# ===========================================================================
def test_two_ledgers_yield_different_reports(cost_audit_log_tmp, capsys):
    # Ledger A: 2 rows, all clean.
    _write_ledger(
        cost_audit_log_tmp,
        [_correct_session_row("ok"), _correct_artifact_row("ok")],
    )
    report_a = cost_audit.format_report(cost_audit.audit_cost_log())

    # Ledger B: 4 rows with a malformed line, a mismatch, and an orphan.
    bad = _correct_session_row("mm")
    bad["cost_llm_usd"] = 0.999
    _write_ledger(
        cost_audit_log_tmp,
        ["@@ nope @@", _correct_session_row("ok"), bad, _correct_artifact_row("nobody")],
    )
    report_b = cost_audit.format_report(cost_audit.audit_cost_log())

    # rows read differs in the known direction (2 -> 4).
    assert _rows_read(report_a) == 2
    assert _rows_read(report_b) == 4
    assert _rows_read(report_b) > _rows_read(report_a)

    # rows valid differs (A all clean; B has 3 offending lines -> 1 valid).
    assert _rows_valid(report_a) == 2
    assert _rows_valid(report_b) == 1
    assert _rows_valid(report_a) > _rows_valid(report_b)

    # Per-category counts differ accordingly (A all zero; B has one each).
    for cat in cost_audit.FINDING_CATEGORIES:
        assert _category_count(report_a, cat) == 0
    assert _category_count(report_b, cost_audit.CATEGORY_MALFORMED) == 1
    assert _category_count(report_b, cost_audit.CATEGORY_COST_MISMATCH) == 1
    assert _category_count(report_b, cost_audit.CATEGORY_ORPHAN_ARTIFACT) == 1


# ===========================================================================
# c9: hermeticity — the report assertions target counts and "line <n>" reasons
# rather than any absolute machine-specific path (the path header is tolerated).
# ===========================================================================
def test_report_assertions_do_not_depend_on_absolute_path(cost_audit_log_tmp):
    _write_ledger(cost_audit_log_tmp, [_correct_artifact_row("nobody")])
    report = cost_audit.format_report(cost_audit.audit_cost_log())
    # We assert on counts + line reasons; the header merely echoes the tmp path.
    assert _rows_read(report) == 1
    assert _offending_line_numbers(report) == [1]
    # The offending entry names its line, category, and reason.
    (finding,) = cost_audit.audit_cost_log().findings
    assert f"line {finding.line_number}" in report
    assert finding.category in report
