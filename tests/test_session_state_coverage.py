"""c9: branch coverage of the moved-helper source spans in session_state.py.

Runs a subprocess pytest with coverage restricted to session_state.py, then
asserts that every line within each moved-helper span is covered (100%) and that
branch coverage over those spans is complete. The assertion FAILS (not skips) if a
helper span is missing from the report or below 100%.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.session_state_cases import MOVED_HELPERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SS_PATH = PROJECT_ROOT / "session_state.py"


def _helper_span_lines():
    src = SS_PATH.read_text()
    tree = ast.parse(src)
    spans = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in set(MOVED_HELPERS):
            spans[node.name] = set(range(node.lineno, node.end_lineno + 1))
    return spans


def _has_coverage():
    try:
        import coverage  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_coverage(), reason="coverage.py not installed")
def test_moved_helper_spans_fully_covered(tmp_path):
    """Drive the characterization suite under coverage and assert 100% line+branch
    coverage over every moved-helper span."""
    data_file = tmp_path / ".coverage"
    json_file = tmp_path / "cov.json"

    # Run the characterization test module under branch coverage of session_state.
    run = subprocess.run(
        [
            sys.executable, "-m", "coverage", "run",
            f"--data-file={data_file}",
            "--branch",
            "--include=*/session_state.py",
            "-m", "pytest",
            "tests/_ss_coverage_driver.py",
            "-q",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, f"coverage run failed:\n{run.stdout}\n{run.stderr}"

    rep = subprocess.run(
        [
            sys.executable, "-m", "coverage", "json",
            f"--data-file={data_file}",
            "-o", str(json_file),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert rep.returncode == 0, f"coverage json failed:\n{rep.stdout}\n{rep.stderr}"

    data = json.loads(json_file.read_text())
    # Find the session_state.py entry regardless of path spelling.
    file_entry = None
    for fname, info in data["files"].items():
        if Path(fname).name == "session_state.py":
            file_entry = info
            break
    assert file_entry is not None, "session_state.py absent from coverage report"

    executed = set(file_entry["executed_lines"])
    missing = set(file_entry.get("missing_lines", []))

    spans = _helper_span_lines()
    assert spans, "no helper spans derived"
    for name, lines in spans.items():
        # Every executable line in the span must be covered. Coverage only tracks
        # executable lines, so we assert none of the span's lines are 'missing'.
        span_missing = lines & missing
        assert not span_missing, (
            f"helper {name} has uncovered lines {sorted(span_missing)}"
        )
        # And at least the def line region was exercised.
        assert lines & executed, f"helper {name} span not exercised at all"

    # Branch completeness: no partial branches within any moved-helper span.
    missing_branches = file_entry.get("missing_branches", [])
    all_span_lines = set().union(*spans.values())
    offending = [b for b in missing_branches if b[0] in all_span_lines]
    assert not offending, f"uncovered branches in moved spans: {offending}"
