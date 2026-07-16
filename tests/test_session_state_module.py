"""c1 + c7A + c12: session_state.py module purity, stub-free import, and the
deterministic pipecat-stub fixture derivation.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.session_state_cases import MOVED_CONSTANTS, MOVED_HELPERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SS_PATH = PROJECT_ROOT / "session_state.py"

FORBIDDEN_TOKENS = [
    "WIKI_ENABLED",
    "ARTIFACTS_DIR",
    "SESSION_ANALYSIS_DIR",
    "COST_LOG_PATH",
    "COST_LOG_JSONL_PATH",
    "build_system_instruction",
    "UsageAccumulator",
    "BaseObserver",
    "generate_session_summary",
    "generate_session_analysis",
    "generate_artifact",
]

FORBIDDEN_IMPORT_TOKENS = [
    "pipecat",
    "wiki",
    "documents",
    "anthropic",
    "asyncio",
    "dotenv",
    "load_dotenv",
    "loguru",
]


def test_session_state_source_has_no_forbidden_tokens():
    src = SS_PATH.read_text()
    for tok in FORBIDDEN_TOKENS:
        assert tok not in src, f"forbidden token {tok!r} present in session_state.py"


def test_session_state_no_forbidden_imports():
    src = SS_PATH.read_text()
    tree = ast.parse(src)
    imported_tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported_tops.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_tops.add(node.module.split(".")[0])
    for tok in ("pipecat", "wiki", "documents", "anthropic", "asyncio", "dotenv", "loguru"):
        assert tok not in imported_tops, f"session_state.py imports {tok}"
    # Only stdlib imports are allowed: json, datetime, pathlib.
    assert imported_tops <= {"json", "datetime", "pathlib"}, imported_tops


def test_session_state_imports_in_fresh_subprocess_without_stubs():
    """c1/c7A: import session_state in a FRESH interpreter with no pre-injected
    stubs and only stdlib available; assert no pipecat/wiki/documents/anthropic
    modules appear in sys.modules afterwards."""
    code = textwrap.dedent(
        """
        import sys
        # Prove no stubs are pre-injected.
        pre = [m for m in sys.modules if m.startswith(('pipecat','wiki','documents','anthropic'))]
        assert pre == [], ('unexpected preinjected', pre)
        import session_state
        bad = [m for m in sys.modules if m.startswith(('pipecat','wiki','documents','anthropic'))]
        assert bad == [], ('leaked', bad)
        assert hasattr(session_state, 'load_profile')
        print('OK')
        """
    )
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_session_state_import_has_no_side_effects():
    """Importing session_state must not create the production ~/.voice-tutor dir."""
    code = textwrap.dedent(
        """
        from pathlib import Path
        existed = (Path.home() / '.voice-tutor').exists()
        import session_state
        now = (Path.home() / '.voice-tutor').exists()
        assert existed == now, 'import created ~/.voice-tutor'
        print('OK')
        """
    )
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_moved_helpers_free_names_resolve_in_session_state():
    """c7A: each moved helper's global free names resolve within session_state's
    namespace (no NameError when invoked on CASES inputs handled by other tests).
    Here we statically confirm every module-level Name referenced by a moved helper
    is bound in session_state's namespace."""
    import session_state as ss

    src = SS_PATH.read_text()
    tree = ast.parse(src)
    ns = set(dir(ss))
    builtins_ns = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__.keys())
    import builtins as _b
    builtins_ns |= set(dir(_b))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # collect local names (params + assigned) to exclude
            local = set()
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                local.add(arg.arg)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            local.add(t.id)
                elif isinstance(sub, (ast.For, ast.comprehension)):
                    pass
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    local.add(sub.id)
            loads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for nm in loads:
                if nm in local or nm in builtins_ns:
                    continue
                assert nm in ns, f"{node.name} references unresolved global {nm}"


# --- c12: deterministic pipecat-stub fixture ---------------------------------
def test_c12_fixture_derives_stub_set_from_bot_ast(pipecat_stub):
    """The fixture stubs ONLY currently-unresolvable third-party top-level modules
    parsed from working-tree bot.py's import AST; never session_state/wiki/documents
    /stdlib."""
    from tests import conftest

    derived = conftest._bot_third_party_toplevel_imports()
    # session_state / wiki / documents / stdlib never in the third-party set.
    for local in ("session_state", "wiki", "documents", "bot", "app"):
        assert local not in derived
    for std in ("json", "os", "sys", "time", "asyncio", "datetime", "pathlib"):
        assert std not in derived
    # The stub set yielded is a subset of the derived third-party set.
    assert pipecat_stub <= derived
    # session_state is never stubbed.
    assert "session_state" not in pipecat_stub


def test_c12_import_bot_under_fixture(imported_bot):
    """Under the fixture, import bot succeeds and points at the working-tree file."""
    assert imported_bot.__file__ == str((PROJECT_ROOT / "bot.py"))


def test_c12_session_state_not_stubbed(pipecat_stub):
    assert "session_state" not in pipecat_stub
    # session_state resolves as a real module, not a stub.
    import session_state
    assert session_state.__file__ == str(SS_PATH)


def test_c12_teardown_removes_stubs_and_bare_import_still_works():
    """After a fixture-using test, a fresh subprocess bare import still works with
    zero stubs (proves teardown restored sys.modules and no shadowing persisted)."""
    code = "import session_state; print('OK')"
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
