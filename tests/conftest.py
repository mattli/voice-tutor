"""Fixtures for the documents.py characterization suite.

These fixtures ONLY provide (a) a hermetic redirection of the documents
directory to a per-test pytest ``tmp_path`` and (b) a guard proving the real
production documents directory is never touched. No ``sys.path`` manipulation,
no production-module shimming, and no autouse behavior-altering patches live
here.
"""

import hashlib

import pytest


def _snapshot(root):
    """Return {relative_posix_path: sha256_hex} for every file under ``root``.

    Missing directories snapshot to an empty mapping. Content-based (not mtime)
    so we can prove the real documents dir is byte-for-byte unchanged.
    """
    snap = {}
    if not root.exists():
        return snap
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            snap[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snap


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    """Redirect documents.DOCUMENTS_DIR to a per-test tmp_path.

    ``documents.py`` resolves the storage directory by reading the module-level
    ``DOCUMENTS_DIR`` attribute at call time inside save_upload/list_documents/
    load_document, so patching that attribute is the real resolution path.

    monkeypatch.setattr guarantees automatic teardown restoring the original
    module-level value after each test.
    """
    import documents

    real_dir = documents.DOCUMENTS_DIR
    before = _snapshot(real_dir)

    target = tmp_path / "documents"
    monkeypatch.setattr(documents, "DOCUMENTS_DIR", target)

    yield target

    # The real production documents dir must be byte-for-byte identical.
    after = _snapshot(real_dir)
    assert after == before, "production documents dir was mutated by a test"


@pytest.fixture
def sample_pdf_bytes():
    """Raw bytes of the committed, real PDF fixture (tests/fixtures/sample.pdf)."""
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "sample.pdf"
    return fixture.read_bytes()


# ---------------------------------------------------------------------------
# session_state.py characterization support (sprint 0: extract pure helpers)
# ---------------------------------------------------------------------------

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOT_PATH = PROJECT_ROOT / "bot.py"

# Project-local modules must NEVER be stubbed — they are imported for real.
_PROJECT_LOCAL = {"session_state", "wiki", "documents", "bot", "app"}


def _stdlib_names():
    """Top-level module names known to be part of the standard library."""
    names = set(getattr(sys, "stdlib_module_names", set()))
    # Fallbacks for names that appear in bot.py's imports.
    names |= {
        "asyncio", "json", "os", "sys", "time", "datetime", "pathlib",
        "types", "ast", "importlib", "subprocess", "hashlib", "locale",
    }
    return names


def _bot_third_party_toplevel_imports():
    """Parse working-tree bot.py's import AST → set of top-level third-party module names.

    Excludes stdlib and project-local modules (session_state/wiki/documents/app/bot).
    Only module-scope imports are considered (function-local imports like the
    ``from pipecat.runner.run import main`` inside ``__main__`` are not needed to
    execute bot's module body).
    """
    src = BOT_PATH.read_text()
    tree = ast.parse(src)
    stdlib = _stdlib_names()
    third_party = set()
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in stdlib and top not in _PROJECT_LOCAL:
                    third_party.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                continue
            top = (node.module or "").split(".")[0]
            if top and top not in stdlib and top not in _PROJECT_LOCAL:
                third_party.add(top)
    return third_party


def _submodule_paths_for(top):
    """Return every dotted module path under ``top`` that bot.py imports.

    Needed so a stub package exposes the exact submodules bot.py reaches
    (e.g. pipecat.frames.frames), plus the ``fromlist`` attribute names.
    """
    src = BOT_PATH.read_text()
    tree = ast.parse(src)
    paths = {}  # dotted module path -> set of attribute names imported from it
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == top:
                    parts = alias.name.split(".")
                    for i in range(1, len(parts) + 1):
                        paths.setdefault(".".join(parts[:i]), set())
        elif isinstance(node, ast.ImportFrom) and not node.level:
            mod = node.module or ""
            if mod.split(".")[0] == top:
                parts = mod.split(".")
                for i in range(1, len(parts) + 1):
                    paths.setdefault(".".join(parts[:i]), set())
                for alias in node.names:
                    paths[mod].add(alias.name)
    return paths


class _StubModule(ModuleType):
    """A permissive stand-in module.

    Any attribute access returns a trivial dummy class/callable so that
    ``from x import Y`` succeeds and ``class Foo(Y)`` (e.g. BaseObserver)
    is a usable base class. These stubs make NO behavioral claim — they exist
    only so bot's module body can execute for the re-export/identity proofs.
    """

    def __getattr__(self, name):
        # Return a trivial class usable as a base class or callable.
        dummy = type(name, (object,), {})
        setattr(self, name, dummy)
        return dummy


def _derive_unresolvable_stub_set():
    """Third-party top-level modules bot.py imports that DON'T resolve here."""
    unresolvable = set()
    for top in _bot_third_party_toplevel_imports():
        if top in sys.modules:
            continue
        try:
            importlib.import_module(top)
        except Exception:
            unresolvable.add(top)
    return unresolvable


@pytest.fixture
def pipecat_stub(monkeypatch):
    """Install lightweight stand-ins for exactly the currently-unresolvable
    third-party top-level modules bot.py imports, enabling ``import bot`` under a
    Pipecat-free environment. No-op if the real stack is installed.

    NEVER stubs session_state/wiki/documents/stdlib. Removes injected stubs on
    teardown (monkeypatch restores sys.modules) so session_state's stub-free
    import and cross-test isolation are preserved.

    Yields the set of module names it stubbed (possibly empty).
    """
    stub_tops = _derive_unresolvable_stub_set()
    injected = []
    for top in stub_tops:
        assert top not in _PROJECT_LOCAL, top
        for dotted in sorted(_submodule_paths_for(top), key=lambda s: s.count(".")):
            if dotted not in sys.modules:
                mod = _StubModule(dotted)
                monkeypatch.setitem(sys.modules, dotted, mod)
                injected.append(dotted)
    # Ensure bot is (re)imported fresh under whatever module set is now active.
    monkeypatch.delitem(sys.modules, "bot", raising=False)
    yield set(stub_tops)
    # monkeypatch teardown restores sys.modules (removes injected stubs, and the
    # 'bot' entry) automatically.
    for dotted in injected:
        pass  # explicit no-op; teardown handled by monkeypatch.setitem


@pytest.fixture
def imported_bot(pipecat_stub):
    """Import bot under the deterministic pipecat-stub fixture and return it."""
    import importlib as _il

    if "bot" in sys.modules:
        del sys.modules["bot"]
    bot = _il.import_module("bot")
    return bot


@pytest.fixture
def deterministic_locale():
    """Pin LC_TIME to a deterministic locale so %B / AM-PM assertions are stable.

    Tries 'C' then 'en_US.UTF-8'; if neither is available the test using this
    fixture is skipped rather than allowed to be environment-flaky.
    """
    import locale as _locale

    saved = _locale.setlocale(_locale.LC_TIME)
    chosen = None
    for cand in ("C", "en_US.UTF-8", "C.UTF-8"):
        try:
            _locale.setlocale(_locale.LC_TIME, cand)
            chosen = cand
            break
        except _locale.Error:
            continue
    if chosen is None:
        pytest.skip("no deterministic LC_TIME locale available")
    yield chosen
    _locale.setlocale(_locale.LC_TIME, saved)


def _ss_snapshot(root):
    """Content snapshot of ``root`` (reuses the sha256 helper semantics)."""
    return _snapshot(root)


@pytest.fixture
def session_state_tmp(tmp_path, monkeypatch):
    """Hermetically redirect session_state's module-level Path constants to tmp.

    Patches VOICE_TUTOR_DIR / TRANSCRIPTS_DIR / PROFILE_PATH / MEMORY_PATH on the
    session_state module (the namespace the moved helpers close over). Guards that
    the real ~/.voice-tutor directory is never created or mutated.

    Yields the tmp root so tests can seed profile/memory/transcript files.
    """
    import session_state as ss

    real_root = Path.home() / ".voice-tutor"
    before = _ss_snapshot(real_root)
    real_existed = real_root.exists()

    root = tmp_path / ".voice-tutor"
    monkeypatch.setattr(ss, "VOICE_TUTOR_DIR", root)
    monkeypatch.setattr(ss, "TRANSCRIPTS_DIR", root / "transcripts")
    monkeypatch.setattr(ss, "PROFILE_PATH", root / "profile.md")
    monkeypatch.setattr(ss, "MEMORY_PATH", root / "memory.md")

    yield root

    # Real production dir must be untouched (not created, not mutated).
    assert real_root.exists() == real_existed, "test created/removed ~/.voice-tutor"
    after = _ss_snapshot(real_root)
    assert after == before, "production ~/.voice-tutor was mutated by a test"
