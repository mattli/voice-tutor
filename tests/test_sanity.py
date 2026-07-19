"""Green-baseline sanity test for the dev-harness characterization run.

Seeded during clone prep so the harness verifier has a passing baseline before
sprint 1 adds real characterization tests. Imports `documents` (the sprint-1
target module) rather than `bot`, because `bot` pulls the full Pipecat stack
that the minimal .harness-venv deliberately does not install.
"""


def test_documents_imports():
    import documents

    assert hasattr(documents, "save_upload")
