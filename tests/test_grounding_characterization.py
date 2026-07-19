"""Characterization tests for the wiki grounding helpers relocated into
grounding.py (sprint 0: extract pure helpers and characterize).

Pins the CURRENT observed behavior of every moved helper with hardcoded
literals, exercises BOTH branches of every conditional helper, and proves the
wiki.py re-export shim is transparent to existing callers.

Hermetic: no network, no LLM, no real ~/second-brain access. grounding.WIKI_DIR
is redirected to a per-test tmp_path via the ``grounding_tmp`` fixture (patched
on the grounding module, where _load_index is DEFINED). No baseline is keyed on
git HEAD or commit state.
"""

import grounding


# The full verbatim USAGE_INSTRUCTIONS text, pinned as a literal so the
# system_prompt_block present-branch assertion is exact end-to-end.
_EXPECTED_USAGE_INSTRUCTIONS = (
    "### How to use the wiki\n\n"
    "- When you need a wiki page, say one short sentence first (e.g. "
    "\"let me pull that up\"), then call `read_wiki_page(path)`. This prevents "
    "silence during the lookup.\n"
    "- Don't open pages speculatively. Open a page only when a specific topic "
    "is central to Matt's current question. One page per question is typical "
    "— don't chain-open multiple pages unless Matt explicitly asks about "
    "multiple topics.\n"
    "- Pass the path relative to the wiki root exactly as shown in the index, "
    "e.g. 'concepts/llm-knowledge-bases.md' or 'landscape/yc-ai-thesis.md'."
)


# ---------------------------------------------------------------------------
# Constant pins (verbatim relocated values).
# ---------------------------------------------------------------------------
def test_usage_instructions_literal():
    assert grounding.USAGE_INSTRUCTIONS == _EXPECTED_USAGE_INSTRUCTIONS


def test_wiki_dir_shape():
    # WIKI_DIR = Path.home() / "second-brain" / "resources" / "wiki"
    from pathlib import Path

    assert grounding.WIKI_DIR == Path.home() / "second-brain" / "resources" / "wiki"


# ---------------------------------------------------------------------------
# _load_index — both branches, exact literals.
# ---------------------------------------------------------------------------
def test_load_index_absent_returns_empty_string(grounding_tmp):
    # No INDEX.md seeded -> exactly "" (empty string, NOT None).
    result = grounding._load_index()
    assert result == ""
    assert result is not None


def test_load_index_present_returns_file_text(grounding_tmp):
    seeded = "# Wiki index\n\n- concepts/llm-knowledge-bases.md\n"
    (grounding_tmp / "INDEX.md").write_text(seeded)
    assert grounding._load_index() == seeded


# ---------------------------------------------------------------------------
# system_prompt_block — both branches, exact literals.
# ---------------------------------------------------------------------------
def test_system_prompt_block_absent_returns_none(grounding_tmp):
    # No INDEX.md -> exactly None.
    assert grounding.system_prompt_block() is None


def test_system_prompt_block_present_exact_format(grounding_tmp):
    index = "# Wiki index\n\n- concepts/llm-knowledge-bases.md\n"
    (grounding_tmp / "INDEX.md").write_text(index)
    expected = (
        f"\n## Matt's knowledge wiki\n\n{index}\n\n{_EXPECTED_USAGE_INSTRUCTIONS}"
    )
    assert grounding.system_prompt_block() == expected


# ---------------------------------------------------------------------------
# c13: end-to-end shim proof — importing via wiki.py is transparent.
# ---------------------------------------------------------------------------
def test_wiki_shim_reexports_are_identity():
    import wiki

    assert wiki._load_index is grounding._load_index
    assert wiki.system_prompt_block is grounding.system_prompt_block
    assert wiki.WIKI_DIR is grounding.WIKI_DIR
    assert wiki.USAGE_INSTRUCTIONS is grounding.USAGE_INSTRUCTIONS


def test_wiki_shim_transparent_present_branch(grounding_tmp):
    import wiki

    index = "# Wiki index via shim\n"
    (grounding_tmp / "INDEX.md").write_text(index)
    # The same concrete output whether called via wiki or grounding.
    expected = (
        f"\n## Matt's knowledge wiki\n\n{index}\n\n{_EXPECTED_USAGE_INSTRUCTIONS}"
    )
    assert wiki.system_prompt_block() == expected
    assert wiki.system_prompt_block() == grounding.system_prompt_block()


def test_wiki_shim_transparent_absent_branch(grounding_tmp):
    import wiki

    assert wiki.system_prompt_block() is None
    assert wiki._load_index() == ""
