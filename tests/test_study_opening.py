def test_kickoff_default_for_regular_mode(imported_bot, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    assert imported_bot.kickoff_message(study=False) == imported_bot.DEFAULT_KICKOFF_MESSAGE


def test_kickoff_study_uses_study_message_when_flag_on(imported_bot, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    assert imported_bot.kickoff_message(study=True) == imported_bot.STUDY_KICKOFF_MESSAGE


def test_kickoff_study_falls_back_when_flag_off(imported_bot, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", False)
    assert imported_bot.kickoff_message(study=True) == imported_bot.DEFAULT_KICKOFF_MESSAGE


def test_default_kickoff_is_unchanged_string(imported_bot):
    assert imported_bot.DEFAULT_KICKOFF_MESSAGE == "Say hello and introduce yourself briefly."


_DOC = {"doc_title": "Graph Engineering", "doc_text": "Graphs are nodes and edges."}


def test_opening_section_present_when_flag_on(imported_bot, session_state_tmp, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    prompt = imported_bot.build_system_instruction(study={**_DOC})
    assert "## Opening the session" in prompt


def test_opening_section_absent_when_flag_off(imported_bot, session_state_tmp, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", False)
    prompt = imported_bot.build_system_instruction(study={**_DOC})
    assert "## Opening the session" not in prompt
    # Flag-off uses the original base verbatim.
    assert prompt.startswith(imported_bot.STUDY_BASE_INSTRUCTION)


def test_flag_off_base_is_the_original_constant(imported_bot):
    # The legacy constant must remain byte-identical (hash continuity depends on it).
    assert imported_bot.STUDY_BASE_INSTRUCTION.startswith("You are a study companion")


def test_with_opening_is_derived_from_the_shipped_base(imported_bot):
    # The passive opener line is present in the original and REMOVED in the derived
    # constant; the opening section is added. This proves WITH_OPENING is exactly
    # one change off the shipped base (no freehand drift), so the first flag-on
    # hash reflects precisely the opening change.
    original = imported_bot.STUDY_BASE_INSTRUCTION
    derived = imported_bot.STUDY_BASE_INSTRUCTION_WITH_OPENING
    assert imported_bot._PASSIVE_OPENER_LINE in original
    assert imported_bot._PASSIVE_OPENER_LINE not in derived
    assert "## Opening the session" in derived
    assert derived != original  # a no-op replace would be a silent bug


_PREV_PARSED = {"covered": ["Nodes and edges", "The diamond pattern"],
                "open_threads": ["Verification architecture"]}
_PREV_FALLBACK = {"fallback_text": "Some recap prose that could not be parsed."}


def test_previously_block_before_document_and_before_claim_map(imported_bot, session_state_tmp, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    prompt = imported_bot.build_system_instruction(
        study={**_DOC, "claims": ["Claim one."], "previously": _PREV_PARSED}
    )
    prev_pos = prompt.index("# Where you left off on this document")
    doc_pos = prompt.index("## Document: Graph Engineering")
    map_pos = prompt.index("## Claim map")
    brevity_pos = prompt.index("# Reminder")
    # Position contract: previously < document < claim map < reminders.
    assert prev_pos < doc_pos < map_pos < brevity_pos
    assert "Nodes and edges" in prompt
    assert "Verification architecture" in prompt


def test_previously_block_renders_fallback_text(imported_bot, session_state_tmp, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    prompt = imported_bot.build_system_instruction(study={**_DOC, "previously": _PREV_FALLBACK})
    assert "# Where you left off on this document" in prompt
    assert "Some recap prose that could not be parsed." in prompt


def test_no_previously_block_when_absent_or_none(imported_bot, session_state_tmp, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    for study in ({**_DOC}, {**_DOC, "previously": None}):
        prompt = imported_bot.build_system_instruction(study=study)
        assert "# Where you left off" not in prompt


def test_no_previously_block_when_flag_off(imported_bot, session_state_tmp, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", False)
    prompt = imported_bot.build_system_instruction(study={**_DOC, "previously": _PREV_PARSED})
    assert "# Where you left off" not in prompt


import hashlib

# Captured from `main` BEFORE any code change — `bot.static_prompt_hash(study=True)`
# on the current tree — and cross-checked against the prompt_hash of real study
# rows in session-log.jsonl (both 2026-07-26 sessions carry this exact value). Pinned
# as a LITERAL, not recomputed from the constants: an accidental byte change to
# STUDY_BASE_INSTRUCTION/BREVITY/STUDY reminders must break this test loudly rather
# than silently breaking continuity with every historical ledger row.
PRE_CHANGE_STUDY_HASH = "4b937a122fd6b7a5297061be1d853e03833214a66de18491af667cbf13b5a3b0"


def test_flag_off_study_hash_equals_pre_change_value(imported_bot, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", False)
    # Equality against the LITERAL is the continuity guard (not a recomputation).
    assert imported_bot.static_prompt_hash(study=True) == PRE_CHANGE_STUDY_HASH


def test_flag_on_study_hash_differs_and_includes_kickoff(imported_bot, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    on_expected = hashlib.sha256(
        (imported_bot.STUDY_BASE_INSTRUCTION_WITH_OPENING
         + imported_bot.BREVITY_REMINDER
         + imported_bot.STUDY_REMINDER
         + imported_bot.STUDY_KICKOFF_MESSAGE).encode("utf-8")
    ).hexdigest()
    got = imported_bot.static_prompt_hash(study=True)
    assert got == on_expected               # flag-on folds in the new base + kickoff
    assert got != PRE_CHANGE_STUDY_HASH      # and is distinct from the historical hash


def test_regular_mode_hash_unaffected_by_flag(imported_bot, monkeypatch):
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", True)
    on = imported_bot.static_prompt_hash(study=False)
    monkeypatch.setattr(imported_bot, "SESSION_OPENING", False)
    off = imported_bot.static_prompt_hash(study=False)
    assert on == off  # non-study mode is untouched by session-opening
