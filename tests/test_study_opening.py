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
