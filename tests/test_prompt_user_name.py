"""The per-user display name (identity.display_name) must be threaded through the
prompts instead of the literal 'Matt'. Exercised via the ``imported_bot`` fixture
(Pipecat-stubbed if the real stack is absent) with profile/memory redirected to an
empty tmp via ``session_state_tmp`` so the assembled prompt is deterministic."""


def test_regular_base_addresses_user_by_name_not_matt(imported_bot, session_state_tmp, monkeypatch):
    monkeypatch.setattr(imported_bot, "WIKI_ENABLED", False)
    prompt = imported_bot.build_system_instruction("jorge")
    assert "let Jorge respond" in prompt
    assert "You know Jorge from prior conversations" in prompt
    assert "Matt" not in prompt


def test_wiki_tagline_uses_user_name_not_matt(imported_bot, session_state_tmp, grounding_tmp, monkeypatch):
    # WIKI_ENABLED on, but an empty tmp wiki so system_prompt_block adds nothing —
    # isolates the tagline's own wording.
    monkeypatch.setattr(imported_bot, "WIKI_ENABLED", True)
    prompt = imported_bot.build_system_instruction("jorge")
    assert "Jorge's personal knowledge wiki" in prompt
    assert "Matt" not in prompt


def test_study_memory_header_uses_user_name_not_matt(imported_bot, session_state_tmp):
    import session_state as ss

    mem = ss.memory_path("jorge")
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("- talked about graph databases\n")

    prompt = imported_bot.build_system_instruction(
        "jorge", study={"doc_title": "T", "doc_text": "D"}
    )
    assert "# Background — Jorge's prior topics" in prompt
    assert "Matt" not in prompt


def test_summary_prompt_uses_user_name_not_matt(imported_bot):
    p = imported_bot._summary_prompt("Jorge", {"turns": []})
    assert "Jorge" in p
    assert "Matt" not in p


def test_analysis_prompt_uses_user_name_not_matt(imported_bot):
    p = imported_bot._analysis_prompt("Jorge", {"session_duration_sec": 1}, [], {"turns": []})
    assert "Jorge" in p
    assert "Matt" not in p
