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
