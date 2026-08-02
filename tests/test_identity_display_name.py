import identity


def test_display_name_titlecases_first_name_slug():
    assert identity.display_name("matt") == "Matt"
    assert identity.display_name("jorge") == "Jorge"
    assert identity.display_name("john") == "John"


def test_display_name_splits_separators_and_titlecases():
    assert identity.display_name("mary-jane") == "Mary Jane"
    assert identity.display_name("sarah_dev") == "Sarah Dev"


def test_display_name_empty_falls_back_to_generic_label():
    # user_id is always a valid slug at real call sites; the fallback is defensive
    # so a name never renders as an empty string inside a prompt.
    assert identity.display_name("") == "the user"
