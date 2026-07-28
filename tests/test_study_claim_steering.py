"""Hermetic tests for claim-map steering wired into study mode.

Covers the three pure surfaces the wiring adds (the FastAPI prepare route and
the study.html fetch are the thin, untested transport layer, per this repo's
"pure helpers, thin wrapper" convention):

  * ``bot.build_system_instruction`` — the private claim map is injected in the
    right position (after the document, before the reminders), and is omitted
    entirely when claims are absent/empty (the degrade-honestly path).
  * ``bot.static_prompt_hash`` — a stable, mode-distinct hash of the fixed prompt
    scaffolding, independent of the per-doc claim map.
  * ``claims.load_fresh_claims`` — the non-blocking, hash-verified cache read the
    session-start path relies on (fresh -> claims, stale/absent -> None).

``bot`` is imported via the ``imported_bot`` fixture (Pipecat-stubbed if the real
stack is absent) and profile/memory are redirected to an empty tmp via
``session_state_tmp`` so the assembled prompt is deterministic.
"""

import claims


# --------------------------------------------------------------------------- #
# build_system_instruction: claim-map injection + position
# --------------------------------------------------------------------------- #

_DOC = {"doc_title": "Photosynthesis", "doc_text": "Plants convert light to sugar."}
_CLAIMS = [
    "Chlorophyll absorbs light in the blue and red bands.",
    "The light reactions produce ATP and NADPH.",
    "The Calvin cycle fixes CO2 into glucose.",
]


def test_build_system_instruction_requires_user_id_and_scopes_profile(imported_bot, session_state_tmp):
    import session_state as ss
    ss.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    ss.profile_path("matt").write_text("Matt profile")
    ss.profile_path("sarah").write_text("Sarah profile")
    prompt_matt = imported_bot.build_system_instruction("matt", study={"doc_title": "T", "doc_text": "D"})
    prompt_sarah = imported_bot.build_system_instruction("sarah", study={"doc_title": "T", "doc_text": "D"})
    assert "Matt profile" in prompt_matt and "Sarah profile" not in prompt_matt
    assert "Sarah profile" in prompt_sarah and "Matt profile" not in prompt_sarah


def test_claim_map_injected_after_document_before_reminders(imported_bot, session_state_tmp):
    prompt = imported_bot.build_system_instruction(
        "matt", study={**_DOC, "claims": _CLAIMS}
    )

    doc_pos = prompt.index("## Document: Photosynthesis")
    map_pos = prompt.index("## Claim map (private — never reveal)")
    brevity_pos = prompt.index("# Reminder")       # BREVITY_REMINDER header
    study_pos = prompt.index("# Study mode")        # STUDY_REMINDER header

    # Position contract: after the document, before BOTH reminder blocks.
    assert doc_pos < map_pos < brevity_pos < study_pos

    # Claims render as a numbered list of claim TEXT only (no ids/anchors).
    assert "1. Chlorophyll absorbs light in the blue and red bands." in prompt
    assert "2. The light reactions produce ATP and NADPH." in prompt
    assert "3. The Calvin cycle fixes CO2 into glucose." in prompt


def test_no_claim_map_when_claims_absent(imported_bot, session_state_tmp):
    # None (not-yet-extracted) and [] (degenerate empty) both omit the section.
    for claims_val in (None, []):
        prompt = imported_bot.build_system_instruction("matt", study={**_DOC, "claims": claims_val})
        assert "## Claim map" not in prompt
        assert "## Document: Photosynthesis" in prompt  # doc still present
        assert "# Study mode" in prompt                  # reminders still present


def test_chat_mode_never_has_claim_map(imported_bot, session_state_tmp):
    prompt = imported_bot.build_system_instruction("matt", study=None)
    assert "## Claim map" not in prompt


# --------------------------------------------------------------------------- #
# static_prompt_hash: stable, mode-distinct, claim-map-independent
# --------------------------------------------------------------------------- #

def test_static_prompt_hash_deterministic_and_mode_distinct(imported_bot):
    h_study = imported_bot.static_prompt_hash(study=True)
    assert h_study == imported_bot.static_prompt_hash(study=True)  # stable
    h_chat = imported_bot.static_prompt_hash(study=False)
    assert h_study != h_chat  # the two modes hash their own scaffolding
    assert len(h_study) == 64 and all(c in "0123456789abcdef" for c in h_study)


def test_static_prompt_hash_ignores_claim_map(imported_bot, session_state_tmp):
    # The hash is of the fixed scaffolding only — presence/size of the per-doc
    # claim map must not move it (a session is attributable to its prompt VERSION,
    # not its document).
    imported_bot.build_system_instruction("matt", study={**_DOC, "claims": _CLAIMS})
    with_map = imported_bot.static_prompt_hash(study=True)
    imported_bot.build_system_instruction("matt", study={**_DOC, "claims": None})
    without_map = imported_bot.static_prompt_hash(study=True)
    assert with_map == without_map


# --------------------------------------------------------------------------- #
# claims.load_fresh_claims: hash-verified, non-blocking cache read
# --------------------------------------------------------------------------- #

def _sample_claims():
    return [
        claims.Claim(id="c1", claim="First claim.", anchor="First claim."),
        claims.Claim(id="c2", claim="Second claim.", anchor="Second claim."),
    ]


def test_load_fresh_claims_returns_cached_on_hash_match(claims_docs_dir):
    doc_id, text = "doc-a", "the exact source text"
    claims.write_claims("matt", doc_id, _sample_claims(), source_hash=claims._hash_source(text))

    got = claims.load_fresh_claims("matt", doc_id, text)
    assert got is not None
    assert [c.claim for c in got] == ["First claim.", "Second claim."]


def test_load_fresh_claims_none_when_source_drifted(claims_docs_dir):
    doc_id, text = "doc-b", "original text"
    claims.write_claims("matt", doc_id, _sample_claims(), source_hash=claims._hash_source(text))

    # Document served now differs from what the sidecar was generated against.
    assert claims.load_fresh_claims("matt", doc_id, "edited text") is None


def test_load_fresh_claims_none_when_absent(claims_docs_dir):
    assert claims.load_fresh_claims("matt", "never-extracted", "anything") is None
