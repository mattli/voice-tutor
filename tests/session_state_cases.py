"""Shared input corpus + manifest for the session_state.py characterization suite.

This is the ONLY input source for c8-c10. Each MOVED_HELPERS name maps to a list
of concrete argument tuples (representative + edge/error cases). Filesystem-touching
helpers take no positional args (they read module-level Path constants redirected by
the ``session_state_tmp`` fixture); their CASES entry documents the seeded-state
scenarios exercised in the branch-coverage tests.

The manifest literals MOVED_HELPERS / MOVED_CONSTANTS are re-derived and pinned to
hardcoded ground-truth in test_session_state_manifest.py.
"""

# --- Manifest: exact names relocated verbatim from git-HEAD bot.py -----------
MOVED_HELPERS = [
    "load_profile",
    "_format_memory_date",
    "append_to_memory",
    "load_memory",
    "_format_session_time",
    "_format_full_transcript_block",
    "load_most_recent_transcript_block",
]

MOVED_CONSTANTS = [
    "PROFILE_PATH",
    "MEMORY_PATH",
    "TRANSCRIPTS_DIR",
    "VOICE_TUTOR_DIR",
]

# --- Fixed sample data -------------------------------------------------------
# A representative transcript used across formatter / block tests.
SAMPLE_TRANSCRIPT = {
    "session_start": "2026-04-14T09:05:00",
    "turns": [
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Hello there"},
    ],
}

# A transcript whose start lands at midnight and noon boundaries.
MIDNIGHT_TRANSCRIPT = {
    "session_start": "2026-01-01T00:07:00",
    "turns": [{"role": "user", "content": "early"}],
}
NOON_TRANSCRIPT = {
    "session_start": "2026-12-25T12:30:00",
    "turns": [{"role": "assistant", "content": "noon"}],
}

# --- CASES: helper name -> list of concrete argument tuples ------------------
# For FS helpers the tuple is empty () and the scenario is set up by the test via
# the hermetic tmp-dir fixture; the entries here enumerate the representative
# invocations so c10 can call both import paths uniformly.
CASES = {
    "load_profile": [()],
    "load_memory": [()],
    "append_to_memory": [
        (SAMPLE_TRANSCRIPT, "- discussed X\n- decided Y"),
        (SAMPLE_TRANSCRIPT, "   surrounding whitespace   "),
    ],
    "load_most_recent_transcript_block": [()],
    "_format_memory_date": [
        ("2026-04-14T09:05:00",),
        ("2026-01-01T00:07:00",),   # midnight -> 12 AM
        ("2026-12-25T12:30:00",),   # noon -> 12 PM
        ("2026-07-04T15:09:00",),   # afternoon -> PM
    ],
    "_format_session_time": [
        ("2026-04-14T09:05:00",),
        ("2026-01-01T00:07:00",),   # midnight -> 12am
        ("2026-12-25T12:30:00",),   # noon -> 12pm
        ("2026-07-04T15:09:00",),   # afternoon -> pm
    ],
    "_format_full_transcript_block": [
        (SAMPLE_TRANSCRIPT,),
        (SAMPLE_TRANSCRIPT, " (most recent)"),
        ({"session_start": "2026-04-14T09:05:00", "turns": []},),
    ],
}
