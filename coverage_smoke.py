"""Bounded credentialed smoke for the coverage judge (Sprint 3 acceptance).

This is the SEPARATE, one-time credentialed-smoke step — NOT part of the hermetic
verifier (``tests/test_coverage_judge.py``) and NOT run per hermetic test. It
makes REAL Haiku calls (one judge invocation per session, on the happy path) via
:func:`coverage_judge.judge_coverage`, then grades the result against the frozen
answer key ``labels.json`` and writes a committed run report the sprint's live
acceptance (c8–c12) is read from.

Run it ONCE:

    uv run --with anthropic python coverage_smoke.py

Cost: ~5 Haiku calls on the happy path (~$0.07). Retries on a malformed reply are
bounded per session by ``JudgeConfig.max_attempts``; the report records the ACTUAL
live-call count, which cannot exceed ``5 * max_attempts``.

API key handling (never echoed, logged, or committed):
  * The key is taken from the process environment ``ANTHROPIC_API_KEY`` if set.
  * Otherwise it is read from the app's ``.env`` at the ABSOLUTE path
    ``/Users/mattli/development/voice-tutor/.env`` (equivalently
    ``$HOME/development/voice-tutor/.env``) — the app's ``.env`` is gitignored and
    does NOT exist inside a harness worktree, so a worktree-relative ``./.env`` is
    never consulted.
  * The key value is passed only to the Anthropic client constructor. It is never
    printed, logged, or written into the report / cost-out.

The v2 prompt is the authored deliverable; every verdict carries the module's
recorded v2 ``judge_prompt_hash``. This script never modifies ``labels.json`` and
never tunes the prompt: if a label disagrees on principled grounds it is surfaced
in the report as a FAILURE for human adjudication, not forced into agreement.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import coverage_judge as cj

# --------------------------------------------------------------------------- #
# Fixture locations (committed under the module's test fixtures) + answer key.
# --------------------------------------------------------------------------- #

_FIXTURE_ROOT = Path(__file__).parent / "tests" / "fixtures" / "coverage"
_TRANSCRIPT_DIR = _FIXTURE_ROOT / "transcripts"
_CLAIMS_DIR = _FIXTURE_ROOT / "claims"
_LABELS_PATH = _FIXTURE_ROOT / "labels.json"

# The four PRIMARY union sessions (judged vs the 63-claim matt doc) and the
# strictness session (judged vs the 71-claim shared doc).
_PRIMARY_SESSIONS = ["f6148c26", "7beee170", "d33800bf", "bb979045"]
_STRICTNESS_SESSION = "12f3a30d"

_MATT_CLAIM_MAP = "matt-2aa66acc"       # 63-claim matt doc
_SHARED_CLAIM_MAP = "shared-ac4b826f"   # 71-claim shared doc

# The 16 upheld claims that must be covered in the PRIMARY union, plus the
# mandatory NOT-covered regression case (c30) and the strictness ceiling.
_UPHELD_COVERED = [
    "c1", "c2", "c3", "c5", "c6", "c8", "c9", "c15", "c17", "c21",
    "c27", "c28", "c31", "c46", "c47", "c48",
]
_MANDATORY_NOT_COVERED = "c30"
_STRICTNESS_MAX_COVERED = 2

_REPORT_PATH = Path(__file__).parent / "RUN_REPORT_smoke.md"
_COST_OUT_PATH = Path(__file__).parent / "coverage_smoke_cost.json"

# App .env absolute path (never a worktree-relative ./.env). Resolved from $HOME
# so it works whether the harness sets HOME to /Users/mattli or not.
_APP_ENV_PATH = Path(os.path.expanduser("~/development/voice-tutor/.env"))


class SmokeError(Exception):
    """A setup/credential failure (missing key or fixture) — distinct from a
    label DISAGREEMENT, which is reported, not raised."""


# --------------------------------------------------------------------------- #
# Credential loading — process env first, then the app .env by ABSOLUTE path.
# --------------------------------------------------------------------------- #

def _load_api_key() -> str:
    """Return the Anthropic API key from the process env or the app .env.

    Order: ``os.environ['ANTHROPIC_API_KEY']`` if present; otherwise parse the
    app ``.env`` at the ABSOLUTE path ``~/development/voice-tutor/.env`` for an
    ``ANTHROPIC_API_KEY=...`` line. The value is returned to the caller (passed
    only to the client constructor); it is NEVER printed or logged. A
    worktree-relative ``./.env`` is deliberately never read.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and key.strip():
        return key.strip()

    if not _APP_ENV_PATH.exists():
        raise SmokeError(
            "ANTHROPIC_API_KEY not in the environment and no app .env at "
            f"{_APP_ENV_PATH}"
        )
    for line in _APP_ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, sep, value = line.partition("=")
        if sep and name.strip() == "ANTHROPIC_API_KEY":
            # Strip optional surrounding quotes; never log the value.
            return value.strip().strip('"').strip("'")
    raise SmokeError(f"no ANTHROPIC_API_KEY line found in {_APP_ENV_PATH}")


# --------------------------------------------------------------------------- #
# One counting client wrapper so the report can record the ACTUAL live-call
# count (initial call + any bounded retries) across all sessions.
# --------------------------------------------------------------------------- #

class _CountingClient:
    """Wrap an Anthropic client, counting every ``messages.create`` call.

    The judge issues one call on the happy path and up to ``max_attempts`` per
    session on malformed output; the total observed count is what the report
    prints. The wrapper touches only the call boundary — no key material passes
    through it.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.messages = _CountingMessages(self)

    @property
    def inner(self):
        return self._inner


class _CountingMessages:
    def __init__(self, owner: _CountingClient):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls += 1
        return self._owner.inner.messages.create(**kwargs)


# --------------------------------------------------------------------------- #
# Loading fixtures + judging.
# --------------------------------------------------------------------------- #

def _load_transcript(session: str):
    path = _TRANSCRIPT_DIR / f"{session}.json"
    if not path.exists():
        raise SmokeError(f"missing transcript fixture: {path}")
    return json.loads(path.read_text())


def _load_claims(stem: str):
    path = _CLAIMS_DIR / f"{stem}.claims.json"
    if not path.exists():
        raise SmokeError(f"missing claim-map fixture: {path}")
    return json.loads(path.read_text())


@dataclass
class SessionResult:
    session: str
    claim_map: str
    covered_ids: list
    not_covered_count: int
    total_claims: int


def _covered_ids_from_verdict(verdict_obj: dict) -> list:
    """The sorted list of claim ids marked covered in one judge verdict object."""
    return sorted(
        v["claim_id"] for v in verdict_obj["verdicts"] if v["covered"]
    )


def _judge_session(session: str, claim_map: str, client, config) -> tuple[SessionResult, dict]:
    """Judge one session against one claim map with ONE judge invocation.

    Returns the per-session result (covered ids etc.) and the full verdict
    object (so the report can carry provenance + the v2 hash).
    """
    claims_data = _load_claims(claim_map)
    transcript_data = _load_transcript(session)
    verdict_obj = cj.judge_coverage(
        claims_data, transcript_data, config=config, client=client
    )
    covered = _covered_ids_from_verdict(verdict_obj)
    total = len(verdict_obj["verdicts"])
    return (
        SessionResult(
            session=session,
            claim_map=claim_map,
            covered_ids=covered,
            not_covered_count=total - len(covered),
            total_claims=total,
        ),
        verdict_obj,
    )


# --------------------------------------------------------------------------- #
# Report + cost writing.
# --------------------------------------------------------------------------- #

def _render_report(
    *,
    per_session: list,
    primary_union: dict,
    strictness: SessionResult,
    upheld_status: dict,
    c30_covered: bool,
    all_agree: bool,
    live_calls: int,
    max_attempts: int,
    v2_hash: str,
    model: str,
) -> str:
    """Render the Markdown run report — the artifact c8–c12 are graded from.

    Includes, per session, the RAW covered_ids returned by that judge invocation
    (so the primary union and strictness count are RECOMPUTABLE from the report),
    the per-claim covered/not-covered table for the 16 upheld + c30 labels, the
    strictness count, the total live-call count, and the v2 judge_prompt_hash.
    """
    lines: list[str] = []
    lines.append("# Run report — credentialed smoke (Sprint 3)")
    lines.append("")
    lines.append(f"**Judged at:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Model:** `{model}`")
    lines.append(f"**Judge prompt (v2) hash:** `{v2_hash}`")
    lines.append(f"**Total live judge calls:** {live_calls} "
                 f"(happy path = 5; bound = 5 × max_attempts = {5 * max_attempts})")
    verdict_word = "PASS — 17/17 label agreement" if all_agree else "FAIL — see disagreements below"
    lines.append(f"**Result:** {verdict_word}")
    lines.append("")

    lines.append("## Per-session raw covered_ids (union + strictness recomputable from here)")
    lines.append("")
    lines.append("| session | claim map | #claims | #covered | covered_ids |")
    lines.append("|---|---|---|---|---|")
    for r in per_session:
        ids = ", ".join(r.covered_ids) if r.covered_ids else "(none)"
        lines.append(
            f"| {r.session} | {r.claim_map} | {r.total_claims} | "
            f"{len(r.covered_ids)} | {ids} |"
        )
    lines.append("")

    lines.append("## PRIMARY union (four primary sessions vs the 63-claim matt map)")
    lines.append("")
    lines.append(f"- covered_ids ({len(primary_union['covered_ids'])}): "
                 f"`{', '.join(primary_union['covered_ids'])}`")
    lines.append(f"- coverage percentage (derived): {primary_union['percentage']}%")
    lines.append("")

    lines.append("### Per-claim label agreement (16 upheld + c30 regression)")
    lines.append("")
    lines.append("| claim | answer key | judge (v2) | agree |")
    lines.append("|---|---|---|---|")
    for cid in _UPHELD_COVERED:
        judged = "covered" if upheld_status[cid] else "not_covered"
        agree = "✅" if upheld_status[cid] else "❌"
        lines.append(f"| {cid} | covered | {judged} | {agree} |")
    c30_judged = "covered" if c30_covered else "not_covered"
    c30_agree = "✅" if not c30_covered else "❌"
    lines.append(f"| c30 | not_covered | {c30_judged} | {c30_agree} |")
    lines.append("")

    lines.append("## Strictness trap (session 12f3a30d vs the 71-claim shared map)")
    lines.append("")
    strict_ids = ", ".join(strictness.covered_ids) if strictness.covered_ids else "(none)"
    strict_ok = "✅" if len(strictness.covered_ids) <= _STRICTNESS_MAX_COVERED else "❌"
    lines.append(f"- covered ({len(strictness.covered_ids)} ≤ "
                 f"{_STRICTNESS_MAX_COVERED}? {strict_ok}): `{strict_ids}`")
    lines.append("")

    lines.append("## Cost")
    lines.append("")
    lines.append(f"- live Haiku judge calls: **{live_calls}** (~$0.07 at ~5 calls)")
    lines.append(f"- cost-out JSON: `{_COST_OUT_PATH.name}`")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if all_agree:
        lines.append(
            "All 17 labels agree: the 16 upheld claims are covered in the primary "
            "union, c30 (the mandatory v2 regression) is NOT covered, and the "
            "strictness trap stays within the ≤ 2 ceiling. Recall was not traded "
            "for the c30 fix. Ready to wire into the app in a later CC session."
        )
    else:
        lines.append(
            "One or more labels DISAGREE (see ❌ rows above). Per the frozen "
            "contract this is a FAILURE surfaced for human adjudication — "
            "labels.json is the frozen answer key and was NOT modified, and the "
            "v2 prompt was NOT tuned to force agreement."
        )
    lines.append("")
    return "\n".join(lines)


def _write_cost(live_calls: int, model: str) -> None:
    """Write the cost-out JSON (calls + model). No key material is included."""
    _COST_OUT_PATH.write_text(
        json.dumps(
            {"model": model, "calls": live_calls,
             "approx_usd": round(0.07 * live_calls / 5.0, 3)},
            indent=2,
        )
        + "\n"
    )


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def run() -> int:
    """Run the bounded credentialed smoke ONCE and write the run report.

    Returns 0 iff all 17 labels agree AND the strictness ceiling holds; 1 on a
    label disagreement (a surfaced FAILURE, report still written); 2 on a
    setup/credential error. NEVER echoes the API key.
    """
    try:
        api_key = _load_api_key()
    except SmokeError as e:
        print(f"coverage_smoke: setup error: {e}", file=sys.stderr)
        return 2

    import anthropic  # lazy: importing this module reads no key / does no I/O

    client = _CountingClient(anthropic.Anthropic(api_key=api_key))
    config = cj.JudgeConfig()  # defaults to Haiku, temp 0, the authored v2 prompt
    v2_hash = config.judge_prompt_hash

    per_session: list[SessionResult] = []
    primary_verdict_objs: list[dict] = []

    # The four PRIMARY sessions vs the 63-claim matt map — one call each.
    for session in _PRIMARY_SESSIONS:
        result, verdict_obj = _judge_session(session, _MATT_CLAIM_MAP, client, config)
        per_session.append(result)
        primary_verdict_objs.append(verdict_obj)
        # Provenance guard: every emitted verdict must carry the v2 hash (c12).
        if verdict_obj["judge_prompt_hash"] != v2_hash:
            raise SmokeError(
                f"{session}: verdict judge_prompt_hash "
                f"{verdict_obj['judge_prompt_hash']!r} != recorded v2 {v2_hash!r}"
            )

    # The strictness session vs the 71-claim shared map — one call.
    strict_result, strict_obj = _judge_session(
        _STRICTNESS_SESSION, _SHARED_CLAIM_MAP, client, config
    )
    per_session.append(strict_result)
    if strict_obj["judge_prompt_hash"] != v2_hash:
        raise SmokeError(
            f"{_STRICTNESS_SESSION}: verdict judge_prompt_hash "
            f"{strict_obj['judge_prompt_hash']!r} != recorded v2 {v2_hash!r}"
        )

    # Recompute the PRIMARY union from the four primary verdict objects via the
    # module's own pure union_coverage (c9/c10/c12).
    primary_union = cj.union_coverage(primary_verdict_objs)
    covered_set = set(primary_union["covered_ids"])

    upheld_status = {cid: (cid in covered_set) for cid in _UPHELD_COVERED}
    c30_covered = _MANDATORY_NOT_COVERED in covered_set

    all_upheld_ok = all(upheld_status.values())
    c30_ok = not c30_covered
    strictness_ok = len(strict_result.covered_ids) <= _STRICTNESS_MAX_COVERED
    all_agree = all_upheld_ok and c30_ok and strictness_ok

    report = _render_report(
        per_session=per_session,
        primary_union=primary_union,
        strictness=strict_result,
        upheld_status=upheld_status,
        c30_covered=c30_covered,
        all_agree=all_agree,
        live_calls=client.calls,
        max_attempts=config.max_attempts,
        v2_hash=v2_hash,
        model=config.model,
    )
    _REPORT_PATH.write_text(report + "\n")
    _write_cost(client.calls, config.model)

    # Console summary (no key material). Small and explicit so the operator sees
    # the pass/fail without opening the report.
    print(f"live judge calls: {client.calls}")
    print(f"primary union covered ({len(primary_union['covered_ids'])}): "
          f"{', '.join(primary_union['covered_ids'])}")
    print(f"c30 covered? {c30_covered} (must be False)")
    print(f"strictness 12f3a30d covered: {len(strict_result.covered_ids)} "
          f"(must be <= {_STRICTNESS_MAX_COVERED})")
    if all_agree:
        print("RESULT: PASS — 17/17 label agreement")
        return 0
    missing = [cid for cid in _UPHELD_COVERED if not upheld_status[cid]]
    print("RESULT: FAIL — surfaced for human adjudication", file=sys.stderr)
    if missing:
        print(f"  upheld labels NOT covered: {', '.join(missing)}", file=sys.stderr)
    if not c30_ok:
        print("  c30 was covered (regression NOT fixed)", file=sys.stderr)
    if not strictness_ok:
        print(f"  strictness exceeded: {len(strict_result.covered_ids)} covered",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(run())
