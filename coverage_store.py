"""Per-session coverage sidecars: write one, read the cross-session union.

The storage + orchestration half of the coverage feature. ``coverage_judge.py``
decides WHAT is covered; this module decides WHERE that lands on disk and how a
document's sessions merge into one number.

Design constraints (mirroring ``claims.py`` / ``study_history.py`` so the repo's
hermetic-test pattern keeps working — see CLAUDE.md "test via pure helpers, not
TestClient"):

  * Standard library + ``coverage_judge`` only at module scope. No ``bot``,
    ``app``, ``pipecat``, ``fastapi``, or ``anthropic`` import — ``import
    coverage_store`` reads no API key and performs no network I/O. The judge's
    Anthropic client is constructed lazily inside ``coverage_judge``, and tests
    inject a fake one.
  * Module-level path constants are read at CALL time so a test can monkeypatch
    ``coverage_store.TRANSCRIPTS_DIR`` to a tmp_path.
  * ``bot.py`` stays a thin caller: it hands over the ids, the claim list, and
    the transcript, and writes the two files this module hands back.

Storage shape — ``~/.voice-tutor/transcripts/<user_id>/<session_id>.coverage.json``,
beside the transcript it was judged from and sharing its filename stem. Per the
design doc the sidecar stores EVIDENCE, not conclusions: per-claim verdicts with
their cited turns, plus provenance (model, judge-prompt hash, the document's
content hash). The percentage is DERIVED at read time by :func:`union_for_document`
and is never stored as the primary record.

Document identity is the claim map's ``source_hash`` (the content hash of the
document the claims were extracted from), NOT the ``document_id`` alone: claim
ids are per-document sequentials (``c1..cN``), so a re-extracted document reuses
ids that mean something different. Sidecars are filtered on BOTH, so coverage
from a superseded claim map is ignored rather than silently merged — the
re-extraction landmine the design doc flags.
"""

import json
import os
from pathlib import Path

import coverage_judge as cj

# Read at call time (never bound at import) so tests can redirect it.
TRANSCRIPTS_DIR = Path.home() / ".voice-tutor" / "transcripts"

# Filename suffix of a per-session coverage sidecar. The stem is the session id,
# matching the transcript (``<session_id>.json``) it was judged from.
COVERAGE_SUFFIX = ".coverage.json"

# Schema version of the sidecar envelope, so a later reader can tell what it is
# looking at without guessing from field presence.
SCHEMA_VERSION = 1


def coverage_path(user_id: str, session_id: str) -> Path:
    """Path of ``user_id``'s ``session_id`` coverage sidecar.

    BOTH ids are collapsed to a single path component (mirroring
    ``claims._claims_path`` and ``session_naming.safe_session_id``) so neither
    half can traverse out of the user's own namespace — the shared choke point
    every read and write in this module funnels through, per CLAUDE.md
    "Client-controllable ids that become file paths".
    """
    stem = Path(str(session_id)).name or "unnamed"
    return TRANSCRIPTS_DIR / Path(str(user_id)).name / f"{stem}{COVERAGE_SUFFIX}"


def build_sidecar(
    *,
    session_id: str,
    user_id: str,
    document_id: str,
    source_hash: str,
    verdict_obj: dict,
    claims_total: int,
    transcript_turns: int,
) -> dict:
    """Assemble the sidecar envelope around one judge verdict object.

    Pure. ``verdict_obj`` is what :func:`coverage_judge.judge_coverage` returns
    (verdicts + judged_at/model/judge_prompt_hash/doc_id/citation_repairs); the
    envelope adds the session/user/document identity a reader needs to find and
    filter it. ``covered_count`` is a convenience for humans reading the file —
    the authoritative number is always recomputed from ``verdicts``.
    """
    verdicts = verdict_obj.get("verdicts", [])
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "user_id": user_id,
        "document_id": document_id,
        # The claim map's source_hash: the identity a union merge is keyed on.
        "source_hash": source_hash,
        "doc_id": verdict_obj.get("doc_id"),
        "claims_total": claims_total,
        "transcript_turns": transcript_turns,
        "judged_at": verdict_obj.get("judged_at"),
        "model": verdict_obj.get("model"),
        "judge_prompt_hash": verdict_obj.get("judge_prompt_hash"),
        "citation_repairs": verdict_obj.get("citation_repairs", []),
        "covered_count": sum(1 for v in verdicts if v.get("covered")),
        "verdicts": verdicts,
    }


def write_sidecar(
    user_id: str, session_id: str, sidecar: dict, *, overwrite: bool = False
) -> Path | None:
    """Persist ``sidecar`` to :func:`coverage_path`; return the path, or None if skipped.

    APPEND-ONLY BY POLICY. A written sidecar is a RECORD of what a session was
    judged to have covered, and it is never silently re-judged: this function
    refuses to overwrite an existing sidecar unless ``overwrite=True`` is passed
    explicitly, returning ``None`` instead.

    That makes the accumulated coverage number MONOTONIC BY CONSTRUCTION — the
    union can only ever grow as sessions are added. It matters because the judge
    is not perfectly reproducible (measured 2026-08-04: re-judging an unchanged
    transcript at temperature 0 varied by one claim), so a silent re-judge could
    make a user's progress bar go DOWN with no session having happened. A bar
    that retreats reads as a broken product, and it would also quietly rewrite
    the evidence an eval label was assigned against.

    The guard lives HERE, at the single choke point every writer funnels
    through, rather than at each call site — containment must not depend on
    every caller remembering (the same principle as the path-traversal guards).
    It also closes a real hole: ``session_id`` is client-supplied, so without
    this a crafted or reused id would clobber a previous session's coverage.
    ``overwrite=True`` is the one sanctioned way to re-judge (see
    ``backfill_coverage.py --force``).

    The write itself is ATOMIC (temp file + ``os.replace``), mirroring
    ``claims.write_claims``: an interrupted write must never leave a half-written
    sidecar, because the union reader would then have to distinguish "corrupt"
    from "no coverage" on a user-facing number. Creates the per-user directory if
    needed.
    """
    path = coverage_path(user_id, session_id)
    if path.exists() and not overwrite:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem
    return path


def load_sidecar(user_id: str, session_id: str) -> dict | None:
    """Return one session's coverage sidecar, or None if absent/unreadable.

    Never raises: a missing, corrupt, or unreadable sidecar means "no coverage
    data for that session", which is exactly how a coverage failure is supposed
    to degrade.
    """
    path = coverage_path(user_id, session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def iter_sidecars(user_id: str):
    """Yield every readable coverage sidecar for ``user_id``, oldest name first.

    Globs the user's transcript directory for ``*.coverage.json``. Unreadable or
    non-object files are SKIPPED, not raised on — one corrupt file must not cost
    the user their whole coverage number.
    """
    user_dir = TRANSCRIPTS_DIR / Path(str(user_id)).name
    if not user_dir.is_dir():
        return
    for path in sorted(user_dir.glob(f"*{COVERAGE_SUFFIX}")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            yield data


def union_for_document(
    user_id: str, document_id: str, source_hash: str | None = None
) -> dict:
    """Union coverage across every session ``user_id`` has run on ``document_id``.

    THE READ PATH. Collects the user's coverage sidecars for this document and
    merges them with the judge module's pure :func:`coverage_judge.union_coverage`
    — a claim is covered if ANY session covered it — then derives the percentage
    at read time. This is the number the live bar opens at on a returning
    session (the design's "starts at the accumulated number, not zero").

    ``source_hash`` — when given, ONLY sidecars judged against that exact claim
    map contribute. Claim ids are per-document sequentials, so a re-extracted
    document's ``c15`` is not the old ``c15``; silently merging across maps would
    produce a false number. Mismatched sidecars are ignored (a quiet, correct
    under-count), and their number is reported as ``stale_sessions`` so the
    condition is observable rather than invisible. Pass ``None`` to merge every
    sidecar for the document regardless of map version.

    Returns::

        {"covered_ids": [...], "percentage": <float>, "claims_total": <int>,
         "sessions": <int>, "stale_sessions": <int>, "session_ids": [...]}

    ``claims_total`` is the size of the claim map (from the sidecars; 0 when
    there are none), and ``percentage`` is 0.0 for a document with no coverage —
    never an error, never a divide-by-zero. Reads files but makes no model call.
    """
    matching: list[dict] = []
    session_ids: list[str] = []
    stale = 0
    claims_total = 0
    for sidecar in iter_sidecars(user_id):
        if sidecar.get("document_id") != document_id:
            continue
        if source_hash is not None and sidecar.get("source_hash") != source_hash:
            stale += 1
            continue
        matching.append(sidecar)
        sid = sidecar.get("session_id")
        if isinstance(sid, str):
            session_ids.append(sid)
        total = sidecar.get("claims_total")
        if isinstance(total, int) and total > claims_total:
            claims_total = total

    if not matching:
        return {
            "covered_ids": [],
            "percentage": 0.0,
            "claims_total": 0,
            "sessions": 0,
            "stale_sessions": stale,
            "session_ids": [],
        }

    # The judge module owns the merge arithmetic AND the cross-document guard.
    # Every sidecar here already agrees on document_id (+ source_hash when given),
    # so allow_unidentified covers the case where a sidecar carries no doc_id
    # stamp; a genuine cross-document mix still raises from union_coverage.
    merged = cj.union_coverage(matching, allow_unidentified=True)
    return {
        "covered_ids": merged["covered_ids"],
        "percentage": merged["percentage"],
        "claims_total": claims_total,
        "sessions": len(matching),
        "stale_sessions": stale,
        "session_ids": session_ids,
    }


# --------------------------------------------------------------------------- #
# Orchestration: judge one session and produce (sidecar, cost row).
#
# Called from bot.py's teardown, OFF the event loop (asyncio.to_thread) — the
# judge is a blocking 10-40s Haiku call. Everything here is synchronous and
# hermetically testable with an injected fake client.
# --------------------------------------------------------------------------- #

# Ledger row kind for a coverage judge call, so its spend is attributable in
# session-log.jsonl alongside kind="session" / kind="artifact" rows.
LEDGER_KIND = "coverage"


def _claim_payload(claim_texts_or_objs, source_hash: str) -> dict:
    """Build the judge's claims envelope from ``claims.Claim`` objects or dicts.

    Accepts what ``claims.load_fresh_claims`` returns (typed ``Claim`` objects,
    which carry ``id`` and ``claim``) or plain dicts, and stamps the document's
    ``source_hash`` so :func:`coverage_judge.judge_coverage` picks it up as the
    verdict set's ``doc_id`` (the cross-document merge guard's identity).
    """
    records = []
    for c in claim_texts_or_objs:
        if isinstance(c, dict):
            records.append({"id": c.get("id"), "claim": c.get("claim", c.get("text"))})
        else:
            records.append({"id": getattr(c, "id", None), "claim": getattr(c, "claim", "")})
    return {"source_hash": source_hash, "claims": records}


def judge_session(
    *,
    user_id: str,
    session_id: str,
    document_id: str,
    source_hash: str,
    claim_objs,
    transcript: dict,
    config=None,
    client=None,
) -> tuple[dict | None, dict]:
    """Judge one finished session and return ``(sidecar_or_None, cost_row)``.

    Runs ONE :func:`coverage_judge.judge_coverage` invocation (bounded internal
    retries) over the session's claim map and transcript, wraps the verdicts in
    the sidecar envelope, and tallies the spend.

    THE FAILURE CONTRACT: any judge failure returns ``(None, cost_row)`` — never
    raises — so a coverage problem degrades to NO COVERAGE DATA for that session
    and can never break the session's teardown, transcript, recap, or ledger.
    The cost row is returned EITHER WAY (a failed run that burned two attempts is
    exactly when spend spiked), carrying ``status`` and, on failure, ``error``.

    The caller writes both artifacts; this function performs no file I/O of its
    own, so a test can exercise it with no filesystem at all.
    """
    tally = cj.UsageTally()
    cfg = config or cj.JudgeConfig()
    claims_payload = _claim_payload(claim_objs, source_hash)
    claims_total = len(claims_payload["claims"])
    turns = transcript.get("turns", []) if isinstance(transcript, dict) else []

    if client is None:
        # Lazy import mirrors coverage_judge/claims: importing this module reads
        # no API key and performs no network I/O.
        import anthropic  # noqa: PLC0415 - intentional lazy import

        client = anthropic.Anthropic()
    counting = cj.CountingClient(client, tally)

    error: str | None = None
    verdict_obj: dict | None = None
    try:
        verdict_obj = cj.judge_coverage(
            claims_payload, transcript, config=cfg, client=counting
        )
    except Exception as e:  # noqa: BLE001 - degrade to no coverage, never break teardown
        error = f"{type(e).__name__}: {e}"

    cost_row = {
        "kind": LEDGER_KIND,
        "session_id": session_id,
        "user_id": user_id,
        "document_id": document_id,
        "model": cfg.model,
        "calls": tally.calls,
        "status": "ok" if verdict_obj is not None else "failed",
        "usage_complete": tally.is_complete(),
    }
    # Omit a token field no call reported, so an unobserved count is absent
    # rather than a confident 0 (the partial-measurement fix).
    if tally.calls_reporting_input:
        cost_row["input_tokens"] = tally.input_tokens
    if tally.calls_reporting_output:
        cost_row["output_tokens"] = tally.output_tokens
    if not tally.is_complete():
        cost_row["calls_reporting_input_tokens"] = tally.calls_reporting_input
        cost_row["calls_reporting_output_tokens"] = tally.calls_reporting_output
    if error is not None:
        cost_row["error"] = error
        return None, cost_row

    sidecar = build_sidecar(
        session_id=session_id,
        user_id=user_id,
        document_id=document_id,
        source_hash=source_hash,
        verdict_obj=verdict_obj,
        claims_total=claims_total,
        transcript_turns=len(turns),
    )
    cost_row["covered_count"] = sidecar["covered_count"]
    cost_row["claims_total"] = claims_total
    return sidecar, cost_row
