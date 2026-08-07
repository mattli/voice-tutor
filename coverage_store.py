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
import sys
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
    session_start: str | None = None,
) -> dict:
    """Assemble the sidecar envelope around one judge verdict object.

    Pure. ``verdict_obj`` is what :func:`coverage_judge.judge_coverage` returns
    (verdicts + judged_at/model/judge_prompt_hash/doc_id/citation_repairs); the
    envelope adds the session/user/document identity a reader needs to find and
    filter it. ``covered_count`` is a convenience for humans reading the file —
    the authoritative number is always recomputed from ``verdicts``.

    ``session_start`` is WHEN THE SESSION RAN, and it is a distinct field from
    ``judged_at`` (when the judge ran) precisely because backfill makes the two
    diverge. It is the key every historical read orders by; storing it here
    makes a sidecar self-describing, and :func:`_session_time_of` falls back to
    the transcript for sidecars written before this field existed.
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
        "session_start": session_start,
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


def _warn(message: str) -> None:
    """Emit a warning about a skipped sidecar to stderr.

    A skipped file is a DEFECT that silently costs coverage, so it must leave a
    trace an operator can find — a counter alone tells you something was dropped
    but never which file, which is the one thing needed to fix it. stderr + a
    ``[coverage]`` tag matches how bot.py logs its own coverage failures.
    """
    print(f"[coverage] WARNING {message}", file=sys.stderr, flush=True)


def _iter_sidecar_files(user_id: str):
    """Yield ``(path, sidecar)`` for every readable coverage sidecar of ``user_id``.

    The path half exists so a caller that rejects a sidecar can NAME it in the
    log. Unreadable or non-object files are skipped here (and warned about) —
    they cannot be attributed to a document, so they are never counted in any
    per-document tally; ``union_for_document``'s ``invalid_sessions`` counts only
    files that ARE this document's and are structurally broken.
    """
    user_dir = TRANSCRIPTS_DIR / Path(str(user_id)).name
    if not user_dir.is_dir():
        return
    for path in sorted(user_dir.glob(f"*{COVERAGE_SUFFIX}")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            _warn(f"unreadable coverage sidecar skipped: {path} ({e})")
            continue
        if not isinstance(data, dict):
            _warn(f"coverage sidecar is not an object, skipped: {path}")
            continue
        yield path, data


def iter_sidecars(user_id: str):
    """Yield every readable coverage sidecar for ``user_id``, oldest name first.

    Globs the user's transcript directory for ``*.coverage.json``. Unreadable or
    non-object files are SKIPPED, not raised on — one corrupt file must not cost
    the user their whole coverage number.
    """
    for _path, data in _iter_sidecar_files(user_id):
        yield data


def _transcript_session_start(user_id: str, session_id: str) -> str:
    """``session_start`` from the transcript beside a sidecar, or ``""``.

    The retroactive half of :func:`_session_time_of`. Sidecars written before
    the envelope carried ``session_start`` have no stamp of their own, but the
    transcript they were judged FROM sits in the same directory under the same
    stem and has always recorded it — so ordering is recoverable for every
    sidecar that already exists, with no migration.
    """
    path = coverage_path(user_id, session_id).with_name(
        f"{Path(str(session_id)).name or 'unnamed'}.json"
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    value = data.get("session_start")
    return value if isinstance(value, str) else ""


def _session_time_of(user_id: str, sidecar: dict) -> str:
    """When the session a sidecar records actually HAPPENED; ``""`` if unknown.

    THE ORDERING KEY for every historical question this module answers, and
    deliberately NOT ``judged_at``. ``judged_at`` is when the JUDGE RAN, which
    equals session order only for sessions judged live at teardown:
    ``backfill_coverage.py`` stamps every session it re-judges with the time of
    the backfill, so a month-old session comes back claiming to be the newest
    thing that ever happened. Ordering on it made a past session's meter show a
    number that had nothing to do with that session (measured on live data
    2026-08-06: a session whose true standing was 16/63 rendered as 1/63,
    because five earlier sessions were backfilled the day AFTER it was judged).

    ``session_start`` is also the field ``sessions.list_study_sessions`` sorts
    the history list by, so the meter and the list it appears in are now keyed
    on the same value and cannot disagree about what "earlier" means.

    Prefers the sidecar's own stamp and falls back to the transcript. Note both
    are NAIVE LOCAL timestamps while ``judged_at`` is UTC-with-offset — never
    mix the two in one comparison; a missing value returns ``""`` (sorts oldest)
    rather than falling back to ``judged_at``, which would silently compare a
    naive string against an offset-bearing one.
    """
    value = sidecar.get("session_start")
    if isinstance(value, str) and value.strip():
        return value.strip()
    session_id = sidecar.get("session_id")
    if isinstance(session_id, str) and session_id:
        return _transcript_session_start(user_id, session_id)
    return ""


def _order_key(user_id: str, sidecar: dict) -> tuple[str, str]:
    """Total order over a document's sessions: ``(session time, session id)``.

    The session id is a tie-break, not a meaningful ordering — it exists so the
    order is TOTAL. Without it, two sidecars sharing a session time (or both
    missing one) are incomparable, and the snapshot and the delta then disagree
    about whether such a pair counts as "before": a snapshot asking "at or
    before" would include a tie while a delta asking "strictly before" excluded
    it, so the two halves of the same screen would be computed against different
    session sets. With a total order, ``at_session`` minus ``prior`` is exactly
    this session's own contribution in every case.
    """
    sid = sidecar.get("session_id")
    return (_session_time_of(user_id, sidecar), sid if isinstance(sid, str) else "")


def _load_entries(user_id: str) -> list[tuple[Path, dict, tuple[str, str]]]:
    """Every readable sidecar with its ordering key resolved, ONE scan.

    Resolving here rather than inside the merge keeps :func:`_union_from` pure
    (the transcript fallback is file I/O) and keeps the cost to one pass even
    when a caller merges the same sidecars for many documents
    (:func:`documents_view`).
    """
    return [
        (path, sidecar, _order_key(user_id, sidecar))
        for path, sidecar in _iter_sidecar_files(user_id)
    ]


def _map_identity_of(sidecar: dict) -> str:
    """The claim-map version a sidecar was judged against, as a grouping key.

    Prefers the verdict set's own ``doc_id`` stamp (what
    ``coverage_judge.union_coverage``'s cross-document guard keys on) and falls
    back to the envelope's ``source_hash``. Both are the same value in practice —
    ``build_sidecar`` stamps them from one source — so the fallback only matters
    for a hand-written or pre-stamp sidecar. Returns ``""`` for a sidecar that
    declares neither, which then groups with its own kind rather than silently
    joining an identified group.
    """
    for key in ("doc_id", "source_hash"):
        value = sidecar.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _newest_map_group(entries: list) -> tuple[list, int]:
    """Split sidecars by claim-map version, keeping the NEWEST group.

    Returns ``(kept, dropped_count)``. "Newest" is the group containing the most
    recent ``judged_at`` (ISO-8601 strings, so lexicographic order is
    chronological); ties break toward the larger group, then by key, so the
    result is deterministic. A sidecar with no ``judged_at`` sorts oldest.

    ``judged_at`` — NOT the session time :func:`_order_key` uses — is the right
    key for THIS question, and the difference is not an oversight. The question
    here is "which claim map is the current one?", and the judge always runs
    against the map that was current when it ran: ``backfill_coverage.py``
    refuses a stale map outright, so a re-judged month-old session carries
    TODAY's map. Ranking those groups by session time would let a group of
    backfilled old sessions look older than the superseded map it replaced, and
    pick the superseded one. Ranking by when the judging happened cannot.

    This is the ``source_hash=None`` path. Merging claim ids ACROSS map versions
    would be a false number (ids are per-document sequentials — a re-extracted
    ``c15`` is not the old ``c15``), and raising is what finding 3 was, so the
    remaining honest option is to answer from ONE version and report the rest as
    stale.
    """
    if len(entries) <= 1:
        return entries, 0
    groups: dict[str, list] = {}
    for entry in entries:
        groups.setdefault(_map_identity_of(entry[1]), []).append(entry)
    if len(groups) == 1:
        return entries, 0

    def _rank(item):
        key, members = item
        newest = max(
            (m.get("judged_at") if isinstance(m.get("judged_at"), str) else "")
            for _p, m, *_ in members
        )
        return (newest, len(members), key)

    winner_key, kept = max(groups.items(), key=_rank)
    return kept, sum(len(v) for k, v in groups.items() if k != winner_key)


def _union_from(
    sidecars: list,
    document_id: str,
    source_hash: str | None,
    as_of: tuple[str, str] | None = None,
) -> dict:
    """Pure merge half of :func:`union_for_document` — no file I/O of its own.

    ``sidecars`` is a list of ``(path, sidecar, order_key)`` triples as produced
    by :func:`_load_entries`; the key is resolved there because the transcript
    fallback needs the filesystem and this half must stay pure.

    Split out so one directory scan can serve many documents
    (:func:`documents_view`) and so the degradation rules are testable without a
    filesystem.

    ``as_of`` — an ``(session time, session id)`` key from :func:`_order_key`.
    Only sessions AT OR BEFORE it in that total order contribute, producing the
    union as it stood when that session ran. This is what makes a past session's
    screen show where the document stood then, instead of a number that changed
    afterwards for reasons that session had nothing to do with.
    """
    matching: list = []
    stale = 0
    for path, sidecar, order_key in sidecars:
        if sidecar.get("document_id") != document_id:
            continue
        if as_of is not None and order_key > as_of:
            continue
        if source_hash is not None:
            # Two independent layers, both non-fatal: the envelope's declared
            # source_hash, and the verdict set's own doc_id stamp. A sidecar that
            # disagrees with either was judged against a different claim map.
            if sidecar.get("source_hash") != source_hash:
                stale += 1
                continue
            declared = _map_identity_of(sidecar)
            if declared and declared != source_hash:
                stale += 1
                continue
        matching.append((path, sidecar, order_key))

    if source_hash is None:
        matching, dropped = _newest_map_group(matching)
        stale += dropped

    covered: set[str] = set()
    judged: set[str] = set()
    session_ids: list[str] = []
    claims_total = 0
    invalid = 0
    sessions = 0
    for path, sidecar, _key in matching:
        try:
            # ONE SET AT A TIME, deliberately. Handing the whole list to
            # union_coverage makes a single malformed verdict raise for the whole
            # document (review finding 3) — the read path promising to degrade
            # and instead becoming a 500 and a blank panel. Per-sidecar, a broken
            # file costs only itself. The merge arithmetic still belongs to the
            # judge module; only the failure boundary moved.
            merged = cj.union_coverage([sidecar])
        except cj.CoverageInputError as e:
            invalid += 1
            _warn(f"structurally invalid coverage sidecar skipped: {path} ({e})")
            continue
        covered.update(merged["covered_ids"])
        judged.update(merged["judged_ids"])
        sessions += 1
        sid = sidecar.get("session_id")
        if isinstance(sid, str):
            session_ids.append(sid)
        total = sidecar.get("claims_total")
        if isinstance(total, int) and total > claims_total:
            claims_total = total

    percentage = (
        round(100.0 * len(covered) / len(judged), cj.COVERAGE_PERCENTAGE_DECIMALS)
        if judged
        else 0.0
    )
    return {
        "covered_ids": sorted(covered),
        "percentage": percentage,
        "claims_total": claims_total,
        "sessions": sessions,
        "stale_sessions": stale,
        "invalid_sessions": invalid,
        "session_ids": session_ids,
    }


def union_for_document(
    user_id: str,
    document_id: str,
    source_hash: str | None = None,
    as_of: tuple[str, str] | None = None,
) -> dict:
    """Union coverage across every session ``user_id`` has run on ``document_id``.

    THE READ PATH. Collects the user's coverage sidecars for this document and
    merges them with the judge module's pure :func:`coverage_judge.union_coverage`
    — a claim is covered if ANY session covered it — then derives the percentage
    at read time. This is the number the live bar opens at on a returning
    session (the design's "starts at the accumulated number, not zero").

    THIS FUNCTION NEVER RAISES on any on-disk state. It backs a user-facing
    number, so every failure mode degrades to a smaller number plus a visible
    counter (design constraint 2 applied to the read path — see the 2026-08-04
    review's finding 3, which this replaced: one malformed sidecar used to raise
    for the whole document, i.e. a 500 and a blank panel).

    ``source_hash`` — when given, ONLY sidecars judged against that exact claim
    map contribute. Claim ids are per-document sequentials, so a re-extracted
    document's ``c15`` is not the old ``c15``; silently merging across maps would
    produce a false number. Mismatched sidecars are ignored (a quiet, correct
    under-count) and counted in ``stale_sessions`` so the condition is observable
    rather than invisible.

    ``source_hash=None`` — NEWEST MAP VERSION WINS. The sidecars are grouped by
    the claim map they were judged against and only the group holding the most
    recent ``judged_at`` contributes; the rest are counted in ``stale_sessions``.
    This deliberately does NOT merge across map versions (an earlier docstring
    promised that, and it is not a safe thing to promise: the merge is exactly
    the false-number case the ``source_hash`` filter exists to prevent). Callers
    that know the map they mean should pass its hash; ``None`` means "answer from
    whatever the current map appears to be", which is what the backfill script
    and any operator-facing read actually want.

    Returns::

        {"covered_ids": [...], "percentage": <float>, "claims_total": <int>,
         "sessions": <int>, "stale_sessions": <int>, "invalid_sessions": <int>,
         "session_ids": [...]}

    ``claims_total`` is the size of the claim map (from the sidecars; 0 when
    there are none) and ``percentage`` is over the JUDGED universe. Callers with
    the live claim map should prefer :func:`as_display`, which divides by the map
    itself. ``stale_sessions`` counts sidecars excluded as a different map
    version (correct behaviour); ``invalid_sessions`` counts this document's
    sidecars that were structurally broken (a defect — each one is also logged
    with its path). Keeping them apart matters: one is the system working, the
    other is the system losing data.

    ``as_of`` — an ordering key from :func:`_order_key` (``(session time,
    session id)``); only sessions at or before it contribute, giving the union
    AS IT STOOD when that session ran. The cutoff is SESSION time, not
    ``judged_at`` — see :func:`_session_time_of` for why that distinction is the
    whole point. Reads files but makes no model call.
    """
    return _union_from(_load_entries(user_id), document_id, source_hash, as_of)


def documents_view(user_id: str, identity_for) -> dict[str, dict]:
    """Accumulated coverage for EVERY document ``user_id`` has coverage on.

    The document picker's read path. ONE directory scan serves the whole list —
    the picker may show many documents, and a per-document round trip would make
    the list an N+1 (each one re-globbing and re-parsing the same sidecars).

    ``identity_for(document_id)`` supplies the document's live claim map identity
    as ``{"source_hash": str, "claims_total": int}``, or ``None`` when the
    document has no fresh map. It is a CALLBACK rather than an import so this
    module keeps its stdlib-only surface (claims.py reads documents; importing it
    here would drag the document layer into the coverage reader).

    A document is OMITTED from the result unless it has at least one contributing
    session AND a known claim total. Absent is not the same as 0% — a document
    that was never studied must render as having no coverage, not as a bar at
    zero — and a percentage with no denominator is not a number worth showing.
    Never raises: an ``identity_for`` that blows up on one document costs that
    document its entry, not the whole list.
    """
    sidecars = _load_entries(user_id)
    doc_ids = sorted(
        {
            s.get("document_id")
            for _p, s, _k in sidecars
            if isinstance(s.get("document_id"), str) and s.get("document_id")
        }
    )
    out: dict[str, dict] = {}
    for doc_id in doc_ids:
        try:
            identity = identity_for(doc_id)
        except Exception as e:  # noqa: BLE001 - one bad document never costs the list
            _warn(f"claim-map identity lookup failed for {doc_id}: {e!r}")
            continue
        if not identity:
            continue
        union = _union_from(sidecars, doc_id, identity.get("source_hash"))
        display = as_display(union, identity.get("claims_total"))
        if display is not None:
            out[doc_id] = display
    return out


def as_display(union: dict, claims_total) -> dict | None:
    """Turn a union into the shape a bar renders, or None if there is nothing to show.

    The percentage is over the CLAIM MAP's size, not the judged universe: the
    display promise is inventory ("16 of 63 claims covered"), and a denominator
    that shifts with what the judge happened to return would make the same
    progress read as different numbers. Rounded with the judge module's single
    convention so the two never drift.

    Returns None when no session contributed or the map size is unknown — the
    caller then omits the element entirely rather than drawing an empty bar.
    """
    if not union or union.get("sessions", 0) <= 0:
        return None
    total = claims_total if isinstance(claims_total, int) and claims_total > 0 else None
    if total is None:
        fallback = union.get("claims_total")
        total = fallback if isinstance(fallback, int) and fallback > 0 else None
    if total is None:
        return None
    covered = len(union.get("covered_ids", []))
    return {
        "covered": covered,
        "total": total,
        "percentage": round(100.0 * covered / total, cj.COVERAGE_PERCENTAGE_DECIMALS),
        "sessions": union.get("sessions", 0),
    }


def session_contribution(
    user_id: str, session_id: str, document_id: str, source_hash: str | None = None
) -> dict | None:
    """What THIS session added to the document's accumulated coverage, or None.

    ``new_claims`` counts the claims this session's sidecar covered that no
    EARLIER session had already covered. "Earlier" is by SESSION time, not by
    when the judge ran (see :func:`_session_time_of`) — ordering on ``judged_at``
    made a backfilled session count every live-judged session as its
    predecessor, zeroing a contribution that was real. Measuring against earlier
    sessions rather than "every other session" is what makes the number stable:
    re-opening a past session months later must not shrink its contribution
    because a later session happened to re-cover the same ground.

    Returns None — never an error — when the sidecar is missing, unreadable,
    structurally invalid, or belongs to a different document. A missing judge
    result means the accumulated total still renders and the delta is simply
    absent, per the brief.
    """
    own = load_sidecar(user_id, session_id)
    if not own or own.get("document_id") != document_id:
        return None
    if source_hash is not None and own.get("source_hash") != source_hash:
        return None
    try:
        mine = cj.union_coverage([own])
    except cj.CoverageInputError as e:
        _warn(f"structurally invalid coverage sidecar for session {session_id}: {e}")
        return None

    own_key = _order_key(user_id, own)
    prior: set[str] = set()
    for path, sidecar, order_key in _load_entries(user_id):
        if sidecar.get("document_id") != document_id:
            continue
        if source_hash is not None and sidecar.get("source_hash") != source_hash:
            continue
        # STRICTLY earlier in the same total order the snapshot uses, so
        # at_session minus prior is exactly this session's own contribution.
        # The session id inside the key makes this exclude the session itself.
        if order_key >= own_key:
            continue
        try:
            prior.update(cj.union_coverage([sidecar])["covered_ids"])
        except cj.CoverageInputError as e:
            _warn(f"structurally invalid coverage sidecar skipped: {path} ({e})")
            continue

    covered_ids = set(mine["covered_ids"])
    return {"covered": len(covered_ids), "new_claims": len(covered_ids - prior)}


# The four states the ended view distinguishes. ``pending`` is the only one
# that means "keep waiting"; everything else is settled.
STATUS_READY = "ready"
STATUS_PENDING = "pending"
STATUS_FAILED = "failed"
STATUS_NONE = "none"


def resolve_status(*, has_sidecar: bool, expects: bool, judge_failed: bool) -> str:
    """Which of the four coverage states this session is in, as a pure decision.

    Lives HERE rather than inline in the ``/telemetry`` route because app.py is
    deliberately untested at the transport layer (CLAUDE.md, "test via pure
    helpers, not TestClient") — a decision left there is a decision no test can
    fail on. The route stays glue over this plus two already-tested primitives.

    Order matters. ``has_sidecar`` wins outright: the file is written AFTER the
    ledger row, so its presence is the only proof the result actually landed,
    and a stale "failed" row must never hide a sidecar that exists. ``expects``
    is checked before ``judge_failed`` so a session that was never going to be
    judged reads as ``none`` rather than borrowing a failure it never had.
    """
    if has_sidecar:
        return STATUS_READY
    if not expects:
        return STATUS_NONE
    return STATUS_FAILED if judge_failed else STATUS_PENDING


def finalize_for_status(block: dict | None, status: str) -> dict | None:
    """Apply the pending-window policy to a :func:`session_view` block.

    While the judge is still running the accumulated ``total`` is WITHHELD. It
    is a real number, but it does not include the session the user just
    finished, and on the screen you land on when you hang up — under a bare
    "Coverage" heading, with no delta beside it — it reads as a statement about
    that session. For those 10-40s the honest answer is nothing at all.

    Every other status passes through untouched, which is what releases the
    total again on ``failed``: a broken judge then costs the delta rather than
    the document's number. A block left with nothing in it collapses to None so
    the caller emits a null field and the UI stays silent.
    """
    if status != STATUS_PENDING or block is None:
        return block
    held = {**block, "total": None}
    if not any(held.get(k) for k in ("total", "at_session", "session")):
        return None
    return held


def session_view(
    user_id: str,
    session_id: str,
    document_id: str | None,
    source_hash: str | None = None,
    claims_total=None,
) -> dict | None:
    """The ended view's coverage block: the document total, this session's
    snapshot, and this session's delta.

    Returns ``{"total": {...} | None, "at_session": {...} | None,
    "session": {...} | None}``, or None when there is nothing to show at all
    (no document, or no coverage anywhere yet) so the caller can emit a null
    field and the UI can stay silent.

    THREE INDEPENDENT HALVES, because two different screens read this:

      * ``total`` — where the document stands NOW. Correct on the screen you
        land on when you hang up: that is the live number at that moment.
      * ``at_session`` — where it stood when THIS session RAN (the union of
        every session at or before it in session-start order). This is what a
        past session's screen needs: the current total is identical on every
        past session, so it says nothing about the one being viewed and reads as
        a claim about it that isn't true. Scrolling back through history should
        show an ascending record, not the same number N times — and because the
        cutoff is session time, that record ascends in the SAME order the
        history list is sorted in.
      * ``session`` — what this session added.

    A session whose judge produced nothing has ``at_session`` and ``session``
    None while ``total`` still renders, so a failed judge never costs the
    document's number.
    """
    if not document_id:
        return None
    union = union_for_document(user_id, document_id, source_hash)
    total = as_display(union, claims_total)
    contribution = session_contribution(
        user_id, session_id, document_id, source_hash=source_hash
    )

    # The snapshot is keyed on WHEN THIS SESSION RAN, so it needs the sidecar;
    # without one there is no moment to take a snapshot at. An unresolvable
    # session time is the same case: a cutoff of "" would place this session
    # before everything and render a near-empty snapshot as if it were history,
    # which is the wrong-number failure this whole ordering fix exists to stop.
    at_session = None
    own = load_sidecar(user_id, session_id)
    if own and own.get("document_id") == document_id:
        as_of = _order_key(user_id, own)
        if as_of[0]:
            at_session = as_display(
                union_for_document(user_id, document_id, source_hash, as_of),
                claims_total,
            )

    if total is None and contribution is None and at_session is None:
        return None
    return {"total": total, "at_session": at_session, "session": contribution}


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

    # Taken from the transcript both callers already hand over, so neither
    # bot.py's teardown nor backfill_coverage.py has to learn a new argument —
    # and backfill therefore stamps the session's REAL start time rather than
    # the time of the backfill, which is the whole ordering fix.
    session_start = transcript.get("session_start") if isinstance(transcript, dict) else None
    sidecar = build_sidecar(
        session_id=session_id,
        user_id=user_id,
        document_id=document_id,
        source_hash=source_hash,
        verdict_obj=verdict_obj,
        claims_total=claims_total,
        transcript_turns=len(turns),
        session_start=session_start if isinstance(session_start, str) else None,
    )
    cost_row["covered_count"] = sidecar["covered_count"]
    cost_row["claims_total"] = claims_total
    return sidecar, cost_row
