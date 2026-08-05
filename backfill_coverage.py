"""Backfill coverage sidecars for study sessions that predate the live judge.

Coverage judging was wired into session teardown on 2026-08-04. Every study
session BEFORE that has a transcript on disk but no `.coverage.json` sidecar, so
the cross-session union — the number the live bar opens at — reads near zero and
tells a returning user they have barely started a document they have worked
through for hours. This script re-judges those past transcripts offline and
writes their sidecars, so the accumulated number reflects real history.

It is the same judge, the same prompt, and the same sidecar writer the live
teardown path uses (``coverage_store.judge_session`` / ``write_sidecar``) — NOT a
parallel implementation. A backfilled sidecar is therefore indistinguishable from
one written live, and re-judging a session later (a new judge prompt, say)
overwrites it the same way.

Usage — DRY RUN FIRST (makes no model calls, writes nothing):

    .venv/bin/python backfill_coverage.py

    .venv/bin/python backfill_coverage.py --execute
    .venv/bin/python backfill_coverage.py --execute --user matt --doc 2aa66acc
    .venv/bin/python backfill_coverage.py --execute --limit 3

Safety properties:
  * DRY RUN BY DEFAULT. Nothing is spent or written without ``--execute``.
  * IDEMPOTENT. A session that already has a sidecar is skipped unless
    ``--force``, so an interrupted run resumes without re-billing.
  * A session whose claim map is missing or STALE is skipped, never judged
    against the wrong map — claim ids are per-document sequentials, so judging
    against a re-extracted map would write a sidecar whose ids mean something
    else. This mirrors the live path's freshness check.
  * One failure does not stop the run; failures are collected and reported.
  * Cost is estimated up front and reported per session as it goes.

The API key is read from the environment or the app ``.env`` (absolute path) and
is never printed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import claims
import coverage_store as cs
from cost_audit import (
    PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK,
    PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK,
)

SESSION_LOG_JSONL_PATH = (
    Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "session-log.jsonl"
)
TRANSCRIPTS_DIR = Path.home() / ".voice-tutor" / "transcripts"
DOCUMENTS_DIR = Path.home() / ".voice-tutor" / "documents"

# Observed on real 63-claim sessions (2026-08-04): ~$0.03 per judged session.
APPROX_USD_PER_SESSION = 0.03


def load_api_key() -> str:
    """Anthropic key from the environment, else the app .env. Never printed."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and key.strip():
        return key.strip()
    env_path = Path(os.path.expanduser("~/development/voice-tutor/.env"))
    if not env_path.exists():
        raise SystemExit(f"no ANTHROPIC_API_KEY in env and no .env at {env_path}")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        name, sep, value = line.partition("=")
        if sep and name.strip() == "ANTHROPIC_API_KEY":
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"no ANTHROPIC_API_KEY line in {env_path}")


def study_sessions():
    """Yield (user_id, document_id, session_id) for every study session logged.

    Reads the append-only ledger, de-duplicated and ordered oldest-first so a
    partial run makes progress in a predictable order.
    """
    if not SESSION_LOG_JSONL_PATH.exists():
        raise SystemExit(f"no session log at {SESSION_LOG_JSONL_PATH}")
    seen = set()
    out = []
    for line in SESSION_LOG_JSONL_PATH.open():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "session" or entry.get("mode") != "study":
            continue
        uid, did, sid = entry.get("user_id"), entry.get("document_id"), entry.get("session_id")
        if not (uid and did and sid) or (uid, did, sid) in seen:
            continue
        seen.add((uid, did, sid))
        out.append((entry.get("session_start") or "", uid, did, sid))
    return [(u, d, s) for _, u, d, s in sorted(out)]


def resolve_document(user_id: str, doc_id: str):
    """Return (doc_text, source_hash, claim_objs) or (None, reason, None).

    Uses claims.load_fresh_claims — the SAME cache-and-freshness check the live
    session-start path uses — so a stale or missing claim map is a skip, never a
    judge against a map whose ids no longer mean what the sidecar would say.
    """
    for namespace in (Path(user_id).name, claims.SHARED_USER_ID):
        txt = DOCUMENTS_DIR / namespace / f"{Path(doc_id).name}.txt"
        if txt.exists():
            text = txt.read_text()
            claim_objs = claims.load_fresh_claims(user_id, doc_id, text)
            if claim_objs is None:
                return None, "claim map missing or stale", None
            return text, claims.source_hash_of(text), claim_objs
    return None, "document text not found on disk", None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="backfill_coverage",
        description="Re-judge past study sessions and write their coverage sidecars.",
    )
    parser.add_argument("--execute", action="store_true",
                        help="actually judge + write (default is a dry run)")
    parser.add_argument("--user", default=None, help="only this user_id")
    parser.add_argument("--doc", default=None, help="only doc ids starting with this prefix")
    parser.add_argument("--limit", type=int, default=None, help="stop after N sessions")
    parser.add_argument("--force", action="store_true",
                        help="re-judge sessions that already have a sidecar")
    args = parser.parse_args(argv)

    candidates = []
    skipped = []
    for user_id, doc_id, session_id in study_sessions():
        if args.user and user_id != args.user:
            continue
        if args.doc and not doc_id.startswith(args.doc):
            continue
        transcript_path = TRANSCRIPTS_DIR / Path(user_id).name / f"{Path(session_id).name}.json"
        if not transcript_path.exists():
            skipped.append((user_id, doc_id, session_id, "no transcript on disk"))
            continue
        if not args.force and cs.coverage_path(user_id, session_id).exists():
            skipped.append((user_id, doc_id, session_id, "already has a sidecar"))
            continue
        # Resolve the claim map HERE, during candidate collection, so the dry run
        # reports the same skips the real run will take. Checking it only at
        # execute time would let the dry run advertise a cost for sessions that
        # cannot be judged at all — an estimate several times the true spend.
        doc_text, source_hash, claim_objs = resolve_document(user_id, doc_id)
        if doc_text is None:
            skipped.append((user_id, doc_id, session_id, source_hash))
            continue
        candidates.append((user_id, doc_id, session_id, transcript_path))

    if args.limit is not None:
        candidates = candidates[: args.limit]

    print(f"{len(candidates)} session(s) to judge, {len(skipped)} skipped")
    for user_id, doc_id, session_id, reason in skipped:
        print(f"  SKIP {user_id}/{session_id[:8]} doc {doc_id[:8]} — {reason}")
    if not candidates:
        return 0
    print(f"\nestimated cost: ~${len(candidates) * APPROX_USD_PER_SESSION:.2f}")

    if not args.execute:
        print("\nDRY RUN — no model calls made, nothing written.")
        for user_id, doc_id, session_id, _ in candidates:
            print(f"  would judge {user_id}/{session_id[:8]} doc {doc_id[:8]}")
        print("\nRe-run with --execute to perform the backfill.")
        return 0

    import anthropic
    client = anthropic.Anthropic(api_key=load_api_key())

    judged = failed = 0
    spent = 0.0
    for user_id, doc_id, session_id, transcript_path in candidates:
        doc_text, source_hash, claim_objs = resolve_document(user_id, doc_id)
        if doc_text is None:
            print(f"  SKIP {user_id}/{session_id[:8]} — {source_hash}")
            continue
        try:
            transcript = json.loads(transcript_path.read_text())
        except (OSError, ValueError) as e:
            print(f"  SKIP {user_id}/{session_id[:8]} — unreadable transcript: {e}")
            continue

        sidecar, cost_row = cs.judge_session(
            user_id=user_id,
            session_id=session_id,
            document_id=doc_id,
            source_hash=source_hash,
            claim_objs=claim_objs,
            transcript=transcript,
            client=client,
        )
        usd = (
            cost_row.get("input_tokens", 0) / 1_000_000 * PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK
            + cost_row.get("output_tokens", 0) / 1_000_000 * PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK
        )
        spent += usd
        if sidecar is None:
            failed += 1
            print(f"  FAIL {user_id}/{session_id[:8]} — {cost_row.get('error')}")
            continue
        cs.write_sidecar(user_id, session_id, sidecar)
        judged += 1
        print(f"  ok   {user_id}/{session_id[:8]} doc {doc_id[:8]} — "
              f"{sidecar['covered_count']}/{sidecar['claims_total']} covered  ${usd:.3f}")

    print(f"\njudged {judged}, failed {failed}, spent ~${spent:.2f}")

    # Report the resulting union per (user, document) — the number the live bar
    # will open at, which is the whole point of the backfill.
    print("\nresulting union per document:")
    pairs = sorted({(u, d) for u, d, _, _ in candidates})
    for user_id, doc_id in pairs:
        doc_text, source_hash, _ = resolve_document(user_id, doc_id)
        if doc_text is None:
            continue
        union = cs.union_for_document(user_id, doc_id, source_hash)
        print(f"  {user_id}/{doc_id[:8]}: {union['percentage']}% "
              f"({len(union['covered_ids'])}/{union['claims_total']} claims, "
              f"{union['sessions']} session(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
