#!/usr/bin/env python3
"""Provider cost reconciliation for Voice Tutor — a standalone diagnostic.

This script is NOT part of the app and is NOT imported by ``app.py`` / ``bot.py``.
It answers a question the internal-consistency audit (``cost_audit.py``)
deliberately cannot: does our local ledger (``cost-log.jsonl``) match what the
providers *actually billed*? ``cost_audit.py`` proves the logger's arithmetic is
honest given each row's own numbers; this proves the numbers themselves are real
by diffing them against Anthropic / Deepgram / Cartesia usage APIs.

Design constraints (see products/voice-tutor/ideas.md "provider cost
reconciliation"):
  * Read-only everywhere — never writes the ledger, never mutates a provider.
  * Network-dependent, credential-dependent — a MANUAL diagnostic, not hermetic,
    not a dev-harness run, not in the app's test suite. Only the pure
    ledger-summing core below has unit tests.
  * Stdlib only (json / urllib / datetime / zoneinfo / argparse) so it stays a
    self-contained script with no pipecat/ML import surface. It imports the
    pricing constants from ``cost_audit.py`` (itself stdlib-only) so the price
    list has a single source of truth.
  * Never prints or logs any API key.

Credentials:
  * ``~/.voice-tutor-secrets.env`` — ANTHROPIC_ADMIN_KEY, CARTESIA_ADMIN_KEY.
  * The app's ``.env`` — DEEPGRAM_API_KEY (the project creds the app already
    uses) and ANTHROPIC_API_KEY (used only to auto-discover which org API-key id
    is Voice Tutor's, by matching the key's partial hint — the raw value is never
    sent anywhere or printed).

Timezone: ledger timestamps are naive local (America/Los_Angeles); provider APIs
report UTC. We convert the ledger's local range to UTC before querying, and by
default reconcile the WHOLE ledger range (min start → max end) so day-boundary
misalignment is a non-issue for the headline totals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Single source of truth for prices + the ledger path: reuse cost_audit.py's
# constants (it is stdlib-only, so importing it introduces no pipecat surface).
import cost_audit

# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

# Where credentials live. The secrets file holds the two admin keys; the app's
# .env holds the Deepgram project key + the app's regular Anthropic key.
SECRETS_ENV_PATH = Path.home() / ".voice-tutor-secrets.env"
APP_ENV_PATH = Path(__file__).resolve().parent / ".env"

# Default reconciliation tolerance (percent). A |%diff| within this band reads as
# a match; beyond it, a discrepancy worth investigating. Overridable at runtime
# via --tolerance-pct or the RECONCILE_TOLERANCE_PCT env var (flag wins) so the
# threshold can be tuned without editing code.
DEFAULT_TOLERANCE_PCT = 1.0

# Provider API bases + pinned versions.
ANTHROPIC_API_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEEPGRAM_API_BASE = "https://api.deepgram.com"
CARTESIA_API_BASE = "https://api.cartesia.ai"
CARTESIA_VERSION = "2026-03-01"  # only value the credit-usage route accepts

# The two model families the ledger distinguishes. Provider usage is grouped by
# model; we bucket a provider model into "haiku" (post-session summary/analysis +
# study-artifact generation) vs "live" (the Sonnet conversation model) by
# substring, so exact dated model ids don't need hard-coding here.
HAIKU_MODEL_SUBSTR = "haiku"


def classify_model(model: str | None) -> str:
    """Return 'haiku' or 'live' for a provider model id string."""
    if model and HAIKU_MODEL_SUBSTR in model.lower():
        return "haiku"
    return "live"


# ===========================================================================
# PURE LEDGER-SUMMING CORE  (this section is what the unit tests cover)
#
# No network, no credentials, no clock. Everything here is a pure function of
# its inputs so it can be tested hermetically.
# ===========================================================================


@dataclass
class TokenBucket:
    """Anthropic token counts for one model family, by billing class."""

    uncached_input: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    output: float = 0.0

    def add(self, uncached_input=0.0, cache_read=0.0, cache_write=0.0, output=0.0):
        self.uncached_input += _num(uncached_input)
        self.cache_read += _num(cache_read)
        self.cache_write += _num(cache_write)
        self.output += _num(output)

    def total(self) -> float:
        return self.uncached_input + self.cache_read + self.cache_write + self.output


@dataclass
class LedgerTotals:
    """Everything the ledger contributes to reconciliation, per provider.

    Anthropic is split into two model-family token buckets (live=Sonnet,
    haiku=auxiliary) because they price differently. Deepgram is billed STT
    minutes; Cartesia is TTS characters (== credits at 1 credit/char). We also
    carry the ledger's *recorded* dollar figures per provider so the report can
    show what we thought we spent alongside the recomputed / provider figures.
    """

    live_tokens: TokenBucket = field(default_factory=TokenBucket)
    haiku_tokens: TokenBucket = field(default_factory=TokenBucket)
    stt_minutes: float = 0.0
    tts_chars: float = 0.0

    # Recorded $ straight from the ledger's stored cost_*_usd fields.
    recorded_anthropic_usd: float = 0.0
    recorded_deepgram_usd: float = 0.0
    recorded_cartesia_usd: float = 0.0

    # Bookkeeping.
    session_rows: int = 0
    artifact_rows: int = 0

    @property
    def tts_credits(self) -> float:
        """Cartesia bills 1 credit per character submitted to TTS."""
        return self.tts_chars


def _num(value) -> float:
    """Coerce a possibly-missing/None numeric field to float (None -> 0.0)."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_ledger_rows(path: Path) -> list[dict]:
    """Parse the JSONL ledger into a list of dict rows.

    Blank lines and any non-object / malformed line are skipped (the audit tool
    owns malformed-row *reporting*; here we simply sum what is well-formed).
    """
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for raw in f:
            if raw.strip() == "":
                continue
            try:
                entry = json.loads(raw)
            except Exception:
                continue
            if isinstance(entry, dict):
                rows.append(entry)
    return rows


def parse_local_ts(value) -> datetime | None:
    """Parse a naive-local ISO timestamp string into a naive datetime.

    Ledger timestamps look like '2026-04-26T16:17:47.912037' (no tz) and denote
    America/Los_Angeles wall-clock time. Returns None if unparseable/missing.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    # Ledger times are naive; strip any tz that sneaks in so comparisons stay
    # naive-local throughout the filtering step.
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _row_kind(row: dict) -> str:
    """Return 'session', 'artifact', or 'legacy' for a ledger row.

    Legacy rows (no 'kind' field) predate the schema and are treated as session
    rows for Sonnet-token / STT / TTS summing (they carry llm_*/stt/tts fields
    but no post_session/artifact fields)."""
    kind = row.get("kind")
    if kind == "artifact":
        return "artifact"
    if kind == "session":
        return "session"
    if "kind" not in row:
        return "legacy"
    return kind  # unknown kind: caller ignores for summing


def filter_rows_by_local_range(
    rows: list[dict], start_local: datetime | None, end_local: datetime | None
) -> list[dict]:
    """Keep rows whose (local) time falls in [start_local, end_local].

    Session/legacy rows are timestamped by ``session_start``. Artifact rows have
    NO timestamp of their own, so they are joined to their session's start time
    via ``session_id`` (an artifact is in-range iff its session is). Rows with no
    resolvable time are kept only when no bounds are given (full-history default).
    A None bound means "unbounded on that side".
    """
    if start_local is None and end_local is None:
        return list(rows)

    # session_id -> session_start (local), for the artifact join.
    session_time: dict = {}
    for row in rows:
        if _row_kind(row) in ("session", "legacy"):
            ts = parse_local_ts(row.get("session_start"))
            sid = row.get("session_id")
            if ts is not None and sid is not None:
                session_time[sid] = ts

    def in_range(ts: datetime | None) -> bool:
        if ts is None:
            return False
        if start_local is not None and ts < start_local:
            return False
        if end_local is not None and ts > end_local:
            return False
        return True

    kept: list[dict] = []
    for row in rows:
        kind = _row_kind(row)
        if kind == "artifact":
            ts = session_time.get(row.get("session_id"))
        else:
            ts = parse_local_ts(row.get("session_start"))
        if in_range(ts):
            kept.append(row)
    return kept


def summarize_ledger(
    rows: list[dict],
    start_local: datetime | None = None,
    end_local: datetime | None = None,
) -> LedgerTotals:
    """Sum a ledger (optionally date-filtered) into per-provider totals.

    Pure and deterministic. This is the function the unit tests exercise.

    Anthropic token attribution:
      * session/legacy rows contribute their four llm_* token classes to the
        LIVE (Sonnet) bucket;
      * session rows also contribute post_session_input/output tokens to the
        HAIKU bucket (as uncached input + output; those calls don't use caching);
      * artifact rows contribute input_tokens/output_tokens to the HAIKU bucket.
    Deepgram: sum ``stt_minutes_billed``.
    Cartesia: sum ``tts_chars`` (modern rows) or ``tts_credits`` (legacy rows,
    which recorded credits directly — 1 credit/char, so numerically the chars).
    """
    totals = LedgerTotals()
    scoped = filter_rows_by_local_range(rows, start_local, end_local)

    for row in scoped:
        kind = _row_kind(row)

        if kind in ("session", "legacy"):
            totals.session_rows += 1
            # Live (Sonnet) conversation tokens.
            totals.live_tokens.add(
                uncached_input=row.get("llm_uncached_input_tokens"),
                cache_read=row.get("llm_cache_read_tokens"),
                cache_write=row.get("llm_cache_write_tokens"),
                output=row.get("llm_output_tokens"),
            )
            # Post-session Haiku summary/analysis (absent on legacy rows -> 0).
            totals.haiku_tokens.add(
                uncached_input=row.get("post_session_input_tokens"),
                output=row.get("post_session_output_tokens"),
            )
            # Deepgram STT.
            totals.stt_minutes += _num(row.get("stt_minutes_billed"))
            # Cartesia TTS: modern tts_chars, else legacy tts_credits.
            if row.get("tts_chars") is not None:
                totals.tts_chars += _num(row.get("tts_chars"))
            else:
                totals.tts_chars += _num(row.get("tts_credits"))
            # Recorded $ (per provider) straight from stored fields.
            totals.recorded_anthropic_usd += _num(row.get("cost_llm_usd")) + _num(
                row.get("cost_post_session_usd")
            )
            totals.recorded_deepgram_usd += _num(row.get("cost_stt_usd"))
            totals.recorded_cartesia_usd += _num(row.get("cost_tts_usd"))

        elif kind == "artifact":
            totals.artifact_rows += 1
            # Study-artifact generation runs on Haiku.
            totals.haiku_tokens.add(
                uncached_input=row.get("input_tokens"),
                output=row.get("output_tokens"),
            )
            totals.recorded_anthropic_usd += _num(row.get("cost_usd"))

        # Unknown kinds contribute nothing.

    return totals


# --- Pricing a token bucket with cost_audit's constants (pure) -------------

_MTOK = 1_000_000


def price_token_bucket(bucket: TokenBucket, family: str) -> float:
    """Dollar-price a TokenBucket using the shared cost_audit price constants.

    ``family`` is 'live' (Sonnet rates) or 'haiku' (Haiku rates). Used to derive
    a $ figure from PROVIDER-reported token counts with the SAME price list the
    ledger used, so a token-count drift surfaces as a dollar drift too.
    """
    if family == "haiku":
        p_in = cost_audit.PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK
        p_out = cost_audit.PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK
        # Haiku calls here don't use prompt caching; fall back to input rate if
        # any cache tokens ever appear.
        p_cr = p_in
        p_cw = p_in
    else:
        p_in = cost_audit.PRICE_ANTHROPIC_INPUT_PER_MTOK
        p_out = cost_audit.PRICE_ANTHROPIC_OUTPUT_PER_MTOK
        p_cr = cost_audit.PRICE_ANTHROPIC_CACHE_READ_PER_MTOK
        p_cw = cost_audit.PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK
    return (
        bucket.uncached_input / _MTOK * p_in
        + bucket.cache_read / _MTOK * p_cr
        + bucket.cache_write / _MTOK * p_cw
        + bucket.output / _MTOK * p_out
    )


# --- Reconciliation math (pure) --------------------------------------------


@dataclass
class ReconLine:
    """One reconciled quantity: ledger vs provider in a native unit."""

    label: str
    unit: str
    ledger: float
    provider: float

    @property
    def abs_diff(self) -> float:
        return self.provider - self.ledger

    @property
    def pct_diff(self) -> float | None:
        """Provider-vs-ledger percent difference; None if ledger is 0."""
        if self.ledger == 0:
            return None
        return (self.provider - self.ledger) / self.ledger * 100.0

    def within(self, tolerance_pct: float) -> bool:
        pct = self.pct_diff
        if pct is None:
            # No ledger baseline: a match only if the provider side is also 0.
            return self.provider == 0
        return abs(pct) <= tolerance_pct


def verdict_for(lines: list[ReconLine], tolerance_pct: float) -> str:
    """'MATCH' iff every line is within tolerance, else 'DISCREPANCY'."""
    return "MATCH" if all(line.within(tolerance_pct) for line in lines) else "DISCREPANCY"


# ===========================================================================
# CREDENTIALS  (never printed / logged)
# ===========================================================================


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE .env file into a dict. Missing file -> {}.

    Ignores blank lines and ``#`` comments; strips optional surrounding quotes.
    Values are NEVER printed anywhere by this tool.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        val = val.strip().strip('"').strip("'")
        out[key] = val
    return out


def load_credentials() -> dict[str, str]:
    """Merge the app .env and the secrets .env, plus the live environment.

    Precedence: process environment overrides files (lets a one-off run inject a
    key without touching files). Returns a plain dict — callers pull the specific
    names they need and must never print them.
    """
    merged: dict[str, str] = {}
    merged.update(parse_env_file(APP_ENV_PATH))
    merged.update(parse_env_file(SECRETS_ENV_PATH))
    for name in (
        "ANTHROPIC_ADMIN_KEY",
        "CARTESIA_ADMIN_KEY",
        "DEEPGRAM_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_ID",
        "DEEPGRAM_PROJECT_ID",
    ):
        if os.environ.get(name):
            merged[name] = os.environ[name]
    return merged


# ===========================================================================
# HTTP  (thin urllib wrapper — stdlib only)
# ===========================================================================


class ProviderError(RuntimeError):
    """A provider call failed in a way we surface (non-fatal per provider)."""


def http_get_json(url: str, headers: dict[str, str], timeout: float = 60.0) -> dict:
    """GET ``url`` with ``headers`` and parse the JSON body.

    Raises ProviderError with the status + a short body snippet on HTTP error.
    The snippet never contains our credentials (provider error bodies echo the
    request params, not auth headers), and headers are never logged.
    """
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise ProviderError(f"HTTP {e.code} for {url.split('?')[0]}: {body}") from None
    except urllib.error.URLError as e:
        raise ProviderError(f"network error for {url.split('?')[0]}: {e.reason}") from None


def _encode_query(params: list[tuple[str, str]]) -> str:
    """URL-encode ordered (key, value) pairs (supports repeated keys)."""
    return urllib.parse.urlencode(params)


# ===========================================================================
# ANTHROPIC  (Admin Usage & Cost API)
# ===========================================================================


def _anthropic_headers(admin_key: str) -> dict[str, str]:
    return {
        "x-api-key": admin_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }


def discover_anthropic_api_key_id(admin_key: str, app_key: str | None) -> str | None:
    """Find the org API-key id whose partial hint matches the app's key.

    The Admin API's ``GET /v1/organizations/api_keys`` lists org keys with a
    ``partial_key_hint`` (e.g. 'sk-ant-api03-...AbCd'). We match it against the
    app's real ANTHROPIC_API_KEY by comparing the last few characters — the raw
    key is never sent to the network or printed; only the hint comparison runs
    locally. Returns the api_key_id, or None if it can't be uniquely resolved.
    """
    if not app_key:
        return None
    tail = app_key[-4:]
    matches: list[str] = []
    page: str | None = None
    while True:
        params = [("limit", "100")]
        if page:
            params.append(("page", page))
        url = f"{ANTHROPIC_API_BASE}/v1/organizations/api_keys?{_encode_query(params)}"
        data = http_get_json(url, _anthropic_headers(admin_key))
        for item in data.get("data", []):
            hint = item.get("partial_key_hint") or ""
            if hint.endswith(tail):
                matches.append(item.get("id"))
        if data.get("has_more") and data.get("next_page"):
            page = data["next_page"]
        else:
            break
    if len(matches) == 1:
        return matches[0]
    return None


def fetch_anthropic_usage(
    admin_key: str, api_key_id: str, start_utc: datetime, end_utc: datetime
) -> dict[str, TokenBucket]:
    """Pull provider token usage for one API key, bucketed by model family.

    Uses ``GET /v1/organizations/usage_report/messages`` with an ``api_key_ids``
    filter (so we see ONLY the Voice Tutor key, not the whole org) grouped by
    model, daily buckets, paginating on next_page. Returns {'live': TokenBucket,
    'haiku': TokenBucket}.
    """
    result = {"live": TokenBucket(), "haiku": TokenBucket()}
    page: str | None = None
    while True:
        params = [
            ("starting_at", _rfc3339(start_utc)),
            ("ending_at", _rfc3339(end_utc)),
            ("bucket_width", "1d"),
            ("limit", "31"),
            ("api_key_ids[]", api_key_id),
            ("group_by[]", "model"),
        ]
        if page:
            params.append(("page", page))
        url = f"{ANTHROPIC_API_BASE}/v1/organizations/usage_report/messages?{_encode_query(params)}"
        data = http_get_json(url, _anthropic_headers(admin_key))
        for bucket in data.get("data", []):
            for item in bucket.get("results", []):
                family = classify_model(item.get("model"))
                cache_creation = item.get("cache_creation") or {}
                cache_write = _num(cache_creation.get("ephemeral_1h_input_tokens")) + _num(
                    cache_creation.get("ephemeral_5m_input_tokens")
                )
                result[family].add(
                    uncached_input=item.get("uncached_input_tokens"),
                    cache_read=item.get("cache_read_input_tokens"),
                    cache_write=cache_write,
                    output=item.get("output_tokens"),
                )
        if data.get("has_more") and data.get("next_page"):
            page = data["next_page"]
        else:
            break
    return result


# ===========================================================================
# DEEPGRAM  (per-project usage API)
# ===========================================================================


def _deepgram_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Token {key}"}


def discover_deepgram_project_id(key: str) -> str | None:
    """Return the project id if the key has exactly one project, else None."""
    url = f"{DEEPGRAM_API_BASE}/v1/projects"
    data = http_get_json(url, _deepgram_headers(key))
    projects = data.get("projects", [])
    if len(projects) == 1:
        return projects[0].get("project_id")
    return None


def fetch_deepgram_usage(
    key: str, project_id: str, start_date: str, end_date: str
) -> dict[str, float]:
    """Sum billed STT audio hours for a project over a date range.

    Uses ``GET /v1/projects/{id}/usage/breakdown?start=&end=`` (dates are
    YYYY-MM-DD, UTC). Sums ``hours`` over ``listen`` (STT) results — Voice Tutor
    only uses Deepgram for STT — and also returns ``total_hours`` and requests as
    context. Returns minutes = hours*60.
    """
    params = [("start", start_date), ("end", end_date)]
    url = f"{DEEPGRAM_API_BASE}/v1/projects/{project_id}/usage/breakdown?{_encode_query(params)}"
    data = http_get_json(url, _deepgram_headers(key))
    hours = 0.0
    total_hours = 0.0
    requests = 0.0
    for res in data.get("results", []):
        endpoint = (res.get("grouping") or {}).get("endpoint")
        # Include listen (STT) rows; also include rows with no endpoint label.
        if endpoint in (None, "listen"):
            hours += _num(res.get("hours"))
        total_hours += _num(res.get("total_hours"))
        requests += _num(res.get("requests"))
    return {
        "stt_minutes": hours * 60.0,
        "total_hours": total_hours,
        "requests": requests,
    }


# ===========================================================================
# CARTESIA  (credit-usage route, admin key)
# ===========================================================================


def _cartesia_headers(admin_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {admin_key}",
        "Cartesia-Version": CARTESIA_VERSION,
    }


def fetch_cartesia_usage(
    admin_key: str, start_utc: datetime, end_utc: datetime
) -> dict[str, float]:
    """Sum credits consumed over a range via ``GET /usage/credits``.

    Passes start_ts/end_ts (RFC3339) with no group_by, so ``data`` is flat
    buckets each carrying ``credits``; we sum them. Voice Tutor's Cartesia
    account is dedicated to this app, so the whole account's credit usage IS
    Voice Tutor (no per-key filter needed).
    """
    params = [
        ("start_ts", _rfc3339(start_utc)),
        ("end_ts", _rfc3339(end_utc)),
    ]
    url = f"{CARTESIA_API_BASE}/usage/credits?{_encode_query(params)}"
    data = http_get_json(url, _cartesia_headers(admin_key))
    credits = 0.0
    for bucket in data.get("data", []):
        credits += _num(bucket.get("credits"))
    return {"credits": credits}


# ===========================================================================
# TIME
# ===========================================================================


def _rfc3339(dt: datetime) -> str:
    """Format a tz-aware UTC datetime as an RFC3339 'Z' string."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_to_utc(dt_local_naive: datetime) -> datetime:
    """Interpret a naive datetime as America/Los_Angeles, return tz-aware UTC."""
    return dt_local_naive.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)


def ledger_local_bounds(rows: list[dict]) -> tuple[datetime | None, datetime | None]:
    """Return (min session_start, max session_end) across the ledger, local naive."""
    starts: list[datetime] = []
    ends: list[datetime] = []
    for row in rows:
        s = parse_local_ts(row.get("session_start"))
        e = parse_local_ts(row.get("session_end"))
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)
    return (min(starts) if starts else None, max(ends) if ends else None)


def resolve_range(
    rows: list[dict], start_arg: str | None, end_arg: str | None
) -> tuple[datetime | None, datetime | None]:
    """Resolve the --start/--end args (YYYY-MM-DD, local) into local datetimes.

    A missing --start defaults to the ledger's earliest session; a missing --end
    to its latest. --start is the local midnight of that day (inclusive); --end
    is the END of that local day (23:59:59.999999, inclusive).
    """
    min_start, max_end = ledger_local_bounds(rows)
    if start_arg:
        start_local = datetime.strptime(start_arg, "%Y-%m-%d")
    else:
        start_local = min_start
    if end_arg:
        end_day = datetime.strptime(end_arg, "%Y-%m-%d")
        end_local = end_day + timedelta(days=1) - timedelta(microseconds=1)
    else:
        end_local = max_end
    return start_local, end_local


def provider_utc_window(
    start_local: datetime | None, end_local: datetime | None
) -> tuple[datetime, datetime]:
    """Convert the local range to a padded UTC window for provider queries.

    Pads one day on each side so provider day-snapping (UTC) never clips a
    boundary session; for whole-range totals this padding is harmless because a
    dedicated key/project has no usage outside the ledger's span anyway.
    """
    if start_local is None:
        start_local = datetime(2000, 1, 1)
    if end_local is None:
        end_local = datetime.utcnow() + timedelta(days=1)
    start_utc = local_to_utc(start_local) - timedelta(days=1)
    end_utc = local_to_utc(end_local) + timedelta(days=1)
    return start_utc, end_utc


# ===========================================================================
# REPORT RENDERING
# ===========================================================================


def _fmt(n: float, unit: str) -> str:
    if unit == "$":
        return f"${n:,.4f}"
    if unit == "tokens" or unit == "credits" or unit == "chars":
        return f"{n:,.0f}"
    return f"{n:,.2f}"


def render_provider(
    name: str, lines: list[ReconLine], tolerance_pct: float, notes: list[str]
) -> tuple[str, str]:
    """Render one provider block; return (text, verdict)."""
    verdict = verdict_for(lines, tolerance_pct)
    out = []
    out.append(f"── {name} " + "─" * max(2, 40 - len(name)))
    header = f"{'metric':<26}{'ledger':>16}{'provider':>16}{'Δ abs':>16}{'Δ %':>10}"
    out.append(header)
    for line in lines:
        pct = line.pct_diff
        pct_s = "   n/a" if pct is None else f"{pct:+.2f}%"
        flag = "" if line.within(tolerance_pct) else "  ⚠"
        out.append(
            f"{line.label:<26}"
            f"{_fmt(line.ledger, line.unit):>16}"
            f"{_fmt(line.provider, line.unit):>16}"
            f"{_fmt(line.abs_diff, line.unit):>16}"
            f"{pct_s:>10}"
            f"{flag}"
        )
    for note in notes:
        out.append(f"  note: {note}")
    out.append(f"  VERDICT: {verdict} (tolerance ±{tolerance_pct:g}%)")
    return "\n".join(out), verdict


# ===========================================================================
# MAIN
# ===========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconcile the Voice Tutor local cost ledger against provider usage APIs.",
    )
    p.add_argument("--start", help="Range start YYYY-MM-DD (local). Default: ledger earliest.")
    p.add_argument("--end", help="Range end YYYY-MM-DD (local, inclusive). Default: ledger latest.")
    p.add_argument(
        "--ledger",
        help="Path to cost-log.jsonl (default: cost_audit.COST_LOG_JSONL_PATH).",
    )
    p.add_argument(
        "--tolerance-pct",
        type=float,
        default=None,
        help=f"Match tolerance in percent (default: RECONCILE_TOLERANCE_PCT env or {DEFAULT_TOLERANCE_PCT}).",
    )
    p.add_argument(
        "--providers",
        default="anthropic,deepgram,cartesia",
        help="Comma-separated subset of providers to reconcile.",
    )
    p.add_argument("--json", action="store_true", help="Also emit a machine-readable JSON summary.")
    return p


def resolve_tolerance(arg_value: float | None) -> float:
    if arg_value is not None:
        return arg_value
    env = os.environ.get("RECONCILE_TOLERANCE_PCT")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return DEFAULT_TOLERANCE_PCT


def reconcile_anthropic(totals: LedgerTotals, provider: dict[str, TokenBucket]) -> tuple[list[ReconLine], list[str]]:
    """Build Anthropic recon lines: per-class tokens (both families) + derived $."""
    lg_live, lg_haiku = totals.live_tokens, totals.haiku_tokens
    pv_live, pv_haiku = provider["live"], provider["haiku"]

    def cls(label, lg, pv):
        return ReconLine(label, "tokens", lg, pv)

    lines = [
        cls("live in (uncached)", lg_live.uncached_input, pv_live.uncached_input),
        cls("live cache read", lg_live.cache_read, pv_live.cache_read),
        cls("live cache write", lg_live.cache_write, pv_live.cache_write),
        cls("live output", lg_live.output, pv_live.output),
        cls("haiku in (uncached)", lg_haiku.uncached_input, pv_haiku.uncached_input),
        cls("haiku output", lg_haiku.output, pv_haiku.output),
    ]
    # Derived $ using the shared price list, both sides.
    ledger_derived = price_token_bucket(lg_live, "live") + price_token_bucket(lg_haiku, "haiku")
    provider_derived = price_token_bucket(pv_live, "live") + price_token_bucket(pv_haiku, "haiku")
    lines.append(ReconLine("derived cost", "$", ledger_derived, provider_derived))
    notes = [
        f"ledger recorded ${totals.recorded_anthropic_usd:,.4f} for Anthropic "
        f"(vs derived ${ledger_derived:,.4f} from token counts).",
        "provider $ is derived from provider token counts x our price list "
        "(Anthropic's per-key $ isn't exposed by the usage API).",
    ]
    return lines, notes


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    tolerance = resolve_tolerance(args.tolerance_pct)
    ledger_path = Path(args.ledger) if args.ledger else cost_audit.COST_LOG_JSONL_PATH
    requested = {p.strip().lower() for p in args.providers.split(",") if p.strip()}

    rows = load_ledger_rows(ledger_path)
    if not rows:
        print(f"No ledger rows found at {ledger_path}", file=sys.stderr)
        return 2

    start_local, end_local = resolve_range(rows, args.start, args.end)
    totals = summarize_ledger(rows, start_local, end_local)
    start_utc, end_utc = provider_utc_window(start_local, end_local)
    creds = load_credentials()

    print("Voice Tutor — provider cost reconciliation")
    print(f"ledger:  {ledger_path}")
    rng_lo = start_local.date() if start_local else "?"
    rng_hi = end_local.date() if end_local else "?"
    print(f"range:   {rng_lo} .. {rng_hi} (local)  ->  {_rfc3339(start_utc)} .. {_rfc3339(end_utc)} (UTC query window)")
    print(f"rows:    {totals.session_rows} session/legacy, {totals.artifact_rows} artifact")
    print()

    json_summary: dict = {"range": {"start_local": str(rng_lo), "end_local": str(rng_hi)}, "providers": {}}

    # --- Anthropic ---------------------------------------------------------
    if "anthropic" in requested:
        admin = creds.get("ANTHROPIC_ADMIN_KEY")
        if not admin:
            print("── Anthropic ───\n  SKIPPED: ANTHROPIC_ADMIN_KEY not found.\n")
        else:
            try:
                key_id = creds.get("ANTHROPIC_API_KEY_ID") or discover_anthropic_api_key_id(
                    admin, creds.get("ANTHROPIC_API_KEY")
                )
                if not key_id:
                    print(
                        "── Anthropic ───\n  SKIPPED: could not resolve the Voice Tutor api_key_id "
                        "(set ANTHROPIC_API_KEY_ID to override).\n"
                    )
                else:
                    provider = fetch_anthropic_usage(admin, key_id, start_utc, end_utc)
                    lines, notes = reconcile_anthropic(totals, provider)
                    notes.insert(0, f"filtered to api_key_id={key_id}")
                    text, verdict = render_provider("Anthropic", lines, tolerance, notes)
                    print(text + "\n")
                    json_summary["providers"]["anthropic"] = {
                        "verdict": verdict,
                        "lines": [_line_json(l) for l in lines],
                    }
            except ProviderError as e:
                print(f"── Anthropic ───\n  ERROR: {e}\n")

    # --- Deepgram ----------------------------------------------------------
    if "deepgram" in requested:
        dg_key = creds.get("DEEPGRAM_API_KEY")
        if not dg_key:
            print("── Deepgram ───\n  SKIPPED: DEEPGRAM_API_KEY not found.\n")
        else:
            try:
                project_id = creds.get("DEEPGRAM_PROJECT_ID") or discover_deepgram_project_id(dg_key)
                if not project_id:
                    print(
                        "── Deepgram ───\n  SKIPPED: could not resolve a single project "
                        "(set DEEPGRAM_PROJECT_ID to override).\n"
                    )
                else:
                    dg = fetch_deepgram_usage(
                        dg_key, project_id, start_utc.strftime("%Y-%m-%d"), end_utc.strftime("%Y-%m-%d")
                    )
                    lines = [ReconLine("STT billed minutes", "minutes", totals.stt_minutes, dg["stt_minutes"])]
                    notes = [
                        f"project_id={project_id}; provider total_hours={dg['total_hours']:.2f}, "
                        f"requests={dg['requests']:.0f}.",
                        "ledger stt_minutes_billed vs provider listen-endpoint hours x 60.",
                        f"ledger recorded ${totals.recorded_deepgram_usd:,.4f} for Deepgram STT.",
                    ]
                    text, verdict = render_provider("Deepgram", lines, tolerance, notes)
                    print(text + "\n")
                    json_summary["providers"]["deepgram"] = {
                        "verdict": verdict,
                        "lines": [_line_json(l) for l in lines],
                    }
            except ProviderError as e:
                print(f"── Deepgram ───\n  ERROR: {e}\n")

    # --- Cartesia ----------------------------------------------------------
    if "cartesia" in requested:
        ct_admin = creds.get("CARTESIA_ADMIN_KEY")
        if not ct_admin:
            print("── Cartesia ───\n  SKIPPED: CARTESIA_ADMIN_KEY not found.\n")
        else:
            try:
                ct = fetch_cartesia_usage(ct_admin, start_utc, end_utc)
                # Ledger chars -> credits at 1 credit/char.
                lines = [
                    ReconLine("TTS credits", "credits", totals.tts_credits, ct["credits"]),
                ]
                notes = [
                    "conversion assumption: 1 Cartesia credit == 1 TTS character.",
                    f"ledger tts_chars={totals.tts_chars:,.0f} treated as {totals.tts_credits:,.0f} credits.",
                    f"ledger recorded ${totals.recorded_cartesia_usd:,.4f} for Cartesia TTS.",
                    "whole-account credit usage (VT's Cartesia account is dedicated to this app).",
                ]
                text, verdict = render_provider("Cartesia", lines, tolerance, notes)
                print(text + "\n")
                json_summary["providers"]["cartesia"] = {
                    "verdict": verdict,
                    "lines": [_line_json(l) for l in lines],
                }
            except ProviderError as e:
                print(f"── Cartesia ───\n  ERROR: {e}\n")

    if args.json:
        print("─── JSON ───")
        print(json.dumps(json_summary, indent=2))

    return 0


def _line_json(line: ReconLine) -> dict:
    return {
        "label": line.label,
        "unit": line.unit,
        "ledger": line.ledger,
        "provider": line.provider,
        "abs_diff": line.abs_diff,
        "pct_diff": line.pct_diff,
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
