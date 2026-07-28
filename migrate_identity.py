"""One-time, idempotent identity migration + ledger backfill.

Pure helpers are unit-tested; the ``__main__`` block (added in Task 14) runs them
against the real ~/.voice-tutor and vault dirs, archiving originals first.
"""

import json

DEFAULT_USER_ID = "matt"


def backfill_ledger_user_id(lines: list[str], default_user_id: str = DEFAULT_USER_ID) -> list[str]:
    """Return ledger lines with ``user_id`` added to any JSON row lacking one.

    Idempotent (rows already carrying user_id are unchanged); non-JSON lines pass
    through verbatim; no other field is touched; line count is preserved.
    """
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        try:
            row = json.loads(stripped)
        except Exception:
            out.append(line)
            continue
        if isinstance(row, dict) and "user_id" not in row:
            row = {"user_id": default_user_id, **row}
            out.append(json.dumps(row))
        else:
            out.append(line if not isinstance(row, dict) else json.dumps(row))
    return out
