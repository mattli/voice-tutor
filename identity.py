"""Pure, Pipecat-free identity primitives: invite-token registry, cookie
constants, user_id sanitization, and the cookieless paste-your-code gate page.

Module-level ``TOKENS_PATH`` is read at CALL time so tests can monkeypatch it —
mirroring documents.DOCUMENTS_DIR / sessions.SESSION_LOG_JSONL_PATH.
"""

import json
import re
from pathlib import Path

TOKENS_PATH = Path.home() / ".voice-tutor" / "tokens.json"

COOKIE_NAME = "vt_uid"          # value is the invite TOKEN, resolved server-side
COOKIE_MAX_AGE = 31_536_000     # 1 year

_USER_ID_RE = re.compile(r"^[a-z0-9_-]+$")


def sanitize_user_id(raw: str) -> str | None:
    """Return ``raw`` if it is a filename-safe slug, else None. The user_id is a
    filename (not a secret); this guards path traversal and odd characters."""
    if not raw or not _USER_ID_RE.match(raw):
        return None
    # Belt-and-suspenders: the regex already forbids separators.
    return raw if Path(raw).name == raw else None


def load_registry() -> dict[str, str]:
    """Load the {token: user_id} map. Absent or malformed → empty (fail closed)."""
    path = TOKENS_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def resolve_user(token: str | None, registry: dict[str, str]) -> str | None:
    """Resolve an invite token to a sanitized user_id, or None. A registry value
    that isn't a valid slug resolves to None rather than leaking a bad filename."""
    if not token:
        return None
    user_id = registry.get(token)
    if user_id is None:
        return None
    return sanitize_user_id(user_id)


GATE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Tutor — Enter your invite</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background:#faf7f0; color:#1c1a17;
         display:flex; min-height:100vh; margin:0; align-items:center; justify-content:center; }
  .card { max-width:360px; padding:32px 28px; }
  h1 { font-size:20px; margin:0 0 8px; }
  p { color:#6b6459; font-size:14px; margin:0 0 20px; }
  input { width:100%; padding:10px 12px; font-size:15px; border:1px solid #d9d2c2;
          border-radius:8px; box-sizing:border-box; }
  button { margin-top:12px; width:100%; padding:10px; font-size:15px; border:0;
           border-radius:8px; background:#2d4a6b; color:#fff; cursor:pointer; }
</style></head>
<body><main class="card">
  <h1>Enter your invite link or code</h1>
  <p>Paste the invite code you were sent, or open your invite link again.</p>
  <form onsubmit="event.preventDefault(); var c=document.getElementById('code').value.trim();
                  if(c) location.href='/study/?u='+encodeURIComponent(c);">
    <input id="code" autofocus autocomplete="off" placeholder="invite code">
    <button type="submit">Continue</button>
  </form>
</main></body></html>
"""
