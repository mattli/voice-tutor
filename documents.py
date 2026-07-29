"""Document storage and text extraction for study-mode sessions.

No DB. Doc list is computed at request time from ~/.voice-tutor/documents/*.txt
(extracted text), with the original file kept alongside under
<uuid>-<original-filename> for provenance.
"""

import asyncio
import io
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
from pypdf import PdfReader

DOCUMENTS_DIR = Path.home() / ".voice-tutor" / "documents"
# The shared-document namespace: documents placed under documents/_shared/ are
# offered to EVERY user (unioned into each user's picker) and are loadable by
# every user, as a resolution fallback — user namespace first, then _shared/. All
# per-user STATE about them (sessions, recaps, coverage) stays keyed on user_id;
# only the document text/claims are shared. There is no app write path to
# _shared/ (save_upload never writes here — see save_upload); shared docs are
# placed on the filesystem by hand. ``_shared`` is a reserved user_id
# (identity.sanitize_user_id rejects it) so no minted user can alias this dir.
SHARED_USER_ID = "_shared"
MAX_DOC_CHARS = 150_000
MAX_UPLOAD_BYTES = 5_000_000
ALLOWED_EXTS = {".pdf", ".md", ".txt", ".markdown"}

SUMMARY_PROMPT = (
    "Summarize the following document in 1–2 sentences of plain prose. "
    "No preamble, no quotation marks, no headers — just the summary itself. "
    "Aim for a sentence a reader could glance at to remember what the document is about.\n\n"
    "Document:\n{text}"
)
SUMMARY_MAX_CHARS_IN = 40_000


class UploadError(Exception):
    """Raised for user-correctable upload problems (wrong type, too big, etc.)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _extract_text(filename: str, raw: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _derive_title(text: str, filename: str) -> str:
    lines = text.splitlines()
    start = 0
    # Skip a YAML frontmatter block so we don't return its "---" delimiter.
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                start = j + 1
                break
    # Prefer the first markdown H1 heading.
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.lstrip("#").strip()[:120]
    # Fall back to the first non-empty line, then to the filename.
    for line in lines[start:]:
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return Path(filename).stem


def user_dir(user_id: str) -> Path:
    return DOCUMENTS_DIR / Path(user_id).name


def _shared_dir() -> Path:
    """Directory for the shared-document namespace (documents/_shared/)."""
    return DOCUMENTS_DIR / SHARED_USER_ID


def _summary_path(user_id: str, doc_id: str) -> Path:
    return user_dir(user_id) / f"{doc_id}.summary.txt"


def _generate_summary(text: str) -> str | None:
    """Best-effort Haiku call. Returns the summary, or None on failure."""
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(text=text[:SUMMARY_MAX_CHARS_IN])}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"[doc-summary] failed: {e}", file=sys.stderr, flush=True)
        return None


def save_upload(user_id: str, filename: str, raw: bytes) -> dict:
    """Validate, extract, and persist a document. Returns metadata."""
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UploadError(413, f"file too large (max {MAX_UPLOAD_BYTES} bytes)")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise UploadError(415, f"unsupported file type: {ext or '(none)'}")

    text = _extract_text(filename, raw)
    if len(text) > MAX_DOC_CHARS:
        raise UploadError(
            413,
            f"extracted text too long ({len(text)} chars, max {MAX_DOC_CHARS})",
        )
    if not text:
        raise UploadError(422, "could not extract any text from this file")

    d = user_dir(user_id)
    d.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid.uuid4())
    safe_name = Path(filename).name
    (d / f"{doc_id}-{safe_name}").write_bytes(raw)
    (d / f"{doc_id}.txt").write_text(text)

    summary = _generate_summary(text)
    if summary:
        _summary_path(user_id, doc_id).write_text(summary)

    return {
        "document_id": doc_id,
        "title": _derive_title(text, safe_name),
        "char_count": len(text),
        "summary": summary,
    }


def _scan_documents(d: Path) -> list[dict]:
    """Scan directory ``d`` for documents, returning one entry per ``*.txt``.

    Pure filesystem read — no summary backfill (the caller owns that so it can
    write the sidecar into the correct namespace). Each entry carries an internal
    ``_dir`` key (the owning directory) so the caller can locate the summary
    sidecar for backfill; it is stripped before the entry is returned to callers.
    """
    if not d.exists():
        return []
    docs = []
    # Skip the .summary.txt sidecars; they're not their own documents.
    txt_paths = [p for p in sorted(d.glob("*.txt")) if not p.name.endswith(".summary.txt")]
    for txt_path in txt_paths:
        doc_id = txt_path.stem
        text = txt_path.read_text()
        originals = [p for p in d.glob(f"{doc_id}-*") if p != txt_path and not p.name.endswith(".summary.txt")]
        original = originals[0] if originals else txt_path
        display_name = original.name.removeprefix(f"{doc_id}-")
        summary_path = d / f"{doc_id}.summary.txt"
        summary = summary_path.read_text().strip() if summary_path.exists() else None
        docs.append({
            "document_id": doc_id,
            "title": _derive_title(text, display_name),
            "char_count": len(text),
            "uploaded_at": datetime.fromtimestamp(original.stat().st_mtime).isoformat(),
            "summary": summary,
            "_dir": d,
            "_text": text,
        })
    return docs


async def list_documents(user_id: str) -> list[dict]:
    # Union the user's own docs with the shared namespace (documents/_shared/),
    # so a doc seeded in _shared/ appears in every user's picker. Resolution is
    # user-first: on a doc_id collision the user's own doc SHADOWS the shared one
    # (matching load_document), and the id appears exactly once.
    own = _scan_documents(user_dir(user_id))
    seen = {d["document_id"] for d in own}
    shared = [d for d in _scan_documents(_shared_dir()) if d["document_id"] not in seen]
    docs = own + shared

    # Backfill missing summaries, writing each sidecar into ITS OWN namespace
    # (a shared doc's summary lives in _shared/, a user doc's in the user dir).
    needs_backfill = [(i, d["_text"]) for i, d in enumerate(docs) if d["summary"] is None]
    if needs_backfill:
        results = await asyncio.gather(
            *(asyncio.to_thread(_generate_summary, text) for _, text in needs_backfill)
        )
        for (idx, _text), summary in zip(needs_backfill, results):
            if summary:
                (docs[idx]["_dir"] / f"{docs[idx]['document_id']}.summary.txt").write_text(summary)
                docs[idx]["summary"] = summary

    for d in docs:
        del d["_dir"]
        del d["_text"]

    docs.sort(key=lambda d: d["uploaded_at"], reverse=True)
    return docs


def _load_from_dir(d: Path, doc_id: str) -> tuple[str, str] | None:
    """Load (title, text) for ``doc_id`` from directory ``d``, or None on miss.

    ``doc_id`` is sanitized to a single path component (``Path(doc_id).name``) at
    this helper boundary — the shared choke point every read path funnels through
    (``load_document`` -> here, reached from bot.py/app.py/sessions.py). A crafted
    id like ``../<other_user>/<uuid>`` or an absolute path would otherwise string-
    join its way OUT of ``d`` and read another user's document; collapsing it to
    the final component keeps the lookup inside ``d`` (a miss, hence None, unless
    the caller genuinely owns that name). Mirrors ``user_dir``'s ``Path.name``
    guard on the user_id half and ``app.py``'s ``safe_id = Path(doc_id).name``."""
    doc_id = Path(doc_id).name
    txt_path = d / f"{doc_id}.txt"
    if not txt_path.exists():
        return None
    text = txt_path.read_text()
    originals = [p for p in d.glob(f"{doc_id}-*") if p != txt_path]
    original_name = originals[0].name.removeprefix(f"{doc_id}-") if originals else f"{doc_id}.txt"
    return _derive_title(text, original_name), text


def load_document(user_id: str, doc_id: str) -> tuple[str, str] | None:
    """Return (title, text) or None if not found.

    Resolves the user's own namespace FIRST, then falls back to the shared
    namespace (documents/_shared/) on a miss. A user's own doc with a colliding
    id therefore shadows the shared one (deterministic)."""
    return _load_from_dir(user_dir(user_id), doc_id) or _load_from_dir(_shared_dir(), doc_id)
