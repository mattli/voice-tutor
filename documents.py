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


class DocumentActionError(Exception):
    """Raised for a refused archive/restore, carrying the status the route returns."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


# Removed documents are MOVED here, never deleted. The picker's scan globs
# ``*.txt`` non-recursively, so a subdirectory is invisible to it with no filter
# logic to keep in sync — and `_load_from_dir` builds an exact path, so an
# archived document also stops being loadable for study. Both properties fall
# out of the move itself rather than being enforced.
ARCHIVE_DIRNAME = "_archive"


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


def _archive_root(user_id: str) -> Path:
    return user_dir(user_id) / ARCHIVE_DIRNAME


def _doc_files(d: Path, doc_id: str) -> list[Path]:
    """Every file belonging to ``doc_id`` in directory ``d``.

    The extracted text, the original upload, the summary sidecar and the claim
    map all share the ``<doc_id>`` stem, so matching on it keeps the set complete
    without this function having to know each sidecar's suffix — a new sidecar
    type is archived with its document automatically.

    A BARE prefix match would be wrong: ``<doc_id>*`` also matches a different
    document whose id merely starts with these characters, and archiving one
    document must never move another's files. Only the exact ``<doc_id>.txt``,
    ``<doc_id>.<suffix>`` and ``<doc_id>-<original name>`` forms this module
    actually writes are matched.
    """
    doc_id = Path(doc_id).name
    return sorted(
        p
        for p in d.glob(f"{doc_id}*")
        if p.is_file()
        and (p.name == doc_id or p.name.startswith((f"{doc_id}.", f"{doc_id}-")))
    )


def archive_document(user_id: str, doc_id: str, *, stamp: str | None = None) -> dict:
    """Move ``doc_id`` out of ``user_id``'s picker into their archive. Never deletes.

    Returns ``{"document_id", "archived_at", "files"}``. Raises
    :class:`DocumentActionError` with the status the route should return:

      * **409** for a document in the SHARED namespace. It belongs to every
        user and there is no app write path to that directory, so one user
        removing it would silently remove it for everyone. Refusing is the only
        honest answer until per-user hiding exists.
      * **404** when the id resolves to no document at all.

    The move is reversible by :func:`restore_document` and nothing else is
    touched — transcripts, coverage sidecars, artifacts and ledger rows are
    RECORDS of sessions that really happened, and removing a document does not
    unhappen them. A coverage sidecar for an archived document simply has no
    document to attach to; it neither breaks the union nor shows a bar, because
    both are driven by documents that exist.
    """
    safe_id = Path(doc_id).name
    own = user_dir(user_id)
    if not (own / f"{safe_id}.txt").exists():
        if (_shared_dir() / f"{safe_id}.txt").exists():
            raise DocumentActionError(
                409, "shared documents can't be removed — they belong to every user"
            )
        raise DocumentActionError(404, "document not found")

    stamp = stamp or datetime.now().strftime("%Y-%m-%d-%H%M%S")
    target = _archive_root(user_id) / f"{safe_id}-{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in _doc_files(own, safe_id):
        path.rename(target / path.name)
        moved.append(path.name)
    return {"document_id": safe_id, "archived_at": stamp, "files": moved}


def _newest_archive_dir(user_id: str, doc_id: str) -> Path | None:
    """The most recent archive folder for ``doc_id``, or None if never archived."""
    safe_id = Path(doc_id).name
    root = _archive_root(user_id)
    if not root.is_dir():
        return None
    candidates = sorted(p for p in root.glob(f"{safe_id}-*") if p.is_dir())
    return candidates[-1] if candidates else None


def restore_document(user_id: str, doc_id: str) -> dict:
    """Move the most recently archived copy of ``doc_id`` back into the picker.

    Powers the undo affordance, and is also the manual recovery path long after
    the toast is gone — which is the reason archiving beats deleting. Raises
    :class:`DocumentActionError`: **404** when nothing is archived under that id,
    **409** when a live document already occupies it (restoring would otherwise
    overwrite a document that exists now).
    """
    safe_id = Path(doc_id).name
    source = _newest_archive_dir(user_id, safe_id)
    if source is None:
        raise DocumentActionError(404, "no archived document with that id")
    own = user_dir(user_id)
    if (own / f"{safe_id}.txt").exists():
        raise DocumentActionError(409, "a document with that id is already live")
    own.mkdir(parents=True, exist_ok=True)
    restored = []
    for path in sorted(source.glob("*")):
        if path.is_file():
            path.rename(own / path.name)
            restored.append(path.name)
    if not any(source.iterdir()):
        source.rmdir()
    return {"document_id": safe_id, "files": restored}


def resolve_title(user_id: str, doc_id: str) -> str | None:
    """Title for a document that may since have been archived — HISTORY ONLY.

    Past sessions on an archived document must keep their names; a session that
    really happened should not render as "Unknown document" because the document
    was later put away. Deliberately NOT a fallback inside
    :func:`load_document`: that would make an archived document studyable again,
    which is the one thing archiving is supposed to prevent. Callers that start
    or ground a session must keep using ``load_document``.
    """
    loaded = load_document(user_id, doc_id)
    if loaded is not None:
        return loaded[0]
    source = _newest_archive_dir(user_id, doc_id)
    if source is None:
        return None
    archived = _load_from_dir(source, doc_id)
    return archived[0] if archived else None


def load_document(user_id: str, doc_id: str) -> tuple[str, str] | None:
    """Return (title, text) or None if not found.

    Resolves the user's own namespace FIRST, then falls back to the shared
    namespace (documents/_shared/) on a miss. A user's own doc with a colliding
    id therefore shadows the shared one (deterministic)."""
    return _load_from_dir(user_dir(user_id), doc_id) or _load_from_dir(_shared_dir(), doc_id)
