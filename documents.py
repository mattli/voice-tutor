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


def _summary_path(doc_id: str) -> Path:
    return DOCUMENTS_DIR / f"{doc_id}.summary.txt"


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


def save_upload(filename: str, raw: bytes) -> dict:
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

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid.uuid4())
    safe_name = Path(filename).name
    (DOCUMENTS_DIR / f"{doc_id}-{safe_name}").write_bytes(raw)
    (DOCUMENTS_DIR / f"{doc_id}.txt").write_text(text)

    summary = _generate_summary(text)
    if summary:
        _summary_path(doc_id).write_text(summary)

    return {
        "document_id": doc_id,
        "title": _derive_title(text, safe_name),
        "char_count": len(text),
        "summary": summary,
    }


async def list_documents() -> list[dict]:
    if not DOCUMENTS_DIR.exists():
        return []
    docs = []
    # Skip the .summary.txt sidecars; they're not their own documents.
    txt_paths = [p for p in sorted(DOCUMENTS_DIR.glob("*.txt")) if not p.name.endswith(".summary.txt")]
    needs_backfill: list[tuple[int, str]] = []  # (index, text) for parallel summarization
    for txt_path in txt_paths:
        doc_id = txt_path.stem
        text = txt_path.read_text()
        originals = [p for p in DOCUMENTS_DIR.glob(f"{doc_id}-*") if p != txt_path and not p.name.endswith(".summary.txt")]
        original = originals[0] if originals else txt_path
        display_name = original.name.removeprefix(f"{doc_id}-")
        summary_path = _summary_path(doc_id)
        summary = summary_path.read_text().strip() if summary_path.exists() else None
        if summary is None:
            needs_backfill.append((len(docs), text))
        docs.append({
            "document_id": doc_id,
            "title": _derive_title(text, display_name),
            "char_count": len(text),
            "uploaded_at": datetime.fromtimestamp(original.stat().st_mtime).isoformat(),
            "summary": summary,
        })

    if needs_backfill:
        results = await asyncio.gather(
            *(asyncio.to_thread(_generate_summary, text) for _, text in needs_backfill)
        )
        for (idx, _text), summary in zip(needs_backfill, results):
            if summary:
                _summary_path(docs[idx]["document_id"]).write_text(summary)
                docs[idx]["summary"] = summary

    docs.sort(key=lambda d: d["uploaded_at"], reverse=True)
    return docs


def load_document(doc_id: str) -> tuple[str, str] | None:
    """Return (title, text) or None if not found."""
    txt_path = DOCUMENTS_DIR / f"{doc_id}.txt"
    if not txt_path.exists():
        return None
    text = txt_path.read_text()
    originals = [p for p in DOCUMENTS_DIR.glob(f"{doc_id}-*") if p != txt_path]
    original_name = originals[0].name.removeprefix(f"{doc_id}-") if originals else f"{doc_id}.txt"
    return _derive_title(text, original_name), text
