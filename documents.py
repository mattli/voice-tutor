"""Document storage and text extraction for study-mode sessions.

No DB. Doc list is computed at request time from ~/.voice-tutor/documents/*.txt
(extracted text), with the original file kept alongside under
<uuid>-<original-filename> for provenance.
"""

import io
import re
import uuid
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

DOCUMENTS_DIR = Path.home() / ".voice-tutor" / "documents"
MAX_DOC_CHARS = 150_000
MAX_UPLOAD_BYTES = 5_000_000
ALLOWED_EXTS = {".pdf", ".md", ".txt", ".markdown"}


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
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:120]
    return filename


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

    return {
        "document_id": doc_id,
        "title": _derive_title(text, safe_name),
        "char_count": len(text),
    }


def list_documents() -> list[dict]:
    if not DOCUMENTS_DIR.exists():
        return []
    docs = []
    for txt_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        doc_id = txt_path.stem
        text = txt_path.read_text()
        originals = [p for p in DOCUMENTS_DIR.glob(f"{doc_id}-*") if p != txt_path]
        original = originals[0] if originals else txt_path
        docs.append({
            "document_id": doc_id,
            "title": _derive_title(text, original.name),
            "char_count": len(text),
            "uploaded_at": datetime.fromtimestamp(original.stat().st_mtime).isoformat(),
        })
    docs.sort(key=lambda d: d["uploaded_at"], reverse=True)
    return docs


def load_document(doc_id: str) -> tuple[str, str] | None:
    """Return (title, text) or None if not found."""
    txt_path = DOCUMENTS_DIR / f"{doc_id}.txt"
    if not txt_path.exists():
        return None
    text = txt_path.read_text()
    originals = [p for p in DOCUMENTS_DIR.glob(f"{doc_id}-*") if p != txt_path]
    original_name = originals[0].name if originals else f"{doc_id}.txt"
    return _derive_title(text, original_name), text
