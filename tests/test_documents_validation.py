"""Characterization tests for documents.save_upload validation error paths.

Each rejection branch present in ``save_upload`` (documents.py) maps 1:1 to a
dedicated test below. The code raises the concrete ``documents.UploadError``
(the most-derived type) for every rejection, and ``str(UploadError)`` equals
its ``detail`` (because ``UploadError.__init__`` calls ``super().__init__(detail)``).

Rejection branches in save_upload, cross-checked against the source:

  B1  len(raw) > MAX_UPLOAD_BYTES        -> UploadError(413, "file too large ...")
  B2  ext not in ALLOWED_EXTS            -> UploadError(415, "unsupported file type: ...")
  B3  len(text) > MAX_DOC_CHARS          -> UploadError(413, "extracted text too long ...")
  B4  not text (empty after extraction)  -> UploadError(422, "could not extract any text ...")

Branch count: 4. Each expected message is reconstructed from the same
test-controlled inputs / module constants used by the code, so interpolated
values are asserted verbatim (no substring/startswith/opaque-path copying).
The B2 "no extension" case is included as an explicit variant of the same
branch to pin the ``ext or '(none)'`` interpolation.
"""

import documents
from documents import MAX_DOC_CHARS, MAX_UPLOAD_BYTES, UploadError, save_upload


def test_b1_file_too_large(docs_dir):
    # B1: raw exceeds MAX_UPLOAD_BYTES.
    raw = b"x" * (MAX_UPLOAD_BYTES + 1)
    expected = f"file too large (max {MAX_UPLOAD_BYTES} bytes)"
    try:
        save_upload("a.txt", raw)
        assert False, "expected UploadError"
    except UploadError as exc:
        assert type(exc) is UploadError
        assert exc.status_code == 413
        assert exc.detail == expected
        assert str(exc) == expected


def test_b2_unsupported_extension(docs_dir):
    # B2: extension not in ALLOWED_EXTS. Uses a controlled extension value.
    ext = ".docx"
    expected = f"unsupported file type: {ext}"
    try:
        save_upload("report.docx", b"anything")
        assert False, "expected UploadError"
    except UploadError as exc:
        assert type(exc) is UploadError
        assert exc.status_code == 415
        assert exc.detail == expected
        assert str(exc) == expected


def test_b2_no_extension_reports_none_placeholder(docs_dir):
    # B2 (variant): no extension -> ext == "" -> the code renders "(none)".
    ext = ""  # Path("noext").suffix.lower() == ""
    expected = f"unsupported file type: {ext or '(none)'}"
    try:
        save_upload("noext", b"anything")
        assert False, "expected UploadError"
    except UploadError as exc:
        assert type(exc) is UploadError
        assert exc.status_code == 415
        assert exc.detail == expected
        assert str(exc) == expected


def test_b3_extracted_text_too_long(docs_dir):
    # B3: extracted text length exceeds MAX_DOC_CHARS. A plain-ascii .txt body
    # extracts to itself (no collapse triggered), so len(text) is controlled.
    overflow = MAX_DOC_CHARS + 10
    raw = ("a" * overflow).encode("utf-8")
    text_len = overflow  # extraction of pure-ascii, no leading/trailing ws == input length
    expected = f"extracted text too long ({text_len} chars, max {MAX_DOC_CHARS})"
    try:
        save_upload("big.txt", raw)
        assert False, "expected UploadError"
    except UploadError as exc:
        assert type(exc) is UploadError
        assert exc.status_code == 413
        assert exc.detail == expected
        assert str(exc) == expected


def test_b4_empty_extracted_text(docs_dir):
    # B4: extraction yields empty text (whitespace-only input strips to "").
    expected = "could not extract any text from this file"
    try:
        save_upload("blank.txt", b"   \n\n  ")
        assert False, "expected UploadError"
    except UploadError as exc:
        assert type(exc) is UploadError
        assert exc.status_code == 422
        assert exc.detail == expected
        assert str(exc) == expected
