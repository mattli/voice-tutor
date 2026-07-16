"""Characterization tests for documents._extract_text (the real dispatch).

All extraction — including PDF — is exercised THROUGH documents.py's own
``_extract_text`` entrypoint. No PDF/text extraction library is imported or
called directly here, so the pinned literals reflect production's extractor
exactly.

Dispatch in ``_extract_text`` (documents.py), cross-checked against source:

    ext = Path(filename).suffix.lower()
    if ext == ".pdf":   -> pypdf.PdfReader path
    else:               -> raw.decode("utf-8", errors="replace")
    then: re.sub(r"\n{3,}", "\n\n", text).strip()

The real input signature production callers use is ``_extract_text(filename,
raw_bytes)`` (called from save_upload with the uploaded filename and raw
bytes). The supported formats accepted upstream are ALLOWED_EXTS =
{".pdf", ".md", ".txt", ".markdown"}. Extraction routes each of {.md, .txt,
.markdown} through the SAME non-pdf (else) decode branch, and .pdf through the
PDF branch. The "else" branch is itself the dispatch fallback for any
non-.pdf extension.

Each test asserts full == equality against a hardcoded literal with NO
post-processing/normalization.
"""

from documents import _extract_text


def test_extract_pdf_via_entrypoint(sample_pdf_bytes):
    # .pdf -> PdfReader branch, exercised via the committed fixture.
    # Observed present-day output of the repo's own extractor in-environment.
    result = _extract_text("sample.pdf", sample_pdf_bytes)
    assert result == "Hello Voice Tutor\nThis is a fixture document."


def test_extract_txt_decode_and_collapse():
    # .txt -> else (utf-8 decode) branch; 4+ newlines collapse to 2 and strip.
    raw = b"line1\n\n\n\nline2\n\n\n"
    assert _extract_text("a.txt", raw) == "line1\n\nline2"


def test_extract_md_decode():
    # .md -> else (utf-8 decode) branch; markdown text passes through verbatim.
    raw = "# Title\ncontent".encode("utf-8")
    assert _extract_text("a.md", raw) == "# Title\ncontent"


def test_extract_markdown_decode_with_replacement():
    # .markdown -> else branch; invalid utf-8 byte becomes U+FFFD (errors="replace").
    raw = b"caf\xe9"
    assert _extract_text("a.markdown", raw) == "caf\ufffd"


def test_extract_fallback_non_pdf_extension_uses_decode_branch():
    # Dispatch fallback: any non-.pdf extension routes through the else/decode
    # branch (this is the extension->extractor fallback, not the .pdf branch).
    raw = b"plain text body"
    assert _extract_text("whatever.bin", raw) == "plain text body"
