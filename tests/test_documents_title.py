"""Characterization tests for documents._derive_title.

These pin the CURRENT observed output of the real function imported from
documents.py. They document behavior, not correctness.

Enumeration of distinct output-affecting paths in ``_derive_title``
(documents.py, def _derive_title) — cross-checked against the source below.
The function has exactly 4 return statements; each is a distinct path, and the
truncation / lstrip edge behaviors of those returns are pinned as additional
observed cases:

  P1  YAML frontmatter block skipped, then a later "# " H1 is returned
      (return at ``return stripped.lstrip("#").strip()[:120]`` reached AFTER
      the frontmatter ``start`` advance).
  P2  First markdown "# " H1 heading returned (no frontmatter).
  P3  H1 heading, "#" chars stripped and result truncated to 120 chars.
  P4  Fallback: first non-empty line (after lstrip("#")) returned.
  P5  Fallback first-non-empty line truncated to 120 chars.
  P6  Final fallback: empty text -> Path(filename).stem.
  P7  Unterminated frontmatter (no closing "---"): start stays 0, so the
      opening "---" line itself becomes the first non-empty fallback line.

Path count enumerated here: 7 (4 return statements + 3 edge behaviors of those
returns). The underlying return-statement count in the source is 4.
"""

from documents import _derive_title


def test_p1_frontmatter_then_h1():
    # P1: skip a YAML frontmatter block, then return the first H1 after it.
    text = "---\ntitle: meta\nauthor: x\n---\n# Real Heading\nbody text"
    assert _derive_title(text, "f.txt") == "Real Heading"


def test_p2_h1_heading():
    # P2: first "# " H1 heading, no frontmatter.
    text = "# Hello World\nsome body"
    assert _derive_title(text, "f.txt") == "Hello World"


def test_p3_h1_truncated_to_120():
    # P3: H1 return path is truncated to 120 chars via [:120].
    text = "# " + ("A" * 200)
    assert _derive_title(text, "f.txt") == "A" * 120


def test_p4_first_non_empty_fallback():
    # P4: no H1 -> first non-empty line, stripped and lstrip("#")-ed.
    text = "\n\n  first line  \nsecond line"
    assert _derive_title(text, "f.txt") == "first line"


def test_p4b_non_h1_hash_line_stripped_in_fallback():
    # P4 (variant): a "###" line is NOT an H1 ("# " prefix required), so it
    # falls through to the fallback loop which lstrip("#")-strips it.
    text = "### Heading\nbody"
    assert _derive_title(text, "f.txt") == "Heading"


def test_p5_fallback_truncated_to_120():
    # P5: fallback first-non-empty line is truncated to 120 chars via [:120].
    text = "B" * 200
    assert _derive_title(text, "f.txt") == "B" * 120


def test_p6_empty_text_falls_back_to_filename_stem():
    # P6: empty text -> Path(filename).stem.
    assert _derive_title("", "myfile.txt") == "myfile"


def test_p7_unterminated_frontmatter_returns_delimiter():
    # P7: opening "---" with no closing "---" leaves start=0, so the "---"
    # line itself is the first non-empty fallback line and is returned.
    text = "---\nonly line\nno end delimiter"
    assert _derive_title(text, "f.md") == "---"
