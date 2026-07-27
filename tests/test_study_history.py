import study_history as sh

_RECAP = """# Study session — Graph Engineering
Duration: 22:46

## What we covered
- What a graph is: nodes and edges
- The fake-edge test

## Key points
### Nodes
Long essay that must NOT appear in the parsed result.

## Open threads
- How to resolve hidden edges
- The verification architecture section
"""


def test_parses_covered_and_open_threads():
    out = sh.parse_recap_sections(_RECAP)
    assert out == {
        "covered": ["What a graph is: nodes and edges", "The fake-edge test"],
        "open_threads": [
            "How to resolve hidden edges",
            "The verification architecture section",
        ],
    }
    assert "fallback_text" not in out


def test_open_threads_optional_empty_list_when_absent():
    text = "## What we covered\n- Only this\n\n## Key points\nblah\n"
    out = sh.parse_recap_sections(text)
    assert out == {"covered": ["Only this"], "open_threads": []}


def test_unparseable_returns_truncated_fallback():
    text = "x" * 5000  # no headers at all
    out = sh.parse_recap_sections(text)
    assert out == {"fallback_text": "x" * 1000}
    assert "covered" not in out
