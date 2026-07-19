"""Pure, Pipecat-free wiki grounding helpers relocated verbatim from wiki.py.

These functions and the constants they depend on were moved here with zero
logic changes so they can be characterized in isolation and imported without
pulling in the Pipecat FunctionSchema / tool-wiring plumbing. wiki.py re-imports
every name defined here so existing callers keep working unchanged.

Home of shared wiki grounding logic — deliberately NOT session_state.py, which
owns transcript/profile/memory helpers unrelated to the wiki.
"""

from pathlib import Path

WIKI_DIR = Path.home() / "second-brain" / "resources" / "wiki"

USAGE_INSTRUCTIONS = (
    "### How to use the wiki\n\n"
    "- When you need a wiki page, say one short sentence first (e.g. "
    "\"let me pull that up\"), then call `read_wiki_page(path)`. This prevents "
    "silence during the lookup.\n"
    "- Don't open pages speculatively. Open a page only when a specific topic "
    "is central to Matt's current question. One page per question is typical "
    "— don't chain-open multiple pages unless Matt explicitly asks about "
    "multiple topics.\n"
    "- Pass the path relative to the wiki root exactly as shown in the index, "
    "e.g. 'concepts/llm-knowledge-bases.md' or 'landscape/yc-ai-thesis.md'."
)


def _load_index() -> str:
    index_path = WIKI_DIR / "INDEX.md"
    if not index_path.exists():
        return ""
    return index_path.read_text()


def system_prompt_block() -> str | None:
    """Return the wiki section to embed in the system prompt, or None if there's no index."""
    index = _load_index()
    if not index:
        return None
    return f"\n## Matt's knowledge wiki\n\n{index}\n\n{USAGE_INSTRUCTIONS}"
