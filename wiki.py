"""Wiki knowledge integration for the voice tutor.

Two-phase pattern: INDEX.md is embedded in the system prompt at session start;
individual pages are fetched on-demand via the `read_wiki_page` tool when Claude
decides a topic is central enough to warrant the round-trip.

Kept independent of the speech-to-speech pipeline — bot.py imports this module
only when WIKI_ENABLED is on, and the tool handler receives a generic
`on_call_start(name)` callback so it doesn't have to know about UsageAccumulator
or any other piece of pipeline plumbing.
"""

import sys
from pathlib import Path
from typing import Awaitable, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema

# Pure, Pipecat-free grounding helpers were relocated verbatim into grounding.py.
# Re-export them (including the shared WIKI_DIR constant, on which the retained
# tool_schema/make_tool_handler below still rely) so existing callers of wiki.py
# keep working unchanged. This is a one-way shim: grounding.py never imports wiki.
from grounding import WIKI_DIR, USAGE_INSTRUCTIONS, _load_index, system_prompt_block


def tool_schema() -> FunctionSchema:
    return FunctionSchema(
        name="read_wiki_page",
        description=(
            "Open a page from Matt's knowledge wiki. Pass the path relative to "
            "the wiki root exactly as shown in the index in the system prompt, "
            "e.g. 'concepts/llm-knowledge-bases.md' or 'landscape/yc-ai-thesis.md'."
        ),
        properties={
            "path": {
                "type": "string",
                "description": "Path relative to wiki root, e.g. 'concepts/llm-knowledge-bases.md'.",
            },
        },
        required=["path"],
    )


def make_tool_handler(on_call_start: Callable[[str], None]) -> Callable[..., Awaitable[None]]:
    """Build the read_wiki_page handler.

    on_call_start is invoked with the tool-call argument right before the file
    is read — bot.py wires this to UsageAccumulator.mark_tool_call to record
    latency-to-first-audio. The handler itself stays ignorant of telemetry.
    """
    async def handle(params):
        path = params.arguments.get("path", "")
        requested = (WIKI_DIR / path).resolve()
        wiki_root = WIKI_DIR.resolve()
        try:
            requested.relative_to(wiki_root)
        except ValueError:
            await params.result_callback({"error": "path must be inside the wiki"})
            return
        if not requested.exists():
            await params.result_callback({"error": f"page not found: {path}"})
            return
        on_call_start(path)
        print(f"[wiki-tool] opening {path}", file=sys.stderr, flush=True)
        await params.result_callback({"content": requested.read_text()})

    return handle
