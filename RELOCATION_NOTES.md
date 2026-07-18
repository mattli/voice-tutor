# Sprint 0 — extract pure helpers and characterize

## New module: `grounding.py` (NOT session_state.py)

`session_state.py` already exists and owns transcript/profile/memory helpers,
which are unrelated to wiki grounding. The relocated wiki helpers therefore live
in a new, purpose-named module: **`grounding.py`**.

## wiki.py top-level symbol rulings

| Symbol               | Ruling  | Reason |
|----------------------|---------|--------|
| `WIKI_DIR`           | INCLUDE | Pure `Path` constant. **Ownership moves to `grounding.py`** and is re-exported into `wiki.py` so the retained EXCLUDE functions (`tool_schema`, `make_tool_handler`) still resolve it. |
| `USAGE_INSTRUCTIONS` | INCLUDE | Pure string constant; read by `system_prompt_block`. |
| `_load_index`        | INCLUDE | Pure filesystem read; no Pipecat/LLM/network. |
| `system_prompt_block`| INCLUDE | Pure; calls `_load_index` + reads `USAGE_INSTRUCTIONS`, both of which move with it (no callback into wiki → no cycle). |
| `tool_schema`        | EXCLUDE | Constructs and returns a Pipecat `FunctionSchema`; its `from pipecat.adapters.schemas.function_schema import FunctionSchema` import stays in `wiki.py`. |
| `make_tool_handler`  | EXCLUDE | Imports NO pipecat (so it is *not* "Pipecat-tangled"). It is tool-wiring plumbing built around `params.result_callback` and an injected `on_call_start` callback — pipeline plumbing, not a pure grounding helper. Left in `wiki.py`. |

- **INCLUDE set** = `{WIKI_DIR, USAGE_INSTRUCTIONS, _load_index, system_prompt_block}`
- **Retained / EXCLUDE set** = `{tool_schema, make_tool_handler}`

The move is verbatim: each moved function body and each moved constant's RHS is
byte-identical to the pre-move working-tree `wiki.py`. `wiki.py` gains only a
one-way re-export shim (`from grounding import ...`); `grounding.py` never
imports `wiki`.

## build_system_instruction (bot.py) — EXCLUDE

`build_system_instruction` (bot.py ~line 443) is **EXCLUDED** and left in
bot.py. Its *body* is Pipecat/LLM/network-free and deterministic given inputs,
but body purity alone does not make it includable. The disqualifiers are:

1. **Eager top-level import coupling in bot.py**: bot.py performs top-level
   `import anthropic` (~line 9) and top-level `pipecat.*` imports (~lines
   13–35). Pulling the function into the Pipecat-free `grounding.py` would drag
   that eager import surface along (or require re-homing it), defeating the
   Pipecat-free guarantee.
2. **Entanglement with six bot.py module constants + `wiki.system_prompt_block()`**:
   the body reads `BASE_INSTRUCTION`, `WIKI_TAGLINE`, `WIKI_ENABLED`,
   `STUDY_BASE_INSTRUCTION`, `BREVITY_REMINDER`, `STUDY_REMINDER` and calls
   `wiki.system_prompt_block()`. Moving it into `grounding.py` would require
   moving those six constants too and would introduce a `grounding → wiki`
   dependency (via `wiki.system_prompt_block`), creating exactly the kind of
   cycle the one-way shim forbids.

One-line disqualifier: *bot.py's eager top-level `anthropic` + `pipecat` imports
plus the six-constant / `wiki.system_prompt_block()` entanglement — not any
in-body Pipecat/LLM call, of which there are none.*
