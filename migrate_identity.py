"""One-time, idempotent identity migration + ledger backfill.

Pure helpers are unit-tested; the ``__main__`` block (added in Task 14) runs them
against the real ~/.voice-tutor and vault dirs, archiving originals first.
"""

import json
import re
import shutil
from pathlib import Path

DEFAULT_USER_ID = "matt"

# Matches every legacy analysis filename generation: date-only, date+timestamp,
# date+shortid — all share the "session-analysis-" prefix and ".md" suffix.
_ANALYSIS_NAME_RE = re.compile(r"^session-analysis-.*\.md$")


def backfill_ledger_user_id(lines: list[str], default_user_id: str = DEFAULT_USER_ID) -> list[str]:
    """Return ledger lines with ``user_id`` added to any JSON row lacking one.

    Idempotent (rows already carrying user_id are unchanged); non-JSON lines pass
    through verbatim; no other field is touched; line count is preserved.
    """
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        try:
            row = json.loads(stripped)
        except Exception:
            out.append(line)
            continue
        if isinstance(row, dict) and "user_id" not in row:
            row = {"user_id": default_user_id, **row}
            out.append(json.dumps(row))
        else:
            out.append(line if not isinstance(row, dict) else json.dumps(row))
    return out


def plan_moves(root: Path, user_id: str = DEFAULT_USER_ID) -> list[tuple[Path, Path]]:
    """Map existing flat files under ``root`` (``~/.voice-tutor``) to their
    ``<user_id>/`` destinations for documents/artifacts/transcripts, and the
    singleton profile.md/memory.md to profiles/<user_id>.md, memory/<user_id>.md.

    Pure; iterdir() is non-recursive so an already-nested ``<user_id>/`` dir is
    never itself re-planned -> idempotent.
    """
    uid = Path(user_id).name
    moves: list[tuple[Path, Path]] = []
    # documents/*, artifacts/*, transcripts/* -> <sub>/<uid>/ . is_file() skips
    # the <uid>/ subdir on a re-run, so this is idempotent.
    for sub in ("documents", "artifacts", "transcripts"):
        d = root / sub
        if d.exists():
            for p in d.iterdir():
                if p.is_file():
                    moves.append((p, d / uid / p.name))
    if (root / "profile.md").exists():
        moves.append((root / "profile.md", root / "profiles" / f"{uid}.md"))
    if (root / "memory.md").exists():
        moves.append((root / "memory.md", root / "memory" / f"{uid}.md"))
    return moves


def plan_analysis_moves(analyses_dir: Path, user_id: str = DEFAULT_USER_ID) -> list[tuple[Path, Path]]:
    """Move ONLY analysis files into <user_id>/, never README.md, _archive/, or any
    subdirectory. iterdir() is non-recursive, so files already under <uid>/ (or in
    _archive/) are never seen -> idempotent and safe."""
    uid = Path(user_id).name
    moves: list[tuple[Path, Path]] = []
    if not analyses_dir.exists():
        return moves
    for p in analyses_dir.iterdir():
        if p.is_file() and _ANALYSIS_NAME_RE.match(p.name):
            moves.append((p, analyses_dir / uid / p.name))
    return moves


def run_moves(moves: list[tuple[Path, Path]]) -> int:
    """Execute planned moves (dest parent created; skip if dest already exists)."""
    done = 0
    for src, dst in moves:
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    return done


if __name__ == "__main__":
    # Runs the real, one-time identity migration against ~/.voice-tutor and the
    # vault. Archive-first (copy, never delete) and idempotent -- safe to
    # re-run; a second run reports 0 backfilled / 0 moved / 0 moved.
    #
    # NOT executed as part of Task 14 -- Task 15 runs this against real data
    # with Matt present.
    import datetime

    VOICE_TUTOR_DIR = Path.home() / ".voice-tutor"
    SESSION_ANALYSES_DIR = (
        Path.home() / "second-brain" / "products" / "voice-tutor" / "session-analyses"
    )
    SESSION_LOG_JSONL_PATH = (
        Path.home()
        / "second-brain"
        / "products"
        / "voice-tutor"
        / "validation"
        / "session-log.jsonl"
    )

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    def _archive_copy(src: Path, archive_root: Path) -> None:
        """Copy ``src`` (file or dir) into ``archive_root/<ts>/<src.name>``.

        Copy-only -- never touches or removes the original. A pre-existing
        ``_archive/`` inside ``src`` (e.g. session-analyses/_archive/) is
        excluded so archives don't nest inside archives on repeated runs.
        """
        if not src.exists():
            return
        dest = archive_root / ts / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(
                src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("_archive")
            )
        else:
            shutil.copy2(src, dest)

    print(f"=== Voice Tutor identity migration ({ts}) ===")

    # 1. Archive first (copy, never delete).
    vt_archive = VOICE_TUTOR_DIR / "_archive"
    for name in ("documents", "artifacts", "transcripts", "profile.md", "memory.md"):
        _archive_copy(VOICE_TUTOR_DIR / name, vt_archive)

    vault_archive = SESSION_ANALYSES_DIR.parent / "_archive"
    _archive_copy(SESSION_LOG_JSONL_PATH, vault_archive)
    _archive_copy(SESSION_ANALYSES_DIR, vault_archive)
    print(f"Archived originals into {vt_archive / ts} and {vault_archive / ts}.")

    # 2. Backfill the ledger.
    rows_backfilled = 0
    if SESSION_LOG_JSONL_PATH.exists():
        lines = SESSION_LOG_JSONL_PATH.read_text().splitlines(keepends=True)
        out_lines = backfill_ledger_user_id(lines)
        rows_backfilled = sum(
            1
            for before, after in zip(lines, out_lines)
            if before.rstrip("\n") != after.rstrip("\n")
        )
        # backfill_ledger_user_id strips the trailing newline off any row it
        # touches (passthrough lines keep theirs) -- re-append exactly one \n
        # per line on write-back so rows never collide onto one physical line.
        if out_lines:
            SESSION_LOG_JSONL_PATH.write_text(
                "\n".join(line.rstrip("\n") for line in out_lines) + "\n"
            )
    print(f"Backfilled {rows_backfilled} ledger row(s).")

    # 3. Move files.
    files_moved = run_moves(plan_moves(VOICE_TUTOR_DIR))
    analysis_files_moved = run_moves(plan_analysis_moves(SESSION_ANALYSES_DIR))

    # 4. Summary. Idempotent: a second run reports 0/0/0.
    print(f"Moved {files_moved} file(s) under ~/.voice-tutor into <user_id>/ subdirs.")
    print(f"Moved {analysis_files_moved} session-analysis file(s) into <user_id>/ subdir.")
