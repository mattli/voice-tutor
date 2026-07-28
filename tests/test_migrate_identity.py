import json
import migrate_identity as mig


def test_backfill_adds_user_id_to_untagged_rows():
    lines = [
        json.dumps({"kind": "session", "session_id": "s1"}),
        json.dumps({"kind": "artifact", "session_id": "s1", "document_id": "d1"}),
    ]
    out = [json.loads(l) for l in mig.backfill_ledger_user_id(lines)]
    assert all(row["user_id"] == "matt" for row in out)


def test_backfill_is_idempotent():
    tagged = json.dumps({"kind": "session", "session_id": "s1", "user_id": "sarah"})
    out = mig.backfill_ledger_user_id([tagged])
    assert json.loads(out[0])["user_id"] == "sarah"  # not overwritten


def test_backfill_preserves_all_other_fields_and_line_count():
    row = {"kind": "session", "session_id": "s1", "cost_total_usd": 0.42, "turns": 7}
    out = mig.backfill_ledger_user_id([json.dumps(row)])
    assert len(out) == 1
    got = json.loads(out[0])
    assert got["cost_total_usd"] == 0.42 and got["turns"] == 7


def test_backfill_passes_through_non_json_lines():
    out = mig.backfill_ledger_user_id(["not json\n", ""])
    assert out == ["not json\n", ""]


def test_plan_moves_maps_flat_files_into_user_dir(tmp_path):
    vt = tmp_path / ".voice-tutor"
    (vt / "documents").mkdir(parents=True)
    (vt / "documents" / "d1.txt").write_text("x")
    (vt / "documents" / "d1-orig.md").write_text("x")
    (vt / "artifacts").mkdir()
    (vt / "artifacts" / "s1.md").write_text("x")
    (vt / "profile.md").write_text("p")
    (vt / "memory.md").write_text("m")

    moves = dict(mig.plan_moves(vt, user_id="matt"))
    assert moves[vt / "documents" / "d1.txt"] == vt / "documents" / "matt" / "d1.txt"
    assert moves[vt / "artifacts" / "s1.md"] == vt / "artifacts" / "matt" / "s1.md"
    assert moves[vt / "profile.md"] == vt / "profiles" / "matt.md"
    assert moves[vt / "memory.md"] == vt / "memory" / "matt.md"
    # Idempotent: already-nested files are not re-planned.
    (vt / "documents" / "matt").mkdir()
    (vt / "documents" / "matt" / "d2.txt").write_text("x")
    moves2 = dict(mig.plan_moves(vt, user_id="matt"))
    assert (vt / "documents" / "matt" / "d2.txt") not in moves2


def test_plan_analysis_moves_only_touches_analysis_files(tmp_path):
    d = tmp_path / "session-analyses"
    d.mkdir()
    # All three legacy generations (per the session-analyses README):
    gen_date = d / "session-analysis-2026-07-20.md"                  # date-only
    gen_ts   = d / "session-analysis-2026-07-25-143005.md"           # date + timestamp
    gen_sid  = d / "session-analysis-2026-07-27-143005-abcd1234.md"  # date + shortid
    for p in (gen_date, gen_ts, gen_sid):
        p.write_text("analysis")
    # Non-analysis siblings that must NOT be swept:
    (d / "README.md").write_text("readme")
    (d / "_archive").mkdir()
    (d / "_archive" / "session-analysis-2026-01-01.md").write_text("old")  # inside a subdir → untouched

    moves = dict(mig.plan_analysis_moves(d, user_id="matt"))
    assert set(moves) == {gen_date, gen_ts, gen_sid}
    assert moves[gen_sid] == d / "matt" / gen_sid.name
    assert (d / "README.md") not in moves
    assert (d / "_archive" / "session-analysis-2026-01-01.md") not in moves
    # Idempotent: files already under matt/ are not re-planned.
    (d / "matt").mkdir(exist_ok=True)
    (d / "matt" / "session-analysis-2026-08-01.md").write_text("new")
    assert (d / "matt" / "session-analysis-2026-08-01.md") not in dict(mig.plan_analysis_moves(d, "matt"))


def test_run_moves_moves_files_and_skips_existing_dest(tmp_path):
    src1 = tmp_path / "a.txt"
    src1.write_text("hello")
    dst1 = tmp_path / "sub" / "a.txt"

    src2 = tmp_path / "b.txt"
    src2.write_text("new")
    dst2 = tmp_path / "b-dest.txt"
    dst2.write_text("already here")

    done = mig.run_moves([(src1, dst1), (src2, dst2)])

    assert done == 1
    assert dst1.read_text() == "hello"
    assert not src1.exists()
    # dest already existed -> skipped, src untouched
    assert src2.exists()
    assert dst2.read_text() == "already here"


def test_ledger_write_back_preserves_one_json_object_per_line(tmp_path):
    """Guardrail 3: backfilled/re-tagged lines come back WITHOUT a trailing
    newline from backfill_ledger_user_id — the write-back must re-append
    exactly one \\n per line so rows don't collide onto one physical line."""
    ledger = tmp_path / "session-log.jsonl"
    rows = [
        json.dumps({"kind": "session", "session_id": "s1"}),
        json.dumps({"kind": "session", "session_id": "s2", "user_id": "sarah"}),
    ]
    ledger.write_text("\n".join(rows) + "\n")

    lines = ledger.read_text().splitlines(keepends=True)
    out_lines = mig.backfill_ledger_user_id(lines)
    ledger.write_text("\n".join(line.rstrip("\n") for line in out_lines) + "\n")

    written = ledger.read_text()
    physical_lines = written.splitlines()
    assert len(physical_lines) == 2
    parsed = [json.loads(l) for l in physical_lines]
    assert parsed[0]["user_id"] == "matt"
    assert parsed[1]["user_id"] == "sarah"
    # File ends with exactly one trailing newline, no blank line appended.
    assert written.endswith("\n") and not written.endswith("\n\n")
