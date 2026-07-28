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
