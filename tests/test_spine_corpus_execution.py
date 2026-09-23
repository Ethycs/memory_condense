from datetime import datetime, timezone
import json
from types import SimpleNamespace

from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import RawSummaryFragment, batch_messages
from tools import execute_spine_corpus as execution
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def test_invalid_summary_is_retained_as_diagnostic_and_excluded_from_atoms(tmp_path, monkeypatch):
    requests = []
    for i in range(2):
        turn = Turn(turn_id=str(i), source_id="source", role="user", text=f"distinct raw evidence {i}",
                    created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
        fragment = RawSummaryFragment(RawSectionSpan.from_turn(turn), turn.text)
        request, _ = publish_sealed_json(tmp_path / f"request-{i}.json", {
            "batch_index": i, "model": "codex_sdk/gpt-5.6-terra", "messages": batch_messages([fragment]),
            "raw_spans": [fragment.span.identity_payload()]})
        assert execution.fragments_from_request(request.payload) == (fragment,)
        requests.append(request)
    manifest, _ = publish_sealed_json(tmp_path / "manifest.json", {"gateway": "https://unused.invalid"})
    plan, _ = publish_sealed_json(tmp_path / "execution.json", {"implementation_sha256": "a" * 64})
    monkeypatch.setattr(execution, "prepare", lambda *args: (
        manifest, {"request_count": 2, "atom_count": 2}, requests, plan))
    def fake_run(**kwargs):
        # Use the actual runtime to inspect the intended prompt; provider client
        # construction is never called by this deterministic test completion.
        runtime = kwargs["runtime_factory"](None)
        runtime.close()
        content = json.dumps({"atoms": [{"label": "T0", "summary": "A user assertion.",
                             "support": ["distinct raw evidence 0"]}]})
        return SimpleNamespace(logical_completions=[content]), 0, 1, 0.0
    monkeypatch.setattr(execution, "_run_exactly_authorized", fake_run)
    execution.execute(tmp_path, 0, 2, False)
    artifact = read_sealed_json(tmp_path / "offset-000/atoms-prefix-0002.json")
    assert artifact.payload["status"] == "partial_or_invalid"
    assert len(artifact.payload["atoms"]) == 1
    assert artifact.payload["invalid_request_shas"] == [requests[1].sha256]
    assert not artifact.payload["hierarchy_constructed"] and not artifact.payload["target_gate_passed"]
    execution.execute(tmp_path, 0, 2, False)
    assert read_sealed_json(artifact.path).sha256 == artifact.sha256
