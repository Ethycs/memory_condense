from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import RawSummaryFragment, batch_messages
from tools import repair_spine_batch_support as repair
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


@pytest.mark.parametrize("labels", [[0], [99]])
def test_label_repair_preserves_summary_and_rejects_foreign_support(tmp_path, monkeypatch, labels):
    turn = Turn(turn_id="turn", source_id="source", role="user", text="exact source words",
                created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    fragment = RawSummaryFragment(RawSectionSpan.from_turn(turn), turn.text)
    request, _ = publish_sealed_json(tmp_path / "request.json", {
        "messages": batch_messages([fragment]), "raw_spans": [fragment.span.identity_payload()]})
    destination = tmp_path / "offset-000/support-prefix-0001"
    publish_sealed_json(destination / "preflight.json", {
        "implementation_sha256": hashlib.sha256(Path(repair.__file__).read_bytes()).hexdigest(),
        "model": "codex_sdk/gpt-5.6-terra", "gateway": "https://unused.invalid", "corpus_preflight_sha256": "a" * 64,
        "request_limit": 1, "namespace_request_count": 2,
        "rows": [{"raw_request_path": "request.json", "raw_request_sha256": request.sha256,
            "original_response": json.dumps({"atoms": [{"label": "T0", "summary": "immutable summary",
                                                       "support": ["invented quote"]}]})}],
        "tasks": [{"request_sha256": request.sha256, "atom_index": 0, "pieces": [turn.text],
                   "messages": [{"role": "user", "content": "select a label"}]}]})
    monkeypatch.setattr(repair, "_run_exactly_authorized", lambda **kwargs: (
        SimpleNamespace(logical_completions=[json.dumps({"selected_labels": labels})]), 0, 1, 0.0))
    if labels == [99]:
        with pytest.raises(ValueError, match="labels"):
            repair.run(destination, False)
        assert not (destination / "atoms.json").exists()
    else:
        repair.run(destination, False)
        result = read_sealed_json(destination / "atoms.json")
        assert result.payload["atoms"][0]["summary"] == "immutable summary"
        assert result.payload["support_audit"][0]["support"] == [turn.text]
        assert result.payload["summary_texts_unchanged"] and not result.payload["complete_namespace"]
