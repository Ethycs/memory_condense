from pathlib import Path

import pytest

from tools import run_spine_as_of_full100 as runner
from tools import spine_transport_lineage_v3 as lineage_module
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tests.test_spine_session_transport_stage import reseal


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "remaining"
    monkeypatch.setattr(runner, "BULK", root)
    bindings, rows = [], []
    lineage = {"additional_raw_attempts": 6, "preserved_completed_raw_responses": 403}
    monkeypatch.setattr(lineage_module, "verify_transport_lineage", lambda *args: dict(lineage))
    for offset, count in ((60, 792), (70, 838), (80, 830), (90, 868)):
        corpus = tmp_path / ("recovery/corpus" if offset == 60 else "original")
        binding = {"offset": offset, "request_count": count, "corpus_root": str(corpus),
            "execution_preflight_sha256": f"execution-{offset}", "maximum_new_calls": 389 if offset == 60 else count}
        raw, _ = publish_sealed_json(corpus / f"offset-{offset:03d}" / f"atoms-prefix-{count:04d}.json", {
            "execution_preflight_sha256": binding["execution_preflight_sha256"], "batch_validation_shas": ["fixture"] * count})
        row = {**binding, "raw_completion_sha256": raw.sha256}
        if offset == 60:
            transport, _ = publish_sealed_json(corpus.parent / "execution-complete.json", {"transport_lineage": lineage})
            row["transport_execution_complete_sha256"] = transport.sha256
        bindings.append(binding)
        rows.append(row)
    preflight, _ = publish_sealed_json(root / "preflight.json", {"bindings": bindings})
    monkeypatch.setattr(runner, "BULK_SHA", preflight.sha256)
    completion, _ = publish_sealed_json(root / "complete.json", {
        "preflight_sha256": preflight.sha256, "maximum_new_raw_calls": 2925,
        "additional_original_unknown_attempts": 6, "completed_namespaces": rows})
    return root, completion, lineage


def test_completed_recovery_and_untouched_namespaces_can_satisfy_timing_dependency(tmp_path, monkeypatch):
    _, completion, _ = fixture(tmp_path, monkeypatch)
    assert runner.require_bulk_complete().sha256 == completion.sha256


@pytest.mark.parametrize("change", ["changed_root", "missing_raw_batch", "lost_original_attempt"])
def test_partial_or_misattributed_completion_cannot_release_timing(tmp_path, monkeypatch, change):
    _, complete, lineage = fixture(tmp_path, monkeypatch)
    payload = complete.payload
    row = payload["completed_namespaces"][0]
    if change == "changed_root":
        row["corpus_root"] = str(tmp_path / "other")
    elif change == "missing_raw_batch":
        raw = read_sealed_json(Path(row["corpus_root"]) / "offset-060/atoms-prefix-0792.json")
        raw.payload["batch_validation_shas"].pop()
        changed = reseal(raw.path, raw.payload)
        row["raw_completion_sha256"] = changed.sha256
    else:
        lineage["additional_raw_attempts"] = 5
        transport = read_sealed_json(Path(row["corpus_root"]).parent / "execution-complete.json")
        changed = reseal(transport.path, {"transport_lineage": lineage})
        row["transport_execution_complete_sha256"] = changed.sha256
    reseal(complete.path, payload)
    with pytest.raises(ValueError):
        runner.require_bulk_complete()


