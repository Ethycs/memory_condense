from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import RawSummaryFragment, batch_messages
from tools import admit_spine_corpus_v2 as admission
from tools import repair_spine_summary_budget_v2 as repair
from tools import verify_spine_admission_method_v3 as verification
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes
from tests.test_spine_admission_method_verification import Client, record


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    destination = root / "offset-000"
    fragments = []
    for i in range(9):
        turn = Turn(turn_id=str(i), source_id="source", role="user",
            text=f"RAW_CANARY_MUST_NOT_ENTER_QWEN_{i}", created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
        fragments.append(RawSummaryFragment(RawSectionSpan.from_turn(turn), turn.text))
    request, _ = publish_sealed_json(root / "request.json", {
        "messages": batch_messages(fragments), "model": "codex_sdk/gpt-5.6-terra",
        "raw_spans": [f.span.identity_payload() for f in fragments]})
    corpus, _ = publish_sealed_json(root / "preflight.json", {"fixture": True})
    execution, _ = publish_sealed_json(destination / "execution-prefix-0001.json", {"corpus_preflight_sha256": corpus.sha256})
    namespace = {"request_count": 1, "atom_count": 9,
        "span_population_sha256": identity_sha256([f.span.receipt_sha256 for f in fragments])}
    prepare = lambda *args: (corpus, namespace, [request], execution)
    monkeypatch.setattr(admission, "prepare", prepare)
    monkeypatch.setattr(verification, "prepare", prepare)
    monkeypatch.setattr(repair, "corpus_prepare", prepare)
    body = {"atoms": [{"label": f"T{i}", "summary": f"User reports item {i}: " + "several tasks " * 100,
                      "support": []} for i in range(9)]}
    record(json.dumps(body), checkpoint_dir=destination / "raw-checkpoints" / request.sha256,
        prompt_population=[request.payload["messages"]], model=request.payload["model"],
        max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
        benchmark_provenance={"raw_request_sha256": request.sha256})
    audit, _ = publish_sealed_json(root / "audit.json", {"execution_preflight_sha256": execution.sha256,
        "request_count": 1, "failures": [{"request_sha256": request.sha256,
            "unresolved_schema_failure": False, "invalid_summaries": body["atoms"]}]})
    repair_root = root / "summary-repair"
    preflight = repair.prepare(repair_root, root, audit.path, 0, 1)
    return root, repair_root, preflight


def output(batch):
    return json.dumps({"summaries": [{"label": f"S{i}", "summary": f"User reports item {ordinal}."}
                      for i, ordinal in enumerate(batch["row_indices"])]})


def runtime_kwargs(root, preflight, batch):
    return dict(checkpoint_dir=root / "checkpoints" / batch["batch_sha256"],
        prompt_population=[batch["messages"]], model=repair.MODEL,
        max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
        request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
        benchmark_provenance={"summary_budget_preflight_sha256": preflight.sha256,
                              "summary_budget_batch_sha256": batch["batch_sha256"]})


def test_nine_jobs_use_two_summary_only_calls_and_replay_all_atoms(tmp_path, monkeypatch):
    root, repair_root, preflight = fixture(tmp_path, monkeypatch)
    batches = preflight.payload["batches"]
    assert [len(b["row_indices"]) for b in batches] == [8, 1]
    captured = []

    class CapturingClient(Client):
        def create(self, **kwargs):
            captured.append(kwargs["messages"])
            return super().create(**kwargs)

    clients = iter(CapturingClient(output(batch)) for batch in batches)
    monkeypatch.setattr(repair, "_completion_client", lambda *args: next(clients))
    repairs = repair.run(repair_root, True)
    assert len(captured) == 2
    assert "RAW_CANARY" not in json.dumps(captured)
    assert [r["label"] for r in repairs.payload["rows"]] == [f"T{i}" for i in range(9)]
    assert repairs.payload["successful_batch_count"] == 2
    monkeypatch.setattr(repair, "_completion_client", lambda *args: pytest.fail("replay attempted provider call"))
    assert repair.run(repair_root, False).sha256 == repairs.sha256
    admission.admit(root, 0, 1, repair_root)
    path = root / "offset-000/source-bound-atoms-prefix-0001.json"
    before = path.read_bytes()
    verified = verification.verify(path, repair_root)
    assert verified.payload["budget_compacted_atoms"] == 9
    assert verified.payload["required_compaction_batches"] == 2
    assert verified.payload["compaction_provider_attempts"] == 2
    assert verification.load_verified_method(path, read_sealed_json(path))[1] == verified.sha256
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["missing", "pending", "omitted", "wrong_label"])
def test_second_batch_must_complete_and_validate_before_publication(tmp_path, monkeypatch, failure):
    _, root, preflight = fixture(tmp_path, monkeypatch)
    first, second = preflight.payload["batches"]
    record(output(first), **runtime_kwargs(root, preflight, first))
    if failure == "pending":
        class FailingClient(Client):
            def create(self, **kwargs):
                raise RuntimeError("simulated inference timeout")
        runtime = FastCompletionRuntime(client=FailingClient(""), **runtime_kwargs(root, preflight, second))
        try:
            with pytest.raises(RuntimeError, match="simulated inference timeout"):
                runtime.run()
        finally:
            runtime.close()
    elif failure in ("omitted", "wrong_label"):
        text = '{"summaries": []}' if failure == "omitted" else '{"summaries": [{"label": "S1", "summary": "User reports an item."}]}'
        record(text, **runtime_kwargs(root, preflight, second))
    before = {p: p.read_bytes() for p in (root / "checkpoints").rglob("*.json")}
    monkeypatch.setattr(repair, "_completion_client", lambda *args: pytest.fail("unexpected provider call"))
    with pytest.raises((ValueError, RuntimeError)):
        repair.run(root, False)
    assert not (root / "repairs.json").exists()
    assert before == {p: p.read_bytes() for p in (root / "checkpoints").rglob("*.json")}


def test_resealed_foreign_summary_or_duplicate_job_cannot_reach_qwen(tmp_path, monkeypatch):
    _, root, preflight = fixture(tmp_path, monkeypatch)
    rows = preflight.payload["rows"]
    with pytest.raises(ValueError, match="duplicate"):
        repair.batches_for(rows + [rows[0]])
    changed = deepcopy(preflight.payload)
    changed["rows"][0]["job"]["fragments"][0]["summary"] = "RAW_CANARY_MUST_NOT_ENTER_QWEN " * 100
    changed["batches"] = repair.batches_for(changed["rows"])
    raw = canonical_json_bytes(changed)
    preflight.path.write_bytes(raw)
    preflight.path.with_name(preflight.path.name + ".sha256").write_bytes(
        (hashlib.sha256(raw).hexdigest() + "  " + preflight.path.name + "\n").encode("ascii"))
    monkeypatch.setattr(repair, "_completion_client", lambda *args: pytest.fail("foreign summary reached provider"))
    with pytest.raises(ValueError):
        repair.run(root, True)
    assert not (root / "checkpoints").exists()
