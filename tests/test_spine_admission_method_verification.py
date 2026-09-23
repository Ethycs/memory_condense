from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import Turn
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import RawSummaryFragment, batch_messages
from tools import admit_spine_corpus as admission
from tools import repair_spine_summary_budget as repair
from tools import verify_spine_admission_method as verification
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes


class Client:
    max_retries = 0

    def __init__(self, content):
        self.content = content
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        assert kwargs["max_retries"] == 0
        return self

    def create(self, **kwargs):
        return SimpleNamespace(id="fixture", model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20))


def record(content, **kwargs):
    runtime = FastCompletionRuntime(client=Client(content), **kwargs)
    try:
        runtime.run()
    finally:
        runtime.close()


def fixture(tmp_path, monkeypatch, *, compact=False, complete=True):
    root = tmp_path / ("compacted" if compact else "original")
    destination = root / "offset-000"
    fragments = []
    for i, role in enumerate(("user", "assistant")):
        turn = Turn(turn_id=str(i), source_id="source", role=role,
            text="RAW_CANARY_MUST_NOT_ENTER_QWEN", created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
        fragments.append(RawSummaryFragment(RawSectionSpan.from_turn(turn), turn.text))
    request, _ = publish_sealed_json(root / "request.json", {
        "messages": batch_messages(fragments), "model": "codex_sdk/gpt-5.6-terra",
        "raw_spans": [f.span.identity_payload() for f in fragments]})
    corpus, _ = publish_sealed_json(root / "preflight.json", {"fixture": True})
    execution, _ = publish_sealed_json(destination / "execution-prefix-0001.json", {"corpus_preflight_sha256": corpus.sha256})
    namespace = {"request_count": 1 if complete else 2, "atom_count": 2,
        "span_population_sha256": identity_sha256([f.span.receipt_sha256 for f in fragments])}
    prepare = lambda *args: (corpus, namespace, [request], execution)
    monkeypatch.setattr(admission, "prepare", prepare)
    monkeypatch.setattr(verification, "prepare", prepare)
    monkeypatch.setattr(repair, "corpus_prepare", prepare)
    body = {"atoms": [{"label": "T0", "summary": "User asks for a plan.", "support": ["nonverbatim diagnostic"]},
        {"label": "T1", "summary": "Assistant suggests " + ("several tasks " * 100 if compact else "a plan."), "support": []}]}
    response = json.dumps(body)
    record(response, checkpoint_dir=destination / "raw-checkpoints" / request.sha256,
        prompt_population=[request.payload["messages"]], model=request.payload["model"],
        max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
        benchmark_provenance={"raw_request_sha256": request.sha256})
    repair_root = None
    if compact:
        repair_root = root / "summary-repair"
        audit, _ = publish_sealed_json(root / "audit.json", {"execution_preflight_sha256": execution.sha256,
            "failures": [{"request_sha256": request.sha256, "invalid_summaries": [body["atoms"][1]]}]})
        repair.prepare(repair_root, root, audit.path, 0, 1)
        preflight = read_sealed_json(repair_root / "preflight.json")
        assert "RAW_CANARY_MUST_NOT_ENTER_QWEN" not in str(preflight.payload["messages"])
        record(json.dumps({"summaries": [{"label": "S0", "summary": "Assistant suggests several tasks."}]}),
            checkpoint_dir=repair_root / "checkpoints", prompt_population=[preflight.payload["messages"]],
            model=repair.MODEL, max_prompt_tokens=7000, max_new_tokens=2048, max_concurrency=1, retries=0,
            request_options={"temperature": 0, "extra_body": {"enable_thinking": False}},
            benchmark_provenance={"preflight_sha256": preflight.sha256})
        repair.run(repair_root, False)
    admission.admit(root, 0, 1, repair_root)
    return destination / "source-bound-atoms-prefix-0001.json", repair_root


def test_both_conditional_branches_replay_exactly_without_mutating_atoms(tmp_path, monkeypatch):
    methods = []
    for compact in (False, True):
        path, repair_root = fixture(tmp_path, monkeypatch, compact=compact)
        before = path.read_bytes()
        result = verification.verify(path, repair_root)
        assert path.read_bytes() == before
        assert result.payload["budget_compacted_atoms"] == int(compact)
        assert result.payload["new_provider_calls"] == 0
        methods.append(verification.load_verified_method(path, read_sealed_json(path))[0])
    assert methods[0] == methods[1]


def test_resealed_atom_change_is_rejected_by_completion_replay(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    changed = deepcopy(read_sealed_json(path).payload)
    changed["atoms"][0]["summary"] = "Unsupported replacement."
    raw = canonical_json_bytes(changed)
    path.write_bytes(raw)
    import hashlib
    path.with_name(path.name + ".sha256").write_text(hashlib.sha256(raw).hexdigest() + "  " + path.name + "\n", encoding="ascii")
    with pytest.raises(ValueError):
        verification.verify(path)


def test_partial_verification_cannot_supply_a_full_memory_method(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch, complete=False)
    result = verification.verify(path)
    assert not result.payload["complete_namespace"]
    with pytest.raises(ValueError, match="partial memory"):
        verification.load_verified_method(path, read_sealed_json(path))


def test_rule_changes_and_wrong_repair_root_are_not_normalized_away(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    policy = read_sealed_json(path.parent / "source-binding-policy-v3-prefix-0001.json").payload
    for change in ({"summary_use": "answer evidence"}, {"summary_entailment_verified": True},
                   {"summary_text_changes_allowed": True}, {"implementation": {}},
                   {"format": "memory-condense-spine-source-binding-policy-v4"}):
        with pytest.raises(ValueError, match="unrecognized"):
            verification.conditional_method({**policy, **change})
    with pytest.raises(ValueError, match="exactly the summary-repair root"):
        verification.verify(path, tmp_path / "unbound")
