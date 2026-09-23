from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tools import run_spine_semantic_seed_full100_v2 as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json, SealedArtifactError


def setup_campaign(tmp_path, monkeypatch, *, namespaces=10, failed_offset=None, omitted_offset=None):
    evaluation = runner.evaluation
    monkeypatch.setattr(runner, "PROTOCOL_IMPLEMENTATION", ())
    questions = [{"ordinal": i, "question_id": f"question-{i}", "retrieval_query": "Which adapter?",
        "prompt_question": "Which adapter?"} for i in range(100)]
    probes_path = tmp_path / "probes.json"
    probes, _ = publish_sealed_json(probes_path, {"questions": questions})
    monkeypatch.setattr(evaluation, "PROBES", probes_path)
    monkeypatch.setattr(evaluation, "PROBE_SHA", probes.sha256)
    protocol, _ = publish_sealed_json(tmp_path / "protocol.json", {
        "required_offsets": list(runner.OFFSETS), "question_count": 100, "answer_call_cap": 500,
        "maximum_logical_judgments": 200, "memory_arms": list(evaluation.MEMORY_ARMS),
        "routes": evaluation.ROUTES, "reader_policies": evaluation.reader_policies(), "seed_policy": evaluation.SEED_POLICY,
        "latency_ratio_limit": 1.10, "accuracy_threshold": .95, "all_answers_sealed_before_judging": True,
        "all_requests_frozen_before_first_answer": True, "cached_answers": False,
        "cached_query_vectors": False, "implementation": {}})
    targets = []
    for offset in runner.OFFSETS[:namespaces]:
        target = tmp_path / f"memory-{offset:03d}"
        calls = []
        for i in range(10):
            question = questions[offset + i]
            for arm in evaluation.call_arm_order(i):
                policy = "base" if arm == "short_api" else arm.removesuffix("_api")
                hydration = None if arm == "short_api" else SimpleNamespace(render_context=lambda: "EXACT_RAW")
                messages = evaluation.answer_messages(question, hydration, policy=policy)
                calls.append({"call_index": len(calls), "arm": arm, "question": question,
                    "messages": messages, "messages_sha256": identity_sha256(messages)})
        preflight, _ = publish_sealed_json(target / "preflight.json", {
            "shard_offset": offset, "raw_token_proxy": 1_040_000, "calls": calls})
        publish_sealed_json(tmp_path / "prepared" / f"offset-{offset:03d}.json", {
            "protocol_sha256": protocol.sha256, "offset": offset, "root": str(target),
            "preflight_sha256": preflight.sha256, "raw_token_proxy": 1_040_000,
            "answer_call_cap": 50, "maximum_logical_judgments": 20})
        targets.append(target)
    monkeypatch.setattr(evaluation, "load_preflight", lambda root: read_sealed_json(root / "preflight.json"))
    events = []
    monkeypatch.setattr(runner, "require_idle", lambda: events.append("idle"))
    monkeypatch.setattr(runner, "require_bulk_complete", lambda: SimpleNamespace(sha256="bulk-complete"))
    monkeypatch.setattr(runner, "require_readiness", lambda path: SimpleNamespace(sha256="readiness"))

    def answer(root, max_calls):
        preflight = evaluation.load_preflight(root)
        offset = preflight.payload["shard_offset"]
        assert max_calls == 50
        events.append(("answer", offset))
        if offset == failed_offset:
            (root / "journal").mkdir()
            (root / "journal/000.reserved").write_text(preflight.sha256, encoding="utf-8")
            raise TimeoutError("fixture transport failure")
        calls = preflight.payload["calls"][:-1] if offset == omitted_offset else preflight.payload["calls"]
        for call in calls:
            prefix = root / "journal" / f'{call["call_index"]:03d}'
            request, _ = publish_sealed_json(prefix.with_suffix(".request.json"), {
                "preflight_sha256": preflight.sha256, "call": call})
            publish_sealed_json(prefix.with_suffix(".response.json"), {"request_sha256": request.sha256,
                "messages": call["messages"], "measurement": {"prediction": "Adapter",
                    "prediction_sha256": quote_sha256("Adapter"), "messages_sha256": call["messages_sha256"]}})
        observations = evaluation.recorded(root, preflight)
        publish_sealed_json(root / "answers.json", {"preflight_sha256": preflight.sha256,
            "rows": evaluation.answer_rows(observations), "gold_loaded": False})

    def judge(root, enable):
        assert enable
        population = read_sealed_json(tmp_path / "answer-population.json")
        assert population.payload["answer_count"] == 500
        assert len(population.payload["bindings"]) == 10
        assert sum(len(evaluation.recorded(p, evaluation.load_preflight(p))) for p in targets) == 500
        events.append(("judge", evaluation.load_preflight(root).payload["shard_offset"]))

    def report(roots, output):
        assert len([e for e in events if isinstance(e, tuple) and e[0] == "judge"]) == 10
        publish_sealed_json(output / "joint-full100.json", {"gates": {
            arm: {"joint_gate_passed": False} for arm in evaluation.MEMORY_ARMS}})

    monkeypatch.setattr(evaluation, "run", answer)
    monkeypatch.setattr(evaluation, "judge", judge)
    monkeypatch.setattr(runner.reporting, "report", report)
    return targets, events


def test_all_500_answers_are_authenticated_before_first_judge(tmp_path, monkeypatch):
    _, events = setup_campaign(tmp_path, monkeypatch)
    runner.prepare(tmp_path)
    runner.run(tmp_path, Path("readiness.json"), True)
    assert events == ["idle", *[("answer", n) for n in runner.OFFSETS], *[("judge", n) for n in runner.OFFSETS]]
    assert read_sealed_json(tmp_path / "complete.json").payload["target_gate_passed"] is False
    with pytest.raises(ValueError, match="another release"):
        runner.run(tmp_path, Path("readiness.json"), True)


def test_six_memories_cannot_release_a_full100_campaign(tmp_path, monkeypatch):
    _, events = setup_campaign(tmp_path, monkeypatch, namespaces=6)
    with pytest.raises(SealedArtifactError, match="offset-060.json"):
        runner.prepare(tmp_path)
    assert not events and not (tmp_path / "runner-plan.json").exists()


def test_mid_campaign_failure_preserves_reservation_without_judging(tmp_path, monkeypatch):
    targets, events = setup_campaign(tmp_path, monkeypatch, failed_offset=60)
    runner.prepare(tmp_path)
    with pytest.raises(TimeoutError):
        runner.run(tmp_path, Path("readiness.json"), True)
    assert (targets[6] / "journal/000.reserved").exists()
    assert not any(isinstance(e, tuple) and e[0] == "judge" for e in events)
    assert read_sealed_json(tmp_path / "failure.json").payload["automatic_retry_performed"] is False
    with pytest.raises(ValueError, match="unacknowledged"):
        runner.run(tmp_path, Path("readiness.json"), True)


def test_one_missing_answer_blocks_all_judging(tmp_path, monkeypatch):
    _, events = setup_campaign(tmp_path, monkeypatch, omitted_offset=90)
    runner.prepare(tmp_path)
    with pytest.raises(ValueError, match="all 500"):
        runner.run(tmp_path, Path("readiness.json"), True)
    assert not any(isinstance(e, tuple) and e[0] == "judge" for e in events)
    assert not (tmp_path / "answer-population.json").exists()


@pytest.mark.parametrize("blocker", ["busy", "bulk", "readiness"])
def test_unready_campaign_sends_no_answers(tmp_path, monkeypatch, blocker):
    _, events = setup_campaign(tmp_path, monkeypatch)
    runner.prepare(tmp_path)
    def fail(*args):
        raise ValueError("fixture prerequisite failed")
    monkeypatch.setattr(runner, {"busy": "require_idle", "bulk": "require_bulk_complete",
        "readiness": "require_readiness"}[blocker], fail)
    with pytest.raises(ValueError, match="prerequisite"):
        runner.run(tmp_path, Path("readiness.json"), True)
    assert not any(isinstance(e, tuple) and e[0] == "answer" for e in events)
    assert not (tmp_path / "execution.reserved").exists()
