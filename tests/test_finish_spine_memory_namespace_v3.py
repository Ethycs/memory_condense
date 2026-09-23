from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from tools import finish_spine_memory_namespace_v3 as worker
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json, SealedArtifactError


def fixture(tmp_path, monkeypatch, *, invalid=False, schema=False, raw_ready=True, wrong_admission=False):
    job = tmp_path / "job"
    campaign, raw_campaign, corpus = tmp_path / "campaign", tmp_path / "raw-campaign", tmp_path / "corpus"
    paths = {name: tmp_path / name for name in ("audit", "compact", "finish", "recovery", "leaves",
        "semantic", "users", "facets", "evaluation")}
    protocol = SimpleNamespace(sha256="protocol", payload={"source_admission_method_sha256": "admission-method",
        "leaf_policy_sha256": "leaf-method", "facet_policy_sha256": "facet-method"})
    raw_plan = SimpleNamespace(sha256="raw-plan")
    binding = {"offset": 80, "request_count": 830, "corpus_root": str(corpus),
        "execution_preflight_sha256": "raw-execution", "raw_token_proxy": 1_040_000}
    monkeypatch.setattr(worker, "CAMPAIGN", campaign)
    monkeypatch.setattr(worker, "RAW_CAMPAIGN", raw_campaign)
    monkeypatch.setattr(worker, "inputs", lambda offset: (protocol, raw_plan, binding, corpus, paths))
    if raw_ready:
        raw, _ = publish_sealed_json(corpus / "offset-080/atoms-prefix-0830.json", {
            "execution_preflight_sha256": "raw-execution", "batch_validation_shas": ["fixture"] * 830})
        publish_sealed_json(raw_campaign / "completed-offset-080.json", {**binding, "raw_completion_sha256": raw.sha256})
    calls = []
    def command(module, *args):
        calls.append((module, args))
        assert not (module == "tools.evaluate_spine_semantic_seeds" and args[0] in ("run", "judge"))
        if module == "tools.audit_spine_source_admission_v4":
            failures = [{"unresolved_schema_failure": schema, "invalid_summaries": ["oversized"]}] if invalid or schema else []
            publish_sealed_json(paths["audit"] / "audit.json", {"failures": failures})
        elif module == "tools.finish_spine_compaction_batches" and args[0] == "run":
            publish_sealed_json(paths["finish"] / "complete.json", {"rows": [{"invalid_slots": [0] if invalid else []}]})
        elif module == "tools.admit_spine_corpus_v7":
            publish_sealed_json(corpus / "offset-080/source-bound-atoms-prefix-0830.json", {"fixture": True})
        elif module == "tools.verify_spine_admission_method_v11":
            publish_sealed_json(corpus / "offset-080/conditional-method-v11-prefix-0830.json", {
                "method_sha256": "changed" if wrong_admission else "admission-method"})
        elif module == "tools.compile_spine_leaf_projection":
            assert args[-2:] == ("--max-new-calls", 64)
            publish_sealed_json(paths["leaves"] / "hierarchy.json", {"fixture": True})
        elif module == "tools.compile_spine_semantic_index_v2":
            publish_sealed_json(paths["semantic"] / "index.json", {"hierarchy_compilation_policy_sha256": "leaf-method"})
        elif module == "tools.evaluate_spine_semantic_seeds":
            assert args[0] == "prepare" and "--enable-provider" not in args
            manifest = read_sealed_json(paths["semantic"] / "index.json")
            publish_sealed_json(paths["evaluation"] / "preflight.json", {"shard_offset": 80,
                "calls": [None] * 50, "facets_sha256": "facets", "index_manifest_sha256": manifest.sha256,
                "raw_token_proxy": 1_040_000})
    monkeypatch.setattr(worker, "command", command)
    monkeypatch.setattr(worker.evaluation_runner.evaluation, "load_preflight", lambda root: read_sealed_json(root / "preflight.json"))
    monkeypatch.setattr(worker.evaluation_runner.evaluation, "recorded", lambda *args: [])
    monkeypatch.setattr(worker.evaluation_runner.reporting, "load_index", lambda root: (read_sealed_json(root / "index.json"), None))
    monkeypatch.setattr(worker.evaluation_runner.reporting, "load_facet_policy", lambda *args: ("facet-method", "facet-verification"))
    worker.prepare(job, 80)
    return job, campaign, paths, calls


@pytest.mark.parametrize("invalid", [False, True])
def test_complete_memory_prepares_fifty_requests_after_all_compilers(tmp_path, monkeypatch, invalid):
    job, campaign, _, calls = fixture(tmp_path, monkeypatch, invalid=invalid)
    worker.run(job, 80, True)
    modules = [module for module, _ in calls]
    assert modules[-4:] == ["tools.compile_spine_semantic_index_v2", "tools.compile_spine_user_addresses",
        "tools.compile_spine_facet_addresses", "tools.evaluate_spine_semantic_seeds"]
    assert modules.index("tools.verify_spine_admission_method_v11") < modules.index("tools.compile_spine_leaf_projection")
    if invalid:
        recovery = [args for module, args in calls if module == "tools.recover_spine_summary_budget_v2" and args[0] == "run"]
        assert len(recovery) == 1 and recovery[0][-2:] == ("--max-new-calls", 2)
    prepared = read_sealed_json(campaign / "prepared/offset-080.json")
    assert prepared.payload["answer_call_cap"] == 50
    complete = read_sealed_json(job / "complete.json")
    assert complete.payload["answer_calls_sent"] == complete.payload["judge_calls_sent"] == 0
    assert complete.payload["full100_target_passed"] is False
    before = list(calls)
    with pytest.raises(FileExistsError):
        worker.run(job, 80, True)
    assert calls == before


def test_partial_raw_population_cannot_start_compilation(tmp_path, monkeypatch):
    job, _, _, calls = fixture(tmp_path, monkeypatch, raw_ready=False)
    with pytest.raises(SealedArtifactError):
        worker.run(job, 80, True)
    assert not calls and not (job / "execution.reserved").exists()


def test_schema_failure_stops_before_qwen_compaction(tmp_path, monkeypatch):
    job, campaign, _, calls = fixture(tmp_path, monkeypatch, schema=True)
    with pytest.raises(ValueError, match="schema failures"):
        worker.run(job, 80, True)
    assert [module for module, _ in calls] == ["tools.audit_spine_source_admission_v4"]
    assert read_sealed_json(job / "failure.json").payload["phase"] == "audit"
    assert not (campaign / "prepared/offset-080.json").exists()


def test_incompatible_admission_stops_before_attention_or_answer_preparation(tmp_path, monkeypatch):
    job, _, _, calls = fixture(tmp_path, monkeypatch, wrong_admission=True)
    with pytest.raises(ValueError, match="existing memories"):
        worker.run(job, 80, True)
    assert not any(module.startswith("tools.compile_spine") for module, _ in calls)


def test_failed_child_preserves_reservation_and_stops_following_work(tmp_path, monkeypatch):
    job, campaign, _, calls = fixture(tmp_path, monkeypatch)
    original = worker.command
    def fail(module, *args):
        if module == "tools.compile_spine_leaf_projection":
            raise subprocess.CalledProcessError(1, ["python", "-m", module])
        original(module, *args)
    monkeypatch.setattr(worker, "command", fail)
    with pytest.raises(subprocess.CalledProcessError):
        worker.run(job, 80, True)
    assert read_sealed_json(job / "failure.json").payload["automatic_retry_performed"] is False
    assert (job / "execution.reserved").exists()
    assert not (campaign / "prepared/offset-080.json").exists()
    assert not any(module == "tools.compile_spine_semantic_index_v2" for module, _ in calls)



def test_successor_cannot_recompile_the_already_admitted_eighth_memory():
    with pytest.raises(ValueError, match="two remaining"):
        worker.inputs(70)

