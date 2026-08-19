from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

import tools.mem0_eval.report as report_module
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.context_stress import transcript_tokens
from memory_condense.eval.sample_identity import sample_sha256
from memory_condense.eval.schemas import UsageStats
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_REVISION,
)
from tests import test_mem0_run_shard as runner_fakes
from tools.mem0_eval.report import (
    ExpectedMem0Shard,
    FrozenMem0Population,
    Mem0ReportError,
    canonical_sha256,
    merge_mem0_shard_reports,
    save_mem0_campaign_report,
    validate_mem0_shard_report,
)
from tools.mem0_eval.run_shard import (
    ProviderCallResult,
    ShardProcessGuard,
    run_retrieval_stage,
    run_scoring_stage,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


LOCK_BYTES = b"synthetic-mem0-lock\n"
LOCK_SHA256 = hashlib.sha256(LOCK_BYTES).hexdigest()
ROOT_LOCK_BYTES = b"synthetic-root-lock\n"
ROOT_LOCK_SHA256 = hashlib.sha256(ROOT_LOCK_BYTES).hexdigest()
EXTRACTION_IDENTITY_BODY = {
    "provider": "synthetic",
    "model": "synthetic-extractor",
    "revision": "test-revision",
    "provider_retries": 0,
    "logical_call_boundary": "Memory.llm.generate_response",
    "logical_calls_per_add": 1,
    "http_attempts_certified": False,
}
EXTRACTION_IDENTITY = {
    **EXTRACTION_IDENTITY_BODY,
    "model_identity_sha256": canonical_sha256(EXTRACTION_IDENTITY_BODY),
}
EXTRACTION_IDENTITY_SHA256 = canonical_sha256(EXTRACTION_IDENTITY)
EMBEDDER_IDENTITY_BODY = {
    "provider": "huggingface",
    "model": "BAAI/bge-m3",
    "revision": DEFAULT_MODEL_REVISION,
    "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
    "dimension": 1024,
    "device": "cpu",
    "dtype": "float32",
    "execution": "local_offline",
    "network_calls_authorized": 0,
    "runtime_probe_required": True,
}
EMBEDDER_IDENTITY = {
    **EMBEDDER_IDENTITY_BODY,
    "model_identity_sha256": canonical_sha256(EMBEDDER_IDENTITY_BODY),
}
EMBEDDER_IDENTITY_SHA256 = canonical_sha256(EMBEDDER_IDENTITY)


class _SyntheticPolicy:
    sha256 = runner_fakes.SHA_2
    environment_lock_sha256 = LOCK_SHA256
    tool_implementation_sha256 = ""
    stable_config_sha256 = runner_fakes.SHA_C
    stable_payload: Mapping[str, Any] = runner_fakes.STABLE_PAYLOAD
    extraction_identity: Mapping[str, Any] = EXTRACTION_IDENTITY
    embedder_identity: Mapping[str, Any] = EMBEDDER_IDENTITY
    scoring: Mapping[str, Any] = {
        "responder_identity_sha256": "5" * 64,
        "judge_identity_sha256": "6" * 64,
        "judge_max_output_tokens": 1_024,
    }

    def recheck(self) -> None:
        return None


def _shard(offset: int):
    base = runner_fakes._shard()
    mapping = {
        f"q-{index}": f"q-{offset:03d}-{index}"
        for index in range(10)
    }
    questions = [
        question.model_copy(update={"question_id": mapping[question.question_id]})
        for question in base.parsed_sample.questions
    ]
    sample_id = f"mem0-context-stress-1000000-offset-{offset:03d}"
    sample = base.parsed_sample.model_copy(
        update={
            "sample_id": sample_id,
            "turn_source_ids": [
                value.replace("q-0", mapping["q-0"])
                if isinstance(value, str)
                else value
                for value in base.parsed_sample.turn_source_ids
            ],
            "questions": questions,
        }
    )
    raw_bundle = copy.deepcopy(base.raw_history_bundle)
    raw_bundle["question_id"] = sample_id
    for record in raw_bundle["records"]:
        old = record["source_sample_id"]
        record["source_sample_id"] = mapping[old]
        record["haystack_session_ids"] = [
            value.replace(f"{old}::", f"{mapping[old]}::", 1)
            for value in record["haystack_session_ids"]
        ]
    batches = tuple(
        replace(
            batch,
            source_sample_id=mapping[batch.source_sample_id],
            source=batch.source.replace(
                f"{batch.source_sample_id}::",
                f"{mapping[batch.source_sample_id]}::",
                1,
            ),
        )
        for batch in base.add_batches
    )
    return replace(
        base,
        sample_offset=offset,
        parsed_sample=sample,
        sample_sha256=sample_sha256(sample),
        history_sample_ids=tuple(mapping[f"q-{index}"] for index in range(10)),
        raw_history_bundle=raw_bundle,
        raw_history_bundle_sha256=canonical_sha256(raw_bundle),
        add_batches=batches,
    )


def _expected(shard) -> ExpectedMem0Shard:
    records = shard.raw_history_bundle["records"]
    sessions = sum(len(record["haystack_sessions"]) for record in records)
    turns = sum(
        len(session)
        for record in records
        for session in record["haystack_sessions"]
    )
    return ExpectedMem0Shard(
        sample_offset=shard.sample_offset,
        sample_id=shard.parsed_sample.sample_id,
        sample_sha256=shard.sample_sha256,
        num_turns=len(shard.parsed_sample.turns),
        transcript_tokens=transcript_tokens(shard.parsed_sample),
        questions=tuple(
            {
                "question_id": question.question_id,
                "question": question.question,
                "dated_question": question.dated_question,
                "gold_answer": question.answer,
                "category": question.category,
            }
            for question in shard.parsed_sample.questions
        ),
        history_sample_ids=shard.history_sample_ids,
        raw_history_bundle_sha256=shard.raw_history_bundle_sha256,
        contributor_ids_sha256=canonical_sha256(list(shard.history_sample_ids)),
        records=len(records),
        raw_sessions=sessions,
        raw_turns=turns,
        raw_pairs=shard.add_counts.raw_pairs,
        skipped_empty_pairs=shard.add_counts.skipped_empty_pairs,
        expected_adds=shard.add_counts.add_requests,
    )


def _population(shards) -> FrozenMem0Population:
    expected = {shard.sample_offset: _expected(shard) for shard in shards}
    policy = _SyntheticPolicy()
    policy.tool_implementation_sha256 = runner_fakes.tool_implementation_sha256()
    identity = runner_fakes._source_evaluation_identity()
    plan = SimpleNamespace(
        dataset_sha256=_digest("dataset"),
        split_manifest_sha256=_digest("split"),
        policy_manifest_sha256=runner_fakes.SHA_E,
        implementation_sha256=runner_fakes.SHA_F,
        environment_lock_sha256=ROOT_LOCK_SHA256,
        question_ids=frozenset(
            question_id
            for shard in shards
            for question_id in shard.question_ids
        ),
    )
    return FrozenMem0Population(
        plan=plan,
        shards=expected,
        mem0_policy_path=Path("synthetic-policy.json"),
        mem0_environment_lock_path=Path("pixi.lock"),
        mem0_policy_sha256=policy.sha256,
        mem0_environment_lock_sha256=policy.environment_lock_sha256,
        mem0_tool_implementation_sha256=policy.tool_implementation_sha256,
        source_evaluation_identity=identity,
        mem0_policy=policy,
    )


def _declare_stateless(callback):
    callback.request_token_state_receipt = lambda: {
        "contract": "stateless-request-token-state-v1",
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    return callback


def _run_shard(tmp_path: Path, shard, *, effective_sha256: str) -> Path:
    directory = tmp_path / f"shard-{shard.sample_offset:03d}"
    directory.mkdir()
    mem0_directory = directory / "mem0"
    mem0_directory.mkdir()
    lock_path = mem0_directory / "pixi.lock"
    lock_path.write_bytes(LOCK_BYTES)
    root_directory = directory / "root"
    root_directory.mkdir()
    root_lock_path = root_directory / "pixi.lock"
    root_lock_path.write_bytes(ROOT_LOCK_BYTES)
    original_runtime = runner_fakes._runtime_identity
    runner_fakes._runtime_identity = lambda: {
        **original_runtime(),
        "effective_config_sha256": effective_sha256,
    }
    def adapter_factory(state: Path):
        adapter = runner_fakes._FakeAdapter(state)
        return adapter

    try:
        retrieval_authorization = replace(
            runner_fakes._retrieval_authorization(shard),
            mem0_environment_lock_sha256=LOCK_SHA256,
            source_environment_lock_sha256=ROOT_LOCK_SHA256,
            extraction_model_identity=EXTRACTION_IDENTITY,
            extraction_model_identity_sha256=EXTRACTION_IDENTITY_SHA256,
            embedder_model_identity=EMBEDDER_IDENTITY,
            embedder_model_identity_sha256=EMBEDDER_IDENTITY_SHA256,
        )
        retrieval = run_retrieval_stage(
            shard=shard,
            authorization=retrieval_authorization,
            mem0_environment_lock_path=lock_path,
            owned_state_dir=directory / "state",
            artifact_path=directory / "retrieval.json",
            trace_path=directory / "retrieval.trace.json",
            adapter_factory=adapter_factory,
            process_guard=ShardProcessGuard(f"retrieval-{shard.sample_offset}"),
        )
    finally:
        runner_fakes._runtime_identity = original_runtime

    answer_index = 0

    @_declare_stateless
    def responder(
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        nonlocal answer_index
        del model, max_output_tokens
        result = ProviderCallResult(
            text=f"answer {answer_index}",
            usage=UsageStats(
                input_tokens=100,
                output_tokens=2,
                elapsed_s=0.01,
                calls=1,
            ),
        )
        answer_index += 1
        return result

    @_declare_stateless
    def judge(
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        del model, max_output_tokens
        return ProviderCallResult(
            text="CORRECT same answer",
            usage=UsageStats(
                input_tokens=50,
                output_tokens=3,
                elapsed_s=0.01,
                calls=1,
            ),
        )

    scoring_authorization = replace(
        runner_fakes._scoring_authorization(
            shard, retrieval.artifact_sha256
        ),
        scoring_policy_sha256=runner_fakes.SHA_2,
        mem0_environment_lock_sha256=LOCK_SHA256,
        source_environment_lock_sha256=ROOT_LOCK_SHA256,
        extraction_model_identity=EXTRACTION_IDENTITY,
        extraction_model_identity_sha256=EXTRACTION_IDENTITY_SHA256,
        embedder_model_identity=EMBEDDER_IDENTITY,
        embedder_model_identity_sha256=EMBEDDER_IDENTITY_SHA256,
    )
    scoring = run_scoring_stage(
        shard=shard,
        authorization=scoring_authorization,
        root_environment_lock_path=root_lock_path,
        retrieval_artifact_path=retrieval.artifact_path,
        retrieval_trace_path=retrieval.trace_path,
        report_path=directory / "report.json",
        scoring_trace_path=directory / "scoring.trace.json",
        responder=responder,
        judge=judge,
        process_guard=ShardProcessGuard(f"scoring-{shard.sample_offset}"),
    )
    return scoring.report_path


def _write_runner_json(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def test_native_run_shard_report_validates_and_rebuilds_prompt(tmp_path: Path) -> None:
    shard = _shard(0)
    population = _population((shard,))
    path = _run_shard(tmp_path, shard, effective_sha256=_digest("effective-0"))
    document = json.loads(path.read_text(encoding="utf-8"))

    validated = validate_mem0_shard_report(
        document,
        report_path=path,
        expected=population.shards[0],
        population=population,
    )

    assert len(validated.questions) == 10
    assert all(row["judge_correct"] for row in validated.questions)
    assert validated.runtime_identity["effective_config_sha256"] == _digest(
        "effective-0"
    )

    propagated_object_paths = (
        ("question_results", 0),
        ("question_results", 0, "responder_usage"),
        ("identity",),
        ("model_identity",),
        ("config",),
    )
    for object_path in propagated_object_paths:
        forged_report = copy.deepcopy(document)
        target: Any = forged_report
        for component in object_path:
            target = target[component]
        target["comparison_certified"] = True
        _write_runner_json(path, forged_report)
        with pytest.raises(Mem0ReportError, match="fields do not match"):
            validate_mem0_shard_report(
                forged_report,
                report_path=path,
                expected=population.shards[0],
                population=population,
            )
    _write_runner_json(path, document)

    artifact_path = path.parent / document["retrieval_artifact"]["filename"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    secret_artifact = copy.deepcopy(artifact)
    secret_artifact["identity"]["runtime_identity"]["config"][
        "Authorization"
    ] = "Bearer TOPSECRET"
    secret_artifact_without_hash = dict(secret_artifact)
    del secret_artifact_without_hash["content_sha256"]
    secret_artifact["content_sha256"] = canonical_sha256(
        secret_artifact_without_hash
    )
    secret_payload = _write_runner_json(artifact_path, secret_artifact)
    secret_report = copy.deepcopy(document)
    secret_report["retrieval_artifact"]["sha256"] = hashlib.sha256(
        secret_payload
    ).hexdigest()
    secret_report["retrieval_artifact"]["bytes"] = len(secret_payload)
    _write_runner_json(path, secret_report)
    with pytest.raises(Mem0ReportError, match="secret"):
        validate_mem0_shard_report(
            secret_report,
            report_path=path,
            expected=population.shards[0],
            population=population,
        )

    namespaced_secret_artifact = copy.deepcopy(artifact)
    namespaced_secret_artifact["identity"]["runtime_identity"]["config"][
        "azure_openai_api_key"
    ] = "TOPSECRET"
    namespaced_secret_without_hash = dict(namespaced_secret_artifact)
    del namespaced_secret_without_hash["content_sha256"]
    namespaced_secret_artifact["content_sha256"] = canonical_sha256(
        namespaced_secret_without_hash
    )
    namespaced_secret_payload = _write_runner_json(
        artifact_path, namespaced_secret_artifact
    )
    namespaced_secret_report = copy.deepcopy(document)
    namespaced_secret_report["retrieval_artifact"]["sha256"] = hashlib.sha256(
        namespaced_secret_payload
    ).hexdigest()
    namespaced_secret_report["retrieval_artifact"]["bytes"] = len(
        namespaced_secret_payload
    )
    _write_runner_json(path, namespaced_secret_report)
    with pytest.raises(Mem0ReportError, match="secret"):
        validate_mem0_shard_report(
            namespaced_secret_report,
            report_path=path,
            expected=population.shards[0],
            population=population,
        )

    mismatched_runtime_artifact = copy.deepcopy(artifact)
    mismatched_runtime_artifact["identity"]["runtime_identity"]["config"][
        "temperature"
    ] = 0.5
    mismatched_runtime_without_hash = dict(mismatched_runtime_artifact)
    del mismatched_runtime_without_hash["content_sha256"]
    mismatched_runtime_artifact["content_sha256"] = canonical_sha256(
        mismatched_runtime_without_hash
    )
    mismatched_runtime_payload = _write_runner_json(
        artifact_path, mismatched_runtime_artifact
    )
    mismatched_runtime_report = copy.deepcopy(document)
    mismatched_runtime_report["retrieval_artifact"]["sha256"] = hashlib.sha256(
        mismatched_runtime_payload
    ).hexdigest()
    mismatched_runtime_report["retrieval_artifact"]["bytes"] = len(
        mismatched_runtime_payload
    )
    _write_runner_json(path, mismatched_runtime_report)
    with pytest.raises(Mem0ReportError, match="stable payload"):
        validate_mem0_shard_report(
            mismatched_runtime_report,
            report_path=path,
            expected=population.shards[0],
            population=population,
        )

    original_artifact_payload = _write_runner_json(artifact_path, artifact)
    assert hashlib.sha256(original_artifact_payload).hexdigest() == document[
        "retrieval_artifact"
    ]["sha256"]
    _write_runner_json(path, document)

    row = artifact["retrieval_rows"][0]
    row["context"] = "self-consistent but not rendered from raw_pool"
    row["context_sha256"] = hashlib.sha256(row["context"].encode()).hexdigest()
    row["context_tokens"] = count_tokens(row["context"])
    row_without_hash = dict(row)
    del row_without_hash["retrieval_row_sha256"]
    row["retrieval_row_sha256"] = canonical_sha256(row_without_hash)
    artifact_without_hash = dict(artifact)
    del artifact_without_hash["content_sha256"]
    artifact["content_sha256"] = canonical_sha256(artifact_without_hash)
    artifact_payload = _write_runner_json(artifact_path, artifact)
    document["retrieval_artifact"]["sha256"] = hashlib.sha256(
        artifact_payload
    ).hexdigest()
    document["retrieval_artifact"]["bytes"] = len(artifact_payload)
    _write_runner_json(path, document)

    with pytest.raises(Mem0ReportError, match="independent prompt rebuild"):
        validate_mem0_shard_report(
            document,
            report_path=path,
            expected=population.shards[0],
            population=population,
        )


def test_ten_native_shards_merge_against_synthetic_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards = tuple(_shard(offset) for offset in range(0, 100, 10))
    population = _population(shards)
    paths = [
        _run_shard(
            tmp_path,
            shard,
            effective_sha256=_digest(f"effective-{shard.sample_offset}"),
        )
        for shard in shards
    ]
    monkeypatch.setattr(
        report_module,
        "reconstruct_frozen_mem0_population",
        lambda **_kwargs: population,
    )
    monkeypatch.setattr(
        report_module,
        "_assert_population_unchanged",
        lambda value: value.mem0_policy.recheck(),
    )

    merged = merge_mem0_shard_reports(
        paths,
        benchmark_file="synthetic-dataset.json",
        split_manifest="synthetic-split.json",
        policy_manifest="synthetic-source-policy.json",
        mem0_policy_manifest="synthetic-mem0-policy.json",
        mem0_environment_lock="synthetic-lock",
    )

    assert merged["num_samples"] == 10
    assert merged["num_questions"] == 100
    assert merged["judge_accuracy"] == 1.0
    assert merged["operation_totals"]["mem0_adds"] == 20
    assert merged["operation_totals"]["mem0_local_logical_wrapper_calls"] == 20
    assert merged["operation_totals"]["answer_judge_logical_wrapper_calls"] == 200
    assert merged["prompt_token_proxy_budget_compliance"] is True
    assert merged["production_binding_certified"] is False
    assert merged["runtime_model_identity_probe"]["kind"] == (
        "unavailable_injected_nonproduction"
    )
    assert merged["runtime_model_identity_probe"]["comparison_certified"] is False
    assert merged["zero_persisted_transformer_token_state_verified"] is False
    assert merged["target_status"] == "metric_passed_noncertified"
    assert merged["provenance"]["source_session_date_exposure"] == (
        "diagnostics_only_not_model_input"
    )
    assert merged["provenance"]["retrieved_created_at_exposure"] == (
        "answer_prompt_date_headings"
    )
    assert len(merged["common_question_results"]) == 100
    assert set(merged["common_question_results"][0]) == {
        "question_id",
        "predicted_answer",
        "judge_correct",
        "f1",
        "exact_match",
        "context_tokens",
        "prompt_token_proxy",
        "responder_usage",
        "judge_usage",
    }


def test_campaign_report_save_is_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "campaign.json"
    report = {"status": "complete", "value": 1}

    saved = save_mem0_campaign_report(report, output)
    before = output.read_bytes()
    assert saved == output.resolve()
    assert json.loads(before) == report
    assert not list(tmp_path.glob("*.staging"))

    with pytest.raises(FileExistsError, match="refusing to replace"):
        save_mem0_campaign_report({"status": "complete", "value": 2}, output)

    assert output.read_bytes() == before
    assert not list(tmp_path.glob("*.staging"))


def test_campaign_report_save_rejects_descendant_of_protected_input(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected-reports"
    protected.mkdir()
    output = protected / "campaign.json"

    with pytest.raises(ValueError, match="caller-protected input"):
        save_mem0_campaign_report(
            {"status": "complete"},
            output,
            protected_inputs=(protected,),
        )

    assert not output.exists()
