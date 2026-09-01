from __future__ import annotations

import copy
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.domain.discourse import canonical_json, quote_sha256
from memory_condense.eval.benchmark import BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
from memory_condense.eval.fast_completion_runtime import (
    FAST_COMPLETION_REQUEST_FORMAT,
    FAST_COMPLETION_RESPONSE_FORMAT,
)
from memory_condense.eval.mem0_adapter import (
    MEM0AI_PIN,
    MEM0_ATTRIBUTION_KIND,
    MEM0_BM25_MODEL,
    MEM0_CERTIFIED_RENDERING,
    MEM0_SPACY_MODEL,
    SourceRef,
)
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    merge_typed_evidence_contributions,
)
from tools.matched_eval.typed_memory_final_arm import (
    fit_typed_final_prompt,
    judge_row_projection,
)
from tools.matched_eval.typed_memory_final_judging import TypedFinalJudgeGoldRow
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec
from tools.mem0_eval.prompt_pack import (
    MEM0_MAX_PROMPT_TOKEN_PROXY,
    MEM0_PROMPT_CAP_SEMANTICS,
    MEM0_PROMPT_PACK_PROTOCOL,
    MEM0_REQUEST_WINDOW_SEMANTICS,
    MEM0_SOURCE_JUDGE_MODEL,
    MEM0_SOURCE_RESPONDER_MODEL,
    MEM0_TYPED_EPOCH,
    MEM0_TYPED_PROMPT_PACK_PROTOCOL,
    MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
    pack_mem0_prompt,
    pack_mem0_typed_prompt,
)
from tools.mem0_eval.run_shard import candidate_payload_for_mem0_typed_v1
from tools.mem0_eval.typed_adapter import adapt_mem0_retrieval_row
from tools.mem0_eval.typed_answer_lifecycle import (
    CHECKPOINT_DIR_NAME as ANSWER_CHECKPOINT_DIR_NAME,
    PREFLIGHT_NAME as ANSWER_PREFLIGHT_NAME,
    REPLAY_NAME as ANSWER_REPLAY_NAME,
    RUN_NAME as ANSWER_RUN_NAME,
    build_answer_preflight_payload,
    build_answer_runtime,
    load_verified_answer_preflight,
    load_verified_answer_run,
    materialize_answer_payload,
)
from tools.mem0_eval.typed_cost_ledger import (
    CommonFinalCostLedger,
    CommonProviderStageCost,
    Mem0ReadCostLedger,
    Mem0TypedEpochCostLedger,
    Mem0WriteCostLedger,
)
from tools.mem0_eval.typed_judge_lifecycle import (
    CHECKPOINT_DIR_NAME as JUDGE_CHECKPOINT_DIR_NAME,
    JUDGE_NAME as SOL_JUDGE_NAME,
    JUDGE_REPLAY_NAME as SOL_JUDGE_REPLAY_NAME,
    MAX_JUDGE_PROMPT_TOKENS,
    PREFLIGHT_NAME as SOL_PREFLIGHT_NAME,
    SCORE_NAME as SOL_SCORE_NAME,
    SCORE_REPLAY_NAME as SOL_SCORE_REPLAY_NAME,
    build_judge_preflight_payload,
    build_judge_runtime,
    load_verified_judge_score,
    materialize_judge_payloads,
)
from tools.mem0_eval.typed_epoch_campaign import (
    COMPARISON_SEMANTICS,
    COMMON_INPUT_NAME,
    CONTRIBUTION_BUNDLE_NAME,
    COST_PREFLIGHT_NAME,
    FINAL_COST_NAME,
    PREFLIGHT_NAME,
    REPLAY_NAME,
    RETRIEVAL_BUNDLE_NAME,
    Mem0TypedCampaignError,
    JUDGE_MODEL,
    JUDGE_OUTPUT_TOKEN_RESERVE,
    PARENT_SOURCE_FORMAT,
    PARENT_SOURCE_ROW_FORMAT,
    RESPONDER_MODEL,
    build_common_usage_payload,
    build_parent_population_payload,
    build_retrieval_export_payload,
    compose_campaign,
    finalize_costs,
    load_mem0_typed_contribution_checkpoint,
    _preflight_campaign_from_exports,
    replay_campaign,
)
from tools.run_mem0_typed_epoch import main as mem0_typed_epoch_main


def _evaluation_identity() -> dict[str, object]:
    return {
        "responder_model": MEM0_SOURCE_RESPONDER_MODEL,
        "judge_model": MEM0_SOURCE_JUDGE_MODEL,
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 20,
        "max_prompt_tokens": MEM0_MAX_PROMPT_TOKEN_PROXY,
        "prompt_cap_semantics": MEM0_PROMPT_CAP_SEMANTICS,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(range(0, 100, 10)),
    }


def _runtime_identity() -> dict[str, object]:
    return {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "certified": True,
        "local_owned_state": True,
        "on_disk": True,
        "stable_config_sha256": "a" * 64,
        "effective_config_sha256": "b" * 64,
        "stack": {
            "dependency_versions": {"mem0ai": MEM0AI_PIN},
            "bm25_model": MEM0_BM25_MODEL,
            "spacy_model": MEM0_SPACY_MODEL,
            "bm25_operational": True,
            "entity_extraction_operational": True,
        },
    }


def _publish_fast_checkpoint(path, body: dict[str, object]) -> str:
    receipt = identity_sha256(body)
    payload = {**body, "journal_sha256": receipt}
    path.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    return receipt


def _seed_provider_free_checkpoints(runtime, completions: tuple[str, ...]) -> None:
    """Create exact runtime journals without constructing or calling a client."""

    checkpoint_dir = runtime._checkpoint_dir  # noqa: SLF001 - fixture authority
    provenance = runtime.provenance.model_dump()
    for ordinal, (population_row, completion) in enumerate(
        zip(runtime.population.ordered_rows, completions, strict=True)
    ):
        call_key = identity_sha256(
            {
                "format": FAST_COMPLETION_REQUEST_FORMAT,
                "runtime_identity_sha256": runtime.runtime_identity_sha256,
                "prompt_population_sha256": runtime.population.prompt_population_sha256,
                "messages_sha256": population_row.messages_sha256,
                "prompt_token_proxy": population_row.prompt_token_proxy,
                "max_new_tokens": runtime.provenance.max_new_tokens,
            }
        )
        request_body = {
            "format": FAST_COMPLETION_REQUEST_FORMAT,
            "call_key_sha256": call_key,
            "runtime_identity_sha256": runtime.runtime_identity_sha256,
            "runtime_identity": provenance,
            "prompt_population_sha256": runtime.population.prompt_population_sha256,
            "messages_sha256": population_row.messages_sha256,
            "prompt_token_proxy": population_row.prompt_token_proxy,
            "max_new_tokens": runtime.provenance.max_new_tokens,
        }
        request_receipt = _publish_fast_checkpoint(
            checkpoint_dir / f"{call_key}.request.json",
            request_body,
        )
        response_body = {
            "format": FAST_COMPLETION_RESPONSE_FORMAT,
            "call_key_sha256": call_key,
            "request_journal_sha256": request_receipt,
            "messages_sha256": population_row.messages_sha256,
            "completion": completion,
            "completion_sha256": quote_sha256(completion),
            "requested_model": runtime.provenance.model,
            "response_id": f"provider-free-fixture-{ordinal}",
            "response_model": runtime.provenance.model,
            "finish_reason": "stop",
            "prompt_token_proxy": population_row.prompt_token_proxy,
            "completion_token_proxy": count_tokens(completion),
            "reported_prompt_tokens": None,
            "reported_completion_tokens": None,
            "reported_total_tokens": None,
            "provider_elapsed_s": 0.0,
        }
        _publish_fast_checkpoint(
            checkpoint_dir / f"{call_key}.response.json",
            response_body,
        )


def _window(
    *, source: str = "private-source", turn_start: int = 0, turn_count: int = 2
) -> SourceRef:
    return SourceRef(
        sample_id="sample-a",
        source=source,
        session=source,
        session_index=0,
        original_session_index=3,
        batch_index=(turn_start // 2) + 1,
        date="2025-01-01",
        turn_start=turn_start,
        turn_count=turn_count,
        roles=tuple("user" if index % 2 == 0 else "assistant" for index in range(turn_count)),
    )


def _candidate(
    rank: int,
    memory_id: str,
    text: str,
    *,
    windows: tuple[SourceRef, ...],
    created_at: str = "2099-12-31T00:00:00Z",
) -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        memory_id=memory_id,
        text=text,
        score=0.9 - rank / 100,
        created_at=created_at,
        attribution_kind=MEM0_ATTRIBUTION_KIND,
        request_window_attribution=windows,
    )


def _result(question: str, candidates: list[object]) -> SimpleNamespace:
    return SimpleNamespace(
        query=question,
        raw_pool=tuple(candidates),
        official_longmemeval_protocol=True,
        official_search_protocol=True,
        rendering_mode=MEM0_CERTIFIED_RENDERING,
        certified_rendering=True,
        comparison_certified=True,
        runtime_identity=_runtime_identity(),
        attribution_kind=MEM0_ATTRIBUTION_KIND,
        supports_exact_source_provenance=False,
    )


def _typed_row(
    question: str,
    candidates: list[object],
    *,
    question_id: str = "opaque-question",
) -> dict[str, object]:
    packed = pack_mem0_typed_prompt(
        question,
        _result(question, candidates),
        evaluation_identity=_evaluation_identity(),
    )
    return packed.to_retrieval_row(question_id=question_id, search_latency_s=0.1)


def test_typed_candidate_and_prompt_round_trip_preserve_windows_outside_messages():
    question = "2025-02-01: How much did Alice spend?"
    candidate = _candidate(
        1,
        "memory-a",
        "Alice spent $30 on books.",
        windows=(_window(),),
    )

    legacy = pack_mem0_prompt(
        question,
        _result(question, [candidate]),
        evaluation_identity=_evaluation_identity(),
    ).to_retrieval_row(question_id="opaque-question", search_latency_s=0.1)
    assert legacy["prompt_pack_protocol"] == MEM0_PROMPT_PACK_PROTOCOL
    assert "request_window_attribution" not in legacy["raw_pool"][0]

    payload = candidate_payload_for_mem0_typed_v1(candidate, 1)
    assert payload["request_window_semantics"] == MEM0_REQUEST_WINDOW_SEMANTICS
    assert payload["created_at_source_event_time_authoritative"] is False
    assert payload["request_window_attribution"][0]["source"] == "private-source"

    typed_pack = pack_mem0_typed_prompt(
        question,
        _result(question, [SimpleNamespace(**payload)]),
        evaluation_identity=_evaluation_identity(),
    )
    assert (
        typed_pack.prompt_token_proxy
        + typed_pack.responder_output_token_reserve
        <= MEM0_MAX_PROMPT_TOKEN_PROXY
    )
    assert typed_pack.max_prompt_token_proxy == (
        MEM0_MAX_PROMPT_TOKEN_PROXY - BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
    )
    typed = typed_pack.to_retrieval_row(
        question_id="opaque-question", search_latency_s=0.1
    )
    assert typed["format"] == MEM0_TYPED_RETRIEVAL_ROW_FORMAT
    assert typed["prompt_pack_protocol"] == MEM0_TYPED_PROMPT_PACK_PROTOCOL
    assert typed["typed_epoch"] == MEM0_TYPED_EPOCH
    assert typed["request_window_attribution_preserved"] is True
    assert typed["provenance"]["supports_exact_source_provenance"] is False
    rendered_messages = json.dumps(typed["messages"], sort_keys=True)
    assert "private-source" not in rendered_messages
    assert "sample-a" not in rendered_messages

    rebuilt = _typed_row(
        question,
        [SimpleNamespace(**typed["raw_pool"][0])],
    )
    assert rebuilt == typed


def test_typed_adapter_binds_exact_fields_and_groups_only_overlapping_windows():
    question = "2025-02-01: How much did Alice spend?"
    row = _typed_row(
        question,
        [
            _candidate(1, "m1", "Alice spent $30 on books.", windows=(_window(turn_start=0),)),
            _candidate(2, "m2", "Alice also spent $12 on pens.", windows=(_window(turn_start=1),)),
            _candidate(3, "m3", "Alice spent $8 on tea.", windows=(_window(turn_start=20),)),
        ],
    )
    spec = compile_typed_operator_spec(question)

    adapted = adapt_mem0_retrieval_row(
        spec,
        row,
        sealed_artifact_sha256="c" * 64,
        handle_start=20,
        group_start=40,
    )

    assert [binding.handle_id for binding in adapted.contribution.bindings] == [
        "H020",
        "H021",
        "H022",
    ]
    assert [binding.source_group_handle for binding in adapted.contribution.bindings] == [
        "G040",
        "G040",
        "G041",
    ]
    assert all(
        binding.origin is EvidenceOrigin.MEM0
        and binding.provenance_grade is ProvenanceGrade.INFERRED_MEMORY
        for binding in adapted.contribution.bindings
    )
    first = adapted.local_bindings[0]
    assert first.memory_id == "m1"
    assert first.retrieval_rank == first.search_order == 1
    assert first.score == pytest.approx(0.89)
    assert first.created_at == "2099-12-31T00:00:00Z"
    assert first.search_receipt_sha256 == row["retrieval_row_sha256"]
    assert first.text_sha256 == hashlib.sha256(
        b"Alice spent $30 on books."
    ).hexdigest()
    assert first.request_window_is_fact_evidence is False
    assert first.created_at_source_event_time_authoritative is False
    assert adapted.contribution.frontier_mode is FrontierMode.BOUNDED
    assert adapted.contribution.truncated is True
    assert adapted.permits_absence_claims is False
    assert adapted.retained_transformer_token_state_bytes == 0
    assert adapted.gold_loaded is False
    # created_at is 2099, but it cannot become a source-event date.
    assert all(item.date != "2099-12-31" for item in adapted.contribution.parsed.accepted_items)
    # Provider-visible/contribution projections expose only opaque handles or hashes.
    assert "private-source" not in json.dumps(adapted.projection(), sort_keys=True)
    assert "private-source" not in json.dumps(
        adapted.contribution.projection(), sort_keys=True
    )


def test_request_window_only_grade_never_creates_fact_item_or_closed_frontier():
    question = "2025-02-01: What happened?"
    row = _typed_row(
        question,
        [_candidate(1, "empty", "", windows=(_window(),))],
    )
    adapted = adapt_mem0_retrieval_row(
        compile_typed_operator_spec(question),
        row,
        sealed_artifact_sha256="d" * 64,
        source_pool="raw_pool",
    )

    assert adapted.local_bindings[0].provenance_grade is ProvenanceGrade.REQUEST_WINDOW_ONLY
    assert adapted.contribution.bindings[0].provenance_grade is ProvenanceGrade.REQUEST_WINDOW_ONLY
    assert adapted.contribution.parsed.accepted_items == ()
    assert adapted.frontier_mode == "bounded"
    assert adapted.permits_absence_claims is False


def test_typed_adapter_rejects_tampered_window_authority_and_gold_fields():
    question = "2025-02-01: What happened?"
    row = _typed_row(
        question,
        [_candidate(1, "m1", "Alice visited Rome.", windows=(_window(),))],
    )
    spec = compile_typed_operator_spec(question)

    authority = copy.deepcopy(row)
    authority["raw_pool"][0]["created_at_source_event_time_authoritative"] = True
    authority["packed_pool"][0]["created_at_source_event_time_authoritative"] = True
    body = dict(authority)
    body.pop("retrieval_row_sha256")
    authority["retrieval_row_sha256"] = identity_sha256(body)
    with pytest.raises(MatchedEvalContractError, match="authority"):
        adapt_mem0_retrieval_row(spec, authority, sealed_artifact_sha256="e" * 64)

    gold = copy.deepcopy(row)
    gold["reference_answer"] = "Rome"
    body = dict(gold)
    body.pop("retrieval_row_sha256")
    gold["retrieval_row_sha256"] = identity_sha256(body)
    with pytest.raises(MatchedEvalContractError, match="gold-bearing"):
        adapt_mem0_retrieval_row(spec, gold, sealed_artifact_sha256="e" * 64)


def _write_cost() -> Mem0WriteCostLedger:
    return Mem0WriteCostLedger(
        population_identity_sha256="1" * 64,
        add_attempted=2,
        add_completed=2,
        add_failed=0,
        extraction_attempted=2,
        extraction_completed=2,
        extraction_failed=0,
        extraction_raw_message_token_proxy=120,
        extraction_provider_input_tokens=None,
        extraction_provider_output_tokens=None,
        extraction_usage_status="unavailable_from_mem0_oss_public_api",
        embedding_operations=3,
        embedding_input_token_proxy=80,
        returned_memory_count=3,
        persisted_memory_count=3,
        persisted_storage_bytes=4096,
        add_latency_s=1.0,
        extraction_latency_s=0.8,
        embedding_latency_s=0.1,
        storage_latency_s=0.1,
    )


def _read_cost(**updates: object) -> Mem0ReadCostLedger:
    values: dict[str, object] = {
        "retrieval_artifact_sha256": "2" * 64,
        "search_attempted": 1,
        "search_completed": 1,
        "search_failed": 0,
        "raw_memory_count": 10,
        "raw_memory_token_proxy": 1000,
        "adapted_memory_count": 8,
        "adapted_memory_token_proxy": 800,
        "packed_memory_count": 6,
        "packed_memory_token_proxy": 600,
        "packed_full_prompt_token_proxy": 7000,
        "responder_output_token_reserve": 768,
        "search_latency_s": 0.2,
        "adaptation_latency_s": 0.02,
        "packing_latency_s": 0.01,
    }
    values.update(updates)
    return Mem0ReadCostLedger(**values)


def _provider(role: str) -> CommonProviderStageCost:
    return CommonProviderStageCost(
        role=role,
        model_id=(
            "openai/codex_sdk/gpt-5.6-terra"
            if role == "responder"
            else "openai/codex_sdk/gpt-5.6-sol"
        ),
        logical_calls_attempted=1,
        logical_calls_completed=1,
        logical_calls_failed=0,
        sdk_retry_attempts=0,
        provider_input_tokens=400,
        provider_output_tokens=40,
        latency_s=0.5,
    )


def test_cost_ledgers_seal_all_stages_under_full_request_budget_and_zero_state():
    write = _write_cost()
    read = _read_cost()
    final = CommonFinalCostLedger(
        question_count=1,
        responder=_provider("responder"),
        judge=_provider("judge"),
        max_full_responder_prompt_token_proxy=7000,
        responder_output_token_reserve=768,
        max_full_judge_prompt_token_proxy=500,
        judge_output_token_reserve=64,
    )
    epoch = Mem0TypedEpochCostLedger(
        write=write,
        read=read,
        common_final=final,
        population_identity_sha256="1" * 64,
        retrieval_artifact_sha256="2" * 64,
    )

    for receipt in (
        write.receipt_sha256,
        read.receipt_sha256,
        final.receipt_sha256,
        epoch.receipt_sha256,
    ):
        assert len(receipt) == 64
    assert epoch.projection()["typed_epoch"] == MEM0_TYPED_EPOCH
    assert write.retained_transformer_token_state_bytes == 0
    assert read.retained_transformer_token_state_bytes == 0
    assert final.retained_transformer_token_state_bytes == 0


def test_cost_ledgers_fail_closed_on_budget_calls_and_receipt_tampering():
    with pytest.raises(MatchedEvalContractError, match="hard 8k"):
        _read_cost(packed_full_prompt_token_proxy=7500)

    with pytest.raises(MatchedEvalContractError, match="hard 8k"):
        CommonFinalCostLedger(
            question_count=1,
            responder=_provider("responder"),
            judge=_provider("judge"),
            max_full_responder_prompt_token_proxy=7000,
            responder_output_token_reserve=768,
            max_full_judge_prompt_token_proxy=7990,
            judge_output_token_reserve=64,
        )

    with pytest.raises(MatchedEvalContractError, match="do not close"):
        CommonProviderStageCost(
            role="responder",
            model_id="openai/codex_sdk/gpt-5.6-terra",
            logical_calls_attempted=2,
            logical_calls_completed=1,
            logical_calls_failed=0,
            sdk_retry_attempts=0,
            provider_input_tokens=10,
            provider_output_tokens=2,
            latency_s=0.1,
        )

    with pytest.raises(MatchedEvalContractError, match="receipt changed"):
        Mem0WriteCostLedger(
            **{
                name: getattr(_write_cost(), name)
                for name in (
                    "population_identity_sha256",
                    "add_attempted",
                    "add_completed",
                    "add_failed",
                    "extraction_attempted",
                    "extraction_completed",
                    "extraction_failed",
                    "extraction_raw_message_token_proxy",
                    "extraction_provider_input_tokens",
                    "extraction_provider_output_tokens",
                    "extraction_usage_status",
                    "embedding_operations",
                    "embedding_input_token_proxy",
                    "returned_memory_count",
                    "persisted_memory_count",
                    "persisted_storage_bytes",
                    "add_latency_s",
                    "extraction_latency_s",
                    "embedding_latency_s",
                    "storage_latency_s",
                )
            },
            receipt_sha256="f" * 64,
        )


def _write_observation() -> dict[str, object]:
    return {
        "add_attempted": 2,
        "add_completed": 2,
        "add_failed": 0,
        "extraction_attempted": 2,
        "extraction_completed": 2,
        "extraction_failed": 0,
        "extraction_raw_message_token_proxy": 120,
        "extraction_provider_input_tokens": None,
        "extraction_provider_output_tokens": None,
        "extraction_usage_status": "unavailable_from_mem0_oss_public_api",
        "embedding_operations": 3,
        "embedding_input_token_proxy": 80,
        "returned_memory_count": 3,
        "persisted_memory_count": 3,
        "persisted_storage_bytes": 4096,
        "add_latency_s": 1.0,
        "extraction_latency_s": 0.8,
        "embedding_latency_s": 0.1,
        "storage_latency_s": 0.1,
    }


def _retrieval_cleanup() -> dict[str, object]:
    return {
        "active_scope_cleared": True,
        "adapter_closed": True,
        "external_provider_persistence_certified": False,
        "extraction_meter_restored_before_cleanup": True,
        "ledger_empty": True,
        "owned_state_path_absent": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "state_absent_after": True,
    }


def _sealed_campaign_inputs(tmp_path):
    questions = [
        "2025-02-01: How much did Alice spend?",
        "2025-02-02: What happened during Alice's trip?",
    ]
    retrieval_rows = [
        _typed_row(
            questions[0],
            [
                _candidate(
                    1,
                    "m1",
                    "Alice spent $30 on books.",
                    windows=(_window(source="private-source-a"),),
                )
            ],
        ),
        _typed_row(
            questions[1],
            [
                _candidate(
                    1,
                    "m2",
                    "Alice visited Rome during the trip.",
                    windows=(_window(source="private-source-b"),),
                )
            ],
            question_id="opaque-question-2",
        ),
    ]
    population = "7" * 64
    parent_source_payload = {
        "format": PARENT_SOURCE_FORMAT,
        "gold_loaded": False,
        "question_count": len(questions),
        "questions": [
            {
                "dated_question_sha256": hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest(),
                "format": PARENT_SOURCE_ROW_FORMAT,
                "gold_loaded": False,
                "ordinal": index,
                "prediction": f"parent prediction {index}",
                "prediction_sha256": quote_sha256(f"parent prediction {index}"),
                "question_id": (
                    "opaque-question" if index == 0 else "opaque-question-2"
                ),
                "question_sha256": hashlib.sha256(
                    f"question-{index}".encode()
                ).hexdigest(),
                "retained_transformer_token_state_bytes": 0,
                "route_id": compile_typed_operator_spec(question).style.value,
            }
            for index, question in enumerate(questions)
        ],
        "retained_transformer_token_state_bytes": 0,
    }
    parent_run, _ = publish_sealed_json(
        tmp_path / "treatment-parent-run.json", parent_source_payload
    )
    parent_replay, _ = publish_sealed_json(
        tmp_path / "treatment-parent-replay.json", parent_source_payload
    )
    parent_payload = build_parent_population_payload(
        population_identity_sha256=population,
        parent_run_path=parent_run.path,
        expected_parent_run_sha256=parent_run.sha256,
        parent_replay_path=parent_replay.path,
        expected_parent_replay_sha256=parent_replay.sha256,
        rows=[
            {
                "ordinal": index,
                "question_id": (
                    "opaque-question" if index == 0 else "opaque-question-2"
                ),
                "dated_question": question,
                "question_sha256": hashlib.sha256(
                    f"question-{index}".encode()
                ).hexdigest(),
                "route_id": compile_typed_operator_spec(question).style.value,
            }
            for index, question in enumerate(questions)
        ],
    )
    export_payload = build_retrieval_export_payload(
        population_identity_sha256=population,
        source_shard_sha256="8" * 64,
        retrieval_trace_sha256="9" * 64,
        question_offset=0,
        retrieval_rows=retrieval_rows,
        write_observation=_write_observation(),
        retrieval_cleanup=_retrieval_cleanup(),
    )
    # Replace only fixture files before any campaign output exists.
    export_path = tmp_path / "retrieval-export-final.json"
    export, _ = publish_sealed_json(export_path, export_payload)
    parent, _ = publish_sealed_json(tmp_path / "parent.json", parent_payload)
    return export, parent


def _parent_source_pins(parent: SealedArtifact) -> dict[str, str]:
    origin = parent.payload["parent_origin"]
    return {
        "expected_parent_run_sha256": origin["parent_run_sha256"],
        "expected_parent_replay_sha256": origin["parent_replay_sha256"],
    }


def test_provider_free_campaign_preflight_compose_replay_and_cost_finalize(
    tmp_path,
    monkeypatch,
):
    export, parent = _sealed_campaign_inputs(tmp_path)
    output = tmp_path / "campaign"
    _preflight_campaign_from_exports(
        retrieval_export_paths=[export.path],
        expected_retrieval_export_sha256s=[export.sha256],
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
        **_parent_source_pins(parent),
        output_root=output,
        expected_question_count=2,
    )
    preflight = read_sealed_json(output / PREFLIGHT_NAME)
    bundle = read_sealed_json(output / RETRIEVAL_BUNDLE_NAME)
    common_payload, contribution_payload, cost_payload = compose_campaign(
        preflight_path=preflight.path,
        expected_preflight_sha256=preflight.sha256,
        retrieval_bundle_path=bundle.path,
        expected_retrieval_bundle_sha256=bundle.sha256,
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
        output_root=output,
        expected_question_count=2,
    )
    common = read_sealed_json(output / COMMON_INPUT_NAME)
    contribution = read_sealed_json(output / CONTRIBUTION_BUNDLE_NAME)
    cost = read_sealed_json(output / COST_PREFLIGHT_NAME)

    assert common.payload == common_payload
    assert contribution.payload == contribution_payload
    assert cost.payload == cost_payload
    assert common.payload["contribution_bundle_sha256"] == contribution.sha256
    assert common.payload["provider_calls_completed"] == 0
    assert common.payload["max_full_chat_plus_output_tokens"] <= 8000
    assert common.payload["comparison_semantics"] == COMPARISON_SEMANTICS
    assert common.payload["model"] == RESPONDER_MODEL == "codex_sdk/gpt-5.6-terra"
    assert preflight.payload["judge_model"] == JUDGE_MODEL == "codex_sdk/gpt-5.6-sol"
    assert preflight.payload["judge_output_token_reserve"] == JUDGE_OUTPUT_TOKEN_RESERVE == 1024
    assert parent.payload["comparison_semantics"] == COMPARISON_SEMANTICS
    assert parent.payload["parent_origin"]["parent_run_sha256"] == parent.payload[
        "parent_origin"
    ]["parent_replay_sha256"]
    assert common.payload["questions"][0]["validation_contract"]
    assert cost.payload["write_cost"]["add_calls"]["completed"] == 2
    assert cost.payload["read_cost"]["search_calls"]["completed"] == 2
    assert cost.payload["read_cost"]["frontier_mode"] == "bounded"
    rendered = json.dumps(
        [row["messages"] for row in common.payload["questions"]],
        sort_keys=True,
    )
    assert "private-source-a" not in rendered
    assert "private-source-b" not in rendered

    answer_preflight_payload, answer_prompts = build_answer_preflight_payload(
        common,
        expected_question_count=2,
        max_concurrency=2,
    )
    assert answer_preflight_payload["comparison_semantics"] == "common_parent"
    assert answer_preflight_payload["model"] == "codex_sdk/gpt-5.6-terra"
    assert answer_preflight_payload["output_token_reserve"] == 768
    assert answer_preflight_payload["observed_max_complete_envelope_tokens"] <= 8000
    assert not (output / ANSWER_CHECKPOINT_DIR_NAME).exists()
    with pytest.raises(MatchedEvalContractError, match="runtime policy changed"):
        build_answer_preflight_payload(
            common,
            expected_question_count=2,
            model="openai/codex_sdk/gpt-5.6-terra",
            max_concurrency=2,
        )
    answer_preflight, _ = publish_sealed_json(
        output / ANSWER_PREFLIGHT_NAME,
        answer_preflight_payload,
    )
    loaded_answer_preflight, loaded_prompts, answer_rows = (
        load_verified_answer_preflight(
            answer_preflight.path,
            answer_preflight.sha256,
            expected_question_count=2,
        )
    )
    assert loaded_prompts == answer_prompts
    answer_runtime = build_answer_runtime(
        loaded_answer_preflight,
        loaded_prompts,
        checkpoint_dir=output / ANSWER_CHECKPOINT_DIR_NAME,
        client=None,
        max_concurrency=2,
        expected_question_count=2,
    )
    completions = ("not valid JSON", "not valid JSON")
    try:
        assert answer_runtime.provenance.model == "codex_sdk/gpt-5.6-terra"
        assert answer_runtime.provenance.max_new_tokens == 768
        assert answer_runtime.provenance.retries == 0
        assert answer_runtime.request_token_state_receipt()[
            "retained_transformer_token_state_bytes"
        ] == 0
        answer_runtime_provenance = answer_runtime.provenance.model_dump()
        answer_runtime_identity = answer_runtime.runtime_identity_sha256
        answer_runtime_population = answer_runtime.population.model_dump()
        _seed_provider_free_checkpoints(answer_runtime, completions)
        checkpoint_only_batch = answer_runtime.run()
    finally:
        answer_runtime.close()
    assert checkpoint_only_batch.provenance.model_dump() == answer_runtime_provenance
    assert checkpoint_only_batch.runtime_identity_sha256 == answer_runtime_identity
    assert checkpoint_only_batch.prompt_population.model_dump() == answer_runtime_population
    with pytest.raises(MatchedEvalContractError, match="complete checkpoint hits"):
        materialize_answer_payload(
            loaded_answer_preflight,
            answer_rows,
            SimpleNamespace(),
            expected_question_count=2,
        )
    answer_run_payload = materialize_answer_payload(
        loaded_answer_preflight,
        answer_rows,
        checkpoint_only_batch,
        expected_question_count=2,
    )
    assert all(
        row["prediction"] == parent.payload["questions"][row["ordinal"]][
            "parent_prediction"
        ]
        for row in answer_run_payload["questions"]
    )
    answer_run, _ = publish_sealed_json(output / ANSWER_RUN_NAME, answer_run_payload)
    answer_replay, _ = publish_sealed_json(
        output / ANSWER_REPLAY_NAME,
        answer_run_payload,
    )
    verified_run, verified_replay, answer_judge_rows = load_verified_answer_run(
        output,
        common_input_path=common.path,
        expected_common_input_sha256=common.sha256,
        expected_preflight_sha256=answer_preflight.sha256,
        expected_run_sha256=answer_run.sha256,
        expected_replay_sha256=answer_replay.sha256,
        expected_question_count=2,
    )
    assert verified_run.sha256 == verified_replay.sha256
    assert len(answer_judge_rows) == 2

    forged_root = tmp_path / "forged-answer-without-checkpoints"
    publish_sealed_json(forged_root / ANSWER_PREFLIGHT_NAME, answer_preflight.payload)
    forged_run, _ = publish_sealed_json(
        forged_root / ANSWER_RUN_NAME,
        answer_run.payload,
    )
    forged_replay, _ = publish_sealed_json(
        forged_root / ANSWER_REPLAY_NAME,
        answer_run.payload,
    )
    with pytest.raises(MatchedEvalContractError, match="checkpoint directory"):
        load_verified_answer_run(
            forged_root,
            common_input_path=common.path,
            expected_common_input_sha256=common.sha256,
            expected_preflight_sha256=answer_preflight.sha256,
            expected_run_sha256=forged_run.sha256,
            expected_replay_sha256=forged_replay.sha256,
            expected_question_count=2,
        )

    forged_binding_root = tmp_path / "forged-answer-prompt-binding"
    publish_sealed_json(
        forged_binding_root / ANSWER_PREFLIGHT_NAME,
        answer_preflight.payload,
    )
    shutil.copytree(
        output / ANSWER_CHECKPOINT_DIR_NAME,
        forged_binding_root / ANSWER_CHECKPOINT_DIR_NAME,
    )
    forged_binding_payload = copy.deepcopy(answer_run.payload)
    forged_binding_payload["questions"][0]["prompt_row_receipt_sha256"] = (
        answer_rows[1]["prompt_row_receipt_sha256"]
    )
    unsigned_forged_row = dict(forged_binding_payload["questions"][0])
    unsigned_forged_row.pop("source_row_sha256")
    forged_binding_payload["questions"][0]["source_row_sha256"] = identity_sha256(
        unsigned_forged_row
    )
    forged_binding_payload["judge_rows"][0] = judge_row_projection(
        forged_binding_payload["questions"][0]
    )
    forged_binding_run, _ = publish_sealed_json(
        forged_binding_root / ANSWER_RUN_NAME,
        forged_binding_payload,
    )
    forged_binding_replay, _ = publish_sealed_json(
        forged_binding_root / ANSWER_REPLAY_NAME,
        forged_binding_payload,
    )
    with pytest.raises(MatchedEvalContractError, match="checkpoint/preflight projection"):
        load_verified_answer_run(
            forged_binding_root,
            common_input_path=common.path,
            expected_common_input_sha256=common.sha256,
            expected_preflight_sha256=answer_preflight.sha256,
            expected_run_sha256=forged_binding_run.sha256,
            expected_replay_sha256=forged_binding_replay.sha256,
            expected_question_count=2,
        )

    gold_rows = tuple(
        TypedFinalJudgeGoldRow(
            ordinal=index,
            question_id=row["question_id"],
            question=f"question-{index}",
            question_sha256=row["question_sha256"],
            dated_question=parent.payload["questions"][index]["dated_question"],
            dated_question_sha256=row["dated_question_sha256"],
            reference=f"parent prediction {index}",
            reference_sha256=quote_sha256(f"parent prediction {index}"),
            category="single-session-user",
        )
        for index, row in enumerate(answer_judge_rows)
    )
    gold_population_sha256 = identity_sha256(
        [
            {
                "category": row.category,
                "dated_question_sha256": row.dated_question_sha256,
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "reference_sha256": row.reference_sha256,
            }
            for row in gold_rows
        ]
    )
    sol_preflight_payload, sol_prompts = build_judge_preflight_payload(
        answer_run=verified_run,
        answer_replay=verified_replay,
        source_rows=answer_judge_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        expected_question_count=2,
        max_concurrency=2,
    )
    assert sol_preflight_payload["model"] == "codex_sdk/gpt-5.6-sol"
    assert sol_preflight_payload["max_judge_prompt_tokens"] == 8000
    assert MAX_JUDGE_PROMPT_TOKENS + 1024 == 9024
    assert sol_preflight_payload["max_judge_complete_envelope_tokens"] == 9024
    assert sol_preflight_payload["observed_max_complete_envelope_tokens"] <= 9024
    assert sol_preflight_payload["sdk_retries"] == 0
    with pytest.raises(MatchedEvalContractError, match="runtime policy"):
        build_judge_preflight_payload(
            answer_run=verified_run,
            answer_replay=verified_replay,
            source_rows=answer_judge_rows,
            gold_rows=gold_rows,
            gold_population_sha256=gold_population_sha256,
            expected_question_count=2,
            model="openai/codex_sdk/gpt-5.6-sol",
            max_concurrency=2,
        )
    sol_preflight, _ = publish_sealed_json(
        output / SOL_PREFLIGHT_NAME,
        sol_preflight_payload,
    )
    sol_runtime = build_judge_runtime(
        sol_preflight,
        sol_prompts,
        checkpoint_dir=output / JUDGE_CHECKPOINT_DIR_NAME,
        client=None,
        max_concurrency=2,
        expected_question_count=2,
    )
    try:
        assert sol_runtime.provenance.model == JUDGE_MODEL
        assert sol_runtime.provenance.max_new_tokens == 1024
        assert sol_runtime.provenance.max_prompt_token_proxy == 8000
        assert sol_runtime.provenance.retries == 0
        _seed_provider_free_checkpoints(sol_runtime, ("CORRECT", "INCORRECT"))
        sol_batch = sol_runtime.run()
    finally:
        sol_runtime.close()
    with pytest.raises(MatchedEvalContractError, match="complete checkpoint hits"):
        materialize_judge_payloads(
            sol_preflight,
            sol_preflight_payload["prompt_rows"],
            SimpleNamespace(),
            expected_question_count=2,
        )
    sol_judge_payload, sol_score_payload = materialize_judge_payloads(
        sol_preflight,
        sol_preflight_payload["prompt_rows"],
        sol_batch,
        expected_question_count=2,
    )
    sol_judge, _ = publish_sealed_json(output / SOL_JUDGE_NAME, sol_judge_payload)
    sol_score, _ = publish_sealed_json(output / SOL_SCORE_NAME, sol_score_payload)
    sol_judge_replay, _ = publish_sealed_json(
        output / SOL_JUDGE_REPLAY_NAME,
        sol_judge_payload,
    )
    sol_score_replay, _ = publish_sealed_json(
        output / SOL_SCORE_REPLAY_NAME,
        sol_score_payload,
    )
    monkeypatch.setattr(
        "tools.mem0_eval.typed_judge_lifecycle.load_locked_typed_final_gold",
        lambda **_kwargs: (gold_rows, gold_population_sha256),
    )
    (
        verified_judge,
        verified_judge_replay,
        verified_score,
        verified_score_replay,
        verified_score_rows,
    ) = load_verified_judge_score(
        output,
        common_input_path=common.path,
        expected_common_input_sha256=common.sha256,
        answer_output_root=output,
        expected_answer_preflight_sha256=answer_preflight.sha256,
        expected_answer_run_sha256=answer_run.sha256,
        expected_answer_replay_sha256=answer_replay.sha256,
        dataset_path=tmp_path / "locked-dataset-fixture.json",
        split_path=tmp_path / "locked-split-fixture.json",
        expected_preflight_sha256=sol_preflight.sha256,
        expected_judge_sha256=sol_judge.sha256,
        expected_judge_replay_sha256=sol_judge_replay.sha256,
        expected_score_sha256=sol_score.sha256,
        expected_score_replay_sha256=sol_score_replay.sha256,
        expected_question_count=2,
    )
    assert verified_judge.sha256 == verified_judge_replay.sha256
    assert verified_score.sha256 == verified_score_replay.sha256
    assert [row["correct"] for row in verified_score_rows] == [True, False]
    assert verified_score.payload["correct"] == 1
    assert verified_score.payload["accuracy"] == 0.5

    local_checkpoint = load_mem0_typed_contribution_checkpoint(
        contribution_bundle_path=contribution.path,
        expected_contribution_bundle_sha256=contribution.sha256,
        retrieval_bundle_path=bundle.path,
        expected_retrieval_bundle_sha256=bundle.sha256,
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
        expected_question_count=2,
    )
    assert len(local_checkpoint.rows) == 2
    assert all(
        row.story_key_mode == "exact_request_window_receipt_local_v1"
        and row.contribution.mechanism_id == "mem0-typed-v1"
        and row.contribution.provider_prompt_count == 0
        for row in local_checkpoint.rows
    )
    first_local = local_checkpoint.rows[0]
    shared_packet = merge_typed_evidence_contributions(
        first_local.operator_spec,
        (first_local.contribution,),
        output_token_reserve=1,
    )
    shared_fit = fit_typed_final_prompt(
        dated_question=parent.payload["questions"][0]["dated_question"],
        parent_prediction=parent.payload["questions"][0]["parent_prediction"],
        packet=shared_packet,
        mechanism_by_handle={
            binding.handle_id: first_local.contribution.mechanism_id
            for binding in first_local.contribution.bindings
        },
        local_story_keys_by_group=first_local.local_story_keys_by_group,
        forbidden_provider_literals=tuple(
            dict.fromkeys(
                value
                for binding in first_local.local_story_source_bindings
                for value in (
                    binding["local_story_source"]["sample_id"],
                    binding["local_story_source"]["source"],
                    binding["local_story_source"]["session"],
                )
            )
        ),
        minimum_usable_items_per_mechanism=1,
    )
    assert [dict(message) for message in shared_fit.messages] == common.payload[
        "questions"
    ][0]["messages"]
    assert shared_fit.projection(include_local=False) == common.payload[
        "questions"
    ][0]["provider_projection"]
    namespace_by_question = {
        row.question_id: hashlib.sha256(
            f"namespace-{row.ordinal}".encode()
        ).hexdigest()
        for row in local_checkpoint.rows
    }
    cross_lane_checkpoint = load_mem0_typed_contribution_checkpoint(
        contribution_bundle_path=contribution.path,
        expected_contribution_bundle_sha256=contribution.sha256,
        retrieval_bundle_path=bundle.path,
        expected_retrieval_bundle_sha256=bundle.sha256,
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
        namespace_id_by_question_id=namespace_by_question,
        expected_question_count=2,
    )
    first = cross_lane_checkpoint.rows[0]
    expected_story_key = identity_sha256(
        {
            "namespace_id": namespace_by_question[first.question_id],
            "source_id": "private-source-a",
        }
    )
    assert first.story_key_mode == "exact_treatment_namespace_source_v1"
    assert expected_story_key in next(
        iter(first.local_story_keys_by_group.values())
    )
    with pytest.raises(
        Mem0TypedCampaignError,
        match="exactly cover",
    ):
        load_mem0_typed_contribution_checkpoint(
            contribution_bundle_path=contribution.path,
            expected_contribution_bundle_sha256=contribution.sha256,
            retrieval_bundle_path=bundle.path,
            expected_retrieval_bundle_sha256=bundle.sha256,
            parent_population_path=parent.path,
            expected_parent_population_sha256=parent.sha256,
            namespace_id_by_question_id={
                local_checkpoint.rows[0].question_id: "a" * 64
            },
            expected_question_count=2,
        )

    tampered_payload = copy.deepcopy(contribution.payload)
    tampered_row = tampered_payload["questions"][0]
    tampered_row["question_sha256"] = "f" * 64
    tampered_body = dict(tampered_row)
    tampered_body.pop("contribution_row_sha256")
    tampered_row["contribution_row_sha256"] = identity_sha256(tampered_body)
    tampered, _ = publish_sealed_json(
        tmp_path / "tampered-contribution.json",
        tampered_payload,
    )
    with pytest.raises(
        Mem0TypedCampaignError,
        match="not replay-identical",
    ):
        load_mem0_typed_contribution_checkpoint(
            contribution_bundle_path=tampered.path,
            expected_contribution_bundle_sha256=tampered.sha256,
            retrieval_bundle_path=bundle.path,
            expected_retrieval_bundle_sha256=bundle.sha256,
            parent_population_path=parent.path,
            expected_parent_population_sha256=parent.sha256,
            expected_question_count=2,
        )

    replay = replay_campaign(
        preflight_path=preflight.path,
        expected_preflight_sha256=preflight.sha256,
        retrieval_bundle_path=bundle.path,
        expected_retrieval_bundle_sha256=bundle.sha256,
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
        contribution_bundle_path=contribution.path,
        expected_contribution_bundle_sha256=contribution.sha256,
        common_input_path=common.path,
        expected_common_input_sha256=common.sha256,
        cost_preflight_path=cost.path,
        expected_cost_preflight_sha256=cost.sha256,
        output_root=output,
        expected_question_count=2,
    )
    assert replay["byte_identical"] is True
    assert replay["physical_provider_calls"] == 0
    assert read_sealed_json(output / REPLAY_NAME).payload == replay

    usage_payload = build_common_usage_payload(
        common_input_sha256=common.sha256,
        question_count=2,
        responder={
            "logical_calls_attempted": 2,
            "logical_calls_completed": 2,
            "logical_calls_failed": 0,
            "sdk_retry_attempts": 0,
            "provider_input_tokens": 1000,
            "provider_output_tokens": 60,
            "latency_s": 1.2,
            "max_full_prompt_token_proxy": common.payload[
                "max_prompt_token_proxy"
            ],
            "output_token_reserve": 768,
        },
        judge={
            "logical_calls_attempted": 2,
            "logical_calls_completed": 2,
            "logical_calls_failed": 0,
            "sdk_retry_attempts": 0,
            "provider_input_tokens": 700,
            "provider_output_tokens": 20,
            "latency_s": 0.8,
            "max_full_prompt_token_proxy": 500,
            "output_token_reserve": 1024,
        },
    )
    with pytest.raises(
        MatchedEvalContractError,
        match="exact successful full-population run",
    ):
        build_common_usage_payload(
            common_input_sha256=common.sha256,
            question_count=2,
            responder={
                key: (
                    3
                    if key == "logical_calls_attempted"
                    else 1
                    if key == "logical_calls_failed"
                    else usage_payload["responder"][key]
                )
                for key in (
                    "logical_calls_attempted",
                    "logical_calls_completed",
                    "logical_calls_failed",
                    "sdk_retry_attempts",
                    "provider_input_tokens",
                    "provider_output_tokens",
                    "latency_s",
                    "max_full_prompt_token_proxy",
                    "output_token_reserve",
                )
            },
            judge={
                key: usage_payload["judge"][key]
                for key in (
                    "logical_calls_attempted",
                    "logical_calls_completed",
                    "logical_calls_failed",
                    "sdk_retry_attempts",
                    "provider_input_tokens",
                    "provider_output_tokens",
                    "latency_s",
                    "max_full_prompt_token_proxy",
                    "output_token_reserve",
                )
            },
        )
    usage, _ = publish_sealed_json(tmp_path / "usage.json", usage_payload)
    final = finalize_costs(
        common_input_path=common.path,
        expected_common_input_sha256=common.sha256,
        cost_preflight_path=cost.path,
        expected_cost_preflight_sha256=cost.sha256,
        common_usage_path=usage.path,
        expected_common_usage_sha256=usage.sha256,
        output_root=output,
        expected_question_count=2,
    )
    assert final["epoch_cost"]["typed_epoch"] == MEM0_TYPED_EPOCH
    assert final["common_final_cost"]["question_count"] == 2
    assert read_sealed_json(output / FINAL_COST_NAME).payload == final


def test_campaign_rejects_legacy_rows_gold_and_wrong_locked_hash(tmp_path):
    question = "2025-02-01: What happened?"
    row = _typed_row(
        question,
        [_candidate(1, "m1", "Alice visited Rome.", windows=(_window(),))],
    )
    row["format"] = "memory-condense-mem0-retrieval-row-v2"
    body = dict(row)
    body.pop("retrieval_row_sha256")
    row["retrieval_row_sha256"] = identity_sha256(body)
    with pytest.raises(Mem0TypedCampaignError, match="not v3"):
        build_retrieval_export_payload(
            population_identity_sha256="1" * 64,
            source_shard_sha256="2" * 64,
            retrieval_trace_sha256="3" * 64,
            question_offset=0,
            retrieval_rows=[row],
            write_observation=_write_observation(),
            retrieval_cleanup=_retrieval_cleanup(),
        )

    bad_cleanup = _retrieval_cleanup()
    bad_cleanup["state_absent_after"] = False
    valid_row = _typed_row(
        question,
        [_candidate(1, "m2", "Alice visited Rome.", windows=(_window(),))],
    )
    with pytest.raises(Mem0TypedCampaignError, match="state_absent_after"):
        build_retrieval_export_payload(
            population_identity_sha256="1" * 64,
            source_shard_sha256="2" * 64,
            retrieval_trace_sha256="3" * 64,
            question_offset=0,
            retrieval_rows=[valid_row],
            write_observation=_write_observation(),
            retrieval_cleanup=bad_cleanup,
        )

    parent_source = {
        "format": PARENT_SOURCE_FORMAT,
        "gold_loaded": False,
        "question_count": 1,
        "questions": [
            {
                "dated_question_sha256": hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest(),
                "format": PARENT_SOURCE_ROW_FORMAT,
                "gold_loaded": False,
                "ordinal": 0,
                "prediction": "parent",
                "prediction_sha256": quote_sha256("parent"),
                "question_id": "q",
                "question_sha256": "4" * 64,
                "retained_transformer_token_state_bytes": 0,
                "route_id": compile_typed_operator_spec(question).style.value,
            }
        ],
        "retained_transformer_token_state_bytes": 0,
    }
    parent_run, _ = publish_sealed_json(tmp_path / "parent-run.json", parent_source)
    parent_replay, _ = publish_sealed_json(
        tmp_path / "parent-replay.json", parent_source
    )
    with pytest.raises(Mem0TypedCampaignError, match="fields changed"):
        build_parent_population_payload(
            population_identity_sha256="1" * 64,
            parent_run_path=parent_run.path,
            expected_parent_run_sha256=parent_run.sha256,
            parent_replay_path=parent_replay.path,
            expected_parent_replay_sha256=parent_replay.sha256,
            rows=[
                {
                    "ordinal": 0,
                    "question_id": "q",
                    "dated_question": question,
                    "question_sha256": "4" * 64,
                    "route_id": compile_typed_operator_spec(question).style.value,
                    "reference_answer": "Rome",
                }
            ],
        )

    arbitrary_parent = copy.deepcopy(parent_source)
    arbitrary_parent["format"] = "arbitrary-gold-blind-parent-v1"
    arbitrary_run, _ = publish_sealed_json(
        tmp_path / "arbitrary-parent-run.json", arbitrary_parent
    )
    arbitrary_replay, _ = publish_sealed_json(
        tmp_path / "arbitrary-parent-replay.json", arbitrary_parent
    )
    with pytest.raises(Mem0TypedCampaignError, match="origin changed"):
        build_parent_population_payload(
            population_identity_sha256="1" * 64,
            parent_run_path=arbitrary_run.path,
            expected_parent_run_sha256=arbitrary_run.sha256,
            parent_replay_path=arbitrary_replay.path,
            expected_parent_replay_sha256=arbitrary_replay.sha256,
            rows=[
                {
                    "ordinal": 0,
                    "question_id": "q",
                    "dated_question": question,
                    "question_sha256": "4" * 64,
                    "route_id": compile_typed_operator_spec(question).style.value,
                }
            ],
        )

    export, parent = _sealed_campaign_inputs(tmp_path)
    with pytest.raises(Mem0TypedCampaignError, match="SHA-256 changed"):
        _preflight_campaign_from_exports(
            retrieval_export_paths=[export.path],
            expected_retrieval_export_sha256s=["f" * 64],
            parent_population_path=parent.path,
            expected_parent_population_sha256=parent.sha256,
            **_parent_source_pins(parent),
            output_root=tmp_path / "never",
            expected_question_count=2,
            dry_run=True,
        )
    with pytest.raises(Mem0TypedCampaignError, match="explicitly authorized"):
        _preflight_campaign_from_exports(
            retrieval_export_paths=[export.path],
            expected_retrieval_export_sha256s=[export.sha256],
            parent_population_path=parent.path,
            expected_parent_population_sha256=parent.sha256,
            expected_parent_run_sha256="f" * 64,
            expected_parent_replay_sha256=parent.payload["parent_origin"][
                "parent_replay_sha256"
            ],
            output_root=tmp_path / "never-parent-pin",
            expected_question_count=2,
            dry_run=True,
        )


def test_campaign_cli_compose_dry_run_is_provider_free(tmp_path, capsys):
    export, parent = _sealed_campaign_inputs(tmp_path)
    output = tmp_path / "campaign-cli"
    _preflight_campaign_from_exports(
        retrieval_export_paths=[export.path],
        expected_retrieval_export_sha256s=[export.sha256],
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
        **_parent_source_pins(parent),
        output_root=output,
        expected_question_count=2,
    )
    preflight = read_sealed_json(output / PREFLIGHT_NAME)
    bundle = read_sealed_json(output / RETRIEVAL_BUNDLE_NAME)

    assert (
        mem0_typed_epoch_main(
            [
                "compose",
                "--preflight",
                str(preflight.path),
                "--expected-preflight-sha256",
                preflight.sha256,
                "--retrieval-bundle",
                str(bundle.path),
                "--expected-retrieval-bundle-sha256",
                bundle.sha256,
                "--parent-population",
                str(parent.path),
                "--expected-parent-population-sha256",
                parent.sha256,
                "--output-root",
                str(output / "dry"),
                "--expected-question-count",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is True
    assert printed["physical_provider_calls"] == 0
    assert printed["gold_loaded"] is False
    assert not (output / "dry").exists()
