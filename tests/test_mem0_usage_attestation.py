from __future__ import annotations

import copy
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FAST_COMPLETION_RESPONSE_FORMAT,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.live import DEFAULT_GATEWAY_URL
from tools.matched_eval.typed_memory_final_arm import (
    HARD_PROMPT_TOKEN_CAP,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
)
from tools.mem0_eval.typed_answer_lifecycle import (
    PREFLIGHT_NAME as ANSWER_PREFLIGHT_NAME,
    REPLAY_NAME as ANSWER_REPLAY_NAME,
    RUN_FORMAT as ANSWER_RUN_FORMAT,
    RUN_NAME as ANSWER_RUN_NAME,
)
from tools.mem0_eval.typed_epoch_campaign import (
    COMPARISON_SEMANTICS,
    JUDGE_MODEL,
    JUDGE_OUTPUT_TOKEN_RESERVE,
    RESPONDER_MODEL,
)
from tools.mem0_eval.typed_cost_ledger import Mem0ReadCostLedger, Mem0WriteCostLedger
from tools.mem0_eval.typed_judge_lifecycle import (
    JUDGE_FORMAT,
    JUDGE_NAME,
    JUDGE_REPLAY_NAME,
    MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS,
    MAX_JUDGE_PROMPT_TOKENS,
    PREFLIGHT_NAME as JUDGE_PREFLIGHT_NAME,
    SCORE_NAME,
    SCORE_REPLAY_NAME,
)
from tools.mem0_eval.typed_usage_attestation import (
    Mem0UsageAttestationError,
    VerifiedMem0UsageAttestation,
    _derive_stage,
    load_verified_final_cost,
    load_verified_usage_attestation,
    publish_usage_attestation,
    publish_verified_final_cost,
)


QUESTION_COUNT = 100
PARENT_SHA = "e" * 64


def _publish_journal(path: Path, body: dict[str, object]) -> str:
    receipt = identity_sha256(body)
    path.write_bytes(canonical_json_bytes({**body, "journal_sha256": receipt}))
    return receipt


def _stage(
    root: Path,
    *,
    role: str,
    common_sha256: str,
) -> tuple[SealedArtifact, dict[str, object]]:
    is_answer = role == "responder"
    model = RESPONDER_MODEL if is_answer else JUDGE_MODEL
    prompt_cap = MAX_CHAT_PROMPT_TOKENS if is_answer else MAX_JUDGE_PROMPT_TOKENS
    output_cap = OUTPUT_TOKEN_RESERVE if is_answer else JUDGE_OUTPUT_TOKEN_RESERVE
    complete_cap = (
        HARD_PROMPT_TOKEN_CAP
        if is_answer
        else MAX_JUDGE_COMPLETE_ENVELOPE_TOKENS
    )
    experiment = ANSWER_RUN_FORMAT if is_answer else JUDGE_FORMAT
    prompts = tuple(
        (
            {"role": "system", "content": f"sealed {role} fixture"},
            {"role": "user", "content": f"opaque question {ordinal}"},
        )
        for ordinal in range(QUESTION_COUNT)
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=prompt_cap
    )
    prompt_rows = [
        {
            "ordinal": ordinal,
            "messages_sha256": row.messages_sha256,
            "prompt_token_proxy": row.prompt_token_proxy,
            "route_id": "direct_fact" if ordinal % 2 == 0 else "temporal_event",
            **({"demand_class": "direct_fact"} if not is_answer else {}),
        }
        for ordinal, row in enumerate(population.ordered_rows)
    ]
    observed_complete = max(
        row["prompt_token_proxy"] + output_cap for row in prompt_rows
    )
    preflight_payload = {
        "common_input_sha256": common_sha256,
        "gateway_url": DEFAULT_GATEWAY_URL,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "max_chat_prompt_tokens": prompt_cap if is_answer else None,
        "max_judge_complete_envelope_tokens": complete_cap if not is_answer else None,
        "max_judge_prompt_tokens": prompt_cap if not is_answer else None,
        "model": model,
        "observed_max_complete_envelope_tokens": observed_complete,
        "output_token_reserve": output_cap,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": prompt_rows,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "sdk_retries": 0,
    }
    name = ANSWER_PREFLIGHT_NAME if is_answer else JUDGE_PREFLIGHT_NAME
    preflight, _ = publish_sealed_json(root / name, preflight_payload)
    runtime = FastCompletionRuntime(
        checkpoint_dir=root / f"{role}-calls",
        prompt_population=prompts,
        model=model,
        client=None,
        max_prompt_tokens=prompt_cap,
        max_new_tokens=output_cap,
        max_concurrency=4,
        retries=0,
        benchmark_provenance={
            "arm": f"mem0_{role}_fixture",
            "authorized_unique_calls": QUESTION_COUNT,
            "common_input_sha256": common_sha256,
            "comparison_semantics": COMPARISON_SEMANTICS,
            "experiment_format": experiment,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "preflight_artifact_sha256": preflight.sha256,
        },
    )
    try:
        for ordinal, population_row in enumerate(runtime.population.ordered_rows):
            request_body = runtime._request_body(  # noqa: SLF001 - fixture authority
                population_row.messages_sha256
            )
            call_key = request_body["call_key_sha256"]
            request_sha = _publish_journal(
                runtime._checkpoint_dir / f"{call_key}.request.json",  # noqa: SLF001
                request_body,
            )
            completion = f"answer {ordinal}" if is_answer else "CORRECT"
            completion_proxy = count_tokens(completion)
            response_body = {
                "format": FAST_COMPLETION_RESPONSE_FORMAT,
                "call_key_sha256": call_key,
                "request_journal_sha256": request_sha,
                "messages_sha256": population_row.messages_sha256,
                "completion": completion,
                "completion_sha256": quote_sha256(completion),
                "requested_model": model,
                "response_id": f"sealed-{role}-{ordinal}",
                "response_model": model,
                "finish_reason": "stop",
                "prompt_token_proxy": population_row.prompt_token_proxy,
                "completion_token_proxy": completion_proxy,
                "reported_prompt_tokens": population_row.prompt_token_proxy,
                "reported_completion_tokens": completion_proxy,
                "reported_total_tokens": (
                    population_row.prompt_token_proxy + completion_proxy
                ),
                "provider_elapsed_s": (ordinal + 1) / 1000,
            }
            _publish_journal(
                runtime._checkpoint_dir / f"{call_key}.response.json",  # noqa: SLF001
                response_body,
            )
        batch = runtime.run()
    finally:
        runtime.close()
    return preflight, batch.model_dump()


def _reseal_receipt(payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("receipt_sha256", None)
    payload["receipt_sha256"] = identity_sha256(body)


def _full100_sources(tmp_path: Path, monkeypatch):
    common, _ = publish_sealed_json(
        tmp_path / "common.json", {"format": "provider-free-common-fixture-v1"}
    )
    answer_root = tmp_path / "answer"
    judge_root = tmp_path / "judge"
    answer_preflight, answer_batch = _stage(
        answer_root, role="responder", common_sha256=common.sha256
    )
    answer_payload = {
        "common_input_sha256": common.sha256,
        "completion_batch": answer_batch,
        "parent_origin_receipt_sha256": PARENT_SHA,
        "question_count": QUESTION_COUNT,
    }
    answer_run, _ = publish_sealed_json(answer_root / ANSWER_RUN_NAME, answer_payload)
    answer_replay, _ = publish_sealed_json(
        answer_root / ANSWER_REPLAY_NAME, answer_payload
    )

    judge_preflight, judge_batch = _stage(
        judge_root, role="judge", common_sha256=common.sha256
    )
    judge_payload = {
        "answer_replay_sha256": answer_replay.sha256,
        "answer_run_sha256": answer_run.sha256,
        "common_input_sha256": common.sha256,
        "completion_batch": judge_batch,
        "parent_origin_receipt_sha256": PARENT_SHA,
        "question_count": QUESTION_COUNT,
    }
    judge, _ = publish_sealed_json(judge_root / JUDGE_NAME, judge_payload)
    judge_replay, _ = publish_sealed_json(
        judge_root / JUDGE_REPLAY_NAME, judge_payload
    )
    score, _ = publish_sealed_json(
        judge_root / SCORE_NAME, {"question_count": QUESTION_COUNT}
    )
    score_replay, _ = publish_sealed_json(
        judge_root / SCORE_REPLAY_NAME, score.payload
    )

    strict_reader_calls: list[dict[str, object]] = []

    def strict_reader(output_root, **kwargs):
        strict_reader_calls.append({"output_root": output_root, **kwargs})
        return judge, judge_replay, score, score_replay, tuple()

    monkeypatch.setattr(
        "tools.mem0_eval.typed_usage_attestation.load_verified_judge_score",
        strict_reader,
    )
    authority = {
        "common_input_path": common.path,
        "expected_common_input_sha256": common.sha256,
        "answer_output_root": answer_root,
        "expected_answer_preflight_sha256": answer_preflight.sha256,
        "expected_answer_run_sha256": answer_run.sha256,
        "expected_answer_replay_sha256": answer_replay.sha256,
        "judge_output_root": judge_root,
        "dataset_path": tmp_path / "unused-dataset.json",
        "split_path": tmp_path / "unused-split.json",
        "expected_judge_preflight_sha256": judge_preflight.sha256,
        "expected_judge_sha256": judge.sha256,
        "expected_judge_replay_sha256": judge_replay.sha256,
        "expected_score_sha256": score.sha256,
        "expected_score_replay_sha256": score_replay.sha256,
    }
    return authority, answer_preflight, answer_batch, strict_reader_calls


def test_full100_journal_usage_attestation_is_derived_and_replay_closed(
    tmp_path, monkeypatch
):
    authority, _preflight, _batch, strict_reader_calls = _full100_sources(
        tmp_path, monkeypatch
    )
    attestation, replay = publish_usage_attestation(
        tmp_path / "usage", **authority
    )
    verified = load_verified_usage_attestation(
        attestation.path,
        attestation.sha256,
        replay.path,
        replay.sha256,
        **authority,
    )
    assert attestation.sha256 == replay.sha256
    assert len(strict_reader_calls) == 2
    for stage in (verified.responder, verified.judge):
        assert stage["calls"] == {
            "attempted": 100,
            "completed": 100,
            "failed": 0,
            "retry_attempts": 0,
            "scope": "journaled_request_response_pairs",
        }
        assert stage["tokens"]["input_accounting_basis"] == "provider_reported"
        assert stage["tokens"]["output_accounting_basis"] == "provider_reported"
        assert stage["tokens"]["accounted_input_tokens"] > 0
        assert stage["tokens"]["accounted_output_tokens"] > 0
        assert stage["latency_s"] > 0
        assert stage["claim_boundary"].startswith(
            "content_authenticated_checkpoint_pairs"
        )
        assert stage["returned_model_accounting"]["returned_models"] == [
            {
                "completed": 100,
                "response_model": stage["model_id"],
            }
        ]
        assert stage["returned_model_accounting"]["claim_scope"].endswith(
            "not_backend_route_attestation"
        )
    assert verified.responder["budget"] == {
        "complete_envelope_token_cap": 8000,
        "hard_prompt_token_cap": 8000,
        "observed_max_complete_envelope_tokens": verified.responder["budget"][
            "observed_max_prompt_token_proxy"
        ]
        + 768,
        "observed_max_prompt_token_proxy": verified.responder["budget"][
            "observed_max_prompt_token_proxy"
        ],
        "output_token_reserve": 768,
        "prompt_token_proxy_cap": 7232,
    }
    assert verified.judge["budget"]["prompt_token_proxy_cap"] == 8000
    assert verified.judge["budget"]["complete_envelope_token_cap"] == 9024

    with pytest.raises(Mem0UsageAttestationError, match="strict journal reader"):
        VerifiedMem0UsageAttestation(
            attestation,
            replay,
            verified.responder,
            verified.judge,
            _token=object(),
        )


def test_usage_attestation_rejects_self_report_and_resealed_forgery(
    tmp_path, monkeypatch
):
    authority, answer_preflight, answer_batch, _calls = _full100_sources(
        tmp_path, monkeypatch
    )
    attestation, replay = publish_usage_attestation(
        tmp_path / "usage", **authority
    )

    bad_batch = copy.deepcopy(answer_batch)
    bad_batch["usage"]["prompt_token_proxy"] += 1
    with pytest.raises(
        Mem0UsageAttestationError, match="aggregate usage differs"
    ):
        _derive_stage(
            role="responder",
            batch_payload=bad_batch,
            preflight=answer_preflight,
        )

    nonterminal = copy.deepcopy(answer_batch)
    nonterminal["unique_records"][0]["finish_reason"] = "length"
    with pytest.raises(Mem0UsageAttestationError, match="journal pair 0"):
        _derive_stage(
            role="responder",
            batch_payload=nonterminal,
            preflight=answer_preflight,
        )

    over_reserve = copy.deepcopy(answer_batch)
    over_reserve["unique_records"][0]["reported_completion_tokens"] = 769
    with pytest.raises(Mem0UsageAttestationError, match="journal pair 0"):
        _derive_stage(
            role="responder",
            batch_payload=over_reserve,
            preflight=answer_preflight,
        )

    over_total = copy.deepcopy(answer_batch)
    first = over_total["unique_records"][0]
    first["reported_total_tokens"] = first["reported_prompt_tokens"] + 769
    with pytest.raises(Mem0UsageAttestationError, match="journal pair 0"):
        _derive_stage(
            role="responder",
            batch_payload=over_total,
            preflight=answer_preflight,
        )

    forged_payload = copy.deepcopy(attestation.payload)
    forged_payload["responder"]["calls"]["attempted"] = 101
    _reseal_receipt(forged_payload["responder"])
    _reseal_receipt(forged_payload)
    forged, _ = publish_sealed_json(
        tmp_path / "forged" / "usage.json", forged_payload
    )
    forged_replay, _ = publish_sealed_json(
        tmp_path / "forged" / "usage-replay.json", forged_payload
    )
    with pytest.raises(Mem0UsageAttestationError, match="strict-reader replay"):
        load_verified_usage_attestation(
            forged.path,
            forged.sha256,
            forged_replay.path,
            forged_replay.sha256,
            **authority,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'responder'"):
        publish_usage_attestation(
            tmp_path / "self-report",
            **authority,
            responder={"attempted": 100, "completed": 100},
        )


def test_journal_attested_final_cost_is_capability_bound_and_replay_closed(
    tmp_path, monkeypatch
):
    authority, _answer_preflight, _answer_batch, _calls = _full100_sources(
        tmp_path, monkeypatch
    )
    attestation, attestation_replay = publish_usage_attestation(
        tmp_path / "usage", **authority
    )
    usage = load_verified_usage_attestation(
        attestation.path,
        attestation.sha256,
        attestation_replay.path,
        attestation_replay.sha256,
        **authority,
    )

    population_sha = "7" * 64
    retrieval_sha = "8" * 64
    contribution_sha = "9" * 64
    write = Mem0WriteCostLedger(
        population_identity_sha256=population_sha,
        add_attempted=100,
        add_completed=100,
        add_failed=0,
        extraction_attempted=100,
        extraction_completed=100,
        extraction_failed=0,
        extraction_raw_message_token_proxy=10_000,
        extraction_provider_input_tokens=None,
        extraction_provider_output_tokens=None,
        extraction_usage_status="unavailable_from_mem0_oss_public_api",
        embedding_operations=100,
        embedding_input_token_proxy=8_000,
        returned_memory_count=200,
        persisted_memory_count=200,
        persisted_storage_bytes=50_000,
        add_latency_s=10.0,
        extraction_latency_s=8.0,
        embedding_latency_s=1.0,
        storage_latency_s=1.0,
    )
    read = Mem0ReadCostLedger(
        retrieval_artifact_sha256=retrieval_sha,
        search_attempted=100,
        search_completed=100,
        search_failed=0,
        raw_memory_count=500,
        raw_memory_token_proxy=20_000,
        adapted_memory_count=400,
        adapted_memory_token_proxy=15_000,
        packed_memory_count=300,
        packed_memory_token_proxy=10_000,
        packed_full_prompt_token_proxy=7_000,
        responder_output_token_reserve=768,
        search_latency_s=5.0,
        adaptation_latency_s=1.0,
        packing_latency_s=1.0,
    )
    monkeypatch.setattr(
        "tools.mem0_eval.typed_usage_attestation._validate_common_input",
        lambda _payload, **_kwargs: {
            "contribution_bundle_sha256": contribution_sha,
            "parent_origin_receipt_sha256": PARENT_SHA,
            "retrieval_bundle_sha256": retrieval_sha,
        },
    )
    monkeypatch.setattr(
        "tools.mem0_eval.typed_usage_attestation._validate_cost_preflight",
        lambda _payload, **_kwargs: (
            {"population_identity_sha256": population_sha},
            write,
            read,
        ),
    )
    cost_preflight, _ = publish_sealed_json(
        tmp_path / "cost-preflight.json", {"format": "sealed-cost-fixture-v1"}
    )
    final, final_replay = publish_verified_final_cost(
        tmp_path / "final",
        common_input_path=authority["common_input_path"],
        expected_common_input_sha256=authority["expected_common_input_sha256"],
        cost_preflight_path=cost_preflight.path,
        expected_cost_preflight_sha256=cost_preflight.sha256,
        usage=usage,
    )
    verified = load_verified_final_cost(
        final.path,
        final.sha256,
        final_replay.path,
        final_replay.sha256,
        common_input_path=authority["common_input_path"],
        expected_common_input_sha256=authority["expected_common_input_sha256"],
        cost_preflight_path=cost_preflight.path,
        expected_cost_preflight_sha256=cost_preflight.sha256,
        usage_attestation_path=attestation.path,
        expected_usage_attestation_sha256=attestation.sha256,
        usage_attestation_replay_path=attestation_replay.path,
        expected_usage_attestation_replay_sha256=attestation_replay.sha256,
        lifecycle_authority=authority,
    )
    assert verified.artifact.sha256 == final_replay.sha256
    assert verified.artifact.payload["journal_usage_attestation_sha256"] == attestation.sha256
    assert verified.artifact.payload["common_final_cost"]["question_count"] == 100
    assert verified.artifact.payload["common_final_cost"]["retained_transformer_token_state_bytes"] == 0
    assert verified.artifact.payload["common_final_cost"][
        "responder_complete_envelope_token_cap"
    ] == 8000
    assert verified.artifact.payload["common_final_cost"][
        "judge_complete_envelope_token_cap"
    ] == 9024

    mutated_usage = load_verified_usage_attestation(
        attestation.path,
        attestation.sha256,
        attestation_replay.path,
        attestation_replay.sha256,
        **authority,
    )
    mutated_usage.responder["calls"] = {
        "attempted": 100,
        "completed": 99,
        "failed": 1,
        "retry_attempts": 0,
        "scope": "journaled_request_response_pairs",
    }
    with pytest.raises(Mem0UsageAttestationError, match="changed after strict"):
        publish_verified_final_cost(
            tmp_path / "mutated-capability",
            common_input_path=authority["common_input_path"],
            expected_common_input_sha256=authority["expected_common_input_sha256"],
            cost_preflight_path=cost_preflight.path,
            expected_cost_preflight_sha256=cost_preflight.sha256,
            usage=mutated_usage,
        )

    forged_payload = copy.deepcopy(final.payload)
    forged_payload["token_accounting"]["responder_input_basis"] = "caller_reported"
    _reseal_receipt(forged_payload)
    forged, _ = publish_sealed_json(
        tmp_path / "forged-final.json", forged_payload
    )
    forged_replay, _ = publish_sealed_json(
        tmp_path / "forged-final-replay.json", forged_payload
    )
    with pytest.raises(Mem0UsageAttestationError, match="journal-derived replay"):
        load_verified_final_cost(
            forged.path,
            forged.sha256,
            forged_replay.path,
            forged_replay.sha256,
            common_input_path=authority["common_input_path"],
            expected_common_input_sha256=authority["expected_common_input_sha256"],
            cost_preflight_path=cost_preflight.path,
            expected_cost_preflight_sha256=cost_preflight.sha256,
            usage_attestation_path=attestation.path,
            expected_usage_attestation_sha256=attestation.sha256,
            usage_attestation_replay_path=attestation_replay.path,
            expected_usage_attestation_replay_sha256=attestation_replay.sha256,
            lifecycle_authority=authority,
        )
