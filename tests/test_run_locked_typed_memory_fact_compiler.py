from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import (
    VALIDATION_CONTRACT_FORMAT,
    render_final_messages,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec
import tools.run_locked_typed_memory_fact_compiler as runner
import tools.run_locked_typed_memory_fact_compiler_sparse as sparse


def _sha(value: str) -> str:
    return quote_sha256(value)


def _source(ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    question = (
        f"[Question asked at 2026/05/{ordinal + 1:02d} (Sat) 14:18]\n"
        f"How much did I spend on the Aurora lamp in case {ordinal}?"
    )
    spec = compile_typed_operator_spec(question)
    slot_ids = [row.slot_id for row in spec.required_slots]
    parent = f"protected parent {ordinal}"
    provider_input = {
        "dated_question": question,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent,
            "prediction_sha256": quote_sha256(parent),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": ["H001", "H002"],
        },
        "story_coherence": {
            "group_links": [],
            "incompatible_group_pairs": [],
        },
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "frontier": {
                "available_handle_ids": ["H001", "H002"],
                "closed": False,
                "mode": "bounded",
            },
            "handles": [
                {
                    "group_handle": "G001",
                    "handle_id": "H001",
                    "origin": "map",
                    "provenance_grade": "exact_citation",
                },
                {
                    "group_handle": "G002",
                    "handle_id": "H002",
                    "origin": "direct_pointer",
                    "provenance_grade": "direct_pointer",
                },
            ],
            "items": [
                {
                    "date": "2026-05-03",
                    "entity_key": "Aurora lamp",
                    "handle_ids": ["H001"],
                    "included": True,
                    "kind": "operand",
                    "numeric_value": 30.0,
                    "status": "completed",
                    "summary": (
                        "On May 3, 2026, I completed buying the Aurora lamp "
                        "for 30 USD."
                    ),
                    "supported_slot_ids": slot_ids,
                    "unit": "USD",
                },
                {
                    "handle_ids": ["H002"],
                    "included": True,
                    "kind": "direct",
                    "status": "completed",
                    "summary": "I bought a blue kettle for the kitchen.",
                    "supported_slot_ids": [],
                },
            ],
            "operator_spec": spec.projection(),
        },
    }
    original_messages = list(render_final_messages(provider_input))
    validation = {
        "answer_shape": "direct",
        "by_handle": {"H001": {}, "H002": {}},
        "cardinality": None,
        "comparison_mode": "none",
        "deterministic_execution_advisory": None,
        "format": VALIDATION_CONTRACT_FORMAT,
        "include_proposed": False,
        "operation": "single_supported_fact",
        "operator_spec_receipt_sha256": spec.receipt_sha256,
        "packet_receipt_sha256": _sha(f"packet-{ordinal}"),
        "question_action_concepts": [],
        "question_terms": ["aurora", "lamp"],
        "required_slot_ids": [],
        "required_slots": [],
        "requires_all_slots": True,
        "scalar_validation_advisory": None,
        "temporal_mode": "none",
    }
    plan_body = {
        "allowed_handle_ids": ["H001", "H002"],
        "composition_row_sha256": _sha(f"composition-{ordinal}"),
        "dated_question_sha256": quote_sha256(question),
        "handle_group_by_id": {"H001": "G001", "H002": "G002"},
        "messages": original_messages,
        "messages_sha256": identity_sha256(original_messages),
        "ordinal": ordinal,
        "parent_prediction": parent,
        "preservation_requirements": {
            "by_handle": {
                "H001": {
                    "personalization_terms": [],
                    "specificity_terms": ["aurora"],
                },
                "H002": {
                    "personalization_terms": [],
                    "specificity_terms": ["kettle"],
                },
            },
            "question_required_terms": [],
        },
        "prompt_token_proxy": count_chat_prompt_token_proxy(original_messages),
        "question_id": f"question-{ordinal:03d}",
        "question_sha256": _sha(f"question-{ordinal}"),
        "route_id": "numeric",
        "story_coherence": {"incompatible_group_pairs": []},
        "typed_composition_receipt_sha256": _sha(f"typed-{ordinal}"),
        "validation_contract": validation,
    }
    plan = {
        **plan_body,
        "prompt_row_receipt_sha256": identity_sha256(plan_body),
    }
    composition = {
        "composition_row_sha256": plan["composition_row_sha256"],
        "provider_projection": {"provider_input": provider_input},
    }
    return composition, plan


def _compiler_response(composition: dict[str, Any]) -> str:
    slot_ids = [
        row["slot_id"]
        for row in composition["provider_projection"]["provider_input"]
        ["typed_evidence"]["operator_spec"]["required_slots"]
    ]
    return json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {
                            "handle_id": "H001",
                            "quote": (
                                "completed buying the Aurora lamp for 30 USD"
                            ),
                        }
                    ],
                    "date": "2026-05-03",
                    "entity": "Aurora lamp",
                    "kind": "operand",
                    "numeric_value": 30.0,
                    "slot_ids": slot_ids,
                    "status": "completed",
                    "text": "I completed buying the Aurora lamp for 30 USD.",
                    "unit": "USD",
                }
            ]
        }
    )


def _compiler_result(
    compiler_prompt: dict[str, Any], completion: str
) -> dict[str, Any]:
    compilation, compilation_projection, packet, packet_projection = (
        runner._parse_compilation(compiler_prompt, completion)
    )
    body = {
        "compilation": compilation_projection,
        "compiler_completion": completion,
        "fact_packet": packet_projection,
        "fact_packet_sha256": identity_sha256(packet_projection),
    }
    return {
        **body,
        "compiler_result_row_sha256": identity_sha256(body),
        "_compilation_receipt": compilation.receipt_sha256,
        "_packet_receipt": packet.receipt_sha256,
    }


def test_compiler_prompt_is_parent_free_and_invalid_answer_fallback_is_exact() -> None:
    composition, source_plan = _source(runner.REMAINING_ORDINALS[0])
    compiler_prompt = runner._compiler_prompt_row(composition, source_plan)
    rendered = json.dumps(compiler_prompt["messages"])

    assert source_plan["parent_prediction"] not in rendered
    assert "protected_parent_fallback" not in rendered
    assert (
        compiler_prompt["prompt_token_proxy"]
        + runner.COMPILER_OUTPUT_TOKEN_RESERVE
        <= 8_000
    )

    invalid = _compiler_result(compiler_prompt, "not-json")
    invalid.pop("_compilation_receipt")
    invalid.pop("_packet_receipt")
    answer = runner._answer_prompt_row(compiler_prompt, invalid)

    assert answer["fact_packet_valid"] is False
    assert answer["byte_identical_source_fallback"] is True
    assert answer["messages"] == source_plan["messages"]
    assert answer["messages_sha256"] == source_plan["messages_sha256"]


def test_valid_fact_answer_uses_only_compiled_evidence_and_original_validator() -> None:
    composition, source_plan = _source(runner.REMAINING_ORDINALS[0])
    compiler_prompt = runner._compiler_prompt_row(composition, source_plan)
    result = _compiler_result(compiler_prompt, _compiler_response(composition))
    result.pop("_compilation_receipt")
    result.pop("_packet_receipt")
    answer = runner._answer_prompt_row(compiler_prompt, result)
    provider = json.loads(answer["messages"][1]["content"])

    assert answer["fact_packet_valid"] is True
    assert answer["byte_identical_source_fallback"] is False
    assert answer["compiled_retained_handle_ids"] == ["H001"]
    assert answer["allowed_handle_ids"] == ["H001"]
    assert answer["handle_group_by_id"] == {"H001": "G001"}
    assert set(answer["preservation_requirements"]["by_handle"]) == {"H001"}
    assert set(answer["validation_contract"]["by_handle"]) == {"H001"}
    assert provider["response_schema"]["used_handle_ids"] == ["H001"]
    assert provider["typed_evidence"]["items"][0]["citations"][0]["handle_id"] == "H001"
    assert provider["typed_evidence"]["items"][0]["citations"][0]["quote"] == (
        "completed buying the Aurora lamp for 30 USD"
    )
    assert provider["protected_parent_fallback"]["prediction"] == source_plan[
        "parent_prediction"
    ]


def test_full24_invalid_compilations_preflight_as_byte_exact_source_fallbacks() -> None:
    compiler_prompts = []
    compiler_results = []
    for ordinal in runner.REMAINING_ORDINALS:
        composition, source_plan = _source(ordinal)
        compiler_prompt = runner._compiler_prompt_row(composition, source_plan)
        result = _compiler_result(compiler_prompt, "not-json")
        result.pop("_compilation_receipt")
        result.pop("_packet_receipt")
        compiler_prompts.append(compiler_prompt)
        compiler_results.append(result)
    compiler_preflight = SimpleNamespace(
        sha256="a" * 64,
        payload={
            "source_composition_artifact_sha256": "d" * 64,
            "source_preflight_artifact_sha256": "e" * 64,
        },
    )
    compiler_run = SimpleNamespace(sha256="b" * 64)
    compiler_replay = SimpleNamespace(sha256="c" * 64)

    payload, prompts = runner._answer_preflight_projection(
        compiler_preflight,
        compiler_run,
        compiler_replay,
        compiler_prompts,
        compiler_results,
        model="terra-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=3,
    )
    validated_prompts, rows = runner._validate_answer_preflight(
        SimpleNamespace(payload=payload)
    )

    assert payload["byte_identical_invalid_fallback_count"] == 24
    assert payload["required_authorized_provider_calls"] == 24
    assert payload["observed_max_complete_envelope_tokens"] <= 8_000
    assert prompts == validated_prompts
    assert len(rows) == 24
    assert all(row["byte_identical_source_fallback"] for row in rows)
    assert all(
        row["messages"] == row["source_prompt_plan"]["messages"] for row in rows
    )


def test_materializer_and_run_verifier_reject_uncompiled_original_handle() -> None:
    answer_rows = []
    completions = []
    records = []
    for index, ordinal in enumerate(runner.REMAINING_ORDINALS):
        composition, source_plan = _source(ordinal)
        compiler_prompt = runner._compiler_prompt_row(composition, source_plan)
        compiler_result = _compiler_result(
            compiler_prompt,
            _compiler_response(composition),
        )
        compiler_result.pop("_compilation_receipt")
        compiler_result.pop("_packet_receipt")
        answer = runner._answer_prompt_row(compiler_prompt, compiler_result)
        assert answer["allowed_handle_ids"] == ["H001"]
        answer_rows.append(answer)
        completion = json.dumps(
            {
                "decision": "replace" if index == 0 else "keep_parent",
                "prediction": (
                    "I bought a blue kettle."
                    if index == 0
                    else answer["parent_prediction"]
                ),
                # H002 belongs to the original v3 universe but was not cited
                # into the compiled fact packet.
                "used_handle_ids": ["H002"] if index == 0 else [],
            }
        )
        completions.append(completion)
        records.append(
            SimpleNamespace(
                call_key_sha256=_sha(f"answer-call-{ordinal}"),
                checkpoint_hit=True,
                completion=completion,
                completion_sha256=quote_sha256(completion),
                messages_sha256=answer["messages_sha256"],
                physical_call=False,
                request_journal_sha256=_sha(f"answer-request-{ordinal}"),
                response_journal_sha256=_sha(f"answer-response-{ordinal}"),
            )
        )
    usage = SimpleNamespace(
        checkpoint_hits=24,
        logical_calls=24,
        physical_calls=0,
        unique_calls=24,
    )
    batch_dump = {
        "logical_completions": completions,
        "unique_records": [],
        "usage": {
            "checkpoint_hits": 24,
            "logical_calls": 24,
            "physical_calls": 0,
            "unique_calls": 24,
        },
    }
    batch = SimpleNamespace(
        logical_completions=tuple(completions),
        model_dump=lambda: batch_dump,
        unique_records=tuple(records),
        usage=usage,
    )
    preflight = SimpleNamespace(
        sha256="f" * 64,
        payload={
            "byte_identical_invalid_fallback_count": 0,
            "compiler_preflight_artifact_sha256": "a" * 64,
            "compiler_replay_artifact_sha256": "b" * 64,
            "compiler_run_artifact_sha256": "c" * 64,
            "source_composition_artifact_sha256": "d" * 64,
            "source_preflight_artifact_sha256": "e" * 64,
        },
    )

    payload = runner._answer_materialization_projection(
        preflight,
        tuple(answer_rows),
        batch,
    )
    first = payload["questions"][0]

    assert payload["compiled_handle_authority_rejection_count"] == 1
    assert first["decision"] == "invalid_keep_parent"
    assert first["parse_error_code"] == "unknown_handle"
    assert first["used_handle_ids"] == []
    assert first["prediction"] == answer_rows[0]["parent_prediction"]
    verified = runner._validate_answer_run(
        SimpleNamespace(payload=payload),
        preflight,
        tuple(answer_rows),
    )
    assert len(verified) == 24


@pytest.mark.parametrize("phase", ["compiler", "answer"])
def test_provider_authorization_fails_before_environment_access(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = SimpleNamespace(
        payload={"gateway_url": "http://sealed-gateway"},
        sha256="a" * 64,
    )
    prompts = tuple(
        (
            {"role": "system", "content": "sealed"},
            {"role": "user", "content": json.dumps({"ordinal": ordinal})},
        )
        for ordinal in runner.REMAINING_ORDINALS
    )
    monkeypatch.setattr(
        runner,
        "_read_phase_preflight",
        lambda *_args, **_kwargs: (artifact, prompts, ()),
    )
    monkeypatch.setattr(
        runner,
        "load_dotenv",
        lambda: pytest.fail("authorization reached environment access"),
    )
    args = SimpleNamespace(
        api_key_env="KEY",
        authorized_provider_calls=23,
        enable_provider=True,
        expected_preflight_sha256="a" * 64,
        output_root="unused",
    )
    with pytest.raises(
        runner.LockedTypedFactCompilerError,
        match="exact authorization for 24 calls",
    ):
        runner._provider_phase(
            args,
            preflight_name="unused.json",
            validator=lambda _artifact: ((), ()),
            checkpoint_dir_name="unused",
            run_format="unused",
            phase=phase,
        )


def test_cli_exposes_two_sealed_planes_and_gold_only_judge_seam() -> None:
    parser = runner._parser()
    commands = [
        "compiler-preflight",
        "compiler-provider-run",
        "compiler-materialize",
        "compiler-replay",
        "answer-preflight",
        "answer-provider-run",
        "answer-materialize",
        "answer-replay",
        "judge-preflight",
    ]
    parsed = []
    for command in commands:
        args = [command]
        if command.endswith("provider-run") or command.endswith("materialize"):
            args.extend(["--expected-preflight-sha256", "a" * 64])
        elif command.endswith("replay"):
            args.extend(
                [
                    "--expected-preflight-sha256",
                    "a" * 64,
                    "--expected-run-sha256",
                    "b" * 64,
                ]
            )
        elif command == "answer-preflight":
            args.extend(
                [
                    "--expected-compiler-preflight-sha256",
                    "a" * 64,
                    "--expected-compiler-run-sha256",
                    "b" * 64,
                    "--expected-compiler-replay-sha256",
                    "c" * 64,
                ]
            )
        elif command == "judge-preflight":
            args.extend(
                [
                    "--expected-answer-preflight-sha256",
                    "a" * 64,
                    "--expected-answer-run-sha256",
                    "b" * 64,
                    "--expected-answer-replay-sha256",
                    "c" * 64,
                ]
            )
        parsed.append(parser.parse_args(args).command)

    assert parsed == commands
    assert runner.REMAINING_ORDINALS == tuple(sorted(runner.REMAINING_ORDINALS))
    assert runner.SUBSET_QUESTION_COUNT == 24


def _sparse_preflight_fixture(
    valid_positions: set[int],
) -> tuple[SimpleNamespace, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    compiler_prompts = []
    compiler_results = []
    for position, ordinal in enumerate(runner.REMAINING_ORDINALS):
        composition, source_plan = _source(ordinal)
        compiler_prompt = runner._compiler_prompt_row(composition, source_plan)
        response = (
            _compiler_response(composition)
            if position in valid_positions
            else "not-json"
        )
        result = _compiler_result(compiler_prompt, response)
        result.pop("_compilation_receipt")
        result.pop("_packet_receipt")
        compiler_prompts.append(compiler_prompt)
        compiler_results.append(result)
    rematerialized = SimpleNamespace(
        sha256="a" * 64,
        payload={
            "invalid_packet_count": 24 - len(valid_positions),
            "valid_packet_count": len(valid_positions),
        },
    )
    replay = SimpleNamespace(sha256="b" * 64)
    payload, _prompts = sparse._answer_preflight_projection(
        rematerialized,
        replay,
        compiler_prompts,
        compiler_results,
        model="terra-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=3,
    )
    preflight = SimpleNamespace(sha256="c" * 64, payload=payload)
    _validated_prompts, physical, all_plans = sparse._validate_answer_preflight(
        preflight
    )
    return preflight, physical, all_plans


def _sparse_batch(
    physical_rows: tuple[dict[str, Any], ...],
    *,
    forge_first_uncompiled_handle: bool = False,
) -> SimpleNamespace:
    completions = []
    records = []
    for index, row in enumerate(physical_rows):
        forged = forge_first_uncompiled_handle and index == 0
        completion = json.dumps(
            {
                "decision": "replace" if forged else "keep_parent",
                "prediction": (
                    "I bought a blue kettle."
                    if forged
                    else row["parent_prediction"]
                ),
                "used_handle_ids": ["H002"] if forged else [],
            }
        )
        completions.append(completion)
        records.append(
            SimpleNamespace(
                call_key_sha256=_sha(f"sparse-call-{row['ordinal']}"),
                checkpoint_hit=True,
                completion=completion,
                completion_sha256=quote_sha256(completion),
                messages_sha256=row["messages_sha256"],
                physical_call=False,
                request_journal_sha256=_sha(f"sparse-request-{row['ordinal']}"),
                response_journal_sha256=_sha(f"sparse-response-{row['ordinal']}"),
            )
        )
    count = len(physical_rows)
    usage = SimpleNamespace(
        checkpoint_hits=count,
        logical_calls=count,
        physical_calls=0,
        unique_calls=count,
    )
    batch_dump = {
        "logical_completions": completions,
        "unique_records": [],
        "usage": {
            "checkpoint_hits": count,
            "logical_calls": count,
            "physical_calls": 0,
            "unique_calls": count,
        },
    }
    return SimpleNamespace(
        logical_completions=tuple(completions),
        model_dump=lambda: batch_dump,
        unique_records=tuple(records),
        usage=usage,
    )


def test_sparse_preflight_selects_only_noncontiguous_valid_packets() -> None:
    preflight, physical, all_plans = _sparse_preflight_fixture({1, 23})
    payload = preflight.payload

    assert len(all_plans) == 24
    assert [row["ordinal"] for row in all_plans] == list(
        runner.REMAINING_ORDINALS
    )
    assert [row["ordinal"] for row in physical] == [
        runner.REMAINING_ORDINALS[1],
        runner.REMAINING_ORDINALS[23],
    ]
    assert payload["valid_packet_count"] == 2
    assert payload["invalid_packet_count"] == 22
    assert payload["required_authorized_provider_calls"] == 2
    assert payload["prompt_population"]["logical_prompt_count"] == 2
    assert payload["prompt_population"]["unique_prompt_count"] == 2
    assert all(row["allowed_handle_ids"] == ["H001"] for row in physical)
    assert all(
        set(row["validation_contract"]["by_handle"]) == {"H001"}
        for row in physical
    )


def test_sparse_materialization_merges_selected_and_local_rows_to_full24() -> None:
    preflight, physical, all_plans = _sparse_preflight_fixture({1, 23})
    batch = _sparse_batch(physical)

    payload = sparse._materialization_projection(
        preflight,
        physical,
        all_plans,
        batch,
    )

    assert payload["answer_provider_population_count"] == 2
    assert payload["historical_physical_answer_call_count"] == 2
    assert payload["local_invalid_fallback_count"] == 22
    assert len(payload["completion_batch"]["logical_completions"]) == 2
    assert [row["ordinal"] for row in payload["questions"]] == list(
        runner.REMAINING_ORDINALS
    )
    local = [
        row
        for row in payload["questions"]
        if row.get("provider_call_performed") is False
    ]
    assert len(local) == 22
    assert all(row["decision"] == "keep_parent" for row in local)
    assert all(
        row["prediction_source"]
        == "typed_fact_invalid_packet_local_keep_parent_v3"
        for row in local
    )
    assert len(
        sparse._validate_answer_run(
            SimpleNamespace(payload=payload),
            preflight,
            physical,
            all_plans,
        )
    ) == 24


def test_sparse_rejects_uncompiled_handle_and_replays_deterministically() -> None:
    preflight, physical, all_plans = _sparse_preflight_fixture({1, 23})
    batch = _sparse_batch(physical, forge_first_uncompiled_handle=True)

    first = sparse._materialization_projection(
        preflight,
        physical,
        all_plans,
        batch,
    )
    second = sparse._materialization_projection(
        preflight,
        physical,
        all_plans,
        batch,
    )
    forged_ordinal = physical[0]["ordinal"]
    forged = next(
        row for row in first["questions"] if row["ordinal"] == forged_ordinal
    )

    assert first == second
    assert identity_sha256(first) == identity_sha256(second)
    assert first["compiled_handle_authority_rejection_count"] == 1
    assert forged["decision"] == "invalid_keep_parent"
    assert forged["parse_error_code"] == "unknown_handle"
    assert forged["used_handle_ids"] == []
    assert forged["prediction"] == physical[0]["parent_prediction"]
    assert len(
        sparse._validate_answer_run(
            SimpleNamespace(payload=first),
            preflight,
            physical,
            all_plans,
        )
    ) == 24
