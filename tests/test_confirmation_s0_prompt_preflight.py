from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.recall_guarded_cumulative import (
    CumulativeRetrievalStageReceipt,
)
from tools import confirmation_cumulative_retrieval as cumulative
from tools import confirmation_s0_prompt_preflight as s0
from tools.confirmation_contracts import (
    RUNTIME_POLICY_FORMAT,
    RUNTIME_POLICY_STATUS,
    RuntimePolicy,
    SealedJson,
    _decode_treatment,
    publish_sealed_json,
    validate_runtime_policy,
)
from tools.matched_eval.contracts import EvidenceItem, MemoryPacket, identity_sha256
from tools.matched_eval.renderer import V4_RENDERER_ID, render_memory_packet_for_id
from tools.plan_confirmation_treatment_pipeline import (
    compile_confirmation_pipeline_preflight,
)
from tools.v4_population_firebreak.canonical import canonical_sha256


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sealed(value: dict[str, object], key: str) -> dict[str, object]:
    return {**value, key: identity_sha256(value)}


@dataclass(frozen=True)
class Fixture:
    root: Path
    policy: RuntimePolicy
    treatment: SealedJson
    treatment_preflight: SealedJson
    cumulative: SealedJson
    semantics: tuple[int, ...]

    def kwargs(self) -> dict[str, object]:
        return {
            "runtime_policy_path": self.policy.path,
            "expected_runtime_policy_sha256": self.policy.runtime_policy_sha256,
            "treatment_input_path": self.treatment.path,
            "expected_treatment_input_sha256": self.treatment.sha256,
            "treatment_preflight_path": self.treatment_preflight.path,
            "expected_treatment_preflight_sha256": self.treatment_preflight.sha256,
            "cumulative_retrieval_path": self.cumulative.path,
            "expected_cumulative_retrieval_sha256": self.cumulative.sha256,
        }


def _policy_payload(treatment: dict[str, object]) -> dict[str, object]:
    treatment_policy = {
        "arbitration_priority": [
            "supported_operator_first_numeric",
            "accepted_typed_final_validator_v5_replacement",
            "byte_exact_protected_parent",
        ],
        "confirmation_guards": {
            "confirmation_role_fixed": True,
            "confirmation_tuning_forbidden": True,
            "gold_or_reference_available_during_prediction": False,
            "judge_available_before_all_predictions_freeze": False,
            "policy_change_requires_new_version": True,
            "question_local_gold_blind_routing_only": True,
            "treatment_projection_only_runtime_input": True,
            "validation_artifacts_runtime_use_forbidden": True,
            "validation_ordinals_runtime_use_forbidden": True,
            "validation_question_ids_runtime_use_forbidden": True,
        },
        "confirmation_population_static_root": {
            "dataset_sha256": treatment["dataset_sha256"],
            "split_manifest_sha256": treatment["split_manifest_sha256"],
            "sample_count": treatment["sample_count"],
            "ordered_question_ids_sha256": treatment[
                "ordered_question_ids_sha256"
            ],
            "ordered_normalized_sample_bindings_sha256": treatment[
                "ordered_normalized_sample_bindings_sha256"
            ],
            "ordered_raw_record_bindings_sha256": treatment[
                "ordered_raw_record_bindings_sha256"
            ],
        },
        "format": "memory-condense-policy-v5-r3-treatment-projection-v1",
        "full100_policy_bindings": {"synthetic": True},
        "numeric_frontier_policy": {"population_size_constant": None},
        "policy_id": "policy-v5-r3",
        "responder_runtime": {
            "gateway_url": "https://central-dev.zt:4000/v1",
            "hard_complete_chat_token_cap": 8000,
            "input_token_cap": 7232,
            "max_concurrency": 4,
            "model": "codex_sdk/gpt-5.6-terra",
            "output_token_reserve": 768,
            "retry_count": 0,
        },
        "typed_final_validator_policy_format": "synthetic-validator-v5",
    }
    body = {
        "format": RUNTIME_POLICY_FORMAT,
        "source_policy_manifest_sha256": _digest(
            f"source-policy:{treatment['ordered_question_ids_sha256']}"
        ),
        "status": RUNTIME_POLICY_STATUS,
        "treatment_policy": treatment_policy,
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
    }
    return {**body, "runtime_policy_identity_sha256": canonical_sha256(body)}


def _namespace_by_question(preflight: SealedJson) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for namespace in preflight.payload["namespaces"]:
        for question_id in namespace["question_ids"]:
            result[question_id] = (
                namespace["namespace_id"],
                namespace["namespace_receipt_sha256"],
            )
    return result


def _build_stage_chain(
    *,
    semantic: int,
    question_id: str,
    question: str,
    dated_question: str,
) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    parent_receipt: str | None = None
    selected: list[dict[str, str]] = []
    parent_ids: tuple[str, ...] = ()
    matched_controls = _digest(f"matched:{semantic}")
    for index, stage_id in enumerate(cumulative.STAGE_IDS):
        additions = [
            {
                "evidence_id": (
                    f"evidence-{semantic}"
                    if index == 0
                    else f"derived-{semantic}-{stage_id}"
                ),
                "format": cumulative.EVIDENCE_FORMAT,
                "source_id": f"session-{semantic}",
                "text": (
                    f"Memory {semantic} is value {semantic}."
                    if index == 0
                    else f"{stage_id} evidence for memory {semantic}."
                ),
            }
        ]
        selected.extend(additions)
        evidence = tuple(
            EvidenceItem(
                evidence_id=item["evidence_id"],
                source_id=item["source_id"],
                text=item["text"],
                token_count=count_tokens(item["text"]),
            )
            for item in selected
        )
        packet = MemoryPacket(
            question_id=question_id,
            question_sha256=quote_sha256(question),
            dated_question=dated_question,
            dated_question_sha256=quote_sha256(dated_question),
            stage_id=stage_id,
            protected_evidence=evidence,
        )
        rendered = render_memory_packet_for_id(packet, renderer_id=V4_RENDERER_ID)
        messages = [dict(message) for message in rendered.messages]
        context = "\n".join(
            f"[{number}] {item.text}" for number, item in enumerate(evidence, start=1)
        )
        selected_ids = tuple(item.evidence_id for item in evidence)
        receipt = CumulativeRetrievalStageReceipt(
            stage_id=stage_id,
            matched_controls_sha256=matched_controls,
            method_evidence_sha256=_digest(f"method:{semantic}:{stage_id}"),
            parent_stage_receipt_sha256=parent_receipt,
            parent_evidence_ids=parent_ids,
            selected_evidence_ids=selected_ids,
            added_evidence_ids=selected_ids[len(parent_ids) :],
            admission_status="root" if index == 0 else "added",
            evidence_projection_sha256=_digest(
                f"evidence-projection:{semantic}:{stage_id}"
            ),
            context_sha256=quote_sha256(context),
            prompt_messages_sha256=identity_sha256(messages),
            context_token_proxy=count_tokens(context),
            max_context_token_proxy=7232,
            prompt_token_proxy=count_chat_prompt_token_proxy(messages),
            max_prompt_token_proxy=7232,
            responder_output_token_reserve=768,
        )
        stages.append(
            {
                "evidence": copy.deepcopy(selected),
                "format": cumulative.STAGE_FORMAT,
                "provider_messages": messages,
                "stage_id": stage_id,
                "stage_receipt": asdict(receipt),
            }
        )
        parent_receipt = receipt.receipt_sha256
        parent_ids = selected_ids
    return stages


def _build_fixture(
    root: Path,
    *,
    semantics: tuple[int, ...],
    id_prefix: str,
    namespace_sizes: tuple[int, ...],
) -> Fixture:
    root.mkdir(parents=True)
    question_ids = tuple(
        f"{id_prefix}-{semantic * 31 + 7}" for semantic in semantics
    )
    samples = []
    for question_id, semantic in zip(question_ids, semantics, strict=True):
        samples.append(
            {
                "sample_id": question_id,
                "turns": [
                    ["user", f"Memory {semantic} is value {semantic}."],
                    ["assistant", "Stored."],
                ],
                "turn_source_ids": [f"session-{semantic}", f"session-{semantic}"],
                "turn_created_at": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                "questions": [
                    {
                        "question_id": question_id,
                        "question": f"What is memory {semantic}?",
                        "question_date": "2026-02-01",
                    }
                ],
            }
        )
    treatment_payload = {
        "format": "memory-condense-v4-confirmation-treatment-input-v1",
        "role": "confirmation",
        "dataset_sha256": _digest(f"dataset:{id_prefix}:{semantics}"),
        "split_manifest_sha256": _digest(f"split:{id_prefix}:{semantics}"),
        "sample_count": len(samples),
        "ordered_question_ids_sha256": canonical_sha256(list(question_ids)),
        "ordered_normalized_sample_bindings_sha256": _digest(
            f"normalized:{id_prefix}:{semantics}"
        ),
        "ordered_raw_record_bindings_sha256": _digest(
            f"raw:{id_prefix}:{semantics}"
        ),
        "sanitized_projection_sha256": canonical_sha256(samples),
        "samples": samples,
    }
    treatment, _ = publish_sealed_json(root / "treatment.json", treatment_payload)
    decoded, _ = _decode_treatment(treatment)
    pipeline_payload = compile_confirmation_pipeline_preflight(
        decoded,
        namespace_sizes=namespace_sizes,
    )
    treatment_preflight, _ = publish_sealed_json(
        root / "treatment-preflight.json",
        pipeline_payload,
    )
    policy_artifact, _ = publish_sealed_json(
        root / "policy.json",
        _policy_payload(treatment_payload),
    )
    policy = validate_runtime_policy(policy_artifact, decoded)
    namespaces = _namespace_by_question(treatment_preflight)
    preflight_rows = {
        row["question_id"]: row for row in treatment_preflight.payload["rows"]
    }
    namespace_bindings: dict[str, tuple[str, str]] = {}
    namespace_checkpoints = []
    for index, namespace in enumerate(treatment_preflight.payload["namespaces"]):
        namespace_id = namespace["namespace_id"]
        store_id = _digest(f"namespace-store:{id_prefix}:{index}")
        checkpoint_sha = _digest(f"namespace-checkpoint:{id_prefix}:{index}")
        namespace_bindings[namespace_id] = (store_id, checkpoint_sha)
        namespace_checkpoints.append(
            {
                "checkpoint_receipt_sha256": _digest(
                    f"checkpoint-receipt:{id_prefix}:{index}"
                ),
                "checkpoint_sha256": checkpoint_sha,
                "namespace_id": namespace_id,
                "namespace_store_id": store_id,
                "namespace_work_receipt_sha256": _digest(
                    f"namespace-work:{id_prefix}:{index}"
                ),
            }
        )

    rows = []
    question_receipts = []
    for question_id, semantic, sample in zip(
        question_ids, semantics, samples, strict=True
    ):
        namespace_id, _namespace_receipt = namespaces[question_id]
        namespace_store_id, checkpoint_sha = namespace_bindings[namespace_id]
        question = sample["questions"][0]["question"]
        dated_question = f"[Question asked at 2026-02-01]\n{question}"
        stages = _build_stage_chain(
            semantic=semantic,
            question_id=question_id,
            question=question,
            dated_question=dated_question,
        )
        preflight_row = preflight_rows[question_id]
        question_body = {
            "base_retrieval_receipt_sha256": _digest(
                f"base-retrieval:{id_prefix}:{semantic}"
            ),
            "content_binding_sha256": preflight_row["content_binding_sha256"],
            "dated_question": dated_question,
            "dated_question_sha256": quote_sha256(dated_question),
            "format": cumulative.QUESTION_FORMAT,
            "physical_provider_calls": 0,
            "predecessor_receipt": {
                "format": "synthetic-causal-coverage-predecessor-v1",
                "receipt_sha256": _digest(f"predecessor:{id_prefix}:{semantic}"),
            },
            "question": question,
            "question_id": question_id,
            "question_id_sha256": quote_sha256(question_id),
            "question_sha256": quote_sha256(question),
            "retrieval_receipt": {
                "format": "synthetic-cumulative-retrieval-v1",
                "receipt_sha256": _digest(f"retrieval:{id_prefix}:{semantic}"),
            },
            "row_receipt_sha256": preflight_row["row_receipt_sha256"],
            "stage_ids": list(cumulative.STAGE_IDS),
            "stages": stages,
        }
        question_row = _sealed(question_body, "question_receipt_sha256")
        question_receipts.append(question_row["question_receipt_sha256"])
        rows.append(
            {
                "format": cumulative.MERGED_ROW_FORMAT,
                "namespace_checkpoint_sha256": checkpoint_sha,
                "namespace_id": namespace_id,
                "namespace_store_id": namespace_store_id,
                "question": question_row,
                "source_question_receipt_sha256": question_row[
                    "question_receipt_sha256"
                ],
            }
        )
    workset_identity = _digest(f"workset:{id_prefix}:{semantics}")
    population_body = {
        "dataset_sha256": treatment_payload["dataset_sha256"],
        "format": cumulative.POPULATION_IDENTITY_FORMAT,
        "namespace_store_ids": [
            binding[0] for binding in namespace_bindings.values()
        ],
        "ordered_row_receipt_sha256s": [
            preflight_rows[question_id]["row_receipt_sha256"]
            for question_id in question_ids
        ],
        "preflight_sha256": treatment_preflight.sha256,
        "sanitized_projection_sha256": treatment_payload[
            "sanitized_projection_sha256"
        ],
        "split_manifest_sha256": treatment_payload["split_manifest_sha256"],
        "workset_identity_sha256": workset_identity,
    }
    population_identity = _sealed(
        population_body, "population_identity_sha256"
    )
    cumulative_body = {
        "backend_identity_sha256": _digest(f"backend:{id_prefix}"),
        "format": cumulative.MERGED_FORMAT,
        "freeze_sha256": policy.sha256,
        "gold_loaded": False,
        "namespace_checkpoints": namespace_checkpoints,
        "namespace_count": len(namespace_checkpoints),
        "physical_provider_calls": 0,
        "population_identity": population_identity,
        "population_identity_sha256": population_identity[
            "population_identity_sha256"
        ],
        "preflight_sha256": treatment_preflight.sha256,
        "question_count": len(rows),
        "question_order_sha256": canonical_sha256(question_receipts),
        "question_receipt_sha256s": question_receipts,
        "questions": rows,
        "stage_ids": list(cumulative.STAGE_IDS),
        "workset_identity_sha256": workset_identity,
    }
    cumulative_artifact, _ = publish_sealed_json(
        root / "cumulative.json",
        _sealed(cumulative_body, "merge_receipt_sha256"),
    )
    return Fixture(
        root=root,
        policy=policy,
        treatment=treatment,
        treatment_preflight=treatment_preflight,
        cumulative=cumulative_artifact,
        semantics=semantics,
    )


def _publish_cumulative_mutation(
    fixture: Fixture,
    *,
    name: str,
    mutate: object,
) -> SealedJson:
    value = copy.deepcopy(fixture.cumulative.payload)
    mutate(value)
    body = {
        key: child
        for key, child in value.items()
        if key != "merge_receipt_sha256"
    }
    value["merge_receipt_sha256"] = identity_sha256(body)
    artifact, _ = publish_sealed_json(fixture.root / name, value)
    return artifact


def test_compiles_exact_arbitrary_n_inert_terra_preflight(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "exact",
        semantics=(0, 1, 2, 3),
        id_prefix="unfamiliar",
        namespace_sizes=(2, 2),
    )
    population = s0.load_generic_matched_s0_population(**fixture.kwargs())
    preflight = population.preflight_projection()

    assert population.question_count == 4
    assert preflight["execution"] == {
        "logical_prompt_count": 4,
        "unique_prompt_count": 4,
        "would_call_count": 4,
        "would_call_count_status": "exact",
        "count_basis": "one-unique-terra-prompt-per-complete-s0-row",
        "physical_provider_calls": 0,
        "provider_execution_available": False,
        "authorization_released": False,
        "retained_request_token_state_bytes": 0,
    }
    assert preflight["runtime"] == {
        "gateway_url": "https://central-dev.zt:4000/v1",
        "hard_complete_chat_token_cap": 8000,
        "input_token_cap": 7232,
        "max_concurrency": 4,
        "model": "codex_sdk/gpt-5.6-terra",
        "output_token_reserve": 768,
        "retry_count": 0,
    }
    assert len(preflight["ordered_rows"]) == 4
    assert "terra_messages" not in preflight
    for source, row in zip(population.rows, preflight["ordered_rows"], strict=True):
        provider_input = row["provider_input"]
        assert provider_input["format"] == (
            "memory-condense-confirmation-terra-provider-input-v1"
        )
        assert provider_input["messages"] == [
            dict(message) for message in source.rendered_prompt.messages
        ]
        assert provider_input["messages_sha256"] == row["messages_sha256"]
        body = {
            key: value
            for key, value in provider_input.items()
            if key != "provider_input_receipt_sha256"
        }
        assert provider_input["provider_input_receipt_sha256"] == identity_sha256(
            body
        )


def test_preflight_publish_and_replay_are_byte_identical(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "replay",
        semantics=(0, 1, 2),
        id_prefix="replay",
        namespace_sizes=(2, 1),
    )
    source, created = s0.publish_confirmation_s0_preflight(
        fixture.root / "s0-preflight.json",
        **fixture.kwargs(),
    )
    replay, replay_created = s0.replay_confirmation_s0_preflight(
        source_preflight_path=source.path,
        expected_source_preflight_sha256=source.sha256,
        replay_output_path=fixture.root / "s0-preflight-replay.json",
        **fixture.kwargs(),
    )
    assert created is True
    assert replay_created is True
    assert replay.sha256 == source.sha256
    assert replay.payload == source.payload


def test_label_bearing_cumulative_field_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "gold",
        semantics=(0, 1),
        id_prefix="gold",
        namespace_sizes=(2,),
    )

    def mutate(value: dict[str, object]) -> None:
        value["gold_answer"] = "forbidden"

    changed = _publish_cumulative_mutation(
        fixture,
        name="gold-bearing.json",
        mutate=mutate,
    )
    kwargs = fixture.kwargs()
    kwargs["cumulative_retrieval_path"] = changed.path
    kwargs["expected_cumulative_retrieval_sha256"] = changed.sha256
    with pytest.raises(ValueError, match="gold-bearing field"):
        s0.load_generic_matched_s0_population(**kwargs)


@pytest.mark.parametrize("mutation", ["missing", "reordered"])
def test_incomplete_or_reordered_rows_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_fixture(
        tmp_path / mutation,
        semantics=(0, 1, 2),
        id_prefix=mutation,
        namespace_sizes=(2, 1),
    )

    def mutate(value: dict[str, object]) -> None:
        if mutation == "missing":
            value["questions"] = value["questions"][:-1]
        else:
            value["questions"] = list(reversed(value["questions"]))

    changed = _publish_cumulative_mutation(
        fixture,
        name=f"{mutation}-rows.json",
        mutate=mutate,
    )
    kwargs = fixture.kwargs()
    kwargs["cumulative_retrieval_path"] = changed.path
    kwargs["expected_cumulative_retrieval_sha256"] = changed.sha256
    with pytest.raises(s0.ConfirmationS0PreflightError, match="incomplete|order"):
        s0.load_generic_matched_s0_population(**kwargs)


def test_renumbering_and_permutation_do_not_change_question_local_prompts(
    tmp_path: Path,
) -> None:
    original = _build_fixture(
        tmp_path / "original",
        semantics=(0, 1, 2, 3),
        id_prefix="alpha",
        namespace_sizes=(2, 2),
    )
    renamed = _build_fixture(
        tmp_path / "renamed",
        semantics=(0, 1, 2, 3),
        id_prefix="zeta",
        namespace_sizes=(2, 2),
    )
    permuted = _build_fixture(
        tmp_path / "permuted",
        semantics=(2, 0, 3, 1),
        id_prefix="permuted",
        namespace_sizes=(2, 2),
    )
    populations = [
        s0.load_generic_matched_s0_population(**fixture.kwargs())
        for fixture in (original, renamed, permuted)
    ]

    def prompt_by_question(population: s0.GenericMatchedS0Population) -> dict[str, str]:
        return {
            row.packet.dated_question: row.rendered_prompt.messages_sha256
            for row in population.rows
        }

    assert prompt_by_question(populations[0]) == prompt_by_question(populations[1])
    assert prompt_by_question(populations[0]) == prompt_by_question(populations[2])
    assert [row.packet.dated_question for row in populations[2].rows] == [
        populations[0].rows[index].packet.dated_question
        for index in (2, 0, 3, 1)
    ]


def test_source_store_cannot_cross_namespace_boundary(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "namespace",
        semantics=(0, 1, 2, 3),
        id_prefix="namespace",
        namespace_sizes=(2, 2),
    )

    def mutate(value: dict[str, object]) -> None:
        rows = value["questions"]
        target = rows[2]
        target["namespace_store_id"] = rows[0]["namespace_store_id"]

    changed = _publish_cumulative_mutation(
        fixture,
        name="crossed-source.json",
        mutate=mutate,
    )
    kwargs = fixture.kwargs()
    kwargs["cumulative_retrieval_path"] = changed.path
    kwargs["expected_cumulative_retrieval_sha256"] = changed.sha256
    with pytest.raises(
        s0.ConfirmationS0PreflightError,
        match="escapes its authenticated namespace|crosses namespace",
    ):
        s0.load_generic_matched_s0_population(**kwargs)


def test_runtime_policy_is_a_distinct_source_bound_sanitized_artifact(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(
        tmp_path / "runtime-policy",
        semantics=(0, 1),
        id_prefix="runtime-policy",
        namespace_sizes=(2,),
    )

    assert fixture.policy.runtime_policy_sha256 != fixture.policy.sha256
    assert fixture.policy.payload["source_policy_manifest_sha256"] == fixture.policy.sha256
    assert not {
        "validation_lineage",
        "validation_result",
        "confirmation_population",
        "manifest_identity_sha256",
    } & set(fixture.policy.payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_answer", "must never cross", "evaluator material"),
        ("validation_result", {"correct": 1}, "forbidden field"),
        ("artifact_path", "validation/results.json", "filesystem field"),
    ],
)
def test_runtime_policy_recursively_rejects_evaluator_and_path_material(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _build_fixture(
        tmp_path / field,
        semantics=(0,),
        id_prefix=field,
        namespace_sizes=(1,),
    )
    payload = copy.deepcopy(fixture.policy.payload)
    treatment_policy = dict(payload["treatment_policy"])
    bindings = dict(treatment_policy["full100_policy_bindings"])
    bindings[field] = value
    treatment_policy["full100_policy_bindings"] = bindings
    payload["treatment_policy"] = treatment_policy
    payload["treatment_projection_sha256"] = canonical_sha256(treatment_policy)
    body = {
        key: item
        for key, item in payload.items()
        if key != "runtime_policy_identity_sha256"
    }
    payload["runtime_policy_identity_sha256"] = canonical_sha256(body)
    artifact, _ = publish_sealed_json(tmp_path / f"bad-{field}.json", payload)
    treatment, _ = _decode_treatment(fixture.treatment)

    with pytest.raises(ValueError, match=message):
        validate_runtime_policy(artifact, treatment)


def _all_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    actions = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                actions.extend(_all_actions(child))
    return actions


def test_cli_has_no_provider_execution_surface() -> None:
    destinations = {action.dest for action in _all_actions(s0.build_parser())}
    assert not destinations & {
        "api_key",
        "authorized_provider_calls",
        "enable_provider",
        "execute",
        "provider",
        "retry",
        "token",
    }
    source = Path(s0.__file__).read_text(encoding="utf-8").casefold()
    assert "import litellm" not in source
    assert "import openai" not in source
