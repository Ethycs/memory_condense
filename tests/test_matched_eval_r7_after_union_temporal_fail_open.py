from __future__ import annotations

import copy
import hashlib
import json

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.after_union_fact_closure import (
    SealedLeafDisposition,
    SelectedHLeaf,
    build_after_union_selection,
)
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.r7_after_union_a1 import (
    DISPOSITIONS_FORMAT,
    FORMAT as A1_FORMAT,
    R7AfterUnionA1Error,
    _disposition_lookup,
)
from tools.matched_eval.r7_a1a_raw_retained_answer import (
    R7A1ARawRetainedError,
    _dispositions as a1a_dispositions,
    build_r7_a1a_raw_retained_payload,
)
from tools.matched_eval.r7_after_union_temporal_fail_open import (
    EFFECTIVE_DISPOSITIONS_FORMAT,
    POLICY_ID,
    TemporalFailOpenContractError,
    build_temporal_fail_open_artifacts,
    validate_temporal_fail_open_effective_artifact,
)


def _sha(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sealed(body: dict[str, object], receipt_key: str) -> dict[str, object]:
    return {**body, receipt_key: identity_sha256(body)}


def _fixtures(
    question: str = (
        "[Question asked at 2023/03/25 (Sat) 18:26]\n"
        "What kitchen appliance did I buy 10 days ago?"
    ),
    *,
    leaf_dates: tuple[str, ...] = (
        "2023-03-15",
        "2023-05-25",
        "2023-03-15",
    ),
    base_dispositions: tuple[str, ...] = (
        "definitely_irrelevant",
        "definitely_irrelevant",
        "relevant",
    ),
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    question_sha = quote_sha256(question)
    source_sha = "1" * 64
    source_replay_sha = source_sha
    selected_leaves: list[SelectedHLeaf] = []
    for index, event_date in enumerate(leaf_dates, start=1):
        selected_leaves.append(
            SelectedHLeaf(
                f"H{index:03d}",
                f"G{index:03d}",
                f"selected memory leaf {index}",
                f"{index + 2}" * 64,
                boundary_labels=(
                    f"group:G{index:03d}",
                    f"date:{event_date}",
                ),
            )
        )
    provisional_dispositions = [
        SealedLeafDisposition(
            leaf.handle_id,
            leaf.receipt_sha256,
            question_sha,
            "r7-a1-deterministic-all-uncertain-v1",
            "uncertain",
        )
        for leaf in selected_leaves
    ]
    semantic = build_after_union_selection(
        question,
        selected_leaves,
        provisional_dispositions,
    ).projection()
    leaves = [leaf.projection() for leaf in selected_leaves]
    selected_sha = identity_sha256(leaves)
    leaf_ids = [leaf["handle_id"] for leaf in leaves]
    classifier_messages = [
        {"content": "classify every supplied leaf", "role": "system"},
        {"content": "sealed classifier input", "role": "user"},
    ]
    request = _sealed(
        {
            "answer_output_token_reserve": 768,
            "boundary_labels_for_scheduling_only": list(
                dict.fromkeys(
                    label
                    for leaf in selected_leaves
                    for label in leaf.boundary_labels
                )
            ),
            "classifier_output_token_reserve": 1024,
            "format": f"{A1_FORMAT}-classifier-request-v1",
            "hard_total_token_cap": 8000,
            "leaf_handle_ids": leaf_ids,
            "messages": classifier_messages,
            "payload_class": "after_union_leaf_relevance_strict_json_v1",
            "prompt_token_proxy": 10,
            "question_sha256": question_sha,
            "selected_union_population_sha256": selected_sha,
            "shard_id": f"C{question_sha[:12]}-000",
            "shard_population_sha256": identity_sha256(leaf_ids),
            "topic_labels_for_scheduling_only": [],
            "topic_labels_have_exclusion_authority": False,
        },
        "request_sha256",
    )
    request_sha = request["request_sha256"]
    question_body = {
        "actionable_compiler_request_count": 0,
        "classifier_request_count": 1,
        "classifier_request_population_sha256": identity_sha256([request_sha]),
        "classifier_requests": [request],
        "compiler_request_count": 0,
        "compiler_request_results": [],
        "compiler_requests": [],
        "compiler_workload_status": "provisional_fail_open_preview",
        "dated_question": question,
        "dated_question_sha256": question_sha,
        "disposition_counts": {
            "definitely_irrelevant": 0,
            "relevant": 0,
            "uncertain": len(leaves),
        },
        "fact_closure": {},
        "format": f"{A1_FORMAT}-question-v1",
        "hard_total_token_cap": 8000,
        "missing_classifier_request_sha256s": [request_sha],
        "missing_compiler_request_sha256s": [],
        "operator_execution": None,
        "operator_obligations": [],
        "operator_packet": None,
        "operator_packet_error": "no_compiled_facts",
        "operator_spec": {},
        "output_token_reserve": 768,
        "provider_calls_performed_by_core": 0,
        "question_id": "test-question-1",
        "question_sha256": question_sha,
        "request_population_sha256": identity_sha256([]),
        "retained_transformer_token_state_bytes": 0,
        "selected_leaf_count": len(leaves),
        "selected_population_sha256": selected_sha,
        "semantic_selection": semantic,
        "topic_labels_have_exclusion_authority": False,
        "union_population_built_before_exclusion": True,
    }
    a1_question = _sealed(question_body, "question_receipt_sha256")
    a1_body = {
        "actionable_compiler_request_count": 0,
        "classifier_payload_class": "after_union_leaf_relevance_strict_json_v1",
        "classifier_request_count": 1,
        "classifier_request_population_sha256": identity_sha256([request_sha]),
        "compiler_output_artifact_sha256": None,
        "compiler_payload_class": "typed_fact_compiler_strict_json_v1",
        "compiler_request_count": 0,
        "compiler_workload_status": "provisional_fail_open_pending_classifier",
        "construction_status": (
            "preflight_external_classification_then_compilation_required"
        ),
        "disposition_artifact_sha256": None,
        "disposition_classifier_id": "r7-a1-deterministic-all-uncertain-v1",
        "expected_question_count": 1,
        "format": A1_FORMAT,
        "gold_loaded": False,
        "hard_total_token_cap": 8000,
        "max_leaves_per_classifier_shard": 48,
        "max_leaves_per_compiler_shard": 8,
        "missing_classifier_call_count": 1,
        "missing_classifier_request_sha256s": [request_sha],
        "missing_compiler_call_count": 0,
        "missing_external_call_count": 1,
        "missing_external_request_sha256s": [request_sha],
        "operator_obligations_closed": False,
        "output_token_reserve": 768,
        "provider_calls_performed_by_core": 0,
        "question_count": 1,
        "question_population_sha256": identity_sha256(
            [a1_question["question_receipt_sha256"]]
        ),
        "questions": [a1_question],
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": {
            "benchmark_fields_loaded": False,
            "ordinal_routing_enabled": False,
            "protected_parent_loaded": False,
            "semantic_atom_manifest_loaded": False,
            "source_allowlist_loaded": False,
            "topic_labels_have_exclusion_authority": False,
        },
        "selected_leaf_count": len(leaves),
        "selected_population_sha256": identity_sha256([selected_sha]),
        "selected_populations_resolved": False,
        "source_artifact_sha256": source_sha,
        "source_replay_artifact_sha256": source_replay_sha,
        "union_before_exclusion": True,
    }
    a1 = _sealed(a1_body, "construction_identity_sha256")
    a1_replay = copy.deepcopy(a1)
    a1_sha = _sha(a1)

    disposition_rows = [
        {
            "disposition": disposition,
            "handle_id": leaf["handle_id"],
            "leaf_receipt_sha256": leaf["receipt_sha256"],
        }
        for leaf, disposition in zip(leaves, base_dispositions, strict=True)
    ]
    output = {
        "leaf_dispositions": [
            {
                "disposition": row["disposition"],
                "handle_id": row["handle_id"],
            }
            for row in disposition_rows
        ]
    }
    completion = json.dumps(output, sort_keys=True, separators=(",", ":"))
    response_body = {
        "call_key_sha256": "a" * 64,
        "classifier_output": completion,
        "classifier_output_sha256": quote_sha256(completion),
        "dispositions": disposition_rows,
        "format": (
            "memory-condense-r7-after-union-a1-classifier-lifecycle-v1-"
            "response-row-v1"
        ),
        "leaf_bindings": [
            {
                "handle_id": row["handle_id"],
                "leaf_receipt_sha256": row["leaf_receipt_sha256"],
            }
            for row in disposition_rows
        ],
        "messages_sha256": identity_sha256(classifier_messages),
        "question_sha256": question_sha,
        "request_journal_sha256": "b" * 64,
        "request_sha256": request_sha,
        "response_journal_sha256": "c" * 64,
        "selected_union_population_sha256": selected_sha,
        "source_artifact_sha256": source_sha,
    }
    response = _sealed(response_body, "response_row_receipt_sha256")
    disposition_question = {
        "classifier_request_sha256s": [request_sha],
        "dispositions": disposition_rows,
        "question_sha256": question_sha,
        "selected_union_population_sha256": selected_sha,
    }
    dispositions = {
        "a1_construction_artifact_sha256": a1_sha,
        "a1_replay_artifact_sha256": a1_sha,
        "classifier_id": "test-generic-classifier-v1",
        "classifier_request_population_sha256": a1[
            "classifier_request_population_sha256"
        ],
        "completion_runtime_identity_sha256": "4" * 64,
        "derived_provider_call_count": 1,
        "disposition_population_sha256": identity_sha256(
            [disposition_question]
        ),
        "format": DISPOSITIONS_FORMAT,
        "journal_owner_identity_sha256": "5" * 64,
        "lifecycle_format": (
            "memory-condense-r7-after-union-a1-classifier-lifecycle-v1"
        ),
        "model": "test-model",
        "model_prompt_population_sha256": "6" * 64,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": "7" * 64,
        "prompt_population_sha256": "8" * 64,
        "provider_calls_performed_by_core": 0,
        "question_count": 1,
        "questions": [disposition_question],
        "response_population_sha256": identity_sha256(
            [response["response_row_receipt_sha256"]]
        ),
        "responses": [response],
        "release_authorization_artifact_sha256": "9" * 64,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": {
            "gold_loaded": False,
            "ordinal_routing_enabled": False,
            "protected_parent_loaded": False,
            "reference_loaded": False,
            "semantic_atom_manifest_loaded": False,
            "source_allowlist_loaded": False,
        },
        "source_artifact_sha256": source_sha,
        "source_replay_artifact_sha256": source_replay_sha,
    }
    disposition_replay = copy.deepcopy(dispositions)
    return a1, a1_replay, dispositions, disposition_replay


def _build(
    a1: dict[str, object],
    a1_replay: dict[str, object],
    dispositions: dict[str, object],
    disposition_replay: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    return build_temporal_fail_open_artifacts(
        a1,
        _sha(a1),
        a1_replay,
        _sha(a1_replay),
        dispositions,
        _sha(dispositions),
        disposition_replay,
        _sha(disposition_replay),
    )


def _validate_effective(
    effective: dict[str, object],
    fixtures: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
    *,
    replay: dict[str, object] | None = None,
) -> dict[str, object]:
    a1, a1_replay, dispositions, disposition_replay = fixtures
    exact_replay = effective if replay is None else replay
    return validate_temporal_fail_open_effective_artifact(
        effective,
        _sha(effective),
        exact_replay,
        _sha(exact_replay),
        a1,
        _sha(a1),
        a1_replay,
        _sha(a1_replay),
        dispositions,
        _sha(dispositions),
        disposition_replay,
        _sha(disposition_replay),
    )


def _rebind_a1_parents(
    a1: dict[str, object],
    dispositions: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    body = {
        key: value
        for key, value in a1.items()
        if key != "construction_identity_sha256"
    }
    rebound_a1 = _sealed(body, "construction_identity_sha256")
    rebound_dispositions = copy.deepcopy(dispositions)
    rebound_dispositions["a1_construction_artifact_sha256"] = _sha(rebound_a1)
    rebound_dispositions["a1_replay_artifact_sha256"] = _sha(rebound_a1)
    rebound_dispositions["source_artifact_sha256"] = rebound_a1[
        "source_artifact_sha256"
    ]
    rebound_dispositions["source_replay_artifact_sha256"] = rebound_a1[
        "source_replay_artifact_sha256"
    ]
    return rebound_a1, rebound_dispositions


def test_exact_target_date_can_only_fail_open() -> None:
    fixtures = _fixtures()
    effective, report = _build(*fixtures)
    rows = effective["questions"][0]["effective_dispositions"]  # type: ignore[index]

    assert [row["effective_disposition"] for row in rows] == [  # type: ignore[index]
        "unresolved",
        "definitely_irrelevant",
        "relevant",
    ]
    assert [row["base_disposition"] for row in rows] == [  # type: ignore[index]
        "definitely_irrelevant",
        "definitely_irrelevant",
        "relevant",
    ]
    assert effective["format"] == EFFECTIVE_DISPOSITIONS_FORMAT
    assert effective["temporal_fail_open_override_count"] == 1
    assert report["override_count"] == 1
    assert report["protected_selected_leaf_count"] == 2


def test_effective_overlay_never_rewrites_or_copies_provider_responses() -> None:
    a1, a1_replay, dispositions, disposition_replay = _fixtures()
    effective, _report = _build(a1, a1_replay, dispositions, disposition_replay)

    assert "responses" not in effective
    assert "classifier_id" not in effective
    assert effective["base_disposition_artifact_sha256"] == _sha(dispositions)
    assert effective["base_disposition_replay_artifact_sha256"] == _sha(
        disposition_replay
    )
    assert effective["effective_classifier_id"] == (
        f"test-generic-classifier-v1+{POLICY_ID}"
    )


def test_non_temporal_question_leaves_effective_dispositions_unchanged() -> None:
    fixtures = _fixtures(
        "[Question asked at 2023/03/25 (Sat) 18:26]\nWhat appliance did I buy?"
    )
    effective, report = _build(*fixtures)
    rows = effective["questions"][0]["effective_dispositions"]  # type: ignore[index]

    assert all(row["reason"] == "unchanged" for row in rows)  # type: ignore[index]
    assert all(
        row["base_disposition"] == row["effective_disposition"] for row in rows  # type: ignore[index]
    )
    assert report["override_count"] == 0


def test_lookback_window_includes_both_bounds_but_not_older_leaf() -> None:
    fixtures = _fixtures(
        "[Question asked at 2023/05/30 (Tue) 18:26]\n"
        "What did I acquire in the last month?",
        leaf_dates=("2023-05-30", "2023-04-29", "2023-04-28"),
        base_dispositions=(
            "definitely_irrelevant",
            "definitely_irrelevant",
            "definitely_irrelevant",
        ),
    )
    effective, _report = _build(*fixtures)
    rows = effective["questions"][0]["effective_dispositions"]  # type: ignore[index]

    assert [row["effective_disposition"] for row in rows] == [  # type: ignore[index]
        "unresolved",
        "unresolved",
        "definitely_irrelevant",
    ]


def test_construction_is_byte_deterministic_and_does_not_mutate_inputs() -> None:
    fixtures = _fixtures()
    before = copy.deepcopy(fixtures)
    first = _build(*fixtures)
    second = _build(*fixtures)

    assert canonical_json_bytes(first[0]) == canonical_json_bytes(second[0])
    assert canonical_json_bytes(first[1]) == canonical_json_bytes(second[1])
    assert fixtures == before
    assert first[1]["effective_disposition_artifact_sha256"] == _sha(first[0])


@pytest.mark.parametrize(
    "target,field",
    [
        ("a1", "benchmark_fields_loaded"),
        ("a1", "ordinal_routing_enabled"),
        ("a1", "protected_parent_loaded"),
        ("a1", "semantic_atom_manifest_loaded"),
        ("a1", "source_allowlist_loaded"),
        ("dispositions", "gold_loaded"),
        ("dispositions", "ordinal_routing_enabled"),
        ("dispositions", "protected_parent_loaded"),
        ("dispositions", "reference_loaded"),
        ("dispositions", "semantic_atom_manifest_loaded"),
        ("dispositions", "source_allowlist_loaded"),
    ],
)
def test_every_forbidden_firewall_flag_fails_closed(
    target: str, field: str
) -> None:
    a1, a1_replay, dispositions, disposition_replay = _fixtures()
    if target == "a1":
        a1["runtime_firewall"][field] = True  # type: ignore[index]
        a1_replay = copy.deepcopy(a1)
        changed_a1_sha = _sha(a1)
        dispositions["a1_construction_artifact_sha256"] = changed_a1_sha
        dispositions["a1_replay_artifact_sha256"] = changed_a1_sha
        disposition_replay = copy.deepcopy(dispositions)
    else:
        dispositions["runtime_firewall"][field] = True  # type: ignore[index]
        disposition_replay = copy.deepcopy(dispositions)

    with pytest.raises((TemporalFailOpenContractError, ValueError)):
        _build(a1, a1_replay, dispositions, disposition_replay)


@pytest.mark.parametrize(
    "mutation",
    [
        "a1_replay_differs",
        "disposition_replay_differs",
        "foreign_a1_binding",
        "foreign_request",
        "foreign_population",
        "foreign_leaf_receipt",
        "invalid_disposition",
        "reordered_dispositions",
        "stale_disposition_population",
        "response_disposition_mismatch",
        "wrong_question_count",
    ],
)
def test_tampered_replay_population_or_provider_linkage_fails_closed(
    mutation: str,
) -> None:
    a1, a1_replay, dispositions, disposition_replay = _fixtures()
    if mutation == "a1_replay_differs":
        a1_replay["question_count"] = 2
    elif mutation == "disposition_replay_differs":
        disposition_replay["question_count"] = 2
    elif mutation == "foreign_a1_binding":
        dispositions["a1_construction_artifact_sha256"] = "8" * 64
    elif mutation == "foreign_request":
        dispositions["questions"][0]["classifier_request_sha256s"] = ["8" * 64]  # type: ignore[index]
        dispositions["disposition_population_sha256"] = identity_sha256(
            dispositions["questions"]
        )
    elif mutation == "foreign_population":
        dispositions["questions"][0]["selected_union_population_sha256"] = "8" * 64  # type: ignore[index]
        dispositions["disposition_population_sha256"] = identity_sha256(
            dispositions["questions"]
        )
    elif mutation == "foreign_leaf_receipt":
        dispositions["questions"][0]["dispositions"][0]["leaf_receipt_sha256"] = "8" * 64  # type: ignore[index]
        dispositions["disposition_population_sha256"] = identity_sha256(
            dispositions["questions"]
        )
    elif mutation == "invalid_disposition":
        dispositions["questions"][0]["dispositions"][0]["disposition"] = "bogus"  # type: ignore[index]
        dispositions["disposition_population_sha256"] = identity_sha256(
            dispositions["questions"]
        )
    elif mutation == "reordered_dispositions":
        dispositions["questions"][0]["dispositions"].reverse()  # type: ignore[index]
        dispositions["disposition_population_sha256"] = identity_sha256(
            dispositions["questions"]
        )
    elif mutation == "stale_disposition_population":
        dispositions["disposition_population_sha256"] = "8" * 64
    elif mutation == "response_disposition_mismatch":
        dispositions["responses"][0]["dispositions"][0]["disposition"] = "relevant"  # type: ignore[index]
        body = {
            key: value
            for key, value in dispositions["responses"][0].items()  # type: ignore[index]
            if key != "response_row_receipt_sha256"
        }
        dispositions["responses"][0]["response_row_receipt_sha256"] = identity_sha256(body)  # type: ignore[index]
        dispositions["response_population_sha256"] = identity_sha256(
            [dispositions["responses"][0]["response_row_receipt_sha256"]]  # type: ignore[index]
        )
    elif mutation == "wrong_question_count":
        dispositions["question_count"] = 2

    if mutation not in {"a1_replay_differs", "disposition_replay_differs"}:
        disposition_replay = copy.deepcopy(dispositions)

    with pytest.raises((TemporalFailOpenContractError, ValueError)):
        _build(a1, a1_replay, dispositions, disposition_replay)


def test_impossible_boundary_date_fails_closed() -> None:
    fixtures = _fixtures(
        leaf_dates=("2023-02-31", "2023-05-25", "2023-03-15")
    )
    with pytest.raises((TemporalFailOpenContractError, ValueError)):
        _build(*fixtures)


def test_effective_pair_is_rederived_from_all_four_parent_artifacts() -> None:
    fixtures = _fixtures()
    effective, _report = _build(*fixtures)

    assert _validate_effective(effective, fixtures) == effective


def test_a1a_effective_consumer_requires_replay_and_base_pair() -> None:
    fixtures = _fixtures()
    a1, _a1_replay, _dispositions, _disposition_replay = fixtures
    effective, _report = _build(*fixtures)

    with pytest.raises(R7A1ARawRetainedError, match="require A1/effective/base"):
        build_r7_a1a_raw_retained_payload(
            a1,
            _sha(a1),
            _sha(a1),
            effective,
            _sha(effective),
            expected_question_count=1,
        )


def test_a1a_effective_consumer_rederives_and_accepts_exact_overlay() -> None:
    fixtures = _fixtures()
    a1, a1_replay, dispositions, disposition_replay = fixtures
    effective, _report = _build(*fixtures)

    payload = build_r7_a1a_raw_retained_payload(
        a1,
        _sha(a1),
        _sha(a1_replay),
        effective,
        _sha(effective),
        a1_preflight_replay_payload=a1_replay,
        disposition_replay_payload=effective,
        disposition_replay_artifact_sha256=_sha(effective),
        base_disposition_payload=dispositions,
        base_disposition_artifact_sha256=_sha(dispositions),
        base_disposition_replay_payload=disposition_replay,
        base_disposition_replay_artifact_sha256=_sha(disposition_replay),
        expected_question_count=1,
    )

    assert payload["density_totals"]["fixed_union_leaf_count"] == 3
    assert payload["density_totals"]["retained_leaf_count"] == 2
    assert payload["density_totals"]["pruned_leaf_count"] == 1


def test_forged_date_inapplicable_fail_open_transition_is_rejected() -> None:
    fixtures = _fixtures()
    effective, _report = _build(*fixtures)
    forged = copy.deepcopy(effective)
    question = forged["questions"][0]  # type: ignore[index]
    transition = question["effective_dispositions"][1]  # type: ignore[index]
    transition["effective_disposition"] = "unresolved"
    transition["reason"] = "question_derived_temporal_target_match"
    transition_body = {
        key: value
        for key, value in transition.items()
        if key != "transition_receipt_sha256"
    }
    transition["transition_receipt_sha256"] = identity_sha256(transition_body)
    question["effective_disposition_population_sha256"] = identity_sha256(
        question["effective_dispositions"]
    )
    question_body = {
        key: value
        for key, value in question.items()
        if key != "question_effective_disposition_receipt_sha256"
    }
    question["question_effective_disposition_receipt_sha256"] = identity_sha256(
        question_body
    )
    forged["effective_disposition_population_sha256"] = identity_sha256(
        forged["questions"]
    )
    forged["temporal_fail_open_override_count"] += 1  # type: ignore[operator]

    with pytest.raises(TemporalFailOpenContractError, match="re-derived"):
        _validate_effective(forged, fixtures)


def test_effective_replay_payload_must_be_byte_identical() -> None:
    fixtures = _fixtures()
    effective, _report = _build(*fixtures)
    replay = copy.deepcopy(effective)
    replay["temporal_fail_open_override_count"] = 99

    with pytest.raises(TemporalFailOpenContractError, match="byte-identical"):
        _validate_effective(effective, fixtures, replay=replay)


@pytest.mark.parametrize(
    "missing",
    [
        "gold_loaded",
        "union_before_exclusion",
        "construction_status",
        "compiler_workload_status",
        "question_population_sha256",
        "selected_population_sha256",
        "construction_identity_sha256",
    ],
)
def test_incomplete_a1_contract_fails_closed(missing: str) -> None:
    a1, _a1_replay, dispositions, _disposition_replay = _fixtures()
    del a1[missing]
    if missing != "construction_identity_sha256":
        a1, dispositions = _rebind_a1_parents(a1, dispositions)

    with pytest.raises((TemporalFailOpenContractError, ValueError)):
        _build(a1, copy.deepcopy(a1), dispositions, copy.deepcopy(dispositions))


def test_a1_source_construction_and_replay_identity_is_required() -> None:
    a1, _a1_replay, dispositions, _disposition_replay = _fixtures()
    a1["source_replay_artifact_sha256"] = "2" * 64
    a1, dispositions = _rebind_a1_parents(a1, dispositions)

    with pytest.raises(TemporalFailOpenContractError, match="source construction/replay"):
        _build(a1, copy.deepcopy(a1), dispositions, copy.deepcopy(dispositions))


def test_downstream_consumers_reject_legacy_format_temporal_overlay() -> None:
    a1, _a1_replay, dispositions, _disposition_replay = _fixtures()
    legacy_overlay = copy.deepcopy(dispositions)
    legacy_overlay["base_disposition_artifact_sha256"] = _sha(dispositions)
    legacy_overlay["temporal_fail_open_policy_id"] = "legacy-v1"
    legacy_overlay["temporal_fail_open_override_count"] = 1

    with pytest.raises(R7AfterUnionA1Error, match="legacy-format"):
        _disposition_lookup(legacy_overlay, a1["source_artifact_sha256"])
    with pytest.raises(R7A1ARawRetainedError, match="legacy-format"):
        a1a_dispositions(
            legacy_overlay,
            a1_preflight_artifact_sha256=_sha(a1),
            a1_preflight_replay_artifact_sha256=_sha(a1),
            source_r7_artifact_sha256=a1["source_artifact_sha256"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["policy_sha", "override_count", "missing_population", "missing_question_count"],
)
def test_effective_overlay_structural_receipts_fail_closed_downstream(
    mutation: str,
) -> None:
    fixtures = _fixtures()
    a1, _a1_replay, _dispositions, _disposition_replay = fixtures
    effective, _report = _build(*fixtures)
    changed = copy.deepcopy(effective)
    if mutation == "policy_sha":
        changed["policy_sha256"] = "8" * 64
    elif mutation == "override_count":
        changed["temporal_fail_open_override_count"] = 999
    elif mutation == "missing_population":
        del changed["effective_disposition_population_sha256"]
    elif mutation == "missing_question_count":
        del changed["question_count"]

    with pytest.raises((R7AfterUnionA1Error, ValueError)):
        _disposition_lookup(changed, a1["source_artifact_sha256"])
    with pytest.raises((R7A1ARawRetainedError, ValueError)):
        a1a_dispositions(
            changed,
            a1_preflight_artifact_sha256=_sha(a1),
            a1_preflight_replay_artifact_sha256=_sha(a1),
            source_r7_artifact_sha256=a1["source_artifact_sha256"],
        )


def test_forged_a1_semantic_selection_replay_is_rejected() -> None:
    a1, _a1_replay, dispositions, _disposition_replay = _fixtures()
    question = a1["questions"][0]  # type: ignore[index]
    semantic = question["semantic_selection"]
    result = semantic["semantic_result"]
    result["classifier_id"] = "forged-classifier"
    result_body = {
        key: value for key, value in result.items() if key != "receipt_sha256"
    }
    result["receipt_sha256"] = identity_sha256(result_body)
    semantic_body = {
        key: value
        for key, value in semantic.items()
        if key != "receipt_sha256"
    }
    semantic["receipt_sha256"] = identity_sha256(semantic_body)
    question_body = {
        key: value
        for key, value in question.items()
        if key != "question_receipt_sha256"
    }
    question["question_receipt_sha256"] = identity_sha256(question_body)
    a1["question_population_sha256"] = identity_sha256(
        [question["question_receipt_sha256"]]
    )
    a1, dispositions = _rebind_a1_parents(a1, dispositions)

    with pytest.raises(
        TemporalFailOpenContractError,
        match="semantic selection differs from deterministic replay",
    ):
        _build(a1, copy.deepcopy(a1), dispositions, copy.deepcopy(dispositions))


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_base_top_key",
        "extra_question_key",
        "extra_response_key",
        "foreign_response_question",
        "provider_output_extra_key",
    ],
)
def test_base_lifecycle_schema_and_request_bindings_fail_closed(
    mutation: str,
) -> None:
    a1, a1_replay, dispositions, _disposition_replay = _fixtures()
    response = dispositions["responses"][0]  # type: ignore[index]
    if mutation == "extra_base_top_key":
        dispositions["unexpected"] = "forbidden-extension"
    elif mutation == "extra_question_key":
        dispositions["questions"][0]["unexpected"] = "forbidden-extension"  # type: ignore[index]
        dispositions["disposition_population_sha256"] = identity_sha256(
            dispositions["questions"]
        )
    elif mutation == "extra_response_key":
        response["unexpected"] = "forbidden-extension"
    elif mutation == "foreign_response_question":
        response["question_sha256"] = "d" * 64
    elif mutation == "provider_output_extra_key":
        parsed = json.loads(response["classifier_output"])
        parsed["metadata"] = {}
        completion = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        response["classifier_output"] = completion
        response["classifier_output_sha256"] = quote_sha256(completion)
    if mutation in {
        "extra_response_key",
        "foreign_response_question",
        "provider_output_extra_key",
    }:
        response_body = {
            key: value
            for key, value in response.items()
            if key != "response_row_receipt_sha256"
        }
        response["response_row_receipt_sha256"] = identity_sha256(response_body)
        dispositions["response_population_sha256"] = identity_sha256(
            [response["response_row_receipt_sha256"]]
        )

    with pytest.raises((TemporalFailOpenContractError, ValueError)):
        _build(
            a1,
            a1_replay,
            dispositions,
            copy.deepcopy(dispositions),
        )
