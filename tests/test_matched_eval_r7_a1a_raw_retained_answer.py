from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
)
from tools.matched_eval.r7_after_union_a1 import (
    DISPOSITIONS_FORMAT,
    build_r7_after_union_a1_payload,
)
from tools.matched_eval.r7_a1a_raw_retained_answer import (
    R7A1ARawRetainedError,
    build_r7_a1a_raw_retained_payload,
    replay_r7_a1a_raw_retained_payload,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec
from tools.run_r7_a1a_raw_retained_answer import (
    CONSTRUCTION_NAME,
    REPLAY_NAME,
    run,
)


def _sha(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _source(summaries: tuple[str, ...]) -> dict[str, Any]:
    dated = (
        "[Question asked at 2025-01-09] "
        "What did I buy and where did I travel?"
    )
    spec = compile_typed_operator_spec(dated)
    handles: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    groups: list[str] = []
    for index, summary in enumerate(summaries):
        handle = f"H{index + 1:03d}"
        group = f"G{index + 1:03d}"
        groups.append(group)
        handles.append(
            {
                "group_handle": group,
                "handle_id": handle,
                "origin": "map",
                "provenance_grade": "exact_citation",
            }
        )
        items.append(
            {
                "date": f"2025-01-{(index % 28) + 1:02d}",
                "entity_key": f"Topic {index + 1}",
                "handle_ids": [handle],
                "included": True,
                "kind": "event",
                "relation": "completed event",
                "status": "completed",
                "summary": summary,
                "supported_slot_ids": [],
            }
        )
    typed = {
        "conflict_policy": "quarantine",
        "format": "synthetic-r7-typed-evidence-v1",
        "frontier": {
            "available_handle_ids": [row["handle_id"] for row in handles],
            "closed": False,
            "mode": "bounded",
            "omitted_handle_ids": [],
            "represented_handle_ids": [row["handle_id"] for row in handles],
            "truncated": False,
        },
        "handles": handles,
        "items": items,
        "operator_spec": spec.projection(),
    }
    question_sha = quote_sha256(dated)
    question = {
        "dated_question_sha256": question_sha,
        "ordinal": 777,
        "question_id": "q-a1a-fixture",
        "terminal_answer_plan": {
            "dated_question_sha256": question_sha,
            "parent_prediction": "MUST-NOT-ENTER-A1A",
            "provider_input": {
                "dated_question": dated,
                "protected_parent_fallback": "MUST-NOT-ENTER-A1A",
                "story_coherence": {
                    "group_links": (
                        [
                            {
                                "group_handles": groups[:2],
                                "relation": "same event across topic boundary",
                            }
                        ]
                        if len(groups) >= 2
                        else []
                    ),
                    "incompatible_group_pairs": [],
                    "link_overlays": [],
                },
                "typed_evidence": typed,
            },
            "reference_answer": "MUST-NOT-ENTER-A1A",
        },
    }
    return {
        "format": "memory-condense-reduced-semantic-global-terminal-assay-v2",
        "gold_loaded": False,
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": 1,
        "questions": [question],
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan_count": 1,
    }


def _a1_preflight(source: dict[str, Any]) -> dict[str, Any]:
    source_sha = _sha(source)
    return build_r7_after_union_a1_payload(
        source,
        source_sha,
        source_sha,
        expected_question_count=1,
    )


def _dispositions(
    source: dict[str, Any],
    preflight: dict[str, Any],
    decisions: dict[str, str],
) -> dict[str, Any]:
    row = preflight["questions"][0]
    leaves = row["semantic_selection"]["leaves"]
    return {
        "a1_construction_artifact_sha256": _sha(preflight),
        "a1_replay_artifact_sha256": _sha(preflight),
        "classifier_id": "sealed-a1a-fixture-classifier-v1",
        "format": DISPOSITIONS_FORMAT,
        "provider_calls_performed_by_core": 0,
        "questions": [
            {
                "classifier_request_sha256s": [
                    request["request_sha256"]
                    for request in row["classifier_requests"]
                ],
                "dispositions": [
                    {
                        "disposition": decisions.get(
                            leaf["handle_id"], "unresolved"
                        ),
                        "handle_id": leaf["handle_id"],
                        "leaf_receipt_sha256": leaf["receipt_sha256"],
                    }
                    for leaf in leaves
                ],
                "question_sha256": row["question_sha256"],
                "selected_union_population_sha256": row[
                    "selected_population_sha256"
                ],
            }
        ],
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": {
            "gold_loaded": False,
            "ordinal_routing_enabled": False,
            "protected_parent_loaded": False,
            "reference_loaded": False,
            "semantic_atom_manifest_loaded": False,
            "source_allowlist_loaded": False,
        },
        "source_artifact_sha256": _sha(source),
    }


def _runtime(
    summaries: tuple[str, ...],
    decisions: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _source(summaries)
    preflight = _a1_preflight(source)
    dispositions = _dispositions(source, preflight, decisions)
    payload = build_r7_a1a_raw_retained_payload(
        preflight,
        _sha(preflight),
        _sha(preflight),
        dispositions,
        _sha(dispositions),
        expected_question_count=1,
    )
    return payload, preflight, dispositions


def test_runtime_keeps_r_and_unresolved_and_excludes_only_i_after_union() -> None:
    payload, preflight, dispositions = _runtime(
        (
            "I bought the cobalt kettle.",
            "I traveled to Kyoto after the conference.",
            "The spare receipt concerned a notebook.",
        ),
        {
            "H001": "relevant",
            "H002": "unresolved",
            "H003": "definitely_irrelevant",
        },
    )

    row = payload["questions"][0]
    selection = row["classified_selection"]["semantic_result"]
    request = row["prompt_request"]
    control = row["control_prompt_request"]
    provider = json.loads(request["messages"][1]["content"])
    control_provider = json.loads(control["messages"][1]["content"])
    assert selection["retained_leaf_cell_ids"] == ["H001", "H002"]
    assert selection["pruned_leaf_cell_ids"] == ["H003"]
    assert request["allowed_handle_ids"] == ["H001", "H002"]
    assert control["allowed_handle_ids"] == ["H001", "H002", "H003"]
    assert request["fixed_union_leaf_population_sha256"] == control[
        "fixed_union_leaf_population_sha256"
    ]
    assert request["presented_handle_population_sha256"] != control[
        "presented_handle_population_sha256"
    ]
    assert request["messages"][0] == control["messages"][0]
    assert set(provider) == set(control_provider)
    assert request["arm"] == "raw_retained_treatment"
    assert control["arm"] == "fixed_union_renderer_control"
    assert control["execution_authority"].startswith("sealed_control_non_actionable")
    assert [evidence["handle_id"] for evidence in provider["evidence"]] == [
        "H001",
        "H002",
    ]
    assert [evidence["relevance_disposition"] for evidence in provider["evidence"]] == [
        "relevant",
        "unresolved",
    ]
    assert [
        evidence["handle_id"] for evidence in control_provider["evidence"]
    ] == ["H001", "H002", "H003"]
    assert [
        evidence["relevance_disposition"]
        for evidence in control_provider["evidence"]
    ] == ["relevant", "unresolved", "definitely_irrelevant"]
    assert control_provider["frontier"]["presented_leaf_count"] == 3
    assert provider["frontier"]["presented_leaf_count"] == 2
    assert len(provider["graph_links"]) == 1
    assert control_provider["graph_links"] == provider["graph_links"]
    assert row["density_metrics"]["fixed_union_leaf_count"] == 3
    assert row["density_metrics"]["retained_leaf_count"] == 2
    assert row["density_metrics"]["pruned_leaf_count"] == 1
    assert row["density_metrics"]["provider_payload_token_reduction"] > 0
    assert row["density_metrics"]["renderer_matched_prompt_token_reduction"] > 0
    assert request["prompt_token_proxy"] + request["output_token_reserve"] <= 8_000
    for prompt_request, provider_input, expected_handles in (
        (request, provider, ["H001", "H002"]),
        (control, control_provider, ["H001", "H002", "H003"]),
    ):
        assert prompt_request["presented_handle_population_sha256"] == (
            identity_sha256(expected_handles)
        )
        assert prompt_request["provider_input_sha256"] == identity_sha256(
            provider_input
        )
        assert prompt_request["messages_sha256"] == identity_sha256(
            prompt_request["messages"]
        )
        assert prompt_request["prompt_token_proxy"] == (
            count_chat_prompt_token_proxy(prompt_request["messages"])
        )
        assert (
            prompt_request["prompt_token_proxy"]
            + prompt_request["output_token_reserve"]
            <= prompt_request["hard_total_token_cap"]
        )
        request_body = dict(prompt_request)
        request_body.pop("request_sha256")
        assert prompt_request["request_sha256"] == identity_sha256(request_body)
    assert "MUST-NOT-ENTER-A1A" not in str(request["messages"])
    assert "ordinal" not in request
    assert "reference" not in request
    assert replay_r7_a1a_raw_retained_payload(
        payload,
        preflight,
        _sha(preflight),
        _sha(preflight),
        dispositions,
        _sha(dispositions),
    ) == payload


def test_all_unresolved_leaves_are_preserved_without_top_k() -> None:
    payload, _preflight, _dispositions_payload = _runtime(
        (
            "First disconnected topic memory.",
            "Second disconnected topic memory.",
            "Third disconnected topic memory.",
        ),
        {},
    )
    row = payload["questions"][0]
    assert row["prompt_request"]["allowed_handle_ids"] == [
        "H001",
        "H002",
        "H003",
    ]
    assert row["classified_selection"]["semantic_result"][
        "pruned_leaf_cell_ids"
    ] == []
    assert payload["density_totals"]["leaf_retention_ratio"] == 1.0
    assert payload["density_totals"]["provider_payload_token_reduction"] == 0
    assert row["prompt_request"]["messages"] == row["control_prompt_request"][
        "messages"
    ]


def test_retained_union_over_cap_fails_instead_of_silent_ranking() -> None:
    summaries = tuple(
        (f"Memory {index}: " + ("amber context " * 110)).strip()
        for index in range(60)
    )
    source = _source(summaries)
    preflight = _a1_preflight(source)
    dispositions = _dispositions(source, preflight, {})
    with pytest.raises(R7A1ARawRetainedError, match="exceeds 8K"):
        build_r7_a1a_raw_retained_payload(
            preflight,
            _sha(preflight),
            _sha(preflight),
            dispositions,
            _sha(dispositions),
            expected_question_count=1,
        )


def test_pruning_one_endpoint_removes_only_that_graph_link() -> None:
    payload, _preflight, _dispositions_payload = _runtime(
        ("First linked memory.", "Second linked memory.", "Other memory."),
        {"H001": "relevant", "H002": "definitely_irrelevant"},
    )
    request = payload["questions"][0]["prompt_request"]
    provider = json.loads(request["messages"][1]["content"])
    assert request["allowed_handle_ids"] == ["H001", "H003"]
    assert provider["graph_links"] == []
    assert request["graph_bindings"] == []
    assert len(payload["questions"][0]["control_prompt_request"]["graph_bindings"]) == 1


def test_cli_seals_byte_identical_runtime_and_replay(tmp_path: Path) -> None:
    source = _source(("I bought a kettle.", "I traveled to Kyoto."))
    preflight = _a1_preflight(source)
    dispositions = _dispositions(source, preflight, {})
    a1, _ = publish_sealed_json(tmp_path / "a1.json", preflight)
    a1_replay, _ = publish_sealed_json(tmp_path / "a1-replay.json", preflight)
    disposition_artifact, _ = publish_sealed_json(
        tmp_path / "dispositions.json", dispositions
    )
    disposition_replay, _ = publish_sealed_json(
        tmp_path / "dispositions-replay.json", dispositions
    )
    output = tmp_path / "output"
    result = run(
        Namespace(
            a1_construction=a1.path,
            a1_replay=a1_replay.path,
            dispositions=disposition_artifact.path,
            dispositions_replay=disposition_replay.path,
            expected_question_count=1,
            output_root=output,
        )
    )
    construction = read_sealed_json(output / CONSTRUCTION_NAME)
    replay = read_sealed_json(output / REPLAY_NAME)
    assert result["replay_byte_identical"] is True
    assert construction.sha256 == replay.sha256
    assert construction.payload == replay.payload
    assert result["prompt_request_count"] == 1
    assert result["control_prompt_request_count"] == 1
    assert result["provider_calls_performed_by_core"] == 0
    assert construction.payload["disposition_replay_artifact_sha256"] == (
        disposition_artifact.sha256
    )


def test_disposition_leaf_receipt_is_authenticated() -> None:
    source = _source(("I bought a kettle.",))
    preflight = _a1_preflight(source)
    dispositions = _dispositions(source, preflight, {})
    dispositions["questions"][0]["dispositions"][0][
        "leaf_receipt_sha256"
    ] = "f" * 64
    with pytest.raises(R7A1ARawRetainedError, match="leaf binding"):
        build_r7_a1a_raw_retained_payload(
            preflight,
            _sha(preflight),
            _sha(preflight),
            dispositions,
            _sha(dispositions),
            expected_question_count=1,
        )


def test_dispositions_bind_both_a1_construction_and_replay() -> None:
    source = _source(("I bought a kettle.",))
    preflight = _a1_preflight(source)
    dispositions = _dispositions(source, preflight, {})
    dispositions["a1_replay_artifact_sha256"] = "e" * 64

    with pytest.raises(R7A1ARawRetainedError, match="construction/replay"):
        build_r7_a1a_raw_retained_payload(
            preflight,
            _sha(preflight),
            _sha(preflight),
            dispositions,
            _sha(dispositions),
            expected_question_count=1,
        )


def test_authenticated_a1_preflight_cannot_smuggle_benchmark_fields() -> None:
    source = _source(("I bought a kettle.",))
    preflight = _a1_preflight(source)
    dispositions = _dispositions(source, preflight, {})
    contaminated = deepcopy(preflight)
    contaminated["reference_answer"] = "must never cross the runtime boundary"
    unsigned = dict(contaminated)
    unsigned.pop("construction_identity_sha256")
    contaminated["construction_identity_sha256"] = identity_sha256(unsigned)

    with pytest.raises(MatchedEvalContractError):
        build_r7_a1a_raw_retained_payload(
            contaminated,
            _sha(contaminated),
            _sha(contaminated),
            dispositions,
            _sha(dispositions),
            expected_question_count=1,
        )
