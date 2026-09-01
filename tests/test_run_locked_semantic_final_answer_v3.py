from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_final_answer_v3 as answer
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.typed_memory_final_arm import judge_row_projection
from tests.test_run_locked_semantic_final_answer import (
    _FakeBatch,
    _canonical_semantic_question,
)


def _sha(label: str) -> str:
    return quote_sha256(label)


def _reseal_frontier(frontier: dict[str, Any]) -> None:
    body = dict(frontier)
    body.pop("receipt_sha256", None)
    frontier["receipt_sha256"] = identity_sha256(body)


def _reseal_search(search: dict[str, Any], ordinal: int) -> None:
    search["format"] = answer.STORED_SEARCH_FORMAT
    search["canonical_result_projection_sha256"] = _sha(
        f"canonical semantic result {ordinal}"
    )
    search["attempted_selection_receipt_sha256"] = _sha(
        f"attempted selection population {ordinal}"
    )
    search["receipt_sha256"] = _sha(f"canonical semantic search v3 {ordinal}")
    body = dict(search)
    body.pop("stored_projection_receipt_sha256", None)
    search["stored_projection_receipt_sha256"] = identity_sha256(body)


def _reseal_closure(row: dict[str, Any]) -> None:
    closure = row["classified_closure"]
    assert type(closure) is dict
    search = row["semantic_residual_search"]
    assert type(search) is dict
    frontier = search["classified_frontier"]
    assert type(frontier) is dict
    dedup = row["additive_composition_local_audit"]["post_selection_dedup"]
    fitted = row["fitted_typed_prompt"]
    assert type(fitted) is dict
    closure["format"] = answer.CLASSIFIED_CLOSURE_FORMAT
    closure["semantic_residual_search_receipt_sha256"] = search["receipt_sha256"]
    closure["classified_frontier_receipt_sha256"] = frontier["receipt_sha256"]
    closure["post_selection_dedup_audit_receipt_sha256"] = dedup["receipt_sha256"]
    closure["retained_segment_receipt_sha256s"] = list(
        frontier["retained_segment_receipt_sha256s"]
    )
    allowed_body = {
        "format": (
            f"{answer.CLASSIFIED_CLOSURE_FORMAT}-terminal-allowed-handles-v1"
        ),
        "terminal_allowed_handle_ids": closure["terminal_allowed_handle_ids"],
    }
    closure["terminal_allowed_handle_ids_sha256"] = identity_sha256(allowed_body)
    protection_body = {
        "classified_frontier_receipt_sha256": frontier["receipt_sha256"],
        "format": f"{answer.CLASSIFIED_CLOSURE_FORMAT}-protection-source-v1",
        "post_selection_dedup_audit_receipt_sha256": dedup["receipt_sha256"],
        "retained_segment_receipt_sha256s": closure[
            "retained_segment_receipt_sha256s"
        ],
        "rows": closure["rows"],
        "semantic_residual_search_receipt_sha256": search["receipt_sha256"],
    }
    protection = identity_sha256(protection_body)
    closure["protection_source_receipt_sha256"] = protection
    fitted["protection_source_receipt_sha256"] = protection
    body = dict(closure)
    body.pop("receipt_sha256", None)
    closure["receipt_sha256"] = identity_sha256(body)


def _reseal_question(row: dict[str, Any]) -> None:
    body = dict(row)
    body.pop("question_receipt_sha256", None)
    row["question_receipt_sha256"] = identity_sha256(body)


def _v3_semantic_question(
    ordinal: int,
    *,
    search_level_protected_duplicate: bool = False,
) -> dict[str, Any]:
    row = copy.deepcopy(_canonical_semantic_question(ordinal))
    search = row["semantic_residual_search"]
    local = row["semantic_residual_local_audit"]
    closure = row["classified_closure"]
    assert type(search) is dict and type(local) is dict and type(closure) is dict

    if search_level_protected_duplicate:
        evidence = search["evidence"].pop(0)
        segment = evidence["segment_receipt_sha256"]
        closure_row = closure["rows"][0]
        duplicate_body = {
            "cell_id": evidence["cell_id"],
            "format": "memory-condense-semantic-residual-protected-duplicate-v1",
            "protected_binding_receipt_sha256": evidence[
                "citation_binding_receipt_sha256"
            ],
            "protected_candidate_id": evidence["candidate_id"],
            "reason": "exact_span_already_in_protected_evidence",
            "segment_receipt_sha256": segment,
            "span_identity_sha256": _sha(f"protected span {ordinal}"),
        }
        duplicate = {
            **duplicate_body,
            "receipt_sha256": identity_sha256(duplicate_body),
        }
        search["protected_duplicates"] = [duplicate]
        search["local_binding_receipt_sha256s"] = search[
            "local_binding_receipt_sha256s"
        ][1:]
        frontier = search["classified_frontier"]
        frontier["packed_segment_receipt_sha256s"].remove(segment)
        frontier["protected_duplicate_segment_receipt_sha256s"] = [segment]
        frontier["protected_duplicate_audit_receipt_sha256s"] = [
            duplicate["receipt_sha256"]
        ]
        _reseal_frontier(frontier)
        closure_row["residual_evidence_receipt_sha256"] = duplicate[
            "receipt_sha256"
        ]
        closure_row["residual_binding_receipt_sha256"] = duplicate[
            "protected_binding_receipt_sha256"
        ]
        closure_row["residual_item_receipt_sha256"] = closure_row[
            "visible_item_receipt_sha256"
        ]
        closure_row["dedup_exclusion_sha256"] = duplicate["receipt_sha256"]

    _reseal_search(search, ordinal)
    local.update(
        {
            "attempted_selection": [],
            "attempted_selection_manifest": {
                "format": "synthetic-attempted-selection-manifest-v3",
                "rows": [],
            },
            "capacity_certificate": {
                "format": "synthetic-capacity-certificate-v3"
            },
            "classified_frontier": copy.deepcopy(search["classified_frontier"]),
            "compact_result_receipt_sha256": search["receipt_sha256"],
            "format": answer.LOCAL_AUDIT_FORMAT,
            "protected_duplicates": copy.deepcopy(search["protected_duplicates"]),
            "protected_parent_inventory": {
                "format": "synthetic-protected-parent-inventory-v3"
            },
            "query": copy.deepcopy(row["semantic_query"]),
        }
    )
    local_body = dict(local)
    local_body.pop("receipt_sha256", None)
    local["receipt_sha256"] = identity_sha256(local_body)
    _reseal_closure(row)
    _reseal_question(row)
    return row


def _v3_fallback_question(ordinal: int) -> dict[str, Any]:
    row = _v3_semantic_question(ordinal)
    row.update(
        {
            "additive_composition": None,
            "additive_composition_local_audit": None,
            "classified_closure": None,
            "fallback_reason": "protected_semantic_residual_exceeds_terminal_cap",
            "fitted_typed_prompt": None,
            "mode": answer.PARENT_PASSTHROUGH_MODE,
            "terminal_prompt": None,
        }
    )
    _reseal_question(row)
    return row


def _construction(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...], answer.ConstructionLoader]:
    rows = (
        _v3_semantic_question(42, search_level_protected_duplicate=True),
        _v3_fallback_question(65),
        _v3_semantic_question(74),
        _v3_fallback_question(79),
    )
    payload = {
        "format": answer.CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "ordinals": list(answer.QUESTION_ORDINALS),
        "question_count": answer.QUESTION_COUNT,
        "questions_sha256": identity_sha256(list(rows)),
    }
    artifact, created = publish_sealed_json(
        tmp_path / "reduced-semantic-binary-search-construction-v3.json",
        payload,
    )
    assert created

    def loader(
        path: Path,
        *,
        expected_sha256: str,
    ) -> tuple[SealedArtifact, Sequence[Mapping[str, Any]]]:
        assert path == artifact.path
        assert expected_sha256 == artifact.sha256
        return artifact, tuple(json.loads(json.dumps(rows, sort_keys=True)))

    return artifact, rows, loader


def _plans(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact, _rows, loader = _construction(tmp_path)
    return answer.load_answer_plans(
        artifact.path,
        artifact.sha256,
        construction_loader=loader,
    )


def _preflight(
    tmp_path: Path,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    construction, plans = _plans(tmp_path)
    payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )
    preflight = SealedArtifact(
        Path("synthetic-semantic-preflight-v3.json"),
        identity_sha256(payload),
        payload,
    )
    return construction, preflight, plans


def test_v3_paths_and_four_question_contract_are_distinct() -> None:
    assert answer.DEFAULT_CONSTRUCTION.name == (
        "reduced-semantic-binary-search-construction-v3.json"
    )
    assert answer.DEFAULT_OUTPUT.name == "locked-semantic-final-answer-v3"
    assert answer.RUN_NAME == "locked-semantic-final-answer-v3.json"
    assert answer.REPLAY_NAME == "locked-semantic-final-answer-replay-v3.json"
    assert answer.QUESTION_ORDINALS == (42, 65, 74, 79)
    assert answer.FORMAT == "memory-condense-locked-semantic-final-terra-answer-v3"


def test_preflight_seals_only_unique_terminal_semantic_prompts(tmp_path: Path) -> None:
    construction, plans = _plans(tmp_path)
    payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )

    assert tuple(row["ordinal"] for row in plans) == answer.QUESTION_ORDINALS
    assert payload["semantic_question_count"] == 2
    assert payload["parent_passthrough_count"] == 2
    assert payload["required_authorized_provider_calls"] == 2
    assert payload["prompt_population"]["logical_prompt_count"] == 2
    assert payload["prompt_population"]["unique_prompt_count"] == 2
    assert payload["observed_max_complete_envelope_tokens"] <= 8_000
    assert payload["provider_calls"] == 0
    assert payload["gold_loaded"] is False
    assert payload["retained_transformer_token_state_bytes"] == 0


@pytest.mark.parametrize("mutation", ["extra_key", "search_type", "closure_row"])
def test_v3_source_rejects_key_type_and_schema_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact, rows, _loader = _construction(tmp_path)
    tampered = copy.deepcopy(list(rows))
    row = tampered[0]
    if mutation == "extra_key":
        row["unexpected"] = True
    elif mutation == "search_type":
        row["semantic_residual_search"] = []
    else:
        row["classified_closure"]["rows"][0].pop("exact_text_sha256")
        _reseal_closure(row)
    _reseal_question(row)

    def loader(path: Path, *, expected_sha256: str):
        assert path == artifact.path and expected_sha256 == artifact.sha256
        return artifact, tuple(tampered)

    with pytest.raises(answer.LockedSemanticFinalAnswerV3Error):
        answer.load_answer_plans(
            artifact.path,
            artifact.sha256,
            construction_loader=loader,
        )


def test_search_level_protected_duplicate_must_remain_exactly_provider_visible(
    tmp_path: Path,
) -> None:
    artifact, rows, _loader = _construction(tmp_path)
    assert rows[0]["semantic_residual_search"]["protected_duplicates"]
    plans = answer.load_answer_plans(
        artifact.path,
        artifact.sha256,
        construction_loader=lambda path, *, expected_sha256: (artifact, rows),
    )[1]
    assert plans[0]["mode"] == answer.SEMANTIC_MODE

    tampered = copy.deepcopy(list(rows))
    row = tampered[0]
    closure_row = row["classified_closure"]["rows"][0]
    closure_row["exact_text_sha256"] = _sha("lossy hidden substitute")
    _reseal_closure(row)
    _reseal_question(row)

    with pytest.raises(
        answer.LockedSemanticFinalAnswerV3Error,
        match="provider-visible",
    ):
        answer.load_answer_plans(
            artifact.path,
            artifact.sha256,
            construction_loader=lambda path, *, expected_sha256: (
                artifact,
                tuple(tampered),
            ),
        )


def test_checkpoint_only_materialization_parses_semantic_and_copies_passthrough(
    tmp_path: Path,
) -> None:
    _construction_artifact, preflight, plans = _preflight(tmp_path)
    physical = tuple(row for row in plans if row["mode"] == answer.SEMANTIC_MODE)
    residual_handle = next(
        handle
        for handle in physical[0]["allowed_handle_ids"]
        if handle.startswith("H950")
    )
    completions = (
        json.dumps(
            {
                "decision": "replace",
                "prediction": "blue mug",
                "used_handle_ids": [residual_handle],
            },
            separators=(",", ":"),
        ),
        "not valid JSON",
    )
    batch = _FakeBatch(physical, completions)
    payload = answer._materialization_projection(preflight, plans, batch)

    assert payload["question_count"] == 4
    assert payload["ordinals"] == list(answer.QUESTION_ORDINALS)
    assert payload["required_authorized_provider_calls"] == 2
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert payload["parent_passthrough_count"] == 2
    assert payload["semantic_question_count"] == 2
    assert payload["validated_replacement_count"] == 1
    assert payload["invalid_completion_parent_fallback_count"] == 1
    assert payload["changed_prediction_count"] == 1

    for row, projected in zip(payload["questions"], payload["judge_rows"], strict=True):
        unsigned = dict(row)
        declared = unsigned.pop("source_row_sha256")
        assert declared == identity_sha256(unsigned)
        assert projected == judge_row_projection(row)

    replaced = payload["questions"][0]
    assert replaced["ordinal"] == 42
    assert replaced["prediction"] == "blue mug"
    assert replaced["changed_from_parent"] is True
    assert replaced["solver_valid"] is True

    first_passthrough = payload["questions"][1]
    assert first_passthrough["ordinal"] == 65
    assert first_passthrough["prediction"] == plans[1]["parent_prediction"]
    assert first_passthrough["prediction_source"] == plans[1][
        "parent_prediction_source"
    ]
    assert first_passthrough["call_key_sha256"] is None
    assert first_passthrough["completion_parser"] == "none"

    invalid = payload["questions"][2]
    assert invalid["ordinal"] == 74
    assert invalid["prediction"] == plans[2]["parent_prediction"]
    assert invalid["changed_from_parent"] is False
    assert invalid["solver_valid"] is False


def test_preflight_gold_firewall_and_byte_identical_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, plans = _plans(tmp_path)
    preflight_payload = answer._preflight_projection(
        construction,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )
    poisoned = copy.deepcopy(preflight_payload)
    poisoned["reference_answer"] = "gold must not enter"
    with pytest.raises(MatchedEvalContractError, match="gold-bearing field"):
        answer._validate_preflight(
            SealedArtifact(Path("poisoned.json"), identity_sha256(poisoned), poisoned)
        )

    preflight, created = publish_sealed_json(
        tmp_path / answer.PREFLIGHT_NAME,
        preflight_payload,
    )
    assert created
    physical = tuple(row for row in plans if row["mode"] == answer.SEMANTIC_MODE)
    batch = _FakeBatch(physical)
    run_payload = answer._materialization_projection(preflight, plans, batch)
    run, created = publish_sealed_json(tmp_path / answer.RUN_NAME, run_payload)
    assert created

    monkeypatch.setattr(
        answer,
        "load_answer_plans",
        lambda path, expected_sha256: (construction, plans),
    )
    monkeypatch.setattr(answer, "_checkpoint_batch", lambda *args, **kwargs: batch)
    args = SimpleNamespace(
        construction=construction.path,
        expected_construction_sha256=construction.sha256,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=run.sha256,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
        model=answer.DEFAULT_MODEL,
        output_root=tmp_path,
    )
    result = answer.run_replay(args)

    assert result["byte_identical"] is True
    assert result["physical_provider_calls"] == 0
    assert result["run_sha256"] == result["replay_sha256"] == run.sha256
