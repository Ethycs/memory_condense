from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from inspect import signature
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_matched_eval_closure_live import _parent_plane
from tests.test_matched_eval_query_expansion import (
    _FakePartitionSearch,
    _StructuredClient,
    _candidate,
    _population,
)
from tests.test_matched_eval_query_fact_adapter import _build
from tests.test_matched_eval_query_operator_refinement_live import _direct_plane
from tools.matched_eval import live
from tools.matched_eval.contracts import (
    StageDisposition,
    canonical_json_bytes,
    identity_sha256,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    VerifiedEvidenceMapPlane,
    VerifiedEvidenceMapRow,
    build_evidence_map_plan,
    parse_evidence_map,
)
from tools.matched_eval.query_expansion import (
    preflight_query_expansion,
    run_query_expansion,
)
from tools.matched_eval.query_map_source_gate_adapter import (
    CONSOLIDATED_OBLIGATION_MODE,
    FORMAT,
    LEGACY_OBLIGATION_MODE,
    PARENT_VERIFICATION_RULE_ID,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    STRICT_STATE_CHAIN_PROFILE,
    QueryMapSourceGateAdapterError,
    adapt_query_map_solver_v2,
    _uses_state_chain_direct_authority,
)
from tools.matched_eval.query_payload_live import build_query_payload_answer_plan
from tools.matched_eval.source_gate_controller import ObligationKind


def _sha(value: str) -> str:
    return quote_sha256(value)


def _map_plan(tmp_path: Path, query_plan: dict[str, list[str]]):
    source, namespace, population = _population(tmp_path)
    output = tmp_path / "query"
    preflight = preflight_query_expansion(population, output_root=output)
    protected = source.rows[0].packet.protected_evidence[0]
    duplicate = _candidate(
        chunk_id="chunk-0",
        source_id=protected.source_id,
        text=protected.text,
        score=0.99,
    )
    novel = _candidate(
        chunk_id="cross-prefix-chunk",
        source_id="unrelated-history::episode-7",
        text="I planted rosemary and mint two weeks ago.",
        score=0.91,
    )
    run = run_query_expansion(
        population,
        output_root=output,
        retrievers_by_namespace={
            namespace.namespace_id: _FakePartitionSearch(namespace, (duplicate, novel))
        },
        enable_provider=True,
        authorized_provider_calls=1,
        client=_StructuredClient(
            json.dumps(query_plan, separators=(",", ":"), sort_keys=True)
        ),
        max_concurrency=1,
    ).run_artifact
    adapter = _build(source, population, preflight, run)
    direct_plan = build_query_payload_answer_plan(adapter, _parent_plane(source))
    direct_plane = _direct_plane(direct_plan)
    runtime = live._thaw_json(direct_plane.runtime_ledger)
    direct_plane = replace(
        direct_plane,
        runtime_ledger_sha256=sha256(canonical_json_bytes(runtime)).hexdigest(),
    )
    return run, build_evidence_map_plan(direct_plan, direct_plane)


def _map_plane(map_plan, candidates: tuple[str, ...]) -> VerifiedEvidenceMapPlane:
    planned = map_plan.rows[0]
    alias = planned.aliases[-1]
    evidence = planned.retained_query_delta[-1]
    completion = json.dumps(
        {
            "items": [
                {
                    "alias": alias.alias,
                    "candidate": candidate,
                    "citation": evidence.text,
                    "kind": "extract_span",
                }
                for candidate in candidates
            ]
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    parsed = parse_evidence_map(
        completion,
        answer_kind="extract_span",
        evidence_text_by_alias={alias.alias: evidence.text},
    )
    status = "validated_items" if parsed.accepted_items else "no_valid_items"
    packet = planned.direct_plan_row.adapter.source.packet
    row = VerifiedEvidenceMapRow(
        ordinal=planned.ordinal,
        question_id=packet.question_id,
        question_sha256=packet.question_sha256,
        dated_question_sha256=packet.dated_question_sha256,
        route_id=planned.route.style.value,
        answer_kind="extract_span",
        accepted_items=parsed.accepted_items,
        rejected_items=parsed.rejected_items,
        map_status=status,
        map_parse_receipt_sha256=parsed.parse_receipt_sha256,
        map_plan_row_receipt_sha256=planned.receipt_sha256,
        direct_parent_prediction_sha256=planned.direct_answer_row.prediction_sha256,
        source_row_sha256=identity_sha256(
            {
                "accepted": [item.item_sha256 for item in parsed.accepted_items],
                "map_plan_row": planned.receipt_sha256,
            }
        ),
        runtime_row_id=_sha("map-runtime-row"),
        call_key_sha256=_sha("map-call-key"),
        request_journal_sha256=_sha("map-request"),
        response_journal_sha256=_sha("map-response"),
    )
    runtime = live._freeze_json({"adapter_test": True})
    runtime_sha = sha256(canonical_json_bytes(live._thaw_json(runtime))).hexdigest()
    run_sha = _sha("terminal-map-run")
    return VerifiedEvidenceMapPlane(
        run_sha256=run_sha,
        replay_sha256=run_sha,
        runtime_ledger_sha256=runtime_sha,
        runtime_ledger=runtime,
        parent_answer_run_sha256=map_plan.direct_plane.run_sha256,
        adapter_population_id=map_plan.direct_plan.adapter_population.population_id,
        retrieval_sha256=(
            map_plan.direct_plan.adapter_population.source_population.retrieval_sha256
        ),
        snapshot_id=map_plan.snapshot.snapshot_id,
        rows=(row,),
        parent_plane=map_plan.direct_plane,
    )


def _plan(
    *,
    entities: tuple[str, ...] = ("rosemary", "mint"),
    dates: tuple[str, ...] = (),
    operators: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    return {
        "queries": ["garden plant history"],
        "entities": list(entities),
        "dates": list(dates),
        "operators": list(operators),
    }


def test_fully_grounded_map_is_a_zero_call_no_op(tmp_path: Path) -> None:
    query_run, map_plan = _map_plan(tmp_path, _plan())
    map_plane = _map_plane(map_plan, ("rosemary and mint",))

    adapted = adapt_query_map_solver_v2(query_run, map_plan, map_plane)
    row = adapted.rows[0]

    assert row.disposition is StageDisposition.NO_OP
    assert row.activation is None
    assert row.obligations
    assert row.satisfied_obligation_ids == tuple(
        item.obligation_id for item in row.obligations
    )
    assert row.unresolved_obligation_ids == ()
    assert row.source_packet_id == (
        map_plan.rows[0].direct_plan_row.adapter.source.packet.packet_id
    )
    assert row.map_packet_id == map_plan.rows[0].packet_id
    assert row.parent_packet_id == row.map_packet_id
    assert row.source_packet_id != row.map_packet_id
    assert row.upstream_fact_frontier_receipt_sha256 == identity_sha256(
        {
            "accepted_item_sha256s": [
                item.item_sha256 for item in map_plane.rows[0].accepted_items
            ],
            "bounded_frontier_exhaustive": False,
            "format": f"{FORMAT}-map-fact-frontier",
            "map_packet_id": map_plan.rows[0].packet_id,
            "map_parse_receipt_sha256": map_plane.rows[0].map_parse_receipt_sha256,
            "map_plan_row_receipt_sha256": map_plan.rows[0].receipt_sha256,
            "map_source_row_sha256": map_plane.rows[0].source_row_sha256,
            "structured_temporal_metadata_available": False,
        }
    )
    assert adapted.provider_calls == 0
    assert adapted.retained_transformer_token_state_bytes == 0
    assert adapted.activated_rows == ()
    assert adapted.no_op_rows == (row,)


def test_consolidated_mode_compiles_one_any_anchor_and_parent_verifies(
    tmp_path: Path,
) -> None:
    query_run, map_plan = _map_plan(tmp_path, _plan())
    # The accepted candidate agrees exactly with the sealed direct parent;
    # its exact citation still carries both query anchors.
    adapted = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ("sealed direct prediction 0",)),
        obligation_mode=CONSOLIDATED_OBLIGATION_MODE,
    )
    row = adapted.rows[0]

    assert adapted.obligation_compilation_mode == CONSOLIDATED_OBLIGATION_MODE
    assert len(row.obligations) == 1
    support = row.obligations[0]
    assert support.kind is ObligationKind.SUPPORT
    assert support.match_terms == ("rosemary", "mint")
    assert support.required_match_term_count == 1
    assert row.parent_prediction_verification is not None
    assert row.parent_prediction_verification.rule_id == (
        PARENT_VERIFICATION_RULE_ID
    )
    assert row.parent_prediction_verification.mechanically_agrees is True
    assert row.disposition is StageDisposition.NO_OP
    assert row.activation is None
    projection = row.projection()
    assert projection["obligation_compilation_mode"] == (
        CONSOLIDATED_OBLIGATION_MODE
    )
    assert projection["parent_prediction_verification_receipt_sha256"]


def test_consolidated_parent_disagreement_keeps_support_unresolved(
    tmp_path: Path,
) -> None:
    query_run, map_plan = _map_plan(tmp_path, _plan())
    mapped = _map_plane(map_plan, ("rosemary and mint",))

    legacy = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        mapped,
        obligation_mode=LEGACY_OBLIGATION_MODE,
    )
    consolidated = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        mapped,
        obligation_mode=CONSOLIDATED_OBLIGATION_MODE,
    )

    assert legacy.rows[0].disposition is StageDisposition.NO_OP
    row = consolidated.rows[0]
    assert len(row.obligations) == 1
    assert row.parent_prediction_verification is not None
    assert row.parent_prediction_verification.mechanically_agrees is False
    assert row.satisfied_obligation_ids == ()
    assert row.unresolved_obligation_ids == (row.obligations[0].obligation_id,)
    assert row.disposition is StageDisposition.ADDED
    assert row.activation is not None
    assert row.reason == "parent_prediction_verification_disagreed"
    assert consolidated.receipt_sha256 != legacy.receipt_sha256


def test_consolidated_mode_adds_at_most_one_typed_operation_obligation(
    tmp_path: Path,
) -> None:
    query_run, map_plan = _map_plan(
        tmp_path,
        _plan(
            dates=("last Tuesday",),
            operators=("timeline", "earliest", "before_after"),
        ),
    )
    row = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ("sealed direct prediction 0",)),
        obligation_mode=CONSOLIDATED_OBLIGATION_MODE,
    ).rows[0]

    assert len(row.obligations) == 2
    assert row.obligations[0].kind is ObligationKind.SUPPORT
    assert row.obligations[1].kind is ObligationKind.TEMPORAL
    assert row.obligations[1].requires_temporal_metadata is True
    assert row.obligations[1].requires_complete_frontier is True
    assert row.obligations[1].obligation_id in row.unresolved_obligation_ids


def test_state_chain_direct_authority_requires_exact_unsubmitted_coordinates() -> None:
    assert _uses_state_chain_direct_authority(
        route_style="state_chain",
        map_submitted=False,
        map_status="not_submitted_state_chain",
        profile=STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    ) is True
    assert _uses_state_chain_direct_authority(
        route_style="state_chain",
        map_submitted=False,
        map_status="not_submitted_state_chain",
        profile=STRICT_STATE_CHAIN_PROFILE,
    ) is False
    assert _uses_state_chain_direct_authority(
        route_style="direct_extract",
        map_submitted=False,
        map_status="not_submitted_state_chain",
        profile=STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    ) is False
    assert _uses_state_chain_direct_authority(
        route_style="state_chain",
        map_submitted=True,
        map_status="validated_items",
        profile=STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    ) is False


def test_empty_map_is_a_retrieval_gap_and_activates(tmp_path: Path) -> None:
    query_run, map_plan = _map_plan(tmp_path, _plan())
    adapted = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ()),
    )
    row = adapted.rows[0]

    assert row.disposition is StageDisposition.ADDED
    assert row.activation is not None
    assert row.satisfied_obligation_ids == ()
    assert row.unresolved_obligation_ids == tuple(
        item.obligation_id for item in row.obligations
    )
    assert row.activation.parent_packet_id == (
        map_plan.rows[0].packet_id
    )
    assert row.activation.unresolved_obligation_ids == row.unresolved_obligation_ids


def test_partial_map_activates_only_the_missing_entity_obligation(
    tmp_path: Path,
) -> None:
    query_run, map_plan = _map_plan(
        tmp_path,
        _plan(entities=("rosemary", "basil")),
    )
    row = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ("rosemary",)),
    ).rows[0]

    assert tuple(item.match_terms for item in row.obligations) == (
        ("rosemary",),
        ("basil",),
    )
    assert row.satisfied_obligation_ids == (row.obligations[0].obligation_id,)
    assert row.unresolved_obligation_ids == (row.obligations[1].obligation_id,)
    assert row.activation is not None
    assert row.activation.obligation_ids == tuple(
        item.obligation_id for item in row.obligations
    )


def test_temporal_operator_needs_structured_map_proof_not_date_words(
    tmp_path: Path,
) -> None:
    query_run, map_plan = _map_plan(
        tmp_path,
        _plan(dates=("two weeks ago",), operators=("timeline",)),
    )
    row = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ("rosemary and mint two weeks ago",)),
    ).rows[0]

    assert row.satisfied_obligation_ids == tuple(
        item.obligation_id for item in row.obligations[:2]
    )
    operation = row.obligations[-1]
    assert operation.kind is ObligationKind.TEMPORAL
    assert operation.requires_temporal_metadata is True
    assert operation.minimum_fact_count == 1
    assert row.unresolved_obligation_ids == (operation.obligation_id,)
    assert row.activation is not None


def test_adapter_is_gold_blind_and_rejects_changed_parent_or_item_seals(
    tmp_path: Path,
) -> None:
    query_run, map_plan = _map_plan(tmp_path, _plan())
    map_plane = _map_plane(map_plan, ("rosemary and mint",))
    adapted = adapt_query_map_solver_v2(query_run, map_plan, map_plane)

    assert tuple(signature(adapt_query_map_solver_v2).parameters) == (
        "query_run",
        "map_plan",
        "map_plane",
        "obligation_mode",
        "state_chain_profile",
    )
    rendered = json.dumps(adapted.projection(), sort_keys=True)
    assert map_plan.rows[0].direct_answer_row.prediction not in rendered
    with pytest.raises(QueryMapSourceGateAdapterError, match="parent chain"):
        adapt_query_map_solver_v2(
            query_run,
            map_plan,
            replace(map_plane, parent_answer_run_sha256=_sha("changed-parent")),
        )
    item = map_plane.rows[0].accepted_items[0]
    changed_row = replace(
        map_plane.rows[0],
        accepted_items=(replace(item, candidate="changed candidate"),),
    )
    with pytest.raises(QueryMapSourceGateAdapterError, match="item seal"):
        adapt_query_map_solver_v2(
            query_run,
            map_plan,
            replace(map_plane, rows=(changed_row,)),
        )
    with pytest.raises(QueryMapSourceGateAdapterError, match="exact zero"):
        replace(adapted, provider_calls=False)
