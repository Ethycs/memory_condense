"""Verified bridge from query-guided exact spans to the shared payload plane.

The query-guided scan is a large provider-free construction artifact.  This
module checks its sealed run/replay/runtime plane, rehydrates every persisted
candidate into its exact ``EvidenceSpan`` identity, and only then projects the
post-selection, post-S0-dedup admitted spans into the existing query-payload
answer contract.

No function here accepts gold, a reference answer, a prior prediction, a
provider client, or a source/question-prefix filter.  The returned value is an
exact ``QueryFactAdapterPopulation`` so the mature split query-payload runtime
and its ``VerifiedQueryPayloadAnswerPlane`` can be reused without a second
answer implementation.
"""

from __future__ import annotations

import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_em_fact_memory import episodic_neighborhood
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastEvidence,
)

from tools._locked_em_repair_adapter import LockedEMQuestionView, LockedEMStageView
from tools._routed_repair_prompts import build_routed_fact_compression_prompt
from tools._routed_repair_routing import route_question

from .artifacts import SealedArtifact, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from .ledger import (
    RuntimeLedgerEntry,
    _validated_runtime_ledger,
    build_runtime_ledger,
)
from .population import EXPECTED_QUESTION_COUNT, SOURCE_STAGE_ID
from .query_expansion import (
    ARM_LABEL as QUERY_ARM_LABEL,
    STAGE_ID as QUERY_STAGE_ID,
    QueryExpansionPopulation,
    load_preflighted_query_expansion_population,
)
from .query_expansion_repack_v2 import (
    VerifiedQueryExpansionParent,
    verify_query_expansion_parent,
)
from .query_fact_adapter import (
    DEFAULT_COMPRESSION_PROMPT_CAP,
    QueryFactAdapterPopulation,
    QueryFactAdapterRow,
    _evidence_projection,
    _root_evidence,
)
from .query_guided_scan import (
    ARM_LABEL as GUIDED_ARM_LABEL,
    DEFAULT_TOKEN_CAP,
    MECHANISM_ID as GUIDED_MECHANISM_ID,
    PLAN_ID as GUIDED_PLAN_ID,
    RENDERER_ID as GUIDED_RENDERER_ID,
    ROW_FORMAT as GUIDED_ROW_FORMAT,
    RUN_FORMAT as GUIDED_RUN_FORMAT,
    RUN_NAME as GUIDED_RUN_NAME,
    RUN_REPLAY_NAME as GUIDED_RUN_REPLAY_NAME,
    RUNTIME_LEDGER_NAME as GUIDED_RUNTIME_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME as GUIDED_RUNTIME_LEDGER_REPLAY_NAME,
    STAGE_ID as GUIDED_STAGE_ID,
    QueryGuidedCandidate,
)


ADAPTER_FORMAT = "memory-condense-query-guided-payload-adapter-v1"
QUERY_POPULATION_FORMAT = "memory-condense-query-guided-payload-population-v1"
DELTA_TIER = "query_guided_scan_delta"


class QueryGuidedPayloadAdapterError(MatchedEvalContractError):
    """Raised when the guided construction plane loses exact provenance."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise QueryGuidedPayloadAdapterError(message)


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except MatchedEvalContractError as exc:
        raise QueryGuidedPayloadAdapterError(str(exc)) from exc


def _ids(value: object, label: str) -> tuple[str, ...]:
    _require(type(value) is list, f"{label} must be an exact array")
    result = tuple(_sha(row, f"{label} item") for row in value)  # type: ignore[union-attr]
    _require(len(result) == len(set(result)), f"{label} must be ordered and unique")
    return result


def _object_rows(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    _require(
        type(value) is list and all(type(row) is dict for row in value),
        f"{label} must be an array of exact objects",
    )
    return tuple(value)  # type: ignore[return-value]


def _ordered_subsequence(values: Sequence[str], parent: Sequence[str]) -> bool:
    iterator = iter(parent)
    return all(any(candidate == value for candidate in iterator) for value in values)


def _load_exact_artifact_payload(path: Path) -> dict[str, Any]:
    """Decode already hash-pinned bytes without a second canonical copy."""

    try:
        with path.open("rb") as stream:
            payload = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueryGuidedPayloadAdapterError(
            "guided run is not strict JSON"
        ) from exc
    _require(type(payload) is dict, "guided run must be an exact object")
    return payload


def _verify_exact_artifact_file(
    artifact_path: str | Path,
    *,
    expected_sha256: str,
) -> None:
    """Verify an identical replay without parsing a second 82 MB JSON tree."""

    target = Path(artifact_path)
    _require(
        target.is_file() and not target.is_symlink(),
        f"artifact must be a regular file: {target}",
    )
    _require(file_sha256(target) == expected_sha256, f"artifact bytes changed: {target}")
    sidecar = target.with_name(target.name + ".sha256")
    expected_sidecar = f"{expected_sha256}  {target.name}\n".encode("ascii")
    _require(
        sidecar.is_file()
        and not sidecar.is_symlink()
        and sidecar.read_bytes() == expected_sidecar,
        f"artifact digest sidecar changed: {sidecar}",
    )


def _verify_replay_file(
    replay_path: str | Path,
    *,
    expected_sha256: str,
) -> None:
    """Backward-compatible focused-test boundary for replay verification."""

    _verify_exact_artifact_file(
        replay_path,
        expected_sha256=expected_sha256,
    )


@dataclass(frozen=True, slots=True)
class VerifiedQueryGuidedConstruction:
    run_path: Path
    run_sha256: str
    runtime_ledger: SealedArtifact
    parent: VerifiedQueryExpansionParent


def verify_query_guided_construction(
    population: QueryExpansionPopulation,
    *,
    query_parent_root: str | Path,
    guided_root: str | Path,
    expected_query_preflight_sha256: str,
    expected_query_run_sha256: str,
    expected_query_runtime_ledger_sha256: str,
    expected_guided_run_sha256: str,
    expected_guided_runtime_ledger_sha256: str,
) -> VerifiedQueryGuidedConstruction:
    """Verify both sealed construction planes and rebuild the guided ledger."""

    if type(population) is not QueryExpansionPopulation:
        raise TypeError("population must be an exact QueryExpansionPopulation")
    parent = verify_query_expansion_parent(
        population,
        parent_output_root=query_parent_root,
        expected_preflight_sha256=expected_query_preflight_sha256,
        expected_run_sha256=expected_query_run_sha256,
        expected_runtime_ledger_sha256=expected_query_runtime_ledger_sha256,
    )
    output = Path(guided_root)
    expected_run = _sha(expected_guided_run_sha256, "expected guided run")
    expected_runtime = _sha(
        expected_guided_runtime_ledger_sha256,
        "expected guided runtime ledger",
    )
    run_path = output / GUIDED_RUN_NAME
    _verify_exact_artifact_file(run_path, expected_sha256=expected_run)
    _verify_replay_file(
        output / GUIDED_RUN_REPLAY_NAME,
        expected_sha256=expected_run,
    )
    runtime = read_sealed_json(output / GUIDED_RUNTIME_LEDGER_NAME)
    _require(runtime.sha256 == expected_runtime, "guided runtime ledger changed")
    _verify_replay_file(
        output / GUIDED_RUNTIME_LEDGER_REPLAY_NAME,
        expected_sha256=expected_runtime,
    )

    _validated_runtime_ledger(runtime.payload)
    return VerifiedQueryGuidedConstruction(
        run_path,
        expected_run,
        runtime,
        parent,
    )


def _candidate_from_projection(raw: Mapping[str, Any]) -> QueryGuidedCandidate:
    _require(type(raw) is dict, "guided candidate must be an exact object")
    span_raw = raw.get("span")
    _require(type(span_raw) is dict, "guided candidate span must be exact")
    try:
        span = EvidenceSpan(
            chunk_id=span_raw["chunk_id"],
            start_char=span_raw["start_char"],
            end_char=span_raw["end_char"],
            quote_sha256=span_raw["quote_sha256"],
            ordinal=span_raw["ordinal"],
            source_id=span_raw["source_id"],
            turn_start_char=span_raw["turn_start_char"],
            turn_id=span_raw["turn_id"],
            role=span_raw["role"],
            created_at=span_raw["created_at"],
        )
        _require(
            type(raw.get("query_coverage")) is float
            and type(raw.get("excerpt_density")) is float,
            "guided candidate surface scores must remain exact floats",
        )
        candidate = QueryGuidedCandidate(
            evidence_id=raw["evidence_id"],
            atom_id=raw["atom_id"],
            source_id=raw["source_id"],
            partition_id=raw["partition_id"],
            text=raw["text"],
            token_count=raw["token_count"],
            span=span,
            best_query_index=raw["best_query_index"],
            best_query_sha256=raw["best_query_sha256"],
            overlap_term_count=raw["overlap_term_count"],
            matching_query_count=raw["matching_query_count"],
            aggregate_overlap_count=raw["aggregate_overlap_count"],
            query_coverage=raw["query_coverage"],
            excerpt_density=raw["excerpt_density"],
            exact_phrase_match=raw["exact_phrase_match"],
            source_rank=raw["source_rank"],
            span_rank=raw["span_rank"],
        )
    except (KeyError, TypeError, ValueError, MatchedEvalContractError) as exc:
        raise QueryGuidedPayloadAdapterError(
            "guided candidate cannot be reconstructed exactly"
        ) from exc
    _require(span.identity_payload() == span_raw, "guided span projection changed")
    _require(candidate.projection() == raw, "guided candidate projection changed")
    return candidate


def _project_row(
    prompt: Any,
    raw: Mapping[str, Any],
    parent_raw: Mapping[str, Any],
    *,
    max_prompt_tokens: int,
) -> QueryFactAdapterRow:
    source = prompt.source
    label = f"guided row {source.ordinal}"
    _require(type(raw) is dict and raw.get("format") == GUIDED_ROW_FORMAT, f"{label} format changed")
    receipt = _sha(raw.get("receipt_sha256"), f"{label} receipt")
    unsigned = dict(raw)
    unsigned.pop("receipt_sha256", None)
    _require(identity_sha256(unsigned) == receipt, f"{label} self-seal changed")
    _require(
        raw.get("ordinal") == source.ordinal
        and raw.get("question_id") == source.packet.question_id
        and raw.get("question_sha256") == source.packet.question_sha256
        and raw.get("dated_question_sha256") == source.packet.dated_question_sha256
        and raw.get("parent_packet_id") == source.packet.packet_id
        and raw.get("namespace_id") == prompt.namespace.namespace_id,
        f"{label} source binding changed",
    )
    _require(
        raw.get("parent_query_expansion_row_receipt_sha256")
        == parent_raw.get("receipt_sha256")
        and raw.get("materialized_queries") == parent_raw.get("materialized_queries")
        and raw.get("parent_routing_receipts") == parent_raw.get("routing_receipts"),
        f"{label} query-route parent changed",
    )
    for key in (
        "gold_loaded",
        "known_history_filter_used",
        "question_id_filter_used",
        "source_prefix_filter_used",
    ):
        _require(raw.get(key) is False, f"{label} must attest {key}=false")
    _require(raw.get("provider_calls") == 0, f"{label} gained a provider call")
    _require(raw.get("retained_transformer_token_state_bytes") == 0, f"{label} retained transformer token state")
    _require(raw.get("stage_id") == GUIDED_STAGE_ID, f"{label} stage changed")
    _require(raw.get("candidate_token_cap") == DEFAULT_TOKEN_CAP, f"{label} candidate budget changed")
    _require(raw.get("dedup_timing") == "after_bounded_selection", f"{label} changed dedup timing")

    candidate_ids = _ids(raw.get("candidate_ids"), f"{label} candidates")
    selected = _ids(raw.get("selected_before_dedup_candidate_ids"), f"{label} selected")
    excluded = _ids(raw.get("dedup_excluded_candidate_ids"), f"{label} dedup exclusions")
    not_admitted = _ids(raw.get("not_admitted_candidate_ids"), f"{label} not admitted")
    admitted_ids = _ids(raw.get("admitted_candidate_ids"), f"{label} admitted")
    _require(_ordered_subsequence(selected, candidate_ids), f"{label} selected outside candidate order")
    for values, child in ((excluded, "dedup"), (not_admitted, "not-admitted"), (admitted_ids, "admitted")):
        _require(_ordered_subsequence(values, selected), f"{label} {child} IDs escaped selection")
    _require(
        not (set(excluded) & set(not_admitted) or set(excluded) & set(admitted_ids) or set(not_admitted) & set(admitted_ids))
        and set(selected) == set(excluded) | set(not_admitted) | set(admitted_ids),
        f"{label} selected lifecycle changed",
    )

    raw_candidates = _object_rows(raw.get("candidates"), f"{label} candidate projections")
    _require(len(raw_candidates) == len(candidate_ids), f"{label} candidate projection count changed")
    selected_set = set(selected)
    selected_candidates: dict[str, QueryGuidedCandidate] = {}
    for expected_id, candidate_raw in zip(candidate_ids, raw_candidates, strict=True):
        candidate = _candidate_from_projection(candidate_raw)
        _require(candidate.evidence_id == expected_id, f"{label} candidate projection order changed")
        if expected_id in selected_set:
            selected_candidates[expected_id] = candidate
    _require(set(selected_candidates) == selected_set, f"{label} selected candidate projections changed")

    admitted_raw = _object_rows(raw.get("admitted_candidates"), f"{label} admitted projections")
    _require(len(admitted_raw) == len(admitted_ids), f"{label} admitted projection count changed")
    for expected_id, projection in zip(admitted_ids, admitted_raw, strict=True):
        _require(
            selected_candidates[expected_id].projection() == projection,
            f"{label} admitted projection changed",
        )

    root = _root_evidence(source)
    root_coordinates = {
        (evidence.source_id, quote_sha256(evidence.text)): evidence.evidence_id
        for evidence in root
    }
    bindings = raw.get("dedup_alias_bindings")
    _require(
        type(bindings) is list
        and all(type(value) is list and len(value) == 2 for value in bindings),
        f"{label} dedup bindings changed",
    )
    binding_pairs = tuple((value[0], value[1]) for value in bindings)
    _require(tuple(value[0] for value in binding_pairs) == excluded, f"{label} dedup binding order changed")
    for candidate_id, protected_id in binding_pairs:
        candidate = selected_candidates[candidate_id]
        _require(
            root_coordinates.get((candidate.source_id, quote_sha256(candidate.text)))
            == protected_id,
            f"{label} dedup binding lost its exact S0 coordinate",
        )
    for candidate_id in admitted_ids:
        candidate = selected_candidates[candidate_id]
        _require(
            (candidate.source_id, quote_sha256(candidate.text)) not in root_coordinates,
            f"{label} admitted an exact protected-S0 duplicate",
        )

    selected_tokens = sum(selected_candidates[value].token_count for value in selected)
    admitted_tokens = sum(selected_candidates[value].token_count for value in admitted_ids)
    _require(raw.get("selected_before_dedup_token_count") == selected_tokens, f"{label} selected token accounting changed")
    _require(raw.get("tokens_used") == admitted_tokens <= DEFAULT_TOKEN_CAP, f"{label} admitted token accounting changed")
    disposition = raw.get("disposition")
    _require(
        disposition in {StageDisposition.ADDED.value, StageDisposition.NO_OP.value}
        and (disposition == StageDisposition.ADDED.value) == bool(admitted_ids),
        f"{label} disposition changed",
    )

    admitted = tuple(
        FastEvidence(
            selected_candidates[value].evidence_id,
            selected_candidates[value].source_id,
            selected_candidates[value].text,
        )
        for value in admitted_ids
    )
    root_stage = LockedEMStageView(
        stage_id=SOURCE_STAGE_ID,
        stage_receipt_sha256=source.source_stage_receipt_sha256,
        evidence_projection_sha256=_evidence_projection(root),
        evidence=root,
    )
    cumulative = root + admitted
    guided_stage = LockedEMStageView(
        stage_id=GUIDED_STAGE_ID,
        stage_receipt_sha256=receipt,
        evidence_projection_sha256=_evidence_projection(cumulative),
        evidence=cumulative,
    )
    question = LockedEMQuestionView(
        ordinal=source.ordinal,
        question_id=source.packet.question_id,
        question_sha256=source.packet.question_sha256,
        dated_question_sha256=source.packet.dated_question_sha256,
        retrieval_question_part_sha256=source.question_part_sha256,
        dated_question=source.packet.dated_question,
        stages=(root_stage, guided_stage),
    )
    observed_root, observed_delta = episodic_neighborhood(
        question,  # type: ignore[arg-type]
        stage_id=GUIDED_STAGE_ID,
    )
    _require(observed_root == root and observed_delta == admitted, f"{label} cumulative EM projection changed")
    route = route_question(question.dated_question)
    compression_prompt = build_routed_fact_compression_prompt(
        question,  # type: ignore[arg-type]
        route,
        stage_id=GUIDED_STAGE_ID,
        max_prompt_tokens=max_prompt_tokens,
    )
    binding = {
        "admitted_ids": list(admitted_ids),
        "compression_prompt_receipt_sha256": compression_prompt.receipt_sha256,
        "dedup_excluded_ids": list(excluded),
        "format": ADAPTER_FORMAT + "-row",
        "guided_row_receipt_sha256": receipt,
        "ordinal": source.ordinal,
        "question_id": source.packet.question_id,
        "route_receipt_sha256": route.receipt_sha256,
        "selected_before_dedup_ids": list(selected),
        "source_packet_id": source.packet.packet_id,
    }
    assert_gold_blind(binding, path=f"query_guided_payload_adapter.row[{source.ordinal}]")
    return QueryFactAdapterRow(
        source=source,
        question=question,
        route=route,
        compression_prompt=compression_prompt,
        selected_before_dedup_ids=selected,
        dedup_excluded_ids=excluded,
        not_admitted_ids=not_admitted,
        admitted_delta=admitted,
        query_row_receipt_sha256=receipt,
        binding_sha256=identity_sha256(binding),
    )


def _guided_runtime_entry(
    prompt: Any,
    raw: Mapping[str, Any],
    *,
    source_row_sha256: str,
) -> RuntimeLedgerEntry:
    """Rebuild the exact guided stage row from one streamed question."""

    receipt = _sha(raw.get("receipt_sha256"), "guided runtime row receipt")
    candidates = _ids(raw.get("candidate_ids"), "guided runtime candidates")
    selected = _ids(
        raw.get("selected_before_dedup_candidate_ids"),
        "guided runtime selected IDs",
    )
    excluded = _ids(
        raw.get("dedup_excluded_candidate_ids"),
        "guided runtime dedup IDs",
    )
    not_admitted = _ids(
        raw.get("not_admitted_candidate_ids"),
        "guided runtime not-admitted IDs",
    )
    admitted = _ids(raw.get("admitted_candidate_ids"), "guided runtime admitted IDs")
    delta_sha = identity_sha256(
        {
            "admitted_candidate_ids": list(admitted),
            "dedup_excluded_candidate_ids": list(excluded),
            "selected_before_dedup_candidate_ids": list(selected),
            "stage_id": GUIDED_STAGE_ID,
        }
    )
    packet_sha = identity_sha256(
        {
            "admitted_candidate_ids": list(admitted),
            "parent_query_expansion_row_receipt_sha256": raw[
                "parent_query_expansion_row_receipt_sha256"
            ],
            "stage_id": GUIDED_STAGE_ID,
        }
    )
    return RuntimeLedgerEntry(
        event_type="stage",
        ordinal=prompt.source.ordinal,
        question_id=prompt.source.packet.question_id,
        question_sha256=prompt.source.packet.question_sha256,
        arm_label=GUIDED_ARM_LABEL,
        parent_arm_label=QUERY_ARM_LABEL,
        stage_id=GUIDED_STAGE_ID,
        parent_stage_id=QUERY_STAGE_ID,
        mechanism_id=GUIDED_MECHANISM_ID,
        delta_kind="membership",
        renderer_id=GUIDED_RENDERER_ID,
        legacy_renderer=False,
        disposition=StageDisposition(str(raw["disposition"])),
        candidate_ids=candidates,
        selected_before_dedup_ids=selected,
        dedup_excluded_ids=excluded,
        not_admitted_ids=not_admitted,
        admitted_ids=admitted,
        token_cap=int(raw["candidate_token_cap"]),
        tokens_used=int(raw["tokens_used"]),
        reported_tokens_used=int(raw["tokens_used"]),
        local_model_calls=0,
        provider_calls=0,
        provider_prompt_cap=0,
        provider_prompt_reserved=0,
        global_provider_prompt_cap=0,
        historical_provider_calls=1,
        parent_packet_sha256=str(
            raw["parent_query_expansion_row_receipt_sha256"]
        ),
        packet_sha256=packet_sha,
        delta_sha256=delta_sha,
        stage_receipt_sha256=receipt,
        source_row_sha256=source_row_sha256,
        reason=str(raw["reason"]),
    )


def build_query_guided_payload_adapter(
    population: QueryExpansionPopulation,
    construction: VerifiedQueryGuidedConstruction,
    *,
    max_prompt_tokens: int = DEFAULT_COMPRESSION_PROMPT_CAP,
) -> QueryFactAdapterPopulation:
    """Project a fully verified guided construction into the shared adapter."""

    if type(population) is not QueryExpansionPopulation:
        raise TypeError("population must be an exact QueryExpansionPopulation")
    if type(construction) is not VerifiedQueryGuidedConstruction:
        raise TypeError("construction must be exact VerifiedQueryGuidedConstruction")
    if type(max_prompt_tokens) is not int or not 1 <= max_prompt_tokens <= DEFAULT_COMPRESSION_PROMPT_CAP:
        raise QueryGuidedPayloadAdapterError(
            f"max_prompt_tokens must be an integer from 1 through {DEFAULT_COMPRESSION_PROMPT_CAP}"
        )
    parent_rows = _object_rows(
        construction.parent.run.payload.get("questions"),
        "parent query questions",
    )
    _require(
        len(parent_rows) == len(population.rows),
        "guided adapter population size changed",
    )
    payload = _load_exact_artifact_payload(construction.run_path)
    guided_rows = _object_rows(payload.get("questions"), "guided questions")
    envelope = {key: value for key, value in payload.items() if key != "questions"}
    _require(
        len(guided_rows) == len(population.rows),
        "guided adapter population size changed",
    )
    projected_rows: list[QueryFactAdapterRow] = []
    runtime_entries: list[RuntimeLedgerEntry] = []
    for prompt, raw, parent_raw in zip(
        population.rows,
        guided_rows,
        parent_rows,
        strict=True,
    ):
        source_sha = identity_sha256(dict(raw))
        projected_rows.append(
            _project_row(
                prompt,
                raw,
                parent_raw,
                max_prompt_tokens=max_prompt_tokens,
            )
        )
        runtime_entries.append(
            _guided_runtime_entry(
                prompt,
                raw,
                source_row_sha256=source_sha,
            )
        )
    streamed_count = len(projected_rows)
    assert_gold_blind(envelope, path="query_guided_payload_adapter.guided_run")
    _require(envelope.get("format") == GUIDED_RUN_FORMAT, "guided run format changed")
    _require(envelope.get("arm_label") == GUIDED_ARM_LABEL, "guided arm label changed")
    _require(envelope.get("gold_loaded") is False, "guided run crossed the gold firewall")
    _require(envelope.get("provider_calls") == envelope.get("new_provider_calls") == 0, "guided scan gained provider calls")
    _require(envelope.get("routing_retrieval_rerun") is False, "guided scan reran routing retrieval")
    _require(envelope.get("source_prefix_filter_used") is False, "guided scan used a source-prefix filter")
    _require(envelope.get("retained_transformer_token_state_bytes") == 0, "guided scan retained transformer token state")
    _require(envelope.get("source_population_id") == population.source_population.population_id, "guided scan changed the source population")
    _require(envelope.get("question_count") == streamed_count, "guided scan question count changed")
    _require(
        envelope.get("parent_bindings")
        == {
            "preflight_sha256": construction.parent.preflight.sha256,
            "run_sha256": construction.parent.run.sha256,
            "runtime_ledger_sha256": construction.parent.runtime_ledger.sha256,
        },
        "guided scan changed its query-expansion parent bindings",
    )
    rebuilt_runtime = build_runtime_ledger(
        snapshot_id=population.source_population.snapshot.snapshot_id,
        plan_id=GUIDED_PLAN_ID,
        entries=runtime_entries,
        source_artifacts=(
            {
                "role": "sealed_retrieval",
                "sha256": population.source_population.retrieval_sha256,
            },
            {
                "role": "parent_query_preflight",
                "sha256": construction.parent.preflight.sha256,
            },
            {
                "role": "parent_query_run",
                "sha256": construction.parent.run.sha256,
            },
            {
                "role": "parent_query_runtime",
                "sha256": construction.parent.runtime_ledger.sha256,
            },
            {
                "role": "query_guided_scan_run",
                "sha256": construction.run_sha256,
            },
        ),
    )
    _require(
        rebuilt_runtime == construction.runtime_ledger.payload,
        "guided runtime ledger does not reconstruct from exact rows",
    )
    rows = tuple(projected_rows)
    # Release the repeated full candidate catalog before answer replay opens
    # its run/runtime artifacts.  The adapter retains only exact admitted
    # evidence and immutable receipts.
    payload.clear()
    del guided_rows
    del raw
    gc.collect()
    prompt_population = preflight_fast_completion_prompts(
        [row.compression_prompt.as_mappings() for row in rows],
        max_prompt_tokens=max_prompt_tokens,
    )
    _require(
        tuple(row.compression_prompt.messages_sha256 for row in rows)
        == tuple(row.messages_sha256 for row in prompt_population.ordered_rows),
        "guided adapter compression prompt order changed",
    )
    query_population_id = identity_sha256(
        {
            "format": QUERY_POPULATION_FORMAT,
            "guided_run_sha256": construction.run_sha256,
            "guided_runtime_ledger_sha256": construction.runtime_ledger.sha256,
            "parent_query_population_id": population.population_id,
        }
    )
    body = {
        "compression_prompt_population_sha256": prompt_population.prompt_population_sha256,
        "format": ADAPTER_FORMAT,
        "guided_mechanism_id": GUIDED_MECHANISM_ID,
        "guided_run_sha256": construction.run_sha256,
        "guided_runtime_ledger_sha256": construction.runtime_ledger.sha256,
        "max_prompt_tokens": max_prompt_tokens,
        "parent_query_population_id": population.population_id,
        "parent_query_preflight_sha256": construction.parent.preflight.sha256,
        "parent_query_run_sha256": construction.parent.run.sha256,
        "parent_query_runtime_ledger_sha256": construction.parent.runtime_ledger.sha256,
        "question_binding_sha256s": [row.binding_sha256 for row in rows],
        "query_population_id": query_population_id,
        "retrieval_sha256": population.source_population.retrieval_sha256,
        "source_population_id": population.source_population.population_id,
    }
    assert_gold_blind(body, path="query_guided_payload_adapter.population")
    return QueryFactAdapterPopulation(
        source_population=population.source_population,
        query_preflight_sha256=construction.parent.preflight.sha256,
        query_run_sha256=construction.run_sha256,
        query_population_id=query_population_id,
        query_prompt_population_sha256=(
            population.prompt_population.prompt_population_sha256
        ),
        rows=rows,
        compression_prompt_population=prompt_population,
        max_prompt_tokens=max_prompt_tokens,
        population_id=identity_sha256(body),
    )


def load_query_guided_payload_adapter(
    retrieval_path: str | Path,
    *,
    query_parent_root: str | Path,
    guided_root: str | Path,
    expected_retrieval_sha256: str,
    expected_source_population_id: str,
    expected_query_preflight_sha256: str,
    expected_query_run_sha256: str,
    expected_query_runtime_ledger_sha256: str,
    expected_query_population_id: str,
    expected_query_prompt_population_sha256: str,
    expected_guided_run_sha256: str,
    expected_guided_runtime_ledger_sha256: str,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    max_prompt_tokens: int = DEFAULT_COMPRESSION_PROMPT_CAP,
) -> QueryFactAdapterPopulation:
    """Load and verify the complete provider-free guided adapter population."""

    population, preflight = load_preflighted_query_expansion_population(
        retrieval_path,
        output_root=query_parent_root,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    _require(
        population.source_population.population_id
        == _sha(expected_source_population_id, "expected source population"),
        "guided adapter source population changed",
    )
    _require(
        preflight.sha256
        == _sha(expected_query_preflight_sha256, "expected query preflight")
        and population.population_id
        == _sha(expected_query_population_id, "expected query population")
        and population.prompt_population.prompt_population_sha256
        == _sha(
            expected_query_prompt_population_sha256,
            "expected query prompt population",
        ),
        "guided adapter query population changed",
    )
    construction = verify_query_guided_construction(
        population,
        query_parent_root=query_parent_root,
        guided_root=guided_root,
        expected_query_preflight_sha256=expected_query_preflight_sha256,
        expected_query_run_sha256=expected_query_run_sha256,
        expected_query_runtime_ledger_sha256=expected_query_runtime_ledger_sha256,
        expected_guided_run_sha256=expected_guided_run_sha256,
        expected_guided_runtime_ledger_sha256=(
            expected_guided_runtime_ledger_sha256
        ),
    )
    return build_query_guided_payload_adapter(
        population,
        construction,
        max_prompt_tokens=max_prompt_tokens,
    )


__all__ = [
    "ADAPTER_FORMAT",
    "DELTA_TIER",
    "QueryGuidedPayloadAdapterError",
    "VerifiedQueryGuidedConstruction",
    "build_query_guided_payload_adapter",
    "load_query_guided_payload_adapter",
    "verify_query_guided_construction",
]
