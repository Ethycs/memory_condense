"""Verified bridge from partition-scan v2 into the shared payload plane.

The partition scan is a provider-free construction artifact.  This module
loads its sealed generation, reconstructs every exact ``EvidenceSpan`` through
``load_partition_scan_v2_generation``, verifies the complete
candidate/selection/admission lifecycle, and checks that protected-S0
deduplication happened only after selection.  It then projects the admitted
partition spans into ``QueryFactAdapterPopulation`` so the existing direct
payload prompt, split provider lifecycle, replay, and consolidated answer
judge can be reused without a second answer implementation.

The ``query_*`` fields on that shared adapter are compatibility slots.  Here
they are all explicitly bound to the sealed partition generation and to a
derived partition-population identity; no query-expansion artifact or provider
call is involved.
"""

from __future__ import annotations

from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
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

from .contracts import MatchedEvalContractError, assert_gold_blind, identity_sha256
from .partition_scan import TOKEN_CAP
from .partition_scan_v2 import (
    MECHANISM_ID,
    STAGE_ID,
    PartitionScanV2Generation,
    PartitionScanV2Question,
    load_partition_scan_v2_generation,
)
from .population import (
    EXPECTED_QUESTION_COUNT,
    SOURCE_STAGE_ID,
    MatchedS0Population,
    MatchedS0Row,
    load_s0_population,
)
from .query_fact_adapter import (
    DEFAULT_COMPRESSION_PROMPT_CAP,
    QueryFactAdapterPopulation,
    QueryFactAdapterRow,
    _evidence_projection,
    _root_evidence,
)


ADAPTER_FORMAT = "memory-condense-partition-scan-v2-payload-adapter-v1"
POPULATION_FORMAT = "memory-condense-partition-scan-v2-payload-population-v1"
PROMPT_BINDING_FORMAT = (
    "memory-condense-partition-scan-v2-payload-prompt-binding-v1"
)
DELTA_TIER = "partition_scan_v2_delta"


class PartitionPayloadAdapterError(MatchedEvalContractError):
    """Raised when the partition payload loses its sealed provenance."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise PartitionPayloadAdapterError(message)


def _project_row(
    source: MatchedS0Row,
    question: PartitionScanV2Question,
    *,
    max_prompt_tokens: int,
) -> QueryFactAdapterRow:
    label = f"partition payload row {source.ordinal}"
    _require(
        question.ordinal == source.ordinal
        and question.question_id == source.packet.question_id
        and question.packet_id == source.packet.packet_id,
        f"{label} changed its matched-S0 binding",
    )
    trace = question.trace
    _require(
        trace.provider_prompt_count == 0 and trace.token_cap == TOKEN_CAP,
        f"{label} changed the provider-free partition budget",
    )
    by_id = {candidate.evidence_id: candidate for candidate in question.candidates}
    _require(
        tuple(by_id) == trace.candidate_ids,
        f"{label} changed exact candidate order",
    )
    selected = trace.selected_before_dedup_ids
    excluded = trace.dedup_excluded_ids
    not_admitted = trace.not_admitted_ids
    admitted_ids = trace.admitted_ids
    _require(
        all(value in by_id for value in selected),
        f"{label} selected an unknown exact candidate",
    )
    _require(
        trace.tokens_used == sum(by_id[value].token_count for value in admitted_ids),
        f"{label} changed admitted token accounting",
    )

    root = _root_evidence(source)
    protected_by_id = {row.evidence_id: row for row in source.packet.protected_evidence}
    _require(
        len(protected_by_id) == len(source.packet.protected_evidence),
        f"{label} protected S0 IDs repeat",
    )
    bindings = question.dedup_alias_bindings
    _require(
        tuple(candidate_id for candidate_id, _protected_id in bindings) == excluded,
        f"{label} changed post-selection dedup order",
    )
    for candidate_id, protected_id in bindings:
        candidate = by_id[candidate_id]
        protected = protected_by_id.get(protected_id)
        _require(
            candidate_id in selected
            and protected is not None
            and candidate.source_id == protected.source_id
            and candidate.text in protected.text,
            f"{label} dedup alias lost its exact protected-S0 binding",
        )
    for candidate_id in admitted_ids:
        candidate = by_id[candidate_id]
        _require(
            not any(
                protected.source_id == candidate.source_id
                and candidate.text in protected.text
                for protected in source.packet.protected_evidence
            ),
            f"{label} admitted a protected-S0 duplicate",
        )
        _require(
            quote_sha256(candidate.text) == candidate.span.quote_sha256
            and count_tokens(candidate.text) == candidate.token_count,
            f"{label} changed exact candidate text provenance",
        )

    admitted = tuple(
        FastEvidence(
            by_id[value].evidence_id,
            by_id[value].source_id,
            by_id[value].text,
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
    row_receipt = question.question_identity_sha256
    partition_stage = LockedEMStageView(
        stage_id=STAGE_ID,
        stage_receipt_sha256=row_receipt,
        evidence_projection_sha256=_evidence_projection(cumulative),
        evidence=cumulative,
    )
    em_question = LockedEMQuestionView(
        ordinal=source.ordinal,
        question_id=source.packet.question_id,
        question_sha256=source.packet.question_sha256,
        dated_question_sha256=source.packet.dated_question_sha256,
        retrieval_question_part_sha256=source.question_part_sha256,
        dated_question=source.packet.dated_question,
        stages=(root_stage, partition_stage),
    )
    observed_root, observed_delta = episodic_neighborhood(
        em_question,  # type: ignore[arg-type]
        stage_id=STAGE_ID,
    )
    _require(
        observed_root == root and observed_delta == admitted,
        f"{label} changed its cumulative S0-plus-partition projection",
    )
    route = route_question(em_question.dated_question)
    compression_prompt = build_routed_fact_compression_prompt(
        em_question,  # type: ignore[arg-type]
        route,
        stage_id=STAGE_ID,
        max_prompt_tokens=max_prompt_tokens,
    )
    binding = {
        "admitted_ids": list(admitted_ids),
        "compression_prompt_receipt_sha256": compression_prompt.receipt_sha256,
        "dedup_excluded_ids": list(excluded),
        "format": ADAPTER_FORMAT + "-row",
        "ordinal": source.ordinal,
        "partition_question_identity_sha256": row_receipt,
        "question_id": source.packet.question_id,
        "route_receipt_sha256": route.receipt_sha256,
        "selected_before_dedup_ids": list(selected),
        "source_packet_id": source.packet.packet_id,
    }
    assert_gold_blind(binding, path=f"partition_payload_adapter.row[{source.ordinal}]")
    return QueryFactAdapterRow(
        source=source,
        question=em_question,
        route=route,
        compression_prompt=compression_prompt,
        selected_before_dedup_ids=selected,
        dedup_excluded_ids=excluded,
        not_admitted_ids=not_admitted,
        admitted_delta=admitted,
        query_row_receipt_sha256=row_receipt,
        binding_sha256=identity_sha256(binding),
    )


def build_partition_payload_adapter(
    source_population: MatchedS0Population,
    generation: PartitionScanV2Generation,
    *,
    max_prompt_tokens: int = DEFAULT_COMPRESSION_PROMPT_CAP,
) -> QueryFactAdapterPopulation:
    """Project a fully verified v2 generation into the shared answer adapter."""

    if type(source_population) is not MatchedS0Population:
        raise TypeError("source_population must be an exact MatchedS0Population")
    if type(generation) is not PartitionScanV2Generation:
        raise TypeError("generation must be an exact PartitionScanV2Generation")
    if (
        type(max_prompt_tokens) is not int
        or not 1 <= max_prompt_tokens <= DEFAULT_COMPRESSION_PROMPT_CAP
    ):
        raise PartitionPayloadAdapterError(
            "max_prompt_tokens must fit the shared routed-prompt cap"
        )
    _require(
        generation.retrieval_sha256 == source_population.retrieval_sha256
        and generation.population_identity_sha256
        == source_population.snapshot.population_identity_sha256
        and len(generation.questions) == len(source_population.rows),
        "partition generation changed the matched source population",
    )
    rows = tuple(
        _project_row(source, question, max_prompt_tokens=max_prompt_tokens)
        for source, question in zip(
            source_population.rows,
            generation.questions,
            strict=True,
        )
    )
    prompts = preflight_fast_completion_prompts(
        [row.compression_prompt.as_mappings() for row in rows],
        max_prompt_tokens=max_prompt_tokens,
    )
    _require(
        tuple(row.compression_prompt.messages_sha256 for row in rows)
        == tuple(row.messages_sha256 for row in prompts.ordered_rows),
        "partition adapter compression prompt order changed",
    )
    generation_sha = (
        generation.source_generation_sha256
        or generation.generation_identity_sha256
    )
    prompt_binding_sha = identity_sha256(
        {
            "format": PROMPT_BINDING_FORMAT,
            "generation_sha256": generation_sha,
            "question_binding_sha256s": [row.binding_sha256 for row in rows],
        }
    )
    partition_population_id = identity_sha256(
        {
            "format": POPULATION_FORMAT,
            "generation_sha256": generation_sha,
            "mechanism_id": MECHANISM_ID,
            "population_identity_sha256": generation.population_identity_sha256,
            "source_population_id": source_population.population_id,
        }
    )
    body = {
        "compression_prompt_population_sha256": prompts.prompt_population_sha256,
        "format": ADAPTER_FORMAT,
        "generation_sha256": generation_sha,
        "max_prompt_tokens": max_prompt_tokens,
        "mechanism_id": MECHANISM_ID,
        "partition_population_id": partition_population_id,
        "prompt_binding_sha256": prompt_binding_sha,
        "question_binding_sha256s": [row.binding_sha256 for row in rows],
        "retrieval_sha256": source_population.retrieval_sha256,
        "source_population_id": source_population.population_id,
    }
    assert_gold_blind(body, path="partition_payload_adapter.population")
    return QueryFactAdapterPopulation(
        source_population=source_population,
        query_preflight_sha256=generation_sha,
        query_run_sha256=generation_sha,
        query_population_id=partition_population_id,
        query_prompt_population_sha256=prompt_binding_sha,
        rows=rows,
        compression_prompt_population=prompts,
        max_prompt_tokens=max_prompt_tokens,
        population_id=identity_sha256(body),
    )


def load_partition_payload_adapter(
    retrieval_path: str | Path,
    *,
    generation_path: str | Path,
    expected_retrieval_sha256: str,
    expected_source_population_id: str,
    expected_generation_sha256: str,
    expected_eligibility_manifest_sha256: str,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    max_prompt_tokens: int = DEFAULT_COMPRESSION_PROMPT_CAP,
) -> QueryFactAdapterPopulation:
    """Load and fully verify the sealed locked partition payload population."""

    source = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    _require(
        source.population_id == expected_source_population_id,
        "partition payload source population identity changed",
    )
    generation = load_partition_scan_v2_generation(
        str(generation_path),
        expected_generation_sha256=expected_generation_sha256,
        population=source,
        expected_eligibility_manifest_sha256=(
            expected_eligibility_manifest_sha256
        ),
    )
    return build_partition_payload_adapter(
        source,
        generation,
        max_prompt_tokens=max_prompt_tokens,
    )


__all__ = [
    "ADAPTER_FORMAT",
    "DELTA_TIER",
    "PartitionPayloadAdapterError",
    "build_partition_payload_adapter",
    "load_partition_payload_adapter",
]
