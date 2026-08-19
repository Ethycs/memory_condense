"""Offline metrics for source-grounded diffuse retrieval packets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean
from collections.abc import Callable
from typing import Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import (
    ClosurePlan,
    EvidencePacket,
    EvidenceSpan,
    quote_sha256,
)
from memory_condense.search.packing.evidence_packet import render_evidence_context


@dataclass(frozen=True, slots=True)
class GoldEvidenceSet:
    """One annotated minimal sufficient evidence solution."""

    atom_ids: frozenset[str]
    relation_ids: frozenset[str] = frozenset()
    atom_weights: Mapping[str, float] | None = None
    relation_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.atom_ids:
            raise ValueError("a gold minimal set requires at least one atom")
        for label, values, known in (
            ("atom_weights", self.atom_weights or {}, self.atom_ids),
            ("relation_weights", self.relation_weights or {}, self.relation_ids),
        ):
            if not set(values) <= set(known):
                raise ValueError(f"{label} contains an unknown ID")
            if any(not math.isfinite(float(value)) or value <= 0 for value in values.values()):
                raise ValueError(f"{label} must contain finite positive weights")


@dataclass(frozen=True, slots=True)
class DiffuseRetrievalGold:
    question_id: str
    snapshot_sha256: str
    artifact_id: str | None
    required_obligation_ids: frozenset[str]
    minimal_sets: tuple[GoldEvidenceSet, ...]
    evidence_path_relation_ids: frozenset[str] = frozenset()
    revision_terminal_unit_ids: frozenset[str] = frozenset()
    contradiction_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        snapshot_sha256 = str(self.snapshot_sha256).casefold()
        if re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256) is None:
            raise ValueError("snapshot_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "snapshot_sha256", snapshot_sha256)
        if self.artifact_id is not None:
            artifact_id = str(self.artifact_id).strip()
            if not artifact_id:
                raise ValueError("artifact_id must be non-empty when supplied")
            object.__setattr__(self, "artifact_id", artifact_id)
        if not self.required_obligation_ids:
            raise ValueError("gold requires at least one required obligation")
        if not self.minimal_sets:
            raise ValueError("gold requires at least one minimal sufficient set")
        normalized_pairs = tuple(
            sorted(
                {
                    tuple(sorted((left, right)))
                    for left, right in self.contradiction_pairs
                    if left and right and left != right
                }
            )
        )
        object.__setattr__(self, "contradiction_pairs", normalized_pairs)


@dataclass(frozen=True, slots=True)
class DiffuseRetrievalMetrics:
    question_id: str
    minimal_set_hit: float
    soft_closure: float
    required_obligation_coverage: float
    required_obligation_complete: float
    evidence_path_recall: float
    revision_terminal_recall: float
    contradiction_pair_recall: float
    evidence_item_precision: float
    distractor_item_fraction: float
    false_complete: float
    context_token_proxy: int
    prompt_token_proxy: int | None
    prompt_workspace_token_proxy: int | None
    max_prompt_token_proxy: int | None
    hard_budget_compliant: bool
    source_span_hash_valid: bool


def _recall(selected: set[str], gold: frozenset[str]) -> float:
    if not gold:
        return 1.0
    return len(selected & set(gold)) / len(gold)


def _soft_set_coverage(
    gold: GoldEvidenceSet,
    *,
    selected_atoms: set[str],
    selected_relations: set[str],
) -> float:
    atom_weights = {
        atom_id: float((gold.atom_weights or {}).get(atom_id, 1.0))
        for atom_id in gold.atom_ids
    }
    relation_weights = {
        relation_id: float((gold.relation_weights or {}).get(relation_id, 1.0))
        for relation_id in gold.relation_ids
    }
    denominator = sum(atom_weights.values()) + sum(relation_weights.values())
    numerator = sum(
        weight for atom_id, weight in atom_weights.items() if atom_id in selected_atoms
    ) + sum(
        weight
        for relation_id, weight in relation_weights.items()
        if relation_id in selected_relations
    )
    return numerator / denominator if denominator else 1.0


def measure_diffuse_retrieval(
    gold: DiffuseRetrievalGold,
    *,
    plan: ClosurePlan,
    packet: EvidencePacket,
    hydrate_span: Callable[[EvidenceSpan], str],
) -> DiffuseRetrievalMetrics:
    """Measure the final packet, never the larger pre-budget candidate graph."""
    if plan.snapshot.snapshot_sha256 != gold.snapshot_sha256:
        raise ValueError("gold evidence belongs to another frozen graph snapshot")
    if gold.artifact_id != plan.artifact_id:
        raise ValueError("gold evidence belongs to another discourse artifact")
    _validate_packet_against_plan(plan, packet)
    selected_atoms = {item.atom_id for item in packet.atoms}
    selected_relations = {
        relation_id for bundle in packet.bundles for relation_id in bundle.relation_ids
    }
    selected_units = {
        unit_id for bundle in packet.bundles for unit_id in bundle.unit_ids
    }
    selected_obligations = {
        obligation_id
        for bundle in packet.bundles
        for obligation_id in bundle.obligation_ids
    }
    minimal_hit = any(
        set(item.atom_ids) <= selected_atoms
        and set(item.relation_ids) <= selected_relations
        for item in gold.minimal_sets
    )
    soft_closure = max(
        _soft_set_coverage(
            item,
            selected_atoms=selected_atoms,
            selected_relations=selected_relations,
        )
        for item in gold.minimal_sets
    )
    obligation_coverage = _recall(
        selected_obligations,
        gold.required_obligation_ids,
    )
    contradiction_recall = (
        mean(
            float(left in selected_units and right in selected_units)
            for left, right in gold.contradiction_pairs
        )
        if gold.contradiction_pairs
        else 1.0
    )
    hash_valid = True
    for item in packet.atoms:
        try:
            authoritative = hydrate_span(item.span)
        except Exception:  # noqa: BLE001 - a resolver failure means invalid evidence
            hash_valid = False
            break
        if (
            authoritative != item.text
            or item.span.quote_sha256 != quote_sha256(authoritative)
        ):
            hash_valid = False
            break
    selected_item_count = len(selected_atoms) + len(selected_relations)
    evidence_item_precision = max(
        (
            (
                len(selected_atoms & set(item.atom_ids))
                + len(selected_relations & set(item.relation_ids))
            )
            / selected_item_count
            if selected_item_count
            else 0.0
        )
        for item in gold.minimal_sets
    )
    return DiffuseRetrievalMetrics(
        question_id=gold.question_id,
        minimal_set_hit=float(minimal_hit),
        soft_closure=soft_closure,
        required_obligation_coverage=obligation_coverage,
        required_obligation_complete=float(
            set(gold.required_obligation_ids) <= selected_obligations
        ),
        evidence_path_recall=_recall(
            selected_relations,
            gold.evidence_path_relation_ids,
        ),
        revision_terminal_recall=_recall(
            selected_units,
            gold.revision_terminal_unit_ids,
        ),
        contradiction_pair_recall=contradiction_recall,
        evidence_item_precision=evidence_item_precision,
        distractor_item_fraction=1.0 - evidence_item_precision,
        false_complete=float(packet.receipt.complete_claimed and not minimal_hit),
        context_token_proxy=packet.receipt.context_token_proxy,
        prompt_token_proxy=packet.receipt.prompt_token_proxy,
        prompt_workspace_token_proxy=(
            packet.receipt.prompt_workspace_token_proxy
        ),
        max_prompt_token_proxy=packet.receipt.max_prompt_token_proxy,
        hard_budget_compliant=(
            packet.receipt.context_token_proxy
            <= packet.receipt.max_context_token_proxy
            and (
                packet.receipt.prompt_workspace_token_proxy is None
                or (
                    packet.receipt.max_prompt_token_proxy is not None
                    and packet.receipt.prompt_workspace_token_proxy
                    <= packet.receipt.max_prompt_token_proxy
                )
            )
        ),
        source_span_hash_valid=hash_valid,
    )


def _validate_packet_against_plan(
    plan: ClosurePlan,
    packet: EvidencePacket,
) -> None:
    if packet.receipt.plan_sha256 != plan.plan_sha256:
        raise ValueError("packet receipt belongs to another closure plan")
    plan_atoms = {item.atom_id: item for item in plan.atoms}
    plan_bundles = {item.bundle_id: item for item in plan.bundles}
    if any(plan_atoms.get(item.atom_id) != item for item in packet.atoms):
        raise ValueError("packet contains an atom outside its closure plan")
    if any(plan_bundles.get(item.bundle_id) != item for item in packet.bundles):
        raise ValueError("packet contains a bundle outside its closure plan")
    selected_atoms = {item.atom_id for item in packet.atoms}
    if any(
        not set(bundle.atom_ids) <= selected_atoms
        for bundle in packet.bundles
    ):
        raise ValueError("packet contains a partial atomic evidence bundle")
    expected_context = render_evidence_context(packet.atoms, packet.bundles)
    if packet.context != expected_context:
        raise ValueError("packet context is not the canonical selected evidence")
    encoding = packet.receipt.tokenizer_identity.split(":", 1)[0]
    if count_tokens(packet.context, encoding=encoding) != (
        packet.receipt.context_token_proxy
    ):
        raise ValueError("packet context token proxy does not match its receipt")


@dataclass(frozen=True, slots=True)
class DiffuseRetrievalAggregate:
    questions: int
    minimal_set_hit: float
    soft_closure: float
    required_obligation_coverage: float
    required_obligation_complete: float
    evidence_path_recall: float
    revision_terminal_recall: float
    contradiction_pair_recall: float
    evidence_item_precision: float
    distractor_item_fraction: float
    false_complete_rate: float
    mean_context_token_proxy: float
    mean_prompt_token_proxy: float | None
    mean_prompt_workspace_token_proxy: float | None
    prompt_token_proxy_availability: float
    hard_budget_compliance: float
    source_span_hash_validity: float


def aggregate_diffuse_retrieval(
    rows: Sequence[DiffuseRetrievalMetrics],
) -> DiffuseRetrievalAggregate:
    if not rows:
        raise ValueError("at least one diffuse retrieval result is required")
    prompt_counts = [
        item.prompt_token_proxy
        for item in rows
        if item.prompt_token_proxy is not None
    ]
    workspace_counts = [
        item.prompt_workspace_token_proxy
        for item in rows
        if item.prompt_workspace_token_proxy is not None
    ]
    return DiffuseRetrievalAggregate(
        questions=len(rows),
        minimal_set_hit=mean(item.minimal_set_hit for item in rows),
        soft_closure=mean(item.soft_closure for item in rows),
        required_obligation_coverage=mean(
            item.required_obligation_coverage for item in rows
        ),
        required_obligation_complete=mean(
            item.required_obligation_complete for item in rows
        ),
        evidence_path_recall=mean(item.evidence_path_recall for item in rows),
        revision_terminal_recall=mean(
            item.revision_terminal_recall for item in rows
        ),
        contradiction_pair_recall=mean(
            item.contradiction_pair_recall for item in rows
        ),
        evidence_item_precision=mean(item.evidence_item_precision for item in rows),
        distractor_item_fraction=mean(
            item.distractor_item_fraction for item in rows
        ),
        false_complete_rate=mean(item.false_complete for item in rows),
        mean_context_token_proxy=mean(item.context_token_proxy for item in rows),
        mean_prompt_token_proxy=(mean(prompt_counts) if prompt_counts else None),
        mean_prompt_workspace_token_proxy=(
            mean(workspace_counts) if workspace_counts else None
        ),
        prompt_token_proxy_availability=len(prompt_counts) / len(rows),
        hard_budget_compliance=mean(float(item.hard_budget_compliant) for item in rows),
        source_span_hash_validity=mean(
            float(item.source_span_hash_valid) for item in rows
        ),
    )


__all__ = [
    "DiffuseRetrievalAggregate",
    "DiffuseRetrievalGold",
    "DiffuseRetrievalMetrics",
    "GoldEvidenceSet",
    "aggregate_diffuse_retrieval",
    "measure_diffuse_retrieval",
]
