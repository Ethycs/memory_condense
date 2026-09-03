"""Validated result models and evidence projections for cumulative retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from memory_condense.domain._discourse_identity import make_bundle_id
from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    EvidenceBundle,
    EvidencePacket,
    ObligationResult,
    identity_sha256,
    quote_sha256,
)
from memory_condense.eval._recall_guarded_cumulative_contracts import (
    _CUMULATIVE_STAGE_IDS,
    _atom_evidence_id,
    _freeze_messages,
    _nonempty,
    _ordered_unique,
    _protected_evidence_id,
    CausalCoveragePredecessor,
    CumulativeRetrievalLadder,
    NovelClosureProjection,
    NovelClosureProjectionReceipt,
    ProtectedExcerpt,
    RecallGuardedCumulativeReceipt,
)
from memory_condense.eval._retrieval_qa_prompt import (
    QA_NO_CONTEXT,
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
)
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPlan,
    EpisodeRetrievalPlan,
)


def _novel_closure_projection(
    plan: ClosurePlan,
    protected_excerpts: Sequence[ProtectedExcerpt],
    admitted_atoms: Sequence[Any],
) -> NovelClosureProjection:
    """Project only already visible evidence while preserving supplemental spans.

    A protected v3 excerpt may be a sentence or token-truncated prefix of its
    source chunk.  Therefore chunk identity alone is not a duplicate test.  An
    atom is excluded only when its exact bytes are already covered by a
    protected excerpt, or when that exact atom was admitted by an earlier
    cumulative stage.
    """

    protected = tuple(protected_excerpts)
    prior_atoms = tuple(admitted_atoms)
    prior_atom_ids = {item.atom_id for item in prior_atoms}
    excluded = tuple(
        atom.atom_id
        for atom in plan.atoms
        if atom.atom_id in prior_atom_ids
        or any(
            atom.span.chunk_id == excerpt.chunk_id
            and atom.text in excerpt.text
            for excerpt in protected
        )
    )
    excluded_set = set(excluded)
    candidate_atoms = tuple(
        atom for atom in plan.atoms if atom.atom_id not in excluded_set
    )
    allowed_atom_ids = {item.atom_id for item in candidate_atoms}
    old_to_new: dict[str, str] = {}
    projected_by_id: dict[str, EvidenceBundle] = {}
    mixed_old_bundle_ids: list[str] = []
    mixed_new_bundle_ids: set[str] = set()
    for bundle in plan.bundles:
        atom_ids = tuple(
            atom_id for atom_id in bundle.atom_ids if atom_id in allowed_atom_ids
        )
        if not atom_ids:
            continue
        mixed = atom_ids != bundle.atom_ids
        if mixed:
            mixed_old_bundle_ids.append(bundle.bundle_id)
        bundle_id = make_bundle_id(
            atom_ids=atom_ids,
            obligation_ids=bundle.obligation_ids,
            unit_ids=() if mixed else bundle.unit_ids,
            relation_ids=() if mixed else bundle.relation_ids,
        )
        projected = replace(
            bundle,
            bundle_id=bundle_id,
            atom_ids=atom_ids,
            unit_ids=() if mixed else bundle.unit_ids,
            relation_ids=() if mixed else bundle.relation_ids,
            utility=0.0 if mixed else bundle.utility,
        )
        if mixed:
            mixed_new_bundle_ids.add(bundle_id)
        current = projected_by_id.get(bundle_id)
        if current is None or projected.utility > current.utility:
            projected_by_id[bundle_id] = projected
        old_to_new[bundle.bundle_id] = bundle_id
    bundles = tuple(projected_by_id.values())
    referenced_atom_ids = {
        atom_id for bundle in bundles for atom_id in bundle.atom_ids
    }
    atoms = tuple(
        atom for atom in candidate_atoms if atom.atom_id in referenced_atom_ids
    )
    bundle_by_id = {item.bundle_id: item for item in bundles}
    obligation_by_id = {
        item.obligation_id: item for item in plan.query_program.obligations
    }
    results: list[ObligationResult] = []
    for result in plan.obligation_results:
        bundle_ids = tuple(
            dict.fromkeys(
                old_to_new[item]
                for item in result.bundle_ids
                if item in old_to_new
            )
        )
        owned = tuple(
            bundle_by_id[item]
            for item in bundle_ids
            if result.obligation_id in bundle_by_id[item].obligation_ids
        )
        unit_ids = tuple(
            dict.fromkeys(item for bundle in owned for item in bundle.unit_ids)
        )
        relation_ids = tuple(
            dict.fromkeys(item for bundle in owned for item in bundle.relation_ids)
        )
        support = max(len(unit_ids), len(relation_ids), len(bundle_ids))
        status = result.status
        reason = result.reason
        depends_on_predecessor = any(
            item.bundle_id in mixed_new_bundle_ids for item in owned
        )
        if status == "satisfied" and (
            depends_on_predecessor
            or support < obligation_by_id[result.obligation_id].min_count
        ):
            status = "not_found"
            reason = "protected_predecessor_dependency"
        results.append(
            replace(
                result,
                status=status,
                unit_ids=unit_ids,
                relation_ids=relation_ids,
                bundle_ids=bundle_ids,
                reason=reason,
            )
        )
    required = {
        item.obligation_id
        for item in plan.query_program.obligations
        if item.required
    }
    satisfied = {
        item.obligation_id for item in results if item.status == "satisfied"
    }
    complete = bool(
        required <= satisfied
        and plan.stopping_reason == "complete"
        and plan.scope_witnesses
        and all(item.exhaustive for item in plan.scope_witnesses)
    )
    projected_plan = replace(
        plan,
        atoms=atoms,
        bundles=bundles,
        obligation_results=tuple(results),
        direct_chunk_ids=tuple(
            item
            for item in plan.direct_chunk_ids
            if item in {atom.span.chunk_id for atom in atoms}
        ),
        complete_claimed=complete,
        plan_sha256="",
    )
    protected_projection = identity_sha256(
        {
            "protected_excerpts": [
                item.identity_payload() for item in protected
            ],
            "admitted_atoms": [item.identity_payload() for item in prior_atoms],
        }
    )
    receipt = NovelClosureProjectionReceipt(
        source_plan_sha256=plan.plan_sha256,
        protected_evidence_projection_sha256=protected_projection,
        excluded_atom_ids=excluded,
        mixed_bundle_ids=tuple(mixed_old_bundle_ids),
        projected_plan_sha256=projected_plan.plan_sha256,
    )
    return NovelClosureProjection(plan=projected_plan, receipt=receipt)


def _addition_prompt_prefix(
    prompt_question: str,
    protected_context: str,
    protected_count: int,
) -> tuple[str, str]:
    if QA_USER_TEMPLATE.count("{context}") != 1:
        raise RuntimeError("QA_USER_TEMPLATE must contain exactly one context slot")
    prefix_template, suffix_template = QA_USER_TEMPLATE.split("{context}", 1)
    prefix = prefix_template.format(question=prompt_question)
    suffix = suffix_template.format(question=prompt_question)
    if protected_context:
        prefix += protected_context + f"\n[{protected_count + 1}] "
    else:
        prefix += "[1] "
    return prefix, suffix


def _stage_evidence_projection_sha256(
    predecessor: CausalCoveragePredecessor,
    admitted_atoms: Sequence[Any],
) -> str:
    return identity_sha256(
        {
            "protected_excerpts": [
                item.identity_payload() for item in predecessor.excerpts
            ],
            "admitted_atoms": [item.identity_payload() for item in admitted_atoms],
        }
    )


@dataclass(frozen=True, slots=True)
class RecallGuardedCumulativeRetrieval:
    """Gold-blind final prompt and all immutable intermediate evidence."""

    predecessor: CausalCoveragePredecessor
    episode_expansion: EpisodeRetrievalPlan
    representative_expansion: EpisodeRepresentativeRetrievalPlan
    closure_plans: tuple[ClosurePlan, ...]
    novel_projections: tuple[NovelClosureProjection, ...]
    addition_packets: tuple[EvidencePacket | None, ...]
    ladder: CumulativeRetrievalLadder
    prompt_question: str
    context: str
    messages: tuple[Mapping[str, str], ...]
    receipt: RecallGuardedCumulativeReceipt

    def __post_init__(self) -> None:
        if self.receipt.matched_controls_sha256 != (
            self.predecessor.receipt.matched_controls_sha256
        ):
            raise ValueError("cumulative result changed matched controls")
        if self.receipt.representative_runtime_certified != (
            self.representative_expansion.runtime_binding_certified
        ):
            raise ValueError("cumulative result changed representative runtime binding")
        plans = tuple(self.closure_plans)
        projections = tuple(self.novel_projections)
        packets = tuple(self.addition_packets)
        if len(plans) != 3 or len(projections) != 3 or len(packets) != 3:
            raise ValueError("cumulative result requires three additive methods")
        if tuple(item.plan_sha256 for item in plans) != (
            self.receipt.closure_plan_sha256s
        ):
            raise ValueError("cumulative result changed a closure plan")
        if tuple(item.receipt.receipt_sha256 for item in projections) != (
            self.receipt.novel_projection_receipt_sha256s
        ):
            raise ValueError("cumulative result changed a novel projection")
        if any(
            projection.receipt.source_plan_sha256 != plan.plan_sha256
            for plan, projection in zip(plans, projections, strict=True)
        ):
            raise ValueError("novel projection belongs to another closure plan")
        packet_hashes = tuple(
            None if item is None else item.receipt.receipt_sha256
            for item in packets
        )
        if packet_hashes != self.receipt.addition_packet_receipt_sha256s:
            raise ValueError("cumulative result changed an addition packet")
        for packet in packets:
            if packet is None:
                continue
            packet_receipt = packet.receipt
            if packet_receipt.max_context_token_proxy > (
                self.receipt.max_context_token_proxy
            ):
                raise ValueError("addition packet changed the cumulative context cap")
            if packet_receipt.max_prompt_token_proxy != (
                self.receipt.max_prompt_token_proxy
                + self.receipt.responder_output_token_reserve
            ):
                raise ValueError("addition packet changed the cumulative prompt cap")
            if packet_receipt.responder_output_token_reserve != (
                self.receipt.responder_output_token_reserve
            ):
                raise ValueError("addition packet changed the responder reserve")
        if len(self.ladder.stages) != 4 or tuple(
            item.stage_id for item in self.ladder.stages
        ) != _CUMULATIVE_STAGE_IDS:
            raise ValueError("cumulative result changed the four-stage ladder")
        root_stage = self.ladder.stages[0]
        if (
            root_stage.max_context_token_proxy,
            root_stage.max_prompt_token_proxy,
            root_stage.responder_output_token_reserve,
        ) != (
            self.receipt.max_context_token_proxy,
            self.receipt.max_prompt_token_proxy,
            self.receipt.responder_output_token_reserve,
        ):
            raise ValueError("cumulative receipt changed the ladder hard budgets")
        if self.predecessor.receipt.max_prompt_token_proxy != (
            self.receipt.max_prompt_token_proxy
        ):
            raise ValueError("cumulative receipt changed the predecessor prompt cap")
        if self.predecessor.receipt.responder_output_token_reserve != (
            self.receipt.responder_output_token_reserve
        ):
            raise ValueError("cumulative receipt changed the responder reserve")
        if tuple(
            item.admission_status for item in self.ladder.stages[1:]
        ) != self.receipt.stage_admission_statuses:
            raise ValueError("cumulative receipt changed stage admission decisions")
        prompt = _nonempty(self.prompt_question, "prompt_question")
        if identity_sha256({"prompt_question": prompt}) != (
            self.predecessor.receipt.prompt_question_sha256
        ):
            raise ValueError("cumulative result changed the prompt question")

        protected_ids = tuple(
            _protected_evidence_id(item) for item in self.predecessor.excerpts
        )
        if self.receipt.protected_chunk_ids != (
            self.predecessor.receipt.protected_chunk_ids
        ):
            raise ValueError("cumulative receipt changed protected chunk IDs")
        if protected_ids != self.receipt.protected_evidence_ids:
            raise ValueError("cumulative result changed protected coordinates")
        current_context = self.predecessor.protected_context
        current_messages = self.predecessor.messages
        current_coordinates = protected_ids
        entry_count = len(self.predecessor.excerpts)
        admitted_atoms: list[Any] = []
        expected_stage_contexts = [current_context or QA_NO_CONTEXT]
        expected_stage_messages = [current_messages]
        expected_stage_projections = [
            _stage_evidence_projection_sha256(self.predecessor, ())
        ]
        expected_added_coordinates: list[tuple[str, ...]] = []
        expected_admission_statuses: list[str] = []
        for plan, projection, packet in zip(
            plans,
            projections,
            packets,
            strict=True,
        ):
            expected_projection = _novel_closure_projection(
                plan,
                self.predecessor.excerpts,
                admitted_atoms,
            )
            if expected_projection.receipt.receipt_sha256 != (
                projection.receipt.receipt_sha256
            ):
                raise ValueError("cumulative result changed a projection proof")
            if expected_projection.plan.plan_sha256 != projection.plan.plan_sha256:
                raise ValueError("cumulative result changed a projected plan")
            if packet is not None and packet.receipt.plan_sha256 != (
                projection.plan.plan_sha256
            ):
                raise ValueError("addition packet belongs to another projected plan")
            packet_atoms = () if packet is None else tuple(packet.atoms)
            added_coordinates = tuple(_atom_evidence_id(item) for item in packet_atoms)
            if set(added_coordinates) & set(current_coordinates):
                raise ValueError("cumulative result admitted duplicate evidence")
            if packet_atoms:
                prefix, suffix = _addition_prompt_prefix(
                    prompt,
                    current_context,
                    entry_count,
                )
                next_messages = _freeze_messages(
                    (
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": prefix + packet.context + suffix,
                        },
                    )
                )
                if identity_sha256(list(next_messages)) != (
                    packet.receipt.prompt_messages_sha256
                ):
                    raise ValueError("addition packet changed its cumulative prompt")
                current_context = (
                    f"{current_context}\n[{entry_count + 1}] {packet.context}"
                    if current_context
                    else f"[1] {packet.context}"
                )
                current_messages = next_messages
                current_coordinates = (*current_coordinates, *added_coordinates)
                entry_count += 1
                admitted_atoms.extend(packet_atoms)
                expected_admission_statuses.append("added")
            elif not projection.plan.atoms or not projection.plan.bundles:
                expected_admission_statuses.append("no_novel_evidence")
            else:
                expected_admission_statuses.append("budget_exhausted")
            expected_added_coordinates.append(added_coordinates)
            expected_stage_contexts.append(current_context or QA_NO_CONTEXT)
            expected_stage_messages.append(current_messages)
            expected_stage_projections.append(
                _stage_evidence_projection_sha256(
                    self.predecessor,
                    admitted_atoms,
                )
            )

        if tuple(expected_admission_statuses) != (
            self.receipt.stage_admission_statuses
        ):
            raise ValueError("cumulative receipt changed stage admission decisions")

        stages = self.ladder.stages
        if stages[0].selected_evidence_ids != protected_ids:
            raise ValueError("root stage changed protected evidence")
        for index, projection in enumerate(projections, 1):
            stage = stages[index]
            if stage.method_evidence_sha256 != projection.receipt.receipt_sha256:
                raise ValueError("cumulative stage changed its method evidence")
            if stage.added_evidence_ids != expected_added_coordinates[index - 1]:
                raise ValueError("cumulative stage changed admitted atom coordinates")
        for stage, stage_context, stage_messages in zip(
            stages,
            expected_stage_contexts,
            expected_stage_messages,
            strict=True,
        ):
            if quote_sha256(stage_context) != stage.context_sha256:
                raise ValueError("cumulative stage context projection changed")
            if identity_sha256(list(stage_messages)) != stage.prompt_messages_sha256:
                raise ValueError("cumulative stage prompt projection changed")
            if count_tokens(stage_context) != stage.context_token_proxy:
                raise ValueError("cumulative stage context accounting changed")
            if count_chat_prompt_token_proxy(stage_messages) != stage.prompt_token_proxy:
                raise ValueError("cumulative stage prompt accounting changed")
        if tuple(item.evidence_projection_sha256 for item in stages) != tuple(
            expected_stage_projections
        ):
            raise ValueError("cumulative stage evidence projection changed")
        atom_ids = tuple(item.atom_id for item in admitted_atoms)
        if atom_ids != self.receipt.added_atom_ids:
            raise ValueError("cumulative receipt changed admitted atom IDs")
        added_chunk_ids = _ordered_unique(
            tuple(item.span.chunk_id for item in admitted_atoms)
        )
        if added_chunk_ids != self.receipt.added_chunk_ids:
            raise ValueError("cumulative receipt changed admitted chunk IDs")
        if self.receipt.final_chunk_ids != _ordered_unique(
            (*self.receipt.protected_chunk_ids, *added_chunk_ids)
        ):
            raise ValueError("cumulative receipt changed final chunk IDs")
        if current_coordinates != self.receipt.final_evidence_ids:
            raise ValueError("cumulative receipt changed final evidence coordinates")
        if identity_sha256(
            [item.identity_payload() for item in admitted_atoms]
        ) != self.receipt.addition_evidence_projection_sha256:
            raise ValueError("cumulative receipt changed addition evidence")
        if current_context or admitted_atoms:
            expected_final_context = current_context or QA_NO_CONTEXT
        else:  # pragma: no cover - predecessor prompt already covers this case
            expected_final_context = QA_NO_CONTEXT
        if self.context != expected_final_context:
            raise ValueError("cumulative result is not assembled from its stage prefix")
        frozen_messages = _freeze_messages(self.messages)
        if tuple(frozen_messages) != tuple(current_messages):
            raise ValueError("cumulative result messages are not the final stage prompt")
        if quote_sha256(self.context) != self.receipt.final_context_sha256:
            raise ValueError("cumulative result context changed")
        if identity_sha256(list(frozen_messages)) != self.receipt.prompt_messages_sha256:
            raise ValueError("cumulative result messages changed")
        if count_tokens(self.context) != self.receipt.context_token_proxy:
            raise ValueError("cumulative result context accounting changed")
        if count_chat_prompt_token_proxy(frozen_messages) != self.receipt.prompt_token_proxy:
            raise ValueError("cumulative result prompt accounting changed")
        object.__setattr__(self, "closure_plans", plans)
        object.__setattr__(self, "novel_projections", projections)
        object.__setattr__(self, "addition_packets", packets)
        object.__setattr__(self, "prompt_question", prompt)
        object.__setattr__(self, "messages", frozen_messages)

    def provider_messages(self) -> list[dict[str, str]]:
        return [dict(message) for message in self.messages]

    def provider_messages_by_stage(self) -> dict[str, list[dict[str, str]]]:
        """Return detached provider-ready prompts for every cumulative arm."""

        prompt = self.prompt_question
        context = self.predecessor.protected_context
        entry_count = len(self.predecessor.excerpts)
        messages = self.predecessor.messages
        views: dict[str, list[dict[str, str]]] = {
            self.ladder.stages[0].stage_id: [dict(item) for item in messages]
        }
        for stage, packet in zip(
            self.ladder.stages[1:],
            self.addition_packets,
            strict=True,
        ):
            if packet is not None and packet.atoms:
                prefix, suffix = _addition_prompt_prefix(
                    prompt,
                    context,
                    entry_count,
                )
                messages = _freeze_messages(
                    (
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": prefix + packet.context + suffix,
                        },
                    )
                )
                context = (
                    f"{context}\n[{entry_count + 1}] {packet.context}"
                    if context
                    else f"[1] {packet.context}"
                )
                entry_count += 1
            if identity_sha256(list(messages)) != stage.prompt_messages_sha256:
                raise RuntimeError("cumulative stage prompt cannot be reconstructed")
            views[stage.stage_id] = [dict(item) for item in messages]
        return views

    @property
    def retrieved_source_ids(self) -> tuple[str, ...]:
        values = [item.source_id for item in self.predecessor.excerpts]
        for packet in self.addition_packets:
            if packet is None:
                continue
            values.extend(
                atom.span.source_id
                for atom in packet.atoms
                if atom.span.source_id is not None
            )
        return tuple(dict.fromkeys(values))

    @property
    def addition_packet(self) -> EvidencePacket | None:
        """Compatibility projection: the last non-empty additive packet."""

        return next(
            (item for item in reversed(self.addition_packets) if item is not None),
            None,
        )


@dataclass(frozen=True, slots=True)
class RecallGuardedCumulativeStageMetrics:
    stage_id: str
    answer_present: bool
    best_evidence_f1: float
    retrieved_source_ids: tuple[str, ...]
    evidence_source_recall: float | None
    answer_value_components_expected: int | None
    answer_value_components_found: int | None
    answer_value_component_recall: float | None
    all_answer_value_components: bool | None
    answer_value_component_hit_mask: tuple[bool, ...]
    answer_value_metric_kind: str
    context_token_proxy: int
    prompt_token_proxy: int


@dataclass(frozen=True, slots=True)
class RecallGuardedCumulativeMetrics:
    question_id: str
    retrieval_receipt_sha256: str
    answer_present: bool
    best_evidence_f1: float
    expected_source_ids: tuple[str, ...]
    retrieved_source_ids: tuple[str, ...]
    evidence_source_recall: float | None
    any_evidence_source: bool | None
    all_evidence_sources: bool | None
    answer_value_components_expected: int | None
    answer_value_components_found: int | None
    answer_value_component_recall: float | None
    all_answer_value_components: bool | None
    answer_value_component_hit_mask: tuple[bool, ...]
    answer_value_metric_kind: str
    protected_excerpts: int
    added_atoms: int
    hard_budget_compliant: bool
    context_token_proxy: int
    prompt_token_proxy: int
    prompt_workspace_token_proxy: int
    stages: tuple[RecallGuardedCumulativeStageMetrics, ...]
