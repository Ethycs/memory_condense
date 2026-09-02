"""Common typed-memory final-answer plane.

This module is deliberately provider agnostic.  It composes already verified
typed evidence, exposes only opaque ``H``/``G`` handles to the model, trims the
weakest evidence until the *complete* chat prompt plus its output reserve fits
the hard 8k envelope, and validates a strict final decision.  Exact citations
and source locators remain in the caller-supplied local audit projection.

The direct parent prediction is protected fallback state, never evidence.  A
malformed, unsupported, conflicted, or detail-erasing completion therefore
returns the parent byte-for-byte instead of turning uncertainty into an
abstention or an ungrounded rewrite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.text_numbers import NUMBER_WORDS

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_operator_adapter import (
    ConflictPolicy,
    ContentCoherence,
    EvidenceHandleBinding,
    EvidenceStatus,
    FrontierMode,
    NumericRole,
    ParsedTypedItems,
    ProviderPayloadMode,
    TypedEvidenceItem,
    TypedEvidencePacket,
    build_typed_evidence_packet,
    compact_evidence_content_projection,
    compact_typed_evidence_projection,
    conservative_numeric_value,
)
from .typed_action_semantics import (
    canonical_action_concepts,
    completed_action_concepts,
)
from .typed_operator_executor import (
    ExecutionStatus,
    OperatorExecutionReceipt,
    execute_typed_operator,
)
from .typed_numeric_semantics import NumericQualifier
from .typed_downstream_operator import (
    compile_downstream_operator_overlay,
    execute_downstream_typed_operator,
)
from .typed_operator_spec import (
    AnswerShape,
    ComparisonMode,
    SlotKind,
    TemporalMode,
    normalized_terms,
)
from .selected_evidence_discourse_links import (
    SelectedEvidenceDiscourseLinks,
)


FORMAT = "memory-condense-typed-memory-final-arm-v1"
COMPOSITION_FORMAT = f"{FORMAT}-composition-v1"
PROMPT_ROW_FORMAT = f"{FORMAT}-prompt-row-v1"
DECISION_FORMAT = f"{FORMAT}-decision-v1"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row-v1"
JUDGE_ROW_FORMAT = f"{FORMAT}-judge-row-v1"
VALIDATOR_POLICY_FORMAT = f"{FORMAT}-validator-policy-v3"

HARD_PROMPT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_PROMPT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE
PACKET_CONSTRUCTION_OUTPUT_TOKEN_RESERVE = 1
EXPECTED_QUESTION_COUNT = 100
STORY_LINK_TOKEN_CAP = 256
LOCAL_RETENTION_PRIORITY_WIDTH = 24
VALIDATION_CONTRACT_FORMAT = f"{FORMAT}-completion-validation-contract-v3"

RESOURCE_PRESERVING_SYSTEM_PROMPT_V2 = (
    "Answer one dated long-memory question from the supplied typed evidence. "
    "The protected parent prediction is fallback-not-evidence. Evidence is "
    "identified only by opaque H handles; opaque G handles express story or "
    "source co-membership without revealing source identities. Prefer one G "
    "group, or the smallest consistently linked set of G groups, that jointly "
    "satisfies the question's entity, role, time, and operator slots. Never mix "
    "mutually inconsistent values from unrelated groups. An evidence summary "
    "may be an exact retrieved source chunk rather than one extracted atomic "
    "fact; read that chunk as the supplied substitute context and use only "
    "claims it explicitly supports. A deterministic "
    "execution or scalar advisory is usable only when its cited handles remain "
    "supported by the evidence. When replacing from a supplied advisory, copy "
    "its prediction and used_handle_ids exactly. BOUNDED is a provenance "
    "qualifier, not automatic "
    "insufficiency: replacement is allowed when cited handles coherently answer "
    "the question and all explicit required slots. Keep the protected parent "
    "exactly only when a required slot remains unresolved, conflicts cannot be "
    "resolved, or the cited facts are genuinely insufficient. Preserve safe "
    "numeric wording: approximate, lower-bound, and upper-bound evidence must "
    "not be rewritten as an exact scalar. When the answering evidence supplies "
    "an exact resource title or URL, preserve both exactly in the replacement. "
    "Return one JSON "
    "object and no markdown, "
    "with exactly decision, prediction, used_handle_ids. decision must be "
    "keep_parent or replace. keep_parent requires the exact parent prediction "
    "and an empty handle list. replace requires a nonempty concise prediction "
    "and one or more supplied usable handles."
)
SYSTEM_PROMPT = RESOURCE_PRESERVING_SYSTEM_PROMPT_V2

# Sealed terminal artifacts created before exact resource preservation was
# added must continue to reconstruct their original provider messages.  Keep
# this as an explicit prompt version rather than rewriting historical seals.
_RESOURCE_PRESERVATION_INSTRUCTION = (
    "When the answering evidence supplies an exact resource title or URL, "
    "preserve both exactly in the replacement. "
)
LEGACY_SYSTEM_PROMPT_V1 = RESOURCE_PRESERVING_SYSTEM_PROMPT_V2.replace(
    _RESOURCE_PRESERVATION_INSTRUCTION,
    "",
)

_FORBIDDEN_PROVIDER_KEYS = frozenset(
    {
        "namespace_id",
        "partition_id",
        "question_id",
        "source_id",
        "source_prefix",
        "store_path",
    }
)


class TypedMemoryFinalArmError(MatchedEvalContractError):
    """A typed final-arm invariant changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedMemoryFinalArmError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    _require(
        all(type(value) is str and bool(value) for value in result)
        and len(set(result)) == len(result),
        f"{label} must be ordered unique text",
    )
    return result


def _reject_provider_locator_keys(value: object, *, path: str = "provider") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            _require(
                normalized not in _FORBIDDEN_PROVIDER_KEYS,
                f"raw locator key escaped into {path}.{key}",
            )
            _reject_provider_locator_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_provider_locator_keys(child, path=f"{path}[{index}]")


def _plain_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    result = tuple(dict(row) for row in messages)
    _require(
        bool(result)
        and all(
            set(row) == {"role", "content"}
            and row["role"] in {"system", "user", "assistant"}
            and type(row["content"]) is str
            for row in result
        ),
        "final prompt messages changed schema",
    )
    return result


def _usable_items(packet: TypedEvidencePacket) -> tuple[TypedEvidenceItem, ...]:
    return tuple(
        item
        for item in packet.items
        if item.included
        and (
            not item.content_conflict
            or packet.conflict_policy is ConflictPolicy.FAIL_OPEN
        )
        and item.status is not EvidenceStatus.CANCELLED
        and (
            item.status is not EvidenceStatus.PROPOSED
            or packet.operator_spec.include_proposed
        )
    )


def _mechanism_priority(mechanism_id: str) -> int:
    """Stable evidence-retention order; a larger value is stronger."""

    value = mechanism_id.casefold()
    if "active_reconstruction" in value:
        return 60
    if "full_store" in value or "slot_closure" in value:
        return 50
    if "tail" in value:
        return 40
    if "source" in value or "guided" in value or "direct" in value:
        return 30
    if "map" in value:
        return 20
    return 10


def _weakest_item_key(
    item: TypedEvidenceItem,
    *,
    mechanism_by_handle: Mapping[str, str],
    ordinal_by_receipt: Mapping[str, int],
    local_retention_priority_by_handle: Mapping[str, tuple[int, ...]],
) -> tuple[int, int, tuple[int, ...], int, int, int, str]:
    mechanisms = tuple(
        mechanism_by_handle.get(handle, "unknown") for handle in item.handle_ids
    )
    local_priority = max(
        (
            local_retention_priority_by_handle.get(
                handle,
                (0,) * LOCAL_RETENTION_PRIORITY_WIDTH,
            )
            for handle in item.handle_ids
        ),
        default=(0,) * LOCAL_RETENTION_PRIORITY_WIDTH,
    )
    return (
        int(item.included),
        int(item.content_coherence is not ContentCoherence.CONFLICT),
        local_priority,
        len(item.supported_slot_ids),
        max((_mechanism_priority(row) for row in mechanisms), default=0),
        -ordinal_by_receipt[item.receipt_sha256],
        item.receipt_sha256,
    )


def _packet_subset(
    packet: TypedEvidencePacket,
    items: tuple[TypedEvidenceItem, ...],
) -> TypedEvidencePacket:
    identity_subset = tuple(
        item.receipt_sha256 for item in items
    ) == tuple(item.receipt_sha256 for item in packet.items)
    used_handles = {handle for item in items for handle in item.handle_ids}
    bindings = (
        packet.local_bindings
        if identity_subset
        else tuple(
            binding
            for binding in packet.local_bindings
            if binding.handle_id in used_handles
        )
    )
    parsed = ParsedTypedItems(
        items,
        packet.rejected_items,
        identity_sha256(
            {
                "accepted_item_receipt_sha256s": [
                    item.receipt_sha256 for item in items
                ],
                "format": f"{FORMAT}-budget-subset-v1",
                "original_packet_receipt_sha256": packet.receipt_sha256,
                "rejected_item_receipt_sha256s": [
                    item.rejection_sha256 for item in packet.rejected_items
                ],
            }
        ),
    )
    return build_typed_evidence_packet(
        packet.operator_spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(
            packet.sealed_input_artifact_sha256s
        ),
        frontier_mode=packet.frontier.mode,
        conflict_policy=packet.conflict_policy,
        # The complete wrapped chat, not the typed sub-payload, owns the real
        # 768-token output reserve below.  A one-token construction reserve
        # prevents the packet builder's first-fit salvage from silently
        # dropping later mechanisms before weakest-item fitting can see them.
        output_token_reserve=PACKET_CONSTRUCTION_OUTPUT_TOKEN_RESERVE,
        truncated=bool(packet.frontier.truncated or not identity_subset),
        provider_payload_mode=packet.provider_payload_mode,
    )


def _group_items(
    packet: TypedEvidencePacket,
) -> dict[str, tuple[TypedEvidenceItem, ...]]:
    group_by_handle = {
        row.handle_id: row.source_group_handle for row in packet.handles
    }
    mutable: dict[str, list[TypedEvidenceItem]] = {}
    for item in _usable_items(packet):
        for group in dict.fromkeys(group_by_handle[handle] for handle in item.handle_ids):
            mutable.setdefault(group, []).append(item)
    return {key: tuple(value) for key, value in mutable.items()}


def _item_story_labels(item: TypedEvidenceItem) -> frozenset[str]:
    labels = tuple(
        value
        for value in (item.entity_key, item.group_key)
        if type(value) is str and value
    )
    return frozenset(
        term
        for label in labels
        for term in normalized_terms(label)
    )


def _values_conflict(
    left: TypedEvidenceItem,
    right: TypedEvidenceItem,
) -> bool:
    if not set(left.supported_slot_ids) & set(right.supported_slot_ids):
        return False
    left_entity = left.entity_key or left.group_key
    right_entity = right.entity_key or right.group_key
    if not left_entity or not right_entity:
        return False
    if normalized_terms(left_entity) != normalized_terms(right_entity):
        # Distinct members/events may legitimately support the same generic
        # predicate slot (set join, count, and ordering questions).
        return False
    if (
        left.relation is not None
        and right.relation is not None
        and normalized_terms(left.relation) != normalized_terms(right.relation)
    ):
        return False
    if (
        left.numeric_role.value != "none"
        and right.numeric_role.value != "none"
        and left.numeric_role is not right.numeric_role
    ):
        return False
    if (
        left.status.value != "unknown"
        and right.status.value != "unknown"
        and left.status is not right.status
    ):
        return False
    if left.date is not None and right.date is not None and left.date != right.date:
        # Different event times identify different assertions.  A temporal
        # sequence is not itself a contradiction.
        return False
    if left.numeric_value is not None and right.numeric_value is not None:
        return (
            float(left.numeric_value) != float(right.numeric_value)
            or left.unit != right.unit
        )
    if left.participant_count is not None and right.participant_count is not None:
        return left.participant_count != right.participant_count
    return False


def _story_coherence_planes(
    packet: TypedEvidencePacket,
    *,
    local_story_keys_by_group: Mapping[str, Sequence[str]] | None = None,
    selected_evidence_discourse_links: (
        SelectedEvidenceDiscourseLinks | None
    ) = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Build opaque CAV/coherence links without question/source-prefix affinity."""

    grouped = _group_items(packet)
    groups = tuple(sorted(grouped))
    story_keys = {
        group: frozenset((local_story_keys_by_group or {}).get(group, ()))
        for group in groups
    }
    content_link_candidates: list[tuple[dict[str, str], str]] = []
    conflict_candidates: list[tuple[dict[str, str], str]] = []
    typed_link_candidates: list[
        tuple[dict[str, Any], Mapping[str, Any], str]
    ] = []
    filtered_typed_link_receipts: list[str] = []
    if selected_evidence_discourse_links is not None:
        _require(
            type(selected_evidence_discourse_links)
            is SelectedEvidenceDiscourseLinks,
            "selected discourse-link compilation changed type",
        )
        current_handles = {row.handle_id for row in packet.handles}
        binding_by_link = {
            str(row["link_id"]): row
            for row in selected_evidence_discourse_links.local_bindings
        }
        relation_priority = {
            "revises": 0,
            "contradicts": 1,
            "depends_on": 2,
            "causes": 3,
            "resolves": 4,
            "evaluates": 5,
            "sequence": 6,
        }
        for link in sorted(
            selected_evidence_discourse_links.links,
            key=lambda row: (
                relation_priority.get(row.relation, 5),
                row.link_id,
            ),
        ):
            # Relations are atomic: if fitting removes either endpoint, the
            # whole link disappears rather than being rewritten into a claim
            # the linker never emitted.
            if not set(link.handle_ids) <= current_handles:
                filtered_typed_link_receipts.append(link.receipt_sha256)
                continue
            binding = binding_by_link.get(link.link_id)
            _require(
                binding is not None
                and binding.get("provider_link_receipt_sha256")
                == link.receipt_sha256,
                "selected discourse link lost its exact local binding",
            )
            typed_link_candidates.append(
                (link.projection(), binding, link.receipt_sha256)
            )
    groups_by_exact_key: dict[str, list[str]] = {}
    for group in groups:
        for key in sorted(story_keys[group]):
            require_sha256(key, "local story co-membership receipt")
            groups_by_exact_key.setdefault(key, []).append(group)
    exact_candidates: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    exact_key_members = [
        (key, tuple(dict.fromkeys(members)))
        for key, members in groups_by_exact_key.items()
        if len(tuple(dict.fromkeys(members))) >= 2
    ]
    # The receipt key is deliberately opaque, so lexical hash order carries no
    # semantic value.  Spend the bounded CAV/link allowance on the overlay that
    # reconnects the most selected groups (and then the most represented
    # evidence items) before using the hash solely as a stable final tie-break.
    # This lets an evidence-derived history component outrank a narrower
    # same-source overlay without exposing either locator to the provider.
    exact_key_members.sort(
        key=lambda row: (
            -len(row[1]),
            -sum(len(grouped[group]) for group in row[1]),
            row[0],
        )
    )
    for key, members in exact_key_members:
        link_id = f"L{len(exact_candidates) + 1:03d}"
        overlay = {
            "group_handles": list(members),
            "link_id": link_id,
            "relation": "exact_local_candidate_comembership",
        }
        local_body = {
            "format": f"{FORMAT}-story-link-binding-v1",
            "group_handles": list(members),
            "link_id": link_id,
            "local_story_key_receipt_sha256": key,
        }
        candidate_receipt = identity_sha256(
            {
                "format": f"{FORMAT}-story-link-candidate-v1",
                "kind": "exact_local_overlay",
                "local_binding_receipt_sha256": identity_sha256(local_body),
                "provider_projection": overlay,
            }
        )
        exact_candidates.append((overlay, local_body, candidate_receipt))
    for left_index, left_group in enumerate(groups):
        left_items = grouped[left_group]
        left_labels = frozenset(
            term for item in left_items for term in _item_story_labels(item)
        )
        for right_group in groups[left_index + 1 :]:
            right_items = grouped[right_group]
            right_labels = frozenset(
                term for item in right_items for term in _item_story_labels(item)
            )
            exact_local = bool(story_keys[left_group] & story_keys[right_group])
            content_link = bool(left_labels & right_labels)
            pair = (left_group, right_group)
            if content_link and not exact_local:
                link = {
                    "basis": "content_entity_coherence",
                    "left_group": left_group,
                    "right_group": right_group,
                }
                content_link_candidates.append(
                    (
                        link,
                        identity_sha256(
                            {
                                "format": f"{FORMAT}-story-link-candidate-v1",
                                "kind": "content_coherence",
                                "provider_projection": link,
                            }
                        ),
                    )
                )
            if any(
                _values_conflict(left, right)
                for left in left_items
                for right in right_items
            ):
                conflict = {
                        "left_group": left_group,
                        "right_group": right_group,
                        "reason": "overlapping_slot_inconsistent_value",
                    }
                conflict_candidates.append(
                    (
                        conflict,
                        identity_sha256(
                            {
                                "format": f"{FORMAT}-story-link-candidate-v1",
                                "kind": "incompatible_group_pair",
                                "provider_projection": conflict,
                            }
                        ),
                    )
                )
    exact_overlays: list[dict[str, Any]] = []
    content_links: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    typed_links: list[dict[str, Any]] = []
    local_overlay_bindings: list[dict[str, Any]] = []
    admitted_receipts: list[str] = []
    dropped_receipts: list[str] = []
    dropped_conflict_receipts: list[str] = []

    def story_projection(
        incompatible: Sequence[Mapping[str, Any]],
        discourse: Sequence[Mapping[str, Any]],
        overlays: Sequence[Mapping[str, Any]],
        links: Sequence[Mapping[str, Any]],
        *,
        omitted_conflict_policy: str,
    ) -> tuple[dict[str, Any], int]:
        _require(
            omitted_conflict_policy in {"block", "clear"},
            "story omitted-conflict policy changed",
        )
        provider: dict[str, Any] = {
            "group_links": list(links),
            "incompatible_group_pairs": list(incompatible),
            "link_overlays": list(overlays),
            "link_token_cap": STORY_LINK_TOKEN_CAP,
            "link_token_proxy": 0,
            "omitted_conflict_policy": omitted_conflict_policy,
            "policy": (
                "prefer_smallest_linked_set; never_mix_incompatible_groups; "
                "omitted_conflict_policy_block_requires_parent"
            ),
        }
        # Preserve byte-identical legacy behavior when no selected-evidence
        # discourse links were supplied or survived whole-link filtering.
        if discourse:
            provider["typed_links"] = list(discourse)
        for _ in range(8):
            observed = count_tokens(_canonical_json(provider))
            if provider["link_token_proxy"] == observed:
                return provider, observed
            provider["link_token_proxy"] = observed
        observed = count_tokens(_canonical_json(provider))
        _require(
            provider["link_token_proxy"] == observed,
            "story link token proxy failed to converge",
        )
        return provider, observed

    def fits(
        incompatible: Sequence[Mapping[str, Any]],
        discourse: Sequence[Mapping[str, Any]],
        overlays: Sequence[Mapping[str, Any]],
        links: Sequence[Mapping[str, Any]],
    ) -> bool:
        _projection, tokens = story_projection(
            incompatible,
            discourse,
            overlays,
            links,
            # Use the safety-blocking form during allocation so any omitted
            # contradiction cannot make a previously fitted row overflow.
            omitted_conflict_policy="block",
        )
        return tokens <= STORY_LINK_TOKEN_CAP

    # Safety conflicts receive first claim, then exact typed discourse links,
    # exact prompt-external source links, and bounded content-only O(G^2)
    # links. Within the discourse lane, semantic relations precede sequence.
    for conflict, candidate_receipt in conflict_candidates:
        trial = tuple((*conflicts, conflict))
        if fits(trial, typed_links, exact_overlays, content_links):
            conflicts.append(conflict)
            admitted_receipts.append(candidate_receipt)
        else:
            dropped_receipts.append(candidate_receipt)
            dropped_conflict_receipts.append(candidate_receipt)
    for link, local_binding, candidate_receipt in typed_link_candidates:
        trial = tuple((*typed_links, link))
        if fits(conflicts, trial, exact_overlays, content_links):
            typed_links.append(link)
            admitted_receipts.append(candidate_receipt)
            local_overlay_bindings.append(dict(local_binding))
        else:
            dropped_receipts.append(candidate_receipt)
    for overlay, local_body, candidate_receipt in exact_candidates:
        trial = tuple((*exact_overlays, overlay))
        if fits(conflicts, typed_links, trial, content_links):
            exact_overlays.append(overlay)
            admitted_receipts.append(candidate_receipt)
            binding_body = {
                **local_body,
                "candidate_receipt_sha256": candidate_receipt,
            }
            local_overlay_bindings.append(
                {
                    **binding_body,
                    "receipt_sha256": identity_sha256(binding_body),
                }
            )
        else:
            dropped_receipts.append(candidate_receipt)
    for link, candidate_receipt in content_link_candidates:
        trial = tuple((*content_links, link))
        if fits(conflicts, typed_links, exact_overlays, trial):
            content_links.append(link)
            admitted_receipts.append(candidate_receipt)
        else:
            dropped_receipts.append(candidate_receipt)

    provider, link_tokens = story_projection(
        conflicts,
        typed_links,
        exact_overlays,
        content_links,
        omitted_conflict_policy=(
            "block" if dropped_conflict_receipts else "clear"
        ),
    )
    _require(
        link_tokens <= STORY_LINK_TOKEN_CAP,
        "story coherence escaped its independent link allowance",
    )
    budget_body = {
        "admitted_link_receipt_sha256s": admitted_receipts,
        "content_candidate_count": len(content_link_candidates),
        "conflict_candidate_count": len(conflict_candidates),
        "dropped_conflict_receipt_sha256s": dropped_conflict_receipts,
        "dropped_link_count": len(dropped_receipts),
        "dropped_link_receipt_sha256s": dropped_receipts,
        "exact_candidate_ordering": (
            "member_group_count_then_represented_item_count_then_receipt"
        ),
        "exact_local_candidate_count": len(exact_candidates),
        "format": f"{FORMAT}-story-link-budget-v1",
        "link_token_cap": STORY_LINK_TOKEN_CAP,
        "link_token_proxy": link_tokens,
    }
    if selected_evidence_discourse_links is not None:
        budget_body.update(
            {
                "selected_discourse_link_compilation_receipt_sha256": (
                    selected_evidence_discourse_links.receipt_sha256
                ),
                "typed_link_admitted_count": len(typed_links),
                "typed_link_candidate_count": len(typed_link_candidates),
                "typed_link_filtered_receipt_sha256s": (
                    filtered_typed_link_receipts
                ),
            }
        )
    local_overlay_bindings.append(
        {**budget_body, "receipt_sha256": identity_sha256(budget_body)}
    )
    return provider, tuple(local_overlay_bindings)


def story_coherence_projection(
    packet: TypedEvidencePacket,
    *,
    local_story_keys_by_group: Mapping[str, Sequence[str]] | None = None,
    selected_evidence_discourse_links: (
        SelectedEvidenceDiscourseLinks | None
    ) = None,
) -> dict[str, Any]:
    """Return the provider-visible opaque CAV/coherence overlay."""

    provider, _local = _story_coherence_planes(
        packet,
        local_story_keys_by_group=local_story_keys_by_group,
        selected_evidence_discourse_links=selected_evidence_discourse_links,
    )
    return provider


def _execution_advisory(
    execution: OperatorExecutionReceipt,
) -> dict[str, Any] | None:
    if execution.status is not ExecutionStatus.SUPPORTED:
        return None
    return {
        "advisory_only": True,
        "executor": execution.executor.value,
        "prediction": execution.prediction,
        "receipt_sha256": execution.receipt_sha256,
        "status": execution.status.value,
        "used_handle_ids": list(execution.used_handle_ids),
    }


def candidate_preservation_requirements(
    packet: TypedEvidencePacket,
    *,
    dated_question: str | None = None,
) -> dict[str, Any]:
    if dated_question is not None:
        require_text(dated_question, "candidate-preservation dated question")
        _require(
            quote_sha256(dated_question) == packet.operator_spec.question_sha256,
            "candidate-preservation question/spec binding changed",
        )
    usable = _usable_items(packet)
    by_handle: dict[str, dict[str, list[str]]] = {}
    for item in usable:
        specific = (
            item.specificity_terms
            if packet.operator_spec.specificity_required
            else ()
        )
        personal = (
            item.personalization_anchors
            if packet.operator_spec.personalization_required
            else ()
        )
        exact_identifiers: tuple[str, ...] = ()
        exact_titles: tuple[str, ...] = ()
        if (
            dated_question is not None
            and packet.operator_spec.answer_shape is AnswerShape.DIRECT
            and packet.operator_spec.operation == "single_supported_fact"
        ):
            exact_identifiers, exact_titles = _selected_exact_resource_anchors(
                dated_question,
                item.summary,
            )
        for handle in item.handle_ids:
            target = by_handle.setdefault(
                handle,
                {"personalization_terms": [], "specificity_terms": []},
            )
            for key, values in (
                ("personalization_terms", personal),
                ("specificity_terms", specific),
            ):
                target[key] = list(dict.fromkeys((*target[key], *values)))
            if exact_identifiers:
                target["exact_identifier_anchors"] = list(
                    dict.fromkeys(
                        (
                            *target.get("exact_identifier_anchors", []),
                            *exact_identifiers,
                        )
                    )
                )
            if exact_titles:
                target["exact_title_anchors"] = list(
                    dict.fromkeys(
                        (*target.get("exact_title_anchors", []), *exact_titles)
                    )
                )
    return {
        "by_handle": by_handle,
        # Reserved for obligations compiled from the question itself.  It is
        # deliberately empty in v1; evidence from an unused distractor can
        # never impose wording on a concise answer.
        "question_required_terms": [],
    }


def _item_answer_anchor_terms(
    packet: TypedEvidencePacket,
    item: TypedEvidenceItem,
) -> tuple[str, ...]:
    """Return conservative answer-bearing terms, never question-only terms."""

    question_terms = {
        term
        for slot in packet.operator_spec.required_slots
        for term in slot.match_terms
    }
    values: list[str] = []
    for value in (
        item.entity_key,
        item.group_key,
        item.date,
        item.relation,
    ):
        if value:
            values.extend(normalized_terms(value))
    values.extend(item.specificity_terms)
    values.extend(item.personalization_anchors)
    values.extend(
        term
        for term in normalized_terms(item.summary)
        if term not in question_terms
    )
    if item.numeric_value is not None:
        scalar = float(item.numeric_value)
        values.append(
            str(int(scalar)) if scalar.is_integer() else str(scalar).rstrip("0").rstrip(".")
        )
    if item.participant_count is not None:
        values.append(str(item.participant_count))
    return tuple(dict.fromkeys(value for value in values if value))


def _positive_scalar_validation_advisory(
    packet: TypedEvidencePacket,
    execution: OperatorExecutionReceipt,
) -> dict[str, Any] | None:
    """Validate positive scalar operations without claiming a closed frontier.

    A bounded frontier blocks absence/set completeness, but it does not make two
    explicitly cited comparison operands or an explicit duration endpoint
    unusable.  Re-execution is therefore restricted to scalar answer shapes and
    remains labelled as a local positive-value check, never as frontier closure.
    """

    if execution.status is ExecutionStatus.SUPPORTED:
        return None
    spec = packet.operator_spec
    explicit_numeric_slots = tuple(
        slot
        for slot in spec.required_slots
        if slot.kind in {SlotKind.OPERAND, SlotKind.COMPARISON_SIDE}
        and slot.requires_numeric
    )
    fixed_arity_scalar = spec.comparison_mode in {
        ComparisonMode.DIFFERENCE,
        ComparisonMode.BOOLEAN_GREATER,
    } or (
        spec.answer_shape is AnswerShape.DURATION
        and spec.temporal_mode is TemporalMode.INTERVAL
    ) or (
        spec.answer_shape is AnswerShape.NUMBER
        and spec.comparison_mode is ComparisonMode.NONE
        and bool(explicit_numeric_slots)
    )
    if (
        packet.frontier.closed
        or execution.reason != "frontier_not_closed"
        or not fixed_arity_scalar
    ):
        return None
    usable_slots = {
        slot_id for item in _usable_items(packet) for slot_id in item.supported_slot_ids
    }
    required_slots = {slot.slot_id for slot in spec.required_slots}
    if not required_slots <= usable_slots:
        return None
    parsed = ParsedTypedItems(
        packet.items,
        packet.rejected_items,
        identity_sha256(
            {
                "format": f"{FORMAT}-positive-scalar-validation-parse-v1",
                "packet_receipt_sha256": packet.receipt_sha256,
            }
        ),
    )
    validation_packet = build_typed_evidence_packet(
        spec,
        packet.local_bindings,
        parsed,
        sealed_input_artifact_sha256s=packet.sealed_input_artifact_sha256s,
        frontier_mode=FrontierMode.EXHAUSTIVE,
        conflict_policy=packet.conflict_policy,
        output_token_reserve=1,
        truncated=False,
    )
    checked = execute_typed_operator(spec, validation_packet)
    if checked.status is not ExecutionStatus.SUPPORTED:
        return None
    return {
        "basis": "bounded_positive_scalar_check_no_frontier_upgrade",
        "comparison_mode": spec.comparison_mode.value,
        "prediction": checked.prediction,
        "prediction_sha256": quote_sha256(checked.prediction),
        "receipt_sha256": checked.receipt_sha256,
        "used_handle_ids": list(checked.used_handle_ids),
    }


_EXPLICIT_TWO_MEMBER_RE = re.compile(
    r"\b(?:a|an|one)\s+[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,3}"
    r"\s+(?:and|along\s+with)\s+(?:a|an|one)\s+"
    r"[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,3}\b",
    re.IGNORECASE,
)
_LIST_CARDINALITY_RE = re.compile(
    r"\b(?:what|which)\s+are\s+(?:the\s+)?"
    r"(?P<count>\d+|" + "|".join(NUMBER_WORDS) + r")\b",
    re.IGNORECASE,
)


def _explicit_member_count(summary: str) -> int | None:
    """Return only a grammar-explicit two-member enumeration.

    Plural morphology is intentionally not cardinality.  In particular,
    ``return boots`` remains one obligation, while ``a peace lily and a
    succulent`` carries two explicit object determiners.
    """

    return 2 if _EXPLICIT_TWO_MEMBER_RE.search(summary) is not None else None


def _validation_cardinality(
    dated_question: str,
    answer_shape: AnswerShape,
    compiled_cardinality: int | None,
) -> int | None:
    if compiled_cardinality is not None or answer_shape not in {
        AnswerShape.SET_LIST,
        AnswerShape.ORDERED_LIST,
    }:
        return compiled_cardinality
    match = _LIST_CARDINALITY_RE.search(dated_question)
    if match is None:
        return None
    raw = match.group("count").casefold()
    return int(raw) if raw.isdigit() else NUMBER_WORDS[raw]


def _semantic_validation_row(item: TypedEvidenceItem) -> dict[str, Any]:
    summary_terms = normalized_terms(item.summary)
    semantic_body = {
        "action_concepts": list(canonical_action_concepts(item.summary)),
        "completed_action_concepts": list(
            completed_action_concepts(item.summary)
        ),
        "date": item.date,
        "entity_terms": list(normalized_terms(item.entity_key or "")),
        "explicit_member_count": _explicit_member_count(item.summary),
        "group_terms": list(normalized_terms(item.group_key or "")),
        "kind": item.kind.value,
        "numeric_role": item.numeric_role.value,
        "numeric_qualifier": item.numeric_qualifier.value,
        "numeric_value": item.numeric_value,
        "participant_count": item.participant_count,
        "relation_terms": list(normalized_terms(item.relation or "")),
        "status": item.status.value,
        "summary_terms": list(summary_terms),
        "supported_slot_ids": list(item.supported_slot_ids),
        "unit": item.unit,
    }
    return {
        **semantic_body,
        "item_receipt_sha256": item.receipt_sha256,
        # Cross-mechanism duplicates have the same typed meaning even though
        # their local item/binding receipts differ.
        "semantic_unit_sha256": identity_sha256(semantic_body),
    }


def completion_validation_contract(
    packet: TypedEvidencePacket,
    execution: OperatorExecutionReceipt,
    *,
    dated_question: str,
) -> dict[str, Any]:
    """Seal the exact prompt-external contract used to accept replacements."""

    if type(packet) is not TypedEvidencePacket:
        raise TypeError("packet must be an exact TypedEvidencePacket")
    if type(execution) is not OperatorExecutionReceipt:
        raise TypeError("execution must be an exact OperatorExecutionReceipt")
    require_text(dated_question, "typed final validation question")
    _require(
        quote_sha256(dated_question) == packet.operator_spec.question_sha256,
        "validation question/spec binding changed",
    )
    usable = _usable_items(packet)
    by_handle: dict[str, dict[str, Any]] = {}
    for item in usable:
        for handle in item.handle_ids:
            row = by_handle.setdefault(
                handle,
                {
                    "answer_anchor_terms": [],
                    "numeric_value_rows": [],
                    "semantic_rows": [],
                    "status_values": [],
                    "supported_slot_ids": [],
                    "usable_item_receipt_sha256s": [],
                },
            )
            row["answer_anchor_terms"] = list(
                dict.fromkeys(
                    (*row["answer_anchor_terms"], *_item_answer_anchor_terms(packet, item))
                )
            )
            row["status_values"] = list(
                dict.fromkeys((*row["status_values"], item.status.value))
            )
            row["supported_slot_ids"] = list(
                dict.fromkeys((*row["supported_slot_ids"], *item.supported_slot_ids))
            )
            row["usable_item_receipt_sha256s"].append(item.receipt_sha256)
            row["semantic_rows"].append(_semantic_validation_row(item))
            if item.numeric_value is not None:
                row["numeric_value_rows"].append(
                    {
                        "item_receipt_sha256": item.receipt_sha256,
                        "numeric_qualifier": item.numeric_qualifier.value,
                        "numeric_value": item.numeric_value,
                        "supported_slot_ids": list(item.supported_slot_ids),
                        "unit": item.unit,
                    }
                )
    contract = {
        "answer_shape": packet.operator_spec.answer_shape.value,
        "by_handle": by_handle,
        "cardinality": _validation_cardinality(
            dated_question,
            packet.operator_spec.answer_shape,
            packet.operator_spec.cardinality,
        ),
        "comparison_mode": packet.operator_spec.comparison_mode.value,
        "deterministic_execution_advisory": _execution_advisory(execution),
        "format": VALIDATION_CONTRACT_FORMAT,
        "include_proposed": packet.operator_spec.include_proposed,
        "operation": packet.operator_spec.operation,
        "operator_spec_receipt_sha256": packet.operator_spec.receipt_sha256,
        "packet_receipt_sha256": packet.receipt_sha256,
        "question_action_concepts": list(
            canonical_action_concepts(dated_question)
        ),
        "question_terms": list(normalized_terms(dated_question)),
        "required_slot_ids": [slot.slot_id for slot in packet.operator_spec.required_slots],
        "required_slots": [
            {
                "kind": slot.kind.value,
                "label_terms": list(normalized_terms(slot.label)),
                "match_terms": list(slot.match_terms),
                "relation_constraint": slot.relation_constraint,
                "requires_numeric": slot.requires_numeric,
                "slot_id": slot.slot_id,
            }
            for slot in packet.operator_spec.required_slots
        ],
        "requires_all_slots": packet.operator_spec.requires_all_slots,
        "scalar_validation_advisory": _positive_scalar_validation_advisory(
            packet, execution
        ),
        "temporal_mode": packet.operator_spec.temporal_mode.value,
    }
    assert_gold_blind(contract, path="typed_final_completion_validation_contract")
    return contract


def provider_input_projection(
    *,
    dated_question: str,
    parent_prediction: str,
    packet: TypedEvidencePacket,
    story_coherence: Mapping[str, Any],
    execution: OperatorExecutionReceipt,
) -> dict[str, Any]:
    require_text(dated_question, "typed final dated question")
    require_text(parent_prediction, "typed final parent prediction")
    value = {
        "dated_question": dated_question,
        "deterministic_execution_advisory": _execution_advisory(execution),
        "format": PROMPT_ROW_FORMAT,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent_prediction,
            "prediction_sha256": quote_sha256(parent_prediction),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": ["H001"],
        },
        "scalar_validation_advisory": _positive_scalar_validation_advisory(
            packet, execution
        ),
        "story_coherence": dict(story_coherence),
        "typed_evidence": compact_typed_evidence_projection(packet),
    }
    _reject_provider_locator_keys(value)
    assert_gold_blind(value, path="typed_final_provider_input")
    return value


def render_final_messages(
    provider_input: Mapping[str, Any],
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[dict[str, str], ...]:
    _reject_provider_locator_keys(provider_input)
    require_text(system_prompt, "typed final system prompt")
    messages = (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _canonical_json(provider_input)},
    )
    return _plain_messages(messages)


def _forbidden_literal_check(
    messages: Sequence[Mapping[str, str]],
    forbidden_provider_literals: Sequence[str],
) -> None:
    rendered = _canonical_json(_plain_messages(messages))
    for value in forbidden_provider_literals:
        _require(type(value) is str, "forbidden provider literal changed type")
        if len(value) >= 4:
            _require(value not in rendered, "raw locator value escaped into prompt")


def _range_prefix(value: str) -> int:
    _require(len(value) >= 2 and value[0] in {"H", "G"}, "opaque ID changed")
    try:
        return int(value[1:]) // 100_000
    except ValueError as exc:  # pragma: no cover - typed packet already guards it
        raise TypedMemoryFinalArmError("opaque ID changed") from exc


def validate_disjoint_contribution_ranges(
    packet: TypedEvidencePacket,
    mechanism_by_handle: Mapping[str, str],
) -> None:
    handle_ids = tuple(row.handle_id for row in packet.handles)
    _require(
        set(handle_ids) == set(mechanism_by_handle),
        "mechanism map must bind every retained opaque handle exactly",
    )
    handle_ranges: dict[str, set[int]] = {}
    group_ranges: dict[str, set[int]] = {}
    for handle in packet.handles:
        mechanism = require_text(
            mechanism_by_handle[handle.handle_id], "typed contribution mechanism"
        )
        handle_ranges.setdefault(mechanism, set()).add(_range_prefix(handle.handle_id))
        group_ranges.setdefault(mechanism, set()).add(
            _range_prefix(handle.source_group_handle)
        )
    _require(
        all(len(values) == 1 for values in handle_ranges.values())
        and all(len(values) == 1 for values in group_ranges.values()),
        "one contribution crossed its opaque H/G range",
    )
    _require(
        len({next(iter(values)) for values in handle_ranges.values()})
        == len(handle_ranges)
        and len({next(iter(values)) for values in group_ranges.values()})
        == len(group_ranges),
        "typed contributions do not have globally disjoint H/G ranges",
    )


@dataclass(frozen=True, slots=True)
class FittedTypedFinalPrompt:
    packet: TypedEvidencePacket
    execution: OperatorExecutionReceipt
    story_coherence: Mapping[str, Any]
    provider_input: Mapping[str, Any]
    messages: tuple[dict[str, str], ...]
    prompt_token_proxy: int
    allowed_handle_ids: tuple[str, ...]
    handle_group_by_id: Mapping[str, str]
    preservation_requirements: Mapping[str, Any]
    validation_contract: Mapping[str, Any]
    mechanism_by_handle: Mapping[str, str]
    story_link_local_bindings: tuple[Mapping[str, Any], ...]
    protection_source_receipt_sha256: str | None
    protected_item_receipt_sha256s: tuple[str, ...]
    protected_binding_receipt_sha256s: tuple[str, ...]
    dropped_item_receipt_sha256s: tuple[str, ...]
    dropped_binding_receipt_sha256s: tuple[str, ...]
    local_retention_priority_receipt_sha256: str
    receipt_sha256: str

    def projection(self, *, include_local: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "allowed_handle_ids": list(self.allowed_handle_ids),
            "dropped_binding_receipt_sha256s": list(
                self.dropped_binding_receipt_sha256s
            ),
            "dropped_item_receipt_sha256s": list(
                self.dropped_item_receipt_sha256s
            ),
            "execution_receipt_sha256": self.execution.receipt_sha256,
            "format": PROMPT_ROW_FORMAT,
            "full_chat_plus_output_tokens": (
                self.prompt_token_proxy + OUTPUT_TOKEN_RESERVE
            ),
            "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
            "messages_sha256": identity_sha256(list(self.messages)),
            "output_token_reserve": OUTPUT_TOKEN_RESERVE,
            "packet_receipt_sha256": self.packet.receipt_sha256,
            "protection_source_receipt_sha256": (
                self.protection_source_receipt_sha256
            ),
            "protected_binding_receipt_sha256s": list(
                self.protected_binding_receipt_sha256s
            ),
            "protected_item_receipt_sha256s": list(
                self.protected_item_receipt_sha256s
            ),
            "local_retention_priority_receipt_sha256": (
                self.local_retention_priority_receipt_sha256
            ),
            "preservation_requirements": dict(self.preservation_requirements),
            "validation_contract": dict(self.validation_contract),
            "prompt_token_proxy": self.prompt_token_proxy,
            "provider_input": dict(self.provider_input),
            "receipt_sha256": self.receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "story_coherence": dict(self.story_coherence),
        }
        if include_local:
            value.update(
                {
                    "handle_group_by_id": dict(self.handle_group_by_id),
                    "local_bindings": [
                        row.projection() for row in self.packet.local_bindings
                    ],
                    "mechanism_by_handle": dict(self.mechanism_by_handle),
                    "story_link_local_bindings": [
                        dict(row) for row in self.story_link_local_bindings
                    ],
                }
            )
        return value


def fit_typed_final_prompt(
    *,
    dated_question: str,
    parent_prediction: str,
    packet: TypedEvidencePacket,
    mechanism_by_handle: Mapping[str, str],
    local_story_keys_by_group: Mapping[str, Sequence[str]] | None = None,
    selected_evidence_discourse_links: (
        SelectedEvidenceDiscourseLinks | None
    ) = None,
    local_retention_priority_by_handle: Mapping[str, Sequence[int]] | None = None,
    forbidden_provider_literals: Sequence[str] = (),
    minimum_usable_items_per_mechanism: int = 0,
    protected_item_receipt_sha256s: Sequence[str] = (),
    protection_source_receipt_sha256: str | None = None,
) -> FittedTypedFinalPrompt:
    """Trim the weakest items until the exact wrapped chat fits 8k.

    The protected parent and dated question are never shortened.  Removed
    bindings remain auditable through their receipt IDs returned by this
    function; retained bindings are never rewritten.
    """

    if type(packet) is not TypedEvidencePacket:
        raise TypeError("packet must be an exact TypedEvidencePacket")
    _require(
        type(minimum_usable_items_per_mechanism) is int
        and minimum_usable_items_per_mechanism >= 0,
        "per-mechanism protected minimum changed",
    )
    mechanism = dict(mechanism_by_handle)
    validate_disjoint_contribution_ranges(packet, mechanism)
    if selected_evidence_discourse_links is not None:
        _require(
            type(selected_evidence_discourse_links)
            is SelectedEvidenceDiscourseLinks
            and {
                handle
                for link in selected_evidence_discourse_links.links
                for handle in link.handle_ids
            }
            <= {row.handle_id for row in packet.handles},
            "selected discourse links escaped the fitted packet population",
        )
    local_priorities = {
        handle: tuple(values)
        for handle, values in (
            local_retention_priority_by_handle or {}
        ).items()
    }
    _require(
        set(local_priorities) <= set(mechanism)
        and all(
            type(handle) is str
            and len(priority) == LOCAL_RETENTION_PRIORITY_WIDTH
            and all(type(value) is int for value in priority)
            for handle, priority in local_priorities.items()
        ),
        "local retention priorities must bind packet handles at fixed width",
    )
    retention_priority_body = {
        "fixed_width": LOCAL_RETENTION_PRIORITY_WIDTH,
        "format": f"{FORMAT}-local-retention-priority-v1",
        "rows": [
            {"handle_id": handle, "priority": list(priority)}
            for handle, priority in sorted(local_priorities.items())
        ],
    }
    retention_priority_receipt_sha256 = identity_sha256(
        retention_priority_body
    )
    original_bindings = {
        row.handle_id: row for row in packet.local_bindings
    }
    original_items = tuple(packet.items)
    protected_receipts = tuple(protected_item_receipt_sha256s)
    _require(
        len(set(protected_receipts)) == len(protected_receipts)
        and all(type(value) is str for value in protected_receipts),
        "protected item receipts must be ordered unique text",
    )
    for value in protected_receipts:
        require_sha256(value, "protected fitted item receipt")
    _require(
        (not protected_receipts and protection_source_receipt_sha256 is None)
        or (
            bool(protected_receipts)
            and type(protection_source_receipt_sha256) is str
        ),
        "protected fitted items require exactly one protection source receipt",
    )
    if protection_source_receipt_sha256 is not None:
        require_sha256(
            protection_source_receipt_sha256,
            "fitted protection source receipt",
        )
    usable_original_receipts = {
        item.receipt_sha256 for item in _usable_items(packet)
    }
    _require(
        set(protected_receipts) <= usable_original_receipts,
        "protected fitted item is missing or unusable",
    )
    protected_receipt_set = set(protected_receipts)
    protected_handles = {
        handle
        for item in original_items
        if item.receipt_sha256 in protected_receipt_set
        for handle in item.handle_ids
    }
    protected_binding_receipts = tuple(
        binding.receipt_sha256
        for binding in packet.local_bindings
        if binding.handle_id in protected_handles
    )
    ordinals = {
        row.receipt_sha256: index for index, row in enumerate(original_items)
    }
    kept = list(original_items)
    dropped_items: list[str] = []

    while True:
        current = _packet_subset(packet, tuple(kept))
        current_mechanism = {
            handle.handle_id: mechanism[handle.handle_id]
            for handle in current.handles
        }
        validate_disjoint_contribution_ranges(current, current_mechanism)
        current_groups = {row.source_group_handle for row in current.handles}
        local_keys = {
            group: tuple((local_story_keys_by_group or {}).get(group, ()))
            for group in current_groups
        }
        story, story_link_local_bindings = _story_coherence_planes(
            current,
            local_story_keys_by_group=local_keys,
            selected_evidence_discourse_links=(
                selected_evidence_discourse_links
            ),
        )
        downstream_overlay = compile_downstream_operator_overlay(
            dated_question,
            current.operator_spec,
        )
        execution = execute_downstream_typed_operator(
            current.operator_spec,
            current,
            downstream_overlay,
        )
        provider_input = provider_input_projection(
            dated_question=dated_question,
            parent_prediction=parent_prediction,
            packet=current,
            story_coherence=story,
            execution=execution,
        )
        messages = render_final_messages(provider_input)
        _forbidden_literal_check(messages, forbidden_provider_literals)
        prompt_tokens = count_chat_prompt_token_proxy(messages)
        if prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP:
            break
        _require(
            bool(kept),
            "protected question/parent/protocol exceed the hard 8k envelope",
        )
        usable_now = set(_usable_items(current))
        count_by_mechanism: dict[str, int] = {}
        mechanisms_by_item: dict[str, tuple[str, ...]] = {}
        for item in kept:
            item_mechanisms = tuple(
                dict.fromkeys(mechanism[handle] for handle in item.handle_ids)
            )
            mechanisms_by_item[item.receipt_sha256] = item_mechanisms
            if item in usable_now:
                for mechanism_id in item_mechanisms:
                    count_by_mechanism[mechanism_id] = (
                        count_by_mechanism.get(mechanism_id, 0) + 1
                    )
        removable = tuple(
            item
            for item in kept
            if item.receipt_sha256 not in protected_receipt_set
            and (
                item not in usable_now
                or all(
                    count_by_mechanism.get(mechanism_id, 0)
                    > minimum_usable_items_per_mechanism
                    for mechanism_id in mechanisms_by_item[item.receipt_sha256]
                )
            )
        )
        _require(
            bool(removable),
            "per-mechanism protected minima cannot fit the hard 8k envelope",
        )
        weakest = min(
            removable,
            key=lambda item: _weakest_item_key(
                item,
                mechanism_by_handle=mechanism,
                ordinal_by_receipt=ordinals,
                local_retention_priority_by_handle=local_priorities,
            ),
        )
        kept.remove(weakest)
        dropped_items.append(weakest.receipt_sha256)

    retained_handles = {row.handle_id for row in current.handles}
    _require(
        protected_receipt_set
        <= {item.receipt_sha256 for item in current.items},
        "hard prompt fit dropped a protected lane minimum",
    )
    dropped_bindings = tuple(
        binding.receipt_sha256
        for handle, binding in original_bindings.items()
        if handle not in retained_handles
    )
    allowed = tuple(
        dict.fromkeys(
            handle
            for item in _usable_items(current)
            for handle in item.handle_ids
        )
    )
    handle_group = {
        row.handle_id: row.source_group_handle
        for row in current.handles
        if row.handle_id in set(allowed)
    }
    requirements = candidate_preservation_requirements(
        current,
        dated_question=dated_question,
    )
    validation_contract = completion_validation_contract(
        current,
        execution,
        dated_question=dated_question,
    )
    receipt = identity_sha256(
        {
            "allowed_handle_ids": list(allowed),
            "dropped_binding_receipt_sha256s": list(dropped_bindings),
            "dropped_item_receipt_sha256s": dropped_items,
            "execution_receipt_sha256": execution.receipt_sha256,
            "format": f"{PROMPT_ROW_FORMAT}-fit",
            "messages_sha256": identity_sha256(list(messages)),
            "packet_receipt_sha256": current.receipt_sha256,
            "protection_source_receipt_sha256": (
                protection_source_receipt_sha256
            ),
            "protected_binding_receipt_sha256s": list(
                protected_binding_receipts
            ),
            "protected_item_receipt_sha256s": list(protected_receipts),
            "prompt_token_proxy": prompt_tokens,
            "minimum_usable_items_per_mechanism": (
                minimum_usable_items_per_mechanism
            ),
            "local_retention_priority_receipt_sha256": (
                retention_priority_receipt_sha256
            ),
            "validation_contract_sha256": identity_sha256(validation_contract),
        }
    )
    return FittedTypedFinalPrompt(
        current,
        execution,
        story,
        provider_input,
        messages,
        prompt_tokens,
        allowed,
        handle_group,
        requirements,
        validation_contract,
        current_mechanism,
        story_link_local_bindings,
        protection_source_receipt_sha256,
        protected_receipts,
        protected_binding_receipts,
        tuple(dropped_items),
        dropped_bindings,
        retention_priority_receipt_sha256,
        receipt,
    )


@dataclass(frozen=True, slots=True)
class ParsedTypedFinalDecision:
    valid: bool
    decision: Literal["keep_parent", "replace", "invalid"]
    prediction: str
    used_handle_ids: tuple[str, ...]
    validation_basis: str
    error_code: str
    receipt_sha256: str


def _invalid_decision(code: str) -> ParsedTypedFinalDecision:
    require_text(code, "typed final parse error")
    return ParsedTypedFinalDecision(
        False,
        "invalid",
        "",
        (),
        "invalid",
        code,
        identity_sha256(
            {
                "error_code": code,
                "format": f"{DECISION_FORMAT}-invalid",
                "validator_policy_format": VALIDATOR_POLICY_FORMAT,
            }
        ),
    )


def _incompatible_pairs(story: Mapping[str, Any]) -> frozenset[frozenset[str]]:
    raw = story.get("incompatible_group_pairs", [])
    _require(type(raw) is list, "story conflict pairs changed type")
    pairs: set[frozenset[str]] = set()
    for row in raw:
        _require(
            type(row) is dict
            and type(row.get("left_group")) is str
            and type(row.get("right_group")) is str,
            "story conflict pair changed schema",
        )
        pairs.add(frozenset((row["left_group"], row["right_group"])))
    return frozenset(pairs)


_ANSWER_GLUE_TERMS = frozenset(
    {
        "answer",
        "approximately",
        "both",
        "item",
        "member",
        "night",
        "piece",
        "total",
    }
)
_LIST_SEPARATOR_RE = re.compile(r"\s*(?:,|;|\n|→|\band\b)\s*", re.IGNORECASE)
_APPROXIMATE_OUTPUT_RE = re.compile(
    r"\b(?:about|around|approximately|approx\.?|roughly|nearly|almost)\b",
    re.I,
)
_LOWER_BOUND_OUTPUT_RE = re.compile(
    r"\b(?:over|above|more\s+than|at\s+least|minimum)\b", re.I
)
_UPPER_BOUND_OUTPUT_RE = re.compile(
    r"\b(?:under|below|less\s+than|at\s+most|up\s+to|maximum)\b", re.I
)
_EXACT_URL_RE = re.compile(
    r"https?://[^\s<>\[\]{}\"']+",
    re.I,
)
_DIRECT_RESOURCE_QUERY_RE = re.compile(
    r"\b(?:article|book|document|episode|guide|link|manual|paper|playlist|"
    r"recipe|report|resource|site|song|title|track|tutorial|url|video|"
    r"website)\b",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(
    r"\[(?P<title>[^\]\r\n]{2,240})\]\(\s*"
    r"(?P<url>https?://[^\s<>\[\]{}\"']+?)\s*\)",
    re.IGNORECASE,
)
_QUOTED_TITLE_RE = re.compile(
    r'(?:"(?P<straight>[^"\r\n]{2,240})"|'
    r"“(?P<curly>[^”\r\n]{2,240})”)",
)
_BOLD_TITLE_RE = re.compile(r"\*\*(?P<title>[^*\r\n]{2,240})\*\*")


def _exact_urls(value: str) -> tuple[str, ...]:
    """Return byte-exact URL anchors without surrounding prose punctuation."""

    return tuple(
        dict.fromkeys(
            match.group(0).rstrip(".,;:!?)]}")
            for match in _EXACT_URL_RE.finditer(value)
            if match.group(0).rstrip(".,;:!?)]}")
        )
    )


def _explicit_resource_title(prefix: str, line: str, url: str) -> str | None:
    """Extract only explicitly delimited resource titles next to one URL."""

    for match in _MARKDOWN_LINK_RE.finditer(line):
        if match.group("url").rstrip(".,;:!?)]}") == url:
            title = match.group("title").strip()
            if len(normalized_terms(title)) >= 2:
                return title
    quoted = tuple(_QUOTED_TITLE_RE.finditer(prefix))
    if quoted:
        match = quoted[-1]
        title = (match.group("straight") or match.group("curly")).strip()
        if len(normalized_terms(title)) >= 2:
            return title
    bold = tuple(_BOLD_TITLE_RE.finditer(prefix))
    if bold:
        title = bold[-1].group("title").strip()
        if len(normalized_terms(title)) >= 2:
            return title
    return None


def _resource_anchor_candidates(summary: str) -> tuple[dict[str, Any], ...]:
    candidates: dict[str, dict[str, Any]] = {}
    for match in _EXACT_URL_RE.finditer(summary):
        url = match.group(0).rstrip(".,;:!?)]}")
        if not url:
            continue
        line_start = summary.rfind("\n", 0, match.start()) + 1
        line_end = summary.find("\n", match.end())
        if line_end < 0:
            line_end = len(summary)
        line = summary[line_start:line_end]
        prefix = summary[line_start:match.start()]
        title = _explicit_resource_title(prefix, line, url)
        candidate = {
            "exact_identifier": url,
            "exact_title": title,
            "terms": tuple(normalized_terms(line)),
        }
        prior = candidates.get(url)
        if prior is None or (prior["exact_title"] is None and title is not None):
            candidates[url] = candidate
    return tuple(candidates.values())


def _selected_exact_resource_anchors(
    dated_question: str,
    summary: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Bind a direct resource answer to one evidence-derived linked entry.

    A selected exact chunk may contain a list of resources.  The sole entry is
    unambiguous; for a list, require a unique question-overlap winner.  Ties
    remain unbound rather than forcing every distractor identifier into a
    concise answer.
    """

    if _DIRECT_RESOURCE_QUERY_RE.search(dated_question) is None:
        return (), ()
    candidates = _resource_anchor_candidates(summary)
    if not candidates:
        return (), ()
    if len(candidates) == 1:
        selected = candidates[0]
    else:
        question_terms = set(normalized_terms(dated_question))
        scores = tuple(
            len(question_terms & set(candidate["terms"]))
            for candidate in candidates
        )
        best = max(scores)
        if best < 1 or scores.count(best) != 1:
            return (), ()
        selected = candidates[scores.index(best)]
    title = selected["exact_title"]
    return (
        (selected["exact_identifier"],),
        (title,) if type(title) is str and title else (),
    )


def _protected_parent_urls(value: str) -> tuple[str, ...]:
    """Return exact URL anchors, excluding surrounding prose punctuation."""

    return _exact_urls(value)


def _safe_numeric_qualifier_error(
    prediction: str,
    qualifiers: Sequence[str],
) -> str | None:
    values = {NumericQualifier(value) for value in qualifiers}
    values.discard(NumericQualifier.EXACT)
    if not values:
        return None
    lower = NumericQualifier.LOWER_BOUND in values
    upper = NumericQualifier.UPPER_BOUND in values
    if lower and upper:
        return "typed_numeric_qualifier_unsafe"
    if (
        NumericQualifier.APPROXIMATE in values
        and _APPROXIMATE_OUTPUT_RE.search(prediction) is None
    ):
        return "typed_numeric_approximation_erased"
    if lower and _LOWER_BOUND_OUTPUT_RE.search(prediction) is None:
        return "typed_numeric_lower_bound_erased"
    if upper and _UPPER_BOUND_OUTPUT_RE.search(prediction) is None:
        return "typed_numeric_upper_bound_erased"
    return None


def _comparison_qualifiers(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    values: list[str] = []
    for row in left_rows:
        values.append(row["numeric_qualifier"])
    for row in right_rows:
        qualifier = NumericQualifier(row["numeric_qualifier"])
        flipped = {
            NumericQualifier.LOWER_BOUND: NumericQualifier.UPPER_BOUND,
            NumericQualifier.UPPER_BOUND: NumericQualifier.LOWER_BOUND,
        }.get(qualifier, qualifier)
        values.append(flipped.value)
    return tuple(values)


def _semantic_terms(row: Mapping[str, Any]) -> frozenset[str]:
    values: list[str] = []
    for key in (
        "summary_terms",
        "entity_terms",
        "group_terms",
        "relation_terms",
    ):
        terms = row.get(key)
        _require(
            type(terms) is list
            and all(type(term) is str and bool(term) for term in terms),
            "semantic validation terms changed",
        )
        values.extend(terms)
    return frozenset(values)


def _validated_semantic_rows(
    by_handle_contract: Mapping[str, Any],
    used: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    unique: dict[str, dict[str, Any]] = {}
    allowed_statuses = {status.value for status in EvidenceStatus}
    allowed_roles = {role.value for role in NumericRole}
    allowed_qualifiers = {qualifier.value for qualifier in NumericQualifier}
    for handle in used:
        handle_contract = by_handle_contract[handle]
        semantic_rows = handle_contract.get("semantic_rows")
        _require(
            type(semantic_rows) is list and bool(semantic_rows),
            "completion semantic rows changed type",
        )
        for raw in semantic_rows:
            _require(type(raw) is dict, "completion semantic row changed type")
            row = dict(raw)
            receipt = row.get("item_receipt_sha256")
            semantic_unit = row.get("semantic_unit_sha256")
            numeric = row.get("numeric_value")
            participant_count = row.get("participant_count")
            explicit_count = row.get("explicit_member_count")
            unit = row.get("unit")
            date = row.get("date")
            slots = row.get("supported_slot_ids")
            actions = row.get("action_concepts")
            completed_actions = row.get("completed_action_concepts")
            _require(
                type(receipt) is str
                and bool(receipt)
                and type(semantic_unit) is str
                and bool(semantic_unit)
                and row.get("status") in allowed_statuses
                and row.get("numeric_role") in allowed_roles
                and row.get("numeric_qualifier") in allowed_qualifiers
                and (numeric is None or type(numeric) in {int, float})
                and (
                    participant_count is None
                    or type(participant_count) is int
                    and participant_count >= 0
                )
                and (
                    explicit_count is None
                    or type(explicit_count) is int
                    and explicit_count >= 2
                )
                and (unit is None or type(unit) is str and bool(unit))
                and (date is None or type(date) is str and bool(date))
                and type(slots) is list
                and all(type(slot) is str and bool(slot) for slot in slots)
                and type(actions) is list
                and all(type(action) is str and bool(action) for action in actions)
                and type(completed_actions) is list
                and all(
                    type(action) is str and bool(action)
                    for action in completed_actions
                ),
                "completion semantic row changed schema",
            )
            _semantic_terms(row)
            prior = unique.setdefault(receipt, row)
            _require(prior == row, "completion semantic item changed across handles")
    return tuple(unique.values())


def _rows_relevant_to_question(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    question_actions = set(contract["question_action_concepts"])
    action_matched = tuple(
        row
        for row in rows
        if question_actions & set(row["action_concepts"])
    )
    if action_matched:
        return action_matched
    required = set(contract["required_slot_ids"])
    slot_matched = tuple(
        row for row in rows if required & set(row["supported_slot_ids"])
    )
    return slot_matched or tuple(rows)


def _rows_in_complete_proof_scope(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Conservatively identify rows an aggregate/order proof must reconcile.

    Required-slot support is the strongest available v3 obligation signal. If
    slots exist, an action-only distractor cannot enlarge the proof universe.
    With no compiled slots, canonical question actions provide the next safest
    scope. Falling back to every row is fail-closed for completeness claims.
    """

    required = set(contract["required_slot_ids"])
    if required:
        slot_matched = tuple(
            row
            for row in rows
            if required & set(row["supported_slot_ids"])
        )
        if slot_matched:
            question_actions = set(contract["question_action_concepts"])
            if question_actions:
                action_and_slot = tuple(
                    row
                    for row in slot_matched
                    if question_actions & set(row["action_concepts"])
                )
                if action_and_slot:
                    return action_and_slot
            return slot_matched
    question_actions = set(contract["question_action_concepts"])
    if question_actions:
        action_matched = tuple(
            row
            for row in rows
            if question_actions & set(row["action_concepts"])
        )
        if action_matched:
            return action_matched
    return tuple(rows)


def _complete_proof_scope_error(
    allowed_rows: Sequence[Mapping[str, Any]],
    used_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> str | None:
    if contract["operation"] not in {
        "count_or_aggregate",
        "deduplicated_member_join",
        "order_or_select",
    }:
        return None
    required_units = {
        row["semantic_unit_sha256"]
        for row in _rows_in_complete_proof_scope(allowed_rows, contract)
    }
    used_units = {row["semantic_unit_sha256"] for row in used_rows}
    if not required_units <= used_units:
        return "aggregate_scope_incomplete"
    return None


def _semantic_row_is_user_grounded(row: Mapping[str, Any]) -> bool:
    """Require a typed user-memory citation, not a generic recommendation."""

    return "user" in _semantic_terms(row)


def _validation_datetime(value: str) -> datetime | None:
    normalized = value.replace("/", "-")
    for format_string in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%B %Y",
        "%b %Y",
    ):
        try:
            return datetime.strptime(normalized, format_string)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _coherent_slot_scalar(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, str | None] | None:
    baseline = {
        (float(row["numeric_value"]), row["unit"])
        for row in rows
        if row["numeric_value"] is not None
        and row["numeric_role"] == NumericRole.BASELINE.value
    }
    end = {
        (float(row["numeric_value"]), row["unit"])
        for row in rows
        if row["numeric_value"] is not None
        and row["numeric_role"] == NumericRole.END.value
    }
    deltas = {
        (float(row["numeric_value"]), row["unit"])
        for row in rows
        if row["numeric_value"] is not None
        and row["numeric_role"] == NumericRole.DELTA.value
    }
    operands = {
        (float(row["numeric_value"]), row["unit"])
        for row in rows
        if row["numeric_value"] is not None
        and row["numeric_role"]
        in {NumericRole.OPERAND.value, NumericRole.NONE.value}
    }
    if baseline or end:
        if len(baseline) == len(end) == 1 and not deltas and not operands:
            baseline_value, baseline_unit = next(iter(baseline))
            end_value, end_unit = next(iter(end))
            if baseline_unit == end_unit:
                return end_value - baseline_value, end_unit
        return None
    if len(deltas) == 1 and not operands:
        return next(iter(deltas))
    if len(operands) == 1 and not deltas:
        return next(iter(operands))
    return None


def _comparison_entailment_error(
    prediction: str,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> str | None:
    side_slots = tuple(
        slot
        for slot in contract["required_slots"]
        if slot["kind"] == SlotKind.COMPARISON_SIDE.value
    )
    if len(side_slots) != 2:
        return "typed_comparison_slots"
    sides: list[tuple[float, str | None]] = []
    item_sets: list[set[str]] = []
    for slot in side_slots:
        selected = tuple(
            row
            for row in rows
            if slot["slot_id"] in row["supported_slot_ids"]
        )
        if not selected:
            return "typed_comparison_operand"
        scalar = _coherent_slot_scalar(selected)
        if scalar is None:
            return "typed_comparison_operand"
        sides.append(scalar)
        item_sets.append({row["item_receipt_sha256"] for row in selected})
    if item_sets[0] & item_sets[1] or sides[0][1] != sides[1][1]:
        return "typed_comparison_operand"
    delta = sides[0][0] - sides[1][0]
    mode = contract["comparison_mode"]
    left_rows = tuple(
        row
        for row in rows
        if side_slots[0]["slot_id"] in row["supported_slot_ids"]
    )
    right_rows = tuple(
        row
        for row in rows
        if side_slots[1]["slot_id"] in row["supported_slot_ids"]
    )
    qualifiers = _comparison_qualifiers(
        left_rows if delta >= 0 else right_rows,
        right_rows if delta >= 0 else left_rows,
    )
    if mode == ComparisonMode.BOOLEAN_GREATER.value:
        if any(
            value != NumericQualifier.EXACT.value for value in qualifiers
        ):
            return "typed_qualified_boolean_comparison_unsafe"
        candidate = prediction.casefold().strip().rstrip(".!?")
        expected = "yes" if delta > 0 else "no"
        return None if candidate == expected else "typed_boolean_entailment"
    candidate = conservative_numeric_value(prediction)
    if candidate is None or abs(candidate - abs(delta)) > 1e-9:
        return "typed_difference_entailment"
    return _safe_numeric_qualifier_error(prediction, qualifiers)


def _numeric_entailment_error(
    prediction: str,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> str | None:
    candidate = conservative_numeric_value(prediction)
    if candidate is None:
        return "typed_numeric_prediction"
    relevant = _rows_relevant_to_question(rows, contract)
    by_semantic_unit: dict[str, Mapping[str, Any]] = {}
    for row in relevant:
        by_semantic_unit.setdefault(row["semantic_unit_sha256"], row)
    units = tuple(by_semantic_unit.values())
    possible: set[float] = set()
    numeric = tuple(
        float(row["numeric_value"])
        for row in units
        if row["numeric_value"] is not None
    )
    possible.update(numeric)
    if numeric:
        possible.add(sum(numeric))
    count_contributions: list[float] = []
    for row in units:
        if row["numeric_value"] is not None:
            count_contributions.append(float(row["numeric_value"]))
        elif row["participant_count"] is not None:
            count_contributions.append(float(row["participant_count"]))
        elif row["explicit_member_count"] is not None:
            count_contributions.append(float(row["explicit_member_count"]))
        else:
            count_contributions.append(1.0)
    if count_contributions:
        possible.add(float(len(units)))
        possible.add(sum(count_contributions))
    if not any(abs(candidate - value) <= 1e-9 for value in possible):
        return "typed_numeric_entailment"
    return _safe_numeric_qualifier_error(
        prediction,
        tuple(
            row["numeric_qualifier"]
            for row in units
            if row["numeric_value"] is not None
        ),
    )


def _list_entailment_error(
    prediction: str,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> str | None:
    members = tuple(
        segment.strip(" .:-")
        for segment in _LIST_SEPARATOR_RE.split(prediction)
        if segment.strip(" .:-")
    )
    cardinality = contract["cardinality"]
    if cardinality is not None and len(members) != cardinality:
        return "typed_list_cardinality"
    if not members:
        return "typed_list_entailment"
    if (
        cardinality is not None
        and contract["operation"] == "deduplicated_member_join"
    ):
        identities = {
            tuple(row["entity_terms"])
            if row["entity_terms"]
            else (row["semantic_unit_sha256"],)
            for row in rows
        }
        if len(identities) != cardinality:
            return "typed_set_frontier_cardinality"
    question_terms = set(contract["question_terms"])
    required_actions = set(contract["question_action_concepts"])
    member_candidates: list[tuple[Mapping[str, Any], ...]] = []
    for member in members:
        terms = set(normalized_terms(member)) - _ANSWER_GLUE_TERMS
        specific = terms - question_terms or terms
        if not specific:
            return "typed_list_entailment"
        candidates: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if required_actions and not (
                required_actions & set(row["action_concepts"])
            ):
                continue
            if specific & set(_semantic_terms(row)):
                candidates.setdefault(row["semantic_unit_sha256"], row)
        if not candidates:
            return "typed_list_entailment"
        member_candidates.append(tuple(candidates.values()))
    if (
        contract["answer_shape"] == AnswerShape.ORDERED_LIST.value
        or contract["temporal_mode"] == TemporalMode.ORDER.value
    ):
        previous: datetime | None = None
        used_units: set[str] = set()
        for candidates in member_candidates:
            dated = sorted(
                (
                    (moment, row["semantic_unit_sha256"])
                    for row in candidates
                    if row["date"] is not None
                    and (moment := _validation_datetime(row["date"])) is not None
                    and row["semantic_unit_sha256"] not in used_units
                    and (previous is None or moment > previous)
                ),
                key=lambda pair: (pair[0], pair[1]),
            )
            if not dated:
                return "typed_order_entailment"
            previous, selected_unit = dated[0]
            used_units.add(selected_unit)
    return None


def _text_entailment_error(
    prediction: str,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> str | None:
    present = set(normalized_terms(prediction)) - _ANSWER_GLUE_TERMS
    if not present:
        return "typed_text_entailment"
    question_terms = set(contract["question_terms"])
    answer_terms = present - question_terms or present
    relevant = _rows_relevant_to_question(rows, contract)
    evidence_terms = set().union(*(_semantic_terms(row) for row in relevant))
    evidence_actions = {
        action for row in relevant for action in row["action_concepts"]
    }
    candidate_actions = set(canonical_action_concepts(prediction))
    if not answer_terms & evidence_terms and not candidate_actions & evidence_actions:
        return "typed_text_entailment"
    return None


def _typed_entailment_error(
    prediction: str,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> str | None:
    comparison_mode = contract["comparison_mode"]
    if comparison_mode in {
        ComparisonMode.DIFFERENCE.value,
        ComparisonMode.BOOLEAN_GREATER.value,
    }:
        return _comparison_entailment_error(prediction, rows, contract)
    answer_shape = contract["answer_shape"]
    if answer_shape in {AnswerShape.NUMBER.value, AnswerShape.DURATION.value}:
        return _numeric_entailment_error(prediction, rows, contract)
    if answer_shape in {AnswerShape.SET_LIST.value, AnswerShape.ORDERED_LIST.value}:
        return _list_entailment_error(prediction, rows, contract)
    return _text_entailment_error(prediction, rows, contract)


def parse_typed_final_completion(
    completion: str,
    *,
    parent_prediction: str,
    allowed_handle_ids: Sequence[str],
    handle_group_by_id: Mapping[str, str],
    story_coherence: Mapping[str, Any],
    preservation_requirements: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
) -> ParsedTypedFinalDecision:
    """Validate strict JSON; every invalid path deterministically keeps parent."""

    if type(completion) is not str:
        raise TypeError("completion must be exact text")
    require_text(parent_prediction, "typed final parent prediction")
    allowed = _ordered_unique(allowed_handle_ids, "allowed final handles")
    _require(
        set(handle_group_by_id) == set(allowed)
        and all(type(value) is str and value for value in handle_group_by_id.values()),
        "final handle/group binding changed",
    )
    contract = dict(validation_contract)
    by_handle_contract = contract.get("by_handle")
    required_slot_ids = contract.get("required_slot_ids")
    required_slots = contract.get("required_slots")
    requires_all_slots = contract.get("requires_all_slots")
    include_proposed = contract.get("include_proposed")
    question_terms = contract.get("question_terms")
    question_actions = contract.get("question_action_concepts")
    cardinality = contract.get("cardinality")
    _require(
        contract.get("format") == VALIDATION_CONTRACT_FORMAT
        and contract.get("answer_shape") in {shape.value for shape in AnswerShape}
        and contract.get("comparison_mode")
        in {mode.value for mode in ComparisonMode}
        and type(contract.get("operation")) is str
        and bool(contract.get("operation"))
        and type(by_handle_contract) is dict
        and set(by_handle_contract) == set(allowed)
        and type(required_slot_ids) is list
        and len(set(required_slot_ids)) == len(required_slot_ids)
        and all(type(row) is str and bool(row) for row in required_slot_ids)
        and type(required_slots) is list
        and [row.get("slot_id") for row in required_slots] == required_slot_ids
        and all(
            type(row) is dict
            and row.get("kind") in {kind.value for kind in SlotKind}
            and type(row.get("label_terms")) is list
            and type(row.get("match_terms")) is list
            and type(row.get("requires_numeric")) is bool
            for row in required_slots
        )
        and type(question_terms) is list
        and all(type(row) is str and bool(row) for row in question_terms)
        and type(question_actions) is list
        and all(type(row) is str and bool(row) for row in question_actions)
        and (cardinality is None or type(cardinality) is int and cardinality >= 1)
        and type(requires_all_slots) is bool
        and type(include_proposed) is bool
        and contract.get("temporal_mode") in {mode.value for mode in TemporalMode},
        "completion validation contract changed",
    )
    try:
        raw = json.loads(
            completion,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, ValueError):
        return _invalid_decision("invalid_json")
    if type(raw) is not dict or set(raw) != {
        "decision",
        "prediction",
        "used_handle_ids",
    }:
        return _invalid_decision("root_schema")
    decision = raw["decision"]
    prediction = raw["prediction"]
    used_raw = raw["used_handle_ids"]
    if (
        type(decision) is not str
        or type(prediction) is not str
        or type(used_raw) is not list
        or any(type(value) is not str for value in used_raw)
        or len(set(used_raw)) != len(used_raw)
    ):
        return _invalid_decision("value_schema")
    used = tuple(used_raw)
    if not set(used) <= set(allowed):
        return _invalid_decision("unknown_handle")
    if decision == "replace" and prediction == parent_prediction:
        receipt = identity_sha256(
            {
                "declared_decision": "replace",
                "decision": "keep_parent",
                "format": DECISION_FORMAT,
                "prediction_sha256": quote_sha256(parent_prediction),
                "used_handle_ids": [],
                "validation_basis": "normalized_identical_replace",
                "validator_policy_format": VALIDATOR_POLICY_FORMAT,
            }
        )
        return ParsedTypedFinalDecision(
            True,
            "keep_parent",
            parent_prediction,
            (),
            "normalized_identical_replace",
            "none",
            receipt,
        )
    validation_basis = "keep_parent_contract"
    if decision == "keep_parent":
        if prediction != parent_prediction or used:
            return _invalid_decision("keep_parent_contract")
    elif decision == "replace":
        if (
            not prediction
            or prediction.strip() != prediction
            or prediction == parent_prediction
            or not used
        ):
            return _invalid_decision("replace_contract")
        used_slots: set[str] = set()
        for handle in used:
            handle_contract = by_handle_contract.get(handle)
            if type(handle_contract) is not dict:
                raise TypedMemoryFinalArmError(
                    "completion handle contract changed type"
                )
            item_receipts = handle_contract.get("usable_item_receipt_sha256s")
            statuses = handle_contract.get("status_values")
            slots = handle_contract.get("supported_slot_ids")
            anchors = handle_contract.get("answer_anchor_terms")
            numeric_rows = handle_contract.get("numeric_value_rows")
            semantic_rows = handle_contract.get("semantic_rows")
            _require(
                type(item_receipts) is list
                and bool(item_receipts)
                and all(type(row) is str and bool(row) for row in item_receipts)
                and type(statuses) is list
                and bool(statuses)
                and all(type(row) is str and bool(row) for row in statuses)
                and EvidenceStatus.CANCELLED.value not in statuses
                and (
                    include_proposed
                    or EvidenceStatus.PROPOSED.value not in statuses
                )
                and type(slots) is list
                and all(type(row) is str and bool(row) for row in slots)
                and type(anchors) is list
                and all(type(row) is str and bool(row) for row in anchors)
                and type(numeric_rows) is list
                and type(semantic_rows) is list
                and bool(semantic_rows),
                "completion handle usability contract changed",
            )
            used_slots.update(slots)
        if requires_all_slots and not set(required_slot_ids) <= used_slots:
            return _invalid_decision("required_slot_coverage")
        if story_coherence.get("omitted_conflict_policy") == "block":
            return _invalid_decision("story_conflict_overflow")
        used_groups = tuple(dict.fromkeys(handle_group_by_id[row] for row in used))
        conflicts = _incompatible_pairs(story_coherence)
        for left_index, left in enumerate(used_groups):
            for right in used_groups[left_index + 1 :]:
                if frozenset((left, right)) in conflicts:
                    return _invalid_decision("incompatible_story_groups")
        deterministic = contract.get("deterministic_execution_advisory")
        scalar = contract.get("scalar_validation_advisory")
        if deterministic is not None:
            _require(
                type(deterministic) is dict
                and deterministic.get("status") == ExecutionStatus.SUPPORTED.value
                and type(deterministic.get("prediction")) is str
                and type(deterministic.get("used_handle_ids")) is list,
                "deterministic completion advisory changed",
            )
            if (
                prediction != deterministic["prediction"]
                or set(used) != set(deterministic["used_handle_ids"])
            ):
                return _invalid_decision("deterministic_advisory_disagreement")
            validation_basis = "deterministic_execution_agreement"
        elif scalar is not None:
            scalar_used = scalar.get("used_handle_ids") if type(scalar) is dict else None
            _require(
                type(scalar) is dict
                and scalar.get("basis")
                == "bounded_positive_scalar_check_no_frontier_upgrade"
                and type(scalar.get("prediction")) is str
                and bool(scalar.get("prediction"))
                and type(scalar_used) is list
                and all(type(row) is str and bool(row) for row in scalar_used)
                and len(set(scalar_used)) == len(scalar_used)
                and set(scalar_used) <= set(allowed),
                "scalar completion advisory changed",
            )
            if prediction == scalar["prediction"] and set(used) == set(scalar_used):
                validation_basis = "bounded_positive_scalar_agreement"
            else:
                validation_basis = "model_attested"
        else:
            validation_basis = "model_attested"
        semantic_rows = _validated_semantic_rows(by_handle_contract, used)
        present = set(normalized_terms(prediction))
        by_handle = preservation_requirements.get("by_handle", {})
        question_required = preservation_requirements.get(
            "question_required_terms", ()
        )
        if type(by_handle) is not dict or not isinstance(
            question_required, (list, tuple)
        ):
            raise TypedMemoryFinalArmError(
                "candidate preservation requirements changed type"
            )
        required: list[str] = list(question_required)
        exact_identifier_anchors: list[str] = []
        exact_title_anchors: list[str] = []
        has_personalization_citation = False
        for handle in used:
            row = by_handle.get(handle, {})
            if type(row) is not dict:
                raise TypedMemoryFinalArmError(
                    "candidate preservation handle requirements changed type"
                )
            for label in ("specificity_terms", "personalization_terms"):
                values = row.get(label, ())
                if not isinstance(values, (list, tuple)):
                    raise TypedMemoryFinalArmError(
                        "candidate preservation terms changed type"
                    )
                if label == "personalization_terms" and values:
                    has_personalization_citation = True
                required.extend(values)
            for label, target in (
                ("exact_identifier_anchors", exact_identifier_anchors),
                ("exact_title_anchors", exact_title_anchors),
            ):
                values = row.get(label, ())
                if not isinstance(values, (list, tuple)):
                    raise TypedMemoryFinalArmError(
                        "candidate preservation exact anchors changed type"
                    )
                if any(type(value) is not str or not value for value in values):
                    raise TypedMemoryFinalArmError(
                        "candidate preservation exact anchors changed schema"
                    )
                target.extend(values)
        if any(type(value) is not str or not value for value in required):
            raise TypedMemoryFinalArmError(
                "candidate preservation terms changed type"
            )
        if not set(required) <= present:
            return _invalid_decision("candidate_preservation")
        if (
            contract["answer_shape"] == AnswerShape.DIRECT.value
            and contract["operation"] == "single_supported_fact"
        ):
            if not set(exact_identifier_anchors) <= set(_exact_urls(prediction)):
                return _invalid_decision("evidence_identifier_anchor_loss")
            folded_prediction = prediction.casefold()
            if any(
                title.casefold() not in folded_prediction
                for title in exact_title_anchors
            ):
                return _invalid_decision("evidence_title_anchor_loss")
        if (
            contract["operation"] == "single_supported_fact"
            and not set(_protected_parent_urls(parent_prediction))
            <= set(_protected_parent_urls(prediction))
        ):
            return _invalid_decision("parent_anchor_loss")
        if (
            contract["operation"] == "preference_or_causal_synthesis"
            and not has_personalization_citation
            and not any(_semantic_row_is_user_grounded(row) for row in semantic_rows)
        ):
            return _invalid_decision("personalization_citation_missing")
        if validation_basis == "model_attested":
            entailment_error = _typed_entailment_error(
                prediction,
                semantic_rows,
                contract,
            )
            if entailment_error is not None:
                return _invalid_decision(entailment_error)
            if contract["operation"] in {
                "count_or_aggregate",
                "deduplicated_member_join",
                "order_or_select",
            }:
                scope_error = _complete_proof_scope_error(
                    _validated_semantic_rows(by_handle_contract, allowed),
                    semantic_rows,
                    contract,
                )
                if scope_error is not None:
                    return _invalid_decision(scope_error)
    else:
        return _invalid_decision("decision")
    receipt = identity_sha256(
        {
            "decision": decision,
            "format": DECISION_FORMAT,
            "prediction_sha256": quote_sha256(prediction),
            "used_handle_ids": list(used),
            "validation_basis": validation_basis,
            "validator_policy_format": VALIDATOR_POLICY_FORMAT,
        }
    )
    return ParsedTypedFinalDecision(
        True,
        decision,
        prediction,
        used,
        validation_basis,
        "none",
        receipt,
    )


def materialize_typed_final_result_row(
    plan_row: Mapping[str, Any],
    completion: str,
    *,
    completion_receipt_sha256: str,
    call_key_sha256: str,
    request_journal_sha256: str,
    response_journal_sha256: str,
) -> dict[str, Any]:
    """Materialize one full-population answer row from a sealed prompt row."""

    for value, label in (
        (completion_receipt_sha256, "completion receipt"),
        (call_key_sha256, "completion call key"),
        (request_journal_sha256, "completion request journal"),
        (response_journal_sha256, "completion response journal"),
    ):
        require_sha256(value, label)
    parent = require_text(plan_row.get("parent_prediction"), "final parent prediction")
    parsed = parse_typed_final_completion(
        completion,
        parent_prediction=parent,
        allowed_handle_ids=tuple(plan_row.get("allowed_handle_ids", ())),
        handle_group_by_id=dict(plan_row.get("handle_group_by_id", {})),
        story_coherence=dict(plan_row.get("story_coherence", {})),
        preservation_requirements=dict(
            plan_row.get("preservation_requirements", {})
        ),
        validation_contract=dict(plan_row.get("validation_contract", {})),
    )
    valid_replace = parsed.valid and parsed.decision == "replace"
    prediction = parsed.prediction if valid_replace else parent
    if valid_replace:
        source = (
            "typed_final_deterministic_validated_replacement_v1"
            if parsed.validation_basis == "deterministic_execution_agreement"
            else "typed_final_scalar_validated_replacement_v1"
            if parsed.validation_basis == "bounded_positive_scalar_agreement"
            else "typed_final_model_attested_replacement_v1"
        )
        decision = "replace"
        used = parsed.used_handle_ids
    elif parsed.valid:
        source = "typed_final_validated_keep_parent_v1"
        decision = "keep_parent"
        used = ()
    else:
        source = "typed_final_invalid_keep_parent_v1"
        decision = "invalid_keep_parent"
        used = ()
    body = {
        "call_key_sha256": call_key_sha256,
        "changed_from_parent": prediction != parent,
        "completion_receipt_sha256": completion_receipt_sha256,
        "dated_question_sha256": require_sha256(
            plan_row.get("dated_question_sha256"), "final dated question"
        ),
        "decision": decision,
        "format": RESULT_ROW_FORMAT,
        "ordinal": plan_row.get("ordinal"),
        "parent_prediction_sha256": quote_sha256(parent),
        "parse_error_code": parsed.error_code,
        "parse_receipt_sha256": parsed.receipt_sha256,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": source,
        "prompt_row_receipt_sha256": require_sha256(
            plan_row.get("prompt_row_receipt_sha256"), "final prompt row"
        ),
        "question_id": require_text(plan_row.get("question_id"), "final question ID"),
        "question_sha256": require_sha256(
            plan_row.get("question_sha256"), "final question"
        ),
        "request_journal_sha256": request_journal_sha256,
        "response_journal_sha256": response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "route_id": require_text(plan_row.get("route_id"), "final route"),
        "solver_valid": parsed.valid,
        "used_handle_ids": list(used),
        "validation_basis": parsed.validation_basis,
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    _require(
        type(body["ordinal"]) is int and int(body["ordinal"]) >= 0,
        "final result ordinal changed",
    )
    body["source_row_sha256"] = identity_sha256(body)
    assert_gold_blind(body, path="typed_final_result_row")
    return body


def judge_row_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Gold-free, stable 100-row seam consumed by the common Sol judge."""

    value = {
        "changed_from_parent": row.get("changed_from_parent"),
        "dated_question_sha256": row.get("dated_question_sha256"),
        "format": JUDGE_ROW_FORMAT,
        "ordinal": row.get("ordinal"),
        "parent_prediction_sha256": row.get("parent_prediction_sha256"),
        "prediction": row.get("prediction"),
        "prediction_sha256": row.get("prediction_sha256"),
        "prediction_source": row.get("prediction_source"),
        "question_id": row.get("question_id"),
        "question_sha256": row.get("question_sha256"),
        "route_id": row.get("route_id"),
        "source_row_sha256": row.get("source_row_sha256"),
    }
    require_text(value["prediction"], "judge prediction")
    require_text(value["prediction_source"], "judge prediction source")
    require_text(value["question_id"], "judge question ID")
    require_text(value["route_id"], "judge route")
    for key in (
        "dated_question_sha256",
        "parent_prediction_sha256",
        "prediction_sha256",
        "question_sha256",
        "source_row_sha256",
    ):
        require_sha256(value[key], f"judge {key}")
    _require(
        type(value["ordinal"]) is int
        and type(value["changed_from_parent"]) is bool,
        "judge row scalar changed",
    )
    assert_gold_blind(value, path="typed_final_judge_row")
    return value


__all__ = [
    "COMPOSITION_FORMAT",
    "DECISION_FORMAT",
    "EXPECTED_QUESTION_COUNT",
    "FORMAT",
    "FittedTypedFinalPrompt",
    "HARD_PROMPT_TOKEN_CAP",
    "JUDGE_ROW_FORMAT",
    "LOCAL_RETENTION_PRIORITY_WIDTH",
    "MAX_CHAT_PROMPT_TOKENS",
    "OUTPUT_TOKEN_RESERVE",
    "PACKET_CONSTRUCTION_OUTPUT_TOKEN_RESERVE",
    "PROMPT_ROW_FORMAT",
    "ParsedTypedFinalDecision",
    "RESOURCE_PRESERVING_SYSTEM_PROMPT_V2",
    "RESULT_ROW_FORMAT",
    "STORY_LINK_TOKEN_CAP",
    "SYSTEM_PROMPT",
    "TypedMemoryFinalArmError",
    "VALIDATION_CONTRACT_FORMAT",
    "VALIDATOR_POLICY_FORMAT",
    "candidate_preservation_requirements",
    "compact_evidence_content_projection",
    "compact_typed_evidence_projection",
    "completion_validation_contract",
    "fit_typed_final_prompt",
    "LEGACY_SYSTEM_PROMPT_V1",
    "judge_row_projection",
    "materialize_typed_final_result_row",
    "parse_typed_final_completion",
    "provider_input_projection",
    "render_final_messages",
    "story_coherence_projection",
    "validate_disjoint_contribution_ranges",
]
