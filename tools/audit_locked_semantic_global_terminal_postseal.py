#!/usr/bin/env python3
"""Post-seal, provider-free source and exact-fact visibility audit for exact11.

The target-owner plan is deliberately unavailable until the terminal
construction/replay pair has passed its normal strict reader.  Source IDs are
useful for diagnosing retrieval coverage, but they are not semantic fact
proof: one source can expose the wrong span.  Promotion therefore requires a
separately sealed manifest of 26 answer-bearing semantic atoms and their exact
source/turn/content/date equivalence locators.  The two relation-link messages
remain in the raw31 diagnostic ledger but do not independently authorize an
atom.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import build_exact11_semantic_atom_manifest as atom_cli  # noqa: E402
from tools import run_reduced_semantic_global_terminal_assay as terminal_cli  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.semantic_global_terminal_adapter import (  # noqa: E402
    HARD_PROMPT_TOKEN_CAP,
    OUTPUT_TOKEN_RESERVE,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    PACKET_CONSTRUCTION_OUTPUT_TOKEN_RESERVE,
    render_final_messages,
)
from tools.matched_eval.typed_operator_adapter import (  # noqa: E402
    COMPACT_FINAL_PROVIDER_FORMAT,
)


FORMAT = "memory-condense-semantic-global-terminal-postseal-audit-v2"
WITNESS_MANIFEST_FORMAT = "memory-condense-exact11-target-witness-manifest-v1"
TARGET_PLAN_FORMAT = "memory-condense-retrieval-target-owner-plan-v1"
SOURCE_TARGET_COUNT = 26
POSITIVE_WITNESS_COUNT = 31
DIRECT_ANSWER_WITNESS_COUNT = 29
RELATION_LINK_WITNESS_COUNT = 2
NEGATIVE_WITNESS_COUNT = 1
SEMANTIC_ATOM_COUNT = atom_cli.EXPECTED_ATOM_COUNT
DEFAULT_TARGET_PLAN = (
    Path(__file__).resolve().parents[1]
    / "docs/10 - Research Log/data/longmemeval-locked-100-target-owner-plan-v1.json"
)
DEFAULT_TARGET_PLAN_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)
DEFAULT_TARGET_PLAN_IDENTITY_SHA256 = (
    "2cabfbb103929c68dea47368502875444903ced282c708cba45ef26bee14d888"
)
DEFAULT_WITNESS_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docs/10 - Research Log/data/longmemeval-exact11-target-witness-manifest-v1.json"
)
DEFAULT_WITNESS_MANIFEST_SHA256 = (
    "f6add6368971d9b0b827bc0042c5e2a2e409f26df4f2a30ef18224c34c64bd60"
)
DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256 = (
    "3b39b8fba2ee0bc67cb6413883973c6da3b9ee4afbe6517aed28ed0b217ee935"
)
DEFAULT_SEMANTIC_ATOM_MANIFEST = atom_cli.DEFAULT_OUTPUT
DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256 = (
    "c40bbfc78f07eccbd6b2e489b79f4ad1ba5221dea2aeb707c64ecf84ac514008"
)
DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256 = (
    "f3e8ad4975d953eac16a98003626d7fb3ebc39b4a335e6fcea703e40f487c69c"
)
DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256 = (
    "e2a13b57f44f4b863df22b7d7e906bb6cd74e15c9b895add37bface21907c73c"
)
LOCKED_DATASET_FILE_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
SOURCE_AUDIT_NAME = "semantic-global-terminal-postseal-source-audit-v1.json"
FACT_AUDIT_NAME = "semantic-global-terminal-postseal-fact-audit-v2.json"


class SemanticGlobalTerminalPostSealAuditError(MatchedEvalContractError):
    """A sealed input, target population, provenance chain, or gate changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise SemanticGlobalTerminalPostSealAuditError(message)


def _dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value), f"{label} must be nonempty text")
    return value  # type: ignore[return-value]


def _integer(value: object, label: str) -> int:
    _require(type(value) is int, f"{label} must be an exact integer")
    return value  # type: ignore[return-value]


def _boolean(value: object, label: str) -> bool:
    _require(type(value) is bool, f"{label} must be an exact boolean")
    return value  # type: ignore[return-value]


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except (TypeError, ValueError, MatchedEvalContractError) as exc:
        raise SemanticGlobalTerminalPostSealAuditError(
            f"{label} must be an exact SHA-256"
        ) from exc


def _self_sha(row: Mapping[str, Any], key: str, label: str) -> str:
    observed = _sha(row.get(key), label)
    expected = identity_sha256(
        {name: value for name, value in row.items() if name != key}
    )
    _require(observed == expected, f"{label} self-authentication changed")
    return observed


def _ordered_unique_text(value: object, label: str) -> tuple[str, ...]:
    rows = _list(value, label)
    _require(
        all(type(row) is str and bool(row) for row in rows)
        and len(rows) == len(set(rows)),
        f"{label} must be ordered unique text",
    )
    return tuple(rows)


def _compact_item(item: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "content_coherence": item.get("content_coherence"),
        "handle_ids": item.get("handle_ids"),
        "included": item.get("included"),
        "kind": item.get("kind"),
        "status": item.get("status"),
        "summary": item.get("summary"),
        "supported_slot_ids": item.get("supported_slot_ids"),
        "value_authority": item.get("value_authority"),
    }
    for key in (
        "date",
        "entity_key",
        "group_key",
        "numeric_role",
        "numeric_qualifier",
        "numeric_value",
        "participant_count",
        "personalization_anchors",
        "relation",
        "specificity_terms",
        "unit",
    ):
        child = item.get(key)
        if key == "numeric_role" and child == "none":
            child = None
        if key == "numeric_qualifier" and item.get("numeric_value") is None:
            child = None
        if key in {"personalization_anchors", "specificity_terms"} and child == []:
            child = None
        if child is not None:
            value[key] = child
    return value


def _compact_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    operator = _dict(packet.get("operator_spec"), "terminal packet operator")
    frontier = _dict(packet.get("frontier"), "terminal packet frontier")
    bindings = tuple(
        _dict(row, "terminal packet local binding")
        for row in _list(packet.get("local_bindings"), "terminal packet bindings")
    )
    items = tuple(
        _dict(row, "terminal packet item")
        for row in _list(packet.get("items"), "terminal packet items")
    )
    represented = {
        handle
        for item in items
        for handle in _ordered_unique_text(
            item.get("handle_ids"), "terminal packet item handles"
        )
    }
    compact_handles = [
        {
            "group_handle": row.get("source_group_handle"),
            "handle_id": row.get("handle_id"),
            "origin": row.get("origin"),
            "provenance_grade": row.get("provenance_grade"),
        }
        for row in bindings
        if row.get("handle_id") in represented
    ]
    required_slots = [
        {
            "kind": row.get("kind"),
            "label": row.get("label"),
            "match_terms": row.get("match_terms"),
            "minimum_match_term_count": row.get("minimum_match_term_count"),
            "relation_constraint": row.get("relation_constraint"),
            "requires_numeric": row.get("requires_numeric"),
            "slot_id": row.get("slot_id"),
        }
        for row in (
            _dict(raw, "terminal packet required slot")
            for raw in _list(
                operator.get("required_slots"),
                "terminal packet required slots",
            )
        )
    ]
    compact_operator = {
        "absence_decision_requires_closed_frontier": operator.get(
            "absence_decision_requires_closed_frontier"
        ),
        "answer_shape": operator.get("answer_shape"),
        "cardinality": operator.get("cardinality"),
        "comparison_mode": operator.get("comparison_mode"),
        "include_proposed": operator.get("include_proposed"),
        "operation": operator.get("operation"),
        "ordering": operator.get("ordering"),
        "personalization_required": operator.get("personalization_required"),
        "query_timestamp": operator.get("query_timestamp"),
        "required_evidence_role": operator.get("required_evidence_role"),
        "required_slots": required_slots,
        "requires_all_slots": operator.get("requires_all_slots"),
        "requires_complete_frontier": operator.get("requires_complete_frontier"),
        "specificity_required": operator.get("specificity_required"),
        "style": operator.get("style"),
        "temporal_mode": operator.get("temporal_mode"),
        "temporal_window_days": operator.get("temporal_window_days"),
    }
    compact_frontier = {
        "available_handle_ids": frontier.get("available_handle_ids"),
        "closed": frontier.get("closed"),
        "mode": frontier.get("mode"),
        "omitted_handle_ids": frontier.get("omitted_handle_ids"),
        "rejected_item_count": len(
            _list(
                frontier.get("rejected_item_receipt_sha256s"),
                "terminal rejected item receipts",
            )
        ),
        "represented_handle_ids": frontier.get("represented_handle_ids"),
        "truncated": frontier.get("truncated"),
        "unresolved_slot_ids": frontier.get("unresolved_slot_ids"),
    }
    return {
        "conflict_policy": packet.get("conflict_policy"),
        "format": COMPACT_FINAL_PROVIDER_FORMAT,
        "frontier": compact_frontier,
        "handles": compact_handles,
        "items": [_compact_item(row) for row in items],
        "operator_spec": compact_operator,
    }


def _verified_target_plan(
    path: Path,
    expected_file_sha256: str,
    expected_identity_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == _sha(expected_file_sha256, "target-plan artifact"),
        "immutable target-plan artifact binding changed",
    )
    plan = artifact.payload
    identity = _self_sha(plan, "plan_sha256", "target-plan identity")
    _require(
        identity == _sha(expected_identity_sha256, "target-plan identity argument")
        and plan.get("format") == TARGET_PLAN_FORMAT
        and plan.get("provider_calls") == 0
        and plan.get("runtime_use_forbidden") is True
        and plan.get("gold_target_tags_posthoc_only") is True
        and plan.get("answer_run_or_judge_inputs_loaded") is False
        and plan.get("question_count") == 100,
        "immutable target-plan contract changed",
    )
    ordered = tuple(
        _dict(row, "target-plan ordered question")
        for row in _list(plan.get("ordered_question_keys"), "target-plan questions")
    )
    _require(
        len(ordered) == 100
        and tuple(row.get("ordinal") for row in ordered) == tuple(range(100))
        and len({row.get("question_id") for row in ordered}) == 100
        and all(type(row.get("question_id")) is str and row.get("question_id") for row in ordered),
        "target-plan ordered population changed",
    )
    targets = tuple(
        _dict(row, "target-plan target")
        for row in _list(plan.get("desired_targets"), "target-plan targets")
    )
    _require(
        plan.get("desired_target_count") == len(targets),
        "target-plan target count changed",
    )
    seen_targets: set[str] = set()
    for row in targets:
        body = {
            key: row.get(key)
            for key in (
                "ordinal",
                "question_id",
                "target_kind",
                "target_id",
                "primary_owner",
            )
        }
        target_sha = _sha(row.get("target_sha256"), "target-plan target")
        basis = _dict(row.get("assignment_basis"), "target assignment basis")
        _require(
            target_sha == identity_sha256(body)
            and row.get("assignment_basis_sha256") == identity_sha256(basis)
            and target_sha not in seen_targets
            and type(row.get("ordinal")) is int
            and 0 <= row["ordinal"] < 100
            and row.get("question_id") == ordered[row["ordinal"]].get("question_id")
            and type(row.get("target_kind")) is str
            and type(row.get("target_id")) is str
            and bool(row.get("target_id")),
            "target-plan target authentication changed",
        )
        seen_targets.add(target_sha)
    exact = tuple(
        row
        for row in targets
        if row.get("ordinal") in terminal_cli.EXACT_ORDINALS
        and row.get("target_kind") == "source_id"
    )
    _require(
        len(exact) == SOURCE_TARGET_COUNT
        and {row["ordinal"] for row in exact} == set(terminal_cli.EXACT_ORDINALS),
        "exact11 source-target population changed from 26",
    )
    return artifact, exact


def _verified_witness_manifest(
    path: Path,
    expected_file_sha256: str,
    *,
    target_plan: SealedArtifact,
    source_targets: Sequence[Mapping[str, Any]],
) -> tuple[SealedArtifact, dict[str, Any]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == _sha(expected_file_sha256, "witness manifest artifact"),
        "witness manifest artifact binding changed",
    )
    manifest = artifact.payload
    _self_sha(manifest, "manifest_identity_sha256", "witness manifest identity")
    positive = tuple(
        _dict(row, "positive target witness")
        for row in _list(
            manifest.get("positive_witnesses"), "positive target witnesses"
        )
    )
    negative = tuple(
        _dict(row, "negative target witness")
        for row in _list(
            manifest.get("negative_witnesses"), "negative target witnesses"
        )
    )
    target_keys = {
        (int(row["ordinal"]), str(row["question_id"]), str(row["target_id"]))
        for row in source_targets
    }
    _require(
        len(positive) == POSITIVE_WITNESS_COUNT
        and len(negative) == NEGATIVE_WITNESS_COUNT,
        "witness manifest positive/negative population changed",
    )
    positive_by_target: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    negative_by_target: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    seen_positive_receipts: set[str] = set()
    seen_negative_receipts: set[str] = set()
    direct_sources: set[tuple[int, str, str]] = set()
    link_sources: set[tuple[int, str, str]] = set()
    for row in positive:
        _require(
            set(row)
            == {
                "content_char_count",
                "content_sha256",
                "format",
                "has_answer",
                "ordinal",
                "question_id",
                "role",
                "session_turn_index",
                "target_source_id",
                "witness_kind",
                "witness_receipt_sha256",
            },
            "positive witness schema changed",
        )
        key = (
            _integer(row.get("ordinal"), "witness ordinal"),
            _text(row.get("question_id"), "witness question ID"),
            _text(row.get("target_source_id"), "witness target source ID"),
        )
        kind = _text(row.get("witness_kind"), "positive witness kind")
        receipt = _self_sha(
            row, "witness_receipt_sha256", "positive witness receipt"
        )
        _sha(row.get("content_sha256"), "positive witness content")
        _require(
            key in target_keys
            and row.get("format") == f"{WITNESS_MANIFEST_FORMAT}-witness-v1"
            and kind in {"answer_atom", "relation_link"}
            and _integer(row.get("session_turn_index"), "witness turn index") >= 0
            and _text(row.get("role"), "witness role") in {"user", "assistant"}
            and _integer(row.get("content_char_count"), "witness character count") > 0
            and type(row.get("has_answer")) is bool
            and (kind == "answer_atom") == (row.get("has_answer") is True)
            and receipt not in seen_positive_receipts,
            "positive witness authentication/semantics changed",
        )
        seen_positive_receipts.add(receipt)
        positive_by_target.setdefault(key, []).append(row)
        (direct_sources if kind == "answer_atom" else link_sources).add(key)
    for row in negative:
        _require(
            set(row)
            == {
                "content_char_count",
                "content_sha256",
                "exclusion_reason",
                "format",
                "has_answer",
                "ordinal",
                "question_id",
                "role",
                "session_turn_index",
                "target_source_id",
                "witness_kind",
                "witness_receipt_sha256",
            },
            "negative witness schema changed",
        )
        key = (
            _integer(row.get("ordinal"), "negative witness ordinal"),
            _text(row.get("question_id"), "negative witness question ID"),
            _text(
                row.get("target_source_id"), "negative witness target source ID"
            ),
        )
        receipt = _self_sha(
            row, "witness_receipt_sha256", "negative witness receipt"
        )
        content_sha = _sha(
            row.get("content_sha256"), "negative witness content"
        )
        _require(
            key in target_keys
            and row.get("format")
            == f"{WITNESS_MANIFEST_FORMAT}-negative-witness-v1"
            and _integer(
                row.get("session_turn_index"), "negative witness turn index"
            )
            >= 0
            and _text(row.get("role"), "negative witness role")
            in {"user", "assistant"}
            and _integer(
                row.get("content_char_count"), "negative witness character count"
            )
            > 0
            and type(row.get("has_answer")) is bool
            and bool(_text(row.get("witness_kind"), "negative witness kind"))
            and bool(_text(row.get("exclusion_reason"), "negative exclusion reason"))
            and receipt not in seen_negative_receipts
            and all(
                child.get("content_sha256") != content_sha for child in positive
            ),
            "negative witness authentication/semantics changed",
        )
        seen_negative_receipts.add(receipt)
        negative_by_target.setdefault(key, []).append(row)
    direct_count = sum(row.get("witness_kind") == "answer_atom" for row in positive)
    link_count = sum(row.get("witness_kind") == "relation_link" for row in positive)
    _require(
        manifest.get("format") == WITNESS_MANIFEST_FORMAT
        and manifest.get("dataset_file_sha256") == LOCKED_DATASET_FILE_SHA256
        and manifest.get("target_plan_file_sha256") == target_plan.sha256
        and manifest.get("target_plan_identity_sha256")
        == target_plan.payload.get("plan_sha256")
        and manifest.get("target_plan_population_identity_sha256")
        == target_plan.payload.get("population_identity_sha256")
        and manifest.get("exact_ordinals")
        == list(terminal_cli.EXACT_ORDINALS)
        and manifest.get("positive_witness_count") == POSITIVE_WITNESS_COUNT
        and manifest.get("direct_answer_witness_count")
        == DIRECT_ANSWER_WITNESS_COUNT
        and manifest.get("relation_link_witness_count")
        == RELATION_LINK_WITNESS_COUNT
        and manifest.get("negative_witness_count") == NEGATIVE_WITNESS_COUNT
        and manifest.get("source_target_count") == SOURCE_TARGET_COUNT
        and manifest.get("direct_answer_source_count") == 24
        and manifest.get("link_only_source_count") == 2
        and manifest.get("provider_calls") == 0
        and manifest.get("analysis_is_posthoc_only") is True
        and manifest.get("runtime_use_forbidden") is True
        and manifest.get("gold_loaded") is True
        and type(manifest.get("witness_policy")) is dict
        and _self_sha(
            _dict(manifest.get("witness_policy"), "witness policy"),
            "receipt_sha256",
            "witness policy receipt",
        )
        == manifest["witness_policy"]["receipt_sha256"]
        and direct_count == DIRECT_ANSWER_WITNESS_COUNT
        and link_count == RELATION_LINK_WITNESS_COUNT
        and len(direct_sources) == 24
        and len(link_sources) == 2
        and not direct_sources & link_sources
        and set(positive_by_target) == target_keys,
        "witness manifest population/binding changed",
    )
    return artifact, {
        "negative_by_target": {
            key: tuple(value) for key, value in negative_by_target.items()
        },
        "positive_by_target": {
            key: tuple(value) for key, value in positive_by_target.items()
        },
    }


def _verified_semantic_atom_manifest(
    path: Path,
    expected_file_sha256: str,
    expected_identity_sha256: str,
    expected_population_sha256: str,
    *,
    target_plan: SealedArtifact,
    witness_manifest: SealedArtifact,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    """Authenticate the static atom/equivalence declaration and bindings."""

    try:
        artifact = atom_cli.load_verified_manifest(
            path,
            expected_file_sha256,
            expected_target_plan_sha256=target_plan.sha256,
            expected_target_plan_identity_sha256=str(
                target_plan.payload.get("plan_sha256")
            ),
            expected_raw_witness_manifest_sha256=witness_manifest.sha256,
            expected_raw_witness_manifest_identity_sha256=str(
                witness_manifest.payload.get("manifest_identity_sha256")
            ),
        )
    except MatchedEvalContractError as exc:
        raise SemanticGlobalTerminalPostSealAuditError(
            "semantic atom manifest is absent or invalid"
        ) from exc
    payload = artifact.payload
    _require(
        payload.get("manifest_identity_sha256")
        == _sha(expected_identity_sha256, "semantic atom manifest identity")
        and payload.get("atom_population_sha256")
        == _sha(expected_population_sha256, "semantic atom population")
        and payload.get("atom_count") == SEMANTIC_ATOM_COUNT,
        "semantic atom manifest identity or population changed",
    )
    atoms = tuple(
        _dict(row, "semantic atom")
        for row in _list(payload.get("atoms"), "semantic atoms")
    )
    return artifact, atoms


def _utc_instant(value: object, label: str) -> str:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SemanticGlobalTerminalPostSealAuditError(
            f"{label} must be an ISO-8601 instant"
        ) from exc
    _require(parsed.tzinfo is not None, f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat()


def _evidence_temporal_identity(
    value: object,
    label: str,
) -> tuple[str, str]:
    """Keep calendar dates disjoint from timezone-qualified instants."""

    text = _text(value, label)
    if len(text) == 10:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            parsed_date = None
        if parsed_date is not None and parsed_date.isoformat() == text:
            return "calendar_date", text
    return "utc_instant", _utc_instant(text, label)


def _is_usable(
    item: Mapping[str, Any],
    conflict_policy: object,
    *,
    include_proposed: object,
) -> bool:
    return bool(
        item.get("included") is True
        and (
            item.get("content_conflict") is not True
            or conflict_policy == "fail_open"
        )
        and item.get("status") != "cancelled"
        and (item.get("status") != "proposed" or include_proposed is True)
    )


def _audit_terminal_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    ordinal = _integer(plan.get("ordinal"), "terminal plan ordinal")
    question_id = _text(plan.get("question_id"), "terminal plan question ID")
    compilation = _dict(plan.get("terminal_compilation"), "terminal compilation")
    local_audit = _dict(compilation.get("local_audit"), "terminal local audit")
    packet = _dict(compilation.get("packet"), "terminal packet")
    _require(
        local_audit.get("packet") == packet,
        f"terminal local/canonical packet diverged at ordinal {ordinal}",
    )
    _self_sha(packet, "receipt_sha256", "terminal packet receipt")
    _require(
        packet.get("provider_payload_mode") == "compact_final"
        and packet.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and packet.get("output_token_reserve")
        == PACKET_CONSTRUCTION_OUTPUT_TOKEN_RESERVE
        and packet.get("provider_prompt_count") == 0
        and packet.get("retained_transformer_token_state_bytes") == 0
        and packet.get("gold_loaded") is False,
        f"terminal packet budget/state contract changed at ordinal {ordinal}",
    )
    handles = tuple(
        _dict(row, "terminal opaque handle")
        for row in _list(packet.get("handles"), "terminal packet handles")
    )
    bindings = tuple(
        _dict(row, "terminal evidence binding")
        for row in _list(packet.get("local_bindings"), "terminal packet bindings")
    )
    items = tuple(
        _dict(row, "terminal evidence item")
        for row in _list(packet.get("items"), "terminal packet items")
    )
    handle_ids = tuple(_text(row.get("handle_id"), "terminal handle ID") for row in handles)
    binding_ids = tuple(_text(row.get("handle_id"), "terminal binding handle") for row in bindings)
    _require(
        len(set(handle_ids)) == len(handle_ids) and handle_ids == binding_ids,
        f"terminal handle/binding population changed at ordinal {ordinal}",
    )
    binding_by_handle = dict(zip(binding_ids, bindings, strict=True))
    for handle, binding in zip(handles, bindings, strict=True):
        binding_receipt = _self_sha(
            binding, "receipt_sha256", "terminal evidence binding receipt"
        )
        _require(
            handle.get("binding_receipt_sha256") == binding_receipt
            and handle.get("source_group_handle") == binding.get("source_group_handle")
            and handle.get("origin") == binding.get("origin")
            and handle.get("provenance_grade") == binding.get("provenance_grade")
            and binding.get("provenance_grade") == "exact_citation",
            f"terminal opaque handle lost exact provenance at ordinal {ordinal}",
        )
    item_by_handle: dict[str, list[dict[str, Any]]] = {handle: [] for handle in handle_ids}
    for item in items:
        _self_sha(item, "receipt_sha256", "terminal evidence item receipt")
        item_handles = _ordered_unique_text(
            item.get("handle_ids"), "terminal item handles"
        )
        _require(
            set(item_handles) <= set(handle_ids),
            f"terminal item cites unknown handle at ordinal {ordinal}",
        )
        for handle in item_handles:
            item_by_handle[handle].append(item)
    _require(
        all(item_by_handle.values()),
        f"terminal packet retains an unrepresented handle at ordinal {ordinal}",
    )

    provider_input = _dict(plan.get("provider_input"), "terminal provider input")
    expected_provider_packet = _compact_packet(packet)
    _require(
        provider_input.get("typed_evidence") == expected_provider_packet,
        f"terminal local packet is not byte-identically provider-visible at ordinal {ordinal}",
    )
    provider_packet = _dict(
        provider_input.get("typed_evidence"), "terminal provider packet"
    )
    provider_items = tuple(
        _dict(row, "terminal provider item")
        for row in _list(provider_packet.get("items"), "terminal provider items")
    )
    provider_item_by_handle: dict[str, list[dict[str, Any]]] = {
        handle: [] for handle in handle_ids
    }
    for item in provider_items:
        for handle in _ordered_unique_text(
            item.get("handle_ids"), "terminal provider item handles"
        ):
            _require(
                handle in provider_item_by_handle,
                f"provider item cites unknown handle at ordinal {ordinal}",
            )
            provider_item_by_handle[handle].append(item)
    _require(
        all(provider_item_by_handle.values()),
        f"terminal provider packet hides a retained handle at ordinal {ordinal}",
    )
    allowed = _ordered_unique_text(
        plan.get("allowed_handle_ids"), "terminal allowed handles"
    )
    operator = _dict(packet.get("operator_spec"), "terminal packet operator")
    usable_handles = tuple(
        dict.fromkeys(
            handle
            for item in items
            if _is_usable(
                item,
                packet.get("conflict_policy"),
                include_proposed=operator.get("include_proposed"),
            )
            for handle in _ordered_unique_text(
                item.get("handle_ids"), "terminal usable item handles"
            )
        )
    )
    _require(
        allowed == usable_handles,
        f"terminal allowed handles differ from usable packet at ordinal {ordinal}",
    )

    local_rows = tuple(
        _dict(row, "terminal local row")
        for row in _list(local_audit.get("local_rows"), "terminal local rows")
    )
    final_row_by_handle: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    retained_candidate_receipts: list[str] = []
    seen_local_candidate_receipts: set[str] = set()
    for row in local_rows:
        binding = _dict(row.get("binding"), "terminal local source binding")
        candidate = _dict(row.get("candidate"), "terminal local candidate")
        binding_receipt = _self_sha(
            binding, "receipt_sha256", "terminal local source binding receipt"
        )
        candidate_receipt = _self_sha(
            candidate, "receipt_sha256", "terminal local candidate receipt"
        )
        selected_before_dedup = _boolean(
            row.get("selected_by_independent_plane_budget"),
            "terminal independent-plane selection",
        )
        retained_after_dedup = _boolean(
            row.get("retained_after_post_selection_dedup"),
            "terminal post-dedup retention",
        )
        typed_value = row.get("typed_terminal")
        _require(
            candidate_receipt not in seen_local_candidate_receipts
            and (typed_value is not None) == retained_after_dedup
            and (not retained_after_dedup or selected_before_dedup),
            f"terminal outer selection/dedup mapping changed at ordinal {ordinal}",
        )
        seen_local_candidate_receipts.add(candidate_receipt)
        if typed_value is None:
            continue
        typed = _dict(typed_value, "terminal typed local row")
        _require(
            typed.get("binding") == binding
            and typed.get("candidate") == candidate,
            f"terminal typed row diverged from its outer authority at ordinal {ordinal}",
        )
        handle = _text(typed.get("final_handle_id"), "terminal final handle")
        admitted = _boolean(
            typed.get("admitted_to_compact_packet"), "terminal compact admission"
        )
        retained = _boolean(
            typed.get("retained_in_final_prompt"), "terminal final retention"
        )
        candidate_quote_sha = _sha(
            candidate.get("quote_sha256"), "terminal candidate quote"
        )
        local_source_id = _text(binding.get("source_id"), "terminal source ID")
        _require(
            candidate.get("binding_receipt_sha256") == binding_receipt
            and binding.get("quote_sha256") == candidate_quote_sha
            and (handle in binding_by_handle) == retained
            and (not retained or admitted)
            and (not retained or handle not in final_row_by_handle),
            f"terminal local selection/final mapping changed at ordinal {ordinal}",
        )
        chain: dict[str, Any] | None = None
        if retained:
            final_row_by_handle[handle] = row
            evidence_binding = binding_by_handle[handle]
            source_span = _dict(binding.get("span"), "terminal source span")
            terminal_source_date_utc = _utc_instant(
                source_span.get("created_at"),
                "terminal exact source date",
            )
            _require(
                terminal_source_date_utc
                == _utc_instant(
                    candidate.get("created_at"),
                    "terminal candidate source date",
                ),
                f"terminal candidate/source date diverged at ordinal {ordinal}",
            )
            evidence_receipt = _sha(
                evidence_binding.get("receipt_sha256"),
                "terminal evidence binding receipt",
            )
            exact_items = item_by_handle[handle]
            exact_provider_items = provider_item_by_handle[handle]
            _require(
                evidence_binding.get("local_source_locator_sha256") == binding_receipt
                and evidence_binding.get("evidence_receipt_sha256") == candidate_receipt
                and evidence_binding.get("payload_sha256")
                == identity_sha256(candidate)
                and evidence_binding.get("citation_sha256") == candidate_quote_sha
                and all(
                    quote_sha256(_text(item.get("summary"), "terminal item summary"))
                    == candidate_quote_sha
                    for item in exact_items
                )
                and all(
                    quote_sha256(
                        _text(item.get("summary"), "terminal provider summary")
                    )
                    == candidate_quote_sha
                    for item in exact_provider_items
                ),
                f"terminal exact citation chain changed at ordinal {ordinal}",
            )
            provider_quote = _text(
                exact_provider_items[0].get("summary"),
                "terminal provider-visible quote",
            )
            _require(
                all(item.get("summary") == provider_quote for item in exact_provider_items),
                f"one handle maps to divergent provider quotes at ordinal {ordinal}",
            )
            temporal_identities = tuple(
                _evidence_temporal_identity(
                    item.get("date"), "terminal provider-visible evidence date"
                )
                for item in exact_provider_items
                if item.get("date") is not None
            )
            retained_candidate_receipts.append(candidate_receipt)
            chain = {
                "candidate_receipt_sha256": candidate_receipt,
                "citation_binding_receipt_sha256": binding_receipt,
                "evidence_binding_receipt_sha256": evidence_receipt,
                "final_handle_id": handle,
                "packet_item_receipt_sha256s": [
                    _sha(item.get("receipt_sha256"), "terminal item receipt")
                    for item in exact_items
                ],
                "plane": candidate.get("plane"),
                "provider_item_sha256s": [
                    identity_sha256(item) for item in exact_provider_items
                ],
                "provider_usable": handle in set(allowed),
                "provider_visible_calendar_dates": list(
                    dict.fromkeys(
                        value
                        for kind, value in temporal_identities
                        if kind == "calendar_date"
                    )
                ),
                "provider_visible_quote": provider_quote,
                "provider_visible_quote_sha256": candidate_quote_sha,
                "provider_visible_utc_instants": list(
                    dict.fromkeys(
                        value
                        for kind, value in temporal_identities
                        if kind == "utc_instant"
                    )
                ),
                "terminal_source_date_utc": terminal_source_date_utc,
                "terminal_source_id": local_source_id,
            }
        selected_rows.append(
            {
                "admitted_to_compact_packet": admitted,
                "binding_receipt_sha256": binding_receipt,
                "candidate_receipt_sha256": candidate_receipt,
                "final_chain": chain,
                "final_handle_id": handle,
                "plane": candidate.get("plane"),
                "quote_sha256": candidate_quote_sha,
                "retained_after_post_selection_dedup": retained_after_dedup,
                "retained_in_final_prompt": retained,
                "selected_by_independent_plane_budget": selected_before_dedup,
                "source_id": local_source_id,
            }
        )
    declared_retained_receipts = _ordered_unique_text(
        compilation.get("retained_row_receipt_sha256s"),
        "terminal retained row receipts",
    )
    _require(
        len(retained_candidate_receipts) == len(declared_retained_receipts)
        and set(retained_candidate_receipts) == set(declared_retained_receipts)
        and set(final_row_by_handle) == set(handle_ids),
        f"terminal retained-row receipt population changed at ordinal {ordinal}",
    )

    prompt_tokens = _integer(plan.get("prompt_token_proxy"), "terminal prompt tokens")
    reserve = _integer(plan.get("output_token_reserve"), "terminal output reserve")
    hard_cap = _integer(plan.get("hard_prompt_token_cap"), "terminal hard cap")
    messages = render_final_messages(provider_input)
    _require(
        count_chat_prompt_token_proxy(messages) == prompt_tokens
        and prompt_tokens + reserve <= hard_cap
        and hard_cap == HARD_PROMPT_TOKEN_CAP
        and reserve == OUTPUT_TOKEN_RESERVE
        and compilation.get("new_provider_calls") == 0
        and compilation.get("retained_transformer_token_state_bytes") == 0,
        f"terminal chat budget/state invariant changed at ordinal {ordinal}",
    )
    return {
        "hard_prompt_token_cap": hard_cap,
        "ordinal": ordinal,
        "output_token_reserve": reserve,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider_input),
        "question_id": question_id,
        "selected_rows": selected_rows,
    }


def build_audit(
    *,
    construction: SealedArtifact,
    replay: SealedArtifact,
    plans: Sequence[Mapping[str, Any]],
    target_plan: SealedArtifact,
    source_targets: Sequence[Mapping[str, Any]],
    witness_manifest: SealedArtifact | None = None,
    witness_index: Mapping[str, Any] | None = None,
    semantic_atom_manifest: SealedArtifact | None = None,
    semantic_atoms: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _require(
        construction.sha256 == replay.sha256
        and construction.payload == replay.payload,
        "terminal construction/replay changed before post-seal audit",
    )
    _require(
        len(plans) == len(terminal_cli.EXACT_ORDINALS)
        and tuple(plan.get("ordinal") for plan in plans)
        == terminal_cli.EXACT_ORDINALS,
        "terminal exact11 plan population changed",
    )
    _require(
        (semantic_atom_manifest is None) == (semantic_atoms is None)
        and (semantic_atom_manifest is None or witness_manifest is not None),
        "semantic atom audit requires both manifests or neither",
    )
    audited_plans = tuple(_audit_terminal_plan(plan) for plan in plans)
    by_ordinal = {row["ordinal"]: row for row in audited_plans}
    target_question_ids = {
        row["ordinal"]: row["question_id"] for row in source_targets
    }
    _require(
        all(
            by_ordinal[ordinal]["question_id"] == question_id
            for ordinal, question_id in target_question_ids.items()
        ),
        "target plan differs from sealed terminal question identities",
    )

    target_literals = tuple(
        dict.fromkeys(
            str(value)
            for row in source_targets
            for value in (
                row.get("question_id"),
                row.get("target_id"),
                row.get("target_sha256"),
                row.get("assignment_basis_sha256"),
            )
        )
    )
    if witness_index is not None:
        positive_by_target = _dict(
            witness_index.get("positive_by_target"), "positive witness index"
        )
        negative_by_target = _dict(
            witness_index.get("negative_by_target"), "negative witness index"
        )
        target_literals = tuple(
            dict.fromkeys(
                (
                    *target_literals,
                    *(
                        str(row["content_sha256"])
                        for groups in (positive_by_target, negative_by_target)
                        for rows in groups.values()
                        for row in rows
                    ),
                )
            )
        )
    else:
        positive_by_target = {}
        negative_by_target = {}
    if semantic_atoms is not None:
        target_literals = tuple(
            dict.fromkeys(
                (
                    *target_literals,
                    *(
                        str(value)
                        for atom in semantic_atoms
                        for value in (
                            atom.get("atom_key"),
                            atom.get("atom_receipt_sha256"),
                            atom.get("canonical_claim"),
                            *(
                                locator.get("locator_receipt_sha256")
                                for locator in _list(
                                    atom.get("acceptable_evidence_locators"),
                                    "semantic atom locators",
                                )
                                if type(locator) is dict
                            ),
                        )
                    ),
                )
            )
        )
    for plan in plans:
        provider_json = json.dumps(
            plan.get("provider_input"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        _require(
            all(value not in provider_json for value in target_literals),
            "posthoc source/target label was already present in provider projection",
        )

    target_rows: list[dict[str, Any]] = []
    for target in source_targets:
        ordinal = int(target["ordinal"])
        question_id = str(target["question_id"])
        source_id = str(target["target_id"])
        terminal_source_id = f"{question_id}::{source_id}"
        rows = [
            row
            for row in by_ordinal[ordinal]["selected_rows"]
            if row["source_id"] == terminal_source_id
        ]
        selected_shas = tuple(dict.fromkeys(row["quote_sha256"] for row in rows))
        admitted_shas = tuple(
            dict.fromkeys(
                row["quote_sha256"]
                for row in rows
                if row["admitted_to_compact_packet"]
            )
        )
        final_chains = [
            row["final_chain"] for row in rows if row["final_chain"] is not None
        ]
        final_shas = tuple(
            dict.fromkeys(
                row["provider_visible_quote_sha256"] for row in final_chains
            )
        )
        usable_shas = tuple(
            dict.fromkeys(
                row["provider_visible_quote_sha256"]
                for row in final_chains
                if row["provider_usable"]
            )
        )
        key = (ordinal, question_id, source_id)
        expected_positive = tuple(positive_by_target.get(key, ()))
        expected_negative = tuple(negative_by_target.get(key, ()))
        positive_results: list[dict[str, Any]] = []
        for witness in expected_positive:
            digest = str(witness["content_sha256"])
            positive_results.append(
                {
                    **dict(witness),
                    "admitted_to_compact_packet": digest in admitted_shas,
                    "matching_final_handle_ids": [
                        chain["final_handle_id"]
                        for chain in final_chains
                        if chain["provider_visible_quote_sha256"] == digest
                    ],
                    "selected_after_dedup": digest in selected_shas,
                    "visible_and_usable": digest in usable_shas,
                    "visible_in_final_provider_packet": digest in final_shas,
                }
            )
        negative_results: list[dict[str, Any]] = []
        for witness in expected_negative:
            digest = str(witness["content_sha256"])
            negative_results.append(
                {
                    **dict(witness),
                    "admitted_to_compact_packet": digest in admitted_shas,
                    "matching_final_handle_ids": [
                        chain["final_handle_id"]
                        for chain in final_chains
                        if chain["provider_visible_quote_sha256"] == digest
                    ],
                    "selected_after_dedup": digest in selected_shas,
                    "visible_and_usable": digest in usable_shas,
                    "visible_in_final_provider_packet": digest in final_shas,
                }
            )
        fact_selected = (
            all(row["selected_after_dedup"] for row in positive_results)
            if witness_index is not None
            else None
        )
        fact_visible = (
            all(row["visible_in_final_provider_packet"] for row in positive_results)
            if witness_index is not None
            else None
        )
        fact_usable = (
            all(row["visible_and_usable"] for row in positive_results)
            if witness_index is not None
            else None
        )
        target_rows.append(
            {
                "admitted_quote_sha256s": list(admitted_shas),
                "expected_fact_quote_sha256s": (
                    [row["content_sha256"] for row in positive_results]
                    if witness_index is not None
                    else None
                ),
                "fact_selected_after_dedup": fact_selected,
                "fact_visible_in_final_provider_packet": fact_visible,
                "fact_visible_and_usable": fact_usable,
                "final_provider_chains": final_chains,
                "final_visible_quote_sha256s": list(final_shas),
                "negative_witness_results": (
                    negative_results if witness_index is not None else None
                ),
                "ordinal": ordinal,
                "positive_witness_results": (
                    positive_results if witness_index is not None else None
                ),
                "primary_owner": target.get("primary_owner"),
                "question_id": question_id,
                "selected_quote_sha256s": list(selected_shas),
                "source_admitted_to_compact_packet": bool(admitted_shas),
                "source_selected_after_dedup": bool(selected_shas),
                "source_target_id": source_id,
                "source_target_sha256": target.get("target_sha256"),
                "source_visible_and_usable": bool(usable_shas),
                "source_visible_in_final_provider_packet": bool(final_shas),
                "terminal_source_id": terminal_source_id,
                "usable_quote_sha256s": list(usable_shas),
            }
        )

    semantic_atom_results: list[dict[str, Any]] = []
    if semantic_atoms is not None:
        target_by_terminal_source = {
            str(row["terminal_source_id"]): row for row in target_rows
        }
        for raw_atom in semantic_atoms:
            atom = _dict(raw_atom, "semantic atom")
            atom_receipt = _sha(
                atom.get("atom_receipt_sha256"), "semantic atom receipt"
            )
            acceptable = tuple(
                _dict(row, "semantic atom locator")
                for row in _list(
                    atom.get("acceptable_evidence_locators"),
                    "semantic atom locators",
                )
            )
            selected_receipts: list[str] = []
            admitted_receipts: list[str] = []
            visible_receipts: list[str] = []
            usable_receipts: list[str] = []
            usable_handles: list[str] = []
            for locator in acceptable:
                locator_receipt = _sha(
                    locator.get("locator_receipt_sha256"),
                    "semantic atom locator receipt",
                )
                terminal_source_id = (
                    f"{_text(locator.get('question_id'), 'atom question ID')}::"
                    f"{_text(locator.get('source_id'), 'atom source ID')}"
                )
                target = target_by_terminal_source.get(terminal_source_id)
                _require(
                    target is not None
                    and locator.get("ordinal") == atom.get("ordinal")
                    and locator.get("question_id") == atom.get("question_id"),
                    "semantic atom locator escaped the exact11 target population",
                )
                quote_sha = _sha(
                    locator.get("content_sha256"), "semantic atom quote"
                )
                expected_date = _utc_instant(
                    locator.get("source_date_utc"), "semantic atom source date"
                )
                if quote_sha in target["selected_quote_sha256s"]:
                    selected_receipts.append(locator_receipt)
                if quote_sha in target["admitted_quote_sha256s"]:
                    admitted_receipts.append(locator_receipt)
                matching_chains = [
                    chain
                    for chain in target["final_provider_chains"]
                    if chain.get("terminal_source_id") == terminal_source_id
                    and chain.get("provider_visible_quote_sha256") == quote_sha
                ]
                if matching_chains:
                    visible_receipts.append(locator_receipt)
                usable_chains = [
                    chain
                    for chain in matching_chains
                    if chain.get("provider_usable") is True
                    and chain.get("terminal_source_date_utc") == expected_date
                ]
                if usable_chains:
                    usable_receipts.append(locator_receipt)
                    usable_handles.extend(
                        _text(chain.get("final_handle_id"), "semantic atom handle")
                        for chain in usable_chains
                    )
            semantic_atom_results.append(
                {
                    "admitted_to_compact_packet": bool(admitted_receipts),
                    "atom_manifest_row": dict(atom),
                    "atom_receipt_sha256": atom_receipt,
                    "matching_admitted_locator_receipt_sha256s": list(
                        dict.fromkeys(admitted_receipts)
                    ),
                    "matching_final_handle_ids": list(dict.fromkeys(usable_handles)),
                    "matching_selected_locator_receipt_sha256s": list(
                        dict.fromkeys(selected_receipts)
                    ),
                    "matching_usable_locator_receipt_sha256s": list(
                        dict.fromkeys(usable_receipts)
                    ),
                    "matching_visible_locator_receipt_sha256s": list(
                        dict.fromkeys(visible_receipts)
                    ),
                    "ordinal": _integer(atom.get("ordinal"), "semantic atom ordinal"),
                    "question_id": _text(
                        atom.get("question_id"), "semantic atom question ID"
                    ),
                    "selected_after_dedup": bool(selected_receipts),
                    "visible_and_usable": bool(usable_receipts),
                    "visible_in_final_provider_packet": bool(visible_receipts),
                }
            )

    per_ordinal: list[dict[str, Any]] = []
    for ordinal in terminal_cli.EXACT_ORDINALS:
        rows = [row for row in target_rows if row["ordinal"] == ordinal]
        ordinal_atoms = [
            row for row in semantic_atom_results if row["ordinal"] == ordinal
        ]
        positive_results = [
            witness
            for row in rows
            for witness in (row["positive_witness_results"] or [])
        ]
        negative_results = [
            witness
            for row in rows
            for witness in (row["negative_witness_results"] or [])
        ]
        per_ordinal.append(
            {
                "fact_final_usable_count": (
                    sum(row["visible_and_usable"] is True for row in positive_results)
                    if witness_index is not None
                    else None
                ),
                "fact_selected_count": (
                    sum(row["selected_after_dedup"] is True for row in positive_results)
                    if witness_index is not None
                    else None
                ),
                "negative_witness_final_visible_count": (
                    sum(
                        row["visible_in_final_provider_packet"] is True
                        for row in negative_results
                    )
                    if witness_index is not None
                    else None
                ),
                "ordinal": ordinal,
                "positive_witness_count": (
                    len(positive_results) if witness_index is not None else None
                ),
                "question_id": by_ordinal[ordinal]["question_id"],
                "semantic_atom_count": (
                    len(ordinal_atoms) if semantic_atoms is not None else None
                ),
                "semantic_atom_final_usable_count": (
                    sum(row["visible_and_usable"] is True for row in ordinal_atoms)
                    if semantic_atoms is not None
                    else None
                ),
                "semantic_atom_final_visible_count": (
                    sum(
                        row["visible_in_final_provider_packet"] is True
                        for row in ordinal_atoms
                    )
                    if semantic_atoms is not None
                    else None
                ),
                "semantic_atom_selected_count": (
                    sum(row["selected_after_dedup"] is True for row in ordinal_atoms)
                    if semantic_atoms is not None
                    else None
                ),
                "source_admitted_count": sum(
                    row["source_admitted_to_compact_packet"] for row in rows
                ),
                "source_final_usable_count": sum(
                    row["source_visible_and_usable"] for row in rows
                ),
                "source_final_visible_count": sum(
                    row["source_visible_in_final_provider_packet"] for row in rows
                ),
                "source_selected_count": sum(
                    row["source_selected_after_dedup"] for row in rows
                ),
                "source_target_count": len(rows),
                "witness_manifest_available": witness_index is not None,
            }
        )

    source_selected = sum(row["source_selected_after_dedup"] for row in target_rows)
    source_admitted = sum(
        row["source_admitted_to_compact_packet"] for row in target_rows
    )
    source_visible = sum(
        row["source_visible_in_final_provider_packet"] for row in target_rows
    )
    source_usable = sum(row["source_visible_and_usable"] for row in target_rows)
    positive_results = [
        witness
        for row in target_rows
        for witness in (row["positive_witness_results"] or [])
    ]
    negative_results = [
        witness
        for row in target_rows
        for witness in (row["negative_witness_results"] or [])
    ]
    if semantic_atoms is None:
        semantic_atom_selected_count: int | None = None
        semantic_atom_admitted_count: int | None = None
        semantic_atom_visible_count: int | None = None
        semantic_atom_usable_count: int | None = None
        semantic_atom_status = "indeterminate_missing_authenticated_atom_manifest"
    else:
        semantic_atom_selected_count = sum(
            row["selected_after_dedup"] is True for row in semantic_atom_results
        )
        semantic_atom_admitted_count = sum(
            row["admitted_to_compact_packet"] is True
            for row in semantic_atom_results
        )
        semantic_atom_visible_count = sum(
            row["visible_in_final_provider_packet"] is True
            for row in semantic_atom_results
        )
        semantic_atom_usable_count = sum(
            row["visible_and_usable"] is True for row in semantic_atom_results
        )
        semantic_atom_status = (
            "proven_all_26_semantic_atoms"
            if semantic_atom_usable_count == SEMANTIC_ATOM_COUNT
            else "failed_semantic_atom_visibility_or_usability"
        )
    if witness_index is None:
        fact_selected_count: int | None = None
        fact_visible_count: int | None = None
        fact_usable_count: int | None = None
        negative_visible_count: int | None = None
        semantic_status = "indeterminate_missing_authenticated_witness_manifest"
    else:
        fact_selected_count = sum(
            row["selected_after_dedup"] is True for row in positive_results
        )
        fact_visible_count = sum(
            row["visible_in_final_provider_packet"] is True
            for row in positive_results
        )
        fact_usable_count = sum(
            row["visible_and_usable"] is True for row in positive_results
        )
        negative_visible_count = sum(
            row["visible_in_final_provider_packet"] is True
            for row in negative_results
        )
        semantic_status = (
            "reported_all_31_raw_witnesses"
            if fact_usable_count == POSITIVE_WITNESS_COUNT
            else "reported_incomplete_raw_witness_visibility"
        )
    source_gate = source_usable == SOURCE_TARGET_COUNT
    promotion_gate = bool(
        semantic_atoms is not None
        and semantic_atom_usable_count == SEMANTIC_ATOM_COUNT
    )
    max_prompt = max(row["prompt_token_proxy"] for row in audited_plans)
    body = {
        "analysis_kind": "postseal_gold_informed_local_audit_only",
        "exact_provenance_invariant": True,
        "exact_terminal_ordinals": list(terminal_cli.EXACT_ORDINALS),
        "semantic_atom_manifest_artifact_sha256": (
            semantic_atom_manifest.sha256
            if semantic_atom_manifest is not None
            else None
        ),
        "semantic_atom_manifest_available": semantic_atoms is not None,
        "semantic_atom_manifest_identity_sha256": (
            semantic_atom_manifest.payload.get("manifest_identity_sha256")
            if semantic_atom_manifest is not None
            else None
        ),
        "semantic_atom_population_sha256": (
            semantic_atom_manifest.payload.get("atom_population_sha256")
            if semantic_atom_manifest is not None
            else None
        ),
        "semantic_atom_results": (
            semantic_atom_results if semantic_atoms is not None else None
        ),
        "witness_manifest_artifact_sha256": (
            witness_manifest.sha256 if witness_manifest is not None else None
        ),
        "witness_manifest_identity_sha256": (
            witness_manifest.payload.get("manifest_identity_sha256")
            if witness_manifest is not None
            else None
        ),
        "witness_manifest_available": witness_index is not None,
        "format": FORMAT,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "hard_prompt_token_cap_invariant": True,
        "max_prompt_token_proxy": max_prompt,
        "new_provider_calls": 0,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "per_ordinal": per_ordinal,
        "promotion_gate_passed": promotion_gate,
        "provider_projection_mutated": False,
        "provider_projection_use_forbidden": True,
        "retained_transformer_token_state_bytes": 0,
        "runtime_ranking_reexecuted": False,
        "runtime_use_forbidden": True,
        "raw_witness_coverage_status": semantic_status,
        "semantic_fact_coverage_status": semantic_atom_status,
        "source_level_gate_passed": source_gate,
        "target_literals_absent_from_provider_projection": True,
        "target_plan_artifact_sha256": target_plan.sha256,
        "target_plan_identity_sha256": target_plan.payload.get("plan_sha256"),
        "target_plan_loaded_after_terminal_seal": True,
        "target_rows": target_rows,
        "terminal_construction_sha256": construction.sha256,
        "terminal_replay_sha256": replay.sha256,
        "totals": {
            "fact_final_usable_count": fact_usable_count,
            "fact_final_visible_count": fact_visible_count,
            "fact_selected_count": fact_selected_count,
            "negative_witness_final_visible_count": negative_visible_count,
            "positive_witness_count": (
                len(positive_results) if witness_index is not None else None
            ),
            "raw_witness_final_usable_count": fact_usable_count,
            "raw_witness_final_visible_count": fact_visible_count,
            "raw_witness_selected_count": fact_selected_count,
            "raw_witness_count": (
                len(positive_results) if witness_index is not None else None
            ),
            "semantic_atom_admitted_count": semantic_atom_admitted_count,
            "semantic_atom_count": (
                len(semantic_atom_results) if semantic_atoms is not None else None
            ),
            "semantic_atom_final_usable_count": semantic_atom_usable_count,
            "semantic_atom_final_visible_count": semantic_atom_visible_count,
            "semantic_atom_selected_count": semantic_atom_selected_count,
            "source_admitted_count": source_admitted,
            "source_final_usable_count": source_usable,
            "source_final_visible_count": source_visible,
            "source_selected_count": source_selected,
            "source_target_count": len(target_rows),
        },
        "zero_persisted_transformer_token_state_invariant": True,
    }
    _require(
        len(target_rows) == SOURCE_TARGET_COUNT
        and (
            semantic_atoms is None
            or len(semantic_atom_results) == SEMANTIC_ATOM_COUNT
        )
        and max_prompt + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP,
        "post-seal audit aggregate invariant changed",
    )
    return {**body, "audit_identity_sha256": identity_sha256(body)}


def load_verified_promotion_audit(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_terminal_construction_sha256: str,
    expected_terminal_replay_sha256: str,
    expected_target_plan_sha256: str = DEFAULT_TARGET_PLAN_SHA256,
    expected_target_plan_identity_sha256: str = (
        DEFAULT_TARGET_PLAN_IDENTITY_SHA256
    ),
    expected_witness_manifest_sha256: str = DEFAULT_WITNESS_MANIFEST_SHA256,
    expected_witness_manifest_identity_sha256: str = (
        DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256
    ),
    expected_semantic_atom_manifest_sha256: str = (
        DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
    ),
    expected_semantic_atom_manifest_identity_sha256: str = (
        DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
    ),
    expected_semantic_atom_population_sha256: str = (
        DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
    ),
) -> SealedArtifact:
    """Authenticate the exact promotion result without reopening target literals.

    The post-seal audit is gold-informed local authorization metadata, never a
    provider input.  This reader checks its complete closure arithmetic and its
    bindings to the sealed terminal, target plan, raw witness manifest, and
    semantic-atom manifest.  Raw31 and source26 remain strict diagnostics;
    only exact semantic-atom completeness grants promotion authority.
    """

    artifact = read_sealed_json(Path(path))
    _require(
        artifact.sha256 == _sha(expected_sha256, "post-seal promotion audit"),
        "post-seal promotion audit artifact changed",
    )
    payload = artifact.payload
    _self_sha(payload, "audit_identity_sha256", "post-seal promotion audit")
    construction_sha = _sha(
        expected_terminal_construction_sha256,
        "post-seal terminal construction",
    )
    replay_sha = _sha(
        expected_terminal_replay_sha256,
        "post-seal terminal replay",
    )
    target_plan_sha = _sha(expected_target_plan_sha256, "post-seal target plan")
    target_plan_identity = _sha(
        expected_target_plan_identity_sha256,
        "post-seal target plan identity",
    )
    witness_sha = _sha(
        expected_witness_manifest_sha256,
        "post-seal witness manifest",
    )
    witness_identity = _sha(
        expected_witness_manifest_identity_sha256,
        "post-seal witness manifest identity",
    )
    atom_manifest_sha = _sha(
        expected_semantic_atom_manifest_sha256,
        "post-seal semantic atom manifest",
    )
    atom_manifest_identity = _sha(
        expected_semantic_atom_manifest_identity_sha256,
        "post-seal semantic atom manifest identity",
    )
    atom_population_sha = _sha(
        expected_semantic_atom_population_sha256,
        "post-seal semantic atom population",
    )
    totals = _dict(payload.get("totals"), "post-seal promotion totals")
    target_rows = tuple(
        _dict(row, "post-seal promotion target row")
        for row in _list(payload.get("target_rows"), "post-seal promotion targets")
    )
    per_ordinal = tuple(
        _dict(row, "post-seal promotion ordinal row")
        for row in _list(payload.get("per_ordinal"), "post-seal promotion ordinals")
    )
    atom_results = tuple(
        _dict(row, "post-seal semantic atom result")
        for row in _list(
            payload.get("semantic_atom_results"),
            "post-seal semantic atom results",
        )
    )
    _require(
        payload.get("format") == FORMAT
        and payload.get("analysis_kind")
        == "postseal_gold_informed_local_audit_only"
        and payload.get("promotion_gate_passed") is True
        and payload.get("semantic_fact_coverage_status")
        == "proven_all_26_semantic_atoms"
        and payload.get("semantic_atom_manifest_available") is True
        and payload.get("witness_manifest_available") is True
        and payload.get("exact_provenance_invariant") is True
        and payload.get("hard_prompt_token_cap_invariant") is True
        and payload.get("zero_persisted_transformer_token_state_invariant")
        is True
        and payload.get("provider_projection_mutated") is False
        and payload.get("provider_projection_use_forbidden") is True
        and payload.get("runtime_ranking_reexecuted") is False
        and payload.get("runtime_use_forbidden") is True
        and payload.get("target_literals_absent_from_provider_projection") is True
        and payload.get("target_plan_loaded_after_terminal_seal") is True
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("terminal_construction_sha256") == construction_sha
        and payload.get("terminal_replay_sha256") == replay_sha
        and payload.get("target_plan_artifact_sha256") == target_plan_sha
        and payload.get("target_plan_identity_sha256") == target_plan_identity
        and payload.get("witness_manifest_artifact_sha256") == witness_sha
        and payload.get("witness_manifest_identity_sha256") == witness_identity
        and payload.get("semantic_atom_manifest_artifact_sha256")
        == atom_manifest_sha
        and payload.get("semantic_atom_manifest_identity_sha256")
        == atom_manifest_identity
        and payload.get("semantic_atom_population_sha256")
        == atom_population_sha
        and payload.get("exact_terminal_ordinals")
        == list(terminal_cli.EXACT_ORDINALS)
        and payload.get("hard_prompt_token_cap") == HARD_PROMPT_TOKEN_CAP
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and type(payload.get("max_prompt_token_proxy")) is int
        and int(payload["max_prompt_token_proxy"]) + OUTPUT_TOKEN_RESERVE
        <= HARD_PROMPT_TOKEN_CAP,
        "post-seal promotion authority/budget binding changed",
    )
    _require(
        len(target_rows) == SOURCE_TARGET_COUNT
        and len(atom_results) == SEMANTIC_ATOM_COUNT
        and len(per_ordinal) == len(terminal_cli.EXACT_ORDINALS)
        and tuple(row.get("ordinal") for row in per_ordinal)
        == terminal_cli.EXACT_ORDINALS,
        "post-seal promotion population/order changed",
    )

    target_keys: list[tuple[int, str, str]] = []
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    for row in target_rows:
        ordinal = _integer(row.get("ordinal"), "post-seal target ordinal")
        question_id = _text(
            row.get("question_id"), "post-seal target question ID"
        )
        source_id = _text(
            row.get("source_target_id"), "post-seal target source ID"
        )
        target_keys.append((ordinal, question_id, source_id))
        positives = tuple(
            _dict(value, "post-seal positive witness result")
            for value in _list(
                row.get("positive_witness_results"),
                "post-seal positive witness results",
            )
        )
        negatives = tuple(
            _dict(value, "post-seal negative witness result")
            for value in _list(
                row.get("negative_witness_results"),
                "post-seal negative witness results",
            )
        )
        _require(
            row.get("terminal_source_id") == f"{question_id}::{source_id}"
            and type(row.get("source_visible_in_final_provider_packet")) is bool
            and type(row.get("source_visible_and_usable")) is bool
            and row.get("fact_visible_in_final_provider_packet")
            == all(
                witness.get("visible_in_final_provider_packet") is True
                for witness in positives
            )
            and row.get("fact_visible_and_usable")
            == all(witness.get("visible_and_usable") is True for witness in positives)
            and all(
                witness.get("ordinal") == ordinal
                and witness.get("question_id") == question_id
                and witness.get("target_source_id") == source_id
                and type(witness.get("visible_in_final_provider_packet")) is bool
                and type(witness.get("visible_and_usable")) is bool
                and bool(
                    _sha(
                        witness.get("witness_receipt_sha256"),
                        "post-seal positive witness receipt",
                    )
                )
                and bool(
                    _sha(
                        witness.get("content_sha256"),
                        "post-seal positive witness content",
                    )
                )
                for witness in positives
            ),
            "post-seal promotion target/raw-witness ledger changed",
        )
        positive_rows.extend(positives)
        negative_rows.extend(negatives)

    _require(
        len(set(target_keys)) == len(target_keys)
        and len(positive_rows) == POSITIVE_WITNESS_COUNT
        and len(negative_rows) == NEGATIVE_WITNESS_COUNT
        and len(
            {
                _sha(
                    row.get("witness_receipt_sha256"),
                    "post-seal positive witness receipt",
                )
                for row in positive_rows
            }
        )
        == POSITIVE_WITNESS_COUNT,
        "post-seal promotion target/witness identities changed",
    )

    target_by_terminal_source = {
        _text(row.get("terminal_source_id"), "post-seal terminal source"): row
        for row in target_rows
    }
    atom_receipts: list[str] = []
    for result in atom_results:
        atom = _dict(result.get("atom_manifest_row"), "post-seal atom manifest row")
        atom_receipt = _self_sha(atom, "atom_receipt_sha256", "semantic atom")
        _require(
            result.get("atom_receipt_sha256") == atom_receipt
            and result.get("ordinal") == atom.get("ordinal")
            and result.get("question_id") == atom.get("question_id"),
            "post-seal semantic atom identity changed",
        )
        atom_receipts.append(atom_receipt)
        locators = tuple(
            _dict(row, "post-seal semantic atom locator")
            for row in _list(
                atom.get("acceptable_evidence_locators"),
                "post-seal semantic atom locators",
            )
        )
        locator_by_receipt: dict[str, dict[str, Any]] = {}
        selected: list[str] = []
        admitted: list[str] = []
        visible: list[str] = []
        usable: list[str] = []
        handles: list[str] = []
        for locator in locators:
            locator_receipt = _self_sha(
                locator,
                "locator_receipt_sha256",
                "semantic atom locator",
            )
            locator_by_receipt[locator_receipt] = locator
            question_id = _text(locator.get("question_id"), "atom locator question")
            source_id = _text(locator.get("source_id"), "atom locator source")
            terminal_source_id = f"{question_id}::{source_id}"
            target = target_by_terminal_source.get(terminal_source_id)
            _require(
                target is not None
                and locator.get("ordinal") == atom.get("ordinal")
                and question_id == atom.get("question_id"),
                "semantic atom locator escaped target population",
            )
            quote_sha = _sha(locator.get("content_sha256"), "atom locator quote")
            expected_date = _utc_instant(
                locator.get("source_date_utc"), "atom locator source date"
            )
            if quote_sha in target.get("selected_quote_sha256s", []):
                selected.append(locator_receipt)
            if quote_sha in target.get("admitted_quote_sha256s", []):
                admitted.append(locator_receipt)
            matching = [
                _dict(chain, "post-seal final provider chain")
                for chain in _list(
                    target.get("final_provider_chains"), "final provider chains"
                )
                if type(chain) is dict
                and chain.get("terminal_source_id") == terminal_source_id
                and chain.get("provider_visible_quote_sha256") == quote_sha
            ]
            if matching:
                visible.append(locator_receipt)
            exact_usable = [
                chain
                for chain in matching
                if chain.get("provider_usable") is True
                and chain.get("terminal_source_date_utc") == expected_date
            ]
            if exact_usable:
                usable.append(locator_receipt)
                handles.extend(
                    _text(chain.get("final_handle_id"), "semantic atom handle")
                    for chain in exact_usable
                )
        declared_selected = _ordered_unique_text(
            result.get("matching_selected_locator_receipt_sha256s"),
            "semantic atom selected locators",
        )
        declared_admitted = _ordered_unique_text(
            result.get("matching_admitted_locator_receipt_sha256s"),
            "semantic atom admitted locators",
        )
        declared_visible = _ordered_unique_text(
            result.get("matching_visible_locator_receipt_sha256s"),
            "semantic atom visible locators",
        )
        declared_usable = _ordered_unique_text(
            result.get("matching_usable_locator_receipt_sha256s"),
            "semantic atom usable locators",
        )
        declared_handles = _ordered_unique_text(
            result.get("matching_final_handle_ids"), "semantic atom handles"
        )
        _require(
            set(locator_by_receipt) == {
                _sha(value, "semantic atom locator receipt")
                for value in locator_by_receipt
            }
            and declared_selected == tuple(dict.fromkeys(selected))
            and declared_admitted == tuple(dict.fromkeys(admitted))
            and declared_visible == tuple(dict.fromkeys(visible))
            and declared_usable == tuple(dict.fromkeys(usable))
            and declared_handles == tuple(dict.fromkeys(handles))
            and result.get("selected_after_dedup") == bool(selected)
            and result.get("admitted_to_compact_packet") == bool(admitted)
            and result.get("visible_in_final_provider_packet") == bool(visible)
            and result.get("visible_and_usable") == bool(usable)
            and result.get("visible_and_usable") is True,
            "post-seal semantic atom exact-locator closure changed",
        )
    _require(
        len(set(atom_receipts)) == SEMANTIC_ATOM_COUNT
        and identity_sha256(atom_receipts) == atom_population_sha,
        "post-seal semantic atom population changed",
    )

    source_selected = sum(
        row.get("source_selected_after_dedup") is True for row in target_rows
    )
    source_admitted = sum(
        row.get("source_admitted_to_compact_packet") is True for row in target_rows
    )
    source_visible = sum(
        row.get("source_visible_in_final_provider_packet") is True
        for row in target_rows
    )
    source_usable = sum(
        row.get("source_visible_and_usable") is True for row in target_rows
    )
    fact_selected = sum(
        row.get("selected_after_dedup") is True for row in positive_rows
    )
    fact_visible = sum(
        row.get("visible_in_final_provider_packet") is True
        for row in positive_rows
    )
    fact_usable = sum(row.get("visible_and_usable") is True for row in positive_rows)
    negative_visible = sum(
        row.get("visible_in_final_provider_packet") is True for row in negative_rows
    )
    atom_selected = sum(
        row.get("selected_after_dedup") is True for row in atom_results
    )
    atom_admitted = sum(
        row.get("admitted_to_compact_packet") is True for row in atom_results
    )
    atom_visible = sum(
        row.get("visible_in_final_provider_packet") is True for row in atom_results
    )
    atom_usable = sum(
        row.get("visible_and_usable") is True for row in atom_results
    )
    _require(
        totals
        == {
            "fact_final_usable_count": fact_usable,
            "fact_final_visible_count": fact_visible,
            "fact_selected_count": fact_selected,
            "negative_witness_final_visible_count": negative_visible,
            "positive_witness_count": len(positive_rows),
            "raw_witness_final_usable_count": fact_usable,
            "raw_witness_final_visible_count": fact_visible,
            "raw_witness_selected_count": fact_selected,
            "raw_witness_count": len(positive_rows),
            "semantic_atom_admitted_count": atom_admitted,
            "semantic_atom_count": len(atom_results),
            "semantic_atom_final_usable_count": atom_usable,
            "semantic_atom_final_visible_count": atom_visible,
            "semantic_atom_selected_count": atom_selected,
            "source_admitted_count": source_admitted,
            "source_final_usable_count": source_usable,
            "source_final_visible_count": source_visible,
            "source_selected_count": source_selected,
            "source_target_count": len(target_rows),
        }
        and atom_visible == atom_usable == SEMANTIC_ATOM_COUNT
        and payload.get("source_level_gate_passed")
        == (source_usable == SOURCE_TARGET_COUNT)
        and payload.get("raw_witness_coverage_status")
        == (
            "reported_all_31_raw_witnesses"
            if fact_usable == POSITIVE_WITNESS_COUNT
            else "reported_incomplete_raw_witness_visibility"
        ),
        "post-seal promotion aggregate closure changed",
    )

    target_rows_by_ordinal = {
        ordinal: tuple(row for row in target_rows if row.get("ordinal") == ordinal)
        for ordinal in terminal_cli.EXACT_ORDINALS
    }
    for row in per_ordinal:
        ordinal = _integer(row.get("ordinal"), "post-seal ordinal")
        targets = target_rows_by_ordinal[ordinal]
        positives = [
            witness
            for target in targets
            for witness in target["positive_witness_results"]
        ]
        negatives = [
            witness
            for target in targets
            for witness in target["negative_witness_results"]
        ]
        atoms = [value for value in atom_results if value.get("ordinal") == ordinal]
        _require(
            row.get("witness_manifest_available") is True
            and row.get("source_target_count") == len(targets)
            and row.get("source_selected_count")
            == sum(target.get("source_selected_after_dedup") is True for target in targets)
            and row.get("source_admitted_count")
            == sum(
                target.get("source_admitted_to_compact_packet") is True
                for target in targets
            )
            and row.get("source_final_visible_count")
            == sum(
                target.get("source_visible_in_final_provider_packet") is True
                for target in targets
            )
            and row.get("source_final_usable_count")
            == sum(
                target.get("source_visible_and_usable") is True for target in targets
            )
            and row.get("positive_witness_count") == len(positives)
            and row.get("fact_selected_count")
            == sum(witness.get("selected_after_dedup") is True for witness in positives)
            and row.get("fact_final_usable_count")
            == sum(
                witness.get("visible_and_usable") is True for witness in positives
            )
            and row.get("semantic_atom_count") == len(atoms)
            and row.get("semantic_atom_selected_count")
            == sum(atom.get("selected_after_dedup") is True for atom in atoms)
            and row.get("semantic_atom_final_visible_count")
            == sum(
                atom.get("visible_in_final_provider_packet") is True
                for atom in atoms
            )
            and row.get("semantic_atom_final_usable_count") == len(atoms)
            and row.get("negative_witness_final_visible_count")
            == sum(
                witness.get("visible_in_final_provider_packet") is True
                for witness in negatives
            ),
            f"post-seal promotion ordinal {ordinal} arithmetic changed",
        )
    return artifact


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    # Ordering is a security boundary: do not stat, open, or parse the target
    # plan until both terminal artifacts and their V7 ancestry are sealed and
    # strictly authenticated by the production reader.
    construction, replay, plans = terminal_cli.load_verified_terminal_assay(
        args.terminal_root,
        args.expected_construction_sha256,
        args.expected_replay_sha256,
        v7_source_root=args.v7_source_root,
    )
    target_plan, source_targets = _verified_target_plan(
        Path(args.target_plan),
        args.expected_target_plan_sha256,
        args.expected_target_plan_identity_sha256,
    )
    witness_manifest: SealedArtifact | None = None
    witness_index: dict[str, Any] | None = None
    if args.witness_manifest is not None:
        _require(
            args.expected_witness_manifest_sha256 is not None,
            "witness manifest requires its exact artifact SHA-256",
        )
        witness_manifest, witness_index = _verified_witness_manifest(
            Path(args.witness_manifest),
            args.expected_witness_manifest_sha256,
            target_plan=target_plan,
            source_targets=source_targets,
        )
    else:
        _require(
            args.expected_witness_manifest_sha256 is None,
            "witness SHA was supplied without a manifest",
        )
    semantic_atom_manifest: SealedArtifact | None = None
    semantic_atoms: tuple[dict[str, Any], ...] | None = None
    if args.semantic_atom_manifest is not None:
        _require(
            witness_manifest is not None
            and args.expected_semantic_atom_manifest_sha256 is not None
            and args.expected_semantic_atom_manifest_identity_sha256 is not None
            and args.expected_semantic_atom_population_sha256 is not None,
            "semantic atom manifest requires raw witness manifest and exact bindings",
        )
        semantic_atom_manifest, semantic_atoms = _verified_semantic_atom_manifest(
            Path(args.semantic_atom_manifest),
            args.expected_semantic_atom_manifest_sha256,
            args.expected_semantic_atom_manifest_identity_sha256,
            args.expected_semantic_atom_population_sha256,
            target_plan=target_plan,
            witness_manifest=witness_manifest,
        )
    else:
        _require(
            args.expected_semantic_atom_manifest_sha256 is None
            and args.expected_semantic_atom_manifest_identity_sha256 is None
            and args.expected_semantic_atom_population_sha256 is None,
            "semantic atom binding was supplied without its manifest",
        )
    payload = build_audit(
        construction=construction,
        replay=replay,
        plans=plans,
        target_plan=target_plan,
        source_targets=source_targets,
        witness_manifest=witness_manifest,
        witness_index=witness_index,
        semantic_atom_manifest=semantic_atom_manifest,
        semantic_atoms=semantic_atoms,
    )
    output = (
        Path(args.output)
        if args.output is not None
        else Path(args.terminal_root)
        / (FACT_AUDIT_NAME if witness_manifest is not None else SOURCE_AUDIT_NAME)
    )
    artifact, created = publish_sealed_json(output, payload)
    return {
        "audit_path": str(artifact.path),
        "audit_sha256": artifact.sha256,
        "created": created,
        "fact_final_usable_count": payload["totals"]["fact_final_usable_count"],
        "new_provider_calls": 0,
        "promotion_gate_passed": payload["promotion_gate_passed"],
        "retained_transformer_token_state_bytes": 0,
        "semantic_fact_coverage_status": payload["semantic_fact_coverage_status"],
        "semantic_atom_final_usable_count": payload["totals"][
            "semantic_atom_final_usable_count"
        ],
        "semantic_atom_count": payload["totals"]["semantic_atom_count"],
        "source_final_usable_count": payload["totals"]["source_final_usable_count"],
        "source_target_count": SOURCE_TARGET_COUNT,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terminal-root", type=Path, default=terminal_cli.DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--expected-construction-sha256", required=True)
    parser.add_argument("--expected-replay-sha256", required=True)
    parser.add_argument(
        "--v7-source-root", type=Path, default=terminal_cli.DEFAULT_V7_SOURCE_ROOT
    )
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument(
        "--expected-target-plan-sha256", default=DEFAULT_TARGET_PLAN_SHA256
    )
    parser.add_argument(
        "--expected-target-plan-identity-sha256",
        default=DEFAULT_TARGET_PLAN_IDENTITY_SHA256,
    )
    parser.add_argument(
        "--witness-manifest", "--fact-span-manifest", type=Path
    )
    parser.add_argument(
        "--expected-witness-manifest-sha256",
        "--expected-fact-span-manifest-sha256",
    )
    parser.add_argument("--semantic-atom-manifest", type=Path)
    parser.add_argument("--expected-semantic-atom-manifest-sha256")
    parser.add_argument("--expected-semantic-atom-manifest-identity-sha256")
    parser.add_argument("--expected-semantic-atom-population-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--promotion-gate",
        action="store_true",
        help=(
            "exit nonzero unless the authenticated semantic-atom manifest "
            "proves all 26 required atoms provider-visible and usable"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_audit(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not args.promotion_gate or result["promotion_gate_passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEMANTIC_ATOM_MANIFEST",
    "DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256",
    "DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256",
    "DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256",
    "DEFAULT_TARGET_PLAN",
    "DEFAULT_TARGET_PLAN_IDENTITY_SHA256",
    "DEFAULT_TARGET_PLAN_SHA256",
    "DEFAULT_WITNESS_MANIFEST",
    "DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256",
    "DEFAULT_WITNESS_MANIFEST_SHA256",
    "FACT_AUDIT_NAME",
    "DIRECT_ANSWER_WITNESS_COUNT",
    "FORMAT",
    "LOCKED_DATASET_FILE_SHA256",
    "NEGATIVE_WITNESS_COUNT",
    "POSITIVE_WITNESS_COUNT",
    "RELATION_LINK_WITNESS_COUNT",
    "SEMANTIC_ATOM_COUNT",
    "SOURCE_AUDIT_NAME",
    "SOURCE_TARGET_COUNT",
    "SemanticGlobalTerminalPostSealAuditError",
    "WITNESS_MANIFEST_FORMAT",
    "build_audit",
    "build_parser",
    "load_verified_promotion_audit",
    "main",
    "run_audit",
]
