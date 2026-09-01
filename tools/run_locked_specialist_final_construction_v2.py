#!/usr/bin/env python3
"""Promote repaired specialist/operator routing into the locked 100 rows.

The sealed full-100 specialist-v1 construction is the immutable base.  This
successor preserves every non-target row byte-for-byte and replaces exactly:

* all ten question-only ``LATEST_STATE`` rows with the repaired temporal
  specialist route, grouped into one read/index pass per used namespace; and
* ordinals 42, 65, and 74 with the sealed missing-four v4 operator rows.

Construction is gold-blind, provider-free, and retains no transformer state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_specialist_final_answer as locked_answer  # noqa: E402
from tools import run_locked_specialist_final_construction as v1  # noqa: E402
from tools import run_locked_typed_memory_final_arm as typed_cli  # noqa: E402
from tools import run_reduced_missing4_v4_construction as v4  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools import run_reduced_specialist_answer_v2 as answer_v2  # noqa: E402
from tools import run_reduced_specialist_retrieval_assay as specialist  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    build_full_store_window_index,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval import semantic_residual_search as residual  # noqa: E402
from tools.matched_eval.temporal_insufficiency_specialist import (  # noqa: E402
    MECHANISM_ID as TEMPORAL_MECHANISM_ID,
)
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    TemporalMode,
    compile_typed_operator_spec,
)
from tools.matched_eval.typed_operator_adapter import (  # noqa: E402
    FrontierMode,
    ProviderPayloadMode,
    TypedEvidenceContribution,
    build_typed_evidence_packet,
    parse_typed_items,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    fit_typed_final_prompt,
)


FORMAT = "memory-condense-locked-specialist-final-v2"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction"
CONSTRUCTION_NAME = "locked-specialist-final-construction-v2.json"
STRUCTURAL_GATE_FORMAT = f"{FORMAT}-structural-gate-v1"
TEMPORAL_ROUTE_FORMAT = f"{FORMAT}-typed-latest-state-route-v1"
OPERATOR_ROUTE_FORMAT = f"{FORMAT}-sealed-repaired-operator-route-v1"
SCAN_RECEIPT_FORMAT = f"{FORMAT}-replacement-scan-receipt-v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_CONSTRUCTION = v1.DEFAULT_OUTPUT_ROOT / v1.CONSTRUCTION_NAME
DEFAULT_V4_CONSTRUCTION = v4.DEFAULT_OUTPUT_ROOT / v4.CONSTRUCTION_NAME
DEFAULT_PARENT_ROOT = v1.DEFAULT_PARENT_ROOT
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-v2"
)

EXPECTED_V1_CONSTRUCTION_SHA256 = (
    "21b50c5f6a318bf801c6523aef7680dd3c220f5bba5184a2b032fe341b4b9510"
)
EXPECTED_V4_CONSTRUCTION_SHA256 = (
    "4328f9334b858909a6511ee7114dd5d3dabf37c45393cf543ea05625fdb4cb43"
)
LATEST_STATE_ORDINALS = (19, 21, 24, 46, 51, 57, 70, 79, 83, 98)
REPAIRED_OPERATOR_ORDINALS = (42, 65, 74)
REPLACED_ORDINALS = tuple(sorted((*LATEST_STATE_ORDINALS, *REPAIRED_OPERATOR_ORDINALS)))
PRESERVED_ORDINALS = tuple(
    ordinal for ordinal in v1.ORDINALS if ordinal not in set(REPLACED_ORDINALS)
)
EXPECTED_REPLACEMENT_NAMESPACE_COUNT = 7
EXPECTED_PRESERVED_ROW_COUNT = 87
EXPECTED_REPLACED_ROW_COUNT = 13
EXPECTED_REPAIRED_OPERATOR_COUNT = 3
EXPECTED_SPECIALIST_COUNT = 69
EXPECTED_PROVIDER_PROMPT_COUNT = 72
EXPECTED_PASSTHROUGH_COUNT = 28
HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000


class LockedSpecialistFinalConstructionV2Error(MatchedEvalContractError):
    """A sealed parent, replacement, provenance, or prompt invariant changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSpecialistFinalConstructionV2Error(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _with_receipt(
    body: Mapping[str, Any], key: str = "receipt_sha256"
) -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def _validate_receipt(
    value: object,
    *,
    label: str,
    key: str = "receipt_sha256",
) -> dict[str, Any]:
    row = _exact_dict(value, label)
    body = dict(row)
    declared = require_sha256(body.pop(key, None), label)
    _require(identity_sha256(body) == declared, f"{label} receipt changed")
    return row


def _load_v1(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_V1_CONSTRUCTION_SHA256,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    return v1.load_verified_construction(path, expected_sha256=expected_sha256)


def _load_v4(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_V4_CONSTRUCTION_SHA256,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "missing-four v4 parent"),
        "missing-four v4 construction artifact changed",
    )
    return artifact, v4.validate_construction(artifact)


def _fixed_temporal_route(
    *,
    question_sha256: str,
    legacy_route: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "applicable_specialist_ids": [TEMPORAL_MECHANISM_ID],
        "format": TEMPORAL_ROUTE_FORMAT,
        "legacy_route": dict(legacy_route),
        "question_sha256": require_sha256(question_sha256, "temporal route question"),
        "route_basis": "typed_operator_spec.temporal_mode",
        "temporal_mode": TemporalMode.LATEST_STATE.value,
    }
    return _with_receipt(body)


def _fixed_operator_route(v4_row: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "applicable_mechanism_ids": list(v4_row.get("mechanism_plan", ())),
        "format": OPERATOR_ROUTE_FORMAT,
        "question_sha256": require_sha256(
            v4_row.get("question_sha256"), "operator route question"
        ),
        "route_basis": "sealed_reduced_missing4_v4",
        "source_question_receipt_sha256": require_sha256(
            v4_row.get("question_receipt_sha256"), "operator source question"
        ),
    }
    return _with_receipt(body)


def _terminal_advisories(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    terminal = _exact_dict(row.get("terminal_prompt"), "replacement terminal")
    provider_input = _exact_dict(
        terminal.get("provider_input"), "replacement provider input"
    )
    return tuple(
        _exact_dict(value, "replacement advisory")
        for value in _exact_list(
            provider_input.get("specialist_advisories"),
            "replacement advisories",
        )
    )


def _temporal_replacement(
    *,
    ordinal: int,
    index: Any,
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    base_row: Mapping[str, Any],
    v4_q79: Mapping[str, Any],
) -> dict[str, Any]:
    dated_question, _parent_prediction, question_id = specialist._question_inputs(  # noqa: SLF001
        composition_row
    )
    spec = compile_typed_operator_spec(dated_question)
    _require(
        ordinal in LATEST_STATE_ORDINALS
        and spec.temporal_mode is TemporalMode.LATEST_STATE
        and specialist.applicable_specialist_ids(dated_question)
        == (TEMPORAL_MECHANISM_ID,)
        and base_row.get("mode") == "parent_passthrough"
        and base_row.get("question_id") == question_id,
        f"latest-state replacement routing changed at ordinal {ordinal}",
    )
    generic = v1._specialist_or_passthrough_question(  # noqa: SLF001
        ordinal=ordinal,
        namespace_id=require_sha256(base_row.get("namespace_id"), "namespace"),
        index=index,
        composition_row=composition_row,
        composition_sha256=composition_sha256,
        parent_source=_exact_dict(base_row.get("parent_source"), "base parent source"),
    )
    _require(
        generic.get("mode") == "specialist"
        and generic.get("applicable_specialist_ids") == [TEMPORAL_MECHANISM_ID]
        and type(generic.get("fitted_typed_prompt")) is dict,
        f"latest-state specialist did not reach a terminal prompt at {ordinal}",
    )
    advisories = _terminal_advisories(generic)
    _require(
        len(advisories) == 1
        and advisories[0].get("mechanism_id") == TEMPORAL_MECHANISM_ID,
        f"latest-state temporal advisory changed at ordinal {ordinal}",
    )
    if ordinal == 79:
        bundle = _exact_dict(
            advisories[0].get("temporal_bundle"), "q79 temporal bundle"
        )
        v4_operator = _exact_dict(v4_q79.get("operator"), "q79 v4 operator")
        _require(
            bundle.get("winner_handle_id")
            == v4_operator.get("winner_handle_id")
            and v4_operator.get("global_exhaustiveness_claimed") is False,
            "q79 full-100 temporal winner diverged from sealed v4",
        )
    body = dict(generic)
    generic_receipt = require_sha256(
        body.pop("question_receipt_sha256", None), "generic temporal question"
    )
    body.update(
        {
            "base_v1_question_receipt_sha256": require_sha256(
                base_row.get("question_receipt_sha256"), "base v1 temporal question"
            ),
            "replacement_kind": "typed_latest_state_temporal_specialist",
            "replacement_source_question_receipt_sha256": generic_receipt,
            "route": _fixed_temporal_route(
                question_sha256=require_sha256(
                    composition_row.get("question_sha256"), "temporal question"
                ),
                legacy_route=_exact_dict(generic.get("route"), "legacy temporal route"),
            ),
        }
    )
    _require(
        body["terminal_prompt"]["full_chat_plus_output_tokens"]
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        f"latest-state terminal escaped 8k at ordinal {ordinal}",
    )
    assert_gold_blind(body, path=f"locked_specialist_v2_temporal_{ordinal}")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _q74_terminal_rebased_to_v1_parent(
    *,
    v4_row: Mapping[str, Any],
    parent_source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit the exact sealed q74 lane with the newer v1 parent fallback."""

    source_terminal = _exact_dict(
        v4_row.get("terminal_prompt"), "q74 v4 source terminal"
    )
    source_input = _exact_dict(
        source_terminal.get("provider_input"), "q74 v4 source provider input"
    )
    dated_question = require_text(
        source_input.get("dated_question"), "q74 dated question"
    )
    spec = compile_typed_operator_spec(dated_question)
    provenance = tuple(
        _exact_dict(value, "q74 v4 provenance")
        for value in _exact_list(v4_row.get("local_provenance"), "q74 provenance")
    )
    bindings = tuple(
        reduced_cli._rehydrate_handle_binding(  # noqa: SLF001
            _exact_dict(value.get("typed_binding"), "q74 typed binding")
        )
        for value in provenance
    )
    compact = _exact_dict(source_input.get("typed_evidence"), "q74 typed evidence")
    raw_items = tuple(
        _exact_dict(value, "q74 typed item")
        for value in _exact_list(compact.get("items"), "q74 typed items")
    )
    _require(
        tuple(binding.handle_id for binding in bindings) == ("H950001", "H950002")
        and tuple(item.get("handle_ids") for item in raw_items)
        == (["H950001"], ["H950002"]),
        "q74 sealed two-item lane changed before parent rebinding",
    )
    parsed = parse_typed_items(
        [v4._clean_compact_item(value) for value in raw_items],  # noqa: SLF001
        operator_spec=spec,
        bindings=bindings,
    )
    _require(
        not parsed.rejected_items and len(parsed.accepted_items) == 2,
        "q74 sealed lane failed typed parent rebinding",
    )
    contribution = TypedEvidenceContribution(
        residual.TYPED_ADAPTER_MECHANISM_ID,
        bindings,
        parsed,
        bindings[0].sealed_artifact_sha256,
        FrontierMode.BOUNDED,
        False,
    )
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(contribution.sealed_artifact_sha256,),
        frontier_mode=FrontierMode.BOUNDED,
        truncated=False,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )
    fitted = fit_typed_final_prompt(
        dated_question=dated_question,
        parent_prediction=require_text(parent_source.get("prediction"), "q74 v1 parent"),
        packet=packet,
        mechanism_by_handle={
            binding.handle_id: residual.TYPED_ADAPTER_MECHANISM_ID
            for binding in bindings
        },
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=tuple(
            item.receipt_sha256 for item in parsed.accepted_items
        ),
        protection_source_receipt_sha256=contribution.receipt_sha256,
    )
    advisories = _terminal_advisories(v4_row)
    terminal = v4._terminal_projection(  # noqa: SLF001
        fitted=fitted,
        specialist_advisories=advisories,
    )
    fitted_projection = fitted.projection()
    rebound_input = _exact_dict(
        terminal.get("provider_input"), "q74 rebound provider input"
    )
    rebound_parent = _exact_dict(
        rebound_input.get("protected_parent_fallback"), "q74 rebound parent"
    )
    _require(
        fitted.allowed_handle_ids == ("H950001", "H950002")
        and rebound_parent.get("prediction") == parent_source.get("prediction")
        and rebound_parent.get("prediction_sha256")
        == parent_source.get("prediction_sha256")
        and rebound_input.get("typed_evidence") == source_input.get("typed_evidence")
        and terminal.get("full_chat_plus_output_tokens")
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "q74 parent rebinding changed evidence or escaped 8k",
    )
    return fitted_projection, terminal


def _operator_replacement(
    *,
    ordinal: int,
    base_row: Mapping[str, Any],
    v4_row: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        ordinal in REPAIRED_OPERATOR_ORDINALS
        and base_row.get("mode") == "parent_passthrough"
        and base_row.get("question_id") == v4_row.get("question_id")
        and base_row.get("question_sha256") == v4_row.get("question_sha256")
        and base_row.get("dated_question_sha256")
        == v4_row.get("dated_question_sha256")
        and base_row.get("namespace_id") == v4_row.get("namespace_id"),
        f"sealed v4 operator question diverged from v1 at ordinal {ordinal}",
    )
    terminal = _exact_dict(v4_row.get("terminal_prompt"), "v4 terminal prompt")
    provider_input = _exact_dict(
        terminal.get("provider_input"), "v4 terminal provider input"
    )
    fallback = _exact_dict(
        provider_input.get("protected_parent_fallback"), "v4 protected parent"
    )
    parent_source = _exact_dict(base_row.get("parent_source"), "v1 parent source")
    parent_matches = (
        fallback.get("prediction") == parent_source.get("prediction")
        and fallback.get("prediction_sha256") == parent_source.get("prediction_sha256")
    )
    if parent_matches:
        fitted_prompt = dict(
            _exact_dict(v4_row.get("fitted_typed_prompt"), "v4 fitted prompt")
        )
        terminal_prompt = dict(terminal)
    else:
        _require(
            ordinal == 74,
            f"v4 operator escaped the sealed v1 parent at ordinal {ordinal}",
        )
        fitted_prompt, terminal_prompt = _q74_terminal_rebased_to_v1_parent(
            v4_row=v4_row,
            parent_source=parent_source,
        )
    _require(
        terminal_prompt.get("full_chat_plus_output_tokens")
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        f"v4 operator escaped 8k at ordinal {ordinal}",
    )
    body = {
        "applicable_specialist_ids": [],
        "base_v1_question_receipt_sha256": require_sha256(
            base_row.get("question_receipt_sha256"), "base v1 operator question"
        ),
        "dated_question_sha256": v4_row["dated_question_sha256"],
        "fitted_typed_prompt": fitted_prompt,
        "local_provenance": list(
            _exact_list(v4_row.get("local_provenance"), "v4 local provenance")
        ),
        "methods": list(_exact_list(v4_row.get("methods"), "v4 methods")),
        "mode": "repaired_operator",
        "namespace_id": v4_row["namespace_id"],
        "new_provider_calls": 0,
        "operator": dict(_exact_dict(v4_row.get("operator"), "v4 operator")),
        "ordinal": ordinal,
        "parent_source": dict(parent_source),
        "question_id": v4_row["question_id"],
        "question_sha256": v4_row["question_sha256"],
        "repaired_operator_ids": list(v4_row.get("mechanism_plan", ())),
        "replacement_kind": "sealed_missing4_v4_operator",
        "retained_transformer_token_state_bytes": 0,
        "route": _fixed_operator_route(v4_row),
        "selection": dict(_exact_dict(v4_row.get("selection"), "v4 selection")),
        "terminal_kind": require_text(v4_row.get("terminal_kind"), "terminal kind"),
        "terminal_prompt": terminal_prompt,
        "v4_terminal_rebased_to_v1_parent": not parent_matches,
        "v4_question_receipt_sha256": require_sha256(
            v4_row.get("question_receipt_sha256"), "v4 source question"
        ),
    }
    assert_gold_blind(body, path=f"locked_specialist_v2_operator_{ordinal}")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _replacement_scan_receipt(
    *,
    namespace_id: str,
    ordinals: Sequence[int],
    cache: Any,
    index: Any,
    database_sha256: str,
) -> dict[str, Any]:
    body = {
        "cache_receipt_sha256": cache.cache_receipt_sha256,
        "content_row_count": cache.content_row_count,
        "database_read_passes": 1,
        "format": SCAN_RECEIPT_FORMAT,
        "namespace_id": require_sha256(namespace_id, "scan namespace"),
        "ordinals": list(ordinals),
        "physical_content_token_count": index.physical_content_tokens_indexed,
        "physical_store_row_count": cache.physical_store_row_count,
        "source_database_sha256": require_sha256(database_sha256, "scan database"),
        "window_index_receipt_sha256": index.receipt_sha256,
    }
    return _with_receipt(body)


def build_construction(args: argparse.Namespace) -> dict[str, Any]:
    base_artifact, base_rows = _load_v1(
        Path(args.v1_construction),
        expected_sha256=str(args.expected_v1_construction_sha256),
    )
    v4_artifact, v4_rows = _load_v4(
        Path(args.v4_construction),
        expected_sha256=str(args.expected_v4_construction_sha256),
    )
    v4_by_ordinal = {int(row["ordinal"]): row for row in v4_rows}
    _require(
        tuple(sorted(v4_by_ordinal)) == v4.TARGET_ORDINALS,
        "sealed v4 source population changed",
    )
    (
        composition,
        closure,
        _run,
        _replay,
        composition_rows,
        _closure_rows,
        _run_rows,
        _judge_rows,
    ) = v1._load_parent_inputs(Path(args.parent_root))  # noqa: SLF001
    _require(
        base_artifact.payload.get("bindings")
        == {
            "parent_composition_artifact_sha256": composition.sha256,
            "parent_full_store_input_artifact_sha256": closure.sha256,
            "parent_replay_artifact_sha256": v1.EXPECTED_PARENT_REPLAY_SHA256,
            "parent_run_artifact_sha256": v1.EXPECTED_PARENT_RUN_SHA256,
        },
        "v1 construction escaped its official parent inputs",
    )

    context = typed_cli._guided_context(reduced_cli._guided_args(args))  # noqa: SLF001
    context_by_question = {
        row.source.packet.question_id: row for row in context.population.rows
    }
    namespace_objects = {
        row.namespace_id: row for row in context.population.namespaces
    }
    ordinals_by_namespace: dict[str, list[int]] = defaultdict(list)
    for ordinal in LATEST_STATE_ORDINALS:
        base_row = base_rows[ordinal]
        question_id = require_text(base_row.get("question_id"), "latest-state question")
        population_row = context_by_question.get(question_id)
        _require(
            population_row is not None
            and population_row.namespace.namespace_id == base_row.get("namespace_id")
            and population_row.source.packet.question_sha256
            == base_row.get("question_sha256"),
            f"latest-state row left its locked namespace at ordinal {ordinal}",
        )
        ordinals_by_namespace[population_row.namespace.namespace_id].append(ordinal)
    _require(
        len(ordinals_by_namespace) == EXPECTED_REPLACEMENT_NAMESPACE_COUNT,
        "latest-state replacement namespace population changed",
    )

    sealed_cache = v1._cache_receipts_by_namespace(closure)  # noqa: SLF001
    replacements: dict[int, dict[str, Any]] = {}
    scan_receipts: list[dict[str, Any]] = []
    for namespace_id in sorted(ordinals_by_namespace):
        namespace = namespace_objects[namespace_id]
        database_path = context.store_dirs_by_namespace[namespace_id] / "memory.db"
        with Database(database_path, read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=(
                    context.database_sha256_by_namespace[namespace_id]
                ),
                source_store_receipt_sha256=(
                    namespace.combined_store_receipt_sha256
                ),
            )
        index = build_full_store_window_index(cache)
        sealed = sealed_cache[namespace_id]
        _require(
            sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
            and sealed.get("window_index_receipt_sha256") == index.receipt_sha256
            and sealed.get("content_row_count") == cache.content_row_count
            and sealed.get("physical_store_row_count")
            == cache.physical_store_row_count,
            f"replacement index differs from sealed v1 closure: {namespace_id}",
        )
        namespace_ordinals = tuple(ordinals_by_namespace[namespace_id])
        for ordinal in namespace_ordinals:
            replacements[ordinal] = _temporal_replacement(
                ordinal=ordinal,
                index=index,
                composition_row=composition_rows[ordinal],
                composition_sha256=composition.sha256,
                base_row=base_rows[ordinal],
                v4_q79=v4_by_ordinal[79],
            )
        scan_receipts.append(
            _replacement_scan_receipt(
                namespace_id=namespace_id,
                ordinals=namespace_ordinals,
                cache=cache,
                index=index,
                database_sha256=context.database_sha256_by_namespace[namespace_id],
            )
        )
        del index, cache
        gc.collect()

    for ordinal in REPAIRED_OPERATOR_ORDINALS:
        replacements[ordinal] = _operator_replacement(
            ordinal=ordinal,
            base_row=base_rows[ordinal],
            v4_row=v4_by_ordinal[ordinal],
        )
    _require(
        tuple(sorted(replacements)) == REPLACED_ORDINALS,
        "replacement population lost or duplicated an ordinal",
    )
    questions = [
        replacements.get(ordinal, dict(base_rows[ordinal])) for ordinal in v1.ORDINALS
    ]
    _require(
        all(questions[ordinal] == base_rows[ordinal] for ordinal in PRESERVED_ORDINALS)
        and all(questions[ordinal] != base_rows[ordinal] for ordinal in REPLACED_ORDINALS),
        "v2 did not preserve exactly the non-target v1 rows",
    )

    specialist_count = sum(row.get("mode") == "specialist" for row in questions)
    repaired_count = sum(row.get("mode") == "repaired_operator" for row in questions)
    passthrough_count = sum(row.get("mode") == "parent_passthrough" for row in questions)
    provider_count = specialist_count + repaired_count
    terminal_tokens = [
        int(row["terminal_prompt"]["full_chat_plus_output_tokens"])
        for row in questions
        if row.get("terminal_prompt") is not None
    ]
    _require(
        specialist_count == EXPECTED_SPECIALIST_COUNT
        and repaired_count == EXPECTED_REPAIRED_OPERATOR_COUNT
        and passthrough_count == EXPECTED_PASSTHROUGH_COUNT
        and provider_count == EXPECTED_PROVIDER_PROMPT_COUNT
        and max(terminal_tokens) <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "full-100 v2 mode or prompt accounting changed",
    )
    structural_gate = _with_receipt(
        {
            "all_latest_state_rows_temporal": all(
                questions[ordinal].get("applicable_specialist_ids")
                == [TEMPORAL_MECHANISM_ID]
                for ordinal in LATEST_STATE_ORDINALS
            ),
            "all_terminal_prompts_within_cap": all(
                value <= HARD_COMPLETE_CHAT_TOKEN_CAP for value in terminal_tokens
            ),
            "byte_identical_v1_row_count": EXPECTED_PRESERVED_ROW_COUNT,
            "format": STRUCTURAL_GATE_FORMAT,
            "provider_prompt_count": provider_count,
            "repaired_operator_count": repaired_count,
            "replaced_row_count": EXPECTED_REPLACED_ROW_COUNT,
            "replacement_namespace_scan_count": len(scan_receipts),
            "structural_gate_passed": True,
        }
    )
    base_lifecycle = _exact_dict(
        base_artifact.payload.get("resident_index_lifecycle"), "v1 lifecycle"
    )
    payload: dict[str, Any] = {
        "base_v1_row_population_sha256": identity_sha256(list(base_rows)),
        "bindings": {
            "base_v1_construction_artifact_sha256": base_artifact.sha256,
            "missing4_v4_construction_artifact_sha256": v4_artifact.sha256,
            "parent_composition_artifact_sha256": composition.sha256,
            "parent_full_store_input_artifact_sha256": closure.sha256,
            "parent_replay_artifact_sha256": v1.EXPECTED_PARENT_REPLAY_SHA256,
            "parent_run_artifact_sha256": v1.EXPECTED_PARENT_RUN_SHA256,
        },
        "byte_identical_v1_row_count": EXPECTED_PRESERVED_ROW_COUNT,
        "construction_is_posthoc_outcome_conditioned": False,
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_terminal_complete_envelope_tokens": max(terminal_tokens),
        "new_provider_calls": 0,
        "ordinals": list(v1.ORDINALS),
        "parent_passthrough_count": passthrough_count,
        "preserved_ordinals": list(PRESERVED_ORDINALS),
        "provider_prompt_count": provider_count,
        "question_count": len(questions),
        "questions": questions,
        "repaired_operator_provider_prompt_count": repaired_count,
        "replaced_ordinals": list(REPLACED_ORDINALS),
        "replacement_resident_index_lifecycle": {
            "base_v1_lifecycle_sha256": identity_sha256(base_lifecycle),
            "database_read_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "receipts": scan_receipts,
            "total_database_read_passes": len(scan_receipts),
            "unique_namespace_count": len(scan_receipts),
        },
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "specialist_provider_prompt_count": specialist_count,
        "structural_gate": structural_gate,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="locked_specialist_final_v2_construction")
    return {**payload, "construction_identity_sha256": identity_sha256(payload)}


def validate_construction(
    artifact: SealedArtifact,
    *,
    base_v1: tuple[SealedArtifact, tuple[dict[str, Any], ...]] | None = None,
    source_v4: tuple[SealedArtifact, tuple[dict[str, Any], ...]] | None = None,
) -> tuple[dict[str, Any], ...]:
    base_artifact, base_rows = base_v1 or _load_v1(DEFAULT_V1_CONSTRUCTION)
    v4_artifact, v4_rows = source_v4 or _load_v4(DEFAULT_V4_CONSTRUCTION)
    v4_by_ordinal = {int(row["ordinal"]): row for row in v4_rows}
    payload = artifact.payload
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("construction_identity_sha256", None), "v2 construction identity"
    )
    _require(identity_sha256(unsigned) == declared, "v2 construction identity changed")
    rows = tuple(
        _exact_dict(value, "v2 construction row")
        for value in _exact_list(payload.get("questions"), "v2 questions")
    )
    bindings = _exact_dict(payload.get("bindings"), "v2 bindings")
    lifecycle = _exact_dict(
        payload.get("replacement_resident_index_lifecycle"), "v2 lifecycle"
    )
    scan_receipts = tuple(
        _validate_receipt(value, label="v2 replacement scan")
        for value in _exact_list(lifecycle.get("receipts"), "v2 scan receipts")
    )
    gate = _validate_receipt(payload.get("structural_gate"), label="v2 structural gate")
    _require(
        payload.get("format") == CONSTRUCTION_FORMAT
        and payload.get("construction_is_posthoc_outcome_conditioned") is False
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("selection_and_routing_frozen_before_target_plan_load") is True
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and tuple(payload.get("ordinals", ())) == v1.ORDINALS
        and tuple(payload.get("preserved_ordinals", ())) == PRESERVED_ORDINALS
        and tuple(payload.get("replaced_ordinals", ())) == REPLACED_ORDINALS
        and payload.get("question_count") == len(rows) == 100
        and payload.get("base_v1_row_population_sha256")
        == identity_sha256(list(base_rows))
        and bindings
        == {
            "base_v1_construction_artifact_sha256": base_artifact.sha256,
            "missing4_v4_construction_artifact_sha256": v4_artifact.sha256,
            "parent_composition_artifact_sha256": v1.EXPECTED_PARENT_COMPOSITION_SHA256,
            "parent_full_store_input_artifact_sha256": v1.EXPECTED_PARENT_CLOSURE_SHA256,
            "parent_replay_artifact_sha256": v1.EXPECTED_PARENT_REPLAY_SHA256,
            "parent_run_artifact_sha256": v1.EXPECTED_PARENT_RUN_SHA256,
        }
        and lifecycle.get("database_read_passes_per_used_namespace") == 1
        and lifecycle.get("maximum_simultaneous_namespace_indexes") == 1
        and lifecycle.get("total_database_read_passes")
        == lifecycle.get("unique_namespace_count")
        == len(scan_receipts)
        == EXPECTED_REPLACEMENT_NAMESPACE_COUNT
        and len({row.get("namespace_id") for row in scan_receipts})
        == EXPECTED_REPLACEMENT_NAMESPACE_COUNT,
        "full-100 v2 construction boundary changed",
    )

    specialist_count = 0
    repaired_count = 0
    passthrough_count = 0
    terminal_tokens: list[int] = []
    for ordinal, row in enumerate(rows):
        if ordinal in PRESERVED_ORDINALS:
            _require(
                row == base_rows[ordinal],
                f"preserved v1 row changed at ordinal {ordinal}",
            )
        else:
            row_body = dict(row)
            row_receipt = require_sha256(
                row_body.pop("question_receipt_sha256", None), "v2 replacement row"
            )
            _require(
                identity_sha256(row_body) == row_receipt
                and row.get("ordinal") == ordinal
                and row.get("base_v1_question_receipt_sha256")
                == base_rows[ordinal].get("question_receipt_sha256")
                and row.get("parent_source") == base_rows[ordinal].get("parent_source")
                and row.get("new_provider_calls") == 0
                and row.get("retained_transformer_token_state_bytes") == 0,
                f"replacement row provenance changed at ordinal {ordinal}",
            )
            if ordinal in LATEST_STATE_ORDINALS:
                _require(
                    row.get("mode") == "specialist"
                    and row.get("replacement_kind")
                    == "typed_latest_state_temporal_specialist"
                    and row.get("applicable_specialist_ids")
                    == [TEMPORAL_MECHANISM_ID],
                    f"latest-state replacement mode changed at ordinal {ordinal}",
                )
                route = _validate_receipt(
                    row.get("route"), label=f"temporal route {ordinal}"
                )
                _require(
                    route.get("format") == TEMPORAL_ROUTE_FORMAT
                    and route.get("temporal_mode") == TemporalMode.LATEST_STATE.value
                    and route.get("route_basis") == "typed_operator_spec.temporal_mode",
                    f"typed latest-state route changed at ordinal {ordinal}",
                )
                try:
                    locked_answer._source_plan(row, ordinal)  # noqa: SLF001
                except MatchedEvalContractError as exc:
                    raise LockedSpecialistFinalConstructionV2Error(
                        f"temporal answer seam changed at ordinal {ordinal}: {exc}"
                    ) from exc
                advisories = _terminal_advisories(row)
                _require(
                    len(advisories) == 1
                    and advisories[0].get("mechanism_id")
                    == TEMPORAL_MECHANISM_ID,
                    f"temporal advisory changed at ordinal {ordinal}",
                )
            else:
                source = v4_by_ordinal[ordinal]
                _require(
                    row.get("mode") == "repaired_operator"
                    and row.get("replacement_kind")
                    == "sealed_missing4_v4_operator"
                    and row.get("v4_question_receipt_sha256")
                    == source.get("question_receipt_sha256")
                    and row.get("operator") == source.get("operator")
                    and row.get("selection") == source.get("selection")
                    and row.get("local_provenance")
                    == source.get("local_provenance")
                    and row.get("methods") == source.get("methods"),
                    f"sealed v4 operator plane changed at ordinal {ordinal}",
                )
                if ordinal == 74:
                    rebound_terminal = _exact_dict(
                        row.get("terminal_prompt"), "q74 rebound terminal"
                    )
                    rebound_input = _exact_dict(
                        rebound_terminal.get("provider_input"),
                        "q74 rebound provider input",
                    )
                    source_input = _exact_dict(
                        _exact_dict(
                            source.get("terminal_prompt"), "q74 source terminal"
                        ).get("provider_input"),
                        "q74 source provider input",
                    )
                    rebound_parent = _exact_dict(
                        rebound_input.get("protected_parent_fallback"),
                        "q74 rebound parent",
                    )
                    parent_source = _exact_dict(
                        row.get("parent_source"), "q74 v1 parent source"
                    )
                    _require(
                        row.get("v4_terminal_rebased_to_v1_parent") is True
                        and rebound_input.get("typed_evidence")
                        == source_input.get("typed_evidence")
                        and rebound_input.get("specialist_advisories")
                        == source_input.get("specialist_advisories")
                        and rebound_parent.get("prediction")
                        == parent_source.get("prediction")
                        and rebound_parent.get("prediction_sha256")
                        == parent_source.get("prediction_sha256")
                        and row.get("fitted_typed_prompt", {}).get(
                            "allowed_handle_ids"
                        )
                        == ["H950001", "H950002"],
                        "q74 v1-parent rebinding changed its sealed v4 evidence",
                    )
                else:
                    _require(
                        row.get("v4_terminal_rebased_to_v1_parent") is False
                        and row.get("fitted_typed_prompt")
                        == source.get("fitted_typed_prompt")
                        and row.get("terminal_prompt")
                        == source.get("terminal_prompt"),
                        f"sealed v4 prompt plane changed at ordinal {ordinal}",
                    )
                answer_v2._prompt_plan_row(row, ordinal)  # noqa: SLF001

        mode = row.get("mode")
        specialist_count += mode == "specialist"
        repaired_count += mode == "repaired_operator"
        passthrough_count += mode == "parent_passthrough"
        terminal = row.get("terminal_prompt")
        if type(terminal) is dict:
            tokens = terminal.get("full_chat_plus_output_tokens")
            _require(
                type(tokens) is int and tokens <= HARD_COMPLETE_CHAT_TOKEN_CAP,
                f"terminal prompt cap changed at ordinal {ordinal}",
            )
            terminal_tokens.append(tokens)

    _require(
        payload.get("byte_identical_v1_row_count")
        == gate.get("byte_identical_v1_row_count")
        == EXPECTED_PRESERVED_ROW_COUNT
        and gate.get("replaced_row_count") == EXPECTED_REPLACED_ROW_COUNT
        and gate.get("replacement_namespace_scan_count")
        == EXPECTED_REPLACEMENT_NAMESPACE_COUNT
        and gate.get("structural_gate_passed") is True
        and specialist_count
        == payload.get("specialist_provider_prompt_count")
        == EXPECTED_SPECIALIST_COUNT
        and repaired_count
        == payload.get("repaired_operator_provider_prompt_count")
        == EXPECTED_REPAIRED_OPERATOR_COUNT
        and passthrough_count
        == payload.get("parent_passthrough_count")
        == EXPECTED_PASSTHROUGH_COUNT
        and specialist_count + repaired_count
        == payload.get("provider_prompt_count")
        == gate.get("provider_prompt_count")
        == EXPECTED_PROVIDER_PROMPT_COUNT
        and payload.get("max_terminal_complete_envelope_tokens")
        == max(terminal_tokens),
        "full-100 v2 structural accounting changed",
    )
    assert_gold_blind(payload, path="validated_locked_specialist_final_v2")
    return rows


def load_verified_construction(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    source = Path(path)
    if source.name != CONSTRUCTION_NAME:
        source = source / CONSTRUCTION_NAME
    artifact = read_sealed_json(source)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "full-100 v2 construction"),
        "full-100 v2 construction artifact changed",
    )
    return artifact, validate_construction(artifact)


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_construction(args)
    target = Path(args.output_root) / CONSTRUCTION_NAME
    candidate = SealedArtifact(
        path=target,
        sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload=payload,
    )
    validate_construction(candidate)
    artifact, created = publish_sealed_json(target, payload)
    return {
        "byte_identical_v1_row_count": payload["byte_identical_v1_row_count"],
        "construction_sha256": artifact.sha256,
        "created": created,
        "max_terminal_complete_envelope_tokens": payload[
            "max_terminal_complete_envelope_tokens"
        ],
        "new_provider_calls": 0,
        "parent_passthrough_count": payload["parent_passthrough_count"],
        "provider_prompt_count": payload["provider_prompt_count"],
        "question_count": payload["question_count"],
        "repaired_operator_provider_prompt_count": payload[
            "repaired_operator_provider_prompt_count"
        ],
        "replacement_namespace_scan_count": payload[
            "replacement_resident_index_lifecycle"
        ]["unique_namespace_count"],
        "retained_transformer_token_state_bytes": 0,
        "specialist_provider_prompt_count": payload[
            "specialist_provider_prompt_count"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-construction", type=Path, default=DEFAULT_V1_CONSTRUCTION)
    parser.add_argument(
        "--expected-v1-construction-sha256",
        default=EXPECTED_V1_CONSTRUCTION_SHA256,
    )
    parser.add_argument("--v4-construction", type=Path, default=DEFAULT_V4_CONSTRUCTION)
    parser.add_argument(
        "--expected-v4-construction-sha256",
        default=EXPECTED_V4_CONSTRUCTION_SHA256,
    )
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    specialist._add_store_args(parser)  # noqa: SLF001
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_construct(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_V1_CONSTRUCTION_SHA256",
    "EXPECTED_V4_CONSTRUCTION_SHA256",
    "LATEST_STATE_ORDINALS",
    "PRESERVED_ORDINALS",
    "REPAIRED_OPERATOR_ORDINALS",
    "REPLACED_ORDINALS",
    "LockedSpecialistFinalConstructionV2Error",
    "build_construction",
    "build_parser",
    "load_verified_construction",
    "main",
    "run_construct",
    "validate_construction",
]
