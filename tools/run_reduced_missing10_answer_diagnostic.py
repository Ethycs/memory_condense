#!/usr/bin/env python3
"""Run the sealed delta-only answer diagnostic for the exact missing ten.

The construction phase reads two frozen artifacts only: the v3 reduced
second-read construction and the existing typed-parent composition.  It does
not reopen any 1M-token corpus, retrieve, call a provider, or load benchmark
references.  Each prompt contains the dated question, a protected parent
prediction, and the already-selected raw evidence from four fact treatments.
The much larger parent typed-evidence payload is deliberately excluded so this
diagnostic isolates treatment quality from prompt packing pressure.

Lifecycle::

    construct -> preflight -> provider-run -> materialize -> replay

Provider execution is checkpointed and requires exact authorization for ten
physical calls.  Materialization and replay are checkpoint-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-reduced-missing10-delta-answer-diagnostic-v1"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
RUN_FORMAT = "memory-condense-reduced-missing10-answer-diagnostic-run-v1"

CONSTRUCTION_NAME = "reduced-missing10-delta-construction-v1.json"
PREFLIGHT_NAME = "reduced-missing10-delta-preflight-v1.json"
RUN_NAME = "reduced-missing10-answer-run-v1.json"
REPLAY_NAME = "reduced-missing10-answer-run-replay-v1.json"
CHECKPOINT_DIR_NAME = "reduced-missing10-delta-checkpoints-v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REDUCED_CONSTRUCTION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-second-read-missing10-v3/"
    "reduced-second-read-construction-v3.json"
)
DEFAULT_PARENT_COMPOSITION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/typed-memory-final-v3-shared-surplus/"
    "typed-memory-final-composition-v1.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-missing10-delta-answer-v1"
)

EXPECTED_REDUCED_CONSTRUCTION_SHA256 = (
    "c1f3aeae910c072196e5d9550e5ddd723cb9df14fd79e9c4e0420dd611e013db"
)
EXPECTED_PARENT_COMPOSITION_SHA256 = (
    "730a437e242174d188ae67484d9414d87c74d8ed926d9e4cdc726c7d5260317f"
)
EXPECTED_ORDINALS = (7, 31, 36, 43, 61, 72, 77, 81, 86, 93)
FACT_METHOD_IDS = (
    "fact_derived_second_read",
    "fact_coverage_callback_second_read",
    "fact_provenance_reinjected_second_read",
    "fact_coverage_provenance_second_read",
)

HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_COMPLETE_CHAT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE
PROTECTED_FACT_LANE_TOKEN_CAP = 4_096
EXPECTED_PROVIDER_CALLS = 10

SYSTEM_PROMPT = (
    "Answer the user's dated memory question from the supplied evidence. "
    "The protected parent prediction is a fallback, not evidence. The four "
    "fact lanes were selected independently and may overlap or conflict. "
    "Prefer direct, dated, user-authored evidence and reconcile all relevant "
    "facts. Return exactly one JSON object and no markdown: "
    '{"decision":"keep_parent|replace","prediction":"nonempty answer",'
    '"used_evidence_ids":["F1-001"]}.'
)


class ReducedMissing10DiagnosticError(MatchedEvalContractError):
    """Raised when a frozen input, prompt, checkpoint, or seal diverges."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedMissing10DiagnosticError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _plain_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    rows = tuple(dict(row) for row in messages)
    _require(
        bool(rows)
        and all(
            set(row) == {"role", "content"}
            and row["role"] in {"system", "user", "assistant"}
            and type(row["content"]) is str
            for row in rows
        ),
        "prompt messages changed schema",
    )
    return rows


def _read_frozen(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, label),
        f"{label} artifact changed",
    )
    return artifact


def _validate_reduced_source(artifact: SealedArtifact) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    rows = payload.get("questions")
    _require(
        payload.get("format")
        == "memory-condense-reduced-second-read-retrieval-assay-v3-construction"
        and payload.get("construction_identity_sha256")
        == "a58fbb31b08d7255b54a4dd48952e3039bc65d9de48af647955303a876c3f623"
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and tuple(payload.get("ordinals", ())) == EXPECTED_ORDINALS
        and payload.get("question_count") == EXPECTED_PROVIDER_CALLS
        and type(rows) is list
        and len(rows) == EXPECTED_PROVIDER_CALLS,
        "reduced v3 construction boundary changed",
    )
    result: list[dict[str, Any]] = []
    for expected_ordinal, raw in zip(EXPECTED_ORDINALS, rows, strict=True):
        _require(
            type(raw) is dict and raw.get("ordinal") == expected_ordinal,
            "reduced v3 question order changed",
        )
        methods = raw.get("methods")
        _require(type(methods) is list, "reduced v3 methods changed type")
        by_id = {
            row.get("method_id"): row
            for row in methods
            if type(row) is dict and type(row.get("method_id")) is str
        }
        _require(
            all(method_id in by_id for method_id in FACT_METHOD_IDS),
            "reduced v3 lost a fact treatment",
        )
        detached = dict(raw)
        detached["_fact_methods"] = [by_id[value] for value in FACT_METHOD_IDS]
        result.append(detached)
    return tuple(result)


def _validate_parent_source(artifact: SealedArtifact) -> dict[int, dict[str, Any]]:
    payload = artifact.payload
    rows = payload.get("questions")
    _require(
        payload.get("format")
        == "memory-condense-typed-memory-final-arm-v1-composition-v1"
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and type(rows) is list
        and len(rows) == 100,
        "typed parent composition boundary changed",
    )
    result: dict[int, dict[str, Any]] = {}
    for raw in rows:
        _require(type(raw) is dict, "typed parent row changed type")
        ordinal = raw.get("ordinal")
        _require(
            type(ordinal) is int and ordinal not in result,
            "typed parent ordinals repeat or changed type",
        )
        result[ordinal] = dict(raw)
    _require(
        all(value in result for value in EXPECTED_ORDINALS),
        "typed parent lost an exact-ten row",
    )
    return result


def _span_identity(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        row["namespace_id"],
        row["chunk_id"],
        row["turn_id"],
        row["span_start_char"],
        row["span_end_char"],
        row["quote_sha256"],
        row["quote"],
    )


def _validate_observation(
    raw: object,
    *,
    namespace_id: str,
    method_id: str,
) -> dict[str, Any]:
    _require(type(raw) is dict, f"{method_id} selected observation changed type")
    assert type(raw) is dict
    required_text = (
        "candidate_id",
        "chunk_id",
        "created_at",
        "namespace_id",
        "observation_sha256",
        "quote",
        "quote_sha256",
        "role",
        "source_id",
        "turn_id",
    )
    _require(
        all(type(raw.get(key)) is str and bool(raw[key]) for key in required_text)
        and raw.get("namespace_id") == namespace_id
        and raw.get("role") in {"user", "assistant"}
        and type(raw.get("discovery_rank")) is int
        and int(raw["discovery_rank"]) >= 0
        and type(raw.get("span_start_char")) is int
        and type(raw.get("span_end_char")) is int
        and 0 <= int(raw["span_start_char"]) < int(raw["span_end_char"])
        and raw.get("quote_sha256") == quote_sha256(str(raw["quote"]))
        and raw.get("token_count") == count_tokens(str(raw["quote"])),
        f"{method_id} selected observation changed exact content/provenance",
    )
    for key in ("candidate_id", "chunk_id", "namespace_id", "observation_sha256", "quote_sha256"):
        require_sha256(str(raw[key]), f"{method_id} {key}")
    return dict(raw)


def _provider_evidence(row: Mapping[str, Any], evidence_id: str) -> dict[str, Any]:
    return {
        "created_at": row["created_at"],
        "evidence_id": evidence_id,
        "quote": row["quote"],
        "role": row["role"],
    }


def _lane_content_tokens(rows: Sequence[Mapping[str, Any]]) -> int:
    return count_tokens(_canonical_json(list(rows)))


def _render_messages(provider_input: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    assert_gold_blind(provider_input, path="reduced_missing10_provider_input")
    return _plain_messages(
        (
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _canonical_json(dict(provider_input))},
        )
    )


def _provider_input(
    *,
    dated_question: str,
    parent_prediction: str,
    lanes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = {
        "dated_question": dated_question,
        "fact_lanes": [dict(row) for row in lanes],
        "format": f"{FORMAT}-provider-input",
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent_prediction,
            "prediction_sha256": quote_sha256(parent_prediction),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_evidence_ids": ["F1-001"],
        },
    }
    assert_gold_blind(value, path="reduced_missing10_provider_input")
    return value


def _build_question_row(
    reduced: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = reduced["ordinal"]
    provider_projection = parent.get("provider_projection")
    _require(type(provider_projection) is dict, "parent provider projection changed")
    assert type(provider_projection) is dict
    original_input = provider_projection.get("provider_input")
    _require(type(original_input) is dict, "parent provider input changed")
    assert type(original_input) is dict
    typed_parent = original_input.get("typed_evidence")
    dated_question = original_input.get("dated_question")
    parent_prediction = parent.get("parent_prediction")
    _require(
        type(typed_parent) is dict
        and type(dated_question) is str
        and bool(dated_question)
        and type(parent_prediction) is str
        and bool(parent_prediction)
        and parent.get("parent_prediction_sha256") == quote_sha256(parent_prediction)
        and reduced.get("question_id") == parent.get("question_id")
        and reduced.get("question_sha256") == parent.get("question_sha256")
        and reduced.get("dated_question_sha256")
        == parent.get("dated_question_sha256")
        == quote_sha256(dated_question),
        f"reduced/parent identity changed at ordinal {ordinal}",
    )

    # Read and validate every method selection before deduplicating any row.
    method_inputs: list[tuple[str, str, list[dict[str, Any]]]] = []
    all_input_receipts: list[str] = []
    for raw_method in reduced["_fact_methods"]:
        method_id = raw_method.get("method_id")
        _require(method_id in FACT_METHOD_IDS, "fact method order changed")
        method_receipt = require_sha256(
            raw_method.get("method_receipt_sha256"), f"{method_id} receipt"
        )
        selected = raw_method.get("callback_selected_candidates")
        _require(
            type(selected) is list
            and raw_method.get("callback_selected_candidate_count") == len(selected),
            f"{method_id} callback-selected list/count changed",
        )
        observations = [
            _validate_observation(
                row,
                namespace_id=require_sha256(
                    reduced.get("namespace_id"), "reduced namespace"
                ),
                method_id=str(method_id),
            )
            for row in selected
        ]
        receipts = [row["observation_sha256"] for row in observations]
        _require(
            len(receipts) == len(set(receipts))
            and raw_method.get("callback_selected_candidate_tokens")
            == sum(row["token_count"] for row in observations),
            f"{method_id} repeated or changed a callback-selected observation",
        )
        all_input_receipts.extend(receipts)
        method_inputs.append((str(method_id), method_receipt, observations))

    owner_by_span: dict[tuple[object, ...], tuple[str, str]] = {}
    deduped: dict[str, list[dict[str, Any]]] = {value: [] for value in FACT_METHOD_IDS}
    exclusions: dict[str, list[dict[str, Any]]] = {value: [] for value in FACT_METHOD_IDS}
    for method_id, _receipt, observations in method_inputs:
        for row in observations:
            key = _span_identity(row)
            prior = owner_by_span.get(key)
            span_receipt = identity_sha256(
                {"format": f"{FORMAT}-exact-span", "parts": list(key[:-1])}
            )
            if prior is not None:
                exclusions[method_id].append(
                    {
                        "duplicate_of_observation_sha256": prior[1],
                        "exact_span_identity_sha256": span_receipt,
                        "observation_sha256": row["observation_sha256"],
                        "owner_method_id": prior[0],
                        "reason": "postselection_exact_span_and_quote_duplicate",
                    }
                )
                continue
            owner_by_span[key] = (method_id, row["observation_sha256"])
            deduped[method_id].append(row)
    _require(
        sum(len(rows) for rows in deduped.values())
        + sum(len(rows) for rows in exclusions.values())
        == len(all_input_receipts),
        "postselection dedup partition changed",
    )

    lane_rows: dict[str, list[dict[str, Any]]] = {
        method_id: [] for method_id in FACT_METHOD_IDS
    }
    evidence_source: dict[str, dict[str, Any]] = {}
    protected_ids: dict[str, list[str]] = {value: [] for value in FACT_METHOD_IDS}
    surplus_ids: dict[str, list[str]] = {value: [] for value in FACT_METHOD_IDS}
    budget_omitted: dict[str, list[dict[str, Any]]] = {
        value: [] for value in FACT_METHOD_IDS
    }
    pending: list[tuple[int, int, dict[str, Any], str]] = []
    for method_index, method_id in enumerate(FACT_METHOD_IDS, start=1):
        for row_index, source in enumerate(deduped[method_id], start=1):
            evidence_id = f"F{method_index}-{row_index:03d}"
            rendered = _provider_evidence(source, evidence_id)
            candidate = lane_rows[method_id] + [rendered]
            if _lane_content_tokens(candidate) <= PROTECTED_FACT_LANE_TOKEN_CAP:
                lane_rows[method_id].append(rendered)
                protected_ids[method_id].append(evidence_id)
                evidence_source[evidence_id] = source
            else:
                pending.append(
                    (int(source["discovery_rank"]), method_index, source, evidence_id)
                )

    method_receipts = {method_id: receipt for method_id, receipt, _ in method_inputs}

    def lanes_projection() -> list[dict[str, Any]]:
        return [
            {
                "evidence": deepcopy(lane_rows[method_id]),
                "method_id": method_id,
            }
            for method_id in FACT_METHOD_IDS
        ]

    base_input = _provider_input(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        lanes=lanes_projection(),
    )
    base_messages = _render_messages(base_input)
    _require(
        count_chat_prompt_token_proxy(base_messages) <= MAX_CHAT_PROMPT_TOKENS,
        "four protected fact-lane allocations do not fit the hard prompt cap",
    )

    # Shared surplus begins only after all four independent protected budgets
    # have been allocated.  Original discovery rank provides a gold-blind,
    # deterministic global fill order.
    for _rank, _method_index, source, evidence_id in sorted(
        pending,
        key=lambda value: (
            value[0],
            value[1],
            value[2]["observation_sha256"],
        ),
    ):
        method_id = FACT_METHOD_IDS[_method_index - 1]
        rendered = _provider_evidence(source, evidence_id)
        lane_rows[method_id].append(rendered)
        candidate_input = _provider_input(
            dated_question=dated_question,
            parent_prediction=parent_prediction,
            lanes=lanes_projection(),
        )
        candidate_messages = _render_messages(candidate_input)
        if count_chat_prompt_token_proxy(candidate_messages) <= MAX_CHAT_PROMPT_TOKENS:
            surplus_ids[method_id].append(evidence_id)
            evidence_source[evidence_id] = source
        else:
            lane_rows[method_id].pop()
            budget_omitted[method_id].append(source)

    provider_input = _provider_input(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        lanes=lanes_projection(),
    )
    messages = _render_messages(provider_input)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "delta diagnostic escaped the complete-chat hard cap",
    )
    allowed_ids = tuple(
        row["evidence_id"]
        for method_id in FACT_METHOD_IDS
        for row in lane_rows[method_id]
    )
    lane_ledgers: list[dict[str, Any]] = []
    for method_id, method_receipt, source_rows in method_inputs:
        retained_ids = tuple(
            row["evidence_id"] for row in lane_rows[method_id]
        )
        retained_observations = tuple(
            evidence_source[value]["observation_sha256"] for value in retained_ids
        )
        omitted = tuple(
            row["observation_sha256"] for row in budget_omitted[method_id]
        )
        body = {
            "dedup_exclusions": exclusions[method_id],
            "final_content_token_proxy": _lane_content_tokens(lane_rows[method_id]),
            "input_selected_count": len(source_rows),
            "input_selected_observation_sha256s": [
                row["observation_sha256"] for row in source_rows
            ],
            "method_id": method_id,
            "method_receipt_sha256": method_receipt,
            "omitted_observation_sha256s": list(omitted),
            "post_dedup_count": len(deduped[method_id]),
            "protected_content_token_cap": PROTECTED_FACT_LANE_TOKEN_CAP,
            "protected_content_token_proxy": _lane_content_tokens(
                [
                    row
                    for row in lane_rows[method_id]
                    if row["evidence_id"] in set(protected_ids[method_id])
                ]
            ),
            "protected_evidence_ids": protected_ids[method_id],
            "retained_evidence_ids": list(retained_ids),
            "retained_observation_sha256s": list(retained_observations),
            "shared_surplus_evidence_ids": surplus_ids[method_id],
        }
        lane_ledgers.append(
            {**body, "lane_receipt_sha256": identity_sha256(body)}
        )

    comparison = {
        "delta_only": True,
        "parent_messages_sha256": provider_projection.get("messages_sha256"),
        "parent_prompt_token_proxy": provider_projection.get("prompt_token_proxy"),
        "parent_typed_evidence_included": False,
        "parent_typed_evidence_sha256": identity_sha256(typed_parent),
    }
    _require(
        type(comparison["parent_prompt_token_proxy"]) is int
        and comparison["parent_prompt_token_proxy"] > 0,
        "parent prompt token comparison changed",
    )
    require_sha256(str(comparison["parent_messages_sha256"]), "parent messages")

    body = {
        "allowed_evidence_ids": list(allowed_ids),
        "comparison_input": comparison,
        "dated_question_sha256": parent["dated_question_sha256"],
        "full_chat_plus_output_tokens": prompt_tokens + OUTPUT_TOKEN_RESERVE,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "lane_ledgers": lane_ledgers,
        "messages": list(messages),
        "messages_sha256": identity_sha256(list(messages)),
        "ordinal": ordinal,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_prediction": parent_prediction,
        "parent_prediction_sha256": parent["parent_prediction_sha256"],
        "postselection_dedup": True,
        "prompt_token_proxy": prompt_tokens,
        "provider_input": provider_input,
        "question_id": parent["question_id"],
        "question_sha256": parent["question_sha256"],
    }
    assert_gold_blind(body, path=f"reduced_missing10_question_{ordinal}")
    return {**body, "prompt_row_receipt_sha256": identity_sha256(body)}


def build_construction_payload(
    reduced: SealedArtifact,
    parent: SealedArtifact,
) -> dict[str, Any]:
    reduced_rows = _validate_reduced_source(reduced)
    parents = _validate_parent_source(parent)
    rows = [
        _build_question_row(row, parents[int(row["ordinal"])])
        for row in reduced_rows
    ]
    _require(
        tuple(row["ordinal"] for row in rows) == EXPECTED_ORDINALS
        and len({row["messages_sha256"] for row in rows}) == EXPECTED_PROVIDER_CALLS,
        "constructed exact-ten prompt population changed",
    )
    payload = {
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "maximum_full_chat_plus_output_tokens": max(
            row["full_chat_plus_output_tokens"] for row in rows
        ),
        "method_ids": list(FACT_METHOD_IDS),
        "new_provider_calls": 0,
        "ordinals": list(EXPECTED_ORDINALS),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_composition_artifact_sha256": parent.sha256,
        "postselection_dedup": True,
        "provider_calls": 0,
        "provider_ready": True,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "questions": rows,
        "reduced_v3_construction_artifact_sha256": reduced.sha256,
        "retained_transformer_token_state_bytes": 0,
        "stage": "callback_selected_union_delta_only",
        "treatment": "delta_only_fact_lanes_with_protected_parent_fallback",
    }
    assert_gold_blind(payload, path="reduced_missing10_construction")
    return payload


def _construct(args: argparse.Namespace) -> dict[str, Any]:
    reduced = _read_frozen(
        Path(args.reduced_construction),
        str(args.expected_reduced_construction_sha256),
        label="reduced v3 construction",
    )
    parent = _read_frozen(
        Path(args.parent_composition),
        str(args.expected_parent_composition_sha256),
        label="typed parent composition",
    )
    payload = build_construction_payload(reduced, parent)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME,
        payload,
    )
    return {
        "artifact": artifact.path.as_posix(),
        "construction_sha256": artifact.sha256,
        "created": created,
        "gold_loaded": False,
        "maximum_full_chat_plus_output_tokens": payload[
            "maximum_full_chat_plus_output_tokens"
        ],
        "physical_provider_calls": 0,
        "provider_ready": True,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "retained_transformer_token_state_bytes": 0,
    }


def _validate_construction(artifact: SealedArtifact) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    rows = payload.get("questions")
    _require(
        payload.get("format") == CONSTRUCTION_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("new_provider_calls") == 0
        and payload.get("provider_ready") is True
        and payload.get("retained_transformer_token_state_bytes") == 0
        and tuple(payload.get("ordinals", ())) == EXPECTED_ORDINALS
        and payload.get("question_count") == EXPECTED_PROVIDER_CALLS
        and type(rows) is list
        and len(rows) == EXPECTED_PROVIDER_CALLS,
        "sealed delta construction changed",
    )
    validated: list[dict[str, Any]] = []
    for ordinal, raw in zip(EXPECTED_ORDINALS, rows, strict=True):
        _require(type(raw) is dict, "delta construction row changed type")
        assert type(raw) is dict
        body = dict(raw)
        declared = body.pop("prompt_row_receipt_sha256", None)
        messages = raw.get("messages")
        _require(
            raw.get("ordinal") == ordinal
            and declared == identity_sha256(body)
            and type(messages) is list,
            "delta construction row seal/order changed",
        )
        plain = _plain_messages(messages)
        prompt_tokens = count_chat_prompt_token_proxy(plain)
        _require(
            identity_sha256(list(plain)) == raw.get("messages_sha256")
            and prompt_tokens == raw.get("prompt_token_proxy")
            and prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_COMPLETE_CHAT_TOKEN_CAP,
            "delta construction prompt bytes/budget changed",
        )
        assert_gold_blind(raw, path=f"delta_construction_row_{ordinal}")
        validated.append(dict(raw))
    return tuple(validated)


def _read_construction(output_root: Path, expected_sha256: str) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(output_root / CONSTRUCTION_NAME)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "expected delta construction"),
        "sealed delta construction digest changed",
    )
    return artifact, _validate_construction(artifact)


def _preflight_projection(
    construction: SealedArtifact,
    rows: tuple[dict[str, Any], ...],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    require_text(model, "answer model")
    require_text(gateway_url, "answer gateway")
    _require(type(max_concurrency) is int and max_concurrency > 0, "answer concurrency changed")
    prompts = tuple(_plain_messages(row["messages"]) for row in rows)
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == EXPECTED_PROVIDER_CALLS,
        "delta answer prompts must be ten unique physical calls",
    )
    plan_rows = []
    for source, receipt in zip(rows, population.ordered_rows, strict=True):
        _require(
            source["messages_sha256"] == receipt.messages_sha256
            and source["prompt_token_proxy"] == receipt.prompt_token_proxy,
            "preflight prompt population changed construction bytes",
        )
        body = {
            "allowed_evidence_ids": source["allowed_evidence_ids"],
            "dated_question_sha256": source["dated_question_sha256"],
            "messages": source["messages"],
            "messages_sha256": source["messages_sha256"],
            "ordinal": source["ordinal"],
            "parent_prediction": source["parent_prediction"],
            "parent_prediction_sha256": source["parent_prediction_sha256"],
            "prompt_token_proxy": source["prompt_token_proxy"],
            "question_id": source["question_id"],
            "question_sha256": source["question_sha256"],
            "source_prompt_row_receipt_sha256": source[
                "prompt_row_receipt_sha256"
            ],
        }
        plan_rows.append(
            {**body, "preflight_row_receipt_sha256": identity_sha256(body)}
        )
    payload = {
        "construction_artifact_sha256": construction.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": plan_rows,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="reduced_missing10_preflight")
    return payload


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    construction, rows = _read_construction(
        Path(args.output_root), str(args.expected_construction_sha256)
    )
    payload = _preflight_projection(
        construction,
        rows,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "artifact": artifact.path.as_posix(),
        "construction_sha256": construction.sha256,
        "created": created,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
        "retained_transformer_token_state_bytes": 0,
    }


def _validate_preflight(artifact: SealedArtifact) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    rows = payload.get("physical_prompt_rows")
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == EXPECTED_PROVIDER_CALLS
        and payload.get("required_authorized_provider_calls") == EXPECTED_PROVIDER_CALLS
        and type(rows) is list
        and len(rows) == EXPECTED_PROVIDER_CALLS,
        "sealed delta preflight changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    validated: list[dict[str, Any]] = []
    for ordinal, raw in zip(EXPECTED_ORDINALS, rows, strict=True):
        _require(type(raw) is dict, "delta preflight row changed type")
        assert type(raw) is dict
        body = dict(raw)
        declared = body.pop("preflight_row_receipt_sha256", None)
        messages = _plain_messages(raw.get("messages", ()))
        _require(
            raw.get("ordinal") == ordinal
            and declared == identity_sha256(body)
            and identity_sha256(list(messages)) == raw.get("messages_sha256")
            and count_chat_prompt_token_proxy(messages) == raw.get("prompt_token_proxy")
            and int(raw["prompt_token_proxy"]) <= MAX_CHAT_PROMPT_TOKENS,
            "delta preflight row seal/messages changed",
        )
        prompts.append(messages)
        validated.append(dict(raw))
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.unique_prompt_count == EXPECTED_PROVIDER_CALLS,
        "delta sealed prompt population changed",
    )
    return tuple(prompts), tuple(validated)


def _read_preflight(output_root: Path, expected_sha256: str) -> tuple[SealedArtifact, tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "expected delta preflight"),
        "sealed delta preflight digest changed",
    )
    prompts, rows = _validate_preflight(artifact)
    return artifact, prompts, rows


def _runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        artifact.payload.get("model") == model
        and artifact.payload.get("gateway_url") == gateway_url
        and artifact.payload.get("max_concurrency") == max_concurrency,
        "runtime settings differ from sealed delta preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=output_root / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "reduced_missing10_delta_answer_diagnostic_v1",
            "authorized_unique_calls": EXPECTED_PROVIDER_CALLS,
            "construction_artifact_sha256": artifact.payload[
                "construction_artifact_sha256"
            ],
            "experiment_format": RUN_FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact,
        prompts,
        output_root=Path(args.output_root),
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, _rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == EXPECTED_PROVIDER_CALLS,
        f"provider-run requires exact authorization for {EXPECTED_PROVIDER_CALLS} calls",
    )
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))  # noqa: SLF001
    try:
        batch = _checkpoint_batch(artifact, prompts, args=args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == EXPECTED_PROVIDER_CALLS,
        "delta provider population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
        "retained_transformer_token_state_bytes": 0,
    }


def _parse_completion(
    completion: str,
    *,
    parent_prediction: str,
    allowed_evidence_ids: Sequence[str],
) -> tuple[str, str, tuple[str, ...], str]:
    try:
        raw = json.loads(completion)
    except (json.JSONDecodeError, TypeError):
        return parent_prediction, "invalid_json_parent_fallback", (), "invalid_json"
    if type(raw) is not dict:
        return parent_prediction, "invalid_schema_parent_fallback", (), "not_object"
    decision = raw.get("decision")
    prediction = raw.get("prediction")
    used = raw.get("used_evidence_ids", [])
    allowed = set(allowed_evidence_ids)
    if (
        decision not in {"keep_parent", "replace"}
        or type(prediction) is not str
        or not prediction.strip()
        or prediction.strip() != prediction
        or type(used) is not list
        or any(type(value) is not str for value in used)
        or len(used) != len(set(used))
        or not set(used) <= allowed
    ):
        return parent_prediction, "invalid_schema_parent_fallback", (), "invalid_schema"
    if decision == "keep_parent":
        return parent_prediction, "provider_keep_parent", tuple(used), "valid"
    return prediction, "provider_replace", tuple(used), "valid"


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "unique_records": [
            {
                key: child
                for key, child in row.items()
                if key not in {"checkpoint_hit", "physical_call"}
            }
            for row in value["unique_records"]
        ],
        "usage": {
            key: child
            for key, child in value["usage"].items()
            if key not in {"checkpoint_hits", "physical_calls"}
        },
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "prompt_population": value["prompt_population"],
    }


def _materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == EXPECTED_PROVIDER_CALLS
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == EXPECTED_PROVIDER_CALLS
        and len(batch.unique_records) == EXPECTED_PROVIDER_CALLS,
        "materialization requires ten checkpoint-only completions",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(len(records) == EXPECTED_PROVIDER_CALLS, "completion identities repeat")
    rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    for plan, completion in zip(prompt_rows, batch.logical_completions, strict=True):
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            "checkpoint completion record changed",
        )
        assert record is not None
        prediction, source, used, validity = _parse_completion(
            completion,
            parent_prediction=plan["parent_prediction"],
            allowed_evidence_ids=plan["allowed_evidence_ids"],
        )
        body = {
            "call_key_sha256": record.call_key_sha256,
            "changed_from_parent": prediction != plan["parent_prediction"],
            "completion_receipt_sha256": record.completion_sha256,
            "completion_validation": validity,
            "dated_question_sha256": plan["dated_question_sha256"],
            "messages_sha256": plan["messages_sha256"],
            "ordinal": plan["ordinal"],
            "parent_prediction_sha256": plan["parent_prediction_sha256"],
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "prediction_source": source,
            "preflight_row_receipt_sha256": plan[
                "preflight_row_receipt_sha256"
            ],
            "question_id": plan["question_id"],
            "question_sha256": plan["question_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "source_prompt_row_receipt_sha256": plan[
                "source_prompt_row_receipt_sha256"
            ],
            "used_evidence_ids": list(used),
        }
        row = {**body, "result_row_receipt_sha256": identity_sha256(body)}
        rows.append(row)
        seam = {
            "dated_question_sha256": row["dated_question_sha256"],
            "ordinal": row["ordinal"],
            "prediction": row["prediction"],
            "prediction_sha256": row["prediction_sha256"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        judge_rows.append({**seam, "answer_row_sha256": identity_sha256(seam)})
    payload = {
        "changed_prediction_count": sum(row["changed_from_parent"] for row in rows),
        "completion_batch": _stable_batch(batch),
        "construction_artifact_sha256": preflight.payload[
            "construction_artifact_sha256"
        ],
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["completion_validation"] != "valid" for row in rows
        ),
        "judge_rows": judge_rows,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": EXPECTED_PROVIDER_CALLS,
        "questions": rows,
        "required_authorized_provider_calls": EXPECTED_PROVIDER_CALLS,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="reduced_missing10_run")
    return payload


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    payload = _materialization_projection(preflight, rows, batch)
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "terminal_run_replayed": not created,
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    construction, _rows = _read_construction(
        Path(args.output_root), str(args.expected_construction_sha256)
    )
    preflight, prompts, prompt_rows = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    _require(
        preflight.payload.get("construction_artifact_sha256") == construction.sha256,
        "replay construction/preflight binding changed",
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_projection(preflight, prompt_rows, batch)
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    expected = require_sha256(str(args.expected_run_sha256), "expected delta run")
    _require(
        terminal.sha256 == expected and terminal.payload == rebuilt,
        "terminal delta run differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME,
        terminal.payload,
    )
    _require(
        replay.sha256 == terminal.sha256,
        "answer replay is not byte-identical to the terminal run",
    )
    return {
        "byte_identical": True,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": terminal.sha256,
    }


def _add_runtime_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=live.DEFAULT_TERRA_GATEWAY_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    construct = commands.add_parser("construct", help="build ten delta-only prompts")
    construct.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    construct.add_argument(
        "--reduced-construction", type=Path, default=DEFAULT_REDUCED_CONSTRUCTION
    )
    construct.add_argument(
        "--parent-composition", type=Path, default=DEFAULT_PARENT_COMPOSITION
    )
    construct.add_argument(
        "--expected-reduced-construction-sha256",
        default=EXPECTED_REDUCED_CONSTRUCTION_SHA256,
    )
    construct.add_argument(
        "--expected-parent-composition-sha256",
        default=EXPECTED_PARENT_COMPOSITION_SHA256,
    )

    preflight = commands.add_parser("preflight", help="seal the ten prompt population")
    _add_runtime_settings(preflight)
    preflight.add_argument("--expected-construction-sha256", required=True)

    provider = commands.add_parser("provider-run", help="execute the sealed prompts")
    _add_runtime_settings(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "materialize", help="seal checkpoint-only predictions"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser("replay", help="prove checkpoint-only byte identity")
    _add_runtime_settings(replay)
    replay.add_argument("--expected-construction-sha256", required=True)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "construct":
        result = _construct(args)
    elif args.command == "preflight":
        result = _preflight(args)
    elif args.command == "provider-run":
        result = _provider(args)
    elif args.command == "materialize":
        result = _materialize(args)
    elif args.command == "replay":
        result = _replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "EXPECTED_ORDINALS",
    "FACT_METHOD_IDS",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "ReducedMissing10DiagnosticError",
    "build_construction_payload",
    "main",
]
