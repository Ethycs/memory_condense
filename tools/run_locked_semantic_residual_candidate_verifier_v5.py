#!/usr/bin/env python3
"""Sealed, gold-blind Sol selector over every raw V4 replace candidate.

V5 does not retrieve memory and does not ask a model to write an answer.  It
authenticates the byte-identical locked V4 answer lifecycle, mechanically
freezes every raw Terra completion whose decision was ``replace``, and asks a
strict selector to choose between that exact candidate and the exact V3
current prediction.  The final prediction is therefore always byte-identical
to one of those two sealed strings.  Malformed, unsupported, speculative, or
unverifiable selections fail closed to the current prediction.

The source loader also receipts the V4 protocol-normalization opportunity:
raw ``keep_current`` completions that returned the exact current prediction
but illegally carried handles are canonicalized locally to current plus an
empty handle list.  This consumes no provider call and changes no prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import canonical_json, quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_locked_semantic_residual_answer_v4 as v4  # noqa: E402
from tools.matched_eval import live  # noqa: E402
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
from tools.matched_eval.typed_memory_final_arm import judge_row_projection  # noqa: E402
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    TypedOperatorSpec,
    compile_typed_operator_spec,
)


FORMAT = "memory-condense-locked-semantic-residual-sol-selector-v5"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
CANDIDATE_FORMAT = f"{FORMAT}-candidate-v1"
NORMALIZATION_FORMAT = f"{FORMAT}-keep-current-normalization-v1"
PARSE_FORMAT = f"{FORMAT}-parsed-selection-v1"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row-v1"

PREFLIGHT_NAME = "locked-semantic-residual-selector-preflight-v5.json"
RUN_NAME = "locked-semantic-residual-answer-v5.json"
REPLAY_NAME = "locked-semantic-residual-answer-replay-v5.json"
CHECKPOINT_DIR_NAME = "locked-semantic-residual-selector-checkpoints-v5"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V4_ROOT = v4.DEFAULT_OUTPUT
DEFAULT_V3_PARENT = v4.DEFAULT_ANSWER
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-residual-answer-v5-r1"
)

LOCKED_V4_PREFLIGHT_SHA256 = (
    "52df0b0a4388ab2297a4af41b577839ab8bc1447df69cb49aa14017de3593bcc"
)
LOCKED_V4_RUN_REPLAY_SHA256 = (
    "de717ce73acad9d634f4639bea786bcae94843933d2acd882917c8ed2a25c2e2"
)
LOCKED_V3_PARENT_SHA256 = (
    "07c6f3125e65094880384c1c1c6f7d9be0600475f1fe58d050796fc0f48493d1"
)

QUESTION_COUNT = 100
CANDIDATE_COUNT = 15
NORMALIZATION_COUNT = 13
DEFAULT_MODEL = "codex_sdk/gpt-5.6-sol"
HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_COMPLETE_CHAT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE
RESIDUAL_EVIDENCE_TOKEN_CAP = 2_400
PROTECTED_OWNER_TOKEN_CAP = 2_400

_HANDLE_RE = re.compile(r"^[RP][0-9]{4}$")
_RESIDUAL_HANDLE_RE = re.compile(r"^R[0-9]{4}$")
_PERSONAL_RE = re.compile(r"\b(?:i|i'm|i’ve|i've|me|my|mine)\b", re.I)
_PREFERENCE_RE = re.compile(
    r"\b(?:prefer|preference|like|love|enjoy|want|need|intend|interested|"
    r"avoid|dislike|hate|favorite|favourite|plan|hope)\w*\b",
    re.I,
)
_SPECULATIVE_RE = re.compile(
    r"\b(?:maybe|might|may|could|would like|consider(?:ing)?|thinking about|"
    r"tentative|possibly|perhaps|hope(?:fully)?|intend(?:ing)?|plan(?:ned|ning)?)\b",
    re.I,
)
_STRICT_NUMBER_RE = re.compile(
    r"^[\s]*[$€£]?[\s]*[+-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?%?[\s]*$"
)
_RESULT_NUMBER_RE = re.compile(r"[+-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?")

_SELECTIONS = frozenset({"candidate", "current"})
_SUPPORT_CLASSES = frozenset(
    {"direct", "derived", "paraphrase", "recommendation", "unsupported"}
)
_DERIVATION_OPERATIONS = frozenset(
    {
        "sum",
        "difference",
        "duration_days",
        "duration_months",
        "duration_years",
        "count_distinct",
        "greater_than",
    }
)

SELECTOR_SYSTEM_PROMPT = """You are a strict evidence selector.

Choose only between CURRENT_PREDICTION and CANDIDATE_PREDICTION. Never write,
rewrite, merge, or improve either prediction. Use only the supplied dated
question, typed operator specification, and bounded R/P evidence rows.

Candidate selection rules:
- Every material claim must be supported by the exact cited handles.
- A candidate selection must cite at least one R handle.
- Direct support states the candidate value or fact in evidence. Derived
  support requires a typed derivation whose operands are exact cited values.
- Assistant advice or promotional language cannot establish a personal event,
  preference, purchase, or intention. Personalized recommendations require a
  cited user preference or intent.
- Proposed, hypothetical, or speculative text is not a completed user-memory
  scalar unless the typed specification explicitly includes proposed states.
- Mark semantically equivalent wording with equivalent_to_current=true; it
  will be canonicalized to CURRENT_PREDICTION.
- For globally scoped count/set/all/total requests whose semantic frontier is
  open and has no sealed operand-closure proof, set needs_global_search=true
  and select current. Grounding in a packed subset is not global closure.
- If uncertain, select current and use support_class=unsupported.

Return one strict JSON object matching RESPONSE_SCHEMA. Do not include answer
text, prose, Markdown, or keys outside the schema."""

RESPONSE_SCHEMA = {
    "directly_answers": "boolean",
    "equivalent_to_current": "boolean",
    "needs_global_search": "boolean",
    "personal_scope_supported": "boolean",
    "selection": "candidate|current",
    "support_class": (
        "direct|derived|paraphrase|recommendation|unsupported"
    ),
    "typed_derivation": (
        "null or {operation:sum|difference|duration_days|duration_months|"
        "duration_years|count_distinct|greater_than,operands:[{handle_id,value}],"
        "result:string,unit:string|null}"
    ),
    "unsupported_claims": ["exact unsupported claim strings; empty when supported"],
    "used_handle_ids": ["exact R/P handle IDs"],
}


class LockedSemanticResidualCandidateVerifierV5Error(MatchedEvalContractError):
    """Raised when a sealed V4/V5 source, prompt, or selector contract changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticResidualCandidateVerifierV5Error(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return dict(value)


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return list(value)


def _with_receipt(body: Mapping[str, Any], key: str = "receipt_sha256") -> dict[str, Any]:
    value = dict(body)
    value[key] = identity_sha256(value)
    return value


def _self_hashed(row: Mapping[str, Any], key: str, label: str) -> None:
    body = dict(row)
    declared = body.pop(key, None)
    _require(declared == identity_sha256(body), f"{label} receipt changed")


def _verified(path: Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, label),
        f"{label} SHA-256 changed",
    )
    return artifact


@dataclass(frozen=True, slots=True)
class V4SourceBundle:
    preflight: SealedArtifact
    run: SealedArtifact
    replay: SealedArtifact
    v3_parent: SealedArtifact
    construction: SealedArtifact
    prompts: tuple[tuple[dict[str, str], ...], ...]
    plans: tuple[dict[str, Any], ...]
    batch: FastCompletionBatch


def _v4_runtime_args(root: Path, preflight: SealedArtifact) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=root,
        model=preflight.payload["model"],
        gateway_url=preflight.payload["gateway_url"],
        max_concurrency=preflight.payload["max_concurrency"],
    )


def load_authenticated_v4_sources(
    *,
    v4_root: Path,
    v3_parent_path: Path,
    expected_v4_preflight_sha256: str,
    expected_v4_run_sha256: str,
    expected_v4_replay_sha256: str,
    expected_v3_parent_sha256: str,
) -> V4SourceBundle:
    """Rebuild V4 from its sealed journals and authenticate every parent seam."""

    for supplied, locked, label in (
        (expected_v4_preflight_sha256, LOCKED_V4_PREFLIGHT_SHA256, "V4 preflight"),
        (expected_v4_run_sha256, LOCKED_V4_RUN_REPLAY_SHA256, "V4 run"),
        (expected_v4_replay_sha256, LOCKED_V4_RUN_REPLAY_SHA256, "V4 replay"),
        (expected_v3_parent_sha256, LOCKED_V3_PARENT_SHA256, "V3 parent"),
    ):
        _require(
            require_sha256(supplied, label) == locked,
            f"{label} is not the locked V5 source",
        )

    preflight, prompts, plans = v4._read_preflight(  # noqa: SLF001
        v4_root, expected_v4_preflight_sha256
    )
    run = _verified(v4_root / v4.RUN_NAME, expected_v4_run_sha256, "V4 run")
    replay = _verified(
        v4_root / v4.REPLAY_NAME, expected_v4_replay_sha256, "V4 replay"
    )
    _require(
        run.sha256 == replay.sha256
        and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload),
        "V4 run/replay are not byte-identical",
    )

    construction, construction_replay, gate, v3_parent, source_plans = v4._load_sources(  # noqa: SLF001
        construction_path=Path(v4.DEFAULT_CONSTRUCTION),
        construction_sha256=str(preflight.payload["construction_artifact_sha256"]),
        construction_replay_path=Path(v4.DEFAULT_CONSTRUCTION_REPLAY),
        construction_replay_sha256=str(
            preflight.payload["construction_replay_artifact_sha256"]
        ),
        gate_path=Path(v4.DEFAULT_GATE),
        gate_sha256=str(preflight.payload["gate_artifact_sha256"]),
        answer_path=v3_parent_path,
        answer_sha256=expected_v3_parent_sha256,
    )
    v4._assert_preflight_source_binding(  # noqa: SLF001
        preflight,
        construction,
        construction_replay,
        gate,
        v3_parent,
        source_plans,
    )
    _require(tuple(plans) == tuple(source_plans), "V4 plan population changed")

    batch = v4._checkpoint_batch(  # noqa: SLF001
        preflight,
        prompts,
        args=_v4_runtime_args(v4_root, preflight),
        client=None,
    )
    rebuilt = v4._materialization_payload(preflight, plans, batch)  # noqa: SLF001
    _require(
        canonical_json_bytes(rebuilt) == canonical_json_bytes(run.payload),
        "V4 run differs from its sealed preflight/journals",
    )
    _require(
        run.payload.get("preflight_artifact_sha256") == preflight.sha256
        and run.payload.get("answer_artifact_sha256") == v3_parent.sha256
        and run.payload.get("completion_batch") == v4._stable_batch(batch),  # noqa: SLF001
        "V4 batch or parent binding changed",
    )

    v3_rows = _exact_list(v3_parent.payload.get("questions"), "V3 parent rows")
    v4_rows = _exact_list(run.payload.get("questions"), "V4 rows")
    _require(
        len(v3_rows) == len(v4_rows) == len(plans) == QUESTION_COUNT
        and tuple(row.get("ordinal") for row in v3_rows) == tuple(range(QUESTION_COUNT))
        and tuple(row.get("ordinal") for row in v4_rows) == tuple(range(QUESTION_COUNT)),
        "V3/V4 row population changed",
    )
    for ordinal, (plan, parent, child) in enumerate(
        zip(plans, v3_rows, v4_rows, strict=True)
    ):
        _self_hashed(parent, "source_row_sha256", f"V3 row {ordinal}")
        _self_hashed(child, "source_row_sha256", f"V4 row {ordinal}")
        _require(
            plan["ordinal"] == parent["ordinal"] == child["ordinal"] == ordinal
            and plan["question_id"] == parent["question_id"] == child["question_id"]
            and plan["question_sha256"]
            == parent["question_sha256"]
            == child["question_sha256"]
            and plan["dated_question_sha256"]
            == parent["dated_question_sha256"]
            == child["dated_question_sha256"]
            and plan["source_v3_answer_row_sha256"] == parent["source_row_sha256"]
            and child["source_v3_answer_row_sha256"] == parent["source_row_sha256"]
            and plan["current_prediction"] == parent["prediction"]
            and plan["current_prediction_sha256"] == parent["prediction_sha256"]
            and child["parent_prediction_sha256"] == parent["prediction_sha256"]
            and child["answer_plan_receipt_sha256"]
            == plan["answer_plan_receipt_sha256"],
            f"V3/V4 parent seam changed at ordinal {ordinal}",
        )
    return V4SourceBundle(
        preflight,
        run,
        replay,
        v3_parent,
        construction,
        prompts,
        tuple(dict(row) for row in plans),
        batch,
    )


def _strict_raw_v4_completion(text: str, *, ordinal: int) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LockedSemanticResidualCandidateVerifierV5Error(
            f"raw V4 completion is malformed at ordinal {ordinal}"
        ) from exc
    raw = _exact_dict(value, f"raw V4 completion {ordinal}")
    _require(
        set(raw) == {"decision", "prediction", "used_evidence_handle_ids"}
        and raw["decision"] in {"replace", "keep_current"}
        and type(raw["prediction"]) is str
        and type(raw["used_evidence_handle_ids"]) is list
        and all(
            type(handle) is str and _HANDLE_RE.fullmatch(handle)
            for handle in raw["used_evidence_handle_ids"]
        )
        and len(set(raw["used_evidence_handle_ids"]))
        == len(raw["used_evidence_handle_ids"]),
        f"raw V4 completion schema changed at ordinal {ordinal}",
    )
    return raw


def _evidence_plane(
    plan: Mapping[str, Any], *, ordinal: int
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Recover exact R/P role/time rows from the bound V4 provider message."""

    messages = _exact_list(plan.get("messages"), f"V4 messages {ordinal}")
    _require(
        len(messages) == 2
        and messages[0].get("role") == "system"
        and messages[1].get("role") == "user"
        and identity_sha256(messages) == plan.get("messages_sha256"),
        f"V4 message seam changed at ordinal {ordinal}",
    )
    try:
        provider = json.loads(messages[1]["content"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LockedSemanticResidualCandidateVerifierV5Error(
            f"V4 provider input is not exact JSON at ordinal {ordinal}"
        ) from exc
    provider = _exact_dict(provider, f"V4 provider input {ordinal}")
    _require(
        provider.get("dated_question")
        and quote_sha256(provider["dated_question"]) == plan["dated_question_sha256"]
        and provider.get("current_answer") == plan["current_prediction"]
        and identity_sha256(provider) == plan["provider_input_sha256"],
        f"V4 provider input binding changed at ordinal {ordinal}",
    )

    grounding = _exact_list(
        plan.get("evidence_grounding_rows"), f"V4 grounding rows {ordinal}"
    )
    by_handle: dict[str, dict[str, Any]] = {}
    for raw in grounding:
        row = _exact_dict(raw, f"V4 grounding row {ordinal}")
        handle = row.get("evidence_handle")
        _require(
            set(row)
            == {
                "evidence_handle",
                "handle_class",
                "quote",
                "quote_sha256",
                "source_group_handle",
            }
            and type(handle) is str
            and _HANDLE_RE.fullmatch(handle)
            and handle not in by_handle
            and quote_sha256(row["quote"]) == row["quote_sha256"]
            and row["handle_class"]
            == ("residual" if handle.startswith("R") else "protected_owner"),
            f"V4 grounding row changed at ordinal {ordinal}",
        )
        by_handle[handle] = row

    residual = _exact_list(
        provider.get("residual_evidence"), f"V4 R evidence {ordinal}"
    )
    protected = _exact_list(
        provider.get("protected_owner_evidence"), f"V4 P evidence {ordinal}"
    )
    full: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_class, source_rows in (
        ("residual", residual),
        ("protected_owner", protected),
    ):
        for raw in source_rows:
            row = _exact_dict(raw, f"V4 provider evidence {ordinal}")
            base_keys = {
                "created_at",
                "event_dates",
                "evidence_handle",
                "quote",
                "role",
                "source_group_handle",
            }
            owner_keys = {
                "owner_binding_receipt_sha256",
                "owner_candidate_id",
                "protected_duplicate_receipt_sha256",
                "quote_sha256",
                "segment_receipt_sha256",
            }
            _require(
                set(row)
                == (base_keys if expected_class == "residual" else base_keys | owner_keys),
                f"V4 provider evidence schema changed at ordinal {ordinal}",
            )
            handle = row.get("evidence_handle")
            parent = by_handle.get(str(handle))
            _require(
                parent is not None
                and handle not in seen
                and parent["handle_class"] == expected_class
                and parent["quote"] == row["quote"]
                and parent["source_group_handle"] == row["source_group_handle"]
                and quote_sha256(row["quote"]) == parent["quote_sha256"]
                and (
                    expected_class == "residual"
                    or row["quote_sha256"] == parent["quote_sha256"]
                    and all(
                        re.fullmatch(r"[0-9a-f]{64}", str(row[key]))
                        for key in (
                            "owner_binding_receipt_sha256",
                            "protected_duplicate_receipt_sha256",
                            "segment_receipt_sha256",
                        )
                    )
                    and type(row["owner_candidate_id"]) is str
                    and bool(row["owner_candidate_id"])
                )
                and row["role"] in {"user", "assistant"}
                and (row["created_at"] is None or type(row["created_at"]) is str)
                and type(row["event_dates"]) is list
                and all(type(value) is str for value in row["event_dates"]),
                f"V4 role/time/quote join changed at ordinal {ordinal}",
            )
            seen.add(str(handle))
            body = {
                "created_at": row["created_at"],
                "event_dates": list(row["event_dates"]),
                "evidence_handle": handle,
                "handle_class": expected_class,
                "quote": row["quote"],
                "quote_sha256": parent["quote_sha256"],
                "role": row["role"],
                "source_group_handle": row["source_group_handle"],
            }
            if expected_class == "protected_owner":
                body.update(
                    {
                        "owner_binding_receipt_sha256": row[
                            "owner_binding_receipt_sha256"
                        ],
                        "owner_candidate_id": row["owner_candidate_id"],
                        "protected_duplicate_receipt_sha256": row[
                            "protected_duplicate_receipt_sha256"
                        ],
                        "segment_receipt_sha256": row[
                            "segment_receipt_sha256"
                        ],
                    }
                )
            full.append(_with_receipt(body, "evidence_row_receipt_sha256"))

    _require(
        seen == set(by_handle)
        and tuple(row["evidence_handle"] for row in full)
        == tuple(plan["allowed_evidence_handle_ids"]),
        f"V4 full R/P evidence population changed at ordinal {ordinal}",
    )
    residual_tokens = count_tokens(canonical_json(residual))
    protected_tokens = count_tokens(canonical_json(protected))
    union_tokens = count_tokens(canonical_json(full))
    quote_tokens = count_tokens("\n".join(row["quote"] for row in full))
    metadata_only = [
        {**row, "quote": ""}
        for row in full
    ]
    metadata_tokens = count_tokens(canonical_json(metadata_only))
    _require(
        residual_tokens <= RESIDUAL_EVIDENCE_TOKEN_CAP
        and protected_tokens <= PROTECTED_OWNER_TOKEN_CAP,
        f"V4 inherited evidence plane cap changed at ordinal {ordinal}",
    )
    accounting = _with_receipt(
        {
            "full_union_serialized_token_proxy": union_tokens,
            "metadata_serialized_token_proxy": metadata_tokens,
            "protected_owner_cap": PROTECTED_OWNER_TOKEN_CAP,
            "protected_owner_plane_sha256": identity_sha256(protected),
            "protected_owner_serialized_token_proxy": protected_tokens,
            "quote_content_token_proxy": quote_tokens,
            "residual_cap": RESIDUAL_EVIDENCE_TOKEN_CAP,
            "residual_plane_sha256": identity_sha256(residual),
            "residual_serialized_token_proxy": residual_tokens,
            "row_count": len(full),
            "union_population_sha256": identity_sha256(full),
        }
    )
    return tuple(full), accounting


def _source_receipts(
    bundle: V4SourceBundle,
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    record: Any,
) -> dict[str, Any]:
    body = {
        "answer_plan_receipt_sha256": plan["answer_plan_receipt_sha256"],
        "completion_receipt_sha256": record.completion_sha256,
        "construction_question_receipt_sha256": plan[
            "construction_question_receipt_sha256"
        ],
        "request_journal_sha256": record.request_journal_sha256,
        "response_journal_sha256": record.response_journal_sha256,
        "source_v3_row_sha256": plan["source_v3_answer_row_sha256"],
        "source_v4_row_sha256": result["source_row_sha256"],
        "terminal_prompt_receipt_sha256": plan["terminal_prompt_receipt_sha256"],
        "v3_parent_artifact_sha256": bundle.v3_parent.sha256,
        "v4_preflight_artifact_sha256": bundle.preflight.sha256,
        "v4_replay_artifact_sha256": bundle.replay.sha256,
        "v4_run_artifact_sha256": bundle.run.sha256,
    }
    return _with_receipt(body)


def _selector_messages(payload: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    assert_gold_blind(payload, path="semantic_residual_v5.selector_input")
    return (
        {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(dict(payload))},
    )


def _candidate_plan(
    bundle: V4SourceBundle,
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    record: Any,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = int(plan["ordinal"])
    evidence, accounting = _evidence_plane(plan, ordinal=ordinal)
    used = tuple(raw["used_evidence_handle_ids"])
    allowed = tuple(row["evidence_handle"] for row in evidence)
    _require(
        raw["decision"] == "replace"
        and type(raw["prediction"]) is str
        and bool(raw["prediction"])
        and set(used) <= set(allowed)
        and any(_RESIDUAL_HANDLE_RE.fullmatch(handle) for handle in used),
        f"raw V4 replace candidate contract changed at ordinal {ordinal}",
    )
    dated_question = json.loads(plan["messages"][1]["content"])["dated_question"]
    spec = compile_typed_operator_spec(dated_question)
    construction_row = bundle.construction.payload["questions"][ordinal]
    commitment = _exact_dict(
        construction_row.get("semantic_search_commitment"),
        f"semantic search commitment {ordinal}",
    )
    _self_hashed(
        commitment,
        "receipt_sha256",
        f"semantic search commitment {ordinal}",
    )
    # _self_hashed validates a copy; retain the declared receipt in the prompt.
    commitment = dict(construction_row["semantic_search_commitment"])
    global_closure_required = bool(
        spec.operation == "count_or_aggregate"
        or spec.answer_shape.value == "set_list"
        or re.search(
            r"\b(?:all|every|entire|total number|complete list)\b",
            dated_question,
            re.I,
        )
    )
    _require(
        construction_row.get("question_receipt_sha256")
        == plan["construction_question_receipt_sha256"]
        and commitment.get("packing_closed") is False
        and commitment.get("support_closure_proven") is False
        and type(commitment.get("unpacked_novel_count")) is int
        and commitment.get("unpacked_novel_population_sha256"),
        f"R7 open semantic frontier binding changed at ordinal {ordinal}",
    )
    receipts = _source_receipts(bundle, plan, result, record)
    candidate_body = {
        "candidate_original_used_handle_ids": list(used),
        "candidate_prediction": raw["prediction"],
        "candidate_prediction_sha256": quote_sha256(raw["prediction"]),
        "current_prediction": plan["current_prediction"],
        "current_prediction_sha256": plan["current_prediction_sha256"],
        "dated_question": dated_question,
        "dated_question_sha256": plan["dated_question_sha256"],
        "evidence_grounding_rows": list(evidence),
        "evidence_plane_accounting": accounting,
        "format": CANDIDATE_FORMAT,
        "global_closure_required": global_closure_required,
        "ordinal": ordinal,
        "parent_v4_prediction": result["prediction"],
        "parent_v4_prediction_sha256": result["prediction_sha256"],
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "route_id": plan["route_id"],
        "source_receipts": receipts,
        "semantic_search_commitment": commitment,
        "typed_operator_spec": spec.projection(),
        "typed_operand_closure_proof": None,
    }
    candidate_receipt = identity_sha256(candidate_body)
    provider_payload = {
        "candidate_original_used_handle_ids": list(used),
        "candidate_prediction": raw["prediction"],
        "candidate_receipt_sha256": candidate_receipt,
        "current_prediction": plan["current_prediction"],
        "dated_question": candidate_body["dated_question"],
        "evidence_grounding_rows": list(evidence),
        "evidence_plane_accounting": accounting,
        "format": f"{FORMAT}-provider-input-v1",
        "global_closure_required": global_closure_required,
        "response_schema": RESPONSE_SCHEMA,
        "source_receipts": receipts,
        "semantic_search_commitment": commitment,
        "typed_operator_spec": spec.projection(),
        "typed_operand_closure_proof": None,
    }
    messages = _selector_messages(provider_payload)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        f"V5 selector envelope exceeded 8k at ordinal {ordinal}",
    )
    body = {
        **candidate_body,
        "candidate_receipt_sha256": candidate_receipt,
        "complete_envelope_token_proxy": prompt_tokens + OUTPUT_TOKEN_RESERVE,
        "messages": list(messages),
        "messages_sha256": identity_sha256(list(messages)),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider_payload),
    }
    return _with_receipt(body, "candidate_plan_receipt_sha256")


def _normalization_row(
    bundle: V4SourceBundle,
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    record: Any,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = int(plan["ordinal"])
    used = tuple(raw["used_evidence_handle_ids"])
    _require(
        raw["decision"] == "keep_current"
        and raw["prediction"] == plan["current_prediction"]
        and bool(used),
        f"V4 keep-current normalization source changed at ordinal {ordinal}",
    )
    body = {
        "canonical_prediction": plan["current_prediction"],
        "canonical_prediction_sha256": plan["current_prediction_sha256"],
        "canonical_used_handle_ids": [],
        "format": NORMALIZATION_FORMAT,
        "ordinal": ordinal,
        "original_used_handle_ids": list(used),
        "question_id": plan["question_id"],
        "source_receipts": _source_receipts(bundle, plan, result, record),
        "zero_output_change": True,
        "zero_provider_calls": 0,
    }
    return _with_receipt(body, "normalization_receipt_sha256")


def freeze_v5_population(
    bundle: V4SourceBundle,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Mechanically partition all 68 raw V4 completions, without final filtering."""

    physical = tuple(
        row for row in bundle.plans if row["mode"] == v4.RESIDUAL_MODE
    )
    v4_results = tuple(bundle.run.payload["questions"])
    records = {row.messages_sha256: row for row in bundle.batch.unique_records}
    candidates: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    empty_keep = 0
    _require(
        len(physical) == len(bundle.batch.logical_completions) == 68,
        "V4 physical completion population changed",
    )
    for plan, completion in zip(
        physical, bundle.batch.logical_completions, strict=True
    ):
        ordinal = int(plan["ordinal"])
        result = v4_results[ordinal]
        record = records.get(plan["messages_sha256"])
        _require(
            record is not None
            and record.completion == completion
            and record.completion_sha256 == result["completion_receipt_sha256"]
            and record.call_key_sha256 == result["call_key_sha256"]
            and record.request_journal_sha256 == result["request_journal_sha256"]
            and record.response_journal_sha256 == result["response_journal_sha256"],
            f"V4 completion/journal seam changed at ordinal {ordinal}",
        )
        raw = _strict_raw_v4_completion(completion, ordinal=ordinal)
        if raw["decision"] == "replace":
            candidates.append(_candidate_plan(bundle, plan, result, record, raw))
        elif raw["used_evidence_handle_ids"]:
            normalizations.append(
                _normalization_row(bundle, plan, result, record, raw)
            )
        else:
            _require(
                raw["prediction"] == plan["current_prediction"],
                f"V4 empty keep-current changed at ordinal {ordinal}",
            )
            empty_keep += 1

    _require(
        len(candidates) == CANDIDATE_COUNT
        and len(normalizations) == NORMALIZATION_COUNT
        and empty_keep == 40
        and tuple(row["ordinal"] for row in candidates)
        == tuple(sorted(row["ordinal"] for row in candidates))
        and tuple(row["ordinal"] for row in normalizations)
        == tuple(sorted(row["ordinal"] for row in normalizations)),
        "mechanical V5 raw completion partition changed",
    )
    return tuple(candidates), tuple(normalizations)


_CANDIDATE_BODY_KEYS = frozenset(
    {
        "candidate_original_used_handle_ids",
        "candidate_prediction",
        "candidate_prediction_sha256",
        "current_prediction",
        "current_prediction_sha256",
        "dated_question",
        "dated_question_sha256",
        "evidence_grounding_rows",
        "evidence_plane_accounting",
        "format",
        "global_closure_required",
        "ordinal",
        "parent_v4_prediction",
        "parent_v4_prediction_sha256",
        "question_id",
        "question_sha256",
        "route_id",
        "semantic_search_commitment",
        "source_receipts",
        "typed_operator_spec",
        "typed_operand_closure_proof",
    }
)


def _original_evidence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "created_at",
        "event_dates",
        "evidence_handle",
    ]
    if row["handle_class"] == "protected_owner":
        keys.extend(
            [
                "owner_binding_receipt_sha256",
                "owner_candidate_id",
                "protected_duplicate_receipt_sha256",
            ]
        )
    keys.append("quote")
    if row["handle_class"] == "protected_owner":
        keys.extend(["quote_sha256"])
    keys.extend(["role"])
    if row["handle_class"] == "protected_owner":
        keys.extend(["segment_receipt_sha256"])
    keys.append("source_group_handle")
    return {key: row[key] for key in keys}


def _validate_typed_spec_projection(raw: object) -> dict[str, Any]:
    spec = _exact_dict(raw, "typed operator specification")
    declared = spec.get("receipt_sha256")
    body = dict(spec)
    body.pop("receipt_sha256", None)
    _require(
        declared == identity_sha256(body)
        and spec.get("retained_transformer_token_state_bytes") == 0,
        "typed operator specification receipt changed",
    )
    return spec


def _validate_evidence_rows(
    raw: object,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    handles: set[str] = set()
    base_keys = {
        "created_at",
        "event_dates",
        "evidence_handle",
        "evidence_row_receipt_sha256",
        "handle_class",
        "quote",
        "quote_sha256",
        "role",
        "source_group_handle",
    }
    for raw_row in _exact_list(raw, "V5 evidence rows"):
        row = _exact_dict(raw_row, "V5 evidence row")
        handle = row.get("evidence_handle")
        body = dict(row)
        declared = body.pop("evidence_row_receipt_sha256", None)
        _require(
            set(row)
            == (
                base_keys
                if row.get("handle_class") == "residual"
                else base_keys
                | {
                    "owner_binding_receipt_sha256",
                    "owner_candidate_id",
                    "protected_duplicate_receipt_sha256",
                    "segment_receipt_sha256",
                }
            )
            and declared == identity_sha256(body)
            and type(handle) is str
            and _HANDLE_RE.fullmatch(handle)
            and handle not in handles
            and row["handle_class"]
            == ("residual" if handle.startswith("R") else "protected_owner")
            and quote_sha256(row["quote"]) == row["quote_sha256"]
            and row["role"] in {"user", "assistant"}
            and (row["created_at"] is None or type(row["created_at"]) is str)
            and type(row["event_dates"]) is list
            and all(type(value) is str for value in row["event_dates"]),
            "V5 evidence row changed",
        )
        if row["handle_class"] == "protected_owner":
            _require(
                type(row["owner_candidate_id"]) is str
                and bool(row["owner_candidate_id"])
                and all(
                    re.fullmatch(r"[0-9a-f]{64}", str(row[key]))
                    for key in (
                        "owner_binding_receipt_sha256",
                        "protected_duplicate_receipt_sha256",
                        "segment_receipt_sha256",
                    )
                ),
                "V5 protected-owner provenance changed",
            )
        handles.add(handle)
        rows.append(row)
    _require(bool(rows), "V5 candidate evidence is empty")
    residual = [
        _original_evidence_row(row)
        for row in rows
        if row["handle_class"] == "residual"
    ]
    protected = [
        _original_evidence_row(row)
        for row in rows
        if row["handle_class"] == "protected_owner"
    ]
    metadata_only = [{**row, "quote": ""} for row in rows]
    accounting = {
        "full_union_serialized_token_proxy": count_tokens(canonical_json(rows)),
        "metadata_serialized_token_proxy": count_tokens(canonical_json(metadata_only)),
        "protected_owner_cap": PROTECTED_OWNER_TOKEN_CAP,
        "protected_owner_plane_sha256": identity_sha256(protected),
        "protected_owner_serialized_token_proxy": count_tokens(canonical_json(protected)),
        "quote_content_token_proxy": count_tokens(
            "\n".join(row["quote"] for row in rows)
        ),
        "residual_cap": RESIDUAL_EVIDENCE_TOKEN_CAP,
        "residual_plane_sha256": identity_sha256(residual),
        "residual_serialized_token_proxy": count_tokens(canonical_json(residual)),
        "row_count": len(rows),
        "union_population_sha256": identity_sha256(rows),
    }
    accounting = _with_receipt(accounting)
    _require(
        accounting["residual_serialized_token_proxy"]
        <= RESIDUAL_EVIDENCE_TOKEN_CAP
        and accounting["protected_owner_serialized_token_proxy"]
        <= PROTECTED_OWNER_TOKEN_CAP,
        "V5 evidence plane exceeds its inherited nonborrowable cap",
    )
    return tuple(rows), accounting


def _validate_candidate_plan(raw: object) -> dict[str, Any]:
    plan = _exact_dict(raw, "V5 candidate plan")
    body = dict(plan)
    declared_plan = body.pop("candidate_plan_receipt_sha256", None)
    _require(declared_plan == identity_sha256(body), "V5 candidate plan receipt changed")
    _require(
        set(body)
        == _CANDIDATE_BODY_KEYS
        | {
            "candidate_receipt_sha256",
            "complete_envelope_token_proxy",
            "messages",
            "messages_sha256",
            "output_token_reserve",
            "prompt_token_proxy",
            "provider_input_sha256",
        }
        and body.get("format") == CANDIDATE_FORMAT
        and type(body.get("ordinal")) is int
        and 0 <= body["ordinal"] < QUESTION_COUNT,
        "V5 candidate plan schema changed",
    )
    candidate_body = {key: body[key] for key in _CANDIDATE_BODY_KEYS}
    _require(
        body["candidate_receipt_sha256"] == identity_sha256(candidate_body)
        and quote_sha256(body["candidate_prediction"])
        == body["candidate_prediction_sha256"]
        and quote_sha256(body["current_prediction"])
        == body["current_prediction_sha256"]
        and quote_sha256(body["parent_v4_prediction"])
        == body["parent_v4_prediction_sha256"]
        and quote_sha256(body["dated_question"])
        == body["dated_question_sha256"],
        "V5 candidate text binding changed",
    )
    evidence, accounting = _validate_evidence_rows(body["evidence_grounding_rows"])
    _require(
        accounting == body["evidence_plane_accounting"],
        "V5 evidence accounting changed",
    )
    allowed = tuple(row["evidence_handle"] for row in evidence)
    cited = _exact_list(
        body["candidate_original_used_handle_ids"], "V5 original candidate handles"
    )
    _require(
        bool(cited)
        and len(set(cited)) == len(cited)
        and set(cited) <= set(allowed)
        and any(_RESIDUAL_HANDLE_RE.fullmatch(handle) for handle in cited),
        "V5 original candidate handle binding changed",
    )
    spec = _validate_typed_spec_projection(body["typed_operator_spec"])
    _require(
        spec["question_sha256"] == body["dated_question_sha256"]
        and body["global_closure_required"]
        == bool(
            spec["operation"] == "count_or_aggregate"
            or spec["answer_shape"] == "set_list"
            or re.search(
                r"\b(?:all|every|entire|total number|complete list)\b",
                body["dated_question"],
                re.I,
            )
        ),
        "V5 typed spec/global closure projection changed",
    )
    commitment = _exact_dict(
        body["semantic_search_commitment"], "V5 semantic search commitment"
    )
    commitment_body = dict(commitment)
    commitment_receipt = commitment_body.pop("receipt_sha256", None)
    _require(
        commitment_receipt == identity_sha256(commitment_body)
        and commitment.get("packing_closed") is False
        and commitment.get("support_closure_proven") is False,
        "V5 semantic search commitment changed",
    )
    _require(
        body["typed_operand_closure_proof"] is None,
        "V5 unexpectedly acquired an unsealed operand-closure proof",
    )
    receipts = _exact_dict(body["source_receipts"], "V5 source receipts")
    _self_hashed(receipts, "receipt_sha256", "V5 source receipts")
    provider_payload = {
        "candidate_original_used_handle_ids": cited,
        "candidate_prediction": body["candidate_prediction"],
        "candidate_receipt_sha256": body["candidate_receipt_sha256"],
        "current_prediction": body["current_prediction"],
        "dated_question": body["dated_question"],
        "evidence_grounding_rows": list(evidence),
        "evidence_plane_accounting": accounting,
        "format": f"{FORMAT}-provider-input-v1",
        "global_closure_required": body["global_closure_required"],
        "response_schema": RESPONSE_SCHEMA,
        "semantic_search_commitment": commitment,
        "source_receipts": receipts,
        "typed_operator_spec": spec,
        "typed_operand_closure_proof": body["typed_operand_closure_proof"],
    }
    messages = _selector_messages(provider_payload)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        body["provider_input_sha256"] == identity_sha256(provider_payload)
        and body["messages"] == list(messages)
        and body["messages_sha256"] == identity_sha256(list(messages))
        and body["prompt_token_proxy"] == prompt_tokens
        and body["output_token_reserve"] == OUTPUT_TOKEN_RESERVE
        and body["complete_envelope_token_proxy"]
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "V5 selector prompt/envelope changed",
    )
    assert_gold_blind(plan, path="semantic_residual_v5.candidate_plan")
    return plan


def _validate_normalization_row(raw: object) -> dict[str, Any]:
    row = _exact_dict(raw, "V5 normalization row")
    body = dict(row)
    declared = body.pop("normalization_receipt_sha256", None)
    _require(
        declared == identity_sha256(body)
        and body.get("format") == NORMALIZATION_FORMAT
        and type(body.get("ordinal")) is int
        and body.get("zero_output_change") is True
        and body.get("zero_provider_calls") == 0
        and body.get("canonical_used_handle_ids") == []
        and quote_sha256(body["canonical_prediction"])
        == body["canonical_prediction_sha256"]
        and type(body.get("original_used_handle_ids")) is list
        and bool(body["original_used_handle_ids"])
        and len(set(body["original_used_handle_ids"]))
        == len(body["original_used_handle_ids"]),
        "V5 keep-current normalization row changed",
    )
    receipts = _exact_dict(body["source_receipts"], "V5 normalization sources")
    _self_hashed(receipts, "receipt_sha256", "V5 normalization sources")
    assert_gold_blind(row, path="semantic_residual_v5.normalization")
    return row


def build_preflight_payload(
    bundle: V4SourceBundle,
    candidates: Sequence[Mapping[str, Any]],
    normalizations: Sequence[Mapping[str, Any]],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    _require(
        model == DEFAULT_MODEL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "V5 Sol runtime settings changed",
    )
    require_text(gateway_url, "V5 gateway")
    candidate_rows = tuple(_validate_candidate_plan(dict(row)) for row in candidates)
    normalization_rows = tuple(
        _validate_normalization_row(dict(row)) for row in normalizations
    )
    _require(
        len(candidate_rows) == CANDIDATE_COUNT
        and len(normalization_rows) == NORMALIZATION_COUNT
        and not (
            {row["ordinal"] for row in candidate_rows}
            & {row["ordinal"] for row in normalization_rows}
        ),
        "V5 candidate/normalization partition changed",
    )
    prompts = tuple(tuple(dict(message) for message in row["messages"]) for row in candidate_rows)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    observed_max = max(row["complete_envelope_token_proxy"] for row in candidate_rows)
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == CANDIDATE_COUNT
        and observed_max <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "V5 prompts duplicated or exceeded 8k",
    )
    evidence_accounts = tuple(row["evidence_plane_accounting"] for row in candidate_rows)
    payload = {
        "candidate_count": CANDIDATE_COUNT,
        "candidate_population_sha256": identity_sha256(
            [row["candidate_receipt_sha256"] for row in candidate_rows]
        ),
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "max_evidence_metadata_serialized_token_proxy": max(
            row["metadata_serialized_token_proxy"] for row in evidence_accounts
        ),
        "max_evidence_quote_content_token_proxy": max(
            row["quote_content_token_proxy"] for row in evidence_accounts
        ),
        "max_evidence_union_serialized_token_proxy": max(
            row["full_union_serialized_token_proxy"] for row in evidence_accounts
        ),
        "max_protected_owner_serialized_token_proxy": max(
            row["protected_owner_serialized_token_proxy"] for row in evidence_accounts
        ),
        "max_residual_serialized_token_proxy": max(
            row["residual_serialized_token_proxy"] for row in evidence_accounts
        ),
        "model": model,
        "normalization_count": NORMALIZATION_COUNT,
        "normalization_population_sha256": identity_sha256(
            [row["normalization_receipt_sha256"] for row in normalization_rows]
        ),
        "normalization_rows": list(normalization_rows),
        "observed_max_complete_envelope_tokens": observed_max,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_provider_calls": 0,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_rows": list(candidate_rows),
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": CANDIDATE_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selection_population_frozen_before_provider": True,
        "v3_parent_artifact_sha256": bundle.v3_parent.sha256,
        "v4_preflight_artifact_sha256": bundle.preflight.sha256,
        "v4_replay_artifact_sha256": bundle.replay.sha256,
        "v4_run_artifact_sha256": bundle.run.sha256,
    }
    assert_gold_blind(payload, path="semantic_residual_v5.preflight")
    return payload


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise LockedSemanticResidualCandidateVerifierV5Error(
            "typed derivation decimal must be finite"
        )
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _parse_decimal(value: str) -> Decimal:
    require_text(value, "typed derivation numeric operand")
    _require(_STRICT_NUMBER_RE.fullmatch(value) is not None, "numeric operand changed")
    clean = value.strip().replace(",", "")
    clean = clean.lstrip("$€£").rstrip("%").strip()
    try:
        return Decimal(clean)
    except InvalidOperation as exc:
        raise LockedSemanticResidualCandidateVerifierV5Error(
            "numeric operand is invalid"
        ) from exc


def _parse_date(value: str) -> date:
    require_text(value, "typed derivation date operand")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.date()
    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise LockedSemanticResidualCandidateVerifierV5Error(
                "date operand is invalid"
            ) from exc


def _operand_supported(value: str, evidence: Mapping[str, Any]) -> bool:
    quote = str(evidence["quote"])
    if value.casefold() in quote.casefold():
        return True
    if value in evidence["event_dates"]:
        return True
    created = evidence["created_at"]
    return bool(
        isinstance(created, str)
        and (value == created or value == created[:10])
    )


def _verified_derivation(
    raw: object,
    *,
    used_handles: Sequence[str],
    evidence_by_handle: Mapping[str, Mapping[str, Any]],
    include_proposed: bool,
    candidate_prediction: str,
    operator_spec: Mapping[str, Any],
) -> dict[str, Any]:
    value = _exact_dict(raw, "typed derivation")
    _require(
        set(value) == {"operation", "operands", "result", "unit"}
        and value["operation"] in _DERIVATION_OPERATIONS
        and type(value["result"]) is str
        and bool(value["result"])
        and (value["unit"] is None or type(value["unit"]) is str),
        "typed derivation schema changed",
    )
    operands: list[dict[str, str]] = []
    for raw_operand in _exact_list(value["operands"], "typed derivation operands"):
        operand = _exact_dict(raw_operand, "typed derivation operand")
        _require(
            set(operand) == {"handle_id", "value"}
            and type(operand["handle_id"]) is str
            and type(operand["value"]) is str
            and bool(operand["value"])
            and operand["handle_id"] in used_handles,
            "typed derivation operand schema changed",
        )
        evidence = evidence_by_handle.get(operand["handle_id"])
        _require(
            evidence is not None
            and evidence["role"] == "user"
            and _operand_supported(operand["value"], evidence)
            and (
                include_proposed
                or _SPECULATIVE_RE.search(str(evidence["quote"])) is None
            ),
            "typed derivation operand is not a completed user-role scalar",
        )
        operands.append(
            {"handle_id": operand["handle_id"], "value": operand["value"]}
        )
    _require(bool(operands), "typed derivation has no operands")
    operation = value["operation"]
    raw_values = [row["value"] for row in operands]
    if operation in {"sum", "difference", "greater_than"}:
        numbers = [_parse_decimal(item) for item in raw_values]
        if operation == "sum":
            computed = sum(numbers, Decimal(0))
            computed_result = _canonical_decimal(computed)
        elif operation == "difference":
            _require(len(numbers) == 2, "difference requires two ordered operands")
            computed = numbers[0] - numbers[1]
            computed_result = _canonical_decimal(computed)
        else:
            _require(len(numbers) == 2, "greater-than requires two ordered operands")
            computed_result = "true" if numbers[0] > numbers[1] else "false"
    elif operation in {"duration_days", "duration_months", "duration_years"}:
        _require(len(raw_values) == 2, "duration requires two date operands")
        start, end = sorted((_parse_date(raw_values[0]), _parse_date(raw_values[1])))
        if operation == "duration_days":
            computed_result = str((end - start).days)
        elif operation == "duration_months":
            months = (end.year - start.year) * 12 + end.month - start.month
            if end.day < start.day:
                months -= 1
            computed_result = str(months)
        else:
            years = end.year - start.year
            if (end.month, end.day) < (start.month, start.day):
                years -= 1
            computed_result = str(years)
    else:
        computed_result = str(len({item.casefold().strip() for item in raw_values}))
    _require(
        value["result"] == computed_result,
        "typed derivation result differs from local execution",
    )
    candidate_numbers = {
        _canonical_decimal(Decimal(match.replace(",", "")))
        for match in _RESULT_NUMBER_RE.findall(candidate_prediction)
    }
    result_visible = bool(
        computed_result in candidate_numbers
        or computed_result.casefold() in candidate_prediction.casefold()
        or operation == "greater_than"
        and (
            computed_result == "true"
            and re.search(r"\b(?:yes|true)\b", candidate_prediction, re.I)
            or computed_result == "false"
            and re.search(r"\b(?:no|false)\b", candidate_prediction, re.I)
        )
    )
    _require(
        result_visible,
        "typed derivation result is absent from the frozen candidate",
    )
    # Local execution proves arithmetic consistency, not retrieval closure.
    # V5 carries no independently sealed operand-to-question-slot mapping, so
    # it must not convert two observed values into a completeness proof.
    scoped = False
    body = {
        "computed_result": computed_result,
        "operation": operation,
        "operands": operands,
        "result": value["result"],
        "scoped_operand_closure_proven": scoped,
        "scoped_operand_closure_reason": (
            "operation_scope_not_closed"
        ),
        "unit": value["unit"],
    }
    return _with_receipt(body, "derivation_receipt_sha256")


def _search_trigger(
    plan: Mapping[str, Any],
    *,
    declared: bool,
    scoped_closure_proven: bool = False,
    scoped_closure_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    commitment = plan["semantic_search_commitment"]
    deterministic_open_global = bool(
        (
            plan["global_closure_required"]
            or plan["typed_operator_spec"]["requires_complete_frontier"]
        )
        and plan.get("typed_operand_closure_proof") is None
        and not scoped_closure_proven
        and (
            commitment["packing_closed"] is False
            or commitment["support_closure_proven"] is False
        )
    )
    required = bool(declared or deterministic_open_global)
    reason = (
        (
            "open_global_aggregation_frontier"
            if plan["global_closure_required"]
            else "open_required_frontier_without_scoped_operand_closure"
        )
        if deterministic_open_global
        else "selector_requested_global_search"
        if declared
        else "none"
    )
    body = {
        "candidate_receipt_sha256": plan["candidate_receipt_sha256"],
        "classified_frontier_receipt_sha256": commitment[
            "classified_frontier_receipt_sha256"
        ],
        "declared_by_selector": declared,
        "global_closure_required": plan["global_closure_required"],
        "packing_closed": commitment["packing_closed"],
        "reason": reason,
        "required": required,
        "scoped_closure_proven": scoped_closure_proven,
        "scoped_closure_receipt_sha256": scoped_closure_receipt_sha256,
        "support_closure_proven": commitment["support_closure_proven"],
        "typed_operand_closure_proof_present": (
            plan.get("typed_operand_closure_proof") is not None
        ),
        "unpacked_novel_count": commitment["unpacked_novel_count"],
        "unpacked_novel_population_sha256": commitment[
            "unpacked_novel_population_sha256"
        ],
    }
    return _with_receipt(body, "search_trigger_receipt_sha256")


def parse_selector_completion(
    completion: str,
    *,
    candidate_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly select one sealed string; every failure returns exact current."""

    plan = _validate_candidate_plan(dict(candidate_plan))
    require_text(plan["candidate_plan_receipt_sha256"], "candidate plan receipt")
    evidence = tuple(plan["evidence_grounding_rows"])
    evidence_by_handle = {row["evidence_handle"]: row for row in evidence}
    allowed = frozenset(evidence_by_handle)

    def result(
        *,
        schema_valid: bool,
        requested_selection: str,
        support_class: str,
        equivalent: bool,
        directly_answers: bool,
        personal_scope_supported: bool,
        used: Sequence[str],
        unsupported_claims: Sequence[str],
        derivation: Mapping[str, Any] | None,
        declared_search: bool,
        accepted_candidate: bool,
        reason: str,
    ) -> dict[str, Any]:
        scoped = bool(
            derivation is not None
            and derivation.get("scoped_operand_closure_proven") is True
        )
        trigger = _search_trigger(
            plan,
            declared=declared_search,
            scoped_closure_proven=scoped,
            scoped_closure_receipt_sha256=(
                None if derivation is None else derivation["derivation_receipt_sha256"]
            ),
        )
        accepted = bool(accepted_candidate and not trigger["required"])
        body = {
            "accepted_candidate": accepted,
            "candidate_plan_receipt_sha256": plan[
                "candidate_plan_receipt_sha256"
            ],
            "decision_reason": reason,
            "directly_answers": directly_answers,
            "equivalent_to_current": equivalent,
            "final_selection": "candidate" if accepted else "current",
            "format": PARSE_FORMAT,
            "personal_scope_supported": personal_scope_supported,
            "requested_selection": requested_selection,
            "schema_valid": schema_valid,
            "search_trigger": trigger,
            "support_class": support_class,
            "typed_derivation_receipt": None if derivation is None else dict(derivation),
            "unsupported_claims": list(unsupported_claims),
            "used_handle_ids": list(used),
            "used_protected_owner_handle_ids": [
                handle for handle in used if handle.startswith("P")
            ],
            "used_residual_handle_ids": [
                handle for handle in used if handle.startswith("R")
            ],
        }
        assert_gold_blind(body, path="semantic_residual_v5.parse")
        return _with_receipt(body, "parse_receipt_sha256")

    fallback = dict(
        schema_valid=False,
        requested_selection="current",
        support_class="unsupported",
        equivalent=False,
        directly_answers=False,
        personal_scope_supported=False,
        used=(),
        unsupported_claims=(),
        derivation=None,
        declared_search=False,
        accepted_candidate=False,
    )
    try:
        raw = json.loads(
            completion,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return result(**fallback, reason="malformed_json")
    if type(raw) is not dict or set(raw) != {
        "directly_answers",
        "equivalent_to_current",
        "needs_global_search",
        "personal_scope_supported",
        "selection",
        "support_class",
        "typed_derivation",
        "unsupported_claims",
        "used_handle_ids",
    }:
        return result(**fallback, reason="root_schema")
    if not (
        raw["selection"] in _SELECTIONS
        and raw["support_class"] in _SUPPORT_CLASSES
        and type(raw["directly_answers"]) is bool
        and type(raw["equivalent_to_current"]) is bool
        and type(raw["needs_global_search"]) is bool
        and type(raw["personal_scope_supported"]) is bool
        and type(raw["used_handle_ids"]) is list
        and all(type(handle) is str for handle in raw["used_handle_ids"])
        and len(set(raw["used_handle_ids"])) == len(raw["used_handle_ids"])
        and set(raw["used_handle_ids"]) <= allowed
        and type(raw["unsupported_claims"]) is list
        and all(
            type(claim) is str and bool(claim) and claim.strip() == claim
            for claim in raw["unsupported_claims"]
        )
    ):
        return result(**fallback, reason="value_schema")
    selection = raw["selection"]
    support = raw["support_class"]
    used = tuple(raw["used_handle_ids"])
    claims = tuple(raw["unsupported_claims"])
    equivalent = raw["equivalent_to_current"]
    directly = raw["directly_answers"]
    personal = raw["personal_scope_supported"]
    declared_search = raw["needs_global_search"]

    derivation: dict[str, Any] | None = None
    if support == "derived":
        try:
            derivation = _verified_derivation(
                raw["typed_derivation"],
                used_handles=used,
                evidence_by_handle=evidence_by_handle,
                include_proposed=bool(
                    plan["typed_operator_spec"]["include_proposed"]
                ),
                candidate_prediction=plan["candidate_prediction"],
                operator_spec=plan["typed_operator_spec"],
            )
        except (MatchedEvalContractError, TypeError, ValueError):
            return result(
                schema_valid=True,
                requested_selection=selection,
                support_class=support,
                equivalent=equivalent,
                directly_answers=directly,
                personal_scope_supported=personal,
                used=used,
                unsupported_claims=claims,
                derivation=None,
                declared_search=declared_search,
                accepted_candidate=False,
                reason="typed_derivation_invalid",
            )
    elif raw["typed_derivation"] is not None:
        return result(
            schema_valid=True,
            requested_selection=selection,
            support_class=support,
            equivalent=equivalent,
            directly_answers=directly,
            personal_scope_supported=personal,
            used=used,
            unsupported_claims=claims,
            derivation=None,
            declared_search=declared_search,
            accepted_candidate=False,
            reason="unexpected_typed_derivation",
        )

    cited_text = "\n".join(evidence_by_handle[handle]["quote"] for handle in used)
    uncited_numbers = v4._numeric_anchors(plan["candidate_prediction"]) - v4._numeric_anchors(  # noqa: SLF001
        cited_text
    )
    personal_candidate = bool(_PERSONAL_RE.search(plan["candidate_prediction"]))
    cited_user = tuple(
        evidence_by_handle[handle]
        for handle in used
        if evidence_by_handle[handle]["role"] == "user"
    )
    personalized = bool(
        personal_candidate
        or plan["typed_operator_spec"]["personalization_required"]
        or support == "recommendation"
    )
    preference_supported = any(
        _PREFERENCE_RE.search(str(row["quote"])) is not None for row in cited_user
    )
    accepted = bool(
        selection == "candidate"
        and support != "unsupported"
        and directly
        and not equivalent
        and not claims
        and any(_RESIDUAL_HANDLE_RE.fullmatch(handle) for handle in used)
        and (not uncited_numbers or support == "derived" and derivation is not None)
        and (not personalized or personal and bool(cited_user))
        and (
            support != "recommendation"
            and not plan["typed_operator_spec"]["personalization_required"]
            or preference_supported
        )
    )
    if declared_search:
        reason = "selector_requested_global_search"
    elif equivalent:
        reason = "equivalent_canonical_current"
    elif selection == "current":
        reason = "selector_current"
    elif claims:
        reason = "unsupported_claims"
    elif not directly:
        reason = "does_not_directly_answer"
    elif not any(_RESIDUAL_HANDLE_RE.fullmatch(handle) for handle in used):
        reason = "candidate_requires_residual_handle"
    elif uncited_numbers and derivation is None:
        reason = "uncited_scalar_without_derivation"
    elif personalized and (not personal or not cited_user):
        reason = "personal_scope_not_user_supported"
    elif (
        support == "recommendation"
        or plan["typed_operator_spec"]["personalization_required"]
    ) and not preference_supported:
        reason = "personal_preference_or_intent_not_supported"
    else:
        reason = "candidate_supported" if accepted else "candidate_rejected"
    return result(
        schema_valid=True,
        requested_selection=selection,
        support_class=support,
        equivalent=equivalent,
        directly_answers=directly,
        personal_scope_supported=personal,
        used=used,
        unsupported_claims=claims,
        derivation=derivation,
        declared_search=declared_search,
        accepted_candidate=accepted,
        reason=reason,
    )


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    payload = artifact.payload
    assert_gold_blind(payload, path="semantic_residual_v5.loaded_preflight")
    candidates = tuple(
        _validate_candidate_plan(row)
        for row in _exact_list(payload.get("prompt_rows"), "V5 prompt rows")
    )
    normalizations = tuple(
        _validate_normalization_row(row)
        for row in _exact_list(
            payload.get("normalization_rows"), "V5 normalization rows"
        )
    )
    _require(
        payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("candidate_count")
        == payload.get("required_authorized_provider_calls")
        == len(candidates)
        == CANDIDATE_COUNT
        and payload.get("normalization_count")
        == len(normalizations)
        == NORMALIZATION_COUNT
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("selection_population_frozen_before_provider") is True
        and payload.get("v4_preflight_artifact_sha256")
        == LOCKED_V4_PREFLIGHT_SHA256
        and payload.get("v4_run_artifact_sha256")
        == payload.get("v4_replay_artifact_sha256")
        == LOCKED_V4_RUN_REPLAY_SHA256
        and payload.get("v3_parent_artifact_sha256") == LOCKED_V3_PARENT_SHA256
        and payload.get("candidate_population_sha256")
        == identity_sha256(
            [row["candidate_receipt_sha256"] for row in candidates]
        )
        and payload.get("normalization_population_sha256")
        == identity_sha256(
            [row["normalization_receipt_sha256"] for row in normalizations]
        ),
        "sealed V5 preflight changed",
    )
    prompts = tuple(
        tuple(dict(message) for message in row["messages"]) for row in candidates
    )
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    accounts = tuple(row["evidence_plane_accounting"] for row in candidates)
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.logical_prompt_count
        == population.unique_prompt_count
        == CANDIDATE_COUNT
        and payload.get("observed_max_complete_envelope_tokens")
        == max(row["complete_envelope_token_proxy"] for row in candidates)
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("max_residual_serialized_token_proxy")
        == max(row["residual_serialized_token_proxy"] for row in accounts)
        <= RESIDUAL_EVIDENCE_TOKEN_CAP
        and payload.get("max_protected_owner_serialized_token_proxy")
        == max(row["protected_owner_serialized_token_proxy"] for row in accounts)
        <= PROTECTED_OWNER_TOKEN_CAP
        and payload.get("max_evidence_union_serialized_token_proxy")
        == max(row["full_union_serialized_token_proxy"] for row in accounts)
        and payload.get("max_evidence_quote_content_token_proxy")
        == max(row["quote_content_token_proxy"] for row in accounts)
        and payload.get("max_evidence_metadata_serialized_token_proxy")
        == max(row["metadata_serialized_token_proxy"] for row in accounts),
        "sealed V5 prompt population/accounting changed",
    )
    return prompts, candidates, normalizations


def _read_preflight(
    output_root: Path, expected_sha256: str
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "V5 preflight"),
        "V5 preflight SHA-256 changed",
    )
    prompts, candidates, normalizations = _validate_preflight(artifact)
    return artifact, prompts, candidates, normalizations


def _load_from_args(args: argparse.Namespace) -> V4SourceBundle:
    return load_authenticated_v4_sources(
        v4_root=Path(args.v4_root),
        v3_parent_path=Path(args.v3_parent),
        expected_v4_preflight_sha256=str(args.expected_v4_preflight_sha256),
        expected_v4_run_sha256=str(args.expected_v4_run_sha256),
        expected_v4_replay_sha256=str(args.expected_v4_replay_sha256),
        expected_v3_parent_sha256=str(args.expected_v3_parent_sha256),
    )


def _assert_preflight_source_binding(
    preflight: SealedArtifact,
    bundle: V4SourceBundle,
    candidates: Sequence[Mapping[str, Any]],
    normalizations: Sequence[Mapping[str, Any]],
) -> None:
    rebuilt_candidates, rebuilt_normalizations = freeze_v5_population(bundle)
    _require(
        tuple(candidates) == rebuilt_candidates
        and tuple(normalizations) == rebuilt_normalizations
        and preflight.payload.get("v4_preflight_artifact_sha256")
        == bundle.preflight.sha256
        and preflight.payload.get("v4_run_artifact_sha256") == bundle.run.sha256
        and preflight.payload.get("v4_replay_artifact_sha256")
        == bundle.replay.sha256
        and preflight.payload.get("v3_parent_artifact_sha256")
        == bundle.v3_parent.sha256,
        "V5 preflight differs from authenticated V4 sources",
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "V5 preflight requires a fresh absent checkpoint root",
    )
    bundle = _load_from_args(args)
    candidates, normalizations = freeze_v5_population(bundle)
    payload = build_preflight_payload(
        bundle,
        candidates,
        normalizations,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "candidate_count": CANDIDATE_COUNT,
        "created": created,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "normalization_count": NORMALIZATION_COUNT,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": CANDIDATE_COUNT,
    }


def _runtime(
    preflight: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        str(args.model) == preflight.payload.get("model") == DEFAULT_MODEL
        and str(args.gateway_url) == preflight.payload.get("gateway_url")
        and int(args.max_concurrency) == preflight.payload.get("max_concurrency")
        and len(prompts) == CANDIDATE_COUNT
        and preflight.payload.get("required_authorized_provider_calls")
        == CANDIDATE_COUNT,
        "V5 runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=int(args.max_concurrency),
        retries=0,
        benchmark_provenance={
            "arm": "locked_semantic_residual_sol_selector_v5",
            "authorized_unique_calls": CANDIDATE_COUNT,
            "experiment_format": FORMAT,
            "gateway_url": str(args.gateway_url),
            "preflight_artifact_sha256": preflight.sha256,
            "v3_parent_artifact_sha256": preflight.payload[
                "v3_parent_artifact_sha256"
            ],
            "v4_run_artifact_sha256": preflight.payload[
                "v4_run_artifact_sha256"
            ],
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(preflight, prompts, args=args, client=client)
    try:
        return runtime.run()
    finally:
        runtime.close()


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, candidates, normalizations = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    bundle = _load_from_args(args)
    _assert_preflight_source_binding(
        preflight, bundle, candidates, normalizations
    )
    checkpoint_root = Path(args.output_root) / CHECKPOINT_DIR_NAME
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == CANDIDATE_COUNT,
        "V5 provider-run requires exact authorization for 15 Sol calls",
    )
    _require(
        not checkpoint_root.exists(),
        "V5 provider-run requires its fresh dedicated checkpoint root",
    )
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))  # noqa: SLF001
    try:
        batch = _checkpoint_batch(preflight, prompts, args=args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.physical_calls
        == CANDIDATE_COUNT
        and batch.usage.checkpoint_hits == 0,
        "V5 provider population changed",
    )
    return {
        "checkpoint_hits": 0,
        "physical_provider_calls": CANDIDATE_COUNT,
        "preflight_sha256": preflight.sha256,
        "required_authorized_provider_calls": CANDIDATE_COUNT,
    }


def _result_row(
    parent_v4: Mapping[str, Any],
    parent_v3: Mapping[str, Any],
    *,
    prediction: str,
    prediction_source: str,
    decision: str,
    candidate: Mapping[str, Any] | None = None,
    normalization: Mapping[str, Any] | None = None,
    parsed: Mapping[str, Any] | None = None,
    record: Any | None = None,
) -> dict[str, Any]:
    body = {
        "call_key_sha256": None if record is None else record.call_key_sha256,
        "candidate_original_used_handle_ids": (
            [] if candidate is None else candidate["candidate_original_used_handle_ids"]
        ),
        "candidate_prediction_sha256": (
            None if candidate is None else candidate["candidate_prediction_sha256"]
        ),
        "candidate_receipt_sha256": (
            None if candidate is None else candidate["candidate_receipt_sha256"]
        ),
        "changed_from_parent": prediction != parent_v4["prediction"],
        "changed_from_v3": prediction != parent_v3["prediction"],
        "changed_from_v4": prediction != parent_v4["prediction"],
        "completion_receipt_sha256": (
            None if record is None else record.completion_sha256
        ),
        "decision": decision,
        "dated_question_sha256": parent_v4["dated_question_sha256"],
        "format": RESULT_ROW_FORMAT,
        "normalization_receipt_sha256": (
            None
            if normalization is None
            else normalization["normalization_receipt_sha256"]
        ),
        "normalized_original_used_handle_ids": (
            [] if normalization is None else normalization["original_used_handle_ids"]
        ),
        "ordinal": parent_v4["ordinal"],
        "parent_prediction_sha256": parent_v4["prediction_sha256"],
        "parent_v4_prediction_sha256": parent_v4["prediction_sha256"],
        "physical_provider_calls": 0,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": prediction_source,
        "question_id": parent_v4["question_id"],
        "question_sha256": parent_v4["question_sha256"],
        "request_journal_sha256": (
            None if record is None else record.request_journal_sha256
        ),
        "response_journal_sha256": (
            None if record is None else record.response_journal_sha256
        ),
        "retained_transformer_token_state_bytes": 0,
        "route_id": parent_v4["route_id"],
        "search_trigger": None if parsed is None else parsed["search_trigger"],
        "source_v3_row_sha256": parent_v3["source_row_sha256"],
        "source_v3_answer_row_sha256": parent_v3["source_row_sha256"],
        "source_v4_row_sha256": parent_v4["source_row_sha256"],
        "source_v4_answer_row_sha256": parent_v4["source_row_sha256"],
        "used_evidence_handle_ids": (
            [] if parsed is None else parsed["used_handle_ids"]
        ),
        "used_protected_owner_handle_ids": (
            [] if parsed is None else parsed["used_protected_owner_handle_ids"]
        ),
        "used_residual_handle_ids": (
            [] if parsed is None else parsed["used_residual_handle_ids"]
        ),
        "verifier_parse_receipt_sha256": (
            None if parsed is None else parsed["parse_receipt_sha256"]
        ),
    }
    assert_gold_blind(body, path="semantic_residual_v5.result")
    return _with_receipt(body, "source_row_sha256")


def _materialization_payload(
    preflight: SealedArtifact,
    bundle: V4SourceBundle,
    candidates: Sequence[Mapping[str, Any]],
    normalizations: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == CANDIDATE_COUNT
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == len(batch.unique_records)
        == CANDIDATE_COUNT,
        "V5 materialization requires complete checkpoints and zero calls",
    )
    by_candidate = {row["ordinal"]: row for row in candidates}
    by_normalization = {row["ordinal"]: row for row in normalizations}
    completions = {
        row["ordinal"]: completion
        for row, completion in zip(candidates, batch.logical_completions, strict=True)
    }
    records = {row.messages_sha256: row for row in batch.unique_records}
    v4_rows = tuple(bundle.run.payload["questions"])
    v3_rows = tuple(bundle.v3_parent.payload["questions"])
    results: list[dict[str, Any]] = []
    for ordinal, (parent_v4, parent_v3) in enumerate(
        zip(v4_rows, v3_rows, strict=True)
    ):
        candidate = by_candidate.get(ordinal)
        normalization = by_normalization.get(ordinal)
        if candidate is None and normalization is None:
            results.append(
                _result_row(
                    parent_v4,
                    parent_v3,
                    prediction=parent_v4["prediction"],
                    prediction_source="locked_v4_passthrough_v5",
                    decision="v4_passthrough",
                )
            )
            continue
        if normalization is not None:
            results.append(
                _result_row(
                    parent_v4,
                    parent_v3,
                    prediction=normalization["canonical_prediction"],
                    prediction_source="locked_keep_current_canonicalized_v5",
                    decision="canonical_keep_current",
                    normalization=normalization,
                )
            )
            continue
        _require(candidate is not None, "V5 candidate lookup changed")
        record = records.get(candidate["messages_sha256"])
        completion = completions[ordinal]
        _require(
            record is not None
            and record.completion == completion
            and record.checkpoint_hit is True
            and record.physical_call is False,
            f"V5 checkpoint changed at ordinal {ordinal}",
        )
        parsed = parse_selector_completion(completion, candidate_plan=candidate)
        if parsed["accepted_candidate"]:
            prediction = candidate["candidate_prediction"]
            source = "locked_sol_selected_candidate_v5"
            decision = "candidate"
        else:
            prediction = candidate["current_prediction"]
            source = (
                "locked_sol_equivalent_canonical_current_v5"
                if parsed["equivalent_to_current"]
                else "locked_sol_search_trigger_current_v5"
                if parsed["search_trigger"]["required"]
                else "locked_sol_fail_closed_current_v5"
                if not parsed["schema_valid"]
                else "locked_sol_selected_current_v5"
            )
            decision = "current"
        results.append(
            _result_row(
                parent_v4,
                parent_v3,
                prediction=prediction,
                prediction_source=source,
                decision=decision,
                candidate=candidate,
                parsed=parsed,
                record=record,
            )
        )
    _require(
        tuple(row["ordinal"] for row in results) == tuple(range(QUESTION_COUNT)),
        "V5 result population changed",
    )
    payload = {
        "candidate_count": CANDIDATE_COUNT,
        "candidate_selected_count": sum(row["decision"] == "candidate" for row in results),
        "canonical_keep_current_normalization_count": sum(
            row["decision"] == "canonical_keep_current" for row in results
        ),
        "changed_from_v3_count": sum(row["changed_from_v3"] for row in results),
        "changed_from_v4_count": sum(row["changed_from_v4"] for row in results),
        "completion_batch": v4._stable_batch(batch),  # noqa: SLF001
        "equivalent_canonical_current_count": sum(
            row["prediction_source"]
            == "locked_sol_equivalent_canonical_current_v5"
            for row in results
        ),
        "format": FORMAT,
        "gold_loaded": False,
        "judge_rows": [judge_row_projection(row) for row in results],
        "model": DEFAULT_MODEL,
        "normalization_count": NORMALIZATION_COUNT,
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "questions": results,
        "required_authorized_provider_calls": CANDIDATE_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "search_trigger_count": sum(
            bool(row["search_trigger"] and row["search_trigger"]["required"])
            for row in results
        ),
        "v3_parent_artifact_sha256": bundle.v3_parent.sha256,
        "v4_preflight_artifact_sha256": bundle.preflight.sha256,
        "v4_replay_artifact_sha256": bundle.replay.sha256,
        "v4_run_artifact_sha256": bundle.run.sha256,
    }
    assert_gold_blind(payload, path="semantic_residual_v5.materialization")
    return payload


def _verified_execution_sources(
    args: argparse.Namespace,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    V4SourceBundle,
]:
    preflight, prompts, candidates, normalizations = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    bundle = _load_from_args(args)
    _assert_preflight_source_binding(
        preflight, bundle, candidates, normalizations
    )
    return preflight, prompts, candidates, normalizations, bundle


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, candidates, normalizations, bundle = (
        _verified_execution_sources(args)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    payload = _materialization_payload(
        preflight, bundle, candidates, normalizations, batch
    )
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "candidate_selected_count": payload["candidate_selected_count"],
        "changed_from_v3_count": payload["changed_from_v3_count"],
        "changed_from_v4_count": payload["changed_from_v4_count"],
        "checkpoint_hits": CANDIDATE_COUNT,
        "created": created,
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "search_trigger_count": payload["search_trigger_count"],
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, candidates, normalizations, bundle = (
        _verified_execution_sources(args)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_payload(
        preflight, bundle, candidates, normalizations, batch
    )
    terminal = _verified(
        Path(args.output_root) / RUN_NAME,
        str(args.expected_run_sha256),
        "V5 run",
    )
    _require(
        canonical_json_bytes(rebuilt) == canonical_json_bytes(terminal.payload),
        "V5 run differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(replay.sha256 == terminal.sha256, "V5 replay changed bytes")
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
    }


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--v3-parent", type=Path, default=DEFAULT_V3_PARENT)
    parser.add_argument("--expected-v4-preflight-sha256", required=True)
    parser.add_argument("--expected-v4-run-sha256", required=True)
    parser.add_argument("--expected-v4-replay-sha256", required=True)
    parser.add_argument("--expected-v3-parent-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    _add_sources(preflight)
    provider = commands.add_parser("provider-run")
    _add_runtime(provider)
    _add_sources(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _add_runtime(materialize)
    _add_sources(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)
    replay = commands.add_parser("replay")
    _add_runtime(replay)
    _add_sources(replay)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    else:
        result = run_replay(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_COUNT",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT",
    "FORMAT",
    "LOCKED_V3_PARENT_SHA256",
    "LOCKED_V4_PREFLIGHT_SHA256",
    "LOCKED_V4_RUN_REPLAY_SHA256",
    "NORMALIZATION_COUNT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "build_parser",
    "build_preflight_payload",
    "freeze_v5_population",
    "load_authenticated_v4_sources",
    "main",
    "parse_selector_completion",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
