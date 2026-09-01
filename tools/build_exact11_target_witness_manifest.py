#!/usr/bin/env python3
"""Seal the post-hoc exact-message witness set for the terminal exact-11 assay.

This artifact is evaluation-only.  It is deliberately built from the locked
benchmark's ``has_answer`` annotations *after* runtime artifacts are sealed and
must never be used by retrieval, selection, packing, or answer synthesis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.domain.integrity import file_sha256  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-exact11-target-witness-manifest-v1"
POLICY_FORMAT = f"{FORMAT}-policy-v1"
WITNESS_FORMAT = f"{FORMAT}-witness-v1"
NEGATIVE_WITNESS_FORMAT = f"{FORMAT}-negative-witness-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    r"C:\Users\Keytone\Downloads\memory-condense-rig\datasets"
    r"\longmemeval_s_cleaned.json"
)
DEFAULT_TARGET_PLAN = REPOSITORY_ROOT / (
    "docs/10 - Research Log/data/"
    "longmemeval-locked-100-target-owner-plan-v1.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "docs/10 - Research Log/data/"
    "longmemeval-exact11-target-witness-manifest-v1.json"
)
PINNED_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
PINNED_TARGET_PLAN_FILE_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)
PINNED_TARGET_PLAN_IDENTITY_SHA256 = (
    "2cabfbb103929c68dea47368502875444903ced282c708cba45ef26bee14d888"
)
EXACT_ORDINALS = (14, 28, 40, 49, 53, 54, 67, 69, 82, 94, 97)
EXPECTED_DATASET_QUESTION_COUNT = 500
EXPECTED_SOURCE_TARGET_COUNT = 26
EXPECTED_DIRECT_SOURCE_COUNT = 24
EXPECTED_LINK_ONLY_SOURCE_COUNT = 2
EXPECTED_DIRECT_WITNESS_COUNT = 29
EXPECTED_RELATION_WITNESS_COUNT = 2
EXPECTED_POSITIVE_WITNESS_COUNT = 31
EXPECTED_NEGATIVE_WITNESS_COUNT = 1


class Exact11TargetWitnessManifestError(MatchedEvalContractError):
    """The locked post-hoc witness population or one exact message changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Exact11TargetWitnessManifestError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DeclaredWitness:
    ordinal: int
    question_id: str
    target_source_id: str
    session_turn_index: int
    role: str
    content_sha256: str
    rationale: str

    def projection(self) -> dict[str, object]:
        return {
            "content_sha256": require_sha256(
                self.content_sha256, "declared witness content"
            ),
            "ordinal": self.ordinal,
            "question_id": self.question_id,
            "rationale": self.rationale,
            "role": self.role,
            "session_turn_index": self.session_turn_index,
            "target_source_id": self.target_source_id,
        }


RELATION_WITNESS_DECLARATIONS: tuple[DeclaredWitness, ...] = (
    DeclaredWitness(
        ordinal=53,
        question_id="3a704032",
        target_source_id="answer_c2204106_3",
        session_turn_index=0,
        role="user",
        content_sha256=(
            "af5b78872c00d3220eeb536df70b4f93fa2c9e5d93c784af0a817f0995000c98"
        ),
        rationale=(
            "The answer-session has no has_answer turn; this exact user turn "
            "links the acquired peace lily to its brought-home state."
        ),
    ),
    DeclaredWitness(
        ordinal=67,
        question_id="80ec1f4f",
        target_source_id="answer_990c8992_3",
        session_turn_index=4,
        role="user",
        content_sha256=(
            "a720bd59171b5017431b89e00d76ad14e9424f78ba00970f1385c3a16703e0af"
        ),
        rationale=(
            "The answer-session has no has_answer turn; this exact user turn "
            "links the opening-night event to The Art Cube gallery."
        ),
    ),
)

NEGATIVE_WITNESS_DECLARATIONS: tuple[DeclaredWitness, ...] = (
    DeclaredWitness(
        ordinal=67,
        question_id="80ec1f4f",
        target_source_id="answer_990c8992_3",
        session_turn_index=0,
        role="user",
        content_sha256=(
            "7763bc0082f3c69f650a4eb75aaf11ac13f9523633f1387f7998333d0973e066"
        ),
        rationale=(
            "January Modern Art Museum context is outside the February query "
            "window and cannot satisfy the Art Cube relation witness."
        ),
    ),
)


def _receipt_row(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(body), "witness_receipt_sha256": identity_sha256(body)}


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), "dataset must be a regular file")
    _require(
        file_sha256(path) == PINNED_DATASET_SHA256,
        "locked LongMemEval dataset SHA-256 changed",
    )
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Exact11TargetWitnessManifestError(
            "locked LongMemEval dataset is not readable strict JSON"
        ) from exc
    rows = _exact_list(raw, "LongMemEval dataset")
    _require(
        len(rows) == EXPECTED_DATASET_QUESTION_COUNT,
        "locked LongMemEval dataset population changed",
    )
    expected_keys = {
        "answer",
        "answer_session_ids",
        "haystack_dates",
        "haystack_session_ids",
        "haystack_sessions",
        "question",
        "question_date",
        "question_id",
        "question_type",
    }
    result: list[dict[str, Any]] = []
    question_ids: set[str] = set()
    for index, value in enumerate(rows):
        row = _exact_dict(value, f"dataset question {index}")
        _require(set(row) == expected_keys, "LongMemEval question schema changed")
        question_id = require_text(row.get("question_id"), "dataset question ID")
        _require(question_id not in question_ids, "dataset question IDs are not unique")
        question_ids.add(question_id)
        result.append(row)
    return result


def _load_target_plan(path: Path) -> tuple[dict[str, Any], str]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == PINNED_TARGET_PLAN_FILE_SHA256,
        "target-owner plan file SHA-256 changed",
    )
    plan = artifact.payload
    identity = require_sha256(plan.get("plan_sha256"), "target-owner plan identity")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require(
        plan.get("format") == "memory-condense-retrieval-target-owner-plan-v1"
        and identity == PINNED_TARGET_PLAN_IDENTITY_SHA256
        and identity == identity_sha256(body)
        and plan.get("question_count") == 100,
        "target-owner plan identity or population changed",
    )
    return plan, artifact.sha256


def _messages_for_source(
    dataset_row: Mapping[str, Any], source_id: str
) -> list[dict[str, Any]]:
    session_ids = _exact_list(
        dataset_row.get("haystack_session_ids"), "haystack session IDs"
    )
    sessions = _exact_list(dataset_row.get("haystack_sessions"), "haystack sessions")
    dates = _exact_list(dataset_row.get("haystack_dates"), "haystack dates")
    _require(
        len(session_ids) == len(sessions) == len(dates)
        and all(type(value) is str and bool(value) for value in session_ids),
        "haystack session population changed",
    )
    matches = [index for index, value in enumerate(session_ids) if value == source_id]
    _require(len(matches) == 1, "target source is absent or duplicated in its question")
    raw_messages = _exact_list(sessions[matches[0]], "target source messages")
    result: list[dict[str, Any]] = []
    for value in raw_messages:
        message = _exact_dict(value, "target source message")
        _require(
            set(message) == {"content", "has_answer", "role"}
            and message.get("role") in {"user", "assistant"}
            and type(message.get("content")) is str
            and bool(message["content"])
            and type(message.get("has_answer")) is bool,
            "target source message schema changed",
        )
        result.append(message)
    return result


def _positive_row(
    *,
    ordinal: int,
    question_id: str,
    source_id: str,
    witness_kind: str,
    turn_index: int,
    message: Mapping[str, Any],
) -> dict[str, Any]:
    content = require_text(message.get("content"), "witness content")
    body = {
        "content_char_count": len(content),
        "content_sha256": quote_sha256(content),
        "format": WITNESS_FORMAT,
        "has_answer": message.get("has_answer"),
        "ordinal": ordinal,
        "question_id": question_id,
        "role": message.get("role"),
        "session_turn_index": turn_index,
        "target_source_id": source_id,
        "witness_kind": witness_kind,
    }
    return _receipt_row(body)


def _negative_row(
    declaration: DeclaredWitness, message: Mapping[str, Any]
) -> dict[str, Any]:
    content = require_text(message.get("content"), "negative witness content")
    body = {
        "content_char_count": len(content),
        "content_sha256": quote_sha256(content),
        "exclusion_reason": declaration.rationale,
        "format": NEGATIVE_WITNESS_FORMAT,
        "has_answer": message.get("has_answer"),
        "ordinal": declaration.ordinal,
        "question_id": declaration.question_id,
        "role": message.get("role"),
        "session_turn_index": declaration.session_turn_index,
        "target_source_id": declaration.target_source_id,
        "witness_kind": "negative_temporal_confounder",
    }
    return _receipt_row(body)


def build_manifest(dataset_path: str | Path, target_plan_path: str | Path) -> dict[str, Any]:
    """Build the exact post-hoc witness manifest without provider calls."""

    dataset = _load_dataset(Path(dataset_path))
    target_plan, target_plan_file_sha256 = _load_target_plan(Path(target_plan_path))
    dataset_by_question = {
        require_text(row.get("question_id"), "dataset question ID"): row
        for row in dataset
    }
    desired = _exact_list(target_plan.get("desired_targets"), "desired targets")
    source_targets = [
        _exact_dict(row, "source target")
        for row in desired
        if type(row) is dict
        and row.get("target_kind") == "source_id"
        and row.get("ordinal") in EXACT_ORDINALS
    ]
    _require(
        len(source_targets) == EXPECTED_SOURCE_TARGET_COUNT
        and tuple(sorted({int(row["ordinal"]) for row in source_targets}))
        == EXACT_ORDINALS,
        "exact11 source-target population changed",
    )
    source_keys = {
        (int(row["ordinal"]), str(row["question_id"]), str(row["target_id"]))
        for row in source_targets
    }
    _require(
        len(source_keys) == len(source_targets),
        "exact11 source targets are not unique",
    )
    relation_targets = [
        _exact_dict(row, "relation target")
        for row in desired
        if type(row) is dict
        and row.get("target_kind") == "relation"
        and row.get("ordinal") in EXACT_ORDINALS
    ]

    relation_by_key = {
        (row.ordinal, row.question_id, row.target_source_id): row
        for row in RELATION_WITNESS_DECLARATIONS
    }
    negative_by_key = {
        (row.ordinal, row.question_id, row.target_source_id): row
        for row in NEGATIVE_WITNESS_DECLARATIONS
    }
    _require(
        len(relation_by_key) == EXPECTED_RELATION_WITNESS_COUNT
        and len(negative_by_key) == EXPECTED_NEGATIVE_WITNESS_COUNT
        and set(relation_by_key) <= source_keys
        and set(negative_by_key) <= source_keys,
        "declared relation/negative witness population changed",
    )

    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    direct_source_count = 0
    link_source_count = 0
    for target in source_targets:
        ordinal = target.get("ordinal")
        question_id = target.get("question_id")
        source_id = target.get("target_id")
        _require(
            type(ordinal) is int
            and type(question_id) is str
            and type(source_id) is str,
            "source target identity changed",
        )
        dataset_row = dataset_by_question.get(question_id)
        _require(dataset_row is not None, "target question is absent from dataset")
        answer_session_ids = _exact_list(
            dataset_row.get("answer_session_ids"), "answer session IDs"
        )
        question_source_ids = {
            str(row["target_id"])
            for row in source_targets
            if row.get("question_id") == question_id
        }
        _require(
            all(type(value) is str for value in answer_session_ids)
            and set(answer_session_ids) == question_source_ids,
            "target plan and dataset answer-session populations differ",
        )
        messages = _messages_for_source(dataset_row, source_id)
        answer_turns = [
            (index, message)
            for index, message in enumerate(messages)
            if message["has_answer"] is True
        ]
        key = (ordinal, question_id, source_id)
        declaration = relation_by_key.get(key)
        if answer_turns:
            _require(
                declaration is None,
                "relation-only declaration unexpectedly gained an answer turn",
            )
            direct_source_count += 1
            for turn_index, message in answer_turns:
                _require(message["role"] == "user", "answer witness role changed")
                positive.append(
                    _positive_row(
                        ordinal=ordinal,
                        question_id=question_id,
                        source_id=source_id,
                        witness_kind="answer_atom",
                        turn_index=turn_index,
                        message=message,
                    )
                )
        else:
            _require(
                declaration is not None,
                "answerless target source lacks a declared relation witness",
            )
            link_source_count += 1
            _require(
                0 <= declaration.session_turn_index < len(messages),
                "relation witness turn escaped its source",
            )
            message = messages[declaration.session_turn_index]
            _require(
                message["role"] == declaration.role
                and message["has_answer"] is False
                and quote_sha256(message["content"]) == declaration.content_sha256
                and any(
                    row.get("question_id") == question_id
                    and source_id
                    in _exact_dict(
                        row.get("assignment_basis"), "relation assignment basis"
                    ).get("expected_source_ids", [])
                    for row in relation_targets
                ),
                "declared relation witness or relation binding changed",
            )
            positive.append(
                _positive_row(
                    ordinal=ordinal,
                    question_id=question_id,
                    source_id=source_id,
                    witness_kind="relation_link",
                    turn_index=declaration.session_turn_index,
                    message=message,
                )
            )

        negative_declaration = negative_by_key.get(key)
        if negative_declaration is not None:
            _require(
                0 <= negative_declaration.session_turn_index < len(messages),
                "negative witness turn escaped its source",
            )
            message = messages[negative_declaration.session_turn_index]
            _require(
                message["role"] == negative_declaration.role
                and message["has_answer"] is False
                and quote_sha256(message["content"])
                == negative_declaration.content_sha256,
                "negative witness bytes changed",
            )
            negative.append(_negative_row(negative_declaration, message))

    direct_witness_count = sum(
        row["witness_kind"] == "answer_atom" for row in positive
    )
    relation_witness_count = sum(
        row["witness_kind"] == "relation_link" for row in positive
    )
    _require(
        direct_source_count == EXPECTED_DIRECT_SOURCE_COUNT
        and link_source_count == EXPECTED_LINK_ONLY_SOURCE_COUNT
        and direct_witness_count == EXPECTED_DIRECT_WITNESS_COUNT
        and relation_witness_count == EXPECTED_RELATION_WITNESS_COUNT
        and len(positive) == EXPECTED_POSITIVE_WITNESS_COUNT
        and len(negative) == EXPECTED_NEGATIVE_WITNESS_COUNT
        and len({row["witness_receipt_sha256"] for row in positive}) == len(positive)
        and not (
            {row["content_sha256"] for row in positive}
            & {row["content_sha256"] for row in negative}
        ),
        "exact11 direct/link/negative witness population changed",
    )

    policy_body = {
        "direct_answer_rule": (
            "include every exact target-source message whose locked "
            "has_answer field is true"
        ),
        "format": POLICY_FORMAT,
        "negative_witness_declarations": [
            row.projection() for row in NEGATIVE_WITNESS_DECLARATIONS
        ],
        "relation_witness_declarations": [
            row.projection() for row in RELATION_WITNESS_DECLARATIONS
        ],
        "relation_witness_rule": (
            "only the two explicit posthoc declarations may stand in for an "
            "answerless target source; they are never production routing"
        ),
    }
    witness_policy = {
        **policy_body,
        "receipt_sha256": identity_sha256(policy_body),
    }
    body = {
        "analysis_is_posthoc_only": True,
        "dataset_file_sha256": PINNED_DATASET_SHA256,
        "direct_answer_source_count": direct_source_count,
        "direct_answer_witness_count": direct_witness_count,
        "exact_ordinals": list(EXACT_ORDINALS),
        "format": FORMAT,
        "gold_loaded": True,
        "link_only_source_count": link_source_count,
        "negative_witness_count": len(negative),
        "negative_witnesses": negative,
        "positive_witness_count": len(positive),
        "positive_witnesses": positive,
        "provider_calls": 0,
        "relation_link_witness_count": relation_witness_count,
        "runtime_use_forbidden": True,
        "source_target_count": len(source_targets),
        "target_plan_file_sha256": target_plan_file_sha256,
        "target_plan_identity_sha256": PINNED_TARGET_PLAN_IDENTITY_SHA256,
        "target_plan_population_identity_sha256": require_sha256(
            target_plan.get("population_identity_sha256"),
            "target plan population identity",
        ),
        "witness_policy": witness_policy,
    }
    return {**body, "manifest_identity_sha256": identity_sha256(body)}


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_manifest(args.dataset, args.target_plan)
    artifact, created = publish_sealed_json(args.output, payload)
    return {
        "artifact": str(artifact.path),
        "created": created,
        "manifest_identity_sha256": payload["manifest_identity_sha256"],
        "negative_witness_count": payload["negative_witness_count"],
        "positive_witness_count": payload["positive_witness_count"],
        "provider_calls": 0,
        "sha256": artifact.sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_OUTPUT",
    "DEFAULT_TARGET_PLAN",
    "EXACT_ORDINALS",
    "FORMAT",
    "NEGATIVE_WITNESS_DECLARATIONS",
    "PINNED_DATASET_SHA256",
    "PINNED_TARGET_PLAN_FILE_SHA256",
    "PINNED_TARGET_PLAN_IDENTITY_SHA256",
    "RELATION_WITNESS_DECLARATIONS",
    "build_manifest",
    "main",
    "run_build",
]
