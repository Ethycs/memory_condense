#!/usr/bin/env python3
"""Materialize and replay the sealed confirmation prediction plane.

This provider-free adapter projects one complete, ordered final-answer source
into the exact ``memory-condense-confirmation-predictions-v1`` artifact accepted
by the post-prediction gold gate.  The source format is deliberately narrow and
population-neutral.  Every row binds its prediction to an upstream source-row
receipt and a self-sealed policy-decision receipt, including explicit fallback
disposition.

The adapter accepts no benchmark path, scorer labels, provider client, or
provider execution switch.  Materialization and replay are canonical,
sidecar-sealed, and no-clobber.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools.confirmation_contracts import (  # noqa: E402
    PREDICTIONS_FORMAT,
    RuntimePolicy,
    SealedJson,
    _decode_treatment,
    _verify_preflight,
    publish_sealed_json,
    read_runtime_policy,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.confirmation_canonical import (  # noqa: E402
    assert_snapshot_unchanged,
    canonical_sha256,
    exact_keys,
    require_int,
    require_list,
    require_mapping,
)


FINAL_ANSWER_SOURCE_FORMAT = "memory-condense-confirmation-final-answer-source-v1"
FINAL_ANSWER_ROW_FORMAT = f"{FINAL_ANSWER_SOURCE_FORMAT}-row-v1"
POLICY_DECISION_FORMAT = f"{FINAL_ANSWER_SOURCE_FORMAT}-policy-decision-v1"

_SOURCE_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "policy_manifest_sha256",
    "treatment_file_sha256",
    "treatment_preflight_sha256",
    "question_count",
    "ordered_question_ids_sha256",
    "rows",
    "artifact_identity_sha256",
}
_ROW_KEYS = {
    "format",
    "question_id",
    "prediction",
    "prediction_sha256",
    "source_row_receipt_sha256",
    "policy_decision_receipt",
    "row_receipt_sha256",
}
_DECISION_KEYS = {
    "format",
    "question_id_sha256",
    "source_row_receipt_sha256",
    "selected_source_kind",
    "fallback_used",
    "fallback_reason",
    "receipt_sha256",
}


class ConfirmationPredictionPlaneError(ValueError):
    """The final-answer source or prediction publication failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationPredictionPlaneError(message)


def _text(value: object, label: str) -> str:
    try:
        return require_text(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationPredictionPlaneError(str(exc)) from exc


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationPredictionPlaneError(str(exc)) from exc


def _self_sealed(value: Mapping[str, Any], *, key: str, label: str) -> str:
    digest = _sha(value.get(key), f"{label} {key}")
    body = dict(value)
    body.pop(key, None)
    _require(identity_sha256(body) == digest, f"{label} self-seal differs")
    return digest


@dataclass(frozen=True, slots=True)
class VerifiedFinalAnswerRow:
    question_id: str
    prediction: str
    source_row_receipt_sha256: str
    policy_decision_receipt_sha256: str
    fallback_used: bool

    def __post_init__(self) -> None:
        _text(self.question_id, "final-answer question ID")
        _text(self.prediction, "final-answer prediction")
        _sha(self.source_row_receipt_sha256, "source row receipt SHA-256")
        _sha(
            self.policy_decision_receipt_sha256,
            "policy decision receipt SHA-256",
        )
        _require(type(self.fallback_used) is bool, "fallback disposition must be boolean")


@dataclass(frozen=True, slots=True)
class VerifiedFinalAnswerSource:
    artifact: SealedJson
    policy: RuntimePolicy
    treatment: SealedJson
    preflight: SealedJson
    ordered_question_ids_sha256: str
    rows: tuple[VerifiedFinalAnswerRow, ...]

    @property
    def question_count(self) -> int:
        return len(self.rows)

    @property
    def fallback_count(self) -> int:
        return sum(row.fallback_used for row in self.rows)

    @property
    def fallback_policy_decisions_sha256(self) -> str:
        return identity_sha256(
            [
                row.policy_decision_receipt_sha256
                for row in self.rows
                if row.fallback_used
            ]
        )


@dataclass(frozen=True, slots=True)
class PredictionPlanePublication:
    artifact: SealedJson
    created: bool
    question_count: int
    fallback_count: int
    fallback_policy_decisions_sha256: str


def _decode_decision(
    value: object,
    *,
    row_index: int,
    question_id: str,
    source_row_receipt_sha256: str,
) -> tuple[str, bool]:
    label = f"final-answer row {row_index} policy decision"
    decision = require_mapping(value, label)
    exact_keys(decision, _DECISION_KEYS, label)
    _require(decision["format"] == POLICY_DECISION_FORMAT, f"{label} format changed")
    _require(
        _sha(decision["question_id_sha256"], f"{label} question ID")
        == canonical_sha256({"question_id": question_id}),
        f"{label} question binding differs",
    )
    _require(
        _sha(decision["source_row_receipt_sha256"], f"{label} source row")
        == source_row_receipt_sha256,
        f"{label} source-row binding differs",
    )
    _text(decision["selected_source_kind"], f"{label} selected source kind")
    fallback_used = decision["fallback_used"]
    _require(type(fallback_used) is bool, f"{label} fallback flag must be boolean")
    fallback_reason = decision["fallback_reason"]
    if fallback_used:
        _text(fallback_reason, f"{label} fallback reason")
    else:
        _require(fallback_reason is None, f"{label} has a reason without a fallback")
    receipt = _self_sealed(decision, key="receipt_sha256", label=label)
    return receipt, fallback_used


def _decode_source_row(
    value: object,
    *,
    row_index: int,
    expected_question_id: str,
) -> VerifiedFinalAnswerRow:
    label = f"final-answer row {row_index}"
    row = require_mapping(value, label)
    exact_keys(row, _ROW_KEYS, label)
    _require(row["format"] == FINAL_ANSWER_ROW_FORMAT, f"{label} format changed")
    question_id = _text(row["question_id"], f"{label} question ID")
    _require(question_id == expected_question_id, f"{label} is missing or reordered")
    prediction = _text(row["prediction"], f"{label} prediction")
    _require(
        _sha(row["prediction_sha256"], f"{label} prediction")
        == quote_sha256(prediction),
        f"{label} prediction identity differs",
    )
    source_receipt = _sha(
        row["source_row_receipt_sha256"],
        f"{label} source row receipt",
    )
    decision_receipt, fallback_used = _decode_decision(
        row["policy_decision_receipt"],
        row_index=row_index,
        question_id=question_id,
        source_row_receipt_sha256=source_receipt,
    )
    _self_sealed(row, key="row_receipt_sha256", label=label)
    return VerifiedFinalAnswerRow(
        question_id=question_id,
        prediction=prediction,
        source_row_receipt_sha256=source_receipt,
        policy_decision_receipt_sha256=decision_receipt,
        fallback_used=fallback_used,
    )


def load_verified_final_answer_source(
    *,
    runtime_policy_path: str | Path,
    expected_runtime_policy_sha256: str,
    treatment_input_path: str | Path,
    expected_treatment_input_sha256: str,
    treatment_preflight_path: str | Path,
    expected_treatment_preflight_sha256: str,
    final_answer_source_path: str | Path,
    expected_final_answer_source_sha256: str,
) -> VerifiedFinalAnswerSource:
    """Verify a complete arbitrary-N source without benchmark access."""

    treatment_artifact = read_sealed_json(
        treatment_input_path,
        expected_sha256=expected_treatment_input_sha256,
        label="label-free confirmation treatment",
    )
    treatment, _raw_samples = _decode_treatment(treatment_artifact)
    policy = read_runtime_policy(
        runtime_policy_path,
        expected_runtime_policy_sha256=expected_runtime_policy_sha256,
        treatment=treatment,
    )
    preflight = read_sealed_json(
        treatment_preflight_path,
        expected_sha256=expected_treatment_preflight_sha256,
        label="label-free confirmation preflight",
    )
    _verify_preflight(preflight, treatment)
    source = read_sealed_json(
        final_answer_source_path,
        expected_sha256=expected_final_answer_source_sha256,
        label="complete final-answer source",
    )
    value = source.payload
    try:
        assert_gold_blind(value, path="confirmation_final_answer_source")
    except MatchedEvalContractError as exc:
        raise ConfirmationPredictionPlaneError(str(exc)) from exc
    exact_keys(value, _SOURCE_KEYS, "complete final-answer source")
    _require(value["format"] == FINAL_ANSWER_SOURCE_FORMAT, "unsupported final-answer source format")
    _require(value["status"] == "complete", "final-answer source is incomplete")
    _require(value["gold_loaded"] is False, "final-answer source crossed the gold firewall")
    _require(value["policy_manifest_sha256"] == policy.sha256, "final-answer source binds another policy")
    _require(value["treatment_file_sha256"] == treatment_artifact.sha256, "final-answer source binds another treatment")
    _require(value["treatment_preflight_sha256"] == preflight.sha256, "final-answer source binds another preflight")
    count = require_int(value["question_count"], "final-answer source question count", minimum=1)
    question_ids = tuple(sample.sample_id for sample in treatment.samples)
    _require(count == len(question_ids), "final-answer source population is incomplete")
    order = _sha(value["ordered_question_ids_sha256"], "final-answer source ordered IDs")
    _require(order == treatment.ordered_question_ids_sha256, "final-answer source binds another order")
    _self_sealed(
        value,
        key="artifact_identity_sha256",
        label="complete final-answer source",
    )
    raw_rows = require_list(value["rows"], "final-answer source rows")
    _require(len(raw_rows) == count, "final-answer source has missing rows")
    rows = tuple(
        _decode_source_row(
            raw,
            row_index=index,
            expected_question_id=question_id,
        )
        for index, (raw, question_id) in enumerate(
            zip(raw_rows, question_ids, strict=True)
        )
    )
    observed = tuple(row.question_id for row in rows)
    _require(len(observed) == len(set(observed)), "final-answer source repeats a question ID")
    _require(identity_sha256(list(observed)) == order, "final-answer source row order differs")
    for artifact, label in (
        (policy, "frozen policy manifest"),
        (treatment_artifact, "label-free confirmation treatment"),
        (preflight, "label-free confirmation preflight"),
        (source, "complete final-answer source"),
    ):
        assert_snapshot_unchanged(artifact.snapshot, label)
        assert_snapshot_unchanged(artifact.sidecar, f"{label} digest sidecar")
    return VerifiedFinalAnswerSource(
        artifact=source,
        policy=policy,
        treatment=treatment_artifact,
        preflight=preflight,
        ordered_question_ids_sha256=order,
        rows=rows,
    )


def compile_confirmation_prediction_plane(**kwargs: Any) -> tuple[dict[str, Any], VerifiedFinalAnswerSource]:
    source = load_verified_final_answer_source(**kwargs)
    payload = {
        "format": PREDICTIONS_FORMAT,
        "status": "complete",
        "policy_manifest_sha256": source.policy.sha256,
        "treatment_file_sha256": source.treatment.sha256,
        "treatment_preflight_sha256": source.preflight.sha256,
        "sample_count": source.question_count,
        "ordered_question_ids_sha256": source.ordered_question_ids_sha256,
        "predictions": [
            {"question_id": row.question_id, "prediction": row.prediction}
            for row in source.rows
        ],
    }
    return payload, source


def materialize_confirmation_prediction_plane(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> PredictionPlanePublication:
    payload, source = compile_confirmation_prediction_plane(**kwargs)
    artifact, created = publish_sealed_json(output_path, payload)
    return PredictionPlanePublication(
        artifact=artifact,
        created=created,
        question_count=source.question_count,
        fallback_count=source.fallback_count,
        fallback_policy_decisions_sha256=(
            source.fallback_policy_decisions_sha256
        ),
    )


def replay_confirmation_prediction_plane(
    *,
    source_predictions_path: str | Path,
    expected_source_predictions_sha256: str,
    replay_output_path: str | Path,
    **kwargs: Any,
) -> PredictionPlanePublication:
    source_predictions = read_sealed_json(
        source_predictions_path,
        expected_sha256=expected_source_predictions_sha256,
        label="confirmation prediction plane",
    )
    expected, final_source = compile_confirmation_prediction_plane(**kwargs)
    _require(source_predictions.payload == expected, "confirmation prediction replay differs")
    replay, created = publish_sealed_json(replay_output_path, expected)
    _require(replay.sha256 == source_predictions.sha256, "prediction replay seal differs")
    assert_snapshot_unchanged(source_predictions.snapshot, "confirmation prediction plane")
    assert_snapshot_unchanged(
        source_predictions.sidecar,
        "confirmation prediction plane digest sidecar",
    )
    return PredictionPlanePublication(
        artifact=replay,
        created=created,
        question_count=final_source.question_count,
        fallback_count=final_source.fallback_count,
        fallback_policy_decisions_sha256=(
            final_source.fallback_policy_decisions_sha256
        ),
    )


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-policy", type=Path, required=True)
    parser.add_argument("--expected-runtime-policy-sha256", required=True)
    parser.add_argument("--treatment-input", type=Path, required=True)
    parser.add_argument("--expected-treatment-input-sha256", required=True)
    parser.add_argument("--treatment-preflight", type=Path, required=True)
    parser.add_argument("--expected-treatment-preflight-sha256", required=True)
    parser.add_argument("--final-answer-source", type=Path, required=True)
    parser.add_argument("--expected-final-answer-source-sha256", required=True)


def _input_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime_policy_path": args.runtime_policy,
        "expected_runtime_policy_sha256": args.expected_runtime_policy_sha256,
        "treatment_input_path": args.treatment_input,
        "expected_treatment_input_sha256": args.expected_treatment_input_sha256,
        "treatment_preflight_path": args.treatment_preflight,
        "expected_treatment_preflight_sha256": args.expected_treatment_preflight_sha256,
        "final_answer_source_path": args.final_answer_source,
        "expected_final_answer_source_sha256": args.expected_final_answer_source_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize", help="publish the sealed prediction plane")
    _add_inputs(materialize)
    materialize.add_argument("--output", type=Path, required=True)
    replay = subparsers.add_parser("replay", help="replay a sealed prediction plane")
    _add_inputs(replay)
    replay.add_argument("--source-predictions", type=Path, required=True)
    replay.add_argument("--expected-source-predictions-sha256", required=True)
    replay.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "materialize":
        result = materialize_confirmation_prediction_plane(
            output_path=args.output,
            **_input_kwargs(args),
        )
    elif args.command == "replay":
        result = replay_confirmation_prediction_plane(
            source_predictions_path=args.source_predictions,
            expected_source_predictions_sha256=args.expected_source_predictions_sha256,
            replay_output_path=args.output,
            **_input_kwargs(args),
        )
    else:  # pragma: no cover - argparse owns the choices.
        raise ConfirmationPredictionPlaneError("unknown command")
    return {
        "created": result.created,
        "prediction_plane_sha256": result.artifact.sha256,
        "question_count": result.question_count,
        "fallback_count": result.fallback_count,
        "fallback_policy_decisions_sha256": (
            result.fallback_policy_decisions_sha256
        ),
        "physical_provider_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (ConfirmationPredictionPlaneError, MatchedEvalContractError, ValueError) as exc:
        print(f"confirmation prediction materialization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfirmationPredictionPlaneError",
    "FINAL_ANSWER_ROW_FORMAT",
    "FINAL_ANSWER_SOURCE_FORMAT",
    "POLICY_DECISION_FORMAT",
    "PredictionPlanePublication",
    "VerifiedFinalAnswerRow",
    "VerifiedFinalAnswerSource",
    "build_parser",
    "compile_confirmation_prediction_plane",
    "load_verified_final_answer_source",
    "main",
    "materialize_confirmation_prediction_plane",
    "replay_confirmation_prediction_plane",
]
