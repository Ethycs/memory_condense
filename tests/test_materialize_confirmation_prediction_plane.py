from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_confirmation_s0_prompt_preflight import (
    Fixture as BaseFixture,
    _build_fixture as _build_base_fixture,
)
from tools import materialize_confirmation_prediction_plane as materializer
from tools.confirmation_contracts import (
    PREDICTIONS_FORMAT,
    SealedJson,
    publish_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.v4_population_firebreak.canonical import canonical_sha256


@dataclass(frozen=True)
class Fixture:
    base: BaseFixture
    final_source: SealedJson
    fallback_indices: tuple[int, ...]

    def kwargs(self) -> dict[str, object]:
        return {
            "runtime_policy_path": self.base.policy.path,
            "expected_runtime_policy_sha256": (
                self.base.policy.runtime_policy_sha256
            ),
            "treatment_input_path": self.base.treatment.path,
            "expected_treatment_input_sha256": self.base.treatment.sha256,
            "treatment_preflight_path": self.base.treatment_preflight.path,
            "expected_treatment_preflight_sha256": (
                self.base.treatment_preflight.sha256
            ),
            "final_answer_source_path": self.final_source.path,
            "expected_final_answer_source_sha256": self.final_source.sha256,
        }


def _sealed(value: dict[str, object], key: str) -> dict[str, object]:
    return {**value, key: identity_sha256(value)}


def _source_payload(
    base: BaseFixture,
    *,
    fallback_indices: tuple[int, ...],
) -> dict[str, object]:
    question_ids = tuple(
        sample["sample_id"] for sample in base.treatment.payload["samples"]
    )
    rows = []
    for index, question_id in enumerate(question_ids):
        fallback = index in fallback_indices
        source_receipt = canonical_sha256(
            {"question_id": question_id, "source_row": index}
        )
        decision_body = {
            "format": materializer.POLICY_DECISION_FORMAT,
            "question_id_sha256": canonical_sha256(
                {"question_id": question_id}
            ),
            "source_row_receipt_sha256": source_receipt,
            "selected_source_kind": (
                "protected_parent_fallback" if fallback else "terminal_policy"
            ),
            "fallback_used": fallback,
            "fallback_reason": (
                "terminal-policy-returned-no-supported-replacement"
                if fallback
                else None
            ),
        }
        decision = _sealed(decision_body, "receipt_sha256")
        prediction = f"prediction for {question_id}"
        row_body = {
            "format": materializer.FINAL_ANSWER_ROW_FORMAT,
            "question_id": question_id,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "source_row_receipt_sha256": source_receipt,
            "policy_decision_receipt": decision,
        }
        rows.append(_sealed(row_body, "row_receipt_sha256"))
    body = {
        "format": materializer.FINAL_ANSWER_SOURCE_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "policy_manifest_sha256": base.policy.sha256,
        "treatment_file_sha256": base.treatment.sha256,
        "treatment_preflight_sha256": base.treatment_preflight.sha256,
        "question_count": len(rows),
        "ordered_question_ids_sha256": canonical_sha256(list(question_ids)),
        "rows": rows,
    }
    return _sealed(body, "artifact_identity_sha256")


def _build_fixture(
    root: Path,
    *,
    count: int,
    fallback_indices: tuple[int, ...] = (),
) -> Fixture:
    base = _build_base_fixture(
        root,
        semantics=tuple(range(count)),
        id_prefix=f"arbitrary-{count}",
        namespace_sizes=(count,),
    )
    source, _ = publish_sealed_json(
        root / "final-answer-source.json",
        _source_payload(base, fallback_indices=fallback_indices),
    )
    return Fixture(base=base, final_source=source, fallback_indices=fallback_indices)


def _mutated_source(
    fixture: Fixture,
    *,
    name: str,
    mutate: object,
) -> SealedJson:
    payload = copy.deepcopy(fixture.final_source.payload)
    mutate(payload)
    body = {
        key: value
        for key, value in payload.items()
        if key != "artifact_identity_sha256"
    }
    payload["artifact_identity_sha256"] = identity_sha256(body)
    result, _ = publish_sealed_json(fixture.base.root / name, payload)
    return result


@pytest.mark.parametrize("count", [1, 3, 7])
def test_materializes_exact_population_neutral_prediction_schema(
    tmp_path: Path,
    count: int,
) -> None:
    fixture = _build_fixture(tmp_path / f"n-{count}", count=count)
    result = materializer.materialize_confirmation_prediction_plane(
        output_path=fixture.base.root / "predictions.json",
        **fixture.kwargs(),
    )

    assert result.question_count == count
    assert result.fallback_count == 0
    assert result.artifact.payload["format"] == PREDICTIONS_FORMAT
    assert result.artifact.payload["status"] == "complete"
    assert result.artifact.payload["sample_count"] == count
    assert len(result.artifact.payload["predictions"]) == count
    assert all(
        set(row) == {"question_id", "prediction"}
        for row in result.artifact.payload["predictions"]
    )

    # A standalone materialized plane remains evaluator-ineligible until the
    # production executor seals its run manifest, 17 checkpoints, and handoff.
    assert len(result.artifact.payload["predictions"]) == count


def test_replay_is_byte_identical_and_no_clobber(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "replay", count=4)
    materialized = materializer.materialize_confirmation_prediction_plane(
        output_path=fixture.base.root / "predictions.json",
        **fixture.kwargs(),
    )
    reused = materializer.materialize_confirmation_prediction_plane(
        output_path=fixture.base.root / "predictions.json",
        **fixture.kwargs(),
    )
    replay = materializer.replay_confirmation_prediction_plane(
        source_predictions_path=materialized.artifact.path,
        expected_source_predictions_sha256=materialized.artifact.sha256,
        replay_output_path=fixture.base.root / "predictions-replay.json",
        **fixture.kwargs(),
    )
    assert materialized.created is True
    assert reused.created is False
    assert replay.created is True
    assert replay.artifact.sha256 == materialized.artifact.sha256
    assert replay.artifact.payload == materialized.artifact.payload


def test_fallback_decisions_remain_visible_in_materialization_receipt(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(
        tmp_path / "fallback",
        count=5,
        fallback_indices=(1, 4),
    )
    result = materializer.materialize_confirmation_prediction_plane(
        output_path=fixture.base.root / "predictions.json",
        **fixture.kwargs(),
    )
    fallback_receipts = [
        row["policy_decision_receipt"]["receipt_sha256"]
        for index, row in enumerate(fixture.final_source.payload["rows"])
        if index in fixture.fallback_indices
    ]
    assert result.fallback_count == 2
    assert result.fallback_policy_decisions_sha256 == identity_sha256(
        fallback_receipts
    )
    assert all(
        fixture.final_source.payload["rows"][index]["policy_decision_receipt"][
            "fallback_reason"
        ]
        for index in fixture.fallback_indices
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_missing_duplicate_or_reordered_rows_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_fixture(tmp_path / mutation, count=4)

    def mutate(payload: dict[str, object]) -> None:
        rows = payload["rows"]
        if mutation == "missing":
            payload["rows"] = rows[:-1]
        elif mutation == "duplicate":
            payload["rows"] = [rows[0], rows[0], *rows[2:]]
        else:
            payload["rows"] = list(reversed(rows))

    changed = _mutated_source(
        fixture,
        name=f"{mutation}.json",
        mutate=mutate,
    )
    kwargs = fixture.kwargs()
    kwargs["final_answer_source_path"] = changed.path
    kwargs["expected_final_answer_source_sha256"] = changed.sha256
    with pytest.raises(materializer.ConfirmationPredictionPlaneError, match="missing|reordered"):
        materializer.compile_confirmation_prediction_plane(**kwargs)


def test_tampered_source_bytes_fail_external_seal(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "tamper", count=3)
    with fixture.final_source.path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="external seal"):
        materializer.compile_confirmation_prediction_plane(**fixture.kwargs())


def test_noncanonical_prediction_and_hidden_fallback_fail(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "canonical", count=2)

    def mutate(payload: dict[str, object]) -> None:
        row = payload["rows"][0]
        row["prediction"] = f"{row['prediction']} "
        row["prediction_sha256"] = quote_sha256(row["prediction"])
        row_body = {
            key: value
            for key, value in row.items()
            if key != "row_receipt_sha256"
        }
        row["row_receipt_sha256"] = identity_sha256(row_body)

    changed = _mutated_source(
        fixture,
        name="noncanonical.json",
        mutate=mutate,
    )
    kwargs = fixture.kwargs()
    kwargs["final_answer_source_path"] = changed.path
    kwargs["expected_final_answer_source_sha256"] = changed.sha256
    with pytest.raises(materializer.ConfirmationPredictionPlaneError, match="exact text"):
        materializer.compile_confirmation_prediction_plane(**kwargs)


def _all_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    actions = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                actions.extend(_all_actions(child))
    return actions


def test_cli_has_no_provider_or_benchmark_surface() -> None:
    destinations = {action.dest for action in _all_actions(materializer.build_parser())}
    assert not destinations & {
        "api_key",
        "authorized_provider_calls",
        "dataset",
        "enable_provider",
        "execute",
        "gold",
        "provider",
        "reference",
        "retry",
    }
    source = Path(materializer.__file__).read_text(encoding="utf-8").casefold()
    assert "import litellm" not in source
    assert "import openai" not in source
