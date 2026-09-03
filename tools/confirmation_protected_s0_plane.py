#!/usr/bin/env python3
"""Materialize the protected confirmation S0 answer/query-input plane.

The adapter joins the authoritative cumulative S0--S3 merge, its exact S0
prompt preflight, and a complete sealed Terra completion artifact.  It makes
no provider calls.  The returned in-memory ``MatchedS0Population`` is the
exact source type consumed by ``matched_eval.query_expansion``; the sealed
artifact protects the S0 fallback answers and all provenance needed to replay
the join before the next stage is allowed to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools.confirmation_contracts import (  # noqa: E402
    SealedJson,
    publish_sealed_json,
    read_sealed_json,
)
from tools.confirmation_s0_prompt_preflight import (  # noqa: E402
    GenericMatchedS0Population,
    load_generic_matched_s0_population,
)
from tools.confirmation_cumulative_retrieval import STAGE_IDS  # noqa: E402
from tools.confirmation_terra_completion_lifecycle import (  # noqa: E402
    COMPLETION_FORMAT,
    COMPLETION_ROW_FORMAT,
    S0_PROMPT_FORMAT,
    SealedArtifact,
    VerifiedPromptArtifact,
    compile_lifecycle_preflight,
    read_sealed_artifact,
    verify_prompt_artifact,
)
from tools.matched_eval.contracts import (  # noqa: E402
    ArtifactRef,
    EvaluationMemorySnapshot,
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
)
from tools.matched_eval.live import V4_ANSWER_PLAN_ID, V4_ARM_LABEL  # noqa: E402
from tools.matched_eval.population import (  # noqa: E402
    MatchedS0Population,
    MatchedS0Row,
)
from tools.matched_eval.renderer import V4_RENDERER_ID  # noqa: E402
from tools.confirmation_canonical import (  # noqa: E402
    assert_snapshot_unchanged,
    canonical_sha256,
    exact_keys,
    require_list,
    require_mapping,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-confirmation-protected-s0-answer-plane-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
SNAPSHOT_POLICY_ID = "policy-v5-r3-protected-s0-query-input-v1"
SNAPSHOT_IMPLEMENTATION_ID = "confirmation_protected_s0_plane_v1"

_COMPLETION_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "source_prompt_artifact_sha256",
    "lifecycle_preflight_sha256",
    "provider_release_sha256",
    "runtime",
    "population",
    "ordered_rows",
    "completion_batch",
    "physical_provider_calls_during_materialization",
    "completion_artifact_identity_sha256",
}
_COMPLETION_ROW_KEYS = {
    "format",
    "row_index",
    "source_prompt_row_index",
    "question_id",
    "source_prompt_row_receipt_sha256",
    "messages_sha256",
    "completion",
    "completion_sha256",
    "call_key_sha256",
    "request_journal_sha256",
    "response_journal_sha256",
    "completion_row_receipt_sha256",
}


class ConfirmationProtectedS0Error(ValueError):
    """The protected S0 plane failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationProtectedS0Error(message)


def _self_seal(value: Mapping[str, Any], key: str, label: str) -> str:
    try:
        declared = require_sha256(value.get(key), f"{label} receipt")
    except ValueError as exc:
        raise ConfirmationProtectedS0Error(str(exc)) from exc
    body = dict(value)
    body.pop(key, None)
    _require(canonical_sha256(body) == declared, f"{label} self-seal differs")
    return declared


@dataclass(frozen=True, slots=True)
class ProtectedS0AnswerPlane:
    """A sealed protected answer plane plus its exact query-expansion input."""

    payload: dict[str, Any]
    source_population: MatchedS0Population
    predictions: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(self.payload.get("format") == FORMAT, "protected S0 format changed")
        _require(
            self.source_population.renderer_id == V4_RENDERER_ID,
            "protected S0 renderer changed",
        )
        _require(
            len(self.predictions) == self.source_population.question_count,
            "protected S0 prediction population is incomplete",
        )

    @property
    def query_expansion_source(self) -> MatchedS0Population:
        """Return the exact type accepted by ``build_query_expansion_population``."""

        return self.source_population


def _verify_s0_prompt(
    population: GenericMatchedS0Population,
    *,
    path: str | Path,
    expected_sha256: str,
) -> VerifiedPromptArtifact:
    prompt = verify_prompt_artifact(path, expected_sha256=expected_sha256)
    _require(prompt.source_format == S0_PROMPT_FORMAT, "source prompt is not S0")
    _require(
        prompt.artifact.payload == population.preflight_projection(),
        "sealed S0 prompt differs from authoritative cumulative replay",
    )
    expected_ids = tuple(row.question_id for row in population.rows)
    _require(
        prompt.source_question_ids == prompt.question_ids == expected_ids
        and prompt.source_row_indexes == tuple(range(population.question_count)),
        "sealed S0 prompt population is incomplete or reordered",
    )
    return prompt


def _verify_completion(
    *,
    path: str | Path,
    expected_sha256: str,
    prompt: VerifiedPromptArtifact,
    expected_lifecycle_preflight_sha256: str,
    expected_provider_release_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_artifact(
        path,
        expected_sha256=expected_sha256,
        label="confirmation S0 completion artifact",
    )
    value = artifact.payload
    exact_keys(value, _COMPLETION_KEYS, "confirmation S0 completion artifact")
    lifecycle_sha = require_sha256(
        expected_lifecycle_preflight_sha256,
        "expected S0 lifecycle preflight SHA-256",
    )
    release_sha = require_sha256(
        expected_provider_release_sha256,
        "expected S0 provider release SHA-256",
    )
    expected_population = {
        **compile_lifecycle_preflight(prompt)["population"],
        "question_count": len(prompt.rows),
    }
    _require(
        value["format"] == COMPLETION_FORMAT
        and value["status"] == "complete"
        and value["gold_loaded"] is False
        and value["source_prompt_artifact_sha256"] == prompt.artifact.sha256
        and value["lifecycle_preflight_sha256"] == lifecycle_sha
        and value["provider_release_sha256"] == release_sha
        and value["runtime"] == prompt.runtime
        and value["population"] == expected_population
        and value["physical_provider_calls_during_materialization"] == 0,
        "completion envelope differs from its sealed S0 lifecycle",
    )
    _self_seal(value, "completion_artifact_identity_sha256", "S0 completion artifact")
    try:
        assert_gold_blind(value, path="confirmation_protected_s0_completion")
    except MatchedEvalContractError as exc:
        raise ConfirmationProtectedS0Error(str(exc)) from exc

    raw_rows = require_list(value["ordered_rows"], "S0 completion rows")
    _require(len(raw_rows) == len(prompt.rows), "S0 completion rows are incomplete")
    rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    for index, raw in enumerate(raw_rows):
        row = require_mapping(raw, f"S0 completion row {index}")
        exact_keys(row, _COMPLETION_ROW_KEYS, f"S0 completion row {index}")
        prediction = require_text(row["completion"], f"S0 completion row {index}")
        expected_message_sha = prompt.prompt_population.ordered_rows[
            index
        ].messages_sha256
        _require(
            row["format"] == COMPLETION_ROW_FORMAT
            and row["row_index"] == index
            and row["source_prompt_row_index"] == prompt.source_row_indexes[index]
            and row["question_id"] == prompt.question_ids[index]
            and row["source_prompt_row_receipt_sha256"] == prompt.row_receipts[index]
            and row["messages_sha256"] == expected_message_sha
            and row["completion_sha256"] == quote_sha256(prediction),
            f"S0 completion row {index} differs from its prompt binding",
        )
        for key in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            require_sha256(row[key], f"S0 completion row {index} {key}")
        _self_seal(row, "completion_row_receipt_sha256", f"S0 completion row {index}")
        rows.append(dict(row))
        predictions.append(prediction)

    batch = require_mapping(value["completion_batch"], "S0 completion batch")
    _require(
        batch.get("logical_completions") == predictions
        and batch.get("prompt_population") == prompt.prompt_population.model_dump(),
        "S0 completion batch differs from its ordered completion rows",
    )
    usage = require_mapping(batch.get("usage"), "S0 completion usage")
    _require(
        usage.get("logical_calls") == len(rows)
        and usage.get("unique_calls") == prompt.prompt_population.unique_prompt_count
        and usage.get("physical_calls") == 0
        and usage.get("checkpoint_hits") == prompt.prompt_population.unique_prompt_count,
        "S0 completion batch is not a complete checkpoint-only replay",
    )
    records = require_list(batch.get("unique_records"), "S0 completion records")
    record_by_messages: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(records):
        record = require_mapping(raw_record, f"S0 completion record {index}")
        messages_sha = require_sha256(
            record.get("messages_sha256"), f"S0 completion record {index} messages"
        )
        _require(messages_sha not in record_by_messages, "S0 completion records repeat")
        record_by_messages[messages_sha] = record
    _require(
        len(record_by_messages) == prompt.prompt_population.unique_prompt_count,
        "S0 completion record population is incomplete",
    )
    for index, row in enumerate(rows):
        record = record_by_messages.get(str(row["messages_sha256"]))
        _require(
            record is not None
            and record.get("completion") == row["completion"]
            and record.get("completion_sha256") == row["completion_sha256"]
            and record.get("call_key_sha256") == row["call_key_sha256"]
            and record.get("request_journal_sha256") == row["request_journal_sha256"]
            and record.get("response_journal_sha256") == row["response_journal_sha256"]
            and record.get("checkpoint_hit") is True
            and record.get("physical_call") is False,
            f"S0 completion row {index} differs from its authenticated record",
        )
    return artifact, tuple(rows)


def _matched_population(
    population: GenericMatchedS0Population,
    *,
    completion_sha256: str,
    prompt_sha256: str,
) -> MatchedS0Population:
    snapshot = EvaluationMemorySnapshot(
        population_identity_sha256=identity_sha256(
            {
                "format": f"{FORMAT}-population-identity-v1",
                "ordered_question_ids_sha256": population.ordered_question_ids_sha256,
                "source_cumulative_retrieval_sha256": population.cumulative_retrieval_sha256,
            }
        ),
        question_order_sha256=population.ordered_question_ids_sha256,
        source_artifacts=(
            ArtifactRef(
                role="confirmation_cumulative_retrieval",
                sha256=population.cumulative_retrieval_sha256,
            ),
            ArtifactRef(role="confirmation_s0_prompt", sha256=prompt_sha256),
            ArtifactRef(role="confirmation_s0_completion", sha256=completion_sha256),
        ),
        policy_id=SNAPSHOT_POLICY_ID,
        renderer_id=V4_RENDERER_ID,
        implementation_id=SNAPSHOT_IMPLEMENTATION_ID,
        model_ids=(population.runtime.model,),
    )
    rows = tuple(
        MatchedS0Row(
            ordinal=index,
            question_part_sha256=row.source_question_receipt_sha256,
            source_stage_receipt_sha256=row.s0_stage_receipt_sha256,
            packet=row.packet,
            rendered_prompt=row.rendered_prompt,
        )
        for index, row in enumerate(population.rows)
    )
    return MatchedS0Population(
        retrieval_sha256=population.cumulative_retrieval_sha256,
        snapshot=snapshot,
        rows=rows,
        prompt_population=population.prompt_population,
        max_prompt_tokens=population.runtime.input_token_cap,
        renderer_id=V4_RENDERER_ID,
    )


def build_protected_s0_answer_plane(
    *,
    s0_prompt_path: str | Path,
    expected_s0_prompt_sha256: str,
    s0_completion_path: str | Path,
    expected_s0_completion_sha256: str,
    expected_s0_lifecycle_preflight_sha256: str,
    expected_s0_provider_release_sha256: str,
    **s0_population_inputs: Any,
) -> ProtectedS0AnswerPlane:
    """Authenticate all inputs and build the protected S0/query source plane."""

    population = load_generic_matched_s0_population(**s0_population_inputs)
    prompt = _verify_s0_prompt(
        population,
        path=s0_prompt_path,
        expected_sha256=expected_s0_prompt_sha256,
    )
    completion, completion_rows = _verify_completion(
        path=s0_completion_path,
        expected_sha256=expected_s0_completion_sha256,
        prompt=prompt,
        expected_lifecycle_preflight_sha256=(
            expected_s0_lifecycle_preflight_sha256
        ),
        expected_provider_release_sha256=expected_s0_provider_release_sha256,
    )
    matched = _matched_population(
        population,
        completion_sha256=completion.sha256,
        prompt_sha256=prompt.artifact.sha256,
    )
    rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    for index, (source, matched_row, completion_row) in enumerate(
        zip(population.rows, matched.rows, completion_rows, strict=True)
    ):
        prediction = str(completion_row["completion"])
        body = {
            "format": ROW_FORMAT,
            "row_index": index,
            "question_id": source.question_id,
            "question_sha256": source.packet.question_sha256,
            "dated_question": source.packet.dated_question,
            "dated_question_sha256": source.packet.dated_question_sha256,
            "namespace_id": source.namespace_id,
            "namespace_store_id": source.namespace_store_id,
            "namespace_checkpoint_sha256": source.namespace_checkpoint_sha256,
            "source_question_receipt_sha256": source.source_question_receipt_sha256,
            "s0_stage_receipt_sha256": source.s0_stage_receipt_sha256,
            "final_cumulative_stage_receipt_sha256": source.final_stage_receipt_sha256,
            "packet": asdict(source.packet),
            "packet_id": matched_row.packet.packet_id,
            "prompt_id": matched_row.rendered_prompt.prompt_id,
            "messages_sha256": matched_row.rendered_prompt.messages_sha256,
            "source_prompt_row_receipt_sha256": completion_row[
                "source_prompt_row_receipt_sha256"
            ],
            "completion_row_receipt_sha256": completion_row[
                "completion_row_receipt_sha256"
            ],
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "parent_disposition": "protected_s0_completion",
        }
        rows.append({**body, "row_receipt_sha256": canonical_sha256(body)})
        predictions.append(prediction)

    body = {
        "format": FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "bindings": {
            "policy_manifest_sha256": population.policy_manifest_sha256,
            "treatment_file_sha256": population.treatment_file_sha256,
            "treatment_preflight_sha256": population.treatment_preflight_sha256,
            "cumulative_retrieval_sha256": population.cumulative_retrieval_sha256,
            "s0_prompt_sha256": prompt.artifact.sha256,
            "s0_lifecycle_preflight_sha256": require_sha256(
                expected_s0_lifecycle_preflight_sha256,
                "S0 lifecycle preflight SHA-256",
            ),
            "s0_provider_release_sha256": require_sha256(
                expected_s0_provider_release_sha256,
                "S0 provider release SHA-256",
            ),
            "s0_completion_sha256": completion.sha256,
        },
        "answer_profile": {
            "arm_label": V4_ARM_LABEL,
            "answer_plan_id": V4_ANSWER_PLAN_ID,
            "renderer_id": V4_RENDERER_ID,
            "model": population.runtime.model,
        },
        "population": {
            "question_count": population.question_count,
            "ordered_question_ids_sha256": population.ordered_question_ids_sha256,
            "matched_s0_population_id": matched.population_id,
            "matched_s0_snapshot": matched.snapshot.projection(),
            "matched_s0_snapshot_id": matched.snapshot.snapshot_id,
        },
        "cumulative_stage_ids": list(STAGE_IDS),
        "ordered_rows": rows,
        "protected_parent_population_sha256": canonical_sha256(
            [row["row_receipt_sha256"] for row in rows]
        ),
        "query_expansion_source_population_id": matched.population_id,
        "physical_provider_calls_during_materialization": 0,
    }
    body = json.loads(
        json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    try:
        assert_gold_blind(body, path="confirmation_protected_s0_plane")
    except MatchedEvalContractError as exc:
        raise ConfirmationProtectedS0Error(str(exc)) from exc
    payload = {**body, "artifact_identity_sha256": canonical_sha256(body)}
    return ProtectedS0AnswerPlane(payload, matched, tuple(predictions))


def publish_protected_s0_answer_plane(
    output_path: str | Path,
    **kwargs: Any,
) -> tuple[SealedJson, bool, ProtectedS0AnswerPlane]:
    plane = build_protected_s0_answer_plane(**kwargs)
    artifact, created = publish_sealed_json(output_path, plane.payload)
    return artifact, created, plane


def replay_protected_s0_answer_plane(
    *,
    source_plane_path: str | Path,
    expected_source_plane_sha256: str,
    replay_output_path: str | Path,
    **kwargs: Any,
) -> tuple[SealedJson, bool, ProtectedS0AnswerPlane]:
    source = read_sealed_json(
        source_plane_path,
        expected_sha256=expected_source_plane_sha256,
        label="protected S0 answer plane",
    )
    plane = build_protected_s0_answer_plane(**kwargs)
    _require(source.payload == plane.payload, "protected S0 replay differs")
    replay, created = publish_sealed_json(replay_output_path, plane.payload)
    _require(replay.sha256 == source.sha256, "protected S0 replay seal differs")
    assert_snapshot_unchanged(source.snapshot, "protected S0 answer plane")
    assert_snapshot_unchanged(source.sidecar, "protected S0 answer plane sidecar")
    return replay, created, plane


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-policy", type=Path, required=True)
    parser.add_argument("--expected-runtime-policy-sha256", required=True)
    parser.add_argument("--treatment-input", type=Path, required=True)
    parser.add_argument("--expected-treatment-input-sha256", required=True)
    parser.add_argument("--treatment-preflight", type=Path, required=True)
    parser.add_argument("--expected-treatment-preflight-sha256", required=True)
    parser.add_argument("--cumulative-retrieval", type=Path, required=True)
    parser.add_argument("--expected-cumulative-retrieval-sha256", required=True)
    parser.add_argument("--s0-prompt", type=Path, required=True)
    parser.add_argument("--expected-s0-prompt-sha256", required=True)
    parser.add_argument("--s0-completion", type=Path, required=True)
    parser.add_argument("--expected-s0-completion-sha256", required=True)
    parser.add_argument("--expected-s0-lifecycle-preflight-sha256", required=True)
    parser.add_argument("--expected-s0-provider-release-sha256", required=True)


def _kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime_policy_path": args.runtime_policy,
        "expected_runtime_policy_sha256": args.expected_runtime_policy_sha256,
        "treatment_input_path": args.treatment_input,
        "expected_treatment_input_sha256": args.expected_treatment_input_sha256,
        "treatment_preflight_path": args.treatment_preflight,
        "expected_treatment_preflight_sha256": args.expected_treatment_preflight_sha256,
        "cumulative_retrieval_path": args.cumulative_retrieval,
        "expected_cumulative_retrieval_sha256": args.expected_cumulative_retrieval_sha256,
        "s0_prompt_path": args.s0_prompt,
        "expected_s0_prompt_sha256": args.expected_s0_prompt_sha256,
        "s0_completion_path": args.s0_completion,
        "expected_s0_completion_sha256": args.expected_s0_completion_sha256,
        "expected_s0_lifecycle_preflight_sha256": (
            args.expected_s0_lifecycle_preflight_sha256
        ),
        "expected_s0_provider_release_sha256": (
            args.expected_s0_provider_release_sha256
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    _add_inputs(compile_parser)
    compile_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    _add_inputs(replay_parser)
    replay_parser.add_argument("--source-plane", type=Path, required=True)
    replay_parser.add_argument("--expected-source-plane-sha256", required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "compile":
        artifact, created, plane = publish_protected_s0_answer_plane(
            args.output, **_kwargs(args)
        )
    elif args.command == "replay":
        artifact, created, plane = replay_protected_s0_answer_plane(
            source_plane_path=args.source_plane,
            expected_source_plane_sha256=args.expected_source_plane_sha256,
            replay_output_path=args.output,
            **_kwargs(args),
        )
    else:  # pragma: no cover
        raise ConfirmationProtectedS0Error("unknown command")
    return {
        "artifact_sha256": artifact.sha256,
        "created": created,
        "question_count": plane.source_population.question_count,
        "query_expansion_source_population_id": plane.source_population.population_id,
        "physical_provider_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (ConfirmationProtectedS0Error, MatchedEvalContractError, ValueError) as exc:
        print(f"protected S0 plane failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAT",
    "ROW_FORMAT",
    "ConfirmationProtectedS0Error",
    "ProtectedS0AnswerPlane",
    "build_parser",
    "build_protected_s0_answer_plane",
    "main",
    "publish_protected_s0_answer_plane",
    "replay_protected_s0_answer_plane",
]
