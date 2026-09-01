"""Positive full-100 comparison for authenticated common-parent score planes.

The comparator consumes a small canonical projection, never an arm-specific
report.  Each projection is backed by byte-identical judge and score
run/replay pairs and carries the sealed prediction, validation, judge, gold,
question-order, parent-origin, and exact model/budget identities needed for a
paired result.  The certified reconciliation-V3 treatment has a concrete
adapter.  The terminal-V2 adapter remains deliberately closed until that
artifact family exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    identity_sha256,
    require_sha256,
    require_text,
)

from .common_parent_contract import COMPARISON_SEMANTICS, EXACT_ACCOUNTING
from .typed_epoch_campaign import (
    JUDGE_MODEL,
    JUDGE_OUTPUT_TOKEN_RESERVE,
    PARENT_ORIGIN_FORMAT,
    PARENT_SOURCE_FORMAT,
    PARENT_SOURCE_ROLE,
)
from .typed_judge_lifecycle import (
    JUDGE_FORMAT as MEM0_JUDGE_FORMAT,
    MAX_JUDGE_PROMPT_TOKENS,
    SCORE_FORMAT as MEM0_SCORE_FORMAT,
    load_verified_judge_score,
)
from .typed_usage_attestation import (
    VerifiedMem0UsageAttestation,
    load_verified_usage_attestation,
    reopen_verified_usage_capability,
)


QUESTION_COUNT = 100
SCORE_PLANE_FORMAT = "memory-condense-common-parent-score-plane-v1"
COMPARISON_FORMAT = "memory-condense-common-parent-paired-full100-v2"
TREATMENT_SCORE_PLANE_NAME = "treatment-common-parent-score-plane-v1.json"
MEM0_SCORE_PLANE_NAME = "mem0-common-parent-score-plane-v1.json"
TREATMENT_SCORE_PLANE_REPLAY_NAME = (
    "treatment-common-parent-score-plane-replay-v1.json"
)
MEM0_SCORE_PLANE_REPLAY_NAME = "mem0-common-parent-score-plane-replay-v1.json"
COMPARISON_NAME = "common-parent-paired-full100-v2.json"
COMPARISON_REPLAY_NAME = "common-parent-paired-full100-replay-v2.json"
CERTIFIED_V3_ADAPTER = "certified_reconciliation_v3_full100_v1"
MEM0_TYPED_ADAPTER = "mem0_typed_resumable_full100_v1"
TERMINAL_V2_ADAPTER_STATUS = (
    "closed_until_terminal_v2_full100_judge_score_run_replay_exists"
)
ArmRole = Literal["treatment", "mem0"]

_PLANE_KEYS = {
    "accuracy",
    "adapter_id",
    "arm_role",
    "comparison_semantics",
    "correct",
    "exact_accounting",
    "format",
    "gold_population_sha256",
    "parent_origin_receipt_sha256",
    "question_count",
    "question_order_sha256",
    "rows",
    "source_artifacts",
    "usage_attestation_sha256",
}
_ROW_KEYS = {
    "answer_validation_receipt_sha256",
    "correct",
    "dated_question_sha256",
    "judge_row_sha256",
    "ordinal",
    "prediction_sha256",
    "prediction_source",
    "question_id",
    "question_sha256",
    "reference_sha256",
    "score_plane_row_sha256",
}
_SOURCE_KEYS = {
    "answer_replay_sha256",
    "answer_run_sha256",
    "judge_replay_sha256",
    "judge_run_sha256",
    "preflight_sha256",
    "score_replay_sha256",
    "score_run_sha256",
}


class CommonParentComparisonError(MatchedEvalContractError):
    """A score plane, source replay, parent identity, or paired result changed."""


_VERIFIED_SCORE_PLANE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedScorePlane:
    """Capability returned only after an arm-specific strict source rebuild."""

    artifact: SealedArtifact
    replay: SealedArtifact
    rows: tuple[dict[str, Any], ...]

    def __init__(
        self,
        artifact: SealedArtifact,
        replay: SealedArtifact,
        rows: tuple[dict[str, Any], ...],
        *,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_SCORE_PLANE_TOKEN:
            raise CommonParentComparisonError(
                "score-plane capability requires an arm-specific strict reader"
            )
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "replay", replay)
        object.__setattr__(self, "rows", rows)


def _reopen_verified_score_plane(
    capability: VerifiedScorePlane,
    *,
    expected_arm_role: ArmRole,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    """Reopen both files; frozen containers do not make nested dicts immutable."""

    _require(
        type(capability) is VerifiedScorePlane,
        "score-plane authority is not a strict reader capability",
    )
    artifact = _read(
        capability.artifact.path,
        capability.artifact.sha256,
        f"{expected_arm_role} score-plane capability",
    )
    replay = _read(
        capability.replay.path,
        capability.replay.sha256,
        f"{expected_arm_role} score-plane capability replay",
    )
    rows = validate_score_plane_artifact(
        artifact, expected_arm_role=expected_arm_role
    )
    _require(
        artifact.sha256 == replay.sha256
        and artifact.payload == replay.payload
        and artifact.payload == capability.artifact.payload
        and replay.payload == capability.replay.payload
        and rows == capability.rows,
        f"{expected_arm_role} score-plane capability changed after verification",
    )
    return artifact, replay, rows


def _require(ok: object, message: str) -> None:
    if not ok:
        raise CommonParentComparisonError(message)


def _dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _read(path: str | Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, f"expected {label}"),
        f"{label} SHA-256 changed",
    )
    return artifact


def _exact_parent_origin(
    *,
    parent_run_sha256: str,
    parent_replay_sha256: str,
) -> str:
    run = require_sha256(parent_run_sha256, "parent run")
    replay = require_sha256(parent_replay_sha256, "parent replay")
    _require(run == replay, "common parent run/replay are not byte-identical")
    return identity_sha256(
        {
            "comparison_semantics": COMPARISON_SEMANTICS,
            "format": PARENT_ORIGIN_FORMAT,
            "parent_replay_sha256": replay,
            "parent_run_sha256": run,
            "question_count": QUESTION_COUNT,
            "source_format": PARENT_SOURCE_FORMAT,
            "source_role": PARENT_SOURCE_ROLE,
        }
    )


def _canonical_row(
    *,
    answer_validation_receipt_sha256: str,
    correct: bool,
    dated_question_sha256: str,
    judge_row_sha256: str,
    ordinal: int,
    prediction_sha256: str,
    prediction_source: str,
    question_id: str,
    question_sha256: str,
    reference_sha256: str,
) -> dict[str, Any]:
    body = {
        "answer_validation_receipt_sha256": require_sha256(
            answer_validation_receipt_sha256, "answer validation receipt"
        ),
        "correct": correct,
        "dated_question_sha256": require_sha256(
            dated_question_sha256, "dated question"
        ),
        "judge_row_sha256": require_sha256(judge_row_sha256, "judge row"),
        "ordinal": ordinal,
        "prediction_sha256": require_sha256(prediction_sha256, "prediction"),
        "prediction_source": require_text(prediction_source, "prediction source"),
        "question_id": require_text(question_id, "question ID"),
        "question_sha256": require_sha256(question_sha256, "question"),
        "reference_sha256": require_sha256(reference_sha256, "reference"),
    }
    _require(type(ordinal) is int and type(correct) is bool, "score row scalar changed")
    return {**body, "score_plane_row_sha256": identity_sha256(body)}


def _build_plane(
    *,
    arm_role: ArmRole,
    adapter_id: str,
    parent_origin_receipt_sha256: str,
    gold_population_sha256: str,
    source_artifacts: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    exact_accounting: Mapping[str, Any],
    usage_attestation_sha256: str | None = None,
) -> dict[str, Any]:
    _require(
        arm_role in {"treatment", "mem0"}
        and adapter_id
        in {CERTIFIED_V3_ADAPTER, MEM0_TYPED_ADAPTER}
        and dict(exact_accounting) == EXACT_ACCOUNTING
        and len(rows) == QUESTION_COUNT,
        "canonical score-plane role, accounting, or population changed",
    )
    if arm_role == "mem0":
        usage_sha: str | None = require_sha256(
            usage_attestation_sha256, "Mem0 usage attestation"
        )
    else:
        _require(
            usage_attestation_sha256 is None,
            "treatment score plane cannot claim a Mem0 usage attestation",
        )
        usage_sha = None
    sources = dict(source_artifacts)
    _require(set(sources) == _SOURCE_KEYS, "score-plane source receipt set changed")
    for key, value in sources.items():
        require_sha256(value, f"score-plane {key}")
    _require(
        sources["answer_run_sha256"] == sources["answer_replay_sha256"]
        and sources["judge_run_sha256"] == sources["judge_replay_sha256"]
        and sources["score_run_sha256"] == sources["score_replay_sha256"],
        "score-plane source run/replay receipts differ",
    )
    canonical = [dict(row) for row in rows]
    validated = _validate_rows(canonical)
    correct = sum(row["correct"] for row in validated)
    order = [
        {
            "dated_question_sha256": row["dated_question_sha256"],
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        for row in validated
    ]
    return {
        "accuracy": correct / QUESTION_COUNT,
        "adapter_id": adapter_id,
        "arm_role": arm_role,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "correct": correct,
        "exact_accounting": dict(EXACT_ACCOUNTING),
        "format": SCORE_PLANE_FORMAT,
        "gold_population_sha256": require_sha256(
            gold_population_sha256, "gold population"
        ),
        "parent_origin_receipt_sha256": require_sha256(
            parent_origin_receipt_sha256, "parent origin"
        ),
        "question_count": QUESTION_COUNT,
        "question_order_sha256": identity_sha256(order),
        "rows": validated,
        "source_artifacts": sources,
        "usage_attestation_sha256": usage_sha,
    }


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _require(len(rows) == QUESTION_COUNT, "score plane is not full100")
    result: list[dict[str, Any]] = []
    ids: list[str] = []
    for ordinal, raw in enumerate(rows):
        row = _dict(raw, "score-plane row")
        unsigned = dict(row)
        declared = unsigned.pop("score_plane_row_sha256", None)
        _require(
            set(row) == _ROW_KEYS
            and row.get("ordinal") == ordinal
            and type(row.get("correct")) is bool
            and declared == identity_sha256(unsigned),
            f"score-plane row {ordinal} changed",
        )
        for key in (
            "answer_validation_receipt_sha256",
            "dated_question_sha256",
            "judge_row_sha256",
            "prediction_sha256",
            "question_sha256",
            "reference_sha256",
        ):
            require_sha256(row.get(key), f"score-plane {key}")
        ids.append(require_text(row.get("question_id"), "score-plane question ID"))
        require_text(row.get("prediction_source"), "score-plane prediction source")
        result.append(dict(row))
    _require(len(set(ids)) == QUESTION_COUNT, "score-plane question IDs repeat")
    return result


def validate_score_plane_artifact(
    artifact: SealedArtifact,
    *,
    expected_arm_role: ArmRole,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    rows = _list(payload.get("rows"), "score-plane rows")
    validated = _validate_rows(rows)
    sources = _dict(payload.get("source_artifacts"), "score-plane sources")
    accounting = _dict(payload.get("exact_accounting"), "score-plane accounting")
    correct = sum(row["correct"] for row in validated)
    order = [
        {
            "dated_question_sha256": row["dated_question_sha256"],
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        for row in validated
    ]
    _require(
        set(payload) == _PLANE_KEYS
        and payload.get("format") == SCORE_PLANE_FORMAT
        and payload.get("comparison_semantics") == COMPARISON_SEMANTICS
        and payload.get("arm_role") == expected_arm_role
        and payload.get("adapter_id")
        == (
            CERTIFIED_V3_ADAPTER
            if expected_arm_role == "treatment"
            else MEM0_TYPED_ADAPTER
        )
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("correct") == correct
        and payload.get("accuracy") == correct / QUESTION_COUNT
        and payload.get("question_order_sha256") == identity_sha256(order)
        and accounting == EXACT_ACCOUNTING
        and set(sources) == _SOURCE_KEYS
        and sources.get("answer_run_sha256")
        == sources.get("answer_replay_sha256")
        and sources.get("judge_run_sha256")
        == sources.get("judge_replay_sha256")
        and sources.get("score_run_sha256")
        == sources.get("score_replay_sha256"),
        "canonical score-plane envelope changed",
    )
    if expected_arm_role == "mem0":
        require_sha256(
            payload.get("usage_attestation_sha256"),
            "score-plane Mem0 usage attestation",
        )
    else:
        _require(
            payload.get("usage_attestation_sha256") is None,
            "treatment score plane claimed a Mem0 usage attestation",
        )
    require_sha256(payload.get("gold_population_sha256"), "score-plane gold")
    require_sha256(payload.get("parent_origin_receipt_sha256"), "score-plane parent")
    for key, value in sources.items():
        require_sha256(value, f"score-plane {key}")
    return tuple(validated)


def publish_score_plane(
    output_root: str | Path,
    payload: dict[str, Any],
    *,
    arm_role: ArmRole,
) -> tuple[SealedArtifact, SealedArtifact]:
    name = TREATMENT_SCORE_PLANE_NAME if arm_role == "treatment" else MEM0_SCORE_PLANE_NAME
    replay_name = (
        TREATMENT_SCORE_PLANE_REPLAY_NAME
        if arm_role == "treatment"
        else MEM0_SCORE_PLANE_REPLAY_NAME
    )
    artifact, _ = publish_sealed_json(Path(output_root) / name, payload)
    validate_score_plane_artifact(artifact, expected_arm_role=arm_role)
    replay, _ = publish_sealed_json(Path(output_root) / replay_name, payload)
    _require(artifact.sha256 == replay.sha256, "score-plane replay changed")
    return artifact, replay


def _read_published_plane(
    plane_path: str | Path,
    expected_plane_sha256: str,
    replay_path: str | Path,
    expected_replay_sha256: str,
    *,
    expected_arm_role: ArmRole,
    expected_payload: Mapping[str, Any],
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    plane = _read(plane_path, expected_plane_sha256, "score plane")
    replay = _read(replay_path, expected_replay_sha256, "score plane replay")
    _require(
        plane.sha256 == replay.sha256 and plane.payload == replay.payload,
        "score plane is not byte-identical on replay",
    )
    rows = validate_score_plane_artifact(plane, expected_arm_role=expected_arm_role)
    _require(
        plane.payload == dict(expected_payload),
        "published score plane differs from its arm-specific strict rebuild",
    )
    return plane, replay, rows


def load_verified_score_plane(*_args: object, **_kwargs: object) -> None:
    """Reject the former generic reader; callers must select a strict arm reader."""

    raise CommonParentComparisonError(
        "generic score-plane verification is forbidden; use the strict arm reader"
    )


def build_mem0_score_plane_payload(
    *,
    judge: SealedArtifact,
    judge_replay: SealedArtifact,
    score: SealedArtifact,
    score_replay: SealedArtifact,
    usage: VerifiedMem0UsageAttestation,
) -> dict[str, Any]:
    """Project a strict Mem0 reader result into the shared full100 plane."""

    usage_artifact, usage_replay, _responder_usage, _judge_usage = (
        reopen_verified_usage_capability(usage)
    )
    _require(
        judge.sha256 == judge_replay.sha256
        and judge.payload == judge_replay.payload
        and score.sha256 == score_replay.sha256
        and score.payload == score_replay.payload
        and judge.payload.get("format") == MEM0_JUDGE_FORMAT
        and score.payload.get("format") == MEM0_SCORE_FORMAT
        and judge.payload.get("question_count") == QUESTION_COUNT
        and score.payload.get("question_count") == QUESTION_COUNT
        and judge.payload.get("model_accounting") == EXACT_ACCOUNTING
        and type(usage) is VerifiedMem0UsageAttestation
        and usage_artifact.sha256 == usage_replay.sha256
        and usage_artifact.payload.get("question_count") == QUESTION_COUNT
        and usage_artifact.payload.get("common_input_sha256")
        == judge.payload.get("common_input_sha256")
        and usage_artifact.payload.get("parent_origin_receipt_sha256")
        == judge.payload.get("parent_origin_receipt_sha256")
        and usage_artifact.payload.get("strict_sources", {}).get(
            "answer_run_sha256"
        )
        == judge.payload.get("answer_run_sha256")
        and usage_artifact.payload.get("strict_sources", {}).get(
            "answer_replay_sha256"
        )
        == judge.payload.get("answer_replay_sha256")
        and usage_artifact.payload.get("strict_sources", {}).get(
            "judge_run_sha256"
        )
        == judge.sha256
        and usage_artifact.payload.get("strict_sources", {}).get(
            "judge_replay_sha256"
        )
        == judge_replay.sha256
        and usage_artifact.payload.get("strict_sources", {}).get(
            "score_run_sha256"
        )
        == score.sha256
        and usage_artifact.payload.get("strict_sources", {}).get(
            "score_replay_sha256"
        )
        == score_replay.sha256,
        "Mem0 score source is not the exact verified full100 quad",
    )
    judge_rows = _list(judge.payload.get("questions"), "Mem0 judge rows")
    score_rows = _list(score.payload.get("score_rows"), "Mem0 score rows")
    _require(len(judge_rows) == len(score_rows) == QUESTION_COUNT, "Mem0 score source is not full100")
    rows: list[dict[str, Any]] = []
    for ordinal, (raw, scored) in enumerate(zip(judge_rows, score_rows, strict=True)):
        row = _dict(raw, "Mem0 judge row")
        score_row = _dict(scored, "Mem0 score row")
        _require(
            row.get("ordinal") == score_row.get("ordinal") == ordinal
            and row.get("question_id") == score_row.get("question_id")
            and row.get("question_sha256") == score_row.get("question_sha256")
            and row.get("judge_row_sha256") == score_row.get("judge_row_sha256")
            and row.get("correct") == score_row.get("correct"),
            f"Mem0 score/judge row binding changed at ordinal {ordinal}",
        )
        rows.append(
            _canonical_row(
                answer_validation_receipt_sha256=row[
                    "answer_prompt_row_receipt_sha256"
                ],
                correct=row["correct"],
                dated_question_sha256=row["dated_question_sha256"],
                judge_row_sha256=row["judge_row_sha256"],
                ordinal=ordinal,
                prediction_sha256=row["prediction_sha256"],
                prediction_source=row["prediction_source"],
                question_id=row["question_id"],
                question_sha256=row["question_sha256"],
                reference_sha256=row["reference_sha256"],
            )
        )
    return _build_plane(
        arm_role="mem0",
        adapter_id=MEM0_TYPED_ADAPTER,
        parent_origin_receipt_sha256=judge.payload[
            "parent_origin_receipt_sha256"
        ],
        gold_population_sha256=judge.payload["gold_population_sha256"],
        source_artifacts={
            "answer_replay_sha256": judge.payload["answer_replay_sha256"],
            "answer_run_sha256": judge.payload["answer_run_sha256"],
            "judge_replay_sha256": judge_replay.sha256,
            "judge_run_sha256": judge.sha256,
            "preflight_sha256": judge.payload["preflight_artifact_sha256"],
            "score_replay_sha256": score_replay.sha256,
            "score_run_sha256": score.sha256,
        },
        rows=rows,
        exact_accounting=judge.payload["model_accounting"],
        usage_attestation_sha256=usage_artifact.sha256,
    )


def load_verified_mem0_score_plane(
    plane_path: str | Path,
    expected_plane_sha256: str,
    plane_replay_path: str | Path,
    expected_plane_replay_sha256: str,
    *,
    judge_output_root: str | Path,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    answer_output_root: str | Path,
    expected_answer_preflight_sha256: str,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
    dataset_path: str | Path,
    split_path: str | Path,
    expected_judge_preflight_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
    usage_attestation_path: str | Path,
    expected_usage_attestation_sha256: str,
    usage_attestation_replay_path: str | Path,
    expected_usage_attestation_replay_sha256: str,
) -> VerifiedScorePlane:
    """Strict Mem0 adapter: rebuild answer, gold, journals, judge, and score."""

    judge, judge_replay, score, score_replay, _rows = load_verified_judge_score(
        judge_output_root,
        common_input_path=common_input_path,
        expected_common_input_sha256=expected_common_input_sha256,
        answer_output_root=answer_output_root,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_answer_run_sha256=expected_answer_run_sha256,
        expected_answer_replay_sha256=expected_answer_replay_sha256,
        dataset_path=dataset_path,
        split_path=split_path,
        expected_preflight_sha256=expected_judge_preflight_sha256,
        expected_judge_sha256=expected_judge_sha256,
        expected_judge_replay_sha256=expected_judge_replay_sha256,
        expected_score_sha256=expected_score_sha256,
        expected_score_replay_sha256=expected_score_replay_sha256,
        expected_question_count=QUESTION_COUNT,
    )
    usage = load_verified_usage_attestation(
        usage_attestation_path,
        expected_usage_attestation_sha256,
        usage_attestation_replay_path,
        expected_usage_attestation_replay_sha256,
        common_input_path=common_input_path,
        expected_common_input_sha256=expected_common_input_sha256,
        answer_output_root=answer_output_root,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_answer_run_sha256=expected_answer_run_sha256,
        expected_answer_replay_sha256=expected_answer_replay_sha256,
        judge_output_root=judge_output_root,
        dataset_path=dataset_path,
        split_path=split_path,
        expected_judge_preflight_sha256=expected_judge_preflight_sha256,
        expected_judge_sha256=expected_judge_sha256,
        expected_judge_replay_sha256=expected_judge_replay_sha256,
        expected_score_sha256=expected_score_sha256,
        expected_score_replay_sha256=expected_score_replay_sha256,
    )
    expected = build_mem0_score_plane_payload(
        judge=judge,
        judge_replay=judge_replay,
        score=score,
        score_replay=score_replay,
        usage=usage,
    )
    plane, replay, rows = _read_published_plane(
        plane_path,
        expected_plane_sha256,
        plane_replay_path,
        expected_plane_replay_sha256,
        expected_arm_role="mem0",
        expected_payload=expected,
    )
    return VerifiedScorePlane(
        plane,
        replay,
        rows,
        _token=_VERIFIED_SCORE_PLANE_TOKEN,
    )


def load_certified_v3_treatment_score_plane(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
    judge_output_root: str | Path,
    expected_preflight_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
) -> dict[str, Any]:
    """Replay-authenticate and project the current certified V3 full100 score."""

    # The frozen validation-v3 source predates the treatment judge helper's
    # ``_binary_judge_protocol`` module.  Keep this treatment-only dependency
    # outside the Mem0 launch/preflight import graph; it is needed only when a
    # certified treatment score plane is actually reconstructed.
    from tools import run_locked_specialist_final_judge_v3 as treatment_v3

    answer_run, answer_replay, _source_rows = (
        treatment_v3.load_verified_answer_judge_source(
            answer_run_path=answer_run_path,
            answer_replay_path=answer_replay_path,
            expected_answer_run_sha256=expected_answer_run_sha256,
            expected_answer_replay_sha256=expected_answer_replay_sha256,
        )
    )
    root = Path(judge_output_root)
    preflight = _read(
        root / treatment_v3.PREFLIGHT_NAME,
        expected_preflight_sha256,
        "certified V3 preflight",
    )
    prompts, prompt_rows = treatment_v3.validate_preflight_artifact(preflight)
    _require(
        preflight.payload.get("answer_run_sha256") == answer_run.sha256
        and preflight.payload.get("answer_replay_sha256") == answer_replay.sha256,
        "certified V3 preflight escaped its answer source",
    )
    runtime = treatment_v3.build_runtime(
        preflight,
        prompts,
        output_root=root,
        model=JUDGE_MODEL,
        gateway_url=preflight.payload["gateway_url"],
        max_concurrency=preflight.payload["max_concurrency"],
        client=None,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    rebuilt_judge, rebuilt_score = treatment_v3.materialization_payloads(
        preflight, prompt_rows, batch
    )
    judge = _read(root / treatment_v3.JUDGE_NAME, expected_judge_sha256, "certified V3 judge")
    judge_replay = _read(
        root / treatment_v3.JUDGE_REPLAY_NAME,
        expected_judge_replay_sha256,
        "certified V3 judge replay",
    )
    score = _read(root / treatment_v3.SCORE_NAME, expected_score_sha256, "certified V3 score")
    score_replay = _read(
        root / treatment_v3.SCORE_REPLAY_NAME,
        expected_score_replay_sha256,
        "certified V3 score replay",
    )
    _require(
        judge.sha256 == judge_replay.sha256
        and judge.payload == judge_replay.payload == rebuilt_judge
        and score.sha256 == score_replay.sha256
        and score.payload == score_replay.payload == rebuilt_score
        and batch.provenance.model == JUDGE_MODEL
        and batch.provenance.max_prompt_token_proxy == MAX_JUDGE_PROMPT_TOKENS
        and batch.provenance.max_new_tokens == JUDGE_OUTPUT_TOKEN_RESERVE
        and batch.provenance.retries == 0
        and batch.provenance.retained_transformer_token_state_bytes == 0
        and answer_run.payload.get("hard_complete_chat_token_cap") == 8000
        and answer_run.payload.get("max_chat_prompt_tokens") == 7232
        and answer_run.payload.get("output_token_reserve") == 768
        and answer_run.payload.get("retained_transformer_token_state_bytes") == 0,
        "certified V3 source failed exact model/budget/replay accounting",
    )
    answer_by_ordinal = {
        row["ordinal"]: row for row in answer_run.payload["questions"]
    }
    rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(judge.payload["questions"]):
        answer = answer_by_ordinal.get(ordinal)
        _require(
            type(answer) is dict
            and answer.get("source_row_sha256") == row.get("source_row_sha256"),
            f"certified V3 answer/judge binding changed at ordinal {ordinal}",
        )
        rows.append(
            _canonical_row(
                answer_validation_receipt_sha256=answer[
                    "answer_plan_receipt_sha256"
                ],
                correct=row["correct"],
                dated_question_sha256=row["dated_question_sha256"],
                judge_row_sha256=row["judge_row_sha256"],
                ordinal=ordinal,
                prediction_sha256=row["prediction_sha256"],
                prediction_source=row["prediction_source"],
                question_id=row["question_id"],
                question_sha256=row["question_sha256"],
                reference_sha256=row["reference_sha256"],
            )
        )
    return _build_plane(
        arm_role="treatment",
        adapter_id=CERTIFIED_V3_ADAPTER,
        parent_origin_receipt_sha256=_exact_parent_origin(
            parent_run_sha256=answer_run.sha256,
            parent_replay_sha256=answer_replay.sha256,
        ),
        gold_population_sha256=judge.payload["gold_population_sha256"],
        source_artifacts={
            "answer_replay_sha256": answer_replay.sha256,
            "answer_run_sha256": answer_run.sha256,
            "judge_replay_sha256": judge_replay.sha256,
            "judge_run_sha256": judge.sha256,
            "preflight_sha256": preflight.sha256,
            "score_replay_sha256": score_replay.sha256,
            "score_run_sha256": score.sha256,
        },
        rows=rows,
        exact_accounting=EXACT_ACCOUNTING,
    )


def load_verified_v3_treatment_score_plane(
    plane_path: str | Path,
    expected_plane_sha256: str,
    plane_replay_path: str | Path,
    expected_plane_replay_sha256: str,
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
    judge_output_root: str | Path,
    expected_preflight_sha256: str,
    expected_judge_sha256: str,
    expected_judge_replay_sha256: str,
    expected_score_sha256: str,
    expected_score_replay_sha256: str,
) -> VerifiedScorePlane:
    """Strict treatment adapter: replay V3 journals and byte-compare its plane."""

    expected = load_certified_v3_treatment_score_plane(
        answer_run_path=answer_run_path,
        answer_replay_path=answer_replay_path,
        expected_answer_run_sha256=expected_answer_run_sha256,
        expected_answer_replay_sha256=expected_answer_replay_sha256,
        judge_output_root=judge_output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_judge_sha256=expected_judge_sha256,
        expected_judge_replay_sha256=expected_judge_replay_sha256,
        expected_score_sha256=expected_score_sha256,
        expected_score_replay_sha256=expected_score_replay_sha256,
    )
    plane, replay, rows = _read_published_plane(
        plane_path,
        expected_plane_sha256,
        plane_replay_path,
        expected_plane_replay_sha256,
        expected_arm_role="treatment",
        expected_payload=expected,
    )
    return VerifiedScorePlane(
        plane,
        replay,
        rows,
        _token=_VERIFIED_SCORE_PLANE_TOKEN,
    )


def build_terminal_v2_score_plane(*_args: object, **_kwargs: object) -> dict[str, Any]:
    raise CommonParentComparisonError(TERMINAL_V2_ADAPTER_STATUS)


def build_comparison_payload(
    treatment: VerifiedScorePlane,
    mem0: VerifiedScorePlane,
) -> dict[str, Any]:
    _require(
        type(treatment) is VerifiedScorePlane
        and type(mem0) is VerifiedScorePlane,
        "comparison requires strict arm-specific score-plane authorities",
    )
    treatment_artifact, _treatment_replay, treatment_rows = (
        _reopen_verified_score_plane(treatment, expected_arm_role="treatment")
    )
    mem0_artifact, _mem0_replay, mem0_rows = _reopen_verified_score_plane(
        mem0, expected_arm_role="mem0"
    )
    _require(
        treatment_artifact.payload["parent_origin_receipt_sha256"]
        == mem0_artifact.payload["parent_origin_receipt_sha256"]
        and treatment_artifact.payload["gold_population_sha256"]
        == mem0_artifact.payload["gold_population_sha256"]
        and treatment_artifact.payload["question_order_sha256"]
        == mem0_artifact.payload["question_order_sha256"],
        "paired score planes do not share parent, gold, and question order",
    )
    pairs: list[dict[str, Any]] = []
    wins = losses = ties_correct = ties_incorrect = 0
    for treatment_row, mem0_row in zip(treatment_rows, mem0_rows, strict=True):
        _require(
            all(
                treatment_row[key] == mem0_row[key]
                for key in (
                    "ordinal",
                    "question_id",
                    "question_sha256",
                    "dated_question_sha256",
                    "reference_sha256",
                )
            ),
            "paired question identity changed",
        )
        treatment_correct = treatment_row["correct"]
        mem0_correct = mem0_row["correct"]
        outcome = (
            "mem0_win"
            if mem0_correct and not treatment_correct
            else "mem0_loss"
            if treatment_correct and not mem0_correct
            else "tie_correct"
            if mem0_correct
            else "tie_incorrect"
        )
        wins += outcome == "mem0_win"
        losses += outcome == "mem0_loss"
        ties_correct += outcome == "tie_correct"
        ties_incorrect += outcome == "tie_incorrect"
        pairs.append(
            {
                "mem0_correct": mem0_correct,
                "mem0_judge_row_sha256": mem0_row["judge_row_sha256"],
                "ordinal": mem0_row["ordinal"],
                "outcome": outcome,
                "question_id": mem0_row["question_id"],
                "question_sha256": mem0_row["question_sha256"],
                "treatment_correct": treatment_correct,
                "treatment_judge_row_sha256": treatment_row[
                    "judge_row_sha256"
                ],
            }
        )
    treatment_correct = treatment_artifact.payload["correct"]
    mem0_correct = mem0_artifact.payload["correct"]
    return {
        "certification": {
            "cost_comparison_certified": False,
            "fair_system_comparison_certified": False,
            "paired_accuracy_certified": True,
            "reason": (
                "paired accuracy has strict score/judge/journal authority; "
                "authenticated treatment and Mem0 write/read cost authorities "
                "are not both bound"
            ),
        },
        "comparison_certified": False,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "exact_accounting": dict(EXACT_ACCOUNTING),
        "format": COMPARISON_FORMAT,
        "gold_population_sha256": treatment_artifact.payload[
            "gold_population_sha256"
        ],
        "mem0": {
            "accuracy": mem0_correct / QUESTION_COUNT,
            "correct": mem0_correct,
            "score_plane_sha256": mem0_artifact.sha256,
            "usage_attestation_sha256": mem0_artifact.payload[
                "usage_attestation_sha256"
            ],
        },
        "paired": {
            "accuracy_delta_mem0_minus_treatment": (
                mem0_correct - treatment_correct
            )
            / QUESTION_COUNT,
            "correct_delta_mem0_minus_treatment": mem0_correct - treatment_correct,
            "mem0_losses": losses,
            "mem0_wins": wins,
            "tie_correct": ties_correct,
            "tie_incorrect": ties_incorrect,
        },
        "pairs": pairs,
        "parent_origin_receipt_sha256": treatment_artifact.payload[
            "parent_origin_receipt_sha256"
        ],
        "question_count": QUESTION_COUNT,
        "question_order_sha256": treatment_artifact.payload[
            "question_order_sha256"
        ],
        "treatment": {
            "accuracy": treatment_correct / QUESTION_COUNT,
            "correct": treatment_correct,
            "score_plane_sha256": treatment_artifact.sha256,
        },
    }


def publish_comparison(
    output_root: str | Path,
    treatment: VerifiedScorePlane,
    mem0: VerifiedScorePlane,
) -> tuple[SealedArtifact, SealedArtifact]:
    payload = build_comparison_payload(treatment, mem0)
    result, _ = publish_sealed_json(Path(output_root) / COMPARISON_NAME, payload)
    replay, _ = publish_sealed_json(
        Path(output_root) / COMPARISON_REPLAY_NAME, payload
    )
    _require(result.sha256 == replay.sha256, "paired comparison replay changed")
    return result, replay


def load_verified_comparison(
    comparison_path: str | Path,
    expected_comparison_sha256: str,
    replay_path: str | Path,
    expected_replay_sha256: str,
    *,
    treatment_authority: Mapping[str, Any],
    mem0_authority: Mapping[str, Any],
) -> SealedArtifact:
    _require(
        type(treatment_authority) is dict and type(mem0_authority) is dict,
        "comparison authorities must be exact arm-specific argument objects",
    )
    try:
        treatment = load_verified_v3_treatment_score_plane(
            **dict(treatment_authority)
        )
        mem0 = load_verified_mem0_score_plane(**dict(mem0_authority))
    except TypeError as exc:
        raise CommonParentComparisonError(
            "comparison authority arguments changed"
        ) from exc
    result = _read(comparison_path, expected_comparison_sha256, "comparison")
    replay = _read(replay_path, expected_replay_sha256, "comparison replay")
    rebuilt = build_comparison_payload(treatment, mem0)
    _require(
        result.sha256 == replay.sha256
        and result.payload == replay.payload == rebuilt
        and result.payload.get("comparison_certified") is False
        and result.payload.get("certification")
        == {
            "cost_comparison_certified": False,
            "fair_system_comparison_certified": False,
            "paired_accuracy_certified": True,
            "reason": (
                "paired accuracy has strict score/judge/journal authority; "
                "authenticated treatment and Mem0 write/read cost authorities "
                "are not both bound"
            ),
        },
        "paired accuracy comparison is not an exact scoped replay",
    )
    return result


__all__ = [
    "CERTIFIED_V3_ADAPTER",
    "COMPARISON_FORMAT",
    "COMPARISON_NAME",
    "COMPARISON_REPLAY_NAME",
    "CommonParentComparisonError",
    "EXACT_ACCOUNTING",
    "MEM0_SCORE_PLANE_NAME",
    "MEM0_SCORE_PLANE_REPLAY_NAME",
    "MEM0_TYPED_ADAPTER",
    "QUESTION_COUNT",
    "SCORE_PLANE_FORMAT",
    "TERMINAL_V2_ADAPTER_STATUS",
    "TREATMENT_SCORE_PLANE_NAME",
    "TREATMENT_SCORE_PLANE_REPLAY_NAME",
    "VerifiedScorePlane",
    "build_comparison_payload",
    "build_mem0_score_plane_payload",
    "build_terminal_v2_score_plane",
    "load_certified_v3_treatment_score_plane",
    "load_verified_mem0_score_plane",
    "load_verified_v3_treatment_score_plane",
    "load_verified_comparison",
    "load_verified_score_plane",
    "publish_comparison",
    "publish_score_plane",
    "validate_score_plane_artifact",
]
