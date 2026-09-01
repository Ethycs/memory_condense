"""Provider-free import boundary for the sealed S0/EM/CAV legacy checkpoint.

The three historical arms predate the common matched-eval renderer.  This
module does not execute their old loaders or reconstruct provider calls.  It
only verifies four already sealed artifacts per arm, normalizes the runtime
answer behavior separately from post-hoc judging, and records the renderer
boundary explicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from .artifacts import SealedArtifact, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)


RUN_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-run-v1"
JUDGE_FORMAT = "memory-condense-locked-retrieval-mechanism-sol-judge-v1"
JUDGE_BINDING_FORMAT = (
    "memory-condense-locked-retrieval-mechanism-sol-judge-binding-v1"
)
QUESTION_COUNT = 100
S0_ARM_LABEL = "S0_CONTROL"
EXTERNAL_S1_LABEL = "FIXED_S1_EXTERNAL_ANCHOR"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
)

LegacyDeltaKind = Literal["membership", "representation", "linking"]


class LegacyImportError(MatchedEvalContractError):
    """Raised when a sealed legacy checkpoint violates the migration contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LegacyImportError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    return value  # type: ignore[return-value]


def _rows(value: object, label: str) -> list[dict[str, Any]]:
    _require(type(value) is list, f"{label} must be an array")
    result = value  # type: ignore[assignment]
    _require(
        all(type(row) is dict for row in result),
        f"{label} rows must be objects",
    )
    return result


def _text(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be text")
    return value


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except MatchedEvalContractError as exc:
        raise LegacyImportError(str(exc)) from exc


def _optional_sha(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _sha(value, label)


def _integer(value: object, label: str) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    return value  # type: ignore[return-value]


def _boolean(value: object, label: str) -> bool:
    _require(type(value) is bool, f"{label} must be a boolean")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class LegacyArmSpec:
    """Trusted identity and schema expectations for one historical arm."""

    arm_label: str
    directory_name: str
    parent_arm_label: str | None
    judge_baseline_arm_label: str
    renderer_identity: str
    delta_kind: LegacyDeltaKind
    run_sha256: str
    run_replay_sha256: str
    judge_sha256: str
    judge_replay_sha256: str
    expected_parent_run_sha256: str | None
    baseline_correct: int
    candidate_correct: int
    rescued: int
    regressed: int
    unique_judge_call_count: int
    accepted_for_positive_only_composition: bool
    question_count: int = QUESTION_COUNT

    def __post_init__(self) -> None:
        require_text(self.arm_label, "legacy arm label")
        require_text(self.directory_name, "legacy arm directory")
        if self.parent_arm_label is not None:
            require_text(self.parent_arm_label, "legacy parent arm label")
        require_text(self.judge_baseline_arm_label, "legacy judge baseline label")
        require_text(self.renderer_identity, "legacy renderer identity")
        for label, digest in (
            ("run", self.run_sha256),
            ("run replay", self.run_replay_sha256),
            ("judge", self.judge_sha256),
            ("judge replay", self.judge_replay_sha256),
        ):
            require_sha256(digest, f"legacy {label} SHA-256")
        if self.expected_parent_run_sha256 is not None:
            require_sha256(
                self.expected_parent_run_sha256,
                "legacy expected parent run SHA-256",
            )
        if self.question_count != QUESTION_COUNT:
            raise MatchedEvalContractError(
                f"legacy migrations require exactly {QUESTION_COUNT} questions"
            )
        counts = (
            self.baseline_correct,
            self.candidate_correct,
            self.rescued,
            self.regressed,
            self.unique_judge_call_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise MatchedEvalContractError("legacy score counts must be non-negative")


_S0_RUN_SHA256 = "a713328485ebef452a0dd30626a7ffc20126999162723cb543da4f94a87b8e68"

LEGACY_ARM_REGISTRY: Mapping[str, LegacyArmSpec] = MappingProxyType(
    {
        S0_ARM_LABEL: LegacyArmSpec(
            arm_label=S0_ARM_LABEL,
            directory_name="s0-control-v1",
            parent_arm_label=None,
            judge_baseline_arm_label=EXTERNAL_S1_LABEL,
            renderer_identity="legacy_renderer/s0_qa_v1",
            delta_kind="membership",
            run_sha256=_S0_RUN_SHA256,
            run_replay_sha256=_S0_RUN_SHA256,
            judge_sha256=(
                "1c9ea03121478edd053c666bfffb8eaf1db508f001df76367ce14adc8f5022cb"
            ),
            judge_replay_sha256=(
                "1c9ea03121478edd053c666bfffb8eaf1db508f001df76367ce14adc8f5022cb"
            ),
            expected_parent_run_sha256=None,
            baseline_correct=56,
            candidate_correct=57,
            rescued=5,
            regressed=4,
            unique_judge_call_count=100,
            accepted_for_positive_only_composition=True,
        ),
        "S0_PLUS_EM_FACTS": LegacyArmSpec(
            arm_label="S0_PLUS_EM_FACTS",
            directory_name="s0-plus-em-facts-v1",
            parent_arm_label=S0_ARM_LABEL,
            judge_baseline_arm_label=S0_ARM_LABEL,
            renderer_identity="legacy_renderer/em_facts_v1",
            delta_kind="representation",
            run_sha256=(
                "af2ee321cbd4d624b753ac942072bbe2fd54d49b86384ae7fdb13d6b46cc3db9"
            ),
            run_replay_sha256=(
                "af2ee321cbd4d624b753ac942072bbe2fd54d49b86384ae7fdb13d6b46cc3db9"
            ),
            judge_sha256=(
                "13913b6bc95f1dca8d5c974fdb7bcf8feae4bae44f5f7ca78d6336ce66016cf8"
            ),
            judge_replay_sha256=(
                "13913b6bc95f1dca8d5c974fdb7bcf8feae4bae44f5f7ca78d6336ce66016cf8"
            ),
            expected_parent_run_sha256=_S0_RUN_SHA256,
            baseline_correct=57,
            candidate_correct=60,
            rescued=8,
            regressed=5,
            unique_judge_call_count=43,
            accepted_for_positive_only_composition=True,
        ),
        "S0_PLUS_CAV_LINKS": LegacyArmSpec(
            arm_label="S0_PLUS_CAV_LINKS",
            directory_name="s0-plus-cav-links-v1",
            parent_arm_label=S0_ARM_LABEL,
            judge_baseline_arm_label=S0_ARM_LABEL,
            renderer_identity="legacy_renderer/cav_links_v1",
            delta_kind="linking",
            run_sha256=(
                "6052f52b7835848aa8e9578703c6bb131460e0eb54c8c64e70bc42dfd783ca49"
            ),
            run_replay_sha256=(
                "6052f52b7835848aa8e9578703c6bb131460e0eb54c8c64e70bc42dfd783ca49"
            ),
            judge_sha256=(
                "8f44f3a259b11615f3009c5ae1d047cc6b0a94290b84a3d13b915adbd82fcfb0"
            ),
            judge_replay_sha256=(
                "8f44f3a259b11615f3009c5ae1d047cc6b0a94290b84a3d13b915adbd82fcfb0"
            ),
            expected_parent_run_sha256=_S0_RUN_SHA256,
            baseline_correct=57,
            candidate_correct=53,
            rescued=2,
            regressed=6,
            unique_judge_call_count=31,
            accepted_for_positive_only_composition=False,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class LegacyArtifactPaths:
    run: Path
    run_replay: Path
    judge: Path
    judge_replay: Path

    @classmethod
    def under_root(cls, root: str | Path, spec: LegacyArmSpec) -> "LegacyArtifactPaths":
        arm_root = Path(root) / spec.directory_name
        return cls(
            run=arm_root / "run.json",
            run_replay=arm_root / "run-replay.json",
            judge=arm_root / "semantic-judge-sol.json",
            judge_replay=arm_root / "semantic-judge-sol-replay.json",
        )


@dataclass(frozen=True, slots=True)
class LegacyArtifactIdentity:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LegacyRuntimeObservation:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    retrieval_question_part_sha256: str
    source_binding_sha256: str
    prompt_messages_sha256: str | None
    call_key_sha256: str | None
    request_journal_sha256: str | None
    response_journal_sha256: str | None
    prediction_text: str
    prediction_sha256: str
    parent_prediction_sha256: str | None
    changed_from_parent: bool | None
    source_row_sha256: str

    def projection(
        self, *, include_source_row_sha256: bool = True
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "call_key_sha256": self.call_key_sha256,
            "changed_from_parent": self.changed_from_parent,
            "dated_question_sha256": self.dated_question_sha256,
            "ordinal": self.ordinal,
            "parent_prediction_sha256": self.parent_prediction_sha256,
            "prediction": {
                "sha256": self.prediction_sha256,
                "text": self.prediction_text,
            },
            "prompt_messages_sha256": self.prompt_messages_sha256,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "request_journal_sha256": self.request_journal_sha256,
            "response_journal_sha256": self.response_journal_sha256,
            "retrieval_question_part_sha256": self.retrieval_question_part_sha256,
            "source_binding_sha256": self.source_binding_sha256,
        }
        if include_source_row_sha256:
            result["source_row_sha256"] = self.source_row_sha256
        return result


@dataclass(frozen=True, slots=True)
class LegacyScoreObservation:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction_sha256: str
    baseline_prediction_sha256: str
    baseline_correct: bool
    correct: bool
    changed_from_baseline: bool
    rescued: bool
    regressed: bool
    verdict_source: str
    gold_answer_sha256: str
    question_only_demand_class: str
    evidence_topology_class: str
    baseline_judge_row_sha256: str
    judge_verdict_sha256: str | None
    judge_row_sha256: str

    def projection(
        self, *, include_source_row_sha256: bool = True
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "baseline_correct": self.baseline_correct,
            "baseline_judge_row_sha256": self.baseline_judge_row_sha256,
            "baseline_prediction_sha256": self.baseline_prediction_sha256,
            "changed_from_baseline": self.changed_from_baseline,
            "correct": self.correct,
            "dated_question_sha256": self.dated_question_sha256,
            "evidence_topology_class": self.evidence_topology_class,
            "gold_answer_sha256": self.gold_answer_sha256,
            "judge_verdict_sha256": self.judge_verdict_sha256,
            "ordinal": self.ordinal,
            "prediction_sha256": self.prediction_sha256,
            "question_id": self.question_id,
            "question_only_demand_class": self.question_only_demand_class,
            "question_sha256": self.question_sha256,
            "regressed": self.regressed,
            "rescued": self.rescued,
            "verdict_source": self.verdict_source,
        }
        if include_source_row_sha256:
            result["judge_row_sha256"] = self.judge_row_sha256
        return result


@dataclass(frozen=True, slots=True)
class LegacyScoreAggregate:
    baseline_correct: int
    candidate_correct: int
    rescued: int
    regressed: int
    net_marginal: int
    accepted_for_positive_only_composition: bool

    def projection(self) -> dict[str, object]:
        return {
            "accepted_for_positive_only_composition": (
                self.accepted_for_positive_only_composition
            ),
            "baseline_correct": self.baseline_correct,
            "candidate_correct": self.candidate_correct,
            "net_marginal": self.net_marginal,
            "regressed": self.regressed,
            "rescued": self.rescued,
        }


@dataclass(frozen=True, slots=True)
class LegacyArmMigration:
    spec: LegacyArmSpec
    run_artifact: LegacyArtifactIdentity
    run_replay_artifact: LegacyArtifactIdentity
    judge_artifact: LegacyArtifactIdentity
    judge_replay_artifact: LegacyArtifactIdentity
    population_identity_sha256: str
    retrieval_sha256: str
    runtime_observations: tuple[LegacyRuntimeObservation, ...]
    score_observations: tuple[LegacyScoreObservation, ...]
    score_aggregate: LegacyScoreAggregate
    imported_provider_call_count: int = 0

    def __post_init__(self) -> None:
        if self.imported_provider_call_count != 0:
            raise MatchedEvalContractError(
                "legacy migration imports must make zero provider calls"
            )

    def runtime_projection(self) -> dict[str, object]:
        projection: dict[str, object] = {
            "arm_label": self.spec.arm_label,
            "delta_kind": self.spec.delta_kind,
            "format": "memory-condense-matched-eval-legacy-runtime-v1",
            "imported_provider_call_count": 0,
            "observations": [row.projection() for row in self.runtime_observations],
            "parent_arm_label": self.spec.parent_arm_label,
            "population_identity_sha256": self.population_identity_sha256,
            "question_count": len(self.runtime_observations),
            "renderer_identity": self.spec.renderer_identity,
            "retrieval_sha256": self.retrieval_sha256,
            "source_run_replay_sha256": self.run_replay_artifact.sha256,
            "source_run_sha256": self.run_artifact.sha256,
        }
        assert_gold_blind(projection)
        return projection

    def score_projection(self) -> dict[str, object]:
        return {
            "aggregate": self.score_aggregate.projection(),
            "arm_label": self.spec.arm_label,
            "delta_kind": self.spec.delta_kind,
            "format": "memory-condense-matched-eval-legacy-score-v1",
            "judge_baseline_arm_label": self.spec.judge_baseline_arm_label,
            "observations": [row.projection() for row in self.score_observations],
            "parent_arm_label": self.spec.parent_arm_label,
            "population_identity_sha256": self.population_identity_sha256,
            "question_count": len(self.score_observations),
            "renderer_identity": self.spec.renderer_identity,
            "retrieval_sha256": self.retrieval_sha256,
            "source_judge_replay_sha256": self.judge_replay_artifact.sha256,
            "source_judge_sha256": self.judge_artifact.sha256,
        }

    def manifest_projection(self) -> dict[str, object]:
        return {
            "arm_label": self.spec.arm_label,
            "artifacts": {
                "judge": self.judge_artifact.sha256,
                "judge_replay": self.judge_replay_artifact.sha256,
                "run": self.run_artifact.sha256,
                "run_replay": self.run_replay_artifact.sha256,
            },
            "delta_kind": self.spec.delta_kind,
            "format": "memory-condense-matched-eval-legacy-arm-manifest-v1",
            "imported_provider_call_count": 0,
            "judge_baseline_arm_label": self.spec.judge_baseline_arm_label,
            "parent_arm_label": self.spec.parent_arm_label,
            "population_identity_sha256": self.population_identity_sha256,
            "question_count": len(self.runtime_observations),
            "renderer_identity": self.spec.renderer_identity,
            "retrieval_sha256": self.retrieval_sha256,
        }


@dataclass(frozen=True, slots=True)
class LegacyCheckpointMigration:
    arms: tuple[LegacyArmMigration, ...]
    imported_provider_call_count: int = 0

    def __post_init__(self) -> None:
        expected = tuple(LEGACY_ARM_REGISTRY)
        actual = tuple(arm.spec.arm_label for arm in self.arms)
        if actual != expected:
            raise MatchedEvalContractError(
                f"legacy checkpoint arm order changed: {actual!r} != {expected!r}"
            )
        if self.imported_provider_call_count != 0:
            raise MatchedEvalContractError(
                "legacy checkpoint import must make zero provider calls"
            )

    def arm(self, arm_label: str) -> LegacyArmMigration:
        for migration in self.arms:
            if migration.spec.arm_label == arm_label:
                return migration
        raise KeyError(arm_label)

    def runtime_projection(self) -> dict[str, object]:
        projection: dict[str, object] = {
            "arms": [arm.runtime_projection() for arm in self.arms],
            "format": "memory-condense-matched-eval-legacy-runtime-checkpoint-v1",
            "imported_provider_call_count": 0,
        }
        assert_gold_blind(projection)
        return projection

    def score_projection(self) -> dict[str, object]:
        return {
            "arms": [arm.score_projection() for arm in self.arms],
            "format": "memory-condense-matched-eval-legacy-score-checkpoint-v1",
        }

    def manifest_projection(self) -> dict[str, object]:
        return {
            "arms": [arm.manifest_projection() for arm in self.arms],
            "format": "memory-condense-matched-eval-legacy-campaign-manifest-v1",
            "imported_provider_call_count": 0,
        }


# Public campaign terminology used by the migration CLI.  Retain the older
# checkpoint name as an exact type alias for callers that adopted it first.
LegacyCampaignMigration = LegacyCheckpointMigration


def _artifact_identity(artifact: SealedArtifact) -> LegacyArtifactIdentity:
    return LegacyArtifactIdentity(path=str(artifact.path), sha256=artifact.sha256)


def _read_expected(path: Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == expected_sha256,
        f"{label} SHA-256 changed: {artifact.sha256} != {expected_sha256}",
    )
    return artifact


def _runtime_observations(
    payload: Mapping[str, Any], spec: LegacyArmSpec
) -> tuple[str, str, tuple[LegacyRuntimeObservation, ...]]:
    _require(payload.get("format") == RUN_FORMAT, "legacy run format changed")
    _require(payload.get("arm_label") == spec.arm_label, "legacy run arm changed")
    _require(payload.get("gold_loaded") is False, "legacy run loaded gold")
    _require(
        payload.get("retained_request_token_state_bytes") == 0,
        "legacy run retained request-token state",
    )
    for flag in ("benchmark_categories_loaded", "benchmark_source_labels_loaded"):
        if flag in payload:
            _require(payload.get(flag) is False, f"legacy run enabled {flag}")
    try:
        assert_gold_blind(payload)
    except MatchedEvalContractError as exc:
        raise LegacyImportError(str(exc)) from exc

    parent = payload.get("parent_arm_label")
    _require(parent == spec.parent_arm_label, "legacy run parent arm changed")
    if spec.parent_arm_label is None:
        identity = _mapping(payload.get("arm_identity"), "S0 arm identity")
        _require(identity.get("parent_arm") is None, "S0 arm identity gained a parent")
    else:
        _require(
            payload.get("s0_control_run_sha256") == spec.expected_parent_run_sha256,
            "legacy descendant parent run changed",
        )

    population_sha = _sha(
        payload.get("population_identity_sha256"), "legacy run population identity"
    )
    retrieval_sha = _sha(payload.get("retrieval_sha256"), "legacy run retrieval")
    _require(
        payload.get("question_count") == spec.question_count,
        "legacy run question count changed",
    )
    rows = _rows(payload.get("questions"), "legacy run questions")
    _require(len(rows) == spec.question_count, "legacy run row count changed")

    observations: list[LegacyRuntimeObservation] = []
    seen_ids: set[str] = set()
    for expected_ordinal, raw in enumerate(rows):
        label = f"legacy run question {expected_ordinal}"
        ordinal = _integer(raw.get("ordinal"), f"{label} ordinal")
        _require(ordinal == expected_ordinal, f"{label} order changed")
        question_id = _text(raw.get("question_id"), f"{label} ID")
        _require(question_id not in seen_ids, f"duplicate legacy question ID: {question_id}")
        seen_ids.add(question_id)
        question_sha = _sha(raw.get("question_sha256"), f"{label} question SHA-256")
        dated_sha = _sha(
            raw.get("dated_question_sha256"), f"{label} dated-question SHA-256"
        )
        retrieval_question_sha = _sha(
            raw.get("retrieval_question_part_sha256"),
            f"{label} retrieval-question SHA-256",
        )
        prediction = _mapping(raw.get("prediction"), f"{label} prediction")
        prediction_text = _text(prediction.get("text"), f"{label} prediction text")
        prediction_sha = _sha(
            prediction.get("sha256"), f"{label} prediction SHA-256"
        )
        observed_prediction_sha = hashlib.sha256(
            prediction_text.encode("utf-8")
        ).hexdigest()
        _require(
            observed_prediction_sha == prediction_sha,
            f"{label} prediction digest changed",
        )

        if spec.parent_arm_label is None:
            prompt_field = "provider_messages_sha256"
            call_field = "call_key_sha256"
            request_field = "request_journal_sha256"
            response_field = "response_journal_sha256"
            source_field = "source_binding_sha256"
            parent_prediction_sha = None
            changed_from_parent = None
        else:
            _require(raw.get("arm_label") == spec.arm_label, f"{label} arm changed")
            _require(
                raw.get("parent_arm_label") == spec.parent_arm_label,
                f"{label} parent changed",
            )
            prompt_field = "answer_prompt_messages_sha256"
            call_field = "answer_call_key_sha256"
            request_field = "answer_request_journal_sha256"
            response_field = "answer_response_journal_sha256"
            source_field = "s0_source_binding_sha256"
            parent_prediction_sha = _sha(
                raw.get("s0_control_prediction_sha256"),
                f"{label} parent prediction SHA-256",
            )
            changed_from_parent = _boolean(
                raw.get("changed_from_s0"), f"{label} parent change"
            )
            _require(
                changed_from_parent == (prediction_sha != parent_prediction_sha),
                f"{label} parent-change flag disagrees with predictions",
            )

        observations.append(
            LegacyRuntimeObservation(
                ordinal=ordinal,
                question_id=question_id,
                question_sha256=question_sha,
                dated_question_sha256=dated_sha,
                retrieval_question_part_sha256=retrieval_question_sha,
                source_binding_sha256=_sha(
                    raw.get(source_field), f"{label} source binding SHA-256"
                ),
                prompt_messages_sha256=_optional_sha(
                    raw.get(prompt_field), f"{label} prompt SHA-256"
                ),
                call_key_sha256=_optional_sha(
                    raw.get(call_field), f"{label} call-key SHA-256"
                ),
                request_journal_sha256=_optional_sha(
                    raw.get(request_field), f"{label} request journal SHA-256"
                ),
                response_journal_sha256=_optional_sha(
                    raw.get(response_field), f"{label} response journal SHA-256"
                ),
                prediction_text=prediction_text,
                prediction_sha256=prediction_sha,
                parent_prediction_sha256=parent_prediction_sha,
                changed_from_parent=changed_from_parent,
                source_row_sha256=identity_sha256(raw),
            )
        )
    return population_sha, retrieval_sha, tuple(observations)


def _score_observations(
    payload: Mapping[str, Any],
    spec: LegacyArmSpec,
    runtime: tuple[LegacyRuntimeObservation, ...],
    *,
    run_sha256: str,
    run_replay_sha256: str,
    population_sha256: str,
    retrieval_sha256: str,
) -> tuple[LegacyScoreAggregate, tuple[LegacyScoreObservation, ...]]:
    _require(payload.get("format") == JUDGE_FORMAT, "legacy judge format changed")
    _require(payload.get("arm_label") == spec.arm_label, "legacy judge arm changed")
    _require(
        payload.get("question_count") == spec.question_count,
        "legacy judge question count changed",
    )
    _require(
        payload.get("gold_loaded_only_after_answer_run_replay") is True,
        "legacy judge gold boundary changed",
    )
    _require(
        payload.get("explicit_gold_answer_text_persisted") is False,
        "legacy judge persisted explicit gold text",
    )
    _require(
        payload.get("arm_or_topology_labels_exposed_to_judge") is False,
        "legacy judge exposed arm/topology labels",
    )
    _require(
        payload.get("topology_loaded_only_after_judge_prompt_seal") is True,
        "legacy topology timing changed",
    )
    _require(
        payload.get("retained_request_token_state_bytes") == 0,
        "legacy judge retained request-token state",
    )
    _require(
        payload.get("logical_judgment_count") == spec.unique_judge_call_count,
        "legacy logical judge count changed",
    )
    _require(
        payload.get("unique_sol_completion_count") == spec.unique_judge_call_count,
        "legacy unique judge count changed",
    )
    expected_reuse = spec.parent_arm_label is not None
    _require(
        payload.get("unchanged_verdicts_reused_from_sealed_baseline") is expected_reuse,
        "legacy unchanged-verdict reuse changed",
    )

    campaign = _mapping(payload.get("campaign_binding"), "legacy judge binding")
    _require(
        campaign.get("format") == JUDGE_BINDING_FORMAT,
        "legacy judge binding format changed",
    )
    _require(campaign.get("arm_label") == spec.arm_label, "judge binding arm changed")
    _require(
        campaign.get("baseline_arm_label") == spec.judge_baseline_arm_label,
        "judge binding baseline arm changed",
    )
    _require(campaign.get("arm_run_sha256") == run_sha256, "judge run binding changed")
    _require(
        campaign.get("arm_run_replay_sha256") == run_replay_sha256,
        "judge replay binding changed",
    )
    _require(
        campaign.get("population_identity_sha256") == population_sha256,
        "judge population binding changed",
    )
    _require(
        campaign.get("retrieval_sha256") == retrieval_sha256,
        "judge retrieval binding changed",
    )
    _require(
        campaign.get("question_count") == spec.question_count,
        "judge binding question count changed",
    )
    if spec.expected_parent_run_sha256 is not None:
        _require(
            campaign.get("baseline_run_sha256") == spec.expected_parent_run_sha256,
            "judge parent-run binding changed",
        )
    _require(
        campaign.get("unique_judge_call_count") == spec.unique_judge_call_count,
        "judge binding unique-call count changed",
    )
    _require(campaign.get("retries") == 0, "legacy judge retries changed")

    rows = _rows(payload.get("questions"), "legacy judge questions")
    _require(len(rows) == spec.question_count, "legacy judge row count changed")
    scores: list[LegacyScoreObservation] = []
    for expected_ordinal, (raw, answer) in enumerate(zip(rows, runtime, strict=True)):
        label = f"legacy judge question {expected_ordinal}"
        ordinal = _integer(raw.get("ordinal"), f"{label} ordinal")
        _require(ordinal == expected_ordinal, f"{label} order changed")
        question_id = _text(raw.get("question_id"), f"{label} ID")
        question_sha = _sha(raw.get("question_sha256"), f"{label} question SHA-256")
        dated_sha = _sha(
            raw.get("dated_question_sha256"), f"{label} dated-question SHA-256"
        )
        prediction_sha = _sha(
            raw.get("prediction_sha256"), f"{label} prediction SHA-256"
        )
        _require(
            (ordinal, question_id, question_sha, dated_sha, prediction_sha)
            == (
                answer.ordinal,
                answer.question_id,
                answer.question_sha256,
                answer.dated_question_sha256,
                answer.prediction_sha256,
            ),
            f"{label} is not aligned to its runtime answer",
        )
        baseline_prediction_sha = _sha(
            raw.get("baseline_prediction_sha256"),
            f"{label} baseline prediction SHA-256",
        )
        changed = _boolean(
            raw.get("changed_from_baseline"), f"{label} prediction change"
        )
        _require(
            changed == (prediction_sha != baseline_prediction_sha),
            f"{label} prediction-change flag disagrees with predictions",
        )
        if answer.parent_prediction_sha256 is not None:
            _require(
                baseline_prediction_sha == answer.parent_prediction_sha256,
                f"{label} judge parent prediction changed",
            )
            _require(
                changed == answer.changed_from_parent,
                f"{label} runtime/judge parent change disagrees",
            )

        baseline_correct = _boolean(
            raw.get("baseline_correct"), f"{label} baseline correctness"
        )
        correct = _boolean(raw.get("correct"), f"{label} correctness")
        rescued = _boolean(raw.get("rescued"), f"{label} rescue")
        regressed = _boolean(raw.get("regressed"), f"{label} regression")
        _require(
            rescued == (not baseline_correct and correct),
            f"{label} rescue flag is inconsistent",
        )
        _require(
            regressed == (baseline_correct and not correct),
            f"{label} regression flag is inconsistent",
        )
        scores.append(
            LegacyScoreObservation(
                ordinal=ordinal,
                question_id=question_id,
                question_sha256=question_sha,
                dated_question_sha256=dated_sha,
                prediction_sha256=prediction_sha,
                baseline_prediction_sha256=baseline_prediction_sha,
                baseline_correct=baseline_correct,
                correct=correct,
                changed_from_baseline=changed,
                rescued=rescued,
                regressed=regressed,
                verdict_source=_text(
                    raw.get("verdict_source"), f"{label} verdict source"
                ),
                gold_answer_sha256=_sha(
                    raw.get("gold_answer_sha256"), f"{label} gold SHA-256"
                ),
                question_only_demand_class=_text(
                    raw.get("question_only_demand_class"), f"{label} demand class"
                ),
                evidence_topology_class=_text(
                    raw.get("evidence_topology_class"), f"{label} topology class"
                ),
                baseline_judge_row_sha256=_sha(
                    raw.get("baseline_judge_row_sha256"),
                    f"{label} baseline judge-row SHA-256",
                ),
                judge_verdict_sha256=_optional_sha(
                    raw.get("judge_verdict_sha256"),
                    f"{label} judge verdict SHA-256",
                ),
                judge_row_sha256=identity_sha256(raw),
            )
        )

    baseline_total = sum(row.baseline_correct for row in scores)
    candidate_total = sum(row.correct for row in scores)
    rescued_total = sum(row.rescued for row in scores)
    regressed_total = sum(row.regressed for row in scores)
    aggregate_raw = _mapping(payload.get("aggregate"), "legacy judge aggregate")
    aggregate = LegacyScoreAggregate(
        baseline_correct=_integer(
            aggregate_raw.get("baseline_correct"), "legacy baseline total"
        ),
        candidate_correct=_integer(
            aggregate_raw.get("candidate_correct"), "legacy candidate total"
        ),
        rescued=_integer(aggregate_raw.get("rescued"), "legacy rescue total"),
        regressed=_integer(
            aggregate_raw.get("regressed"), "legacy regression total"
        ),
        net_marginal=_integer(
            aggregate_raw.get("net_marginal"), "legacy net marginal"
        ),
        accepted_for_positive_only_composition=_boolean(
            aggregate_raw.get("accepted_for_positive_only_composition"),
            "legacy acceptance gate",
        ),
    )
    reproduced = (
        baseline_total,
        candidate_total,
        rescued_total,
        regressed_total,
        candidate_total - baseline_total,
    )
    declared = (
        aggregate.baseline_correct,
        aggregate.candidate_correct,
        aggregate.rescued,
        aggregate.regressed,
        aggregate.net_marginal,
    )
    expected = (
        spec.baseline_correct,
        spec.candidate_correct,
        spec.rescued,
        spec.regressed,
        spec.candidate_correct - spec.baseline_correct,
    )
    _require(declared == reproduced, "legacy judge aggregate does not reproduce rows")
    _require(declared == expected, "legacy judge aggregate changed from checkpoint")
    _require(
        aggregate.accepted_for_positive_only_composition
        is spec.accepted_for_positive_only_composition,
        "legacy judge acceptance decision changed",
    )
    return aggregate, tuple(scores)


def import_legacy_artifacts(
    paths: LegacyArtifactPaths, spec: LegacyArmSpec
) -> LegacyArmMigration:
    """Import one explicitly described arm without executing any provider code."""

    run = _read_expected(paths.run, spec.run_sha256, "legacy run")
    run_replay = _read_expected(
        paths.run_replay, spec.run_replay_sha256, "legacy run replay"
    )
    judge = _read_expected(paths.judge, spec.judge_sha256, "legacy judge")
    judge_replay = _read_expected(
        paths.judge_replay, spec.judge_replay_sha256, "legacy judge replay"
    )

    population_sha, retrieval_sha, runtime = _runtime_observations(run.payload, spec)
    replay_population, replay_retrieval, replay_runtime = _runtime_observations(
        run_replay.payload, spec
    )
    _require(
        (
            population_sha,
            retrieval_sha,
            tuple(
                row.projection(include_source_row_sha256=False) for row in runtime
            ),
        )
        == (
            replay_population,
            replay_retrieval,
            tuple(
                row.projection(include_source_row_sha256=False)
                for row in replay_runtime
            ),
        ),
        "legacy run/replay behavior projection changed",
    )

    aggregate, scores = _score_observations(
        judge.payload,
        spec,
        runtime,
        run_sha256=run.sha256,
        run_replay_sha256=run_replay.sha256,
        population_sha256=population_sha,
        retrieval_sha256=retrieval_sha,
    )
    replay_aggregate, replay_scores = _score_observations(
        judge_replay.payload,
        spec,
        replay_runtime,
        run_sha256=run.sha256,
        run_replay_sha256=run_replay.sha256,
        population_sha256=replay_population,
        retrieval_sha256=replay_retrieval,
    )
    _require(
        (
            aggregate,
            tuple(
                row.projection(include_source_row_sha256=False) for row in scores
            ),
        )
        == (
            replay_aggregate,
            tuple(
                row.projection(include_source_row_sha256=False)
                for row in replay_scores
            ),
        ),
        "legacy judge/replay score projection changed",
    )

    migration = LegacyArmMigration(
        spec=spec,
        run_artifact=_artifact_identity(run),
        run_replay_artifact=_artifact_identity(run_replay),
        judge_artifact=_artifact_identity(judge),
        judge_replay_artifact=_artifact_identity(judge_replay),
        population_identity_sha256=population_sha,
        retrieval_sha256=retrieval_sha,
        runtime_observations=runtime,
        score_observations=scores,
        score_aggregate=aggregate,
    )
    # Prove that both public projections are canonical-JSON serializable now,
    # rather than deferring a bad nested type until publication.
    canonical_json_bytes(migration.runtime_projection())
    canonical_json_bytes(migration.score_projection())
    canonical_json_bytes(migration.manifest_projection())
    return migration


def import_legacy_arm(
    arm_label: str, artifact_root: str | Path = DEFAULT_LEGACY_ROOT
) -> LegacyArmMigration:
    """Import one of the three registered historical arms."""

    try:
        spec = LEGACY_ARM_REGISTRY[arm_label]
    except KeyError as exc:
        raise LegacyImportError(f"unregistered legacy arm: {arm_label}") from exc
    return import_legacy_artifacts(LegacyArtifactPaths.under_root(artifact_root, spec), spec)


def _validate_checkpoint_relations(arms: tuple[LegacyArmMigration, ...]) -> None:
    s0 = arms[0]
    s0_runtime = s0.runtime_observations
    s0_scores = s0.score_observations
    for child in arms[1:]:
        _require(
            child.population_identity_sha256 == s0.population_identity_sha256,
            f"{child.spec.arm_label} population differs from S0",
        )
        _require(
            child.retrieval_sha256 == s0.retrieval_sha256,
            f"{child.spec.arm_label} retrieval differs from S0",
        )
        _require(
            child.spec.parent_arm_label == S0_ARM_LABEL,
            f"{child.spec.arm_label} is not an S0 child",
        )
        for parent_answer, child_answer, parent_score, child_score in zip(
            s0_runtime,
            child.runtime_observations,
            s0_scores,
            child.score_observations,
            strict=True,
        ):
            _require(
                (
                    child_answer.ordinal,
                    child_answer.question_id,
                    child_answer.question_sha256,
                    child_answer.dated_question_sha256,
                )
                == (
                    parent_answer.ordinal,
                    parent_answer.question_id,
                    parent_answer.question_sha256,
                    parent_answer.dated_question_sha256,
                ),
                f"{child.spec.arm_label} question order differs from S0",
            )
            _require(
                child_answer.parent_prediction_sha256
                == parent_answer.prediction_sha256,
                f"{child.spec.arm_label} parent prediction differs from S0",
            )
            _require(
                child_score.baseline_prediction_sha256
                == parent_answer.prediction_sha256,
                f"{child.spec.arm_label} judge baseline differs from S0",
            )
            _require(
                child_score.baseline_correct == parent_score.correct,
                f"{child.spec.arm_label} baseline verdict differs from S0",
            )


def import_legacy_checkpoint(
    artifact_root: str | Path = DEFAULT_LEGACY_ROOT,
) -> LegacyCheckpointMigration:
    """Import and cross-bind all three registered arms with zero provider calls."""

    arms = tuple(import_legacy_arm(label, artifact_root) for label in LEGACY_ARM_REGISTRY)
    _validate_checkpoint_relations(arms)
    checkpoint = LegacyCheckpointMigration(arms=arms)
    canonical_json_bytes(checkpoint.runtime_projection())
    canonical_json_bytes(checkpoint.score_projection())
    canonical_json_bytes(checkpoint.manifest_projection())
    return checkpoint


def load_legacy_campaign(
    root: str | Path = DEFAULT_LEGACY_ROOT,
) -> LegacyCampaignMigration:
    """Preferred public entry point for the complete sealed legacy campaign."""

    return import_legacy_checkpoint(root)


def runtime_projection(
    migration: LegacyArmMigration | LegacyCheckpointMigration,
) -> dict[str, object]:
    """Return the serializable, recursively gold-blind runtime projection."""

    projection = migration.runtime_projection()
    assert_gold_blind(projection)
    canonical_json_bytes(projection)
    return projection


def score_projection(
    migration: LegacyArmMigration | LegacyCheckpointMigration,
) -> dict[str, object]:
    """Return the serializable post-hoc score projection."""

    projection = migration.score_projection()
    canonical_json_bytes(projection)
    return projection


def manifest_projection(
    migration: LegacyArmMigration | LegacyCheckpointMigration,
) -> dict[str, object]:
    """Return the serializable source-identity and renderer manifest."""

    projection = migration.manifest_projection()
    canonical_json_bytes(projection)
    return projection


__all__ = [
    "DEFAULT_LEGACY_ROOT",
    "EXTERNAL_S1_LABEL",
    "LEGACY_ARM_REGISTRY",
    "LegacyArmMigration",
    "LegacyArmSpec",
    "LegacyArtifactIdentity",
    "LegacyArtifactPaths",
    "LegacyCampaignMigration",
    "LegacyCheckpointMigration",
    "LegacyImportError",
    "LegacyRuntimeObservation",
    "LegacyScoreAggregate",
    "LegacyScoreObservation",
    "S0_ARM_LABEL",
    "import_legacy_arm",
    "import_legacy_artifacts",
    "import_legacy_checkpoint",
    "load_legacy_campaign",
    "manifest_projection",
    "runtime_projection",
    "score_projection",
]
