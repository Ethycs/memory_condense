"""Gold-blind, read-only case ledger for the original cumulative 1M runs.

The benchmark has several answer artifacts that share a question population
but do *not* share a prompt contract.  This module projects those artifacts
into one per-question lineage without treating a cross-run score difference
as a causal method delta.

Gold-bearing score artifacts are deliberately accepted only by
``augment_case_ledger_posthoc``.  ``build_gold_blind_case_ledger`` rejects
score-shaped fields recursively, so callers cannot accidentally construct the
retrieval/answer lineage while reading labels or metrics.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from memory_condense.eval._artifact_json import (
    canonical_json_bytes as _canonical_json_bytes,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    STAGE_IDS,
    FastRetrievalArtifact,
    load_fast_retrieval_artifact,
)


GOLD_BLIND_LEDGER_FORMAT = "memory-condense-linear-case-ledger-v1"
POSTHOC_LEDGER_FORMAT = "memory-condense-linear-case-ledger-posthoc-v1"

SYNTHESIS_FORMAT = "memory-condense-recall-guarded-episodic-synthesis-v1"
FIXED_ANSWER_FORMAT = "memory-condense-recall-guarded-fixed-stage-final-answers-v1"
CAV_ANSWER_FORMAT = "memory-condense-fast-1m-cav-answers-v1"
HEBBIAN_ANSWER_FORMAT = "memory-condense-fast-1m-hebbian-answers-v1"

RETRIEVAL_SCORE_FORMAT = "memory-condense-recall-guarded-cumulative-1m-score-v1"
SYNTHESIS_SCORE_FORMAT = "memory-condense-recall-guarded-episodic-synthesis-score-v1"
FIXED_JUDGE_FORMAT = (
    "memory-condense-fixed-stage-final-answer-semantic-judge-score-v1"
)
CAV_SCORE_FORMAT = "memory-condense-fast-1m-cav-scores-v1"
HEBBIAN_SCORE_FORMAT = "memory-condense-fast-1m-hebbian-scores-v1"

_ANSWER_FORMATS = {
    SYNTHESIS_FORMAT,
    FIXED_ANSWER_FORMAT,
    CAV_ANSWER_FORMAT,
    HEBBIAN_ANSWER_FORMAT,
}
_SCORE_FORMATS = {
    RETRIEVAL_SCORE_FORMAT,
    SYNTHESIS_SCORE_FORMAT,
    FIXED_JUDGE_FORMAT,
    CAV_SCORE_FORMAT,
    HEBBIAN_SCORE_FORMAT,
}
_STAGE_ALIAS = {stage_id: f"S{index}" for index, stage_id in enumerate(STAGE_IDS)}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GOLD_BLIND_FORBIDDEN_KEYS = {
    "answer_present",
    "best_evidence_f1",
    "category",
    "exact_match",
    "expected_source_ids",
    "f1",
    "gold_answer",
    "gold_answer_sha256",
    "gold_loaded_posthoc",
    "semantic_correct",
}


class CaseLedgerValidationError(ValueError):
    """Raised when a supplied ledger artifact does not prove its lineage."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One explicitly named, sealed artifact consumed by the ledger.

    ``run_id`` is a caller-controlled stable label.  It is used to bind a
    later score artifact to the gold-blind answer artifact it evaluates.
    """

    run_id: str
    path: str | Path
    expected_sha256: str | None = None
    verify_sidecar: bool = True

    def __post_init__(self) -> None:
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("run_id must be a short filesystem-neutral identifier")
        if self.expected_sha256 is not None:
            _require_digest(self.expected_sha256, "expected_sha256")


def _identity_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


def _quote_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CaseLedgerValidationError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise CaseLedgerValidationError(f"{label} must be an array")
    return value


def _require_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise CaseLedgerValidationError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CaseLedgerValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise CaseLedgerValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise CaseLedgerValidationError(f"missing SHA-256 sidecar: {sidecar}")
    try:
        text = sidecar.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise CaseLedgerValidationError("SHA-256 sidecar must be ASCII") from exc
    lines = text.splitlines()
    if len(lines) != 1:
        raise CaseLedgerValidationError("SHA-256 sidecar must contain exactly one line")
    parts = lines[0].split()
    if not parts or parts[0] != digest:
        raise CaseLedgerValidationError("SHA-256 sidecar digest does not match")
    if len(parts) > 1 and parts[-1].lstrip("*") != path.name:
        raise CaseLedgerValidationError("SHA-256 sidecar names a different artifact")


def _read_sealed_json(spec: ArtifactSpec) -> tuple[dict[str, Any], str, Path]:
    path = Path(spec.path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if spec.expected_sha256 is not None and digest != spec.expected_sha256:
        raise CaseLedgerValidationError(
            f"{spec.run_id} SHA-256 mismatch ({digest} != {spec.expected_sha256})"
        )
    if not spec.verify_sidecar and spec.expected_sha256 is None:
        raise CaseLedgerValidationError(
            "an expected SHA-256 is required when sidecar verification is disabled"
        )
    if spec.verify_sidecar:
        _validate_sidecar(path, digest)
    try:
        payload = json.loads(raw, parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CaseLedgerValidationError(f"{path} is not finite UTF-8 JSON") from exc
    root = _require_mapping(payload, str(path))
    if raw != _canonical_json_bytes(root):
        raise CaseLedgerValidationError(f"{path} is not canonical JSON")
    return root, digest, path


def _walk_gold_firewall(value: object, label: str = "artifact") -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CaseLedgerValidationError(f"{label} has a non-string key")
            lowered = key.lower()
            if lowered in {"gold_fields_present", "gold_fields_read"}:
                if item is not False:
                    raise CaseLedgerValidationError(
                        f"{label}.{key} must remain false during ledger construction"
                    )
                continue
            if lowered == "gold_blind":
                if item is not True:
                    raise CaseLedgerValidationError(
                        f"{label}.{key} must remain true during ledger construction"
                    )
                continue
            if lowered in _GOLD_BLIND_FORBIDDEN_KEYS or (
                lowered.startswith("gold_")
            ):
                raise CaseLedgerValidationError(
                    f"gold-bearing field {key!r} is forbidden during ledger construction"
                )
            _walk_gold_firewall(item, f"{label}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _walk_gold_firewall(item, f"{label}[{index}]")


def _stage_alias(stage_id: object, label: str) -> str:
    stage = _require_string(stage_id, label)
    try:
        return _STAGE_ALIAS[stage]
    except KeyError as exc:
        raise CaseLedgerValidationError(f"{label} is not an S0-S3 stage") from exc


def _membership_sha256(kind: str, coordinates: Sequence[str]) -> str:
    return _identity_sha256(
        {
            "format": "memory-condense-case-ledger-membership-v1",
            "coordinate_kind": kind,
            "coordinates": list(coordinates),
        }
    )


def _observation_id(run_id: str, question_id: str, stage: str, arm: str) -> str:
    digest = _identity_sha256(
        {
            "format": "memory-condense-case-ledger-observation-id-v1",
            "run_id": run_id,
            "question_id": question_id,
            "stage": stage,
            "arm": arm,
        }
    )
    return f"obs-{digest[:24]}"


def _comparison_id(
    left_observation_id: str,
    right_observation_id: str,
    relation: str,
) -> str:
    digest = _identity_sha256(
        {
            "format": "memory-condense-case-ledger-comparison-id-v1",
            "left": left_observation_id,
            "right": right_observation_id,
            "relation": relation,
        }
    )
    return f"cmp-{digest[:24]}"


def _retrieval_prompt_contract(artifact: FastRetrievalArtifact) -> str:
    return _identity_sha256(
        {
            "format": "memory-condense-sealed-retrieval-prompt-contract-v1",
            "campaign_format": artifact.campaign_format,
            "stage_ids": list(artifact.stage_ids),
        }
    )


def _question_index(artifact: FastRetrievalArtifact) -> dict[str, Any]:
    return {question.question_id: question for question in artifact.questions}


def _require_population(
    root: Mapping[str, Any],
    artifact: FastRetrievalArtifact,
    *,
    label: str,
    retrieval_field: str = "retrieval_sha256",
) -> None:
    if root.get("question_count") != artifact.question_count:
        raise CaseLedgerValidationError(f"{label} question count changed")
    if root.get("population_identity_sha256") not in (
        None,
        artifact.population_identity_sha256,
    ):
        raise CaseLedgerValidationError(f"{label} population identity changed")
    if root.get(retrieval_field) != artifact.raw_sha256:
        raise CaseLedgerValidationError(f"{label} retrieval binding changed")


def _base_observation(
    *,
    run_id: str,
    family: str,
    artifact_sha256: str,
    question_ordinal: int,
    question_id: str,
    question_sha256: str,
    stage_id: str,
    arm_id: str,
    prediction: str | None,
    prompt_sha256: str,
    source_prompt_sha256: str | None,
    prompt_contract_sha256: str,
    responder_identity_sha256: str | None,
    membership_kind: str,
    membership: Sequence[str],
    intervention: str,
    alias_of: str | None = None,
) -> dict[str, Any]:
    stage = _stage_alias(stage_id, "observation.stage_id")
    if prediction is not None and (type(prediction) is not str or not prediction):
        raise CaseLedgerValidationError("prediction must be null or non-empty text")
    coordinates = tuple(
        _require_string(value, "membership coordinate") for value in membership
    )
    return {
        "observation_id": _observation_id(run_id, question_id, stage, arm_id),
        "run_id": run_id,
        "family": family,
        "artifact_sha256": artifact_sha256,
        "question_ordinal": question_ordinal,
        "question_id": question_id,
        "question_sha256": question_sha256,
        "stage": stage,
        "stage_id": stage_id,
        "arm_id": arm_id,
        "intervention": intervention,
        "prediction": prediction,
        "prediction_sha256": (
            None if prediction is None else _quote_sha256(prediction)
        ),
        "prompt_sha256": _require_digest(prompt_sha256, "prompt_sha256"),
        "source_prompt_sha256": (
            None
            if source_prompt_sha256 is None
            else _require_digest(source_prompt_sha256, "source_prompt_sha256")
        ),
        "prompt_contract_sha256": _require_digest(
            prompt_contract_sha256, "prompt_contract_sha256"
        ),
        "responder_identity_sha256": (
            None
            if responder_identity_sha256 is None
            else _require_digest(
                responder_identity_sha256, "responder_identity_sha256"
            )
        ),
        "membership_kind": membership_kind,
        "membership": list(coordinates),
        "membership_sha256": _membership_sha256(membership_kind, coordinates),
        "alias_of": alias_of,
    }


def _retrieval_run(
    spec: ArtifactSpec,
    artifact: FastRetrievalArtifact,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = _retrieval_prompt_contract(artifact)
    observations: list[dict[str, Any]] = []
    for question in artifact.questions:
        for stage in question.stages:
            observations.append(
                _base_observation(
                    run_id=spec.run_id,
                    family="retrieval",
                    artifact_sha256=artifact.raw_sha256,
                    question_ordinal=question.ordinal,
                    question_id=question.question_id,
                    question_sha256=question.question_sha256,
                    stage_id=stage.stage_id,
                    arm_id="retrieval",
                    prediction=None,
                    prompt_sha256=stage.prompt_messages_sha256,
                    source_prompt_sha256=stage.prompt_messages_sha256,
                    prompt_contract_sha256=contract,
                    responder_identity_sha256=None,
                    membership_kind="evidence_id",
                    membership=stage.evidence_ids,
                    intervention=_STAGE_ALIAS[stage.stage_id],
                )
            )
    run = {
        "run_id": spec.run_id,
        "family": "retrieval",
        "artifact_format": artifact.format,
        "artifact_path": str(Path(spec.path)),
        "artifact_sha256": artifact.raw_sha256,
        "population_identity_sha256": artifact.population_identity_sha256,
        "retrieval_sha256": artifact.raw_sha256,
        "prompt_contract_sha256": contract,
        "responder_identity_sha256": None,
        "stages": list(_STAGE_ALIAS.values()),
        "arms": ["retrieval"],
        "gold_fields_present": False,
    }
    return run, observations


def _ordered_question_rows(
    root: Mapping[str, Any],
    artifact: FastRetrievalArtifact,
    label: str,
) -> list[dict[str, Any]]:
    rows = _require_list(root.get("questions"), f"{label}.questions")
    if len(rows) != artifact.question_count:
        raise CaseLedgerValidationError(f"{label} question population changed")
    ordered: list[dict[str, Any]] = []
    for question, raw in zip(artifact.questions, rows, strict=True):
        row = _require_mapping(raw, f"{label}.questions[{question.ordinal}]")
        if row.get("question_id") != question.question_id or row.get(
            "ordinal", question.ordinal
        ) != question.ordinal:
            raise CaseLedgerValidationError(f"{label} question order changed")
        ordered.append(row)
    return ordered


def _parse_synthesis(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    artifact_sha256: str,
    artifact: FastRetrievalArtifact,
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_population(root, artifact, label=spec.run_id)
    stage_ids = tuple(_require_list(root.get("stage_ids"), "synthesis.stage_ids"))
    if stage_ids != STAGE_IDS[1:]:
        raise CaseLedgerValidationError("synthesis must cover the ordered S1-S3 stages")
    policy_sha = _require_digest(
        root.get("synthesis_prompt_policy_sha256"),
        "synthesis_prompt_policy_sha256",
    )
    request_sha = _require_digest(
        root.get("request_policy_sha256"), "request_policy_sha256"
    )
    contract = _identity_sha256(
        {
            "format": "memory-condense-episodic-synthesis-prompt-contract-v1",
            "synthesis_prompt_policy_sha256": policy_sha,
            "request_policy_sha256": request_sha,
        }
    )
    responder = _require_digest(
        root.get("runtime_identity_sha256"), "runtime_identity_sha256"
    )
    questions = _ordered_question_rows(root, artifact, spec.run_id)
    observations: list[dict[str, Any]] = []
    for question, row in zip(artifact.questions, questions, strict=True):
        stages = _require_list(row.get("stages"), "synthesis question stages")
        if len(stages) != 3:
            raise CaseLedgerValidationError("synthesis question must contain S1-S3")
        by_stage: dict[str, str] = {}
        for raw_stage, expected_stage in zip(stages, STAGE_IDS[1:], strict=True):
            stage = _require_mapping(raw_stage, "synthesis stage")
            if stage.get("stage_id") != expected_stage:
                raise CaseLedgerValidationError("synthesis stage order changed")
            answer = _require_mapping(stage.get("answer"), "synthesis answer")
            prediction = _require_string(answer.get("text"), "synthesis answer text")
            source_stage = question.stage(expected_stage)
            source_prompt_sha = _require_digest(
                stage.get("source_prompt_messages_sha256"),
                "source_prompt_messages_sha256",
            )
            if source_prompt_sha != source_stage.prompt_messages_sha256:
                raise CaseLedgerValidationError(
                    "synthesis source prompt no longer binds the retrieval stage"
                )
            reused = stage.get("reused_from_stage_id")
            alias_of = None
            if reused is not None:
                reused_stage = _stage_alias(reused, "reused_from_stage_id")
                alias_of = by_stage.get(reused_stage)
                if alias_of is None:
                    raise CaseLedgerValidationError(
                        "synthesis reuse must name an earlier stage in the question"
                    )
            observation = _base_observation(
                run_id=spec.run_id,
                family="synthesis",
                artifact_sha256=artifact_sha256,
                question_ordinal=question.ordinal,
                question_id=question.question_id,
                question_sha256=question.question_sha256,
                stage_id=expected_stage,
                arm_id="answer",
                prediction=prediction,
                prompt_sha256=_require_digest(
                    stage.get("prompt_messages_sha256"), "prompt_messages_sha256"
                ),
                source_prompt_sha256=source_prompt_sha,
                prompt_contract_sha256=contract,
                responder_identity_sha256=responder,
                membership_kind="evidence_id",
                membership=source_stage.evidence_ids,
                intervention="structured_synthesis",
                alias_of=alias_of,
            )
            observations.append(observation)
            by_stage[_STAGE_ALIAS[expected_stage]] = observation["observation_id"]
    run = {
        "run_id": spec.run_id,
        "family": "synthesis",
        "artifact_format": SYNTHESIS_FORMAT,
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256,
        "population_identity_sha256": artifact.population_identity_sha256,
        "retrieval_sha256": artifact.raw_sha256,
        "prompt_contract_sha256": contract,
        "responder_identity_sha256": responder,
        "stages": ["S1", "S2", "S3"],
        "arms": ["answer"],
        "gold_fields_present": False,
        "synthesis_prompt_policy_sha256": policy_sha,
        "request_policy_sha256": request_sha,
    }
    return run, observations


def _parse_fixed_answers(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    artifact_sha256: str,
    artifact: FastRetrievalArtifact,
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_population(root, artifact, label=spec.run_id)
    stage_id = _require_string(root.get("fixed_stage_id"), "fixed_stage_id")
    _stage_alias(stage_id, "fixed_stage_id")
    responder = _require_digest(
        root.get("runtime_identity_sha256"), "runtime_identity_sha256"
    )
    answer_policy = _require_digest(
        root.get("responder_prompt_policy_sha256"),
        "responder_prompt_policy_sha256",
    )
    # The fixed-stage runner sends the exact sealed retrieval messages.  Its
    # output/runtime policy remains separately bound in run metadata.
    contract = _retrieval_prompt_contract(artifact)
    observations: list[dict[str, Any]] = []
    for question, row in zip(
        artifact.questions,
        _ordered_question_rows(root, artifact, spec.run_id),
        strict=True,
    ):
        if row.get("fixed_stage_id") != stage_id:
            raise CaseLedgerValidationError("fixed answer stage changed within the run")
        source_stage = question.stage(stage_id)
        prompt_sha = _require_digest(
            row.get("provider_messages_sha256"), "provider_messages_sha256"
        )
        if prompt_sha != source_stage.prompt_messages_sha256:
            raise CaseLedgerValidationError(
                "fixed-stage answer no longer uses the exact retrieval prompt"
            )
        answer = _require_mapping(row.get("answer"), "fixed answer")
        prediction = _require_string(answer.get("text"), "fixed answer text")
        if answer.get("sha256") != _quote_sha256(prediction):
            raise CaseLedgerValidationError("fixed answer digest does not verify")
        observations.append(
            _base_observation(
                run_id=spec.run_id,
                family="fixed_stage",
                artifact_sha256=artifact_sha256,
                question_ordinal=question.ordinal,
                question_id=question.question_id,
                question_sha256=question.question_sha256,
                stage_id=stage_id,
                arm_id="answer",
                prediction=prediction,
                prompt_sha256=prompt_sha,
                source_prompt_sha256=prompt_sha,
                prompt_contract_sha256=contract,
                responder_identity_sha256=responder,
                membership_kind="evidence_id",
                membership=source_stage.evidence_ids,
                intervention="fixed_stage_answer",
            )
        )
    run = {
        "run_id": spec.run_id,
        "family": "fixed_stage",
        "artifact_format": FIXED_ANSWER_FORMAT,
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256,
        "population_identity_sha256": artifact.population_identity_sha256,
        "retrieval_sha256": artifact.raw_sha256,
        "prompt_contract_sha256": contract,
        "responder_identity_sha256": responder,
        "stages": [_STAGE_ALIAS[stage_id]],
        "arms": ["answer"],
        "gold_fields_present": False,
        "responder_prompt_policy_sha256": answer_policy,
    }
    return run, observations


def _answer_rows(root: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    rows = _require_list(root.get("answers"), f"{label}.answers")
    logical = _require_int(root.get("logical_answer_count"), "logical_answer_count")
    if logical != len(rows):
        raise CaseLedgerValidationError(f"{label} logical answer count changed")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"{label}.answers[{index}]")
        if row.get("logical_ordinal") != index:
            raise CaseLedgerValidationError(f"{label} logical answer order changed")
        result.append(row)
    return result


def _parse_cav_answers(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    artifact_sha256: str,
    artifact: FastRetrievalArtifact,
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_population(root, artifact, label=spec.run_id)
    prompt_population = _require_mapping(
        root.get("prompt_population"), "CAV prompt population"
    )
    selected = tuple(
        _require_list(
            prompt_population.get("selected_stage_ids"), "selected_stage_ids"
        )
    )
    if not selected or any(stage not in STAGE_IDS for stage in selected):
        raise CaseLedgerValidationError("CAV selected stages changed")
    feature_sha = _require_digest(
        root.get("feature_manifest_sha256"), "feature_manifest_sha256"
    )
    responder = _require_digest(
        _require_mapping(root.get("completion_batch"), "completion_batch").get(
            "runtime_identity_sha256"
        ),
        "runtime_identity_sha256",
    )
    contract = _identity_sha256(
        {
            "format": "memory-condense-fast-cav-evidence-catalog-contract-v1",
            "prompt_population_format": prompt_population.get("format"),
        }
    )
    answer_rows = _answer_rows(root, spec.run_id)
    logical_prompts = _require_list(
        prompt_population.get("logical_prompts"), "CAV logical prompts"
    )
    if len(logical_prompts) != len(answer_rows):
        raise CaseLedgerValidationError("CAV answer/prompt population changed")
    observations: list[dict[str, Any]] = []
    questions = _question_index(artifact)
    for row, raw_prompt in zip(answer_rows, logical_prompts, strict=True):
        prompt = _require_mapping(raw_prompt, "CAV logical prompt")
        for field in (
            "logical_ordinal",
            "question_ordinal",
            "question_id",
            "stage_id",
            "arm_id",
            "messages_sha256",
        ):
            if prompt.get(field) != row.get(field):
                raise CaseLedgerValidationError(
                    f"CAV answer no longer binds prompt field {field!r}"
                )
        ordinal = _require_int(row.get("question_ordinal"), "question_ordinal")
        question_id = _require_string(row.get("question_id"), "question_id")
        question = questions.get(question_id)
        if question is None or question.ordinal != ordinal:
            raise CaseLedgerValidationError("CAV answer question binding changed")
        stage_id = _require_string(row.get("stage_id"), "stage_id")
        if stage_id not in selected:
            raise CaseLedgerValidationError("CAV answer names an unselected stage")
        arm = _require_string(row.get("arm_id"), "arm_id")
        if arm not in {"original", "base", "treatment"}:
            raise CaseLedgerValidationError("CAV arm is unsupported")
        prediction = _require_string(row.get("prediction"), "prediction")
        if row.get("prediction_sha256") != _quote_sha256(prediction):
            raise CaseLedgerValidationError("CAV prediction digest does not verify")
        source_stage = question.stage(stage_id)
        evidence_ids = tuple(
            _require_string(value, "CAV evidence_id")
            for value in _require_list(prompt.get("evidence_ids"), "CAV evidence_ids")
        )
        if len(evidence_ids) != len(set(evidence_ids)) or set(evidence_ids) != set(
            source_stage.evidence_ids
        ):
            raise CaseLedgerValidationError(
                "CAV prompt must remain a permutation of source-stage evidence"
            )
        observations.append(
            _base_observation(
                run_id=spec.run_id,
                family="cav_proxy_readout",
                artifact_sha256=artifact_sha256,
                question_ordinal=ordinal,
                question_id=question_id,
                question_sha256=question.question_sha256,
                stage_id=stage_id,
                arm_id=arm,
                prediction=prediction,
                prompt_sha256=_require_digest(
                    row.get("messages_sha256"), "messages_sha256"
                ),
                source_prompt_sha256=None,
                prompt_contract_sha256=contract,
                responder_identity_sha256=responder,
                membership_kind="evidence_id",
                membership=evidence_ids,
                intervention=f"cav_linking_text_order_proxy_{arm}",
            )
        )
    run = {
        "run_id": spec.run_id,
        "family": "cav_proxy_readout",
        "method_role": "linking_technique_proxy_readout_ablation",
        "is_retrieval_layer": False,
        "is_direct_activation_injection": False,
        "artifact_format": CAV_ANSWER_FORMAT,
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256,
        "population_identity_sha256": artifact.population_identity_sha256,
        "retrieval_sha256": artifact.raw_sha256,
        "prompt_contract_sha256": contract,
        "responder_identity_sha256": responder,
        "stages": [_STAGE_ALIAS[stage] for stage in selected],
        "arms": ["original", "base", "treatment"],
        "gold_fields_present": False,
        "feature_manifest_sha256": feature_sha,
    }
    return run, observations


def _parse_hebbian_answers(
    spec: ArtifactSpec,
    root: Mapping[str, Any],
    artifact_sha256: str,
    artifact: FastRetrievalArtifact,
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = _require_mapping(root.get("experiment_binding"), "experiment_binding")
    if binding.get("retrieval_sha256") != artifact.raw_sha256:
        raise CaseLedgerValidationError("Hebbian retrieval binding changed")
    if binding.get("population_identity_sha256") != artifact.population_identity_sha256:
        raise CaseLedgerValidationError("Hebbian population identity changed")
    if root.get("question_count") != artifact.question_count:
        raise CaseLedgerValidationError("Hebbian question count changed")
    prompt_population = _require_mapping(
        root.get("prompt_population"), "Hebbian prompt population"
    )
    receipts = _require_list(
        prompt_population.get("question_receipts"), "Hebbian question receipts"
    )
    if len(receipts) != artifact.question_count:
        raise CaseLedgerValidationError("Hebbian question receipt count changed")
    catalog_formats = {
        _require_string(
            _require_mapping(receipt, "Hebbian question receipt").get(
                "catalog_format"
            ),
            "catalog_format",
        )
        for receipt in receipts
    }
    if len(catalog_formats) != 1:
        raise CaseLedgerValidationError("Hebbian catalog contract changed by question")
    contract = _identity_sha256(
        {
            "format": "memory-condense-fast-hebbian-catalog-contract-v1",
            "catalog_format": next(iter(catalog_formats)),
        }
    )
    responder = _require_digest(
        _require_mapping(root.get("completion_batch"), "completion_batch").get(
            "runtime_identity_sha256"
        ),
        "runtime_identity_sha256",
    )
    observations: list[dict[str, Any]] = []
    questions = _question_index(artifact)
    prior_by_prompt: dict[tuple[str, str], str] = {}
    for row in _answer_rows(root, spec.run_id):
        ordinal = _require_int(row.get("question_ordinal"), "question_ordinal")
        question_id = _require_string(row.get("question_id"), "question_id")
        question = questions.get(question_id)
        if question is None or question.ordinal != ordinal:
            raise CaseLedgerValidationError("Hebbian answer question binding changed")
        stage_id = _require_string(row.get("stage_id"), "stage_id")
        if stage_id != STAGE_IDS[0]:
            raise CaseLedgerValidationError("Hebbian answer must remain an S0 sibling")
        arm = _require_string(row.get("arm_id"), "arm_id")
        if arm not in {"base", "h1"}:
            raise CaseLedgerValidationError("Hebbian arm is unsupported")
        prediction = _require_string(row.get("prediction"), "prediction")
        if row.get("prediction_sha256") != _quote_sha256(prediction):
            raise CaseLedgerValidationError("Hebbian prediction digest does not verify")
        messages_sha = _require_digest(row.get("messages_sha256"), "messages_sha256")
        alias_of = prior_by_prompt.get((question_id, messages_sha))
        observation = _base_observation(
            run_id=spec.run_id,
            family="hebbian",
            artifact_sha256=artifact_sha256,
            question_ordinal=ordinal,
            question_id=question_id,
            question_sha256=question.question_sha256,
            stage_id=stage_id,
            arm_id=arm,
            prediction=prediction,
            prompt_sha256=messages_sha,
            source_prompt_sha256=None,
            prompt_contract_sha256=contract,
            responder_identity_sha256=responder,
            membership_kind="chunk_id",
            membership=_require_list(row.get("chunk_ids"), "chunk_ids"),
            intervention="s0_control" if arm == "base" else "hebbian_tail_replacement",
            alias_of=alias_of,
        )
        observations.append(observation)
        prior_by_prompt.setdefault((question_id, messages_sha), observation["observation_id"])
    run = {
        "run_id": spec.run_id,
        "family": "hebbian",
        "artifact_format": HEBBIAN_ANSWER_FORMAT,
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256,
        "population_identity_sha256": artifact.population_identity_sha256,
        "retrieval_sha256": artifact.raw_sha256,
        "prompt_contract_sha256": contract,
        "responder_identity_sha256": responder,
        "stages": ["S0"],
        "arms": ["base", "h1"],
        "gold_fields_present": False,
        "association_artifact_sha256": binding.get("association_artifact_sha256"),
    }
    return run, observations


def _parse_answer_artifact(
    spec: ArtifactSpec,
    artifact: FastRetrievalArtifact,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root, digest, path = _read_sealed_json(spec)
    artifact_format = root.get("format")
    if artifact_format in _SCORE_FORMATS:
        raise CaseLedgerValidationError(
            "score artifacts are forbidden in build_gold_blind_case_ledger"
        )
    if artifact_format not in _ANSWER_FORMATS:
        raise CaseLedgerValidationError(
            f"unsupported gold-blind artifact format: {artifact_format!r}"
        )
    if root.get("gold_fields_present") is not False:
        raise CaseLedgerValidationError("answer artifact must declare gold_fields_present=false")
    _walk_gold_firewall(root)
    if artifact_format == SYNTHESIS_FORMAT:
        return _parse_synthesis(spec, root, digest, artifact, path)
    if artifact_format == FIXED_ANSWER_FORMAT:
        return _parse_fixed_answers(spec, root, digest, artifact, path)
    if artifact_format == CAV_ANSWER_FORMAT:
        return _parse_cav_answers(spec, root, digest, artifact, path)
    return _parse_hebbian_answers(spec, root, digest, artifact, path)


def _membership_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if left["membership_kind"] != right["membership_kind"]:
        return "coordinate_kind_mismatch"
    left_ids = tuple(left["membership"])
    right_ids = tuple(right["membership"])
    if left_ids == right_ids:
        return "same_ordered_membership"
    if len(left_ids) == len(right_ids) and set(left_ids) == set(right_ids):
        return "same_set_reordered"
    if len(left_ids) <= len(right_ids) and left_ids == right_ids[: len(left_ids)]:
        return "ordered_prefix_addition"
    if len(left_ids) == len(right_ids):
        removed = set(left_ids) - set(right_ids)
        added = set(right_ids) - set(left_ids)
        if len(removed) == len(added) == 1:
            return "one_for_one_replacement"
    return "changed_membership"


def _comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    relation: str,
    scope: str,
    causal_comparable: bool,
    reasons: Iterable[str],
    membership_relation: str | None = None,
) -> dict[str, Any]:
    left_id = _require_string(left.get("observation_id"), "left observation id")
    right_id = _require_string(right.get("observation_id"), "right observation id")
    same_contract = left["prompt_contract_sha256"] == right["prompt_contract_sha256"]
    left_responder = left.get("responder_identity_sha256")
    right_responder = right.get("responder_identity_sha256")
    same_responder: bool | None
    if left_responder is None and right_responder is None:
        same_responder = None
    else:
        same_responder = left_responder == right_responder
    reason_values = tuple(dict.fromkeys(str(reason) for reason in reasons))
    return {
        "comparison_id": _comparison_id(left_id, right_id, relation),
        "left_observation_id": left_id,
        "right_observation_id": right_id,
        "question_id": left["question_id"],
        "relation": relation,
        "scope": scope,
        "causal_comparable": causal_comparable,
        "prompt_contract_match": same_contract,
        "responder_identity_match": same_responder,
        "membership_relation": membership_relation
        or _membership_relation(left, right),
        "reasons": list(reason_values),
    }


def _within_run_comparisons(
    run: Mapping[str, Any],
    observations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    family = run["family"]
    by_question: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        by_question.setdefault(observation["question_id"], []).append(observation)
    comparisons: list[dict[str, Any]] = []
    for rows in by_question.values():
        by_key = {(row["stage"], row["arm_id"]): row for row in rows}
        if family == "retrieval":
            for left_stage, right_stage in zip(
                ("S0", "S1", "S2"), ("S1", "S2", "S3"), strict=True
            ):
                comparisons.append(
                    _comparison(
                        by_key[(left_stage, "retrieval")],
                        by_key[(right_stage, "retrieval")],
                        relation="cumulative_retrieval_stage",
                        scope="retrieval_evidence",
                        causal_comparable=True,
                        reasons=("same sealed cumulative ladder",),
                    )
                )
        elif family == "synthesis":
            for left_stage, right_stage in zip(("S1", "S2"), ("S2", "S3"), strict=True):
                comparisons.append(
                    _comparison(
                        by_key[(left_stage, "answer")],
                        by_key[(right_stage, "answer")],
                        relation="cumulative_synthesis_stage",
                        scope="answer",
                        causal_comparable=True,
                        reasons=(
                            "same synthesis manifest and prompt policy",
                            "nested retrieval membership",
                        ),
                    )
                )
        elif family == "cav_proxy_readout":
            for left_arm, right_arm, relation in (
                ("original", "base", "unsteered_text_order_proxy"),
                ("base", "treatment", "cav_linking_text_order_proxy"),
            ):
                stages = sorted(
                    {stage for stage, arm in by_key if arm in {left_arm, right_arm}}
                )
                for stage in stages:
                    left = by_key.get((stage, left_arm))
                    right = by_key.get((stage, right_arm))
                    if left is None or right is None:
                        raise CaseLedgerValidationError("CAV arm population is incomplete")
                    comparisons.append(
                        _comparison(
                            left,
                            right,
                            relation=relation,
                            scope="proxy_readout_answer",
                            causal_comparable=True,
                            reasons=(
                                "same answer manifest and canonical renderer",
                                "CAV is a linking technique represented only by a text-order proxy",
                                "same evidence membership; ordering intervention only",
                            ),
                        )
                    )
        elif family == "hebbian":
            left = by_key.get(("S0", "base"))
            right = by_key.get(("S0", "h1"))
            if left is None or right is None:
                raise CaseLedgerValidationError("Hebbian base/H1 population is incomplete")
            comparisons.append(
                _comparison(
                    left,
                    right,
                    relation="hebbian_tail_replacement",
                    scope="answer",
                    causal_comparable=True,
                    reasons=(
                        "same answer manifest and compact renderer",
                        "bounded budget-neutral membership intervention",
                    ),
                )
            )
    return comparisons


def _lineage_comparisons(
    retrieval_run_id: str,
    observations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    retrieval = {
        (row["question_id"], row["stage"]): row
        for row in observations
        if row["run_id"] == retrieval_run_id
    }
    result: list[dict[str, Any]] = []
    for row in observations:
        if row["run_id"] == retrieval_run_id:
            continue
        source = retrieval.get((row["question_id"], row["stage"]))
        if source is None:
            raise CaseLedgerValidationError("answer observation has no retrieval source stage")
        exact_prompt = row["prompt_sha256"] == source["prompt_sha256"]
        source_binding = row.get("source_prompt_sha256") == source["prompt_sha256"]
        same_membership = row["membership_sha256"] == source["membership_sha256"]
        reasons = ["retrieval stage contains no matched answer outcome"]
        if exact_prompt:
            relation = "exact_retrieval_prompt_projection"
            reasons.append("provider message bytes match the retrieval stage")
        elif source_binding:
            relation = "derived_answer_prompt"
            reasons.append("source prompt is bound but the answer prompt is transformed")
        else:
            relation = "prompt_contract_mismatch"
            reasons.append("answer prompt was freshly rendered under another contract")
        if not same_membership:
            reasons.append("evidence coordinates or membership differ from the retrieval stage")
        result.append(
            _comparison(
                source,
                row,
                relation=relation,
                scope="lineage",
                causal_comparable=False,
                reasons=reasons,
            )
        )
    return result


def _cross_run_comparisons(
    runs: Sequence[Mapping[str, Any]],
    observations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_run: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for row in observations:
        by_run.setdefault(row["run_id"], {})[
            (row["question_id"], row["stage"], row["arm_id"])
        ] = row
    result: list[dict[str, Any]] = []
    answer_runs = [run for run in runs if run["family"] != "retrieval"]
    for left_run, right_run in itertools.combinations(answer_runs, 2):
        if left_run["family"] != right_run["family"]:
            continue
        common = sorted(set(by_run[left_run["run_id"]]) & set(by_run[right_run["run_id"]]))
        for key in common:
            left = by_run[left_run["run_id"]][key]
            right = by_run[right_run["run_id"]][key]
            prompt_match = left["prompt_contract_sha256"] == right["prompt_contract_sha256"]
            responder_match = (
                left["responder_identity_sha256"]
                == right["responder_identity_sha256"]
            )
            reasons = ["separate provider samples in separate answer manifests"]
            relation = "separate_run_unmatched"
            if not prompt_match:
                relation = "prompt_contract_mismatch"
                reasons.append("prompt policy or renderer changed between runs")
            if not responder_match:
                reasons.append("responder runtime identity changed between runs")
            result.append(
                _comparison(
                    left,
                    right,
                    relation=relation,
                    scope="cross_run_answer",
                    causal_comparable=False,
                    reasons=reasons,
                )
            )
    return result


def _seal_ledger(payload: dict[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise ValueError(f"payload already contains {field}")
    payload[field] = _identity_sha256(payload)
    return payload


def build_gold_blind_case_ledger(
    retrieval: ArtifactSpec,
    artifacts: Sequence[ArtifactSpec] = (),
) -> dict[str, Any]:
    """Build a per-question lineage without opening any score or gold input.

    The returned ledger contains predictions because predictions are provider
    outputs, not labels.  It contains no exact-match, F1, semantic verdict,
    category, expected source, or gold-answer field.
    """

    run_ids = [retrieval.run_id, *(spec.run_id for spec in artifacts)]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("ledger run_id values must be unique")
    retrieval_artifact = load_fast_retrieval_artifact(
        retrieval.path,
        expected_sha256=retrieval.expected_sha256,
        verify_sidecar=retrieval.verify_sidecar,
    )
    retrieval_run, retrieval_observations = _retrieval_run(
        retrieval, retrieval_artifact
    )
    runs = [retrieval_run]
    observations = list(retrieval_observations)
    comparisons = _within_run_comparisons(retrieval_run, retrieval_observations)
    for spec in artifacts:
        run, rows = _parse_answer_artifact(spec, retrieval_artifact)
        runs.append(run)
        observations.extend(rows)
        comparisons.extend(_within_run_comparisons(run, rows))
    comparisons.extend(_lineage_comparisons(retrieval.run_id, observations))
    comparisons.extend(_cross_run_comparisons(runs, observations))

    observation_ids = [row["observation_id"] for row in observations]
    if len(set(observation_ids)) != len(observation_ids):
        raise CaseLedgerValidationError("observation identities collided")
    comparison_ids = [row["comparison_id"] for row in comparisons]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise CaseLedgerValidationError("comparison identities collided")

    questions = []
    for question in retrieval_artifact.questions:
        question_rows = [
            row for row in observations if row["question_id"] == question.question_id
        ]
        question_observations = [row["observation_id"] for row in question_rows]
        question_comparisons = [
            row["comparison_id"]
            for row in comparisons
            if row["question_id"] == question.question_id
        ]
        questions.append(
            {
                "ordinal": question.ordinal,
                "question_id": question.question_id,
                "question_sha256": question.question_sha256,
                "question": question.question,
                "observation_ids": question_observations,
                "comparison_ids": question_comparisons,
                "layer_progression": [
                    {
                        "observation_id": row["observation_id"],
                        "run_id": row["run_id"],
                        "family": row["family"],
                        "stage": row["stage"],
                        "arm_id": row["arm_id"],
                        "intervention": row["intervention"],
                    }
                    for row in question_rows
                ],
            }
        )
    payload = {
        "format": GOLD_BLIND_LEDGER_FORMAT,
        "gold_fields_present": False,
        "gold_artifacts_read": 0,
        "population_identity_sha256": retrieval_artifact.population_identity_sha256,
        "retrieval_sha256": retrieval_artifact.raw_sha256,
        "question_count": retrieval_artifact.question_count,
        "run_count": len(runs),
        "observation_count": len(observations),
        "comparison_count": len(comparisons),
        "runs": runs,
        "questions": questions,
        "observations": observations,
        "comparisons": comparisons,
    }
    return _seal_ledger(payload, "ledger_sha256")


def augment_case_ledger_posthoc(
    ledger: Mapping[str, Any],
    score_artifacts: Sequence[ArtifactSpec],
) -> dict[str, Any]:
    """Attach gold-derived metrics through the explicit post-hoc boundary."""

    from memory_condense.eval._linear_case_ledger_posthoc import (
        augment_case_ledger_posthoc as _augment,
    )

    return _augment(ledger, score_artifacts)


__all__ = [
    "ArtifactSpec",
    "CAV_ANSWER_FORMAT",
    "CAV_SCORE_FORMAT",
    "CaseLedgerValidationError",
    "FIXED_ANSWER_FORMAT",
    "FIXED_JUDGE_FORMAT",
    "GOLD_BLIND_LEDGER_FORMAT",
    "HEBBIAN_ANSWER_FORMAT",
    "HEBBIAN_SCORE_FORMAT",
    "POSTHOC_LEDGER_FORMAT",
    "RETRIEVAL_SCORE_FORMAT",
    "SYNTHESIS_FORMAT",
    "SYNTHESIS_SCORE_FORMAT",
    "augment_case_ledger_posthoc",
    "build_gold_blind_case_ledger",
]
