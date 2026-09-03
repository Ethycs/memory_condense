"""Gold-blind adapter from the locked 100Q artifacts to EM v2.

This module intentionally lives under ``tools``.  The sealed fixed-S1 answer
campaign binds ``implementation_sha256()``, which hashes every Python file
under ``src/memory_condense``.  Keeping this read-only adapter outside that
tree lets the historical baseline pass its original strict validator before a
new treatment is constructed.

Only the merged retrieval artifact and its sealed baseline predictions are
accepted.  No benchmark dataset, category, reference answer, or labeled answer
source is an input.  The small question view below exposes exactly the fields
used by ``fast_em_fact_memory``: a dated question and cumulative S0/S1 evidence.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval._recall_guarded_cumulative_synthesis_contracts import (
    extract_stage_question,
)
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_em_fact_memory import (
    DEFAULT_EM_STAGE_ID,
    build_fact_compression_messages,
    episodic_neighborhood,
)
from memory_condense.eval.recall_guarded_cumulative_1m import STAGE_IDS
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastEvidence,
)
from memory_condense.eval.recall_guarded_cumulative_final_answer import (
    FIXED_STAGE_ID,
    RESPONDER_PROMPT_CAP,
    validate_final_answer_artifact,
)
from memory_condense.eval.recall_guarded_cumulative_validation_retrieval import (
    VALIDATION_MERGED_RETRIEVAL_FORMAT,
)
from tools.matched_eval.em_question_view import (
    LockedEMQuestionView,
    LockedEMStageView,
)


ADAPTER_FORMAT = "memory-condense-locked-fixed-s1-em-repair-adapter-v1"
PREFLIGHT_FORMAT = "memory-condense-locked-fixed-s1-em-repair-preflight-v1"
MEMORY_POLICY = "v2"
ANSWER_ARM = "facts"

_FORBIDDEN_RETRIEVAL_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "answer_session_ids",
        "category",
        "evidence_sources",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
    }
)
_SHA256 = frozenset("0123456789abcdef")


class LockedEMRepairAdapterError(ValueError):
    """Raised when sealed provenance or the gold firewall is not exact."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise LockedEMRepairAdapterError(
            f"{label} must be an exact lowercase SHA-256 digest"
        )
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LockedEMRepairAdapterError(f"{label} must be non-empty exact text")
    return value


def _require_source_bytes(value: object, label: str) -> str:
    """Accept nonblank source text without normalizing its sealed bytes."""

    if not isinstance(value, str) or not value.strip():
        raise LockedEMRepairAdapterError(f"{label} must be nonblank text")
    return value


def _forbidden_field(value: object, path: str = "retrieval") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}"
            if name.casefold() in _FORBIDDEN_RETRIEVAL_FIELDS:
                return child_path
            found = _forbidden_field(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            found = _forbidden_field(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _read_canonical_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    target = Path(path)
    expected_digest = _require_sha256(expected_sha256, f"{target} SHA-256")
    if target.is_symlink() or not target.is_file():
        raise LockedEMRepairAdapterError(
            f"artifact must be a regular non-symlink file: {target}"
        )
    raw = target.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedEMRepairAdapterError(
            f"artifact is not strict JSON: {target}"
        ) from exc
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise LockedEMRepairAdapterError(
            f"artifact is not canonical JSON: {target}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_digest:
        raise LockedEMRepairAdapterError(
            f"artifact SHA-256 differs from the expected identity: {target}"
        )
    sidecar = target.with_name(target.name + ".sha256")
    expected_sidecar = f"{digest}  {target.name}\n".encode("ascii")
    if (
        sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.read_bytes() != expected_sidecar
    ):
        raise LockedEMRepairAdapterError(
            f"artifact digest sidecar is invalid: {sidecar}"
        )
    return value, digest


@dataclass(frozen=True, slots=True)
class LockedBaselinePrediction:
    """One baseline prediction, detached from all benchmark gold."""

    text: str
    text_sha256: str
    final_answer_row_sha256: str


@dataclass(frozen=True, slots=True)
class LockedEMRepairRow:
    """Question view plus its independently sealed baseline candidate."""

    question: LockedEMQuestionView
    baseline: LockedBaselinePrediction
    binding_sha256: str


@dataclass(frozen=True, slots=True)
class LockedEMRepairPopulation:
    """Validated, gold-free inputs for a later routed EM repair campaign."""

    retrieval_sha256: str
    baseline_final_answers_sha256: str
    population_identity_sha256: str
    rows: tuple[LockedEMRepairRow, ...]
    binding_sha256: str

    @property
    def question_count(self) -> int:
        return len(self.rows)

    @property
    def questions(self) -> tuple[LockedEMQuestionView, ...]:
        return tuple(row.question for row in self.rows)


def _stage_view(raw: object, *, expected_stage_id: str) -> LockedEMStageView:
    if not isinstance(raw, Mapping) or raw.get("stage_id") != expected_stage_id:
        raise LockedEMRepairAdapterError("locked S0/S1 stage order changed")
    receipt = raw.get("stage_receipt")
    evidence = raw.get("evidence")
    if not isinstance(receipt, Mapping) or not isinstance(evidence, list):
        raise LockedEMRepairAdapterError("locked stage is missing receipt/evidence")
    rows: list[FastEvidence] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping) or set(item) != {
            "evidence_id",
            "source_id",
            "text",
        }:
            raise LockedEMRepairAdapterError(
                f"locked stage evidence {index} has a noncanonical shape"
            )
        rows.append(
            FastEvidence(
                evidence_id=_require_source_bytes(
                    item.get("evidence_id"), f"evidence {index} ID"
                ),
                source_id=_require_source_bytes(
                    item.get("source_id"), f"evidence {index} source ID"
                ),
                text=_require_source_bytes(
                    item.get("text"), f"evidence {index} text"
                ),
            )
        )
    return LockedEMStageView(
        stage_id=expected_stage_id,
        stage_receipt_sha256=_require_sha256(
            receipt.get("receipt_sha256"), f"{expected_stage_id} receipt"
        ),
        evidence_projection_sha256=_require_sha256(
            receipt.get("evidence_projection_sha256"),
            f"{expected_stage_id} evidence projection",
        ),
        evidence=tuple(rows),
    )


def _build_locked_em_repair_population(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
    baseline_final_answers: Mapping[str, Any],
    baseline_final_answers_sha256: str,
    validate_historical_artifact: bool,
) -> LockedEMRepairPopulation:
    """Project gold-free S0/S1 views, optionally invoking the old validator."""

    retrieval_digest = _require_sha256(retrieval_sha256, "retrieval SHA-256")
    baseline_digest = _require_sha256(
        baseline_final_answers_sha256, "baseline final-answer SHA-256"
    )
    if retrieval.get("format") != VALIDATION_MERGED_RETRIEVAL_FORMAT:
        raise LockedEMRepairAdapterError(
            "EM repair requires the merged locked validation retrieval"
        )
    if retrieval.get("gold_fields_present") is not False:
        raise LockedEMRepairAdapterError("retrieval does not attest a gold-free input")
    forbidden = _forbidden_field(retrieval)
    if forbidden is not None:
        raise LockedEMRepairAdapterError(
            f"retrieval contains forbidden answer-analysis field: {forbidden}"
        )
    if _digest(dict(retrieval)) != retrieval_digest:
        raise LockedEMRepairAdapterError("retrieval canonical SHA-256 changed")
    if _digest(dict(baseline_final_answers)) != baseline_digest:
        raise LockedEMRepairAdapterError(
            "baseline final-answer canonical SHA-256 changed"
        )

    # This is intentionally the historical validator, not a partial duplicate.
    # It validates the merged retrieval, every prompt/receipt, the campaign,
    # every baseline response journal binding, and the current sealed source
    # implementation identity before any projection is returned.
    if validate_historical_artifact:
        validate_final_answer_artifact(
            baseline_final_answers,
            retrieval=retrieval,
            artifact_sha256=baseline_digest,
            retrieval_sha256=retrieval_digest,
        )

    if (
        baseline_final_answers.get("gold_fields_present") is not False
        or baseline_final_answers.get("fixed_stage_id") != FIXED_STAGE_ID
        or FIXED_STAGE_ID != DEFAULT_EM_STAGE_ID
    ):
        raise LockedEMRepairAdapterError(
            "baseline is not the sealed gold-free fixed-S1 campaign"
        )
    raw_questions = retrieval.get("questions")
    part_hashes = retrieval.get("question_part_sha256s")
    baseline_rows = baseline_final_answers.get("questions")
    if not all(
        isinstance(value, list)
        for value in (raw_questions, part_hashes, baseline_rows)
    ):
        raise LockedEMRepairAdapterError("locked question populations are incomplete")
    assert isinstance(raw_questions, list)
    assert isinstance(part_hashes, list)
    assert isinstance(baseline_rows, list)
    count = retrieval.get("question_count")
    if (
        type(count) is not int
        or count < 1
        or len(raw_questions) != count
        or len(part_hashes) != count
        or len(baseline_rows) != count
    ):
        raise LockedEMRepairAdapterError("locked question counts differ")

    rows: list[LockedEMRepairRow] = []
    for ordinal, (raw_question, part_sha, baseline_row) in enumerate(
        zip(raw_questions, part_hashes, baseline_rows, strict=True)
    ):
        if not isinstance(raw_question, Mapping) or not isinstance(
            baseline_row, Mapping
        ):
            raise LockedEMRepairAdapterError("locked question row is not an object")
        question_id = _require_text(
            raw_question.get("question_id"), f"question {ordinal} ID"
        )
        if (
            raw_question.get("ordinal") != ordinal
            or baseline_row.get("ordinal") != ordinal
            or baseline_row.get("question_id") != question_id
        ):
            raise LockedEMRepairAdapterError("baseline question order changed")
        raw_stages = raw_question.get("stages")
        if not isinstance(raw_stages, list) or len(raw_stages) != len(STAGE_IDS):
            raise LockedEMRepairAdapterError("retrieval does not contain S0-S3")
        s0 = _stage_view(raw_stages[0], expected_stage_id=STAGE_IDS[0])
        s1 = _stage_view(raw_stages[1], expected_stage_id=FIXED_STAGE_ID)
        dated_s0 = extract_stage_question(raw_stages[0])
        dated_s1 = extract_stage_question(raw_stages[1])
        if dated_s0 != dated_s1:
            raise LockedEMRepairAdapterError("S0 and S1 carry different questions")
        dated_sha = _require_sha256(
            raw_question.get("dated_question_sha256"),
            f"question {ordinal} dated-question SHA-256",
        )
        if quote_sha256(dated_s1) != dated_sha:
            raise LockedEMRepairAdapterError("dated question no longer matches its seal")
        part_digest = _require_sha256(
            part_sha, f"question {ordinal} part SHA-256"
        )
        baseline_answer = baseline_row.get("answer")
        if not isinstance(baseline_answer, Mapping):
            raise LockedEMRepairAdapterError("baseline prediction is missing")
        prediction = _require_text(
            baseline_answer.get("text"), f"question {ordinal} baseline prediction"
        )
        prediction_sha = _require_sha256(
            baseline_answer.get("sha256"),
            f"question {ordinal} baseline prediction SHA-256",
        )
        if quote_sha256(prediction) != prediction_sha:
            raise LockedEMRepairAdapterError("baseline prediction seal changed")
        if (
            baseline_row.get("retrieval_question_part_sha256") != part_digest
            or baseline_row.get("stage_receipt_sha256") != s1.stage_receipt_sha256
            or baseline_row.get("evidence_projection_sha256")
            != s1.evidence_projection_sha256
        ):
            raise LockedEMRepairAdapterError(
                "baseline prediction changed its S1 retrieval binding"
            )
        view = LockedEMQuestionView(
            ordinal=ordinal,
            question_id=question_id,
            question_sha256=_require_sha256(
                raw_question.get("question_sha256"),
                f"question {ordinal} SHA-256",
            ),
            dated_question_sha256=dated_sha,
            retrieval_question_part_sha256=part_digest,
            dated_question=dated_s1,
            stages=(s0, s1),
        )
        # Reuse the exact post-selection EM projection and its protected-prefix
        # assertion now, rather than waiting for a provider-facing caller.
        episodic_neighborhood(view, stage_id=FIXED_STAGE_ID)  # type: ignore[arg-type]
        baseline = LockedBaselinePrediction(
            text=prediction,
            text_sha256=prediction_sha,
            final_answer_row_sha256=identity_sha256(dict(baseline_row)),
        )
        binding_body = {
            "format": ADAPTER_FORMAT + "-question",
            "ordinal": ordinal,
            "question_id": question_id,
            "question_sha256": view.question_sha256,
            "dated_question_sha256": dated_sha,
            "retrieval_question_part_sha256": part_digest,
            "s0_stage_receipt_sha256": s0.stage_receipt_sha256,
            "s1_stage_receipt_sha256": s1.stage_receipt_sha256,
            "s1_evidence_projection_sha256": s1.evidence_projection_sha256,
            "baseline_prediction_sha256": prediction_sha,
            "baseline_final_answer_row_sha256": baseline.final_answer_row_sha256,
        }
        rows.append(
            LockedEMRepairRow(
                question=view,
                baseline=baseline,
                binding_sha256=identity_sha256(binding_body),
            )
        )

    population_sha = _require_sha256(
        retrieval.get("population_identity_sha256"),
        "population identity SHA-256",
    )
    binding_body = {
        "format": ADAPTER_FORMAT,
        "retrieval_sha256": retrieval_digest,
        "baseline_final_answers_sha256": baseline_digest,
        "population_identity_sha256": population_sha,
        "fixed_stage_id": FIXED_STAGE_ID,
        "question_binding_sha256s": [row.binding_sha256 for row in rows],
    }
    return LockedEMRepairPopulation(
        retrieval_sha256=retrieval_digest,
        baseline_final_answers_sha256=baseline_digest,
        population_identity_sha256=population_sha,
        rows=tuple(rows),
        binding_sha256=identity_sha256(binding_body),
    )


def build_locked_em_repair_population(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
    baseline_final_answers: Mapping[str, Any],
    baseline_final_answers_sha256: str,
) -> LockedEMRepairPopulation:
    """Validate both sealed artifacts, then project gold-free S0/S1 views."""

    return _build_locked_em_repair_population(
        retrieval,
        retrieval_sha256=retrieval_sha256,
        baseline_final_answers=baseline_final_answers,
        baseline_final_answers_sha256=baseline_final_answers_sha256,
        validate_historical_artifact=True,
    )


def project_prevalidated_locked_em_repair_population(
    retrieval: Mapping[str, Any],
    *,
    retrieval_sha256: str,
    baseline_final_answers: Mapping[str, Any],
    baseline_final_answers_sha256: str,
    expected_historical_validator_binding_sha256: str,
) -> LockedEMRepairPopulation:
    """Reproject only after a sealed preflight binds historical validation."""

    population = _build_locked_em_repair_population(
        retrieval,
        retrieval_sha256=retrieval_sha256,
        baseline_final_answers=baseline_final_answers,
        baseline_final_answers_sha256=baseline_final_answers_sha256,
        validate_historical_artifact=False,
    )
    expected = _require_sha256(
        expected_historical_validator_binding_sha256,
        "historical validator binding SHA-256",
    )
    if population.binding_sha256 != expected:
        raise LockedEMRepairAdapterError(
            "prevalidated population changed its historical validator binding"
        )
    return population


def load_locked_em_repair_population(
    retrieval_path: str | Path,
    *,
    expected_retrieval_sha256: str,
    baseline_final_answers_path: str | Path,
    expected_baseline_final_answers_sha256: str,
) -> LockedEMRepairPopulation:
    """Read canonical artifacts and return the strictly validated population."""

    retrieval, retrieval_sha = _read_canonical_artifact(
        retrieval_path,
        expected_sha256=expected_retrieval_sha256,
    )
    baseline, baseline_sha = _read_canonical_artifact(
        baseline_final_answers_path,
        expected_sha256=expected_baseline_final_answers_sha256,
    )
    return build_locked_em_repair_population(
        retrieval,
        retrieval_sha256=retrieval_sha,
        baseline_final_answers=baseline,
        baseline_final_answers_sha256=baseline_sha,
    )


def build_compression_prompt_population(
    population: LockedEMRepairPopulation,
) -> tuple[tuple[dict[str, str], ...], ...]:
    """Build the complete gold-free EM-v2 compression prompt population."""

    if not isinstance(population, LockedEMRepairPopulation):
        raise TypeError("population must be a LockedEMRepairPopulation")
    return tuple(
        build_fact_compression_messages(
            row.question,  # type: ignore[arg-type]
            stage_id=FIXED_STAGE_ID,
            policy=MEMORY_POLICY,
        )
        for row in population.rows
    )


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise LockedEMRepairAdapterError("cannot summarize an empty population")
    return {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
        "total": sum(values),
    }


def preflight_locked_em_repair_population(
    population: LockedEMRepairPopulation,
) -> dict[str, Any]:
    """Return whole-population bounds without writes, provider access, or gold."""

    prompts = build_compression_prompt_population(population)
    prompt_preflight = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=RESPONDER_PROMPT_CAP,
    )
    prompt_tokens = [
        row.prompt_token_proxy for row in prompt_preflight.ordered_rows
    ]
    root_counts: list[int] = []
    delta_counts: list[int] = []
    for row in population.rows:
        root, delta = episodic_neighborhood(
            row.question,  # type: ignore[arg-type]
            stage_id=FIXED_STAGE_ID,
        )
        root_counts.append(len(root))
        delta_counts.append(len(delta))
    baseline_hashes = [row.baseline.text_sha256 for row in population.rows]
    return {
        "format": PREFLIGHT_FORMAT,
        "adapter_binding_sha256": population.binding_sha256,
        "retrieval_sha256": population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": population.population_identity_sha256,
        "question_count": population.question_count,
        "source_stage_id": FIXED_STAGE_ID,
        "memory_policy": MEMORY_POLICY,
        "answer_arm": ANSWER_ARM,
        "root_evidence_rows": _distribution(root_counts),
        "post_selection_em_delta_rows": {
            **_distribution(delta_counts),
            "zero_delta_questions": sum(value == 0 for value in delta_counts),
        },
        "compression_prompt_population": prompt_preflight.model_dump(),
        "compression_prompt_token_proxy": _distribution(prompt_tokens),
        "baseline_prediction_population_sha256": identity_sha256(
            baseline_hashes
        ),
        "baseline_prediction_count": len(baseline_hashes),
        "planned_compression_logical_calls": population.question_count,
        "dependent_answer_logical_calls": population.question_count,
        "dependent_answer_prompts_preflighted": False,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
    }


__all__ = [
    "ADAPTER_FORMAT",
    "ANSWER_ARM",
    "LockedBaselinePrediction",
    "LockedEMQuestionView",
    "LockedEMRepairAdapterError",
    "LockedEMRepairPopulation",
    "LockedEMRepairRow",
    "LockedEMStageView",
    "MEMORY_POLICY",
    "PREFLIGHT_FORMAT",
    "build_compression_prompt_population",
    "build_locked_em_repair_population",
    "load_locked_em_repair_population",
    "preflight_locked_em_repair_population",
    "project_prevalidated_locked_em_repair_population",
]
