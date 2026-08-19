"""Independent reconstruction of the locked LongMemEval stress population."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import (
    AuditError,
    canonical_sha256,
    require_int,
    require_list,
    require_mapping,
    require_text,
)
from .prompt import FrozenPromptRuntime


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    question: str
    answer: str
    category: str | None
    evidence: tuple[str, ...]
    evidence_sources: tuple[str, ...]
    question_date: str | None

    @property
    def dated_question(self) -> str:
        if not self.question_date:
            return self.question
        return f"[Question asked at {self.question_date}]\n{self.question}"

    def json_value(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "evidence": list(self.evidence),
            "evidence_sources": list(self.evidence_sources),
            "question_date": self.question_date,
        }


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    turns: tuple[tuple[str, str], ...]
    turn_source_ids: tuple[str | None, ...]
    questions: tuple[Question, ...]

    def json_value(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "turns": [list(turn) for turn in self.turns],
            "turn_source_ids": list(self.turn_source_ids),
            "questions": [question.json_value() for question in self.questions],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.json_value())


@dataclass(frozen=True, slots=True)
class PopulationPlan:
    samples: dict[str, Sample]
    question_to_sample: dict[str, str]
    transcript_tokens: dict[str, int]
    offsets: dict[str, int]


_ROLE_MAP = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "system": "assistant",
}


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                inner = part.get("text") or part.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
        return "\n".join(parts).strip()
    return ""


def _as_answer(value: Any) -> str:
    text = _as_text(value)
    if text:
        return text
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _records(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "samples", "records", "questions", "instances"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        return [value]
    return []


def load_longmemeval_bytes(payload: bytes, label: str = "LongMemEval dataset") -> list[Sample]:
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot parse {label}: {exc}") from exc
    samples: list[Sample] = []
    for index, raw in enumerate(_records(data)):
        if not isinstance(raw, dict):
            continue
        question_text = raw.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            continue
        sample_id = str(raw.get("question_id") or f"longmemeval_{index}")
        turns: list[tuple[str, str]] = []
        sources: list[str | None] = []
        sessions = raw.get("haystack_sessions") or []
        if not isinstance(sessions, list):
            sessions = []
        session_ids = _as_strings(raw.get("haystack_session_ids"))
        session_dates = _as_strings(raw.get("haystack_dates"))
        for session_index, session in enumerate(sessions):
            if not isinstance(session, list):
                continue
            source_id = (
                session_ids[session_index]
                if session_index < len(session_ids)
                else f"session_{session_index + 1}"
            )
            if session_index < len(session_dates) and session_dates[session_index]:
                turns.append(
                    (
                        "system",
                        f"[{source_id} took place at {session_dates[session_index]}]",
                    )
                )
                sources.append(source_id)
            for turn_index, turn in enumerate(session):
                if not isinstance(turn, dict):
                    continue
                text = _as_text(turn.get("content", turn.get("text")))
                if not text:
                    continue
                raw_role = turn.get("role")
                role = (
                    _ROLE_MAP.get(raw_role.strip().lower())
                    if isinstance(raw_role, str)
                    else None
                )
                turns.append((role or ("user" if turn_index % 2 == 0 else "assistant"), text))
                sources.append(source_id)
        answer = _as_answer(raw.get("answer"))
        if not answer:
            continue
        question = Question(
            question_id=sample_id,
            question=question_text.strip(),
            answer=answer,
            category=(
                str(raw["question_type"])
                if raw.get("question_type") is not None
                else None
            ),
            evidence=tuple(_as_strings(raw.get("answer_session_ids"))),
            evidence_sources=tuple(_as_strings(raw.get("answer_session_ids"))),
            question_date=_as_text(raw.get("question_date")) or None,
        )
        samples.append(
            Sample(
                sample_id=sample_id,
                turns=tuple(turns),
                turn_source_ids=tuple(sources),
                questions=(question,),
            )
        )
    return samples


def load_longmemeval(path: str | Path) -> list[Sample]:
    target = Path(path)
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read LongMemEval dataset {target}: {exc}") from exc
    return load_longmemeval_bytes(payload, f"LongMemEval dataset {target}")


def select_locked_validation(
    samples: list[Sample], split_manifest: dict[str, Any]
) -> list[Sample]:
    if split_manifest.get("format") != "memory-condense-locked-benchmark-split-v1":
        raise AuditError("split manifest format mismatch")
    if split_manifest.get("algorithm") != "stratified-largest-remainder-v1":
        raise AuditError("split manifest algorithm mismatch")
    salt = require_text(split_manifest.get("salt"), "split.salt")
    splits = require_mapping(split_manifest.get("splits"), "split.splits")
    split_names = list(splits)
    counts = {
        name: require_int(value, f"split.splits.{name}", minimum=1)
        for name, value in splits.items()
    }
    if sum(counts.values()) != len(samples):
        raise AuditError("locked split counts do not cover the parsed dataset")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise AuditError("dataset sample IDs are not unique")
    if "validation" not in counts:
        raise AuditError("locked split has no validation partition")

    strata: dict[str, list[Sample]] = {}
    for sample in samples:
        categories = sorted(
            {question.category or "uncategorized" for question in sample.questions}
        )
        stratum = "|".join(categories) or "uncategorized"
        strata.setdefault(stratum, []).append(sample)
    population = len(samples)
    quotas: dict[str, dict[str, int]] = {}
    remainders: dict[str, dict[str, float]] = {}
    assigned = {name: 0 for name in split_names}
    leftovers: dict[str, int] = {}
    for stratum, members in strata.items():
        quotas[stratum] = {}
        remainders[stratum] = {}
        for name in split_names:
            ideal = len(members) * counts[name] / population
            base = int(ideal)
            quotas[stratum][name] = base
            remainders[stratum][name] = ideal - base
            assigned[name] += base
        leftovers[stratum] = len(members) - sum(quotas[stratum].values())
    deficits = {name: counts[name] - assigned[name] for name in split_names}
    for stratum in sorted(strata):
        used: set[str] = set()
        for _ in range(leftovers[stratum]):
            choices = [name for name in split_names if deficits[name] > 0]
            pool = [name for name in choices if name not in used] or choices
            if not pool:
                raise AuditError("split apportionment exhausted capacity")
            name = max(
                pool,
                key=lambda candidate: (
                    remainders[stratum][candidate],
                    deficits[candidate],
                    -split_names.index(candidate),
                ),
            )
            quotas[stratum][name] += 1
            deficits[name] -= 1
            used.add(name)
    if any(deficits.values()):
        raise AuditError("split apportionment did not fill every partition")
    partitions: dict[str, list[Sample]] = {name: [] for name in split_names}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda sample: hashlib.sha256(
                f"{salt}\0{stratum}\0{sample.sample_id}".encode("utf-8")
            ).digest(),
        )
        offset = 0
        for name in split_names:
            count = quotas[stratum][name]
            partitions[name].extend(ordered[offset : offset + count])
            offset += count
    return sorted(
        partitions["validation"],
        key=lambda sample: hashlib.sha256(
            f"{salt}\0order\0{sample.sample_id}".encode("utf-8")
        ).digest(),
    )


def compose_stress_sample(
    samples: list[Sample],
    runtime: FrozenPromptRuntime,
    *,
    target_tokens: int,
    max_questions: int,
    question_offset: int,
) -> tuple[Sample, int]:
    turns: list[tuple[str, str]] = []
    sources: list[str | None] = []
    questions: list[Question] = []
    total_tokens = 0
    for sample in samples:
        turns.extend(sample.turns)
        if len(sample.turn_source_ids) != len(sample.turns):
            raise AuditError(f"sample {sample.sample_id} has misaligned source IDs")
        namespaced = tuple(
            None if source_id is None else f"{sample.sample_id}::{source_id}"
            for source_id in sample.turn_source_ids
        )
        sources.extend(namespaced)
        question_stop = question_offset + max_questions
        if len(questions) < question_stop:
            for question in sample.questions:
                if len(questions) >= question_stop:
                    break
                questions.append(
                    Question(
                        question_id=question.question_id,
                        question=question.question,
                        answer=question.answer,
                        category=question.category,
                        evidence=question.evidence,
                        evidence_sources=tuple(
                            f"{sample.sample_id}::{source_id}"
                            for source_id in question.evidence_sources
                        ),
                        question_date=question.question_date,
                    )
                )
        total_tokens += sum(runtime.count_tokens(text) for _role, text in sample.turns)
        if total_tokens >= target_tokens:
            break
    if total_tokens < target_tokens:
        raise AuditError("dataset cannot satisfy the frozen stress-token target")
    questions = questions[question_offset : question_offset + max_questions]
    if not questions:
        raise AuditError("stress question offset is outside the population")
    return (
        Sample(
            sample_id=f"context-stress-{target_tokens}",
            turns=tuple(turns),
            turn_source_ids=tuple(sources),
            questions=tuple(questions),
        ),
        total_tokens,
    )


def build_population_plan(
    dataset_payload: bytes,
    split_manifest: dict[str, Any],
    policy: dict[str, Any],
    runtime: FrozenPromptRuntime,
) -> PopulationPlan:
    evaluation = require_mapping(policy.get("evaluation"), "policy.evaluation")
    offsets = require_list(evaluation.get("sample_offsets"), "policy.evaluation.sample_offsets")
    if not offsets or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in offsets
    ) or len(offsets) != len(set(offsets)):
        raise AuditError("policy sample_offsets must be unique non-negative integers")
    target = require_int(
        evaluation.get("stress_context_tokens"),
        "policy.evaluation.stress_context_tokens",
        minimum=1,
    )
    questions_per = require_int(
        evaluation.get("stress_questions"),
        "policy.evaluation.stress_questions",
        minimum=1,
    )
    question_offset = require_int(
        evaluation.get("stress_question_offset"),
        "policy.evaluation.stress_question_offset",
    )
    validation = select_locked_validation(
        load_longmemeval_bytes(dataset_payload),
        split_manifest,
    )
    samples: dict[str, Sample] = {}
    question_to_sample: dict[str, str] = {}
    transcript_counts: dict[str, int] = {}
    sample_offsets: dict[str, int] = {}
    for offset_value in offsets:
        offset = int(offset_value)
        if offset >= len(validation):
            raise AuditError(f"sample offset {offset} is outside validation")
        sample, tokens = compose_stress_sample(
            validation[offset:],
            runtime,
            target_tokens=target,
            max_questions=questions_per,
            question_offset=question_offset,
        )
        if len(sample.questions) != questions_per:
            raise AuditError(f"stress shard {offset} has the wrong question count")
        digest = sample.sha256
        if digest in samples:
            raise AuditError("two stress offsets produced the same sample identity")
        samples[digest] = sample
        transcript_counts[digest] = tokens
        sample_offsets[digest] = offset
        for question in sample.questions:
            if question.question_id in question_to_sample:
                raise AuditError(f"question is repeated across shards: {question.question_id}")
            question_to_sample[question.question_id] = digest
    expected = require_int(
        evaluation.get("min_target_questions"),
        "policy.evaluation.min_target_questions",
        minimum=1,
    )
    if len(question_to_sample) != expected:
        raise AuditError(
            "reconstructed question population does not equal min_target_questions"
        )
    validation_ids = {
        question.question_id for sample in validation for question in sample.questions
    }
    if set(question_to_sample) != validation_ids:
        raise AuditError("stress shards do not cover the exact validation population")
    return PopulationPlan(
        samples=samples,
        question_to_sample=question_to_sample,
        transcript_tokens=transcript_counts,
        offsets=sample_offsets,
    )
