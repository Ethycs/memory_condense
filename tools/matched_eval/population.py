"""Lightweight sealed adapter from the locked retrieval to matched S0-v2.

This module deliberately does not rebuild the historical retrieval apparatus.
It verifies the publication boundary needed by a matched answer run: sealed
population/order bindings, cumulative evidence prefixes, receipt self-seals,
the protected S0 context, and the exact common-renderer prompt population.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import (
    identity_sha256 as legacy_identity_sha256,
    quote_sha256,
)
from memory_condense.eval.fast_completion_runtime import (
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools.confirmation_prompt_extract import extract_stage_question

from .artifacts import read_sealed_json
from .contracts import (
    ArtifactRef,
    EvaluationMemorySnapshot,
    EvidenceItem,
    MatchedEvalContractError,
    MemoryPacket,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .renderer import (
    RENDERER_ID,
    V3_RENDERER_ID,
    V4_RENDERER_ID,
    RenderedPrompt,
    render_memory_packet_for_id,
)


MERGED_RETRIEVAL_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-100q-retrieval-v1"
)
MERGED_QUESTION_FORMAT = (
    "memory-condense-recall-guarded-cumulative-validation-merged-query-v1"
)
POPULATION_FORMAT = "memory-condense-matched-s0-population-v2"
PREFLIGHT_FORMAT = "memory-condense-matched-s0-preflight-v2"
V3_POPULATION_FORMAT = "memory-condense-matched-s0-population-v3"
V3_PREFLIGHT_FORMAT = "memory-condense-matched-s0-preflight-v3"
V4_POPULATION_FORMAT = "memory-condense-matched-s0-population-v4"
V4_PREFLIGHT_FORMAT = "memory-condense-matched-s0-preflight-v4"
V4_POLICY_ID = "matched_eval_policy_v4"
V4_IMPLEMENTATION_ID = "tools_matched_eval_v4"

STAGE_IDS = (
    "causal_graph_coverage_predecessor",
    "direct_episode_additions",
    "representative_episode_additions",
    "artifact_global_closure_additions",
)
SOURCE_STAGE_ID = STAGE_IDS[0]
EXPECTED_QUESTION_COUNT = 100
DEFAULT_MAX_PROMPT_TOKENS = 8_000
EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)


def _renderer_formats(renderer_id: str) -> tuple[str, str]:
    if renderer_id == RENDERER_ID:
        return POPULATION_FORMAT, PREFLIGHT_FORMAT
    if renderer_id == V3_RENDERER_ID:
        return V3_POPULATION_FORMAT, V3_PREFLIGHT_FORMAT
    if renderer_id == V4_RENDERER_ID:
        return V4_POPULATION_FORMAT, V4_PREFLIGHT_FORMAT
    _fail(f"unsupported matched renderer identity: {renderer_id!r}")


class MatchedPopulationError(MatchedEvalContractError):
    """Raised when a sealed retrieval cannot be projected into matched S0."""


def _fail(message: str) -> None:
    raise MatchedPopulationError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    return value  # type: ignore[return-value]


def _rows(value: object, label: str) -> list[Mapping[str, Any]]:
    if type(value) is not list or not all(type(row) is dict for row in value):
        _fail(f"{label} must be an array of objects")
    return value  # type: ignore[return-value]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{label} must be non-empty exact text")
    return value


def _evidence_text(value: object, label: str) -> str:
    # The locked population contains one deliberately retained zero-length
    # excerpt.  Membership is protected, so preserve its exact text rather
    # than silently dropping or rewriting that evidence coordinate.
    if not isinstance(value, str):
        _fail(f"{label} must be exact text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except MatchedEvalContractError as exc:
        raise MatchedPopulationError(str(exc)) from exc


def _string_ids(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        _fail(f"{label} must be an array")
    ids = tuple(_text(item, f"{label} item") for item in value)
    if len(set(ids)) != len(ids):
        _fail(f"{label} must be unique")
    return ids


def _sealed_receipt(value: object, label: str) -> tuple[Mapping[str, Any], str]:
    receipt = _mapping(value, label)
    declared = _sha(receipt.get("receipt_sha256"), f"{label} receipt SHA-256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    if legacy_identity_sha256(body) != declared:
        _fail(f"{label} receipt self-seal changed")
    return receipt, declared


def _evidence_rows(
    value: object,
    *,
    label: str,
) -> tuple[EvidenceItem, ...]:
    rows = _rows(value, label)
    result: list[EvidenceItem] = []
    for ordinal, row in enumerate(rows):
        evidence_id = _text(row.get("evidence_id"), f"{label} {ordinal} ID")
        source_id = _text(row.get("source_id"), f"{label} {ordinal} source")
        text = _evidence_text(row.get("text"), f"{label} {ordinal} text")
        result.append(
            EvidenceItem(
                evidence_id=evidence_id,
                source_id=source_id,
                text=text,
                token_count=count_tokens(text),
            )
        )
    ids = tuple(row.evidence_id for row in result)
    if len(set(ids)) != len(ids):
        _fail(f"{label} evidence IDs must be unique")
    return tuple(result)


def _raw_question(dated_question: str) -> str:
    first, separator, remainder = dated_question.partition("\n")
    if (
        not separator
        or not first.startswith("[Question asked at ")
        or not first.endswith("]")
        or not remainder
    ):
        _fail("dated question changed its question boundary")
    return remainder


def _validate_root_context(
    evidence: tuple[EvidenceItem, ...],
    receipt: Mapping[str, Any],
) -> None:
    context = "\n".join(
        f"[{ordinal}] {row.text}" for ordinal, row in enumerate(evidence, start=1)
    )
    if (
        quote_sha256(context) != receipt.get("context_sha256")
        or count_tokens(context) != receipt.get("context_token_proxy")
    ):
        _fail("S0 evidence projection/context changed")


@dataclass(frozen=True, slots=True)
class MatchedS0Row:
    ordinal: int
    question_part_sha256: str
    source_stage_receipt_sha256: str
    packet: MemoryPacket
    rendered_prompt: RenderedPrompt

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise MatchedPopulationError("S0 row ordinal cannot be negative")
        require_sha256(self.question_part_sha256, "question part SHA-256")
        require_sha256(
            self.source_stage_receipt_sha256,
            "source-stage receipt SHA-256",
        )
        if self.packet.stage_id != SOURCE_STAGE_ID:
            raise MatchedPopulationError("S0 row changed its source stage")
        if self.rendered_prompt.packet_id != self.packet.packet_id:
            raise MatchedPopulationError("S0 row prompt changed its packet binding")

    def binding_projection(self) -> dict[str, object]:
        result: dict[str, object] = {
            "messages_sha256": self.rendered_prompt.messages_sha256,
            "ordinal": self.ordinal,
            "packet_id": self.packet.packet_id,
            "prompt_id": self.rendered_prompt.prompt_id,
            "prompt_token_proxy": self.rendered_prompt.total_prompt_token_proxy,
            "question_id": self.packet.question_id,
            "question_part_sha256": self.question_part_sha256,
            "source_stage_receipt_sha256": self.source_stage_receipt_sha256,
        }
        if self.rendered_prompt.renderer_id in {
            V3_RENDERER_ID,
            V4_RENDERER_ID,
        }:
            aliases = [
                row.projection() for row in self.rendered_prompt.alias_receipt
            ]
            result["alias_receipt"] = aliases
            result["alias_receipt_sha256"] = identity_sha256(aliases)
        return result


@dataclass(frozen=True, slots=True)
class MatchedS0Population:
    retrieval_sha256: str
    snapshot: EvaluationMemorySnapshot
    rows: tuple[MatchedS0Row, ...]
    prompt_population: FastPromptPopulation
    max_prompt_tokens: int
    renderer_id: str = RENDERER_ID

    def __post_init__(self) -> None:
        require_sha256(self.retrieval_sha256, "retrieval SHA-256")
        if not self.rows:
            raise MatchedPopulationError("S0 population cannot be empty")
        ordinals = tuple(row.ordinal for row in self.rows)
        if any(ordinal < 0 for ordinal in ordinals) or ordinals != tuple(
            sorted(set(ordinals))
        ):
            raise MatchedPopulationError("S0 population row order changed")
        if len({row.packet.question_id for row in self.rows}) != len(self.rows):
            raise MatchedPopulationError("S0 population question IDs must be unique")
        if self.prompt_population.logical_prompt_count != len(self.rows):
            raise MatchedPopulationError("S0 prompt population count changed")
        observed = tuple(
            row.rendered_prompt.messages_sha256 for row in self.rows
        )
        declared = tuple(
            row.messages_sha256 for row in self.prompt_population.ordered_rows
        )
        if observed != declared:
            raise MatchedPopulationError("S0 prompt population order changed")
        if self.max_prompt_tokens != self.prompt_population.max_prompt_token_proxy:
            raise MatchedPopulationError("S0 prompt cap changed")
        _renderer_formats(self.renderer_id)
        if self.snapshot.renderer_id != self.renderer_id or any(
            row.rendered_prompt.renderer_id != self.renderer_id for row in self.rows
        ):
            raise MatchedPopulationError("S0 population renderer binding changed")

    @property
    def question_count(self) -> int:
        return len(self.rows)

    @property
    def population_id(self) -> str:
        population_format, _preflight_format = _renderer_formats(self.renderer_id)
        return identity_sha256(
            {
                "format": population_format,
                "renderer_id": self.renderer_id,
                "retrieval_sha256": self.retrieval_sha256,
                "rows": [row.binding_projection() for row in self.rows],
                "snapshot_id": self.snapshot.snapshot_id,
            }
        )

    def preflight_projection(self) -> dict[str, object]:
        _population_format, preflight_format = _renderer_formats(self.renderer_id)
        projection: dict[str, object] = {
            "format": preflight_format,
            "gold_loaded": False,
            "hard_prompt_token_cap": self.max_prompt_tokens,
            "logical_prompt_count": self.prompt_population.logical_prompt_count,
            "matched_population_id": self.population_id,
            "new_provider_calls": 0,
            "observed_max_prompt_token_proxy": max(
                row.prompt_token_proxy for row in self.prompt_population.ordered_rows
            ),
            "ordered_rows": [row.binding_projection() for row in self.rows],
            "prompt_population": self.prompt_population.model_dump(),
            "prompt_population_sha256": (
                self.prompt_population.prompt_population_sha256
            ),
            "provider_calls": 0,
            "question_count": self.question_count,
            "renderer_id": self.renderer_id,
            "required_authorized_provider_calls": (
                self.prompt_population.unique_prompt_count
            ),
            "retained_request_token_state_bytes": 0,
            "retrieval_sha256": self.retrieval_sha256,
            "snapshot": self.snapshot.projection(),
            "snapshot_id": self.snapshot.snapshot_id,
            "unique_prompt_count": self.prompt_population.unique_prompt_count,
        }
        assert_gold_blind(projection, path="s0_v2_preflight")
        return projection

    @property
    def preflight_sha256(self) -> str:
        return identity_sha256(self.preflight_projection())


def _project_question(
    question: Mapping[str, Any],
    *,
    ordinal: int,
    question_part_sha256: str,
    population_sha256: str,
    renderer_id: str,
) -> MatchedS0Row:
    if question.get("format") != MERGED_QUESTION_FORMAT:
        _fail(f"question {ordinal} format changed")
    if question.get("population_identity_sha256") != population_sha256:
        _fail(f"question {ordinal} population binding changed")
    if _integer(question.get("ordinal"), f"question {ordinal} ordinal") != ordinal:
        _fail(f"question {ordinal} order binding changed")
    if type(question.get("provider_calls")) is not int or question.get("provider_calls") != 0:
        _fail(f"question {ordinal} contains provider calls")

    predecessor, predecessor_sha = _sealed_receipt(
        question.get("predecessor_receipt"),
        f"question {ordinal} predecessor",
    )
    stages = _rows(question.get("stages"), f"question {ordinal} stages")
    if tuple(stage.get("stage_id") for stage in stages) != STAGE_IDS:
        _fail(f"question {ordinal} stage order changed")
    if tuple(question.get("stage_ids", ())) != STAGE_IDS:
        _fail(f"question {ordinal} declared stage order changed")

    parent_ids: tuple[str, ...] = ()
    parent_receipt_sha: str | None = None
    dated_question: str | None = None
    root_evidence: tuple[EvidenceItem, ...] | None = None
    root_receipt_sha: str | None = None
    for stage_index, (expected_stage_id, stage) in enumerate(
        zip(STAGE_IDS, stages, strict=True)
    ):
        label = f"question {ordinal} stage {expected_stage_id}"
        receipt, receipt_sha = _sealed_receipt(stage.get("stage_receipt"), label)
        if receipt.get("stage_id") != expected_stage_id:
            _fail(f"{label} receipt stage binding changed")
        evidence = _evidence_rows(stage.get("evidence"), label=f"{label} evidence")
        evidence_ids = tuple(row.evidence_id for row in evidence)
        selected_ids = _string_ids(
            receipt.get("selected_evidence_ids"), f"{label} selected evidence"
        )
        if evidence_ids != selected_ids:
            _fail(f"{label} evidence membership/order changed")
        declared_parent_ids = _string_ids(
            receipt.get("parent_evidence_ids"), f"{label} parent evidence"
        )
        if declared_parent_ids != parent_ids or evidence_ids[: len(parent_ids)] != parent_ids:
            _fail(f"{label} cumulative parent prefix changed")
        expected_added_ids = evidence_ids[len(parent_ids) :]
        if _string_ids(
            receipt.get("added_evidence_ids"), f"{label} added evidence"
        ) != expected_added_ids:
            _fail(f"{label} selected-then-added suffix changed")
        if receipt.get("parent_stage_receipt_sha256") != parent_receipt_sha:
            _fail(f"{label} parent receipt binding changed")
        messages = stage.get("provider_messages")
        if type(messages) is not list:
            _fail(f"{label} provider messages are missing")
        if legacy_identity_sha256(messages) != receipt.get("prompt_messages_sha256"):
            _fail(f"{label} provider prompt seal changed")
        try:
            stage_question = extract_stage_question(stage)
        except (TypeError, ValueError) as exc:
            raise MatchedPopulationError(f"{label} question extraction failed") from exc
        if dated_question is None:
            dated_question = stage_question
        elif stage_question != dated_question:
            _fail(f"{label} dated question changed across stages")

        if stage_index == 0:
            if receipt.get("method_evidence_sha256") != predecessor_sha:
                _fail(f"{label} predecessor binding changed")
            _validate_root_context(evidence, receipt)
            root_evidence = evidence
            root_receipt_sha = receipt_sha
        parent_ids = evidence_ids
        parent_receipt_sha = receipt_sha

    assert dated_question is not None
    assert root_evidence is not None
    assert root_receipt_sha is not None
    question_id = _text(question.get("question_id"), f"question {ordinal} ID")
    if question.get("question_id_sha256") != legacy_identity_sha256(
        {"question_id": question_id}
    ):
        _fail(f"question {ordinal} ID binding changed")
    question_sha = _sha(
        question.get("question_sha256"), f"question {ordinal} question SHA-256"
    )
    dated_sha = _sha(
        question.get("dated_question_sha256"),
        f"question {ordinal} dated-question SHA-256",
    )
    if quote_sha256(dated_question) != dated_sha:
        _fail(f"question {ordinal} dated-question binding changed")
    if quote_sha256(_raw_question(dated_question)) != question_sha:
        _fail(f"question {ordinal} raw-question binding changed")

    retrieval_receipt, _ = _sealed_receipt(
        question.get("retrieval_receipt"), f"question {ordinal} retrieval"
    )
    if (
        retrieval_receipt.get("predecessor_receipt_sha256") != predecessor_sha
        or tuple(retrieval_receipt.get("final_evidence_ids", ())) != parent_ids
        or retrieval_receipt.get("prompt_messages_sha256")
        != stages[-1]["stage_receipt"].get("prompt_messages_sha256")
    ):
        _fail(f"question {ordinal} final retrieval binding changed")

    packet = MemoryPacket(
        question_id=question_id,
        question_sha256=question_sha,
        dated_question=dated_question,
        dated_question_sha256=dated_sha,
        stage_id=SOURCE_STAGE_ID,
        protected_evidence=root_evidence,
    )
    return MatchedS0Row(
        ordinal=ordinal,
        question_part_sha256=question_part_sha256,
        source_stage_receipt_sha256=root_receipt_sha,
        packet=packet,
        rendered_prompt=render_memory_packet_for_id(
            packet,
            renderer_id=renderer_id,
        ),
    )


def load_s0_population(
    retrieval_path: str | Path,
    *,
    expected_retrieval_sha256: str | None = None,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    renderer_id: str = RENDERER_ID,
) -> MatchedS0Population:
    """Load and render a sealed S0 population without executing providers."""

    _renderer_formats(renderer_id)
    artifact = read_sealed_json(retrieval_path)
    if expected_retrieval_sha256 is not None:
        _sha(expected_retrieval_sha256, "expected retrieval SHA-256")
        if artifact.sha256 != expected_retrieval_sha256:
            _fail("sealed retrieval SHA-256 changed from its pinned checkpoint")
    retrieval = artifact.payload
    assert_gold_blind(retrieval, path="sealed_retrieval")
    if retrieval.get("format") != MERGED_RETRIEVAL_FORMAT:
        _fail("sealed retrieval format changed")
    if retrieval.get("gold_fields_present") is not False:
        _fail("sealed retrieval crossed the gold firewall")
    if type(retrieval.get("provider_calls")) is not int or retrieval.get("provider_calls") != 0:
        _fail("sealed retrieval contains provider calls")
    if type(expected_question_count) is not int or expected_question_count < 1:
        _fail("expected question count must be positive")
    if (
        type(retrieval.get("question_count")) is not int
        or retrieval.get("question_count") != expected_question_count
    ):
        _fail("sealed retrieval question count changed")
    if tuple(retrieval.get("stage_ids", ())) != STAGE_IDS:
        _fail("sealed retrieval stage order changed")

    population = _mapping(retrieval.get("population_identity"), "population identity")
    population_sha = _sha(
        retrieval.get("population_identity_sha256"), "population identity SHA-256"
    )
    if population.get("population_identity_sha256") != population_sha:
        _fail("population identity outer binding changed")
    population_body = dict(population)
    population_body.pop("population_identity_sha256", None)
    if legacy_identity_sha256(population_body) != population_sha:
        _fail("population identity self-seal changed")
    if (
        population.get("gold_fields_present") is not False
        or type(population.get("question_count")) is not int
        or population.get("question_count") != expected_question_count
    ):
        _fail("population identity scope changed")

    ordered_question_ids = _string_ids(
        population.get("ordered_question_id_sha256s"),
        "population ordered question IDs",
    )
    ordered_probes = _string_ids(
        population.get("ordered_question_probe_sha256s"),
        "population ordered question probes",
    )
    if len(ordered_question_ids) != expected_question_count or len(ordered_probes) != expected_question_count:
        _fail("population question order count changed")

    questions = _rows(retrieval.get("questions"), "retrieval questions")
    declared_part_hashes = _string_ids(
        retrieval.get("question_part_sha256s"), "retrieval question-part hashes"
    )
    if len(questions) != expected_question_count or len(declared_part_hashes) != len(questions):
        _fail("retrieval question-part count changed")

    matched_rows: list[MatchedS0Row] = []
    seen_ids: set[str] = set()
    for ordinal, (question, declared_part_sha) in enumerate(
        zip(questions, declared_part_hashes, strict=True)
    ):
        observed_part_sha = hashlib.sha256(canonical_json_bytes(question)).hexdigest()
        if observed_part_sha != declared_part_sha:
            _fail(f"question {ordinal} part binding changed")
        if question.get("question_id_sha256") != ordered_question_ids[ordinal]:
            _fail(f"question {ordinal} ID/order binding changed")
        if question.get("probe_identity_sha256") != ordered_probes[ordinal]:
            _fail(f"question {ordinal} probe/order binding changed")
        question_id = _text(question.get("question_id"), f"question {ordinal} ID")
        if question_id in seen_ids:
            _fail(f"question {ordinal} duplicates question ID {question_id!r}")
        seen_ids.add(question_id)
        matched_rows.append(
            _project_question(
                question,
                ordinal=ordinal,
                question_part_sha256=declared_part_sha,
                population_sha256=population_sha,
                renderer_id=renderer_id,
            )
        )

    rows_tuple = tuple(matched_rows)
    prompt_population = preflight_fast_completion_prompts(
        tuple(row.rendered_prompt.messages for row in rows_tuple),
        max_prompt_tokens=max_prompt_tokens,
    )
    snapshot = EvaluationMemorySnapshot(
        population_identity_sha256=population_sha,
        question_order_sha256=identity_sha256(
            {"ordered_question_id_sha256s": list(ordered_question_ids)}
        ),
        source_artifacts=(
            ArtifactRef(
                role="sealed_retrieval",
                sha256=artifact.sha256,
            ),
        ),
        policy_id=(
            "matched_eval_policy_v2"
            if renderer_id == RENDERER_ID
            else (
                "matched_eval_policy_v3"
                if renderer_id == V3_RENDERER_ID
                else V4_POLICY_ID
            )
        ),
        renderer_id=renderer_id,
        implementation_id=(
            "tools_matched_eval_v2"
            if renderer_id == RENDERER_ID
            else (
                "tools_matched_eval_v3"
                if renderer_id == V3_RENDERER_ID
                else V4_IMPLEMENTATION_ID
            )
        ),
    )
    return MatchedS0Population(
        retrieval_sha256=artifact.sha256,
        snapshot=snapshot,
        rows=rows_tuple,
        prompt_population=prompt_population,
        max_prompt_tokens=max_prompt_tokens,
        renderer_id=renderer_id,
    )


def select_s0_population(
    population: MatchedS0Population,
    ordinals: Sequence[int],
) -> MatchedS0Population:
    """Create a sealed-order diagnostic view without changing source rows."""

    selected_ordinals = tuple(ordinals)
    if (
        not selected_ordinals
        or any(type(value) is not int or value < 0 for value in selected_ordinals)
        or selected_ordinals != tuple(sorted(set(selected_ordinals)))
    ):
        _fail("selected S0 ordinals must be a non-empty sorted unique sequence")
    by_ordinal = {row.ordinal: row for row in population.rows}
    if any(ordinal not in by_ordinal for ordinal in selected_ordinals):
        _fail("selected S0 ordinal is outside the loaded population")
    rows = tuple(by_ordinal[ordinal] for ordinal in selected_ordinals)
    prompt_population = preflight_fast_completion_prompts(
        tuple(row.rendered_prompt.messages for row in rows),
        max_prompt_tokens=population.max_prompt_tokens,
    )
    snapshot = EvaluationMemorySnapshot(
        population_identity_sha256=population.snapshot.population_identity_sha256,
        question_order_sha256=identity_sha256(
            {
                "selected_question_id_sha256s": [
                    identity_sha256({"question_id": row.packet.question_id})
                    for row in rows
                ],
                "source_snapshot_id": population.snapshot.snapshot_id,
            }
        ),
        source_artifacts=population.snapshot.source_artifacts,
        overlay_revisions=population.snapshot.overlay_revisions,
        policy_id=population.snapshot.policy_id,
        renderer_id=population.renderer_id,
        implementation_id=population.snapshot.implementation_id,
        model_ids=population.snapshot.model_ids,
    )
    return MatchedS0Population(
        retrieval_sha256=population.retrieval_sha256,
        snapshot=snapshot,
        rows=rows,
        prompt_population=prompt_population,
        max_prompt_tokens=population.max_prompt_tokens,
        renderer_id=population.renderer_id,
    )


__all__ = [
    "DEFAULT_MAX_PROMPT_TOKENS",
    "EXPECTED_QUESTION_COUNT",
    "EXPECTED_RETRIEVAL_SHA256",
    "MERGED_QUESTION_FORMAT",
    "MERGED_RETRIEVAL_FORMAT",
    "MatchedPopulationError",
    "MatchedS0Population",
    "MatchedS0Row",
    "POPULATION_FORMAT",
    "PREFLIGHT_FORMAT",
    "V3_POPULATION_FORMAT",
    "V3_PREFLIGHT_FORMAT",
    "V4_IMPLEMENTATION_ID",
    "V4_POLICY_ID",
    "V4_POPULATION_FORMAT",
    "V4_PREFLIGHT_FORMAT",
    "SOURCE_STAGE_ID",
    "STAGE_IDS",
    "load_s0_population",
    "select_s0_population",
]
