#!/usr/bin/env python3
"""Build two gold-blind closure additions independently on sealed S0.

The tool never calls an answer provider and never rebuilds a corpus or store.
``preflight`` validates the historical locked artifacts first, seals a
question-only eligibility manifest, and verifies the ten existing shard
stores.  ``retrieve`` reopens one store read-only and runs the existing
cumulative retriever once for each missing eligible question.  Its cumulative
S2/S3 packets are discarded: the representative and artifact-global closure
plans are each selected, deduplicated against exact sealed S0, and admitted
under separate non-borrowing budgets.  Its runtime registry is deliberately
limited to structural candidate attribution; only the separate sealed posthoc
desired-target registry may claim benchmark target-union completeness.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # support ``python tools/run_...py``
    _REPOSITORY = Path(__file__).resolve().parents[1]
    _LOCKED_RUNTIME = os.environ.get("MEMORY_CONDENSE_LOCKED_RUNTIME_SRC")
    sys.path[:0] = [
        *([str(Path(_LOCKED_RUNTIME).resolve())] if _LOCKED_RUNTIME else []),
        str(_REPOSITORY / "src"),
        str(_REPOSITORY),
    ]

import memory_condense

if __package__ in {None, ""} and _LOCKED_RUNTIME:
    # The frozen package supplies every historical retrieval module.  New
    # tool-only artifact validators live only in the current tree, so expose
    # that directory as a fallback after the locked package, never before it.
    _CURRENT_PACKAGE = _REPOSITORY / "src" / "memory_condense"
    if str(_CURRENT_PACKAGE) not in memory_condense.__path__:
        memory_condense.__path__.append(str(_CURRENT_PACKAGE))
    import memory_condense.eval as _memory_condense_eval

    _CURRENT_EVAL = _CURRENT_PACKAGE / "eval"
    if str(_CURRENT_EVAL) not in _memory_condense_eval.__path__:
        _memory_condense_eval.__path__.append(str(_CURRENT_EVAL))
from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    EvidencePacket,
    ObligationResult,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval._recall_guarded_cumulative_contracts import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalStageReceipt,
    ProtectedExcerpt,
    _atom_evidence_id,
    _protected_evidence_id,
)
from memory_condense.eval._recall_guarded_cumulative_ops import _pack_additions
from memory_condense.eval._recall_guarded_cumulative_result import (
    RecallGuardedCumulativeRetrieval,
    _addition_prompt_prefix,
    _novel_closure_projection,
)
from memory_condense.eval._recall_guarded_cumulative_validation_shard import (
    DEFAULT_POLICY,
    DEFAULT_QWEN_CHOICE,
    DEFAULT_QWEN_PREFIX,
    LOCKED_100Q_OFFSETS,
    MAX_CONTEXT_TOKENS,
    MAX_PROMPT_TOKENS,
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    SOURCE_ROUTER_MAX_SOURCES,
    SOURCE_ROUTER_RRF_CONSTANT,
    _UnboundCoverageSelector,
    _closure_policy,
    _episode_policy,
    _representative_policy,
    load_frozen_validation_policy,
)
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT
from memory_condense.eval.recall_guarded_cumulative import (
    causal_graph_context_budget,
    retrieve_recall_guarded_cumulative_packet,
)
from memory_condense.eval.recall_guarded_cumulative_1m import _load_shared_qwen
from memory_condense.eval.recall_guarded_cumulative_1m_source import (
    current_source_binding,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    _query_batch,
    _read_combined_manifest,
    open_recall_guarded_cumulative_store,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.eval import run_fast_1m_em_facts as em_runner
from tools._locked_em_repair_adapter import (
    LockedEMRepairPopulation,
    _read_canonical_artifact,
    load_locked_em_repair_population,
    project_prevalidated_locked_em_repair_population,
)
from tools._routed_repair_routing import RoutedRepairReason, route_question


REPRESENTATIVE_ARM = "S0_PLUS_REPRESENTATIVE_BRIDGE"
GLOBAL_ARM = "S0_PLUS_ARTIFACT_GLOBAL"
ARM_LABELS = (REPRESENTATIVE_ARM, GLOBAL_ARM)
PLAN_INDEX = {REPRESENTATIVE_ARM: 1, GLOBAL_ARM: 2}

ELIGIBILITY_FORMAT = (
    "memory-condense-independent-closure-eligibility-manifest-v1"
)
PREFLIGHT_FORMAT = "memory-condense-independent-closure-arms-preflight-v3"
QUESTION_FORMAT = "memory-condense-independent-closure-question-v3"
SHARD_INDEX_FORMAT = "memory-condense-independent-closure-shard-index-v3"
MERGED_FORMAT = "memory-condense-independent-closure-retrieval-v3"
POLICY_FORMAT = "memory-condense-independent-closure-policy-v3"

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_BASELINE_ANSWERS = Path(
    "eval_results/longmemeval-1m-fixed-s1-validation-20260826/final-answers.json"
)
DEFAULT_STORE_ROOT = DEFAULT_RETRIEVAL.parent
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/independent-closure-v3"
)
DEFAULT_RUNTIME_SOURCE_ROOT = Path(
    "eval_results/locked-campaign-a66ff05-worktree/src"
)
EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
EXPECTED_BASELINE_ANSWERS_SHA256 = (
    "d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd"
)
EXPECTED_QUESTION_COUNT = 100
EXPECTED_ELIGIBLE_COUNT = 57
EXPECTED_RUNTIME_IMPLEMENTATION_SHA256 = (
    "cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09"
)
ADDITION_TOKEN_CAP = 2_048

_publish = em_runner._publish
_read = em_runner._read
_SHA256 = frozenset("0123456789abcdef")
_FORBIDDEN_GOLD_KEYS = frozenset(
    {
        "answer_session_ids",
        "category",
        "evidence_sources",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
    }
)
_ORCHESTRATION_SOURCE_SURFACE = (
    Path("tools/run_locked_independent_closure_arms.py"),
    Path("tools/_locked_em_repair_adapter.py"),
    Path("tools/_routed_repair_routing.py"),
)
_RUNTIME_SOURCE_SURFACE = (
    Path("memory_condense/eval/_recall_guarded_cumulative_ops.py"),
    Path("memory_condense/eval/_recall_guarded_cumulative_result.py"),
    Path("memory_condense/eval/_recall_guarded_cumulative_contracts.py"),
    Path("memory_condense/eval/recall_guarded_cumulative_runtime.py"),
    Path("memory_condense/eval/recall_guarded_cumulative_1m.py"),
    Path("memory_condense/eval/_recall_guarded_cumulative_validation_shard.py"),
    Path("memory_condense/eval/recall_guarded_cumulative_1m_source.py"),
    Path("memory_condense/search/packing/evidence_packet.py"),
    Path("memory_condense/domain/discourse.py"),
)


@dataclass(frozen=True, slots=True)
class _Question:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    retrieval_question_part_sha256: str
    raw_question: str
    dated_question: str
    shard_offset: int
    source_shard_retrieval_sha256: str
    source_question_part_sha256: str
    combined_store_receipt_sha256: str
    compilation_receipt_sha256: str
    s0_stage_receipt: CumulativeRetrievalStageReceipt
    predecessor_receipt: CausalCoveragePredecessorReceipt
    protected_excerpts: tuple[ProtectedExcerpt, ...]
    s0_evidence: tuple[dict[str, str], ...]
    s0_messages: tuple[dict[str, str], ...]
    eligibility: Mapping[str, Any]
    eligible: bool
    historical_elapsed_seconds: float

    @property
    def protected_context(self) -> str:
        return "\n".join(
            f"[{index}] {row.text}"
            for index, row in enumerate(self.protected_excerpts, 1)
        )


@dataclass(frozen=True, slots=True)
class _Population:
    adapter: LockedEMRepairPopulation
    retrieval: Mapping[str, Any]
    questions: tuple[_Question, ...]
    eligibility_manifest: Mapping[str, Any]
    eligibility_sha256: str

    @property
    def eligible(self) -> tuple[_Question, ...]:
        return tuple(row for row in self.questions if row.eligible)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _contains_gold_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_GOLD_KEYS
            or _contains_gold_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_gold_key(child) for child in value)
    return False


def _messages(value: object, *, ordinal: int) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"S0 messages changed at ordinal {ordinal}")
    result: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
            raise ValueError(f"S0 message {index} changed at ordinal {ordinal}")
        role, content = raw.get("role"), raw.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError(f"S0 message {index} changed at ordinal {ordinal}")
        result.append({"role": str(role), "content": content})
    if [item["role"] for item in result] != ["system", "user"]:
        raise ValueError(f"S0 message order changed at ordinal {ordinal}")
    if result[0]["content"] != QA_SYSTEM_PROMPT:
        raise ValueError(f"S0 system prompt changed at ordinal {ordinal}")
    return tuple(result)


def _raw_question(dated: str, expected_sha256: str, ordinal: int) -> str:
    candidates = [dated]
    if dated.startswith("[Question asked at "):
        end = dated.find("]\n", len("[Question asked at "))
        if end >= 0 and end + 2 < len(dated):
            candidates.append(dated[end + 2 :])
    matches = tuple(value for value in candidates if quote_sha256(value) == expected_sha256)
    if len(matches) != 1:
        raise ValueError(
            f"raw question has {len(matches)} hash matches at ordinal {ordinal}"
        )
    return matches[0]


def _validated_sources(
    retrieval_path: Path,
    baseline_answers_path: Path,
    *,
    expected_retrieval_sha256: str,
    expected_baseline_answers_sha256: str,
) -> tuple[LockedEMRepairPopulation, Mapping[str, Any]]:
    """Invoke the historical validator before any output-root mutation."""

    population = load_locked_em_repair_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        baseline_final_answers_path=baseline_answers_path,
        expected_baseline_final_answers_sha256=expected_baseline_answers_sha256,
    )
    retrieval, observed_sha = _read_canonical_artifact(
        retrieval_path,
        expected_sha256=expected_retrieval_sha256,
    )
    if observed_sha != population.retrieval_sha256:
        raise ValueError("retrieval changed after historical validation")
    return population, retrieval


def _prevalidated_sources(
    retrieval_path: Path,
    baseline_answers_path: Path,
    *,
    expected_retrieval_sha256: str,
    expected_baseline_answers_sha256: str,
    expected_historical_validator_binding_sha256: str,
) -> tuple[LockedEMRepairPopulation, Mapping[str, Any]]:
    """Reproject only after the expected v3 preflight sealed validation."""

    retrieval, retrieval_sha = _read_canonical_artifact(
        retrieval_path,
        expected_sha256=expected_retrieval_sha256,
    )
    baseline, baseline_sha = _read_canonical_artifact(
        baseline_answers_path,
        expected_sha256=expected_baseline_answers_sha256,
    )
    population = project_prevalidated_locked_em_repair_population(
        retrieval,
        retrieval_sha256=retrieval_sha,
        baseline_final_answers=baseline,
        baseline_final_answers_sha256=baseline_sha,
        expected_historical_validator_binding_sha256=(
            expected_historical_validator_binding_sha256
        ),
    )
    return population, retrieval


def _question_row(
    ordinal: int,
    adapter_row: Any,
    raw: Mapping[str, Any],
    part_sha256: object,
) -> _Question:
    question = adapter_row.question
    stages = raw.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError(f"retrieval stages changed at ordinal {ordinal}")
    stage = stages[0]
    if not isinstance(stage, Mapping) or stage.get("stage_id") != (
        "causal_graph_coverage_predecessor"
    ):
        raise ValueError(f"sealed S0 stage changed at ordinal {ordinal}")
    receipt_raw = stage.get("stage_receipt")
    predecessor_raw = raw.get("predecessor_receipt")
    evidence_raw = stage.get("evidence")
    if not isinstance(receipt_raw, Mapping) or not isinstance(
        predecessor_raw, Mapping
    ) or not isinstance(evidence_raw, list):
        raise ValueError(f"sealed S0 receipts changed at ordinal {ordinal}")
    stage_receipt = CumulativeRetrievalStageReceipt(**dict(receipt_raw))
    predecessor = CausalCoveragePredecessorReceipt(**dict(predecessor_raw))
    if canonical_json_bytes(asdict(stage_receipt)) != canonical_json_bytes(receipt_raw):
        raise ValueError(f"S0 stage receipt is noncanonical at ordinal {ordinal}")
    if canonical_json_bytes(asdict(predecessor)) != canonical_json_bytes(predecessor_raw):
        raise ValueError(f"S0 predecessor receipt is noncanonical at ordinal {ordinal}")
    messages = _messages(stage.get("provider_messages"), ordinal=ordinal)
    evidence: list[dict[str, str]] = []
    for index, item in enumerate(evidence_raw):
        if not isinstance(item, Mapping) or set(item) != {
            "evidence_id",
            "source_id",
            "text",
        }:
            raise ValueError(f"S0 evidence {index} changed at ordinal {ordinal}")
        if any(not isinstance(item.get(name), str) for name in item):
            raise ValueError(f"S0 evidence {index} changed at ordinal {ordinal}")
        evidence.append({name: str(item[name]) for name in ("evidence_id", "source_id", "text")})
    if len(evidence) != len(predecessor.protected_chunk_ids):
        raise ValueError(f"S0 evidence coordinates changed at ordinal {ordinal}")
    excerpts = tuple(
        ProtectedExcerpt(
            chunk_id=chunk_id,
            source_id=item["source_id"],
            text=item["text"],
        )
        for chunk_id, item in zip(
            predecessor.protected_chunk_ids, evidence, strict=True
        )
    )
    evidence_ids = tuple(_protected_evidence_id(item) for item in excerpts)
    if (
        raw.get("ordinal") != ordinal
        or raw.get("question_id") != question.question_id
        or raw.get("question_sha256") != question.question_sha256
        or raw.get("dated_question_sha256") != question.dated_question_sha256
        or part_sha256 != question.retrieval_question_part_sha256
        or evidence_ids != tuple(item["evidence_id"] for item in evidence)
        or evidence_ids != stage_receipt.selected_evidence_ids
        or identity_sha256(list(messages)) != stage_receipt.prompt_messages_sha256
        or predecessor.prompt_messages_sha256 != stage_receipt.prompt_messages_sha256
        or predecessor.receipt_sha256 != stage_receipt.method_evidence_sha256
        or predecessor.protected_excerpt_projection_sha256
        != identity_sha256([item.identity_payload() for item in excerpts])
        or stage_receipt.evidence_projection_sha256
        != identity_sha256(
            {
                "protected_excerpts": [item.identity_payload() for item in excerpts],
                "admitted_atoms": [],
            }
        )
        or count_chat_prompt_token_proxy(messages) != stage_receipt.prompt_token_proxy
    ):
        raise ValueError(f"sealed S0 binding changed at ordinal {ordinal}")
    raw_question = _raw_question(
        question.dated_question, question.question_sha256, ordinal
    )
    route = route_question(question.dated_question)
    eligible = bool(
        route.reason is RoutedRepairReason.TEMPORAL_ORDER
        or route.modifiers.requires_complete_frontier
    )
    eligibility = route.identity_payload()
    elapsed = raw.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
        raise ValueError(f"historical elapsed time changed at ordinal {ordinal}")
    shard_offset = raw.get("shard_offset")
    if type(shard_offset) is not int or shard_offset not in LOCKED_100Q_OFFSETS:
        raise ValueError(f"shard offset changed at ordinal {ordinal}")
    return _Question(
        ordinal=ordinal,
        question_id=question.question_id,
        question_sha256=question.question_sha256,
        dated_question_sha256=question.dated_question_sha256,
        retrieval_question_part_sha256=question.retrieval_question_part_sha256,
        raw_question=raw_question,
        dated_question=question.dated_question,
        shard_offset=shard_offset,
        source_shard_retrieval_sha256=_require_sha256(
            raw.get("source_shard_retrieval_sha256"), "source shard retrieval"
        ),
        source_question_part_sha256=_require_sha256(
            raw.get("source_question_part_sha256"), "source question part"
        ),
        combined_store_receipt_sha256=_require_sha256(
            raw.get("combined_store_receipt_sha256"), "combined store receipt"
        ),
        compilation_receipt_sha256=_require_sha256(
            raw.get("compilation_receipt_sha256"), "compilation receipt"
        ),
        s0_stage_receipt=stage_receipt,
        predecessor_receipt=predecessor,
        protected_excerpts=excerpts,
        s0_evidence=tuple(evidence),
        s0_messages=messages,
        eligibility=eligibility,
        eligible=eligible,
        historical_elapsed_seconds=float(elapsed),
    )


def _prepare_population(
    *,
    retrieval_path: Path,
    baseline_answers_path: Path,
    expected_retrieval_sha256: str = EXPECTED_RETRIEVAL_SHA256,
    expected_baseline_answers_sha256: str = EXPECTED_BASELINE_ANSWERS_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    expected_eligible_count: int = EXPECTED_ELIGIBLE_COUNT,
    expected_historical_validator_binding_sha256: str | None = None,
) -> _Population:
    adapter, retrieval = (
        _validated_sources(
            retrieval_path,
            baseline_answers_path,
            expected_retrieval_sha256=expected_retrieval_sha256,
            expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        )
        if expected_historical_validator_binding_sha256 is None
        else _prevalidated_sources(
            retrieval_path,
            baseline_answers_path,
            expected_retrieval_sha256=expected_retrieval_sha256,
            expected_baseline_answers_sha256=expected_baseline_answers_sha256,
            expected_historical_validator_binding_sha256=(
                expected_historical_validator_binding_sha256
            ),
        )
    )
    raw_questions = retrieval.get("questions")
    part_hashes = retrieval.get("question_part_sha256s")
    if (
        adapter.question_count != expected_question_count
        or retrieval.get("question_count") != expected_question_count
        or not isinstance(raw_questions, list)
        or not isinstance(part_hashes, list)
        or len(raw_questions) != expected_question_count
        or len(part_hashes) != expected_question_count
    ):
        raise ValueError("locked question population changed")
    questions = tuple(
        _question_row(ordinal, adapter_row, raw, part_sha)
        for ordinal, (adapter_row, raw, part_sha) in enumerate(
            zip(adapter.rows, raw_questions, part_hashes, strict=True)
        )
        if isinstance(raw, Mapping)
    )
    if len(questions) != expected_question_count:
        raise ValueError("locked question row is not an object")
    eligible_count = sum(item.eligible for item in questions)
    if eligible_count != expected_eligible_count:
        raise ValueError(
            f"question-only eligibility population changed ({eligible_count} != "
            f"{expected_eligible_count})"
        )
    rows: list[dict[str, Any]] = []
    for item in questions:
        body = {
            "ordinal": item.ordinal,
            "question_id": item.question_id,
            "question_sha256": item.question_sha256,
            "dated_question_sha256": item.dated_question_sha256,
            "dated_question": item.dated_question,
            "route_receipt": dict(item.eligibility),
            "eligible": item.eligible,
            "eligibility_basis": (
                "explicit_temporal_order_or_complete_frontier_demand"
                if item.eligible
                else "question_does_not_request_distributed_complete_frontier"
            ),
        }
        body["row_identity_sha256"] = identity_sha256(body)
        rows.append(body)
    manifest: dict[str, Any] = {
        "format": ELIGIBILITY_FORMAT,
        "selection_input": "dated_question_text_only",
        "selection_policy": {
            "eligible_when": (
                "route.reason == temporal_order OR "
                "route.modifiers.requires_complete_frontier == true"
            ),
            "focus": "temporal_order_and_dispersed_complete_frontier_demand",
            "source_labels_used": False,
            "gold_topology_used": False,
        },
        "retrieval_sha256": adapter.retrieval_sha256,
        "population_identity_sha256": adapter.population_identity_sha256,
        "question_count": len(questions),
        "eligible_question_count": eligible_count,
        "questions": rows,
        "provider_calls": 0,
        "gold_loaded": False,
    }
    manifest["manifest_identity_sha256"] = identity_sha256(manifest)
    if _contains_gold_key(manifest):
        raise RuntimeError("eligibility manifest crossed the gold firewall")
    return _Population(
        adapter=adapter,
        retrieval=retrieval,
        questions=questions,
        eligibility_manifest=manifest,
        eligibility_sha256=_digest(manifest),
    )


def _repository_relative(path: Path, repository: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the repository") from exc


def _runtime_source_binding(
    runtime_source_root: Path,
    *,
    repository: Path,
    require_imported_runtime: bool = True,
) -> dict[str, Any]:
    source_root = runtime_source_root.resolve()
    package_root = source_root / "memory_condense"
    if not package_root.is_dir():
        raise FileNotFoundError(f"locked runtime package is missing: {package_root}")
    observed = implementation_sha256(package_root)
    if observed != EXPECTED_RUNTIME_IMPLEMENTATION_SHA256:
        raise ValueError(
            "runtime retrieval implementation changed "
            f"({observed} != {EXPECTED_RUNTIME_IMPLEMENTATION_SHA256})"
        )
    imported = Path(str(memory_condense.__file__)).resolve().parent
    if require_imported_runtime and imported != package_root:
        raise ValueError(
            "imported memory_condense package is not the locked runtime source"
        )
    body: dict[str, Any] = {
        "format": "memory-condense-locked-retrieval-runtime-source-v1",
        "runtime_source_root": _repository_relative(
            source_root, repository, "runtime source root"
        ),
        "runtime_package_root": _repository_relative(
            package_root, repository, "runtime package root"
        ),
        "retrieval_implementation_sha256": observed,
        "expected_retrieval_implementation_sha256": (
            EXPECTED_RUNTIME_IMPLEMENTATION_SHA256
        ),
        "frozen_package_must_be_imported_for_retrieval": True,
    }
    body["runtime_source_binding_sha256"] = identity_sha256(body)
    return body


def _source_surface(
    repository: Path, *, runtime_source_root: Path
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _ORCHESTRATION_SOURCE_SURFACE:
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"retrieval source surface is missing: {path}")
        result[relative.as_posix()] = file_sha256(path)
    for relative in _RUNTIME_SOURCE_SURFACE:
        path = runtime_source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"locked runtime source is missing: {path}")
        result[(Path("locked-runtime-src") / relative).as_posix()] = file_sha256(
            path
        )
    return result


def _reference(population: _Population, offset: int) -> Mapping[str, Any]:
    shards = population.retrieval.get("shards")
    if not isinstance(shards, list):
        raise ValueError("merged retrieval omitted shard references")
    matches = tuple(
        item
        for item in shards
        if isinstance(item, Mapping) and item.get("shard_offset") == offset
    )
    if len(matches) != 1:
        raise ValueError(f"merged retrieval has {len(matches)} references for {offset}")
    return matches[0]


def _held_out_queries(population: _Population, offset: int) -> tuple[str, ...]:
    rows = tuple(
        row for row in population.questions if row.shard_offset == offset
    )
    if len(rows) != 10:
        raise ValueError(f"shard {offset} no longer contains ten questions")
    return tuple(
        dict.fromkeys(
            text for row in rows for text in (row.raw_question, row.dated_question)
        )
    )


def _validate_store(
    population: _Population,
    *,
    offset: int,
    store_root: Path,
    policy: Any,
) -> dict[str, Any]:
    reference = _reference(population, offset)
    shard_root = store_root / "shards" / f"offset-{offset:03d}"
    shard_path = shard_root / "retrieval.json"
    expected_shard_sha = _require_sha256(
        reference.get("shard_retrieval_sha256"), "shard retrieval"
    )
    shard, shard_sha = _read_canonical_artifact(
        shard_path, expected_sha256=expected_shard_sha
    )
    if (
        shard_sha != expected_shard_sha
        or shard.get("provider_calls") != 0
        or shard.get("gold_fields_present") is not False
        or shard.get("question_count") != 10
        or shard.get("combined_store_receipt")
        != reference.get("combined_store_receipt")
        or shard.get("combined_store_receipt_sha256")
        != reference.get("combined_store_receipt_sha256")
        or shard.get("compilation_receipt_sha256")
        != reference.get("compilation_receipt_sha256")
    ):
        raise ValueError(f"sealed shard retrieval changed at offset {offset}")
    if _contains_gold_key(shard):
        raise ValueError(f"shard retrieval crossed gold firewall at offset {offset}")
    store_dir = shard_root / "combined-store"
    combined, compilation, _staging, _learning = _read_combined_manifest(store_dir)
    if (
        asdict(combined) != reference.get("combined_store_receipt")
        or combined.receipt_sha256
        != reference.get("combined_store_receipt_sha256")
        or compilation.receipt_sha256 != reference.get("compilation_receipt_sha256")
        or combined.retrieval_policy_sha256 != policy.retrieval_policy_sha256
    ):
        raise ValueError(f"combined store manifest changed at offset {offset}")
    expected_budget = causal_graph_context_budget(policy.config.retrieval)
    if combined.context_budget_sha256 != identity_sha256(
        {
            name: getattr(expected_budget, name)
            for name in expected_budget.__dataclass_fields__
        }
    ):
        raise ValueError(f"combined store context budget changed at offset {offset}")
    query_batch = _query_batch(_held_out_queries(population, offset), policy.config)
    if combined.held_out_query_batch_sha256 != identity_sha256(
        [{"query_sha256": quote_sha256(item)} for item in query_batch]
    ):
        raise ValueError(f"combined store held-out queries changed at offset {offset}")
    database = store_dir / "memory.db"
    index = store_dir / "hnsw_index.bin"
    manifest = store_dir / "combined-cumulative-store.json"
    database_sha = file_sha256(database)
    index_sha = file_sha256(index)
    if (
        database_sha != combined.target_database_sha256
        or index_sha != combined.target_index_sha256
    ):
        raise ValueError(f"combined store bytes changed at offset {offset}")
    eligible = tuple(row.ordinal for row in population.eligible if row.shard_offset == offset)
    return {
        "shard_offset": offset,
        "shard_retrieval_path": shard_path.as_posix(),
        "shard_retrieval_sha256": shard_sha,
        "combined_store_path": store_dir.as_posix(),
        "combined_store_manifest_sha256": file_sha256(manifest),
        "combined_store_receipt_sha256": combined.receipt_sha256,
        "compilation_receipt_sha256": compilation.receipt_sha256,
        "database_sha256": database_sha,
        "database_size_bytes": database.stat().st_size,
        "index_sha256": index_sha,
        "index_size_bytes": index.stat().st_size,
        "held_out_query_count": len(query_batch),
        "eligible_ordinals": list(eligible),
        "eligible_question_count": len(eligible),
        "store_operation": "verified_read_only_reopen_only",
        "corpus_rebuild_allowed": False,
        "store_rebuild_allowed": False,
    }


def _policy_receipt(
    population: _Population,
    policy: Any,
    *,
    runtime_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": POLICY_FORMAT,
        "arm_labels": list(ARM_LABELS),
        "source_stage": "causal_graph_coverage_predecessor",
        "candidate_plan_indices": {
            REPRESENTATIVE_ARM: 1,
            GLOBAL_ARM: 2,
        },
        "candidate_source": "fresh_full_closure_plans_from_one_cumulative_call",
        "starvation_conditioned_s2_s3_packets_reused": False,
        "selection_order": [
            "pack_raw_mechanism_plan_with_exact_s0_protected",
            "record_selected_before_dedup",
            "exclude_exact_s0_overlaps_from_selected_subplan",
            "repack_and_admit_against_exact_s0",
        ],
        "structural_candidate_attribution": {
            "target_kind": "evidence_atom",
            "declared_structural_candidate_universe": (
                "question_scoped_union_of_both_fresh_plan_atom_sets"
            ),
            "primary_attribution_precedence": list(ARM_LABELS),
            "shared_candidate_rule": (
                "representative attribution wins when the same byte-identical "
                "atom is reachable by both plans"
            ),
            "actual_reachability_field": "discovering_methods",
            "discovery_credit": (
                "every selected-before-dedup atom keeps route-local discovery "
                "credit regardless of its terminal disposition"
            ),
            "final_coverage": (
                "exact S0 overlap is covered by S0_CONTROL; only admitted novel "
                "atoms receive mechanism admission credit"
            ),
            "benchmark_target_tags_loaded": False,
            "desired_target_union_completeness_claimed": False,
            "desired_target_registry_format": (
                "memory-condense-retrieval-target-owner-registry-v1"
            ),
            "required_invariants": [
                "zero_unattributed_structural_candidates",
                "zero_duplicate_primary_attributions",
                "pairwise_primary_attribution_sets_are_disjoint",
                "union_primary_attribution_sets_equals_declared_structural_candidate_universe",
                "shared_atom_ids_have_identical_identity_payloads",
                "each_selected_route_target_has_exactly_one_terminal_disposition",
                "selected_route_discovery_credit_is_preserved",
            ],
        },
        "parent_for_each_arm": "exact_sealed_s0_only",
        "cross_arm_budget_borrowing": False,
        "addition_token_cap_per_arm": ADDITION_TOKEN_CAP,
        "total_context_token_cap": MAX_CONTEXT_TOKENS,
        "prompt_token_cap": MAX_PROMPT_TOKENS,
        "responder_output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
        "retrieval_policy_sha256": policy.retrieval_policy_sha256,
        "runtime_source_binding_sha256": runtime_source_binding[
            "runtime_source_binding_sha256"
        ],
        "retrieval_implementation_sha256": runtime_source_binding[
            "retrieval_implementation_sha256"
        ],
        "eligibility_manifest_sha256": population.eligibility_sha256,
        "provider_calls": 0,
        "gold_loaded": False,
    }
    body["policy_receipt_sha256"] = identity_sha256(body)
    return body


def _preflight(
    population: _Population,
    *,
    store_root: Path,
    policy_path: Path,
    qwen_prefix: Path,
    qwen_choice: Path,
    device: str,
    repository: Path,
    runtime_source_root: Path,
) -> dict[str, Any]:
    runtime_binding = _runtime_source_binding(
        runtime_source_root,
        repository=repository,
        require_imported_runtime=False,
    )
    policy = load_frozen_validation_policy(policy_path, device=device)
    if (
        population.retrieval.get("retrieval_policy_sha256")
        != policy.retrieval_policy_sha256
    ):
        raise ValueError("frozen validation policy changed")
    if not qwen_prefix.is_dir() or not qwen_choice.is_dir():
        raise FileNotFoundError("one or both pinned Qwen directories are missing")
    surface_before = _source_surface(
        repository, runtime_source_root=runtime_source_root
    )
    stores = [
        _validate_store(
            population,
            offset=offset,
            store_root=store_root,
            policy=policy,
        )
        for offset in LOCKED_100Q_OFFSETS
    ]
    if (
        _runtime_source_binding(
            runtime_source_root,
            repository=repository,
            require_imported_runtime=False,
        )
        != runtime_binding
        or _source_surface(repository, runtime_source_root=runtime_source_root)
        != surface_before
    ):
        raise RuntimeError("retrieval source surface changed during preflight")
    elapsed = [row.historical_elapsed_seconds for row in population.eligible]
    by_shard = [
        {
            "shard_offset": offset,
            "eligible_question_count": sum(
                row.shard_offset == offset for row in population.eligible
            ),
            "historical_comparable_retrieval_seconds": sum(
                row.historical_elapsed_seconds
                for row in population.eligible
                if row.shard_offset == offset
            ),
        }
        for offset in LOCKED_100Q_OFFSETS
    ]
    policy_receipt = _policy_receipt(
        population,
        policy,
        runtime_source_binding=runtime_binding,
    )
    artifact: dict[str, Any] = {
        "format": PREFLIGHT_FORMAT,
        "eligibility_manifest_sha256": population.eligibility_sha256,
        "eligibility_manifest_identity_sha256": population.eligibility_manifest[
            "manifest_identity_sha256"
        ],
        "retrieval_sha256": population.adapter.retrieval_sha256,
        "baseline_final_answers_sha256": (
            population.adapter.baseline_final_answers_sha256
        ),
        "historical_validator_binding_sha256": population.adapter.binding_sha256,
        "runtime_source_binding": runtime_binding,
        "population_identity_sha256": population.adapter.population_identity_sha256,
        "question_count": len(population.questions),
        "eligible_question_count": len(population.eligible),
        "policy": policy_receipt,
        "source_surface_sha256s": surface_before,
        "frozen_validation_policy_path": policy_path.as_posix(),
        "frozen_validation_policy_sha256": file_sha256(policy_path),
        "qwen_prefix_model_dir": qwen_prefix.as_posix(),
        "qwen_choice_model_dir": qwen_choice.as_posix(),
        "qwen_prefix_checkpoint_sha256": (
            policy.config.retrieval.coverage_selector_prefix_checkpoint_sha256
        ),
        "qwen_choice_checkpoint_sha256": (
            policy.config.retrieval.coverage_selector_choice_checkpoint_sha256
        ),
        "source_embedding_device": str(device).casefold(),
        "stores": stores,
        "runtime_reference": {
            "basis": "sealed_elapsed_seconds_for_same_cumulative_retriever",
            "eligible_question_count": len(elapsed),
            "historical_comparable_retrieval_seconds": sum(elapsed),
            "historical_mean_seconds_per_question": statistics.fmean(elapsed),
            "historical_median_seconds_per_question": statistics.median(elapsed),
            "historical_max_seconds_per_question": max(elapsed),
            "per_shard": by_shard,
            "additional_unmeasured_work": (
                "ten read-only store reopens and model loads; independent "
                "projection/packing is expected to be small"
            ),
        },
        "retrieval_invocations_planned": len(population.eligible),
        "provider_calls": 0,
        "gold_loaded": False,
        "corpus_rebuilds": 0,
        "store_rebuilds": 0,
        "output_root_mutated_before_historical_validation": False,
    }
    artifact["preflight_identity_sha256"] = identity_sha256(artifact)
    if _contains_gold_key(artifact):
        raise RuntimeError("preflight crossed the gold firewall")
    return artifact


def run_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    population = _prepare_population(
        retrieval_path=Path(args.retrieval),
        baseline_answers_path=Path(args.baseline_answers),
        expected_question_count=int(args.expected_question_count),
        expected_eligible_count=int(args.expected_eligible_count),
    )
    repository = Path(__file__).resolve().parents[1]
    artifact = _preflight(
        population,
        store_root=Path(args.store_root),
        policy_path=Path(args.policy),
        qwen_prefix=Path(args.qwen_prefix),
        qwen_choice=Path(args.qwen_choice),
        device=str(args.device),
        repository=repository,
        runtime_source_root=Path(args.runtime_source_root),
    )
    output = Path(args.output_root)
    manifest_sha = _publish(
        output / "eligibility-manifest.json", population.eligibility_manifest
    )
    if manifest_sha != population.eligibility_sha256:
        raise RuntimeError("published eligibility manifest changed")
    preflight_sha = _publish(output / "preflight.json", artifact)
    return artifact, manifest_sha, preflight_sha


def _selected_plan(plan: ClosurePlan, packet: EvidencePacket) -> ClosurePlan:
    """Restrict a closure plan to the packer's atomic pre-dedup selection."""

    selected_bundle_ids = {item.bundle_id for item in packet.bundles}
    bundles = tuple(
        item for item in plan.bundles if item.bundle_id in selected_bundle_ids
    )
    selected_atom_ids = {atom_id for bundle in bundles for atom_id in bundle.atom_ids}
    atoms = tuple(item for item in plan.atoms if item.atom_id in selected_atom_ids)
    bundle_by_id = {item.bundle_id: item for item in bundles}
    obligation_by_id = {
        item.obligation_id: item for item in plan.query_program.obligations
    }
    results: list[ObligationResult] = []
    for result in plan.obligation_results:
        bundle_ids = tuple(
            item for item in result.bundle_ids if item in selected_bundle_ids
        )
        owned = tuple(
            bundle_by_id[item]
            for item in bundle_ids
            if result.obligation_id in bundle_by_id[item].obligation_ids
        )
        unit_ids = tuple(dict.fromkeys(item for row in owned for item in row.unit_ids))
        relation_ids = tuple(
            dict.fromkeys(item for row in owned for item in row.relation_ids)
        )
        support = max(len(unit_ids), len(relation_ids), len(bundle_ids))
        status, reason = result.status, result.reason
        if status == "satisfied" and support < obligation_by_id[
            result.obligation_id
        ].min_count:
            status, reason = "not_found", "selected_before_dedup_budget"
        results.append(
            replace(
                result,
                status=status,
                unit_ids=unit_ids,
                relation_ids=relation_ids,
                bundle_ids=bundle_ids,
                reason=reason,
            )
        )
    required = {
        item.obligation_id
        for item in plan.query_program.obligations
        if item.required
    }
    satisfied = {
        item.obligation_id for item in results if item.status == "satisfied"
    }
    complete = bool(
        required <= satisfied
        and plan.stopping_reason == "complete"
        and plan.scope_witnesses
        and all(item.exhaustive for item in plan.scope_witnesses)
    )
    return replace(
        plan,
        atoms=atoms,
        bundles=bundles,
        obligation_results=tuple(results),
        direct_chunk_ids=tuple(
            item
            for item in plan.direct_chunk_ids
            if item in {atom.span.chunk_id for atom in atoms}
        ),
        complete_claimed=complete,
        plan_sha256="",
    )


def _atom_rows(atoms: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": _atom_evidence_id(atom),
            "atom_id": atom.atom_id,
            "chunk_id": atom.span.chunk_id,
            "source_id": atom.span.source_id,
            "text": atom.text,
            "text_sha256": quote_sha256(atom.text),
            "identity": atom.identity_payload(),
        }
        for atom in atoms
    ]


def _packet_projection(packet: EvidencePacket | None) -> dict[str, Any] | None:
    if packet is None:
        return None
    atoms = _atom_rows(packet.atoms)
    bundles = [item.identity_payload() for item in packet.bundles]
    return {
        "packet_receipt": asdict(packet.receipt),
        "context": packet.context,
        "context_sha256": quote_sha256(packet.context),
        "atom_count": len(atoms),
        "atom_identities_sha256": identity_sha256(
            [item["identity"] for item in atoms]
        ),
        "atoms": atoms,
        "bundle_count": len(bundles),
        "bundle_identities_sha256": identity_sha256(bundles),
        "bundles": bundles,
    }


def _candidate_pool(plan: ClosurePlan) -> dict[str, Any]:
    atoms = [item.identity_payload() for item in plan.atoms]
    bundles = [item.identity_payload() for item in plan.bundles]
    return {
        "source_plan_sha256": plan.plan_sha256,
        "atom_count": len(atoms),
        "atom_identities_sha256": identity_sha256(atoms),
        "atom_identities": atoms,
        "bundle_count": len(bundles),
        "bundle_identities_sha256": identity_sha256(bundles),
        "bundle_identities": bundles,
        "visited_episode_count": len(plan.visited_episode_ids),
        "visited_unit_count": len(plan.visited_unit_ids),
        "visited_relation_count": len(plan.visited_relation_ids),
        "scope_witnesses_sha256": identity_sha256(
            [item.identity_payload() for item in plan.scope_witnesses]
        ),
        "complete_claimed": plan.complete_claimed,
        "stopping_reason": plan.stopping_reason,
    }


def _final_messages(
    question: _Question, packet: EvidencePacket
) -> tuple[dict[str, str], ...]:
    prefix, suffix = _addition_prompt_prefix(
        question.dated_question,
        question.protected_context,
        len(question.protected_excerpts),
    )
    return (
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": prefix + packet.context + suffix},
    )


def _overflow(
    question: _Question,
    packet: EvidencePacket,
    messages: Sequence[Mapping[str, str]],
) -> str | None:
    addition_tokens = count_tokens(packet.context)
    combined = (
        f"{question.protected_context}\n"
        f"[{len(question.protected_excerpts) + 1}] {packet.context}"
    )
    if addition_tokens > ADDITION_TOKEN_CAP:
        return "addition_token_cap"
    if count_tokens(combined) > MAX_CONTEXT_TOKENS:
        return "total_context_token_cap"
    if count_chat_prompt_token_proxy(messages) > MAX_PROMPT_TOKENS:
        return "prompt_token_cap"
    return None


def _noop_arm(
    *,
    label: str,
    candidate: Mapping[str, Any],
    selected: Mapping[str, Any] | None,
    dedup: Mapping[str, Any] | None,
    question: _Question,
    status: str,
    overflow_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "arm_label": label,
        "parent_stage": "exact_sealed_s0",
        "candidate_pool": dict(candidate),
        "selected_before_dedup": selected,
        "dedup": dedup,
        "admission": {
            "status": status,
            "overflow_reason": overflow_reason,
            "addition_token_cap": ADDITION_TOKEN_CAP,
            "addition_token_proxy": 0,
            "total_context_token_proxy": count_tokens(question.protected_context),
            "prompt_token_proxy": count_chat_prompt_token_proxy(
                question.s0_messages
            ),
            "added_evidence": [],
            "provider_messages": list(question.s0_messages),
            "provider_messages_sha256": identity_sha256(
                list(question.s0_messages)
            ),
        },
    }


def _build_arm(
    *,
    label: str,
    plan: ClosurePlan,
    question: _Question,
    condenser: Any,
) -> dict[str, Any]:
    candidate = _candidate_pool(plan)
    protected_tokens = count_tokens(question.protected_context)
    context_cap = min(MAX_CONTEXT_TOKENS, protected_tokens + ADDITION_TOKEN_CAP)
    raw_packet = _pack_additions(
        condenser,
        plan,
        prompt_question=question.dated_question,
        protected_context=question.protected_context,
        protected_count=len(question.protected_excerpts),
        max_context_tokens=context_cap,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
    )
    selected = _packet_projection(raw_packet)
    if raw_packet is None or not raw_packet.atoms:
        return _noop_arm(
            label=label,
            candidate=candidate,
            selected=selected,
            dedup=None,
            question=question,
            status=("no_candidates" if not plan.atoms else "selection_budget_noop"),
        )
    raw_messages = _final_messages(question, raw_packet)
    raw_overflow = _overflow(question, raw_packet, raw_messages)
    if raw_overflow is not None:
        return _noop_arm(
            label=label,
            candidate=candidate,
            selected=selected,
            dedup=None,
            question=question,
            status="overflow_noop",
            overflow_reason=f"selected_before_dedup:{raw_overflow}",
        )
    selected_plan = _selected_plan(plan, raw_packet)
    projection = _novel_closure_projection(
        selected_plan, question.protected_excerpts, ()
    )
    projected_atoms = [item.identity_payload() for item in projection.plan.atoms]
    projected_bundles = [item.identity_payload() for item in projection.plan.bundles]
    dedup = {
        "selected_plan_sha256": selected_plan.plan_sha256,
        "projection_receipt": asdict(projection.receipt),
        "excluded_atom_count": len(projection.receipt.excluded_atom_ids),
        "excluded_atom_ids": list(projection.receipt.excluded_atom_ids),
        "post_dedup_atom_count": len(projected_atoms),
        "post_dedup_atom_identities_sha256": identity_sha256(projected_atoms),
        "post_dedup_atom_identities": projected_atoms,
        "post_dedup_bundle_count": len(projected_bundles),
        "post_dedup_bundle_identities_sha256": identity_sha256(projected_bundles),
        "post_dedup_bundle_identities": projected_bundles,
    }
    final_packet = _pack_additions(
        condenser,
        projection.plan,
        prompt_question=question.dated_question,
        protected_context=question.protected_context,
        protected_count=len(question.protected_excerpts),
        max_context_tokens=context_cap,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
    )
    if final_packet is None or not final_packet.atoms:
        return _noop_arm(
            label=label,
            candidate=candidate,
            selected=selected,
            dedup=dedup,
            question=question,
            status=(
                "no_novel_evidence"
                if not projection.plan.atoms
                else "admission_budget_noop"
            ),
        )
    messages = _final_messages(question, final_packet)
    overflow = _overflow(question, final_packet, messages)
    if overflow is not None:
        return _noop_arm(
            label=label,
            candidate=candidate,
            selected=selected,
            dedup=dedup,
            question=question,
            status="overflow_noop",
            overflow_reason=f"admission:{overflow}",
        )
    combined = (
        f"{question.protected_context}\n"
        f"[{len(question.protected_excerpts) + 1}] {final_packet.context}"
    )
    return {
        "arm_label": label,
        "parent_stage": "exact_sealed_s0",
        "candidate_pool": candidate,
        "selected_before_dedup": selected,
        "dedup": dedup,
        "admission": {
            "status": "added",
            "overflow_reason": None,
            "addition_token_cap": ADDITION_TOKEN_CAP,
            "addition_token_proxy": count_tokens(final_packet.context),
            "total_context_token_proxy": count_tokens(combined),
            "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
            "packet": _packet_projection(final_packet),
            "added_evidence": _atom_rows(final_packet.atoms),
            "provider_messages": list(messages),
            "provider_messages_sha256": identity_sha256(list(messages)),
        },
    }


def _atom_identity_index(
    rows: object, *, label: str, wrapped: bool
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} atom identities are missing")
    order: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{label} atom identity is not an object")
        identity = raw.get("identity") if wrapped else raw
        if not isinstance(identity, Mapping):
            raise RuntimeError(f"{label} atom identity payload is missing")
        atom_id = identity.get("atom_id")
        if not isinstance(atom_id, str) or not atom_id:
            raise RuntimeError(f"{label} atom ID is empty")
        if wrapped and raw.get("atom_id") != atom_id:
            raise RuntimeError(f"{label} atom wrapper changed identity")
        if atom_id in result:
            raise RuntimeError(f"{label} contains duplicate atom IDs")
        order.append(atom_id)
        result[atom_id] = dict(identity)
    return order, result


def _receipt_sha256(value: object, label: str) -> str:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} receipt is missing")
    try:
        return _require_sha256(value.get("receipt_sha256"), f"{label} receipt")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _arm_target_projection(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Seal route-local structural reachability and exhaustive dispositions."""

    label = arm.get("arm_label")
    if label not in ARM_LABELS:
        raise RuntimeError("structural candidate arm label changed")
    candidate = arm.get("candidate_pool")
    if not isinstance(candidate, Mapping):
        raise RuntimeError(f"{label} candidate pool is missing")
    candidate_ids, candidate_by_id = _atom_identity_index(
        candidate.get("atom_identities"), label=f"{label} candidate", wrapped=False
    )
    source_plan_sha256 = _require_sha256(
        candidate.get("source_plan_sha256"), f"{label} source plan"
    )
    source_scope_witnesses_sha256 = _require_sha256(
        candidate.get("scope_witnesses_sha256"), f"{label} scope witnesses"
    )
    candidate_bundles = candidate.get("bundle_identities")
    if (
        candidate.get("atom_count") != len(candidate_ids)
        or candidate.get("atom_identities_sha256")
        != identity_sha256([candidate_by_id[item] for item in candidate_ids])
        or not isinstance(candidate_bundles, list)
        or candidate.get("bundle_count") != len(candidate_bundles)
        or candidate.get("bundle_identities_sha256")
        != identity_sha256(candidate_bundles)
    ):
        raise RuntimeError(f"{label} candidate pool seal changed")

    selected = arm.get("selected_before_dedup")
    selection_receipt_sha256: str | None = None
    if selected is None:
        selected_ids, selected_by_id = [], {}
    elif isinstance(selected, Mapping):
        selected_ids, selected_by_id = _atom_identity_index(
            selected.get("atoms"), label=f"{label} selected", wrapped=True
        )
        if (
            selected.get("atom_count") != len(selected_ids)
            or selected.get("atom_identities_sha256")
            != identity_sha256([selected_by_id[item] for item in selected_ids])
        ):
            raise RuntimeError(f"{label} selected atom seal changed")
        selection_receipt_sha256 = _receipt_sha256(
            selected.get("packet_receipt"), f"{label} selection packet"
        )
    else:
        raise RuntimeError(f"{label} selected packet is malformed")

    dedup = arm.get("dedup")
    dedup_receipt_sha256: str | None = None
    if dedup is None:
        excluded_ids, projected_ids, projected_by_id = [], [], {}
    elif isinstance(dedup, Mapping):
        raw_excluded = dedup.get("excluded_atom_ids")
        if not isinstance(raw_excluded, list) or any(
            not isinstance(item, str) or not item for item in raw_excluded
        ):
            raise RuntimeError(f"{label} S0 exclusions are malformed")
        excluded_ids = list(raw_excluded)
        projected_ids, projected_by_id = _atom_identity_index(
            dedup.get("post_dedup_atom_identities"),
            label=f"{label} post-dedup",
            wrapped=False,
        )
        projection_receipt = dedup.get("projection_receipt")
        dedup_receipt_sha256 = _receipt_sha256(
            projection_receipt, f"{label} dedup projection"
        )
        selected_plan_sha256 = _require_sha256(
            dedup.get("selected_plan_sha256"), f"{label} selected plan"
        )
        projected_bundles = dedup.get("post_dedup_bundle_identities")
        receipt_excluded = (
            projection_receipt.get("excluded_atom_ids")
            if isinstance(projection_receipt, Mapping)
            else None
        )
        if (
            len(excluded_ids) != len(set(excluded_ids))
            or dedup.get("excluded_atom_count") != len(excluded_ids)
            or not isinstance(projection_receipt, Mapping)
            or not isinstance(receipt_excluded, Sequence)
            or isinstance(receipt_excluded, (str, bytes, bytearray))
            or list(receipt_excluded) != excluded_ids
            or projection_receipt.get("source_plan_sha256")
            != selected_plan_sha256
            or dedup.get("post_dedup_atom_count") != len(projected_ids)
            or dedup.get("post_dedup_atom_identities_sha256")
            != identity_sha256([projected_by_id[item] for item in projected_ids])
            or not isinstance(projected_bundles, list)
            or dedup.get("post_dedup_bundle_count") != len(projected_bundles)
            or dedup.get("post_dedup_bundle_identities_sha256")
            != identity_sha256(projected_bundles)
        ):
            raise RuntimeError(f"{label} dedup seal changed")
    else:
        raise RuntimeError(f"{label} dedup projection is malformed")

    admission = arm.get("admission")
    if not isinstance(admission, Mapping):
        raise RuntimeError(f"{label} admission is missing")
    status = admission.get("status")
    allowed_statuses = {
        "added",
        "no_candidates",
        "selection_budget_noop",
        "overflow_noop",
        "no_novel_evidence",
        "admission_budget_noop",
    }
    if status not in allowed_statuses:
        raise RuntimeError(f"{label} admission status changed")
    admitted_ids, admitted_by_id = _atom_identity_index(
        admission.get("added_evidence"), label=f"{label} admitted", wrapped=True
    )
    admission_receipt_sha256: str | None = None
    if status == "added":
        packet = admission.get("packet")
        if not isinstance(packet, Mapping):
            raise RuntimeError(f"{label} admitted packet is missing")
        packet_ids, packet_by_id = _atom_identity_index(
            packet.get("atoms"), label=f"{label} admitted packet", wrapped=True
        )
        if packet_ids != admitted_ids or any(
            canonical_json_bytes(packet_by_id[item])
            != canonical_json_bytes(admitted_by_id[item])
            for item in admitted_ids
        ):
            raise RuntimeError(f"{label} admitted packet evidence changed")
        admission_receipt_sha256 = _receipt_sha256(
            packet.get("packet_receipt"), f"{label} admission packet"
        )
        if not admitted_ids:
            raise RuntimeError(f"{label} added an empty packet")
    elif admitted_ids or admission.get("packet") is not None:
        raise RuntimeError(f"{label} no-op contains admitted evidence")

    candidate_set, selected_set = set(candidate_ids), set(selected_ids)
    excluded_set, projected_set, admitted_set = (
        set(excluded_ids),
        set(projected_ids),
        set(admitted_ids),
    )
    if (
        not selected_set <= candidate_set
        or not excluded_set <= selected_set
        or not projected_set <= selected_set
        or not admitted_set <= projected_set
        or excluded_set & projected_set
        or excluded_set & admitted_set
    ):
        raise RuntimeError(f"{label} target projection escaped its candidate universe")
    for atom_id, projected in (
        *((item, selected_by_id[item]) for item in selected_ids),
        *((item, projected_by_id[item]) for item in projected_ids),
        *((item, admitted_by_id[item]) for item in admitted_ids),
    ):
        if canonical_json_bytes(projected) != canonical_json_bytes(
            candidate_by_id[atom_id]
        ):
            raise RuntimeError(f"{label} atom identity changed across projections")

    if selected_ids and dedup is None and not (
        status == "overflow_noop"
        and str(admission.get("overflow_reason", "")).startswith(
            "selected_before_dedup:"
        )
    ):
        raise RuntimeError(f"{label} selected atoms bypassed S0 dedup")
    if dedup is not None and status == "no_novel_evidence" and projected_ids:
        raise RuntimeError(f"{label} claimed no novel evidence with retained atoms")
    if status in {"no_candidates", "selection_budget_noop"} and selected_ids:
        raise RuntimeError(f"{label} selection no-op contains selected atoms")

    dispositions: list[dict[str, Any]] = []
    for atom_id in candidate_ids:
        selected_here = atom_id in selected_set
        if not selected_here:
            selection_disposition = "not_selected"
            dedup_disposition = "not_applicable"
            admission_disposition = "not_applicable"
            terminal_disposition = "not_selected"
            coverage_source = None
        elif atom_id in excluded_set:
            selection_disposition = "selected_before_dedup"
            dedup_disposition = "excluded_exact_s0_overlap"
            admission_disposition = "not_admitted_exact_s0_covered"
            terminal_disposition = "exact_s0_overlap_after_selection"
            coverage_source = "S0_CONTROL"
        elif dedup is None:
            selection_disposition = "selected_before_dedup"
            dedup_disposition = "not_run_due_selection_overflow"
            admission_disposition = "not_admitted_selection_overflow"
            terminal_disposition = "selection_overflow_noop"
            coverage_source = None
        elif atom_id not in projected_set:
            selection_disposition = "selected_before_dedup"
            dedup_disposition = "removed_during_novel_projection"
            admission_disposition = "not_applicable"
            terminal_disposition = "projection_drop_after_s0_dedup"
            coverage_source = None
        elif atom_id in admitted_set:
            selection_disposition = "selected_before_dedup"
            dedup_disposition = "retained_after_s0_dedup"
            admission_disposition = "admitted"
            terminal_disposition = "admitted_after_dedup"
            coverage_source = label
        elif status == "added":
            selection_disposition = "selected_before_dedup"
            dedup_disposition = "retained_after_s0_dedup"
            admission_disposition = "not_selected_by_final_repack"
            terminal_disposition = "final_repack_budget_drop"
            coverage_source = None
        elif status == "admission_budget_noop":
            selection_disposition = "selected_before_dedup"
            dedup_disposition = "retained_after_s0_dedup"
            admission_disposition = "admission_budget_noop"
            terminal_disposition = "admission_budget_noop"
            coverage_source = None
        elif status == "overflow_noop" and str(
            admission.get("overflow_reason", "")
        ).startswith("admission:"):
            selection_disposition = "selected_before_dedup"
            dedup_disposition = "retained_after_s0_dedup"
            admission_disposition = "admission_overflow_noop"
            terminal_disposition = "admission_overflow_noop"
            coverage_source = None
        else:
            raise RuntimeError(f"{label} selected atom has no terminal disposition")
        dispositions.append(
            {
                "evidence_atom_id": atom_id,
                "atom_identity_sha256": identity_sha256(candidate_by_id[atom_id]),
                "source_plan_sha256": source_plan_sha256,
                "source_scope_witnesses_sha256": source_scope_witnesses_sha256,
                "candidate_pool_atom_identities_sha256": candidate[
                    "atom_identities_sha256"
                ],
                "selection_disposition": selection_disposition,
                "selection_packet_receipt_sha256": selection_receipt_sha256,
                "dedup_disposition": dedup_disposition,
                "dedup_projection_receipt_sha256": dedup_receipt_sha256,
                "admission_disposition": admission_disposition,
                "admission_packet_receipt_sha256": admission_receipt_sha256,
                "terminal_disposition": terminal_disposition,
                "discovery_credit_preserved": selected_here,
                "final_packet_covered": coverage_source is not None,
                "final_coverage_source": coverage_source,
            }
        )
    selected_dispositions = [
        item for item in dispositions if item["selection_disposition"] != "not_selected"
    ]
    if (
        [item["evidence_atom_id"] for item in selected_dispositions] != selected_ids
        or any(not item["discovery_credit_preserved"] for item in selected_dispositions)
    ):
        raise RuntimeError(f"{label} did not preserve selected discovery credit")
    return {
        "reachable_structural_candidate_ids": candidate_ids,
        "selected_target_ids_before_dedup": selected_ids,
        "preserved_discovery_credit_target_ids": selected_ids,
        "exact_s0_overlap_target_ids_after_selection": excluded_ids,
        "post_dedup_candidate_target_ids": projected_ids,
        "admitted_target_ids_after_dedup": admitted_ids,
        "route_target_dispositions": dispositions,
        "route_target_dispositions_sha256": identity_sha256(dispositions),
    }


def _structural_candidate_attribution(
    *,
    population_identity_sha256: str,
    question_id: str,
    question_identity_sha256: str,
    arms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attribute the gold-blind plan union without claiming desired-target recall."""

    population_sha = _require_sha256(
        population_identity_sha256, "population identity"
    )
    question_sha = _require_sha256(question_identity_sha256, "question identity")
    if not isinstance(question_id, str) or not question_id:
        raise RuntimeError("structural candidate question ID is empty")
    labels = tuple(
        str(arm.get("arm_label")) if isinstance(arm, Mapping) else ""
        for arm in arms
    )
    if labels != ARM_LABELS:
        raise RuntimeError("structural attribution requires the exact ordered arms")
    by_label = dict(zip(labels, arms, strict=True))
    projections = {
        label: _arm_target_projection(by_label[label]) for label in ARM_LABELS
    }
    identities = {
        label: _atom_identity_index(
            by_label[label]["candidate_pool"]["atom_identities"],
            label=f"{label} candidate",
            wrapped=False,
        )[1]
        for label in ARM_LABELS
    }
    atom_order = tuple(
        dict.fromkeys(
            atom_id
            for label in ARM_LABELS
            for atom_id in projections[label]["reachable_structural_candidate_ids"]
        )
    )
    attribution_sets: dict[str, list[str]] = {label: [] for label in ARM_LABELS}
    targets: list[dict[str, Any]] = []
    for atom_id in atom_order:
        routes = [
            label
            for label in ARM_LABELS
            if atom_id in projections[label]["reachable_structural_candidate_ids"]
        ]
        identity_payload = identities[routes[0]][atom_id]
        if any(
            canonical_json_bytes(identities[label][atom_id])
            != canonical_json_bytes(identity_payload)
            for label in routes[1:]
        ):
            raise RuntimeError(
                "shared structural atom ID has different identity payloads"
            )
        atom_identity_sha256 = identity_sha256(identity_payload)
        primary = routes[0]
        target_id = identity_sha256(
            {
                "scope": "question_local_structural_candidate",
                "population_identity_sha256": population_sha,
                "question_id": question_id,
                "question_identity_sha256": question_sha,
                "kind": "evidence_atom",
                "atom_identity_sha256": atom_identity_sha256,
            }
        )
        attribution_sets[primary].append(target_id)
        reachability = []
        for label in routes:
            matches = [
                item
                for item in projections[label]["route_target_dispositions"]
                if item["evidence_atom_id"] == atom_id
            ]
            if len(matches) != 1:
                raise RuntimeError("route target disposition is not one-to-one")
            reachability.append({"method": label, **matches[0]})
        primary_route = reachability[0]
        selected_by = [
            item["method"]
            for item in reachability
            if item["selection_disposition"] != "not_selected"
        ]
        admitted_by = [
            item["method"]
            for item in reachability
            if item["admission_disposition"] == "admitted"
        ]
        s0_overlap_discovered_by = [
            item["method"]
            for item in reachability
            if item["dedup_disposition"] == "excluded_exact_s0_overlap"
        ]
        targets.append(
            {
                "target_id": target_id,
                "kind": "evidence_atom",
                "evidence_atom_id": atom_id,
                "evidence_atom_identity_sha256": atom_identity_sha256,
                "evidence_atom_identity": identity_payload,
                "primary_attribution": primary,
                "discovering_methods": routes,
                "secondary_reachability": routes[1:],
                "reachability": reachability,
                "selected_before_dedup_by": selected_by,
                "discovery_credit_preserved_by": selected_by,
                "admitted_after_dedup_by": admitted_by,
                "exact_s0_overlap_discovered_by": s0_overlap_discovered_by,
                "primary_attribution_outcome": {
                    "discovery_credit_preserved": primary_route[
                        "discovery_credit_preserved"
                    ],
                    "mechanism_admission_credit": (
                        primary_route["admission_disposition"] == "admitted"
                    ),
                    "exact_s0_overlap_discovered": (
                        primary_route["dedup_disposition"]
                        == "excluded_exact_s0_overlap"
                    ),
                    "secondary_route_only_admission": bool(admitted_by)
                    and primary not in admitted_by,
                },
            }
        )
    universe = [item["target_id"] for item in targets]
    attributed = [
        item for label in ARM_LABELS for item in attribution_sets[label]
    ]
    unattributed = [
        item["target_id"]
        for item in targets
        if item["primary_attribution"] not in ARM_LABELS
    ]
    duplicate_count = len(attributed) - len(set(attributed))
    intersections = [
        item
        for left_index, left in enumerate(ARM_LABELS)
        for right in ARM_LABELS[left_index + 1 :]
        for item in set(attribution_sets[left]) & set(attribution_sets[right])
    ]
    union_matches = (
        set(attributed) == set(universe) and len(attributed) == len(universe)
    )
    if unattributed or duplicate_count or intersections or not union_matches:
        raise RuntimeError("structural candidate attribution is incomplete")
    body: dict[str, Any] = {
        "registry_role": "runtime_structural_candidate_attribution_only",
        "target_scope": "question_local_fresh_closure_candidate_union",
        "population_identity_sha256": population_sha,
        "question_id": question_id,
        "question_identity_sha256": question_sha,
        "benchmark_target_tags_loaded": False,
        "desired_target_union_completeness_claimed": False,
        "desired_target_registry_format": (
            "memory-condense-retrieval-target-owner-registry-v1"
        ),
        "declared_structural_candidate_count": len(universe),
        "declared_structural_candidate_universe_sha256": identity_sha256(universe),
        "declared_structural_candidate_ids": universe,
        "primary_attribution_sets": attribution_sets,
        "targets": targets,
        "invariants": {
            "unattributed_structural_candidate_count": len(unattributed),
            "duplicate_primary_attribution_count": duplicate_count,
            "pairwise_primary_attribution_intersection_count": len(intersections),
            "primary_attribution_union_equals_declared_structural_candidate_universe": (
                union_matches
            ),
            "shared_atom_identity_mismatch_count": 0,
            "selected_terminal_disposition_missing_count": 0,
            "selected_discovery_credit_loss_count": 0,
        },
    }
    body["manifest_identity_sha256"] = identity_sha256(body)
    return body


def _annotate_arm_targets(arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm in arms:
        row = dict(arm)
        row.update(_arm_target_projection(arm))
        result.append(row)
    return result


def _aggregate_structural_candidate_attribution(
    questions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not questions:
        raise RuntimeError("cannot aggregate an empty structural candidate campaign")
    population_identities = {
        item.get("population_identity_sha256") for item in questions
    }
    if len(population_identities) != 1:
        raise RuntimeError("question structural populations differ")
    population_identity = _require_sha256(
        next(iter(population_identities)), "population identity"
    )
    for question in questions:
        expected = _structural_candidate_attribution(
            population_identity_sha256=population_identity,
            question_id=str(question.get("question_id", "")),
            question_identity_sha256=str(
                question.get("retrieval_question_part_sha256", "")
            ),
            arms=question.get("arms", ()),
        )
        if question.get("structural_candidate_attribution") != expected:
            raise RuntimeError("question structural candidate attribution changed")
    targets = [
        target
        for question in questions
        for target in question["structural_candidate_attribution"]["targets"]
    ]
    universe = [str(item["target_id"]) for item in targets]
    if len(universe) != len(set(universe)):
        raise RuntimeError("merged structural target IDs are not globally unique")
    attribution_sets = {
        label: [
            str(item["target_id"])
            for item in targets
            if item["primary_attribution"] == label
        ]
        for label in ARM_LABELS
    }
    attributed = [
        item for label in ARM_LABELS for item in attribution_sets[label]
    ]
    unattributed = [
        str(item["target_id"])
        for item in targets
        if item.get("primary_attribution") not in ARM_LABELS
    ]
    duplicate_count = len(attributed) - len(set(attributed))
    intersections = [
        item
        for left_index, left in enumerate(ARM_LABELS)
        for right in ARM_LABELS[left_index + 1 :]
        for item in set(attribution_sets[left]) & set(attribution_sets[right])
    ]
    union_matches = (
        set(attributed) == set(universe) and len(attributed) == len(universe)
    )
    if unattributed or duplicate_count or intersections or not union_matches:
        raise RuntimeError("merged structural candidate attribution is incomplete")
    body: dict[str, Any] = {
        "registry_role": "runtime_structural_candidate_attribution_only",
        "target_scope": "merged_question_scoped_fresh_closure_candidate_union",
        "population_identity_sha256": population_identity,
        "benchmark_target_tags_loaded": False,
        "desired_target_union_completeness_claimed": False,
        "desired_target_registry_format": (
            "memory-condense-retrieval-target-owner-registry-v1"
        ),
        "declared_structural_candidate_count": len(universe),
        "declared_structural_candidate_universe_sha256": identity_sha256(universe),
        "declared_structural_candidate_ids": universe,
        "primary_attribution_sets": attribution_sets,
        "targets": targets,
        "invariants": {
            "unattributed_structural_candidate_count": len(unattributed),
            "duplicate_primary_attribution_count": duplicate_count,
            "pairwise_primary_attribution_intersection_count": len(intersections),
            "primary_attribution_union_equals_declared_structural_candidate_universe": (
                union_matches
            ),
        },
    }
    body["manifest_identity_sha256"] = identity_sha256(body)
    return body


def _validate_exact_s0_result(
    question: _Question, result: RecallGuardedCumulativeRetrieval
) -> None:
    if canonical_json_bytes(asdict(result.predecessor.receipt)) != canonical_json_bytes(
        asdict(question.predecessor_receipt)
    ):
        raise RuntimeError("fresh cumulative retrieval changed exact sealed S0 receipt")
    if canonical_json_bytes(asdict(result.ladder.stages[0])) != canonical_json_bytes(
        asdict(question.s0_stage_receipt)
    ):
        raise RuntimeError("fresh cumulative retrieval changed exact sealed S0 stage")
    if list(result.predecessor.messages) != list(question.s0_messages):
        raise RuntimeError("fresh cumulative retrieval changed exact sealed S0 prompt")
    if tuple(result.predecessor.excerpts) != question.protected_excerpts:
        raise RuntimeError("fresh cumulative retrieval changed exact sealed S0 evidence")


def _question_artifact(
    *,
    question: _Question,
    result: RecallGuardedCumulativeRetrieval,
    condenser: Any,
    population: _Population,
    preflight_sha256: str,
    policy_receipt_sha256: str,
    runtime_source_binding: Mapping[str, Any],
    source_surface_sha256s: Mapping[str, str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    _validate_exact_s0_result(question, result)
    raw_arms = [
        _build_arm(
            label=label,
            plan=result.closure_plans[PLAN_INDEX[label]],
            question=question,
            condenser=condenser,
        )
        for label in ARM_LABELS
    ]
    structural_attribution = _structural_candidate_attribution(
        population_identity_sha256=population.adapter.population_identity_sha256,
        question_id=question.question_id,
        question_identity_sha256=question.retrieval_question_part_sha256,
        arms=raw_arms,
    )
    arms = _annotate_arm_targets(raw_arms)
    manifest_row = population.eligibility_manifest["questions"][question.ordinal]
    artifact: dict[str, Any] = {
        "format": QUESTION_FORMAT,
        "ordinal": question.ordinal,
        "question_id": question.question_id,
        "question_sha256": question.question_sha256,
        "dated_question_sha256": question.dated_question_sha256,
        "retrieval_question_part_sha256": question.retrieval_question_part_sha256,
        "population_identity_sha256": population.adapter.population_identity_sha256,
        "eligibility_manifest_sha256": population.eligibility_sha256,
        "eligibility_row_identity_sha256": manifest_row["row_identity_sha256"],
        "preflight_sha256": preflight_sha256,
        "policy_receipt_sha256": policy_receipt_sha256,
        "runtime_source_binding_sha256": runtime_source_binding[
            "runtime_source_binding_sha256"
        ],
        "retrieval_implementation_sha256": runtime_source_binding[
            "retrieval_implementation_sha256"
        ],
        "source_surface_sha256s": dict(source_surface_sha256s),
        "source_bindings": {
            "shard_offset": question.shard_offset,
            "source_shard_retrieval_sha256": (
                question.source_shard_retrieval_sha256
            ),
            "source_question_part_sha256": question.source_question_part_sha256,
            "combined_store_receipt_sha256": (
                question.combined_store_receipt_sha256
            ),
            "compilation_receipt_sha256": question.compilation_receipt_sha256,
        },
        "s0": {
            "stage_id": "causal_graph_coverage_predecessor",
            "stage_receipt": asdict(question.s0_stage_receipt),
            "predecessor_receipt": asdict(question.predecessor_receipt),
            "evidence": list(question.s0_evidence),
            "provider_messages": list(question.s0_messages),
            "provider_messages_sha256": identity_sha256(
                list(question.s0_messages)
            ),
        },
        "retrieval_invocation": {
            "count": 1,
            "elapsed_seconds": elapsed_seconds,
            "closure_plan_sha256s": [
                item.plan_sha256 for item in result.closure_plans
            ],
            "direct_expansion_receipt_sha256": (
                result.episode_expansion.receipt_sha256
            ),
            "representative_expansion_receipt_sha256": (
                result.representative_expansion.receipt_sha256
            ),
            "starvation_conditioned_packets_reused": False,
        },
        "structural_candidate_attribution": structural_attribution,
        "arms": arms,
        "provider_calls": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    artifact["artifact_identity_sha256"] = identity_sha256(artifact)
    if _contains_gold_key(artifact):
        raise RuntimeError("question artifact crossed the gold firewall")
    return artifact


def _verify_pool(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} candidate pool is missing")
    atoms, bundles = value.get("atom_identities"), value.get("bundle_identities")
    if not isinstance(atoms, list) or not isinstance(bundles, list):
        raise ValueError(f"{label} candidate pool is incomplete")
    if (
        value.get("atom_count") != len(atoms)
        or value.get("bundle_count") != len(bundles)
        or value.get("atom_identities_sha256") != identity_sha256(atoms)
        or value.get("bundle_identities_sha256") != identity_sha256(bundles)
    ):
        raise ValueError(f"{label} candidate pool seal changed")


def _validate_question_artifact(
    artifact: Mapping[str, Any],
    *,
    question: _Question,
    population: _Population,
    preflight_sha256: str,
    policy_receipt_sha256: str,
    runtime_source_binding: Mapping[str, Any],
    source_surface_sha256s: Mapping[str, str],
) -> None:
    body = dict(artifact)
    declared = body.pop("artifact_identity_sha256", None)
    manifest_row = population.eligibility_manifest["questions"][question.ordinal]
    if (
        declared != identity_sha256(body)
        or artifact.get("format") != QUESTION_FORMAT
        or artifact.get("ordinal") != question.ordinal
        or artifact.get("question_id") != question.question_id
        or artifact.get("question_sha256") != question.question_sha256
        or artifact.get("dated_question_sha256") != question.dated_question_sha256
        or artifact.get("population_identity_sha256")
        != population.adapter.population_identity_sha256
        or artifact.get("eligibility_manifest_sha256")
        != population.eligibility_sha256
        or artifact.get("eligibility_row_identity_sha256")
        != manifest_row["row_identity_sha256"]
        or artifact.get("preflight_sha256") != preflight_sha256
        or artifact.get("policy_receipt_sha256") != policy_receipt_sha256
        or artifact.get("runtime_source_binding_sha256")
        != runtime_source_binding["runtime_source_binding_sha256"]
        or artifact.get("retrieval_implementation_sha256")
        != runtime_source_binding["retrieval_implementation_sha256"]
        or artifact.get("source_surface_sha256s") != dict(source_surface_sha256s)
        or artifact.get("provider_calls") != 0
        or artifact.get("gold_loaded") is not False
        or _contains_gold_key(artifact)
    ):
        raise ValueError(f"question artifact binding changed at {question.ordinal}")
    s0 = artifact.get("s0")
    if not isinstance(s0, Mapping) or (
        s0.get("stage_receipt") != asdict(question.s0_stage_receipt)
        or s0.get("predecessor_receipt") != asdict(question.predecessor_receipt)
        or s0.get("evidence") != list(question.s0_evidence)
        or s0.get("provider_messages") != list(question.s0_messages)
    ):
        raise ValueError(f"question artifact S0 changed at {question.ordinal}")
    arms = artifact.get("arms")
    if not isinstance(arms, list) or tuple(
        item.get("arm_label") if isinstance(item, Mapping) else None for item in arms
    ) != ARM_LABELS:
        raise ValueError(f"question artifact arms changed at {question.ordinal}")
    expected_attribution = _structural_candidate_attribution(
        population_identity_sha256=population.adapter.population_identity_sha256,
        question_id=question.question_id,
        question_identity_sha256=question.retrieval_question_part_sha256,
        arms=arms,
    )
    if artifact.get("structural_candidate_attribution") != expected_attribution:
        raise ValueError(
            f"question artifact structural candidate attribution changed at "
            f"{question.ordinal}"
        )
    for arm in arms:
        assert isinstance(arm, Mapping)
        label = str(arm["arm_label"])
        _verify_pool(arm.get("candidate_pool"), label)
        projection = _arm_target_projection(arm)
        if any(arm.get(name) != value for name, value in projection.items()):
            raise ValueError(f"{label} target projection changed")
        admission = arm.get("admission")
        if not isinstance(admission, Mapping):
            raise ValueError(f"{label} admission is missing")
        messages = admission.get("provider_messages")
        if (
            not isinstance(messages, list)
            or admission.get("provider_messages_sha256") != identity_sha256(messages)
            or type(admission.get("addition_token_proxy")) is not int
            or admission["addition_token_proxy"] > ADDITION_TOKEN_CAP
            or type(admission.get("total_context_token_proxy")) is not int
            or admission["total_context_token_proxy"] > MAX_CONTEXT_TOKENS
            or type(admission.get("prompt_token_proxy")) is not int
            or admission["prompt_token_proxy"] > MAX_PROMPT_TOKENS
        ):
            raise ValueError(f"{label} admission budget or prompt changed")
        if admission.get("status") != "added" and messages != list(
            question.s0_messages
        ):
            raise ValueError(f"{label} no-op did not preserve exact S0")


def _load_campaign(
    args: argparse.Namespace,
) -> tuple[_Population, Mapping[str, Any], str, Any]:
    output = Path(args.output_root)
    manifest, manifest_sha = _read(output / "eligibility-manifest.json")
    expected_manifest = _require_sha256(
        args.expected_eligibility_sha256, "expected eligibility manifest"
    )
    if manifest_sha != expected_manifest:
        raise ValueError("sealed eligibility manifest digest changed")
    preflight, preflight_sha = _read(output / "preflight.json")
    expected_preflight = _require_sha256(
        args.expected_preflight_sha256, "expected preflight"
    )
    policy = load_frozen_validation_policy(Path(args.policy), device=str(args.device))
    repository = Path(__file__).resolve().parents[1]
    runtime_root = Path(args.runtime_source_root)
    runtime_binding = _runtime_source_binding(
        runtime_root,
        repository=repository,
    )
    surface = _source_surface(repository, runtime_source_root=runtime_root)
    historical_binding = _require_sha256(
        preflight.get("historical_validator_binding_sha256"),
        "historical validator binding",
    )
    if (
        preflight_sha != expected_preflight
        or preflight.get("eligibility_manifest_sha256") != manifest_sha
        or preflight.get("retrieval_sha256") != EXPECTED_RETRIEVAL_SHA256
        or preflight.get("baseline_final_answers_sha256")
        != EXPECTED_BASELINE_ANSWERS_SHA256
        or preflight.get("runtime_source_binding") != runtime_binding
        or preflight.get("source_surface_sha256s") != surface
        or preflight.get("provider_calls") != 0
        or preflight.get("gold_loaded") is not False
    ):
        raise ValueError("sealed preflight changed")
    population = _prepare_population(
        retrieval_path=Path(args.retrieval),
        baseline_answers_path=Path(args.baseline_answers),
        expected_question_count=int(args.expected_question_count),
        expected_eligible_count=int(args.expected_eligible_count),
        expected_historical_validator_binding_sha256=historical_binding,
    )
    if (
        manifest_sha != population.eligibility_sha256
        or canonical_json_bytes(manifest)
        != canonical_json_bytes(population.eligibility_manifest)
    ):
        raise ValueError("sealed eligibility manifest changed")
    return population, preflight, preflight_sha, policy


def _open_store(
    population: _Population,
    *,
    offset: int,
    store_root: Path,
    policy: Any,
    qwen_prefix: Path,
    qwen_choice: Path,
) -> tuple[Any, Any, Any]:
    _, binding = current_source_binding(
        policy.config, qwen_model_dir=qwen_prefix
    )
    embedder = binding.embedder
    prepared = None
    try:
        prepared = open_recall_guarded_cumulative_store(
            store_root / "shards" / f"offset-{offset:03d}" / "combined-store",
            config=policy.config,
            embedder=embedder,
            held_out_queries=_held_out_queries(population, offset),
            coverage_selector=_UnboundCoverageSelector(),
        )
    finally:
        embedder.close()
    try:
        selector, linker = _load_shared_qwen(
            policy.config, qwen_prefix, qwen_choice
        )
    except BaseException:
        if prepared is not None:
            prepared.close()
        raise
    prepared.condenser.set_context_candidate_selector(selector)
    return prepared, selector, linker


def _part_path(output: Path, ordinal: int) -> Path:
    return output / "questions" / f"q{ordinal:03d}.json"


def run_retrieve(args: argparse.Namespace) -> tuple[dict[str, Any], str, int]:
    population, preflight, preflight_sha, policy = _load_campaign(args)
    offset = int(args.shard_offset)
    if offset not in LOCKED_100Q_OFFSETS:
        raise ValueError("--shard-offset must be one of 0,10,...,90")
    store_report = _validate_store(
        population,
        offset=offset,
        store_root=Path(args.store_root),
        policy=policy,
    )
    expected_store = tuple(
        item
        for item in preflight["stores"]
        if item.get("shard_offset") == offset
    )
    if len(expected_store) != 1 or expected_store[0] != store_report:
        raise ValueError("selected store differs from sealed preflight")
    questions = tuple(
        row for row in population.eligible if row.shard_offset == offset
    )
    output = Path(args.output_root)
    missing: list[_Question] = []
    verified_parts: dict[int, tuple[dict[str, Any], str]] = {}
    policy_sha = preflight["policy"]["policy_receipt_sha256"]
    runtime_binding = preflight["runtime_source_binding"]
    surface = preflight["source_surface_sha256s"]
    for question in questions:
        path = _part_path(output, question.ordinal)
        if path.exists():
            artifact, digest = _read(path)
            _validate_question_artifact(
                artifact,
                question=question,
                population=population,
                preflight_sha256=preflight_sha,
                policy_receipt_sha256=policy_sha,
                runtime_source_binding=runtime_binding,
                source_surface_sha256s=surface,
            )
            verified_parts[question.ordinal] = (artifact, digest)
        else:
            missing.append(question)
    if not args.enable_expensive_retrieval:
        raise ValueError("retrieve requires --enable-expensive-retrieval")
    if int(args.authorized_retrieval_questions) != len(missing):
        raise ValueError(
            "--authorized-retrieval-questions must exactly equal the missing "
            f"eligible questions ({args.authorized_retrieval_questions} != "
            f"{len(missing)})"
        )
    physical = 0
    if missing:
        prepared, selector, linker = _open_store(
            population,
            offset=offset,
            store_root=Path(args.store_root),
            policy=policy,
            qwen_prefix=Path(args.qwen_prefix),
            qwen_choice=Path(args.qwen_choice),
        )
        try:
            artifact_id = prepared.compilation.artifact.artifact_id
            for question in missing:
                started = time.perf_counter()
                result = retrieve_recall_guarded_cumulative_packet(
                    prepared.condenser,
                    query=question.raw_question,
                    prompt_question=question.dated_question,
                    retrieval=policy.config.retrieval,
                    artifact_id=artifact_id,
                    max_context_tokens=MAX_CONTEXT_TOKENS,
                    max_prompt_tokens=MAX_PROMPT_TOKENS,
                    responder_output_token_reserve=RESPONDER_OUTPUT_TOKEN_RESERVE,
                    episode_policy=_episode_policy(artifact_id),
                    representative_linker=linker,
                    representative_policy=_representative_policy(artifact_id),
                    source_router_max_sources=SOURCE_ROUTER_MAX_SOURCES,
                    source_router_rrf_constant=SOURCE_ROUTER_RRF_CONSTANT,
                    closure_policy=_closure_policy(),
                    require_certified_coverage_runtime=True,
                    require_owned_representative_runtime=True,
                )
                artifact = _question_artifact(
                    question=question,
                    result=result,
                    condenser=prepared.condenser,
                    population=population,
                    preflight_sha256=preflight_sha,
                    policy_receipt_sha256=policy_sha,
                    runtime_source_binding=runtime_binding,
                    source_surface_sha256s=surface,
                    elapsed_seconds=time.perf_counter() - started,
                )
                digest = _publish(_part_path(output, question.ordinal), artifact)
                verified_parts[question.ordinal] = (artifact, digest)
                physical += 1
                print(
                    f"Question {question.ordinal + 1}/100 {question.question_id}: "
                    f"published {digest}",
                    flush=True,
                )
        finally:
            selector.close()
            prepared.close()
            del linker, selector, prepared
            gc.collect()
    ordered = [verified_parts[row.ordinal] for row in questions]
    index: dict[str, Any] = {
        "format": SHARD_INDEX_FORMAT,
        "shard_offset": offset,
        "eligibility_manifest_sha256": population.eligibility_sha256,
        "preflight_sha256": preflight_sha,
        "policy_receipt_sha256": policy_sha,
        "runtime_source_binding_sha256": runtime_binding[
            "runtime_source_binding_sha256"
        ],
        "retrieval_implementation_sha256": runtime_binding[
            "retrieval_implementation_sha256"
        ],
        "eligible_question_count": len(questions),
        "question_ordinals": [row.ordinal for row in questions],
        "question_artifact_sha256s": [digest for _, digest in ordered],
        "retrieval_invocations_this_run": physical,
        "provider_calls": 0,
        "gold_loaded": False,
    }
    index["index_identity_sha256"] = identity_sha256(index)
    digest = _publish(
        output / "shards" / f"offset-{offset:03d}.json", index
    )
    return index, digest, physical


def run_merge(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    population, preflight, preflight_sha, _policy = _load_campaign(args)
    output = Path(args.output_root)
    policy_sha = preflight["policy"]["policy_receipt_sha256"]
    runtime_binding = preflight["runtime_source_binding"]
    surface = preflight["source_surface_sha256s"]
    rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for question in population.eligible:
        artifact, digest = _read(_part_path(output, question.ordinal))
        _validate_question_artifact(
            artifact,
            question=question,
            population=population,
            preflight_sha256=preflight_sha,
            policy_receipt_sha256=policy_sha,
            runtime_source_binding=runtime_binding,
            source_surface_sha256s=surface,
        )
        rows.append(artifact)
        hashes.append(digest)
    merged: dict[str, Any] = {
        "format": MERGED_FORMAT,
        "arm_labels": list(ARM_LABELS),
        "eligibility_manifest_sha256": population.eligibility_sha256,
        "preflight_sha256": preflight_sha,
        "policy_receipt_sha256": policy_sha,
        "runtime_source_binding_sha256": runtime_binding[
            "runtime_source_binding_sha256"
        ],
        "retrieval_implementation_sha256": runtime_binding[
            "retrieval_implementation_sha256"
        ],
        "question_count": len(rows),
        "question_ordinals": [row["ordinal"] for row in rows],
        "question_artifact_sha256s": hashes,
        "questions": rows,
        "structural_candidate_attribution": (
            _aggregate_structural_candidate_attribution(rows)
        ),
        "retrieval_invocation_count": len(rows),
        "provider_calls": 0,
        "gold_loaded": False,
    }
    merged["artifact_identity_sha256"] = identity_sha256(merged)
    return merged, _publish(output / "retrieval-generation.json", merged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "retrieve", "merge"))
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--baseline-answers", type=Path, default=DEFAULT_BASELINE_ANSWERS
    )
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--runtime-source-root",
        type=Path,
        default=DEFAULT_RUNTIME_SOURCE_ROOT,
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--qwen-prefix", type=Path, default=DEFAULT_QWEN_PREFIX)
    parser.add_argument("--qwen-choice", type=Path, default=DEFAULT_QWEN_CHOICE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-eligibility-sha256")
    parser.add_argument("--expected-preflight-sha256")
    parser.add_argument("--shard-offset", type=int)
    parser.add_argument("--enable-expensive-retrieval", action="store_true")
    parser.add_argument("--authorized-retrieval-questions", type=int, default=0)
    parser.set_defaults(
        expected_question_count=EXPECTED_QUESTION_COUNT,
        expected_eligible_count=EXPECTED_ELIGIBLE_COUNT,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "preflight":
        if (
            args.enable_expensive_retrieval
            or args.authorized_retrieval_questions != 0
            or args.shard_offset is not None
        ):
            raise ValueError("preflight forbids retrieval authorization")
        artifact, manifest_sha, preflight_sha = run_preflight(args)
        seconds = artifact["runtime_reference"][
            "historical_comparable_retrieval_seconds"
        ]
        print(
            f"Independent closure preflight: eligible="
            f"{artifact['eligible_question_count']}; retrieval_calls=0; "
            f"provider_calls=0; historical_compute_hours={seconds / 3600:.2f}; "
            f"eligibility_sha256={manifest_sha}; preflight_sha256={preflight_sha}"
        )
        return 0
    if not args.expected_eligibility_sha256 or not args.expected_preflight_sha256:
        raise ValueError(
            "retrieve/merge require both expected eligibility and preflight SHA-256"
        )
    if args.phase == "retrieve":
        if args.shard_offset is None:
            raise ValueError("retrieve requires --shard-offset")
        artifact, digest, physical = run_retrieve(args)
        print(
            f"Independent closure shard {artifact['shard_offset']:03d}: "
            f"questions={artifact['eligible_question_count']}; "
            f"retrieval_calls={physical}; provider_calls=0; sha256={digest}"
        )
        return 0
    if args.enable_expensive_retrieval or args.authorized_retrieval_questions != 0:
        raise ValueError("merge forbids retrieval authorization")
    artifact, digest = run_merge(args)
    print(
        f"Independent closure merge: questions={artifact['question_count']}; "
        f"provider_calls=0; sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADDITION_TOKEN_CAP",
    "ARM_LABELS",
    "GLOBAL_ARM",
    "REPRESENTATIVE_ARM",
    "main",
    "run_merge",
    "run_preflight",
    "run_retrieve",
]
