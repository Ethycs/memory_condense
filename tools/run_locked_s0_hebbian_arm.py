#!/usr/bin/env python3
"""Run the isolated locked S0-plus-Hebbian retrieval mechanism arm.

The expensive phase builds one causal chronological co-access history for each
of the ten already-sealed validation shard stores.  The build is shard-level
resumable and never rebuilds a corpus or source store.  Query/admission starts
from the exact S0 chunk coordinates (never S3), selects at most 256 graph
neighbors, applies the robust support/co-access gates, removes exact S0
duplicates only after selection, and appends at most one bounded row.

Provider-free phases seal history inputs, retrieval/admission decisions, and a
generic structural discovery ledger.  Answer calls are made only for questions
with one admitted row; every fail-closed question reuses the exact sealed S0
prediction without a dependent call.  No phase loads benchmark gold, category,
source-topology, oracle, or judge fields.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:  # support ``python tools/run_...py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.coaccess_graph import rank_discount
from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_hebbian_h2 import (
    _verified_derived_inputs,
    load_fast_hebbian_h2_history,
)
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    implementation_sha256,
)
from memory_condense.eval import run_fast_1m_hebbian as history_runner
from memory_condense.persistence.db import Database
from memory_condense.search.indexes.retrieval_models import hydrate_chunk_result
from tools import run_locked_retrieval_mechanism_arm as s0_runner
from tools.run_routed_full_source_repair import (
    _distribution,
    _make_provider_client,
    _publish,
    _read,
    _record_by_messages,
    _stable_batch,
)


ARM_LABEL = "S0_PLUS_HEBBIAN"
PARENT_ARM_LABEL = "S0_CONTROL"
HISTORY_PREFLIGHT_FORMAT = (
    "memory-condense-locked-s0-hebbian-history-preflight-v1"
)
PROPOSAL_FORMAT = "memory-condense-locked-s0-hebbian-proposals-v1"
ANSWER_PREFLIGHT_FORMAT = (
    "memory-condense-locked-s0-hebbian-answer-preflight-v1"
)
RUN_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-run-v1"
TARGET_LEDGER_FORMAT = "memory-condense-structural-target-ledger-v1"

DEFAULT_RETRIEVAL = s0_runner.DEFAULT_RETRIEVAL
DEFAULT_BASELINE_ANSWERS = s0_runner.DEFAULT_BASELINE_ANSWERS
DEFAULT_S0_RUN = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-control-v1/run.json"
)
DEFAULT_SHARDS_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "validation-20260822/shards"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-plus-hebbian-v1"
)
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

EXPECTED_RETRIEVAL_SHA256 = s0_runner.EXPECTED_RETRIEVAL_SHA256
EXPECTED_BASELINE_ANSWERS_SHA256 = s0_runner.EXPECTED_BASELINE_ANSWERS_SHA256
EXPECTED_QUESTION_COUNT = 100
SHARD_OFFSETS = tuple(range(0, EXPECTED_QUESTION_COUNT, 10))
MAX_SEED_CHUNKS = 64
MAX_CANDIDATES = 256
MIN_SUPPORT = 2
MIN_COACCESS_COUNT = 2
MAX_ADDITIONS = 1
MAX_ADDED_TOKENS = 384
MAX_PROMPT_TOKENS = 8_000
MAX_ANSWER_OUTPUT_TOKENS = 256
HALF_LIFE_TURNS = 200.0
MIN_SCORE = 0.0  # support/co-access are the robust gates; add no hidden score gate
HISTORY_RETRIEVAL_K = 10
HISTORY_EXPANSION_TOKENS = 1_600
HISTORY_MAX_PROMPT_TOKENS = 128
HISTORY_MAX_EVENT_NODES = 9
HISTORY_NEW_EVENT_NODES = 5
HISTORY_EMBEDDING_DEVICE = "cuda"
HISTORY_EMBEDDING_BATCH_SIZE = 32
MEASURED_SECONDS_PER_TURN = (35 * 60 + 37.8) / 5_400

_FORBIDDEN_KEYS = frozenset(
    {
        "answer_session_ids",
        "category",
        "evidence_sources",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
        "source_completeness",
        "source_topology",
        "oracle",
        "oracle_label",
        "primary_owner",
    }
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase SHA-256 digest")
    return value


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_KEYS
            or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"digest sidecar is absent: {sidecar}")
    expected = file_sha256(path)
    if sidecar.read_bytes() != f"{expected}  {path.name}\n".encode("ascii"):
        raise ValueError(f"digest sidecar changed: {sidecar}")
    return expected


def _validate_common(args: argparse.Namespace) -> None:
    if args.gateway_url != DEFAULT_GATEWAY_URL or args.model != DEFAULT_MODEL:
        raise ValueError("Hebbian arm requires the locked central-dev Terra route")
    if args.expected_question_count != EXPECTED_QUESTION_COUNT:
        raise ValueError("Hebbian arm requires the exact locked-100 population")
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be positive")


def _verified_s0(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], str]:
    """Historically validate S0 once, then replay its immutable journals."""

    plan = s0_runner._prepare(
        retrieval_path=Path(args.retrieval),
        baseline_answers_path=Path(args.baseline_answers),
        expected_retrieval_sha256=str(args.expected_retrieval_sha256),
        expected_baseline_answers_sha256=str(
            args.expected_baseline_answers_sha256
        ),
        expected_question_count=int(args.expected_question_count),
    )
    run_path = Path(args.s0_run)
    source, source_sha = s0_runner._read(run_path)
    if source_sha != _require_sha256(
        args.expected_s0_run_sha256,
        "S0 run expected SHA-256",
    ):
        raise ValueError("S0 run artifact SHA-256 changed")
    batch = s0_runner._runtime(
        plan,
        checkpoint_dir=Path(
            args.s0_checkpoint_dir or run_path.parent / "terra-answer-calls"
        ),
        client=None,
        max_concurrency=int(args.max_concurrency),
    ).run()
    if batch.usage.physical_calls or batch.usage.checkpoint_hits != plan.exact_calls:
        raise RuntimeError("S0 replay did not consume its complete journal set")
    expected = s0_runner._run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("S0 run differs from immutable runtime journals")
    return plan, source, source_sha


def _history_policy() -> dict[str, Any]:
    return {
        "format": "memory-condense-locked-s0-hebbian-history-policy-v1",
        "history_count": len(SHARD_OFFSETS),
        "chronological": True,
        "causal_capture_before_current_user_append": True,
        "source_store_mutated": False,
        "corpus_rebuilt": False,
        "retrieval_k": HISTORY_RETRIEVAL_K,
        "expansion_tokens": HISTORY_EXPANSION_TOKENS,
        "history_max_prompt_tokens": HISTORY_MAX_PROMPT_TOKENS,
        "max_event_nodes": HISTORY_MAX_EVENT_NODES,
        "new_event_nodes": HISTORY_NEW_EVENT_NODES,
        "embedding_device": HISTORY_EMBEDDING_DEVICE,
        "embedding_batch_size": HISTORY_EMBEDDING_BATCH_SIZE,
        "resumption_granularity": "sealed-shard-history-root-v1",
        "validation_shard_adapter": (
            "tool-only-minimal-history-producer-surface-v1"
        ),
        "validation_shard_bytes_unchanged": True,
    }


def _arm_policy() -> dict[str, Any]:
    return {
        "format": "memory-condense-locked-s0-hebbian-policy-v1",
        "seed_stage": "causal_graph_coverage_predecessor",
        "s3_consumed": False,
        "max_s0_seed_chunks": MAX_SEED_CHUNKS,
        "seed_order": "first-in-exact-s0-protected-chunk-order-v1",
        "max_neighbor_candidates": MAX_CANDIDATES,
        "ranking": "graph-score-desc-support-desc-chunk-id-asc-v1",
        "minimum_support": MIN_SUPPORT,
        "minimum_coaccess_count": MIN_COACCESS_COUNT,
        "half_life_turns": HALF_LIFE_TURNS,
        "minimum_score": MIN_SCORE,
        "post_selection_dedup": (
            "exact-s0-chunk-or-source-and-text-projection-v1"
        ),
        "max_appended_rows": MAX_ADDITIONS,
        "max_added_tokens": MAX_ADDED_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_answer_output_tokens": MAX_ANSWER_OUTPUT_TOKENS,
        "s0_membership_order_protected": True,
        "s0_answer_operator_protected": True,
        "shared_residual_borrowing": False,
        "no_dependent_call_without_admission": True,
        "fallback": "exact-sealed-s0-control-prediction-v1",
    }


@dataclass(frozen=True, slots=True)
class _ShardInput:
    shard_id: str
    offset: int
    retrieval_path: Path
    retrieval_sha256: str
    source_store: Path
    artifact: Any
    source: Any
    eligible_query_count: int
    question_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ShardQuestionCoordinate:
    ordinal: int
    question_id: str


@dataclass(frozen=True, slots=True)
class _ShardRetrievalAdapter:
    """Minimal typed surface consumed by the unchanged history producer."""

    source_path: str
    raw_sha256: str
    format: str
    campaign_format: str
    population_identity_sha256: str
    source_store_receipt_sha256: str
    combined_store_receipt_sha256: str
    retrieval_implementation_sha256: str
    retrieval_policy_sha256: str
    transcript_tokens: int
    turn_count: int
    retained_request_token_state_bytes: int
    stage_ids: tuple[str, ...]
    questions: tuple[_ShardQuestionCoordinate, ...]

    @property
    def question_count(self) -> int:
        return len(self.questions)


def _adapt_shard_retrieval(
    path: Path,
    *,
    expected_sha256: str,
    expected_offset: int,
) -> _ShardRetrievalAdapter:
    payload, digest = _read(path, expected_sha256=expected_sha256)
    raw_questions = payload.get("questions")
    stage_ids = payload.get("stage_ids")
    if (
        digest != expected_sha256
        or payload.get("format")
        != "memory-condense-recall-guarded-cumulative-validation-shard-retrieval-v1"
        or payload.get("campaign_format")
        != "memory-condense-recall-guarded-cumulative-1m-validation-campaign-v1"
        or payload.get("shard_offset") != expected_offset
        or payload.get("question_count") != 10
        or payload.get("provider_calls") != 0
        or payload.get("gold_fields_present") is not False
        or not isinstance(raw_questions, list)
        or len(raw_questions) != 10
        or not isinstance(stage_ids, list)
        or len(stage_ids) != 4
    ):
        raise ValueError("locked validation shard retrieval envelope changed")
    questions: list[_ShardQuestionCoordinate] = []
    for local, row in enumerate(raw_questions):
        ordinal = expected_offset + local
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != ordinal
            or row.get("local_ordinal") != local
            or row.get("shard_offset") != expected_offset
            or not isinstance(row.get("question_id"), str)
        ):
            raise ValueError("locked validation shard question order changed")
        questions.append(
            _ShardQuestionCoordinate(
                ordinal=ordinal,
                question_id=str(row["question_id"]),
            )
        )
    required_digests = (
        "population_identity_sha256",
        "source_store_receipt_sha256",
        "combined_store_receipt_sha256",
        "retrieval_implementation_sha256",
        "retrieval_policy_sha256",
    )
    digests = {
        name: _require_sha256(payload.get(name), f"shard {name}")
        for name in required_digests
    }
    transcript_tokens = payload.get("transcript_tokens")
    turn_count = payload.get("turn_count")
    if (
        type(transcript_tokens) is not int
        or transcript_tokens < 1
        or type(turn_count) is not int
        or turn_count < 1
    ):
        raise ValueError("locked validation shard dimensions changed")
    return _ShardRetrievalAdapter(
        source_path=str(path),
        raw_sha256=digest,
        format=str(payload["format"]),
        campaign_format=str(payload["campaign_format"]),
        population_identity_sha256=digests["population_identity_sha256"],
        source_store_receipt_sha256=digests["source_store_receipt_sha256"],
        combined_store_receipt_sha256=digests[
            "combined_store_receipt_sha256"
        ],
        retrieval_implementation_sha256=digests[
            "retrieval_implementation_sha256"
        ],
        retrieval_policy_sha256=digests["retrieval_policy_sha256"],
        transcript_tokens=transcript_tokens,
        turn_count=turn_count,
        retained_request_token_state_bytes=0,
        stage_ids=tuple(str(row) for row in stage_ids),
        questions=tuple(questions),
    )


def _shard_inputs(args: argparse.Namespace) -> tuple[_ShardInput, ...]:
    root = Path(args.shards_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("locked shard root must be a regular directory")
    expected_names = tuple(f"offset-{offset:03d}" for offset in SHARD_OFFSETS)
    observed_names = tuple(
        item.name
        for item in sorted(root.iterdir(), key=lambda row: row.name)
        if item.is_dir()
    )
    if observed_names != expected_names:
        raise ValueError("locked shard directory population changed")
    rows: list[_ShardInput] = []
    seen_questions: list[str] = []
    for offset, shard_id in zip(SHARD_OFFSETS, expected_names, strict=True):
        shard_root = root / shard_id
        retrieval_path = shard_root / "retrieval.json"
        retrieval_sha = _sidecar_digest(retrieval_path)
        artifact = _adapt_shard_retrieval(
            retrieval_path,
            expected_sha256=retrieval_sha,
            expected_offset=offset,
        )
        source_store = shard_root / "combined-store"
        source = history_runner._validate_source_store(source_store, artifact)
        questions = tuple(artifact.questions)
        question_ids = tuple(row.question_id for row in questions)
        if (
            len(questions) != 10
            or tuple(row.ordinal for row in questions)
            != tuple(range(offset, offset + 10))
            or len(set(question_ids)) != len(question_ids)
        ):
            raise ValueError(f"{shard_id} changed its locked question slice")
        eligible = history_runner._eligible_historical_queries(
            source.database_path,
            max_prompt_tokens=HISTORY_MAX_PROMPT_TOKENS,
        )
        rows.append(
            _ShardInput(
                shard_id=shard_id,
                offset=offset,
                retrieval_path=retrieval_path,
                retrieval_sha256=retrieval_sha,
                source_store=source_store,
                artifact=artifact,
                source=source,
                eligible_query_count=len(eligible),
                question_ids=question_ids,
            )
        )
        seen_questions.extend(question_ids)
    if len(seen_questions) != EXPECTED_QUESTION_COUNT or len(
        set(seen_questions)
    ) != EXPECTED_QUESTION_COUNT:
        raise ValueError("shards do not form one unique locked-100 population")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _HistorySource:
    s0_plan: Any
    s0_run: Mapping[str, Any]
    s0_run_sha256: str
    shards: tuple[_ShardInput, ...]
    preflight: Mapping[str, Any]
    preflight_sha256: str


def _history_preflight_body(
    args: argparse.Namespace,
    *,
    s0_plan: Any,
    s0_run_sha256: str,
    shards: tuple[_ShardInput, ...],
) -> dict[str, Any]:
    s0_ids = tuple(row.question_id for row in s0_plan.rows)
    shard_ids = tuple(
        question_id for shard in shards for question_id in shard.question_ids
    )
    if shard_ids != s0_ids:
        raise ValueError("locked shard order differs from sealed S0 order")
    total_turns = sum(row.artifact.turn_count for row in shards)
    total_queries = sum(row.eligible_query_count for row in shards)
    history_policy = _history_policy()
    arm_policy = _arm_policy()
    rows: list[dict[str, Any]] = []
    for shard in shards:
        history_root = Path(args.output_root) / "histories" / shard.shard_id
        rows.append(
            {
                "shard_id": shard.shard_id,
                "offset": shard.offset,
                "question_count": len(shard.question_ids),
                "question_ids_sha256": identity_sha256(list(shard.question_ids)),
                "turn_count": shard.artifact.turn_count,
                "eligible_historical_query_count": shard.eligible_query_count,
                "retrieval_path": str(shard.retrieval_path),
                "retrieval_sha256": shard.retrieval_sha256,
                "combined_store_path": str(shard.source_store),
                "combined_store_receipt_sha256": (
                    shard.source.receipt.receipt_sha256
                ),
                "source_manifest_sha256": shard.source.manifest_sha256,
                "source_database_sha256": (
                    shard.source.receipt.target_database_sha256
                ),
                "source_index_sha256": shard.source.receipt.target_index_sha256,
                "history_output_root": str(history_root),
                "history_build_command_argv_template": [
                    "pixi",
                    "run",
                    "-e",
                    "dev",
                    "python",
                    "tools/run_locked_s0_hebbian_arm.py",
                    "--phase",
                    "history-build",
                    "--expected-s0-run-sha256",
                    s0_run_sha256,
                    "--expected-history-preflight-sha256",
                    "<HISTORY_PREFLIGHT_SHA256>",
                    "--shard-id",
                    shard.shard_id,
                    "--enable-history-build",
                    "--authorized-history-shards",
                    "1",
                ],
            }
        )
    result: dict[str, Any] = {
        "format": HISTORY_PREFLIGHT_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "retrieval_sha256": s0_plan.population.retrieval_sha256,
        "population_identity_sha256": s0_plan.population.population_identity_sha256,
        "historical_validator_binding_sha256": s0_plan.population.binding_sha256,
        "s0_control_run_sha256": s0_run_sha256,
        "source_package_implementation_sha256": implementation_sha256(),
        "environment_lock_sha256": environment_lock_sha256(),
        "tool_source_sha256": file_sha256(Path(__file__)),
        "history_policy": history_policy,
        "history_policy_sha256": identity_sha256(history_policy),
        "arm_policy": arm_policy,
        "arm_policy_sha256": identity_sha256(arm_policy),
        "question_count": EXPECTED_QUESTION_COUNT,
        "shard_count": len(shards),
        "total_turn_count": total_turns,
        "total_eligible_historical_query_count": total_queries,
        "estimated_sequential_seconds": round(
            total_turns * MEASURED_SECONDS_PER_TURN,
            3,
        ),
        "estimate_basis": {
            "measured_turn_count": 5_400,
            "measured_elapsed_seconds": 35 * 60 + 37.8,
            "seconds_per_turn": MEASURED_SECONDS_PER_TURN,
        },
        "shards": rows,
        "history_model_loads": 0,
        "history_embedding_calls": 0,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_forbidden_key(result):
        raise RuntimeError("Hebbian history preflight crossed the gold firewall")
    return result


def _build_history_source(args: argparse.Namespace) -> _HistorySource:
    _validate_common(args)
    s0_plan, s0_run, s0_sha = _verified_s0(args)
    shards = _shard_inputs(args)
    preflight = _history_preflight_body(
        args,
        s0_plan=s0_plan,
        s0_run_sha256=s0_sha,
        shards=shards,
    )
    return _HistorySource(
        s0_plan=s0_plan,
        s0_run=s0_run,
        s0_run_sha256=s0_sha,
        shards=shards,
        preflight=preflight,
        preflight_sha256=_digest(preflight),
    )


def _history_preflight_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "history-preflight.json"


def run_history_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if (
        args.enable_history_build
        or args.authorized_history_shards
        or args.enable_provider
        or args.authorized_provider_calls
    ):
        raise ValueError("history preflight forbids model/provider authorization")
    source = _build_history_source(args)
    return dict(source.preflight), _publish(
        _history_preflight_path(args),
        source.preflight,
    )


def _load_verified_history_preflight(
    args: argparse.Namespace,
) -> _HistorySource:
    source = _build_history_source(args)
    payload, raw_sha = _read(_history_preflight_path(args))
    expected_sha = _require_sha256(
        args.expected_history_preflight_sha256,
        "history preflight expected SHA-256",
    )
    if (
        raw_sha != expected_sha
        or raw_sha != source.preflight_sha256
        or canonical_json_bytes(payload) != canonical_json_bytes(source.preflight)
    ):
        raise ValueError("history preflight differs from exact sealed inputs")
    return source


def _history_args(args: argparse.Namespace, shard: _ShardInput) -> Any:
    return SimpleNamespace(
        phase="history",
        retrieval=shard.retrieval_path,
        expected_retrieval_sha256=shard.retrieval_sha256,
        source_store=shard.source_store,
        output_root=Path(args.output_root) / "histories" / shard.shard_id,
        history_root=None,
        retrieval_k=HISTORY_RETRIEVAL_K,
        expansion_tokens=HISTORY_EXPANSION_TOKENS,
        history_max_prompt_tokens=HISTORY_MAX_PROMPT_TOKENS,
        max_event_nodes=HISTORY_MAX_EVENT_NODES,
        new_event_nodes=HISTORY_NEW_EVENT_NODES,
        embedding_device=HISTORY_EMBEDDING_DEVICE,
        embedding_batch_size=HISTORY_EMBEDDING_BATCH_SIZE,
    )


def run_history_build(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("history build forbids providers")
    if not args.enable_history_build or args.authorized_history_shards != 1:
        raise ValueError(
            "history build requires --enable-history-build and exactly one "
            "authorized shard"
        )
    source = _load_verified_history_preflight(args)
    matches = tuple(row for row in source.shards if row.shard_id == args.shard_id)
    if len(matches) != 1:
        raise ValueError("--shard-id must name one sealed history shard")
    shard = matches[0]
    original_loader = history_runner._load_artifact

    def sealed_shard_loader(path: Path, expected_sha256: str) -> Any:
        if (
            Path(path).resolve() != shard.retrieval_path.resolve()
            or expected_sha256 != shard.retrieval_sha256
        ):
            raise ValueError("history producer requested another retrieval shard")
        return shard.artifact

    history_runner._load_artifact = sealed_shard_loader
    try:
        result = history_runner.run_history(_history_args(args, shard))
    finally:
        history_runner._load_artifact = original_loader
    receipt: dict[str, Any] = {
        "format": "memory-condense-locked-s0-hebbian-history-build-receipt-v1",
        "arm_label": ARM_LABEL,
        "history_preflight_sha256": source.preflight_sha256,
        "shard_id": shard.shard_id,
        "validation_shard_adapter": {
            "format": "memory-condense-locked-history-shard-adapter-v1",
            "source_retrieval_format": shard.artifact.format,
            "source_retrieval_sha256": shard.retrieval_sha256,
            "tool_source_sha256": source.preflight["tool_source_sha256"],
        },
        "history_result": result,
        "provider_calls": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    path = (
        Path(args.output_root)
        / "history-build-receipts"
        / f"{shard.shard_id}.json"
    )
    return receipt, _publish(path, receipt)


@dataclass(frozen=True, slots=True)
class _GraphCandidate:
    rank: int
    source_chunk_id: str
    source_id: str
    raw_text: str
    rendered_text: str
    evidence_text_sha256: str
    score: float
    support: int
    anchor_chunk_id: str
    coaccess_count: int
    last_reinforced_turn: int


@dataclass(frozen=True, slots=True)
class _CandidateDecision:
    candidate: _GraphCandidate
    post_dedup_disposition: str
    added_token_proxy: int | None
    proposed_prompt_token_proxy: int | None
    messages: tuple[dict[str, str], ...] | None
    candidate_receipt_sha256: str

    @property
    def admitted(self) -> bool:
        return self.post_dedup_disposition == "admitted_after_budget"


@dataclass(frozen=True, slots=True)
class _QuestionProposal:
    ordinal: int
    question_id: str
    shard_id: str
    s0_chunk_ids: tuple[str, ...]
    seed_chunk_ids: tuple[str, ...]
    history_receipt_sha256: str
    derived_store_receipt_sha256: str
    association_artifact_id: str
    decisions: tuple[_CandidateDecision, ...]
    admitted: _CandidateDecision | None
    outcome: str
    base_prompt_token_proxy: int
    proposal_receipt_sha256: str

    @property
    def messages(self) -> tuple[dict[str, str], ...] | None:
        return None if self.admitted is None else self.admitted.messages


@dataclass(frozen=True, slots=True)
class _ProposalPlan:
    history_source: _HistorySource
    history_bindings: tuple[Mapping[str, Any], ...]
    questions: tuple[_QuestionProposal, ...]
    artifact: Mapping[str, Any]
    artifact_sha256: str
    valid_ordinals: tuple[int, ...]
    prompts: tuple[tuple[dict[str, str], ...], ...]
    answer_preflight: FastPromptPopulation | None

    @property
    def unique_calls(self) -> int:
        return (
            0
            if self.answer_preflight is None
            else self.answer_preflight.unique_prompt_count
        )


def _source_id(result: Any) -> str:
    source_id = result.durable_source_id.strip()
    if not source_id or len(result.source_hints) > 1:
        raise ValueError(
            f"chunk {result.chunk.chunk_id!r} has ambiguous source provenance"
        )
    return source_id


def _render_candidate(result: Any) -> str:
    turn = result.turn
    if turn is None:
        raise ValueError(
            f"chunk {result.chunk.chunk_id!r} has no chronological turn"
        )
    created_at = turn.created_at.strftime("%Y/%m/%d (%a) %H:%M")
    return f"[Hebbian @ {created_at} | {turn.role}] {result.chunk.text}"


def _candidate_block(rendered_text: str) -> str:
    return (
        "\n\nAdditional retrieved excerpt from causal co-access:\n"
        f"[H1] {rendered_text}"
    )


def _append_candidate_messages(
    base_messages: Sequence[Mapping[str, str]],
    rendered_text: str,
) -> tuple[dict[str, str], ...]:
    messages = tuple(
        {"role": str(row["role"]), "content": str(row["content"])}
        for row in base_messages
    )
    if len(messages) != 2 or tuple(row["role"] for row in messages) != (
        "system",
        "user",
    ):
        raise ValueError("sealed S0 answer operator changed message shape")
    marker = "\n\nQuestion: "
    user = messages[-1]["content"]
    if user.count(marker) != 1:
        raise ValueError("sealed S0 user packet changed its question marker")
    prefix, suffix = user.split(marker, 1)
    appended = {
        "role": "user",
        "content": prefix + _candidate_block(rendered_text) + marker + suffix,
    }
    return (dict(messages[0]), appended)


def _candidate_body(
    candidate: _GraphCandidate,
    *,
    disposition: str,
    added_tokens: int | None,
    proposed_prompt_tokens: int | None,
) -> dict[str, Any]:
    return {
        "format": "memory-condense-locked-s0-hebbian-candidate-v1",
        "rank": candidate.rank,
        "source_chunk_id": candidate.source_chunk_id,
        "source_id": candidate.source_id,
        "evidence_text_sha256": candidate.evidence_text_sha256,
        "score": candidate.score,
        "support": candidate.support,
        "anchor_chunk_id": candidate.anchor_chunk_id,
        "coaccess_count": candidate.coaccess_count,
        "last_reinforced_turn": candidate.last_reinforced_turn,
        "post_dedup_disposition": disposition,
        "added_token_proxy": added_tokens,
        "proposed_prompt_token_proxy": proposed_prompt_tokens,
    }


def _decide_candidates(
    base_messages: Sequence[Mapping[str, str]],
    candidates: Sequence[_GraphCandidate],
    *,
    all_s0_chunk_ids: Sequence[str],
    s0_exact_projections: set[tuple[str, str]],
) -> tuple[tuple[_CandidateDecision, ...], _CandidateDecision | None, str]:
    """Apply post-selection dedup and protected admission to ranked neighbors."""

    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("Hebbian graph returned more than 256 candidates")
    s0_chunks = set(all_s0_chunk_ids)
    seen_projections: set[tuple[str, str]] = set()
    decisions: list[_CandidateDecision] = []
    admitted: _CandidateDecision | None = None
    for candidate in candidates:
        projection = (candidate.source_id, candidate.evidence_text_sha256)
        added_tokens: int | None = None
        proposed_tokens: int | None = None
        messages: tuple[dict[str, str], ...] | None = None
        if candidate.source_chunk_id in s0_chunks:
            disposition = "excluded_post_selection_s0_chunk_duplicate"
        elif projection in s0_exact_projections:
            disposition = "excluded_post_selection_s0_projection_duplicate"
        elif projection in seen_projections:
            disposition = "excluded_post_selection_candidate_duplicate"
        else:
            seen_projections.add(projection)
            candidate_messages = _append_candidate_messages(
                base_messages,
                candidate.rendered_text,
            )
            added_tokens = count_tokens(_candidate_block(candidate.rendered_text))
            proposed_tokens = count_chat_prompt_token_proxy(candidate_messages)
            if admitted is not None:
                disposition = "rejected_addition_cap"
            elif added_tokens > MAX_ADDED_TOKENS:
                disposition = "rejected_added_token_cap"
            elif proposed_tokens > MAX_PROMPT_TOKENS:
                disposition = "rejected_prompt_cap"
            else:
                disposition = "admitted_after_budget"
                messages = candidate_messages
        body = _candidate_body(
            candidate,
            disposition=disposition,
            added_tokens=added_tokens,
            proposed_prompt_tokens=proposed_tokens,
        )
        decision = _CandidateDecision(
            candidate=candidate,
            post_dedup_disposition=disposition,
            added_token_proxy=added_tokens,
            proposed_prompt_token_proxy=proposed_tokens,
            messages=messages,
            candidate_receipt_sha256=identity_sha256(body),
        )
        decisions.append(decision)
        if decision.admitted:
            admitted = decision
    if admitted is not None:
        outcome = "appended"
    elif not decisions:
        outcome = "no_robust_candidate"
    elif all(
        row.post_dedup_disposition.startswith("excluded_post_selection")
        for row in decisions
    ):
        outcome = "no_novel_candidate_after_post_selection_dedup"
    else:
        outcome = "no_budget_admissible_candidate"
    return tuple(decisions), admitted, outcome


def _raw_s0_coordinates(
    raw_question: Mapping[str, Any],
    *,
    expected_ordinal: int,
    expected_question_id: str,
    expected_evidence_ids: tuple[str, ...],
) -> tuple[str, ...]:
    stages = raw_question.get("stages")
    retrieval_receipt = raw_question.get("retrieval_receipt")
    if (
        raw_question.get("ordinal") != expected_ordinal
        or raw_question.get("question_id") != expected_question_id
        or not isinstance(stages, list)
        or not stages
        or not isinstance(stages[0], Mapping)
        or stages[0].get("stage_id") != "causal_graph_coverage_predecessor"
        or not isinstance(retrieval_receipt, Mapping)
    ):
        raise ValueError("locked shard S0 question coordinates changed")
    evidence = stages[0].get("evidence")
    chunk_ids = retrieval_receipt.get("protected_chunk_ids")
    evidence_ids = retrieval_receipt.get("protected_evidence_ids")
    if (
        not isinstance(evidence, list)
        or not isinstance(chunk_ids, list)
        or not isinstance(evidence_ids, list)
        or tuple(evidence_ids) != expected_evidence_ids
        or tuple(
            row.get("evidence_id") if isinstance(row, Mapping) else None
            for row in evidence
        )
        != expected_evidence_ids
        or len(chunk_ids) != len(expected_evidence_ids)
        or len(set(chunk_ids)) != len(chunk_ids)
    ):
        raise ValueError("locked shard S0 evidence/chunk map changed")
    return tuple(str(row) for row in chunk_ids)


def _proposal_receipt_body(
    *,
    ordinal: int,
    question_id: str,
    shard_id: str,
    s0_chunk_ids: tuple[str, ...],
    seed_chunk_ids: tuple[str, ...],
    history_receipt_sha256: str,
    derived_store_receipt_sha256: str,
    association_artifact_id: str,
    decisions: tuple[_CandidateDecision, ...],
    outcome: str,
    base_prompt_tokens: int,
) -> dict[str, Any]:
    return {
        "format": "memory-condense-locked-s0-hebbian-question-proposal-v1",
        "ordinal": ordinal,
        "question_id": question_id,
        "shard_id": shard_id,
        "s0_chunk_ids_sha256": identity_sha256(list(s0_chunk_ids)),
        "s0_chunk_count": len(s0_chunk_ids),
        "seed_chunk_ids": list(seed_chunk_ids),
        "history_receipt_sha256": history_receipt_sha256,
        "derived_store_receipt_sha256": derived_store_receipt_sha256,
        "association_artifact_id": association_artifact_id,
        "candidate_receipt_sha256s": [
            row.candidate_receipt_sha256 for row in decisions
        ],
        "candidate_count_before_post_selection_dedup": len(decisions),
        "admitted_source_chunk_ids": [
            row.candidate.source_chunk_id for row in decisions if row.admitted
        ],
        "outcome": outcome,
        "base_prompt_token_proxy": base_prompt_tokens,
    }


def _proposal_question_artifact(
    question: _QuestionProposal,
    *,
    s0_row: Any,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for decision in question.decisions:
        body = _candidate_body(
            decision.candidate,
            disposition=decision.post_dedup_disposition,
            added_tokens=decision.added_token_proxy,
            proposed_prompt_tokens=decision.proposed_prompt_token_proxy,
        )
        body["candidate_receipt_sha256"] = decision.candidate_receipt_sha256
        candidates.append(body)
    admitted = question.admitted
    return {
        "ordinal": question.ordinal,
        "question_id": question.question_id,
        "question_sha256": s0_row.question_sha256,
        "dated_question_sha256": s0_row.dated_question_sha256,
        "retrieval_question_part_sha256": s0_row.retrieval_question_part_sha256,
        "shard_id": question.shard_id,
        "s0_source_binding_sha256": s0_row.binding_sha256,
        "s0_stage_receipt_sha256": s0_row.stage_receipt_sha256,
        "s0_evidence_projection_sha256": s0_row.evidence_projection_sha256,
        "s0_provider_messages_sha256": s0_row.provider_messages_sha256,
        "s0_chunk_ids": list(question.s0_chunk_ids),
        "seed_chunk_ids": list(question.seed_chunk_ids),
        "seed_count": len(question.seed_chunk_ids),
        "history_receipt_sha256": question.history_receipt_sha256,
        "derived_store_receipt_sha256": question.derived_store_receipt_sha256,
        "association_artifact_id": question.association_artifact_id,
        "candidate_count_before_post_selection_dedup": len(question.decisions),
        "candidates": candidates,
        "admitted_source_chunk_id": (
            None if admitted is None else admitted.candidate.source_chunk_id
        ),
        "admitted_evidence_text_sha256": (
            None if admitted is None else admitted.candidate.evidence_text_sha256
        ),
        "added_token_proxy": (
            None if admitted is None else admitted.added_token_proxy
        ),
        "answer_prompt_messages_sha256": (
            None if admitted is None else identity_sha256(list(admitted.messages or ()))
        ),
        "answer_prompt_token_proxy": (
            None if admitted is None else admitted.proposed_prompt_token_proxy
        ),
        "outcome": question.outcome,
        "proposal_receipt_sha256": question.proposal_receipt_sha256,
    }


def _load_shard_raw_questions(shard: _ShardInput) -> tuple[Mapping[str, Any], ...]:
    payload, digest = _read(shard.retrieval_path)
    questions = payload.get("questions")
    if digest != shard.retrieval_sha256 or not isinstance(questions, list):
        raise ValueError(f"{shard.shard_id} raw retrieval changed")
    if len(questions) != len(shard.question_ids) or any(
        not isinstance(row, Mapping) for row in questions
    ):
        raise ValueError(f"{shard.shard_id} raw question population changed")
    return tuple(questions)


def _build_proposal_plan(args: argparse.Namespace) -> _ProposalPlan:
    source = _load_verified_history_preflight(args)
    s0_rows = source.s0_run.get("questions")
    if not isinstance(s0_rows, list) or len(s0_rows) != EXPECTED_QUESTION_COUNT:
        raise ValueError("sealed S0 run question population changed")
    questions: list[_QuestionProposal] = []
    history_bindings: list[dict[str, Any]] = []
    for shard in source.shards:
        history_root = Path(args.output_root) / "histories" / shard.shard_id
        history_path = history_root / history_runner.HISTORY_FILE_NAME
        history_sha = _sidecar_digest(history_path)
        history_source = load_fast_hebbian_h2_history(
            history_path,
            expected_sha256=history_sha,
        )
        history_runner._validate_reused_history_policy(
            _history_args(args, shard),
            history_source.artifact,
        )
        database_path, derived = _verified_derived_inputs(
            history_root / "derived-store",
            artifact=shard.artifact,
            history=history_source.artifact,
        )
        if (
            history_source.artifact.receipt.implementation_sha256
            != source.preflight["source_package_implementation_sha256"]
            or history_source.artifact.receipt.environment_lock_sha256
            != source.preflight["environment_lock_sha256"]
        ):
            raise ValueError("history producer differs from sealed preflight")
        raw_questions = _load_shard_raw_questions(shard)
        binding: dict[str, Any] = {
            "shard_id": shard.shard_id,
            "retrieval_sha256": shard.retrieval_sha256,
            "history_file_sha256": history_sha,
            "history_artifact_sha256": history_source.artifact.artifact_sha256,
            "history_receipt_sha256": (
                history_source.artifact.receipt.receipt_sha256
            ),
            "history_event_population_sha256": (
                history_source.artifact.receipt.event_population_sha256
            ),
            "derived_store_receipt_sha256": derived.receipt_sha256,
            "derived_database_sha256": derived.derived_database_sha256,
            "association_artifact_id": derived.association_artifact_id,
            "association_artifact_sha256": derived.association_artifact_sha256,
        }
        binding["history_binding_sha256"] = identity_sha256(binding)
        history_bindings.append(binding)
        with Database(database_path, read_only=True) as database:
            if database.current_turn() != shard.artifact.turn_count:
                raise ValueError("derived store terminal turn changed")
            associations = AssociationStore(database)
            if associations.get_artifact(derived.association_artifact_id) is None:
                raise ValueError("derived association artifact is absent")
            for local, raw_question in enumerate(raw_questions):
                ordinal = shard.offset + local
                s0_plan_row = source.s0_plan.rows[ordinal]
                locked_question = source.s0_plan.population.rows[ordinal].question
                s0_stage = locked_question.stages[0]
                if (
                    s0_plan_row.ordinal != ordinal
                    or s0_plan_row.question_id != shard.question_ids[local]
                    or s0_rows[ordinal].get("source_binding_sha256")
                    != s0_plan_row.binding_sha256
                ):
                    raise ValueError("S0/shard question binding changed")
                s0_chunk_ids = _raw_s0_coordinates(
                    raw_question,
                    expected_ordinal=ordinal,
                    expected_question_id=s0_plan_row.question_id,
                    expected_evidence_ids=s0_stage.evidence_ids,
                )
                seed_chunk_ids = s0_chunk_ids[:MAX_SEED_CHUNKS]
                if not seed_chunk_ids:
                    raise ValueError("S0 cannot seed an empty Hebbian query")
                s0_projections: set[tuple[str, str]] = set()
                for chunk_id in s0_chunk_ids:
                    hydrated = hydrate_chunk_result(database, chunk_id, score=0.0)
                    if hydrated is None:
                        raise ValueError(f"S0 chunk is absent: {chunk_id}")
                    s0_projections.add(
                        (_source_id(hydrated), quote_sha256(hydrated.chunk.text))
                    )
                activations = {
                    chunk_id: rank_discount(rank)
                    for rank, chunk_id in enumerate(seed_chunk_ids, start=1)
                }
                neighbors = associations.hebbian_neighbors(
                    activations,
                    derived.association_artifact_id,
                    top_k=MAX_CANDIDATES,
                    exclude=(),
                    now_turn=shard.artifact.turn_count,
                    half_life_turns=HALF_LIFE_TURNS,
                    min_score=MIN_SCORE,
                )
                robust = tuple(
                    row
                    for row in neighbors
                    if row.support >= MIN_SUPPORT
                    and row.coaccess_count >= MIN_COACCESS_COUNT
                )
                expected_order = tuple(
                    sorted(
                        robust,
                        key=lambda row: (-row.score, -row.support, row.chunk_id),
                    )
                )
                if robust != expected_order:
                    raise ValueError("Hebbian graph ranking changed")
                candidates: list[_GraphCandidate] = []
                for rank, neighbor in enumerate(robust, start=1):
                    hydrated = hydrate_chunk_result(
                        database,
                        neighbor.chunk_id,
                        score=neighbor.score,
                    )
                    if hydrated is None:
                        raise ValueError(
                            f"robust candidate is absent: {neighbor.chunk_id}"
                        )
                    candidates.append(
                        _GraphCandidate(
                            rank=rank,
                            source_chunk_id=neighbor.chunk_id,
                            source_id=_source_id(hydrated),
                            raw_text=hydrated.chunk.text,
                            rendered_text=_render_candidate(hydrated),
                            evidence_text_sha256=quote_sha256(hydrated.chunk.text),
                            score=neighbor.score,
                            support=neighbor.support,
                            anchor_chunk_id=neighbor.anchor_chunk_id,
                            coaccess_count=neighbor.coaccess_count,
                            last_reinforced_turn=neighbor.last_reinforced_turn,
                        )
                    )
                decisions, admitted, outcome = _decide_candidates(
                    s0_plan_row.messages,
                    candidates,
                    all_s0_chunk_ids=s0_chunk_ids,
                    s0_exact_projections=s0_projections,
                )
                receipt_body = _proposal_receipt_body(
                    ordinal=ordinal,
                    question_id=s0_plan_row.question_id,
                    shard_id=shard.shard_id,
                    s0_chunk_ids=s0_chunk_ids,
                    seed_chunk_ids=seed_chunk_ids,
                    history_receipt_sha256=binding["history_receipt_sha256"],
                    derived_store_receipt_sha256=derived.receipt_sha256,
                    association_artifact_id=derived.association_artifact_id,
                    decisions=decisions,
                    outcome=outcome,
                    base_prompt_tokens=s0_plan_row.prompt_token_proxy,
                )
                questions.append(
                    _QuestionProposal(
                        ordinal=ordinal,
                        question_id=s0_plan_row.question_id,
                        shard_id=shard.shard_id,
                        s0_chunk_ids=s0_chunk_ids,
                        seed_chunk_ids=seed_chunk_ids,
                        history_receipt_sha256=binding[
                            "history_receipt_sha256"
                        ],
                        derived_store_receipt_sha256=derived.receipt_sha256,
                        association_artifact_id=derived.association_artifact_id,
                        decisions=decisions,
                        admitted=admitted,
                        outcome=outcome,
                        base_prompt_token_proxy=s0_plan_row.prompt_token_proxy,
                        proposal_receipt_sha256=identity_sha256(receipt_body),
                    )
                )
    if tuple(row.ordinal for row in questions) != tuple(
        range(EXPECTED_QUESTION_COUNT)
    ):
        raise ValueError("Hebbian proposal question order changed")
    question_artifacts = [
        _proposal_question_artifact(row, s0_row=source.s0_plan.rows[row.ordinal])
        for row in questions
    ]
    artifact: dict[str, Any] = {
        "format": PROPOSAL_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "history_preflight_sha256": source.preflight_sha256,
        "retrieval_sha256": source.s0_plan.population.retrieval_sha256,
        "population_identity_sha256": (
            source.s0_plan.population.population_identity_sha256
        ),
        "historical_validator_binding_sha256": (
            source.s0_plan.population.binding_sha256
        ),
        "s0_control_run_sha256": source.s0_run_sha256,
        "arm_policy": _arm_policy(),
        "arm_policy_sha256": identity_sha256(_arm_policy()),
        "history_bindings": history_bindings,
        "history_binding_population_sha256": identity_sha256(history_bindings),
        "question_count": len(questions),
        "admitted_question_count": sum(row.admitted is not None for row in questions),
        "outcome_counts": dict(sorted(Counter(row.outcome for row in questions).items())),
        "questions": question_artifacts,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    artifact["proposal_population_sha256"] = identity_sha256(artifact)
    if _contains_forbidden_key(artifact):
        raise RuntimeError("Hebbian proposals crossed the gold firewall")
    valid_ordinals = tuple(
        row.ordinal for row in questions if row.admitted is not None
    )
    prompts = tuple(
        row.messages for row in questions if row.messages is not None
    )
    answer_preflight = None
    if prompts:
        answer_preflight = preflight_fast_completion_prompts(
            prompts,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )
        if (
            answer_preflight.logical_prompt_count != len(valid_ordinals)
            or answer_preflight.unique_prompt_count != len(valid_ordinals)
        ):
            raise ValueError("Hebbian answer prompts are not unique per admission")
    return _ProposalPlan(
        history_source=source,
        history_bindings=tuple(history_bindings),
        questions=tuple(questions),
        artifact=artifact,
        artifact_sha256=_digest(artifact),
        valid_ordinals=valid_ordinals,
        prompts=prompts,
        answer_preflight=answer_preflight,
    )


def _proposals_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "proposals.json"


def _load_verified_proposals(args: argparse.Namespace) -> _ProposalPlan:
    plan = _build_proposal_plan(args)
    source, digest = _read(_proposals_path(args))
    expected = _require_sha256(
        args.expected_proposals_sha256,
        "proposal expected SHA-256",
    )
    if (
        digest != expected
        or digest != plan.artifact_sha256
        or canonical_json_bytes(source) != canonical_json_bytes(plan.artifact)
    ):
        raise ValueError("Hebbian proposals differ from sealed histories")
    return plan


def run_proposal_preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    if (
        args.enable_history_build
        or args.authorized_history_shards
        or args.enable_provider
        or args.authorized_provider_calls
    ):
        raise ValueError("proposal preflight forbids model/provider authorization")
    plan = _build_proposal_plan(args)
    return dict(plan.artifact), _publish(_proposals_path(args), plan.artifact)


def run_answer_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.enable_history_build
        or args.authorized_history_shards
        or args.enable_provider
        or args.authorized_provider_calls
    ):
        raise ValueError("answer preflight forbids model/provider authorization")
    plan = _load_verified_proposals(args)
    answer_tokens = [
        row.admitted.proposed_prompt_token_proxy
        for row in plan.questions
        if row.admitted is not None
    ]
    added_tokens = [
        row.admitted.added_token_proxy
        for row in plan.questions
        if row.admitted is not None
    ]
    result: dict[str, Any] = {
        "format": ANSWER_PREFLIGHT_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "history_preflight_sha256": plan.history_source.preflight_sha256,
        "proposal_artifact_sha256": plan.artifact_sha256,
        "question_count": len(plan.questions),
        "admitted_question_count": len(plan.valid_ordinals),
        "s0_fallback_count": len(plan.questions) - len(plan.valid_ordinals),
        "outcome_counts": dict(
            sorted(Counter(row.outcome for row in plan.questions).items())
        ),
        "added_tokens": _distribution(added_tokens),
        "answer_prompt_tokens": _distribution(answer_tokens),
        "answer_prompt_population": (
            None
            if plan.answer_preflight is None
            else plan.answer_preflight.model_dump()
        ),
        "required_authorized_provider_calls": plan.unique_calls,
        "authorized_call_kind": "terra_s0_plus_hebbian_answer",
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_forbidden_key(result):
        raise RuntimeError("Hebbian answer preflight crossed the gold firewall")
    return result


def _evidence_targets(plan: _ProposalPlan, ordinal: int) -> list[dict[str, Any]]:
    question = plan.history_source.s0_plan.population.rows[ordinal].question
    stage = question.stages[0]
    return [
        {
            "target_id": row.evidence_id,
            "target_id_encoding": "raw_sealed_evidence_id",
            "target_kind": "evidence",
            "discovering_method": "causal_graph_coverage_predecessor",
            "source_target_id": row.source_id,
            "disposition": "protected_s0_unchanged",
            "route_local_receipt_sha256": stage.stage_receipt_sha256,
        }
        for row in stage.evidence
    ]


def _candidate_target(
    decision: _CandidateDecision,
    *,
    disposition: str,
) -> dict[str, Any]:
    candidate = decision.candidate
    return {
        "target_id": candidate.source_chunk_id,
        "target_id_encoding": "raw_sealed_chunk_id",
        "target_kind": "source_chunk",
        "discovering_method": "hebbian_coaccess",
        "source_target_id": candidate.source_id,
        "evidence_text_sha256": candidate.evidence_text_sha256,
        "disposition": disposition,
        "route_local_receipt_sha256": decision.candidate_receipt_sha256,
        "rank": candidate.rank,
        "support": candidate.support,
        "coaccess_count": candidate.coaccess_count,
        "anchor_source_chunk_id": candidate.anchor_chunk_id,
    }


def _structural_target_ledger(plan: _ProposalPlan) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for question in plan.questions:
        before = [
            _candidate_target(
                decision,
                disposition="candidate_before_post_selection_dedup",
            )
            for decision in question.decisions
        ]
        after = [
            _candidate_target(
                decision,
                disposition=decision.post_dedup_disposition,
            )
            for decision in question.decisions
        ]
        admitted = [
            _candidate_target(
                decision,
                disposition="admitted_after_budget",
            )
            for decision in question.decisions
            if decision.admitted
        ]
        body: dict[str, Any] = {
            "ordinal": question.ordinal,
            "question_id": question.question_id,
            "shard_id": question.shard_id,
            "history_receipt_sha256": question.history_receipt_sha256,
            "derived_store_receipt_sha256": (
                question.derived_store_receipt_sha256
            ),
            "proposal_receipt_sha256": question.proposal_receipt_sha256,
            "evidence_targets": _evidence_targets(plan, question.ordinal),
            "candidate_targets_before_post_selection_dedup": before,
            "candidate_targets_after_post_selection_dedup": after,
            "admitted_targets_after_budget": admitted,
            "candidate_target_ids_before_post_selection_dedup_sha256": (
                identity_sha256([row["target_id"] for row in before])
            ),
            "candidate_target_ids_after_post_selection_dedup_sha256": (
                identity_sha256(
                    [
                        row["target_id"]
                        for row in after
                        if not row["disposition"].startswith(
                            "excluded_post_selection"
                        )
                    ]
                )
            ),
            "admitted_target_ids_sha256": identity_sha256(
                [row["target_id"] for row in admitted]
            ),
            "candidate_target_count_before_post_selection_dedup": len(before),
            "candidate_target_count_after_post_selection_dedup": sum(
                not row["disposition"].startswith("excluded_post_selection")
                for row in after
            ),
            "admitted_target_count": len(admitted),
        }
        body["ledger_row_sha256"] = identity_sha256(body)
        rows.append(body)
    result: dict[str, Any] = {
        "format": TARGET_LEDGER_FORMAT,
        "arm_label": ARM_LABEL,
        "source_s0_run_sha256": plan.history_source.s0_run_sha256,
        "source_history_preflight_sha256": (
            plan.history_source.preflight_sha256
        ),
        "source_proposal_artifact_sha256": plan.artifact_sha256,
        "population_identity_sha256": (
            plan.history_source.s0_plan.population.population_identity_sha256
        ),
        "question_count": len(rows),
        "target_id_policy": {
            "evidence_targets": "raw_sealed_evidence_id",
            "hebbian_targets": "raw_sealed_chunk_id",
        },
        "discovery_projection": (
            "candidate_targets_before_post_selection_dedup"
        ),
        "post_dedup_projection": (
            "candidate_targets_after_post_selection_dedup"
        ),
        "admission_projection": "admitted_targets_after_budget",
        "questions": rows,
    }
    result["ledger_sha256"] = identity_sha256(result)
    if _contains_forbidden_key(result):
        raise RuntimeError("Hebbian structural ledger crossed the gold firewall")
    return result


def _target_ledger_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "structural-target-ledger.json"


def run_target_ledger(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if (
        args.enable_history_build
        or args.authorized_history_shards
        or args.enable_provider
        or args.authorized_provider_calls
    ):
        raise ValueError("target ledger forbids model/provider authorization")
    plan = _load_verified_proposals(args)
    ledger = _structural_target_ledger(plan)
    return ledger, _publish(_target_ledger_path(args), ledger)


def _runtime(
    plan: _ProposalPlan,
    args: argparse.Namespace,
    *,
    client: Any | None,
) -> FastCompletionRuntime | None:
    if not plan.prompts:
        return None
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / "terra-answer-calls",
        prompt_population=plan.prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=MAX_ANSWER_OUTPUT_TOKENS,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance={
            "experiment_format": RUN_FORMAT,
            "arm_label": ARM_LABEL,
            "parent_arm_label": PARENT_ARM_LABEL,
            "parent_run_sha256": plan.history_source.s0_run_sha256,
            "history_preflight_sha256": plan.history_source.preflight_sha256,
            "proposal_artifact_sha256": plan.artifact_sha256,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
        },
    )


def _answer_batch(
    plan: _ProposalPlan,
    args: argparse.Namespace,
    *,
    client: Any | None,
) -> FastCompletionBatch | None:
    runtime = _runtime(plan, args, client=client)
    return None if runtime is None else runtime.run()


def _arm_identity(plan: _ProposalPlan) -> dict[str, Any]:
    result = {
        "format": "memory-condense-locked-retrieval-mechanism-arm-identity-v1",
        "arm_label": ARM_LABEL,
        "parent_arm": PARENT_ARM_LABEL,
        "parent_run_sha256": plan.history_source.s0_run_sha256,
        "mechanism": "s0-seeded-robust-causal-hebbian-coaccess-v1",
        "retrieval_sha256": (
            plan.history_source.s0_plan.population.retrieval_sha256
        ),
        "population_identity_sha256": (
            plan.history_source.s0_plan.population.population_identity_sha256
        ),
        "historical_validator_binding_sha256": (
            plan.history_source.s0_plan.population.binding_sha256
        ),
        "history_preflight_sha256": plan.history_source.preflight_sha256,
        "proposal_artifact_sha256": plan.artifact_sha256,
        "arm_policy_sha256": identity_sha256(_arm_policy()),
        "question_proposal_receipt_sha256s": [
            row.proposal_receipt_sha256 for row in plan.questions
        ],
        "question_count": len(plan.questions),
    }
    return result


def _run_artifact(
    plan: _ProposalPlan,
    batch: FastCompletionBatch | None,
) -> dict[str, Any]:
    completions: dict[int, str] = {}
    records: dict[str, Mapping[str, Any]] = {}
    if batch is not None:
        completions = dict(
            zip(plan.valid_ordinals, batch.logical_completions, strict=True)
        )
        records = _record_by_messages(batch)
    ledger = _structural_target_ledger(plan)
    ledger_rows = ledger["questions"]
    s0_rows = plan.history_source.s0_run["questions"]
    questions: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for proposal, s0, ledger_row in zip(
        plan.questions,
        s0_rows,
        ledger_rows,
        strict=True,
    ):
        s0_prediction = s0["prediction"]
        prediction = str(s0_prediction["text"])
        prediction_kind = "sealed_s0_control_fallback"
        fallback_reason: str | None = proposal.outcome
        prompt_sha: str | None = None
        call_key: str | None = None
        request_journal: str | None = None
        response_journal: str | None = None
        if proposal.admitted is not None:
            messages = proposal.admitted.messages
            if messages is None:
                raise RuntimeError("admitted Hebbian row omitted its prompt")
            prompt_sha = identity_sha256(list(messages))
            record = records[prompt_sha]
            call_key = record["call_key_sha256"]
            request_journal = record["request_journal_sha256"]
            response_journal = record["response_journal_sha256"]
            candidate_answer = completions[proposal.ordinal].strip()
            if candidate_answer:
                prediction = candidate_answer
                prediction_kind = "terra_s0_plus_hebbian"
                fallback_reason = None
            else:
                fallback_reason = "empty_answer_response"
        questions.append(
            {
                "ordinal": proposal.ordinal,
                "question_id": proposal.question_id,
                "question_sha256": s0["question_sha256"],
                "dated_question_sha256": s0["dated_question_sha256"],
                "retrieval_question_part_sha256": s0[
                    "retrieval_question_part_sha256"
                ],
                "arm_label": ARM_LABEL,
                "parent_arm_label": PARENT_ARM_LABEL,
                "parent_prediction_sha256": s0_prediction["sha256"],
                "s0_source_binding_sha256": s0["source_binding_sha256"],
                "s0_stage_receipt_sha256": s0["stage_receipt_sha256"],
                "s0_evidence_projection_sha256": s0[
                    "evidence_projection_sha256"
                ],
                "s0_provider_messages_sha256": s0[
                    "provider_messages_sha256"
                ],
                "shard_id": proposal.shard_id,
                "history_receipt_sha256": proposal.history_receipt_sha256,
                "derived_store_receipt_sha256": (
                    proposal.derived_store_receipt_sha256
                ),
                "proposal_receipt_sha256": proposal.proposal_receipt_sha256,
                "seed_count": len(proposal.seed_chunk_ids),
                "candidate_count_before_post_selection_dedup": len(
                    proposal.decisions
                ),
                "admitted_source_chunk_id": (
                    None
                    if proposal.admitted is None
                    else proposal.admitted.candidate.source_chunk_id
                ),
                "added_token_proxy": (
                    None
                    if proposal.admitted is None
                    else proposal.admitted.added_token_proxy
                ),
                "answer_prompt_token_proxy": (
                    None
                    if proposal.admitted is None
                    else proposal.admitted.proposed_prompt_token_proxy
                ),
                "outcome": proposal.outcome,
                "prediction_kind": prediction_kind,
                "s0_fallback_reason": fallback_reason,
                "prediction": {
                    "text": prediction,
                    "sha256": quote_sha256(prediction),
                },
                "changed_from_s0": (
                    quote_sha256(prediction) != s0_prediction["sha256"]
                ),
                "answer_prompt_messages_sha256": prompt_sha,
                "answer_call_key_sha256": call_key,
                "answer_request_journal_sha256": request_journal,
                "answer_response_journal_sha256": response_journal,
                "structural_target_ledger_row_sha256": ledger_row[
                    "ledger_row_sha256"
                ],
            }
        )
        budget_rows.append(
            {
                "ordinal": proposal.ordinal,
                "s0_seed_count": len(proposal.seed_chunk_ids),
                "s0_seed_cap": MAX_SEED_CHUNKS,
                "candidate_count": len(proposal.decisions),
                "candidate_cap": MAX_CANDIDATES,
                "minimum_support": MIN_SUPPORT,
                "minimum_coaccess_count": MIN_COACCESS_COUNT,
                "appended_rows": int(proposal.admitted is not None),
                "append_cap": MAX_ADDITIONS,
                "added_token_proxy": (
                    None
                    if proposal.admitted is None
                    else proposal.admitted.added_token_proxy
                ),
                "added_token_cap": MAX_ADDED_TOKENS,
                "answer_prompt_token_proxy": (
                    None
                    if proposal.admitted is None
                    else proposal.admitted.proposed_prompt_token_proxy
                ),
                "answer_prompt_token_cap": MAX_PROMPT_TOKENS,
                "answer_output_token_cap": MAX_ANSWER_OUTPUT_TOKENS,
                "s0_fallback": prediction_kind
                == "sealed_s0_control_fallback",
                "s0_fallback_reason": fallback_reason,
            }
        )
    identity = _arm_identity(plan)
    artifact: dict[str, Any] = {
        "format": RUN_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "parent_run_sha256": plan.history_source.s0_run_sha256,
        "s0_control_run_sha256": plan.history_source.s0_run_sha256,
        "arm_identity": identity,
        "arm_identity_sha256": identity_sha256(identity),
        "retrieval_sha256": (
            plan.history_source.s0_plan.population.retrieval_sha256
        ),
        "baseline_final_answers_sha256": (
            plan.history_source.s0_plan.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": (
            plan.history_source.s0_plan.population.population_identity_sha256
        ),
        "historical_validator_binding_sha256": (
            plan.history_source.s0_plan.population.binding_sha256
        ),
        "history_preflight_sha256": plan.history_source.preflight_sha256,
        "proposal_artifact_sha256": plan.artifact_sha256,
        "question_count": len(questions),
        "required_answer_calls": plan.unique_calls,
        "settings": {
            "model": DEFAULT_MODEL,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_new_tokens": MAX_ANSWER_OUTPUT_TOKENS,
            "retries": 0,
        },
        "answer_completion_batch": (
            None if batch is None else _stable_batch(batch)
        ),
        "budget": {
            "s0_non_borrowable": True,
            "exact_s0_membership_and_order": True,
            "common_answer_operator": True,
            "shared_residual_borrowing": False,
            "questions": budget_rows,
        },
        "structural_target_ledger": ledger,
        "questions": questions,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_forbidden_key(artifact):
        raise RuntimeError("Hebbian run crossed the gold firewall")
    return artifact


def _run_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "run.json"


def run_treatment(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_history_build or args.authorized_history_shards:
        raise ValueError("answer run forbids history construction")
    plan = _load_verified_proposals(args)
    if not args.enable_provider:
        raise ValueError("run requires --enable-provider")
    if args.authorized_provider_calls != plan.unique_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the dependent "
            f"answer population ({args.authorized_provider_calls} != "
            f"{plan.unique_calls})"
        )
    path = _run_path(args)
    if path.exists() or path.is_symlink():
        raise FileExistsError("sealed Hebbian run already exists; use replay")
    client = None
    if plan.unique_calls:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        if not api_key:
            raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
    try:
        batch = _answer_batch(plan, args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if batch is not None and (
        batch.prompt_population.unique_prompt_count != plan.unique_calls
        or batch.usage.physical_calls + batch.usage.checkpoint_hits
        != plan.unique_calls
    ):
        raise RuntimeError("Hebbian answer journal population changed")
    artifact = _run_artifact(plan, batch)
    return artifact, _publish(path, artifact)


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if (
        args.enable_history_build
        or args.authorized_history_shards
        or args.enable_provider
        or args.authorized_provider_calls
    ):
        raise ValueError("run replay forbids model/provider authorization")
    plan = _load_verified_proposals(args)
    source, source_sha = _read(
        _run_path(args),
        expected_sha256=_require_sha256(
            args.expected_run_sha256,
            "run expected SHA-256",
        ),
    )
    batch = _answer_batch(plan, args, client=None)
    if batch is not None and (
        batch.usage.physical_calls
        or batch.usage.checkpoint_hits != plan.unique_calls
    ):
        raise RuntimeError("Hebbian replay did not consume exact journals")
    expected = _run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("Hebbian run differs from immutable answer journals")
    replay_sha = _publish(Path(args.output_root) / "run-replay.json", source)
    if replay_sha != source_sha:
        raise RuntimeError("Hebbian replay publication changed source digest")
    return source, source_sha


def load_verified_run(
    run_path: str | Path,
    *,
    expected_run_sha256: str,
    retrieval_path: str | Path = DEFAULT_RETRIEVAL,
    baseline_answers_path: str | Path = DEFAULT_BASELINE_ANSWERS,
    checkpoint_dir: str | Path | None = None,
    max_concurrency: int = 4,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    expected_retrieval_sha256: str = EXPECTED_RETRIEVAL_SHA256,
    expected_baseline_answers_sha256: str = (
        EXPECTED_BASELINE_ANSWERS_SHA256
    ),
) -> tuple[dict[str, Any], str]:
    """Strict zero-call loader for the common retrieval-arm judge."""

    target = Path(run_path)
    output_root = target.parent
    if target.resolve() != (output_root / "run.json").resolve():
        raise ValueError("Hebbian loader requires canonical run.json")
    source, source_sha = _read(target, expected_sha256=expected_run_sha256)
    replay, replay_sha = _read(
        output_root / "run-replay.json",
        expected_sha256=expected_run_sha256,
    )
    if (
        replay_sha != source_sha
        or canonical_json_bytes(replay) != canonical_json_bytes(source)
    ):
        raise ValueError("Hebbian answer run/replay differ")
    expected_checkpoint = output_root / "terra-answer-calls"
    if (
        checkpoint_dir is not None
        and Path(checkpoint_dir).resolve() != expected_checkpoint.resolve()
    ):
        raise ValueError("Hebbian loader received another checkpoint directory")
    args = build_parser().parse_args(
        [
            "--phase",
            "replay",
            "--retrieval",
            str(retrieval_path),
            "--expected-retrieval-sha256",
            expected_retrieval_sha256,
            "--baseline-answers",
            str(baseline_answers_path),
            "--expected-baseline-answers-sha256",
            expected_baseline_answers_sha256,
            "--s0-run",
            str(output_root.parent / "s0-control-v1" / "run.json"),
            "--expected-s0-run-sha256",
            str(source["parent_run_sha256"]),
            "--shards-root",
            str(DEFAULT_SHARDS_ROOT),
            "--output-root",
            str(output_root),
            "--expected-history-preflight-sha256",
            str(source["history_preflight_sha256"]),
            "--expected-proposals-sha256",
            str(source["proposal_artifact_sha256"]),
            "--expected-run-sha256",
            expected_run_sha256,
            "--expected-question-count",
            str(expected_question_count),
            "--max-concurrency",
            str(max_concurrency),
        ]
    )
    plan = _load_verified_proposals(args)
    batch = _answer_batch(plan, args, client=None)
    if batch is not None and (
        batch.usage.physical_calls
        or batch.usage.checkpoint_hits != plan.unique_calls
    ):
        raise RuntimeError("Hebbian loader did not consume exact journals")
    expected = _run_artifact(plan, batch)
    if canonical_json_bytes(expected) != canonical_json_bytes(source):
        raise ValueError("Hebbian loader reconstructed another run")
    return source, source_sha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "history-preflight",
            "history-build",
            "proposal-preflight",
            "answer-preflight",
            "target-ledger",
            "run",
            "replay",
        ),
        default="history-preflight",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--baseline-answers",
        type=Path,
        default=DEFAULT_BASELINE_ANSWERS,
    )
    parser.add_argument(
        "--expected-baseline-answers-sha256",
        default=EXPECTED_BASELINE_ANSWERS_SHA256,
    )
    parser.add_argument("--s0-run", type=Path, default=DEFAULT_S0_RUN)
    parser.add_argument("--s0-checkpoint-dir", type=Path)
    parser.add_argument("--expected-s0-run-sha256", required=True)
    parser.add_argument("--shards-root", type=Path, default=DEFAULT_SHARDS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-history-preflight-sha256")
    parser.add_argument("--expected-proposals-sha256")
    parser.add_argument("--expected-run-sha256")
    parser.add_argument("--shard-id")
    parser.add_argument(
        "--expected-question-count",
        type=int,
        default=EXPECTED_QUESTION_COUNT,
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-history-build", action="store_true")
    parser.add_argument("--authorized-history-shards", type=int, default=0)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "history-preflight":
        artifact, digest = run_history_preflight(args)
        print(
            "S0_PLUS_HEBBIAN history preflight sealed: "
            f"sha256={digest}; shards={artifact['shard_count']}; "
            f"turns={artifact['total_turn_count']}; "
            f"eligible_queries={artifact['total_eligible_historical_query_count']}; "
            "history_model_loads=0; provider_calls=0",
            flush=True,
        )
        for row in artifact["shards"]:
            command = [
                digest if value == "<HISTORY_PREFLIGHT_SHA256>" else value
                for value in row["history_build_command_argv_template"]
            ]
            print("Exact history command: " + " ".join(command), flush=True)
        return 0
    if args.phase == "history-build":
        artifact, digest = run_history_build(args)
    elif args.phase == "proposal-preflight":
        artifact, digest = run_proposal_preflight(args)
    elif args.phase == "answer-preflight":
        artifact = run_answer_preflight(args)
        print(
            "S0_PLUS_HEBBIAN answer preflight: "
            f"admitted={artifact['admitted_question_count']}; "
            f"fallback={artifact['s0_fallback_count']}; "
            f"authorized_terra_calls={artifact['required_authorized_provider_calls']}; "
            "provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    elif args.phase == "target-ledger":
        artifact, digest = run_target_ledger(args)
    elif args.phase == "run":
        artifact, digest = run_treatment(args)
    else:
        artifact, digest = run_replay(args)
    print(
        f"{args.phase} verified {ARM_LABEL} artifact {digest}; "
        f"provider_calls={0 if args.phase != 'run' else artifact.get('required_answer_calls', 0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_PREFLIGHT_FORMAT",
    "ARM_LABEL",
    "HISTORY_PREFLIGHT_FORMAT",
    "PROPOSAL_FORMAT",
    "RUN_FORMAT",
    "TARGET_LEDGER_FORMAT",
    "build_parser",
    "load_verified_run",
    "main",
    "run_answer_preflight",
    "run_history_build",
    "run_history_preflight",
    "run_proposal_preflight",
    "run_replay",
    "run_target_ledger",
    "run_treatment",
]
