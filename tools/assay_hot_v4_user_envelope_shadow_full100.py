#!/usr/bin/env python3
"""Gold-blind user-led envelope shadow over the sealed online-graph v4 arm.

``construct`` and ``replay`` authenticate the frozen v4 construction/runtime/
replay trio and the frozen retrieval stores.  They never open benchmark labels
or any earlier score artifact.  ``evaluate`` is the sole post-hoc gold join.
The shadow is additive: every exact packed v4 parent occurrence is protected,
and bounded user-led envelope companions may use only otherwise spare prompt
capacity.
"""

from __future__ import annotations

import argparse
import copy
import gc
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.persistence.db import Database
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_retrieval_full100 as full100
from tools import assay_hot_retrieval_source_seed_hybrid_full100 as v3
from tools import assay_hot_v3_provider_free_witness as packet_tools
from tools import assay_hot_v3_typed_operator_full100 as typed
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.query_expansion import load_locked_query_expansion_context
from tools.matched_eval.query_guided_scan import (
    NamespacePartitionCache,
    cache_namespace_partitions,
)


FORMAT = "memory-condense-hot-v4-user-envelope-shadow-full100-construction-v1"
RUNTIME_FORMAT = "memory-condense-hot-v4-user-envelope-shadow-full100-runtime-v1"
REPLAY_FORMAT = "memory-condense-hot-v4-user-envelope-shadow-full100-replay-v1"
EVALUATION_FORMAT = "memory-condense-hot-v4-user-envelope-shadow-full100-evaluation-v1"
ROW_FORMAT = "memory-condense-hot-v4-user-envelope-shadow-full100-row-v1"
TIMING_FORMAT = "memory-condense-hot-v4-user-envelope-shadow-full100-timing-v1"
POLICY_ID = "sealed-online-graph-v4-plus-user-led-envelope-shadow-v1"

EXPECTED_V4_CONSTRUCTION_SHA256 = (
    "5cc97df65cd97972bc19b8b203ef0a3a8434871dc33f687687689801c5949515"
)
EXPECTED_V4_RUNTIME_SHA256 = (
    "cfede1df484b27f719cca3e0afe40a260078d7fb21c68073d2328c36394d61e4"
)
EXPECTED_V4_REPLAY_SHA256 = (
    "114e4bd3b77dc27b73501944307064b4d93369949d409407dbcc19b4b52f5719"
)
EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
EXPECTED_POPULATION_SHA256 = (
    "9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246"
)
EXPECTED_QUESTION_COUNT = 100

MAX_CONTEXT_TOKENS = 7_000
MAX_PROMPT_WORKSPACE_TOKENS = 8_000
OUTPUT_TOKEN_RESERVE = hot.RESPONDER_OUTPUT_TOKEN_RESERVE
MAX_ENVELOPES = 4
MAX_TURNS_PER_ENVELOPE = 8
MAX_COMPANION_CHUNKS = 16
MAX_COMPANION_TOKENS = 800

DEFAULT_V4_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v3-online-graph-full100-20260907"
)
DEFAULT_STORE_ROOT = v3.DEFAULT_SOURCE_ROOT
DEFAULT_RETRIEVAL = DEFAULT_STORE_ROOT / "retrieval.json"
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v4-user-envelope-shadow-full100-20260907"
)
DEFAULT_SPLIT = full100.DEFAULT_SPLIT

CONSTRUCTION_NAME = "construction.json"
RUNTIME_NAME = "runtime.json"
REPLAY_NAME = "replay.json"
EVALUATION_NAME = "evaluation.json"


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _elapsed(clock: Callable[[], int], started: int) -> int:
    value = clock() - started
    _require(type(value) is int and value >= 0, "runtime clock moved backwards")
    return value


def _percentile95(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)])


def _projection(value: object, *, label: str) -> dict[str, Any]:
    projector = getattr(value, "projection", None)
    _require(callable(projector), f"{label} omitted its projection")
    result = projector()
    _require(type(result) is dict, f"{label} projection changed type")
    return result


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/eval/_retrieval_qa_prompt.py",
        "tools/assay_hot_retrieval_1m.py",
        "tools/assay_hot_v4_user_envelope_shadow_full100.py",
        "tools/matched_eval/contracts.py",
        "tools/matched_eval/hot_v3_user_led_envelope_shadow.py",
        "tools/matched_eval/query_expansion.py",
        "tools/matched_eval/query_guided_scan.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "files": files,
        "format": "memory-condense-hot-v4-user-envelope-shadow-implementation-v1",
        "sha256": identity_sha256(
            [{"path": path, "sha256": files[path]} for path in paths]
        ),
    }


@dataclass(frozen=True, slots=True)
class ParentArtifacts:
    construction: Mapping[str, Any]
    construction_sha256: str
    runtime_sha256: str
    replay_sha256: str
    rows_by_ordinal: Mapping[int, Mapping[str, Any]]


def _load_parent(v4_root: Path) -> ParentArtifacts:
    """Authenticate only the v4 construction/runtime/replay boundary."""

    construction, construction_sha = hot._read_json_artifact(  # noqa: SLF001
        v4_root / "construction.json"
    )
    runtime, runtime_sha = hot._read_json_artifact(  # noqa: SLF001
        v4_root / "runtime.json"
    )
    replay, replay_sha = hot._read_json_artifact(  # noqa: SLF001
        v4_root / "replay.json"
    )
    _require(
        construction_sha == EXPECTED_V4_CONSTRUCTION_SHA256
        and runtime_sha == EXPECTED_V4_RUNTIME_SHA256
        and replay_sha == EXPECTED_V4_REPLAY_SHA256,
        "sealed online-graph v4 artifact digest changed",
    )
    _require(
        construction.get("format")
        == "memory-condense-hot-v3-ordered-story-construction-v4"
        and runtime.get("format")
        == "memory-condense-hot-v3-ordered-story-runtime-v4"
        and replay.get("format")
        == "memory-condense-hot-v3-ordered-story-replay-v4",
        "sealed online-graph v4 artifact format changed",
    )
    _require(
        runtime.get("construction_sha256") == construction_sha
        and replay.get("construction_sha256") == construction_sha
        and construction.get("question_count") == EXPECTED_QUESTION_COUNT
        and replay.get("question_count") == EXPECTED_QUESTION_COUNT
        and construction.get("gold_loaded") is False
        and runtime.get("gold_loaded") is False
        and replay.get("gold_loaded") is False
        and construction.get("model_calls") == 0
        and construction.get("new_provider_calls") == 0
        and runtime.get("model_calls") == 0
        and runtime.get("new_provider_calls") == 0
        and replay.get("model_calls") == 0
        and replay.get("new_provider_calls") == 0,
        "sealed online-graph v4 lifecycle binding changed",
    )
    rows = construction.get("questions")
    replay_rows = replay.get("questions")
    _require(
        type(rows) is list
        and type(replay_rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT
        and len(replay_rows) == EXPECTED_QUESTION_COUNT,
        "sealed online-graph v4 question inventory changed",
    )
    replay_by_ordinal = {int(row["ordinal"]): row for row in replay_rows}
    by_ordinal: dict[int, Mapping[str, Any]] = {}
    for expected_ordinal, row in enumerate(rows):
        _require(type(row) is dict, "sealed online-graph v4 row changed type")
        ordinal = row.get("ordinal")
        _require(
            ordinal == expected_ordinal and ordinal not in by_ordinal,
            "sealed online-graph v4 ordinal sequence changed",
        )
        unsigned = dict(row)
        receipt = unsigned.pop("row_receipt_sha256", None)
        _require(
            receipt == identity_sha256(unsigned),
            f"sealed online-graph v4 row receipt changed at {ordinal}",
        )
        replay_row = replay_by_ordinal.get(ordinal)
        _require(
            type(replay_row) is dict
            and replay_row.get("row_receipt_sha256") == receipt,
            f"sealed online-graph v4 replay row changed at {ordinal}",
        )
        arm = row.get("effective_arm")
        _require(
            type(arm) is dict
            and type(arm.get("packed_evidence")) is list
            and bool(arm["packed_evidence"]),
            f"sealed online-graph v4 effective arm changed at {ordinal}",
        )
        dated_question = typed._extract_dated_question(arm)  # noqa: SLF001
        _require(
            quote_sha256(dated_question) == row.get("prompt_question_sha256"),
            f"sealed online-graph v4 prompt binding changed at {ordinal}",
        )
        hot._validate_arm_payload(  # noqa: SLF001
            arm,
            prompt_question=dated_question,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_prompt_tokens=MAX_PROMPT_WORKSPACE_TOKENS,
        )
        by_ordinal[ordinal] = row
    assert_gold_blind(construction, path="sealed_online_graph_v4_parent")
    return ParentArtifacts(
        construction=construction,
        construction_sha256=construction_sha,
        runtime_sha256=runtime_sha,
        replay_sha256=replay_sha,
        rows_by_ordinal=by_ordinal,
    )


def _build_cache(
    context: object, namespace: object
) -> tuple[NamespacePartitionCache, dict[str, Any]]:
    namespace_id = str(getattr(namespace, "namespace_id"))
    store = getattr(context, "store_dirs_by_namespace")[namespace_id]
    database_sha = getattr(context, "database_sha256_by_namespace")[namespace_id]
    store_receipt = str(getattr(namespace, "combined_store_receipt_sha256"))
    started = time.perf_counter_ns()
    with Database(store / "memory.db", read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            namespace,
            source_database_sha256=database_sha,
            source_store_receipt_sha256=store_receipt,
        )
    return cache, {
        "cache_build_ns": time.perf_counter_ns() - started,
        "cache_receipt_sha256": cache.cache_receipt_sha256,
        "content_row_count": cache.content_row_count,
        "namespace_id": namespace_id,
        "physical_store_row_count": cache.physical_store_row_count,
    }


@dataclass(frozen=True, slots=True)
class RuntimeHooks:
    load_parent: Callable[[Path], ParentArtifacts]
    load_context: Callable[..., object]
    build_cache: Callable[[object, object], tuple[object, Mapping[str, Any]]]
    build_index: Callable[[object], object]
    make_budget: Callable[[], object]
    select: Callable[..., object]
    compose: Callable[..., object]


def _default_hooks() -> RuntimeHooks:
    from tools.matched_eval.hot_v3_user_led_envelope_shadow import (
        UserLedEnvelopeShadowBudget,
        build_user_led_envelope_shadow_index,
        compose_user_led_envelope_shadow,
        select_user_led_envelope_shadow,
    )

    return RuntimeHooks(
        load_parent=_load_parent,
        load_context=load_locked_query_expansion_context,
        build_cache=_build_cache,
        build_index=build_user_led_envelope_shadow_index,
        make_budget=lambda: UserLedEnvelopeShadowBudget(
            max_envelopes=MAX_ENVELOPES,
            max_turns_per_envelope=MAX_TURNS_PER_ENVELOPE,
            max_companion_chunks=MAX_COMPANION_CHUNKS,
            max_companion_tokens=MAX_COMPANION_TOKENS,
        ),
        select=select_user_led_envelope_shadow,
        compose=compose_user_led_envelope_shadow,
    )


def _packet_measure(dated_question: str) -> Callable[[Sequence[Mapping[str, Any]]], tuple[int, int]]:
    def measure(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
        rendered = [str(row["rendered_text"]) for row in rows]
        context_tokens = hot._context_token_proxy(rendered)  # noqa: SLF001
        messages = hot.build_qa_prompt(dated_question, rendered)
        workspace_tokens = (
            hot.count_chat_prompt_token_proxy(messages) + OUTPUT_TOKEN_RESERVE
        )
        return context_tokens, workspace_tokens

    return measure


def _as_evidence(value: object) -> tuple[dict[str, Any], ...]:
    _require(type(value) in {tuple, list}, "shadow packed evidence changed type")
    rows = tuple(copy.deepcopy(dict(row)) for row in value)  # type: ignore[arg-type]
    _require(rows and all(type(row) is dict for row in rows), "shadow packet became empty")
    return rows


def _protect_parent(
    parent: Sequence[Mapping[str, Any]], packed: Sequence[Mapping[str, Any]]
) -> None:
    available: dict[str, int] = {}
    for row in packed:
        digest = identity_sha256(dict(row))
        available[digest] = available.get(digest, 0) + 1
    for row in parent:
        digest = identity_sha256(dict(row))
        _require(available.get(digest, 0) > 0, "shadow dropped or mutated a v4 parent row")
        available[digest] -= 1


def _effective_arm(
    evidence: Sequence[Mapping[str, Any]], *, dated_question: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    copied = [copy.deepcopy(dict(row)) for row in evidence]
    arm, audit = packet_tools._pack_ranked_raw_evidence(  # noqa: SLF001
        copied,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_WORKSPACE_TOKENS,
    )
    _require(
        arm["packed_evidence"] == copied
        and arm["selected_evidence"] == copied
        and arm["dropped_chunk_ids"] == []
        and arm["context_token_proxy"] <= MAX_CONTEXT_TOKENS
        and arm["prompt_workspace_token_proxy"] <= MAX_PROMPT_WORKSPACE_TOKENS,
        "shadow composition escaped the strict final prompt cap",
    )
    hot._validate_arm_payload(  # noqa: SLF001
        arm,
        prompt_question=dated_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_WORKSPACE_TOKENS,
    )
    return arm, audit


def _id_tuple(value: object, *names: str) -> tuple[str, ...]:
    for name in names:
        candidate = getattr(value, name, None)
        if type(candidate) in {tuple, list}:
            result = tuple(str(item) for item in candidate)
            _require(all(result), f"{name} contains an empty identity")
            return result
    raise ValueError(f"shadow helper omitted identity field {names[0]}")


def _question_row(
    *,
    parent_row: Mapping[str, Any],
    namespace_id: str,
    cache: object,
    index: object,
    budget: object,
    hooks: RuntimeHooks,
    clock: Callable[[], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = clock()
    ordinal = int(parent_row["ordinal"])
    parent_arm = parent_row["effective_arm"]
    dated_question = typed._extract_dated_question(parent_arm)  # noqa: SLF001
    # This is the only parent evidence handed to the shadow selector.  The
    # broader v4 candidate set remains deliberately invisible.
    parent = tuple(copy.deepcopy(parent_arm["packed_evidence"]))

    step = clock()
    selection = hooks.select(index, parent, budget=budget)
    selection_ns = _elapsed(clock, step)
    step = clock()
    composition = hooks.compose(
        index,
        selection,
        parent,
        measure_packet=_packet_measure(dated_question),
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_WORKSPACE_TOKENS,
    )
    composition_ns = _elapsed(clock, step)
    packed = _as_evidence(getattr(composition, "packed_evidence", None))
    _protect_parent(parent, packed)
    step = clock()
    arm, packing_audit = _effective_arm(packed, dated_question=dated_question)
    packet_ns = _elapsed(clock, step)

    selection_projection = _projection(selection, label="shadow selection")
    composition_projection = _projection(composition, label="shadow composition")
    assert_gold_blind(selection_projection, path=f"shadow_selection_{ordinal}")
    assert_gold_blind(composition_projection, path=f"shadow_composition_{ordinal}")
    selected_ids = _id_tuple(
        composition,
        "selected_companion_chunk_ids",
    )
    admitted_ids = _id_tuple(
        composition,
        "admitted_companion_chunk_ids",
        "admitted_chunk_ids",
    )
    _require(
        int(getattr(composition, "selected_companion_token_count"))
        <= MAX_COMPANION_TOKENS
        and len(selected_ids) <= MAX_COMPANION_CHUNKS
        and int(getattr(composition, "admitted_companion_token_count"))
        <= int(getattr(composition, "selected_companion_token_count"))
        and len(admitted_ids) <= len(selected_ids)
        and set(admitted_ids).issubset(selected_ids)
        and
        int(getattr(composition, "context_token_count"))
        == arm["context_token_proxy"]
        and int(getattr(composition, "prompt_workspace_token_count"))
        == arm["prompt_workspace_token_proxy"],
        "shadow helper and final packet accounting diverged",
    )
    body = {
        "admitted_companion_chunk_ids": list(admitted_ids),
        "admitted_companion_count": len(admitted_ids),
        "cache_receipt_sha256": str(getattr(cache, "cache_receipt_sha256")),
        "composition": composition_projection,
        "composition_receipt_sha256": str(getattr(composition, "receipt_sha256")),
        "effective_arm": arm,
        "format": ROW_FORMAT,
        "gold_loaded": False,
        "index_receipt_sha256": str(getattr(index, "receipt_sha256")),
        "model_calls": 0,
        "namespace_id": namespace_id,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "packing_audit_sha256": identity_sha256(packing_audit),
        "parent_packed_chunk_ids": list(parent_arm["packed_chunk_ids"]),
        "parent_row_receipt_sha256": parent_row["row_receipt_sha256"],
        "parent_rows_protected": True,
        "policy_id": POLICY_ID,
        "prompt_question_sha256": parent_row["prompt_question_sha256"],
        "question_id": parent_row["question_id"],
        "selected_companion_chunk_ids": list(selected_ids),
        "selected_companion_count": len(selected_ids),
        "selection": selection_projection,
        "selection_receipt_sha256": str(getattr(selection, "receipt_sha256")),
    }
    assert_gold_blind(body, path=f"user_envelope_shadow_{ordinal}")
    row = {**body, "row_receipt_sha256": identity_sha256(body)}
    timing = {
        "composition_ns": composition_ns,
        "format": TIMING_FORMAT,
        "ordinal": ordinal,
        "packet_materialization_ns": packet_ns,
        "question_id": parent_row["question_id"],
        "result_row_receipt_sha256": row["row_receipt_sha256"],
        "selection_ns": selection_ns,
        "total_ns": _elapsed(clock, started),
    }
    return row, timing


def _semantic_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostic_reasons = Counter(
        str(diagnostic["reason"])
        for row in rows
        for diagnostic in row["selection"]["diagnostics"]
    )
    return {
        "admitted_companion_count": sum(
            int(row["admitted_companion_count"]) for row in rows
        ),
        "all_parent_rows_protected": all(
            row["parent_rows_protected"] is True for row in rows
        ),
        "max_context_token_proxy": max(
            int(row["effective_arm"]["context_token_proxy"]) for row in rows
        ),
        "max_prompt_workspace_token_proxy": max(
            int(row["effective_arm"]["prompt_workspace_token_proxy"])
            for row in rows
        ),
        "question_count": len(rows),
        "questions_with_admitted_companions": sum(
            int(row["admitted_companion_count"]) > 0 for row in rows
        ),
        "questions_with_selected_companions": sum(
            int(row["selected_companion_count"]) > 0 for row in rows
        ),
        "selected_companion_count": sum(
            int(row["selected_companion_count"]) for row in rows
        ),
        "selection_diagnostic_reason_counts": dict(sorted(diagnostic_reasons.items())),
        "strict_parent_anchor_rejection_count": sum(
            diagnostic_reasons[reason]
            for reason in ("parent_chunk_mismatch", "parent_chunk_not_physical")
        ),
    }


def _runtime_aggregate(timings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = [int(row["total_ns"]) for row in timings]
    selection = [int(row["selection_ns"]) for row in timings]
    composition = [int(row["composition_ns"]) for row in timings]
    packet = [int(row["packet_materialization_ns"]) for row in timings]
    return {
        "composition_mean_ns": statistics.fmean(composition),
        "composition_p95_ns": _percentile95(composition),
        "packet_materialization_mean_ns": statistics.fmean(packet),
        "packet_materialization_p95_ns": _percentile95(packet),
        "question_count": len(timings),
        "selection_mean_ns": statistics.fmean(selection),
        "selection_p95_ns": _percentile95(selection),
        "warm_mean_ns": statistics.fmean(total),
        "warm_p95_ns": _percentile95(total),
    }


def _materialize(
    *,
    v4_root: Path,
    retrieval_path: Path,
    store_root: Path,
    hooks: RuntimeHooks,
    clock: Callable[[], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_started = clock()
    step = clock()
    parent = hooks.load_parent(v4_root)
    parent_load_ns = _elapsed(clock, step)
    _require(
        tuple(parent.rows_by_ordinal) == tuple(range(EXPECTED_QUESTION_COUNT)),
        "parent loader returned a non-full100 population",
    )
    step = clock()
    context = hooks.load_context(
        retrieval_path,
        store_root=store_root,
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    context_load_ns = _elapsed(clock, step)
    population = getattr(context, "population")
    population_rows = tuple(getattr(population, "rows"))
    _require(
        len(population_rows) == EXPECTED_QUESTION_COUNT,
        "locked context population changed",
    )
    population_by_question = {
        str(row.source.packet.question_id): row for row in population_rows
    }
    _require(
        len(population_by_question) == EXPECTED_QUESTION_COUNT,
        "locked context question IDs repeated",
    )
    by_namespace: dict[str, list[int]] = {}
    namespaces: dict[str, object] = {}
    for ordinal, parent_row in parent.rows_by_ordinal.items():
        question_id = str(parent_row["question_id"])
        population_row = population_by_question.get(question_id)
        _require(population_row is not None, "v4 question absent from locked context")
        _require(
            quote_sha256(str(population_row.source.packet.dated_question))
            == parent_row["prompt_question_sha256"],
            "locked context dated question differs from v4",
        )
        namespace = population_row.namespace
        namespace_id = str(namespace.namespace_id)
        by_namespace.setdefault(namespace_id, []).append(ordinal)
        namespaces[namespace_id] = namespace

    budget = hooks.make_budget()
    budget_projection = _projection(budget, label="shadow budget")
    _require(
        budget_projection
        == {
            "max_companion_chunks": MAX_COMPANION_CHUNKS,
            "max_companion_tokens": MAX_COMPANION_TOKENS,
            "max_envelopes": MAX_ENVELOPES,
            "max_turns_per_envelope": MAX_TURNS_PER_ENVELOPE,
        },
        "user-led envelope lane budget changed",
    )
    results: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    lifecycle: list[dict[str, Any]] = []
    for namespace_id in sorted(by_namespace):
        cache_started = clock()
        cache, cache_timing = hooks.build_cache(context, namespaces[namespace_id])
        cache_ns = _elapsed(clock, cache_started)
        index_started = clock()
        index = hooks.build_index(cache)
        index_ns = _elapsed(clock, index_started)
        lifecycle.append(
            {
                **dict(cache_timing),
                "cache_lifecycle_ns": cache_ns,
                "envelope_index_build_ns": index_ns,
                "envelope_index_receipt_sha256": str(
                    getattr(index, "receipt_sha256")
                ),
                "namespace_id": namespace_id,
                "question_count": len(by_namespace[namespace_id]),
            }
        )
        try:
            for ordinal in by_namespace[namespace_id]:
                results[ordinal] = _question_row(
                    parent_row=parent.rows_by_ordinal[ordinal],
                    namespace_id=namespace_id,
                    cache=cache,
                    index=index,
                    budget=budget,
                    hooks=hooks,
                    clock=clock,
                )
        finally:
            del index, cache
            gc.collect()
    rows = [results[ordinal][0] for ordinal in range(EXPECTED_QUESTION_COUNT)]
    timings = [results[ordinal][1] for ordinal in range(EXPECTED_QUESTION_COUNT)]
    construction = {
        "aggregate": _semantic_aggregate(rows),
        "budget": budget_projection,
        "format": FORMAT,
        "gold_loaded": False,
        "implementation": _implementation_identity(),
        "model_calls": 0,
        "new_provider_calls": 0,
        "ordinals": list(range(EXPECTED_QUESTION_COUNT)),
        "parent_v4_construction_sha256": parent.construction_sha256,
        "parent_v4_replay_sha256": parent.replay_sha256,
        "parent_v4_runtime_sha256": parent.runtime_sha256,
        "policy_id": POLICY_ID,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "question_count": len(rows),
        "questions": rows,
        "retrieval_sha256": EXPECTED_RETRIEVAL_SHA256,
        "status": "sealed_gold_blind_parent_preserving_user_envelope_shadow",
    }
    assert_gold_blind(construction, path="user_envelope_shadow_construction")
    runtime = {
        "cold_setup": {
            "namespace_index_timings": lifecycle,
            "sealed_parent_load_ns": parent_load_ns,
            "sealed_store_context_load_ns": context_load_ns,
        },
        "format": RUNTIME_FORMAT,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "peak_resident_namespace_count": min(1, len(lifecycle)),
        "processed_namespace_count": len(lifecycle),
        "question_timings": timings,
        "total_ns": _elapsed(clock, total_started),
        "warm_aggregate": _runtime_aggregate(timings),
    }
    assert_gold_blind(runtime, path="user_envelope_shadow_runtime")
    return construction, runtime


def construct(
    *,
    v4_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
    hooks: RuntimeHooks | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> tuple[str, str]:
    """Build one full100 shadow in a fresh, never-before-used output root."""

    _require(not output_root.exists(), "output root must be unique and absent")
    construction, runtime = _materialize(
        v4_root=v4_root,
        retrieval_path=retrieval_path,
        store_root=store_root,
        hooks=hooks or _default_hooks(),
        clock=clock,
    )
    construction_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / CONSTRUCTION_NAME, construction
    )
    runtime = {**runtime, "construction_sha256": construction_sha}
    runtime_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / RUNTIME_NAME, runtime
    )
    print(
        "User-envelope shadow: "
        f"{construction['question_count']} rows; selected="
        f"{construction['aggregate']['selected_companion_count']}; admitted="
        f"{construction['aggregate']['admitted_companion_count']}; "
        f"construction={construction_sha}; runtime={runtime_sha}",
        flush=True,
    )
    return construction_sha, runtime_sha


def _load_constructed(
    output_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    construction, construction_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / CONSTRUCTION_NAME
    )
    runtime, runtime_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / RUNTIME_NAME
    )
    _require(
        construction.get("format") == FORMAT
        and runtime.get("format") == RUNTIME_FORMAT
        and runtime.get("construction_sha256") == construction_sha
        and construction.get("gold_loaded") is False
        and runtime.get("gold_loaded") is False
        and construction.get("model_calls") == 0
        and construction.get("new_provider_calls") == 0
        and runtime.get("model_calls") == 0
        and runtime.get("new_provider_calls") == 0,
        "shadow construction/runtime firebreak changed",
    )
    assert_gold_blind(construction, path="loaded_user_envelope_shadow")
    return construction, construction_sha, runtime, runtime_sha


def replay(
    *,
    v4_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
    hooks: RuntimeHooks | None = None,
) -> str:
    """Recompute every deterministic row and require byte-semantic identity."""

    _require(not (output_root / REPLAY_NAME).exists(), "refusing to overwrite replay")
    sealed, sealed_sha, _runtime, runtime_sha = _load_constructed(output_root)
    rebuilt, _discarded_runtime = _materialize(
        v4_root=v4_root,
        retrieval_path=retrieval_path,
        store_root=store_root,
        hooks=hooks or _default_hooks(),
        clock=time.perf_counter_ns,
    )
    _require(rebuilt == sealed, "shadow replay differs from sealed construction")
    rows = [
        {
            "effective_arm_sha256": identity_sha256(row["effective_arm"]),
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "row_receipt_sha256": row["row_receipt_sha256"],
        }
        for row in rebuilt["questions"]
    ]
    body = {
        "byte_identical": True,
        "construction_sha256": sealed_sha,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "model_calls": 0,
        "new_provider_calls": 0,
        "parent_v4_construction_sha256": EXPECTED_V4_CONSTRUCTION_SHA256,
        "question_count": len(rows),
        "questions": rows,
        "runtime_sha256": runtime_sha,
        "status": "exact_shadow_rows_reconstructed",
    }
    assert_gold_blind(body, path="user_envelope_shadow_replay")
    digest = hot._atomic_write_json(output_root / REPLAY_NAME, body)  # noqa: SLF001
    print(f"User-envelope shadow replay: {len(rows)} exact rows; replay={digest}", flush=True)
    return digest


def _metric_regressions(
    parent: Mapping[str, Any], effective: Mapping[str, Any]
) -> dict[str, bool]:
    parent_component = parent["answer_value_component_recall"]
    effective_component = effective["answer_value_component_recall"]
    return {
        "source": float(effective["gold_source_id_recall"])
        + 1e-12
        < float(parent["gold_source_id_recall"]),
        "literal": parent["literal_answer"] is True
        and effective["literal_answer"] is not True,
        "best_f1": float(effective["best_f1"]) + 1e-12
        < float(parent["best_f1"]),
        "component": parent_component is not None
        and (
            effective_component is None
            or float(effective_component) + 1e-12 < float(parent_component)
        ),
    }


def evaluate(
    *,
    dataset: Path,
    split_manifest: Path,
    v4_root: Path,
    output_root: Path,
    population_loader: Callable[..., tuple[object, object, Mapping[str, Any]]] = full100._load_population,  # noqa: SLF001
    question_flattener: Callable[[object], Sequence[object]] = full100._flatten_questions,  # noqa: SLF001
    parent_loader: Callable[[Path], ParentArtifacts] = _load_parent,
) -> str:
    """Open the benchmark only after replay and compare monotone evidence metrics."""

    _require(
        not (output_root / EVALUATION_NAME).exists(),
        "refusing to overwrite evaluation",
    )
    construction, construction_sha, _runtime, runtime_sha = _load_constructed(
        output_root
    )
    replay_artifact, replay_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / REPLAY_NAME
    )
    _require(
        replay_artifact.get("format") == REPLAY_FORMAT
        and replay_artifact.get("byte_identical") is True
        and replay_artifact.get("construction_sha256") == construction_sha
        and replay_artifact.get("runtime_sha256") == runtime_sha,
        "evaluation requires the exact shadow replay",
    )
    parent = parent_loader(v4_root)
    _require(
        construction["parent_v4_construction_sha256"]
        == parent.construction_sha256,
        "evaluation parent differs from construction",
    )
    samples, _identities, population = population_loader(dataset, split_manifest)
    _require(
        population.get("population_identity_sha256") == EXPECTED_POPULATION_SHA256,
        "evaluation population identity changed",
    )
    questions = tuple(question_flattener(samples))
    _require(len(questions) == EXPECTED_QUESTION_COUNT, "evaluation population changed")
    constructed = {int(row["ordinal"]): row for row in construction["questions"]}
    rows: list[dict[str, Any]] = []
    regressions = {name: 0 for name in ("source", "literal", "best_f1", "component")}
    for ordinal, benchmark in enumerate(questions):
        row = constructed[ordinal]
        parent_row = parent.rows_by_ordinal[ordinal]
        _require(
            row["question_id"] == benchmark.question_id
            and row["prompt_question_sha256"]
            == quote_sha256(benchmark.dated_question),
            "evaluation benchmark differs from sealed question",
        )
        parent_score = v3._score_arm(  # noqa: SLF001
            parent_row["effective_arm"], benchmark
        )
        effective_score = v3._score_arm(row["effective_arm"], benchmark)  # noqa: SLF001
        regressed = _metric_regressions(parent_score, effective_score)
        for name, value in regressed.items():
            regressions[name] += int(value)
        rows.append(
            {
                "effective": effective_score,
                "metric_regressions": regressed,
                "ordinal": ordinal,
                "parent": parent_score,
                "question_id": benchmark.question_id,
                "row_receipt_sha256": row["row_receipt_sha256"],
            }
        )
    _require(not any(regressions.values()), "parent protection metric regressed")

    def defined(metric: str, side: str) -> list[float]:
        return [
            float(row[side][metric])
            for row in rows
            if row[side][metric] is not None
        ]

    parent_components = defined("answer_value_component_recall", "parent")
    effective_components = defined("answer_value_component_recall", "effective")
    parent_f1 = defined("best_f1", "parent")
    effective_f1 = defined("best_f1", "effective")

    aggregate = {
        "effective_all_answer_value_components": sum(
            row["effective"]["all_answer_value_components"] is True for row in rows
        ),
        "effective_all_gold_source_reach": sum(
            row["effective"]["all_gold_source_ids_reached"] is True for row in rows
        ),
        "effective_literal_answer_hits": sum(
            row["effective"]["literal_answer"] is True for row in rows
        ),
        "effective_answer_value_component_recall_count": len(effective_components),
        "effective_mean_answer_value_component_recall": (
            statistics.fmean(effective_components) if effective_components else None
        ),
        "effective_mean_best_f1": statistics.fmean(effective_f1),
        "parent_all_answer_value_components": sum(
            row["parent"]["all_answer_value_components"] is True for row in rows
        ),
        "parent_all_gold_source_reach": sum(
            row["parent"]["all_gold_source_ids_reached"] is True for row in rows
        ),
        "parent_literal_answer_hits": sum(
            row["parent"]["literal_answer"] is True for row in rows
        ),
        "parent_answer_value_component_recall_count": len(parent_components),
        "parent_mean_answer_value_component_recall": (
            statistics.fmean(parent_components) if parent_components else None
        ),
        "parent_mean_best_f1": statistics.fmean(parent_f1),
        "prior_success_regressions": regressions,
        "question_count": len(rows),
        "zero_prior_success_regressions": True,
    }
    body = {
        "aggregate": aggregate,
        "construction_sha256": construction_sha,
        "format": EVALUATION_FORMAT,
        "gold_loaded": True,
        "model_calls": 0,
        "new_provider_calls": 0,
        "parent_v4_construction_sha256": parent.construction_sha256,
        "population_identity_sha256": population["population_identity_sha256"],
        "question_count": len(rows),
        "questions": rows,
        "replay_sha256": replay_sha,
        "runtime_sha256": runtime_sha,
        "status": "post_construction_gold_join_zero_parent_regressions",
    }
    digest = hot._atomic_write_json(  # noqa: SLF001
        output_root / EVALUATION_NAME, body
    )
    print(
        "User-envelope shadow evaluation: source="
        f"{aggregate['effective_all_gold_source_reach']}/{len(rows)}; literal="
        f"{aggregate['effective_literal_answer_hits']}; best_f1="
        f"{aggregate['effective_mean_best_f1']:.6f}; regressions=0; evaluation={digest}",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("construct")
    commands.add_parser("replay")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dataset", type=Path, required=True)
    evaluation.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "v4_root": args.v4_root.resolve(),
        "output_root": args.output_root.resolve(),
    }
    if args.command == "construct":
        construct(
            **common,
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
        )
    elif args.command == "replay":
        replay(
            **common,
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
        )
    elif args.command == "evaluate":
        evaluate(
            **common,
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
        )
    else:  # pragma: no cover
        raise AssertionError(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "EVALUATION_FORMAT",
    "FORMAT",
    "POLICY_ID",
    "RUNTIME_FORMAT",
    "construct",
    "evaluate",
    "replay",
]
