"""Add source-balanced surplus to the sealed v6 hot full100 packets.

This successor performs no search and no model call.  It authenticates the
frozen v6 selection, ranks source groups from the stored wide BM25/dense
frontiers, preserves every packed v6 evidence row as a prefix, hydrates only
the admitted surplus IDs, and rebuilds a provider-ready packet under the same
7,000-context/8,000-workspace caps.  Gold is opened only by ``score`` after a
byte-identical provider-free ``replay``.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.persistence.db import Database
from memory_condense.search.adaptive_source_admission import (
    POLICY_ID,
    RankedSourceLane,
    source_balanced_surplus_admission,
)
from memory_condense.search.hot_retrieval import RankedChunkAddress

try:
    from tools import assay_hot_retrieval_1m as hot
    from tools import assay_hot_retrieval_full100 as parent
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    import assay_hot_retrieval_1m as hot
    import assay_hot_retrieval_full100 as parent


SELECTION_FORMAT = "memory-condense-hot-retrieval-adaptive-full100-selection-v1"
RUNTIME_FORMAT = "memory-condense-hot-retrieval-adaptive-full100-runtime-v1"
REPLAY_FORMAT = "memory-condense-hot-retrieval-adaptive-full100-replay-v1"
SCORE_FORMAT = "memory-condense-hot-retrieval-adaptive-full100-score-v1"
EXPECTED_POPULATION_SHA256 = parent.EXPECTED_POPULATION_SHA256
EXPECTED_QUESTION_COUNT = parent.EXPECTED_QUESTION_COUNT
EXPECTED_PARENT_SELECTION_SHA256 = (
    "7062a1b23b231b9870d3e92ca94ac44f12a8a6ad68366787affd37d16ba737bf"
)
DEFAULT_PARENT_ROOT = parent.DEFAULT_OUTPUT_ROOT
DEFAULT_SOURCE_ROOT = parent.DEFAULT_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-retrieval-adaptive-full100-validation-20260905"
)
DEFAULT_SPLIT = parent.DEFAULT_SPLIT
SELECTION_NAME = parent.SELECTION_NAME
RUNTIME_NAME = "runtime.json"
REPLAY_NAME = "replay.json"
SCORE_NAME = "scores.json"
MAX_CONTEXT_TOKENS = parent.MAX_CONTEXT_TOKENS
MAX_PROMPT_TOKENS = parent.MAX_PROMPT_TOKENS
DEFAULT_SURPLUS_BUDGET = 32
DEFAULT_RRF_CONSTANT = 60
DEFAULT_MAX_HITS_PER_SOURCE_PER_LANE = 4
ADMISSION_LANES = ("bm25", "exact_dense", "temporal_event")


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "tools/assay_hot_retrieval_adaptive_full100.py",
        "tools/assay_hot_retrieval_full100.py",
        "tools/assay_hot_retrieval_1m.py",
        "src/memory_condense/search/adaptive_source_admission.py",
        "src/memory_condense/search/hot_retrieval.py",
        "src/memory_condense/search/indexes/retrieval_models.py",
        "src/memory_condense/eval/_retrieval_qa_prompt.py",
        "src/memory_condense/eval/answer_value_coverage.py",
        "src/memory_condense/eval/recall_guarded_cumulative_population.py",
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/domain/integrity.py",
        "src/memory_condense/persistence/db.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "format": "memory-condense-hot-retrieval-adaptive-full100-implementation-v1",
        "files": files,
        "sha256": identity_sha256(
            [{"path": path, "sha256": digest} for path, digest in files.items()]
        ),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_to_repository(path: Path, *, label: str) -> str:
    try:
        relative = path.resolve().relative_to(_repository_root())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    return relative.as_posix()


def _bound_parent_root(binding: Mapping[str, Any]) -> Path:
    value = binding.get("parent_output_relative_path")
    if not isinstance(value, str) or not value:
        raise ValueError("adaptive selection omitted its parent path")
    return hot._safe_relative(  # noqa: SLF001
        _repository_root(), value, label="adaptive parent output"
    )


def _load_parent(parent_root: Path) -> tuple[dict[str, Any], str]:
    selection, digest = parent._load_selection(parent_root)  # noqa: SLF001
    if digest != EXPECTED_PARENT_SELECTION_SHA256:
        raise ValueError(
            f"adaptive parent changed ({digest} != {EXPECTED_PARENT_SELECTION_SHA256})"
        )
    return selection, digest


def _load_compiled_metadata(
    parent_root: Path,
    *,
    shard_offset: int,
    expected_manifest_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = parent_root / "compiled" / f"offset-{shard_offset:03d}"
    manifest, manifest_sha = hot._read_json_artifact(  # noqa: SLF001
        root / "compiled.json"
    )
    chunks_ref = manifest.get("chunk_manifest")
    source_binding = manifest.get("source_binding")
    if (
        manifest_sha != expected_manifest_sha256
        or manifest.get("format") != parent.COMPILED_SHARD_FORMAT
        or manifest.get("shard_offset") != shard_offset
        or not isinstance(chunks_ref, Mapping)
        or not isinstance(source_binding, Mapping)
    ):
        raise ValueError(f"offset {shard_offset:03d} compiled binding changed")
    chunks_path = hot._safe_relative(  # noqa: SLF001
        root,
        str(chunks_ref.get("path", "")),
        label=f"offset {shard_offset:03d} chunk manifest",
    )
    chunks, chunks_sha = hot._read_json_artifact(chunks_path)  # noqa: SLF001
    rows = chunks.get("rows")
    expected_count = int(source_binding.get("chunk_count", -1))
    if (
        chunks_sha != chunks_ref.get("sha256")
        or chunks.get("format") != parent.COMPILED_CHUNKS_FORMAT
        or chunks.get("shard_offset") != shard_offset
        or chunks.get("chunk_count") != expected_count
        or chunks.get("gold_fields_present") is not False
        or not isinstance(rows, list)
        or len(rows) != expected_count
    ):
        raise ValueError(f"offset {shard_offset:03d} chunk metadata changed")
    chunk_ids = [str(row.get("chunk_id", "")) for row in rows]
    if (
        chunk_ids != sorted(set(chunk_ids))
        or identity_sha256(chunk_ids) != chunks.get("chunk_sequence_sha256")
    ):
        raise ValueError(f"offset {shard_offset:03d} chunk sequence changed")
    return {chunk_id: dict(row) for chunk_id, row in zip(chunk_ids, rows, strict=True)}, dict(source_binding)


def _source_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_id")
    if not isinstance(value, str) or not value:
        raise ValueError("candidate omitted source_id")
    return value


def _evidence_id(row: Mapping[str, Any]) -> str:
    value = row.get("chunk_id")
    if not isinstance(value, str) or not value:
        raise ValueError("candidate omitted chunk_id")
    return value


def _adapt_question(
    parent_row: Mapping[str, Any],
    *,
    prompt_question: str,
    database: Database,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    surplus_budget: int,
    rrf_constant: int,
    max_hits_per_source_per_lane: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    started = time.perf_counter_ns()
    parent_arm = parent_row.get("arms", {}).get("a3_protected_union")
    if not isinstance(parent_arm, Mapping):
        raise ValueError("parent row omitted A3")
    parent_evidence = parent_arm.get("packed_evidence")
    if (
        not isinstance(parent_evidence, list)
        or parent_arm.get("selected_evidence") != parent_evidence
        or parent_arm.get("dropped_chunk_ids") != []
    ):
        raise ValueError("adaptive parent is not a fully packed protected prefix")
    wide = parent_row.get("wide_frontier")
    if not isinstance(wide, Mapping):
        raise ValueError("parent row omitted wide frontier")
    lanes: list[RankedSourceLane[Mapping[str, Any]]] = []
    for lane_id in ADMISSION_LANES:
        candidates = wide.get(lane_id)
        if not isinstance(candidates, list) or not all(
            isinstance(row, Mapping) for row in candidates
        ):
            raise ValueError(f"parent wide frontier changed lane {lane_id}")
        lanes.append(RankedSourceLane(lane_id, candidates))

    selection = source_balanced_surplus_admission(
        parent_evidence,
        lanes,
        evidence_id=_evidence_id,
        source_id=_source_id,
        surplus_budget=surplus_budget,
        rrf_constant=rrf_constant,
        max_hits_per_source_per_lane=max_hits_per_source_per_lane,
    )
    selected_audit = selection.audit.selected
    if len(selected_audit) != len(selection.surplus_items):
        raise RuntimeError("surplus selection/audit length changed")
    selection_ns = time.perf_counter_ns() - started

    addresses = tuple(
        RankedChunkAddress(
            chunk_id=_evidence_id(item),
            score=float(Fraction(audit.candidate_rrf)),
            route="adaptive_source_rrf",
        )
        for item, audit in zip(
            selection.surplus_items, selected_audit, strict=True
        )
    )
    hydrated, hydrate_ns = hot._hydrate_addresses(  # noqa: SLF001
        database, addresses, metadata_by_id
    )
    prepare_started = time.perf_counter_ns()
    prepared = [hot._prepare_raw_evidence(result) for result in hydrated]  # noqa: SLF001
    prepare_ns = time.perf_counter_ns() - prepare_started
    rendered = [str(row["rendered_text"]) for row in parent_evidence]
    rendered.extend(item.rendered_text for item in prepared)
    envelope = hot._pack_provider_prompt(  # noqa: SLF001
        rendered,
        prompt_question=prompt_question,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    provider_ready_at = envelope.provider_ready_at_ns
    extra_evidence = [hot._raw_evidence_row(item) for item in prepared]  # noqa: SLF001
    arm = hot._raw_packet_semantic(  # noqa: SLF001
        [*parent_evidence, *extra_evidence], envelope
    )
    parent_ids = [str(row["chunk_id"]) for row in parent_evidence]
    if arm["packed_chunk_ids"][: len(parent_ids)] != parent_ids:
        raise RuntimeError("adaptive packing displaced protected parent evidence")
    semantic = {
        "ordinal": int(parent_row["ordinal"]),
        "shard_offset": int(parent_row["shard_offset"]),
        "local_ordinal": int(parent_row["local_ordinal"]),
        "question_id": str(parent_row["question_id"]),
        "probe_sha256": str(parent_row["probe_sha256"]),
        "retrieval_query_sha256": str(parent_row["retrieval_query_sha256"]),
        "prompt_question_sha256": str(parent_row["prompt_question_sha256"]),
        "parent_provider_payload_sha256": str(
            parent_arm["provider_payload_sha256"]
        ),
        "source_surplus_admission": selection.audit.projection(),
        "arms": {"a3_protected_union": arm},
    }
    timings = {
        "source_surplus_selection_ns": selection_ns,
        "hydrate_surplus_raw_ns": hydrate_ns,
        "prepare_surplus_provider_text_ns": prepare_ns,
        "pack_and_prompt_render_count_ns": envelope.pack_and_prompt_render_count_ns,
        "serialize_ns": envelope.serialize_ns,
        "adaptive_prompt_to_serialized_bytes_ns": provider_ready_at - started,
    }
    return semantic, timings


def _collect(
    *,
    parent_root: Path,
    source_root: Path,
    surplus_budget: int,
    rrf_constant: int,
    max_hits_per_source_per_lane: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent_selection, _parent_sha = _load_parent(parent_root)
    probes, _probes_sha = parent._load_probes(parent_root)  # noqa: SLF001
    catalog, _catalog_sha = parent._load_catalog(parent_root)  # noqa: SLF001
    probe_by_ordinal = {int(row["ordinal"]): row for row in probes["questions"]}
    parent_by_offset = {
        offset: [
            row
            for row in parent_selection["questions"]
            if int(row["shard_offset"]) == offset
        ]
        for offset in parent.LOCKED_100Q_OFFSETS
    }
    identity_by_offset = dict(
        zip(
            parent.LOCKED_100Q_OFFSETS,
            probes["population_identity"]["ordered_shard_identity_sha256s"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for source_row, catalog_row, offset in zip(
        probes["source_bindings"],
        catalog["shards"],
        parent.LOCKED_100Q_OFFSETS,
        strict=True,
    ):
        binding = parent._load_source_binding(  # noqa: SLF001
            source_root,
            offset,
            expected_shard_identity=None,
            expected_shard_identity_sha256=str(identity_by_offset[offset]),
            verify_large_files=True,
        )
        if binding.artifact_row() != source_row:
            raise ValueError(f"offset {offset:03d} source binding changed")
        metadata, compiled_source = _load_compiled_metadata(
            parent_root,
            shard_offset=offset,
            expected_manifest_sha256=str(catalog_row["compiled_sha256"]),
        )
        if compiled_source != binding.artifact_row():
            raise ValueError(f"offset {offset:03d} compiled/source binding changed")
        with Database(binding.database_path, read_only=True) as database:
            for parent_row in parent_by_offset[offset]:
                probe = probe_by_ordinal[int(parent_row["ordinal"])]
                semantic, sample_timings = _adapt_question(
                    parent_row,
                    prompt_question=str(probe["prompt_question"]),
                    database=database,
                    metadata_by_id=metadata,
                    surplus_budget=surplus_budget,
                    rrf_constant=rrf_constant,
                    max_hits_per_source_per_lane=max_hits_per_source_per_lane,
                )
                rows.append(semantic)
                timings.append(
                    {
                        "ordinal": semantic["ordinal"],
                        "question_id": semantic["question_id"],
                        "shard_offset": offset,
                        "timings_ns": sample_timings,
                    }
                )
        print(f"Adapted offset-{offset:03d}: 10/10", flush=True)
    rows.sort(key=lambda row: int(row["ordinal"]))
    timings.sort(key=lambda row: int(row["ordinal"]))
    if [row["ordinal"] for row in rows] != list(range(EXPECTED_QUESTION_COUNT)):
        raise RuntimeError("adaptive selection population changed")
    return rows, timings


def _timing_summary(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize empty timings")
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
        "max": ordered[-1],
    }


def run(
    *,
    parent_root: Path,
    source_root: Path,
    output_root: Path,
    surplus_budget: int,
    rrf_constant: int,
    max_hits_per_source_per_lane: int,
) -> str:
    if (
        isinstance(surplus_budget, bool)
        or not isinstance(surplus_budget, int)
        or surplus_budget < 0
    ):
        raise ValueError("surplus budget must be a non-negative integer")
    if (
        isinstance(rrf_constant, bool)
        or not isinstance(rrf_constant, int)
        or rrf_constant < 0
    ):
        raise ValueError("RRF constant must be a non-negative integer")
    if (
        isinstance(max_hits_per_source_per_lane, bool)
        or not isinstance(max_hits_per_source_per_lane, int)
        or max_hits_per_source_per_lane < 1
    ):
        raise ValueError("per-source lane hit cap must be a positive integer")
    path = output_root / SELECTION_NAME
    if path.exists():
        _body, digest = _load_selection(output_root)
        runtime, _runtime_sha = hot._read_json_artifact(  # noqa: SLF001
            output_root / RUNTIME_NAME
        )
        if (
            runtime.get("format") != RUNTIME_FORMAT
            or runtime.get("selection_sha256") != digest
            or runtime.get("population_identity_sha256")
            != EXPECTED_POPULATION_SHA256
        ):
            raise ValueError("adaptive runtime receipt changed")
        print(f"Adaptive selection verified: {path} ({digest})", flush=True)
        return digest
    parent_selection, parent_sha = _load_parent(parent_root)
    started = time.perf_counter_ns()
    rows, timing_rows = _collect(
        parent_root=parent_root,
        source_root=source_root,
        surplus_budget=surplus_budget,
        rrf_constant=rrf_constant,
        max_hits_per_source_per_lane=max_hits_per_source_per_lane,
    )
    controls = {
        "policy_id": POLICY_ID,
        "parent_policy_id": parent.POLICY_ID,
        "parent_protection": "exact_packed_evidence_prefix",
        "admission_lanes": list(ADMISSION_LANES),
        "surplus_budget": surplus_budget,
        "rrf_constant": rrf_constant,
        "max_hits_per_source_per_lane": max_hits_per_source_per_lane,
        "source_id_semantics": "opaque_exact_equality_only",
        "packing": "ranked_prefix_parent_then_source_representatives",
        "max_context_token_proxy": MAX_CONTEXT_TOKENS,
        "max_prompt_workspace_token_proxy": MAX_PROMPT_TOKENS,
    }
    body = {
        "format": SELECTION_FORMAT,
        "status": "sealed_gold_blind_locked_full100_adaptive_source_surplus_v1",
        "bindings": {
            "parent_output_relative_path": _relative_to_repository(
                parent_root, label="parent output"
            ),
            "parent_selection_sha256": parent_sha,
            "population_identity_sha256": EXPECTED_POPULATION_SHA256,
            "parent_bindings": parent_selection["bindings"],
        },
        "corpus": parent_selection["corpus"],
        "controls": controls,
        "implementation": _implementation_identity(),
        "questions": rows,
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    parent._assert_gold_free_rows(rows)  # noqa: SLF001
    selection_sha = hot._atomic_write_json(path, body)  # noqa: SLF001
    summary = _timing_summary(
        [
            int(row["timings_ns"]["adaptive_prompt_to_serialized_bytes_ns"])
            for row in timing_rows
        ]
    )
    runtime = {
        "format": RUNTIME_FORMAT,
        "status": "provider_free_adaptive_increment_over_sealed_parent",
        "selection_sha256": selection_sha,
        "parent_selection_sha256": parent_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "elapsed_ns": time.perf_counter_ns() - started,
        "samples": timing_rows,
        "adaptive_prompt_to_serialized_bytes_ns": summary,
        "adaptive_p50_milliseconds": float(summary["p50"]) / 1e6,
        "adaptive_p95_milliseconds": int(summary["p95"]) / 1e6,
        "parent_retrieval_latency_excluded": True,
        "provider_rtt_prefill_decode_excluded": True,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    runtime_sha = hot._atomic_write_json(  # noqa: SLF001
        output_root / RUNTIME_NAME, runtime
    )
    print(
        f"Adaptive selection published: {selection_sha}; runtime={runtime_sha}; "
        f"incremental_p95={runtime['adaptive_p95_milliseconds']:.3f}ms",
        flush=True,
    )
    return selection_sha


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / SELECTION_NAME
    )
    bindings = body.get("bindings")
    controls = body.get("controls")
    rows = body.get("questions")
    if (
        body.get("format") != SELECTION_FORMAT
        or body.get("status")
        != "sealed_gold_blind_locked_full100_adaptive_source_surplus_v1"
        or body.get("implementation") != _implementation_identity()
        or body.get("gold_fields_present") is not False
        or body.get("retained_request_token_state_bytes") != 0
        or any(
            body.get(key) != 0
            for key in ("qwen_calls", "responder_calls", "judge_calls", "provider_calls")
        )
        or not isinstance(bindings, Mapping)
        or bindings.get("population_identity_sha256")
        != EXPECTED_POPULATION_SHA256
        or bindings.get("parent_selection_sha256")
        != EXPECTED_PARENT_SELECTION_SHA256
        or not isinstance(controls, Mapping)
        or controls.get("policy_id") != POLICY_ID
        or controls.get("admission_lanes") != list(ADMISSION_LANES)
        or controls.get("max_context_token_proxy") != MAX_CONTEXT_TOKENS
        or controls.get("max_prompt_workspace_token_proxy") != MAX_PROMPT_TOKENS
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in rows]
        != list(range(EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("adaptive full100 selection changed")
    parent_root = _bound_parent_root(bindings)
    parent_selection, parent_sha = _load_parent(parent_root)
    if (
        parent_sha != bindings.get("parent_selection_sha256")
        or parent_selection.get("bindings") != bindings.get("parent_bindings")
        or body.get("corpus") != parent_selection.get("corpus")
    ):
        raise ValueError("adaptive parent binding changed")
    surplus_budget = controls.get("surplus_budget")
    rrf_constant = controls.get("rrf_constant")
    hit_cap = controls.get("max_hits_per_source_per_lane")
    if (
        isinstance(surplus_budget, bool)
        or not isinstance(surplus_budget, int)
        or surplus_budget < 0
        or isinstance(rrf_constant, bool)
        or not isinstance(rrf_constant, int)
        or rrf_constant < 0
        or isinstance(hit_cap, bool)
        or not isinstance(hit_cap, int)
        or hit_cap < 1
    ):
        raise ValueError("adaptive controls changed")
    parent._assert_gold_free_rows(rows)  # noqa: SLF001
    probes, _probes_sha = parent._load_probes(parent_root)  # noqa: SLF001
    for row, parent_row in zip(rows, parent_selection["questions"], strict=True):
        if any(
            row.get(key) != parent_row.get(key)
            for key in (
                "ordinal",
                "shard_offset",
                "local_ordinal",
                "question_id",
                "probe_sha256",
                "retrieval_query_sha256",
                "prompt_question_sha256",
            )
        ):
            raise ValueError("adaptive question binding changed")
        parent_arm = parent_row["arms"]["a3_protected_union"]
        arm = row.get("arms", {}).get("a3_protected_union")
        audit = row.get("source_surplus_admission")
        if not isinstance(arm, Mapping) or not isinstance(audit, Mapping):
            raise ValueError("adaptive question omitted its packet or audit")
        parent_ids = list(parent_arm["packed_chunk_ids"])
        selected_audit = audit.get("selected")
        if (
            row.get("parent_provider_payload_sha256")
            != parent_arm.get("provider_payload_sha256")
            or arm.get("selected_chunk_ids", [])[: len(parent_ids)] != parent_ids
            or arm.get("packed_chunk_ids", [])[: len(parent_ids)] != parent_ids
            or audit.get("parent_evidence_ids") != parent_ids
            or not isinstance(selected_audit, list)
            or not all(isinstance(item, Mapping) for item in selected_audit)
            or [item.get("evidence_id") for item in selected_audit]
            != arm.get("selected_chunk_ids", [])[len(parent_ids) :]
            or audit.get("policy_id") != POLICY_ID
            or audit.get("surplus_budget") != surplus_budget
            or audit.get("rrf_constant") != rrf_constant
            or audit.get("max_hits_per_source_per_lane") != hit_cap
        ):
            raise ValueError("adaptive packet displaced or changed its parent")
        prompt_question = str(probes["questions"][int(row["ordinal"])]["prompt_question"])
        hot._validate_arm_payload(  # noqa: SLF001
            arm,
            prompt_question=prompt_question,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )
    return body, digest


def replay(*, source_root: Path, output_root: Path) -> str:
    expected, selection_sha = _load_selection(output_root)
    controls = expected["controls"]
    parent_root = _bound_parent_root(expected["bindings"])
    rows, _timings = _collect(
        parent_root=parent_root,
        source_root=source_root,
        surplus_budget=int(controls["surplus_budget"]),
        rrf_constant=int(controls["rrf_constant"]),
        max_hits_per_source_per_lane=int(
            controls["max_hits_per_source_per_lane"]
        ),
    )
    expected_bytes = hot._canonical_json_bytes(expected["questions"])  # noqa: SLF001
    replayed_bytes = hot._canonical_json_bytes(rows)  # noqa: SLF001
    if replayed_bytes != expected_bytes:
        raise RuntimeError("adaptive gold-blind replay differs from selection")
    body = {
        "format": REPLAY_FORMAT,
        "status": "byte_identical_gold_blind_adaptive_full100_replay",
        "selection_sha256": selection_sha,
        "question_population_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "replayed_question_population_sha256": hashlib.sha256(replayed_bytes).hexdigest(),
        "byte_identical": True,
        "question_count": EXPECTED_QUESTION_COUNT,
        "gold_fields_present": False,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    digest = hot._atomic_write_json(output_root / REPLAY_NAME, body)  # noqa: SLF001
    print(f"Adaptive replay published: {digest}; byte_identical=true", flush=True)
    return digest


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    output_root: Path,
) -> str:
    selection, selection_sha = _load_selection(output_root)
    replay_body, replay_sha = hot._read_json_artifact(  # noqa: SLF001
        output_root / REPLAY_NAME
    )
    question_bytes = hot._canonical_json_bytes(selection["questions"])  # noqa: SLF001
    if (
        replay_body.get("format") != REPLAY_FORMAT
        or replay_body.get("selection_sha256") != selection_sha
        or replay_body.get("byte_identical") is not True
        or replay_body.get("question_population_sha256")
        != hashlib.sha256(question_bytes).hexdigest()
    ):
        raise ValueError("score requires the sealed adaptive replay")
    samples, _identities, population = parent._load_population(  # noqa: SLF001
        dataset, split_manifest
    )
    questions = parent._flatten_questions(samples)  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    for selected, question in zip(selection["questions"], questions, strict=True):
        if selected.get("question_id") != question.question_id:
            raise ValueError("adaptive selection differs from gold population")
        arm = selected["arms"]["a3_protected_union"]
        evidence = arm["packed_evidence"]
        texts = [str(item["raw_text"]) for item in evidence]
        retrieved = {
            str(item["source_id"]) for item in evidence if item.get("source_id")
        }
        expected_sources = set(str(value) for value in question.evidence_sources)
        recall = (
            None
            if not expected_sources
            else len(expected_sources & retrieved) / len(expected_sources)
        )
        components = answer_value_component_coverage(
            question.answer, len(expected_sources), texts
        )
        rows.append(
            {
                "ordinal": int(selected["ordinal"]),
                "question_id": question.question_id,
                "category": question.category,
                "packed_count": len(evidence),
                "selected_count": len(arm["selected_evidence"]),
                "dropped_count": len(arm["dropped_chunk_ids"]),
                "context_token_proxy": arm["context_token_proxy"],
                "prompt_workspace_token_proxy": arm[
                    "prompt_workspace_token_proxy"
                ],
                "answer_present": contains_answer(texts, question.answer),
                "best_evidence_f1": best_f1(texts, question.answer),
                "evidence_source_recall": recall,
                "all_evidence_sources": None if recall is None else recall == 1.0,
                "answer_value_component_recall": (
                    None if components is None else components.recall
                ),
                "all_answer_value_components": (
                    None if components is None else components.all_components
                ),
            }
        )
    recalls = [
        float(row["evidence_source_recall"])
        for row in rows
        if row["evidence_source_recall"] is not None
    ]
    component_recalls = [
        float(row["answer_value_component_recall"])
        for row in rows
        if row["answer_value_component_recall"] is not None
    ]
    body = {
        "format": SCORE_FORMAT,
        "status": "locked_adaptive_full100_retrieval_diagnostics_not_answer_accuracy",
        "selection_sha256": selection_sha,
        "replay_sha256": replay_sha,
        "population_identity_sha256": population["population_identity_sha256"],
        "questions": rows,
        "aggregate": {
            "question_count": len(rows),
            "packed_all_evidence_source_hits": sum(
                row["all_evidence_sources"] is True for row in rows
            ),
            "packed_literal_answer_hits": sum(bool(row["answer_present"]) for row in rows),
            "mean_best_evidence_f1": statistics.fmean(
                float(row["best_evidence_f1"]) for row in rows
            ),
            "mean_evidence_source_recall": statistics.fmean(recalls),
            "mean_answer_value_component_recall": (
                None
                if not component_recalls
                else statistics.fmean(component_recalls)
            ),
            "mean_packed_count": statistics.fmean(
                int(row["packed_count"]) for row in rows
            ),
            "total_dropped_count": sum(int(row["dropped_count"]) for row in rows),
            "max_context_token_proxy": max(
                int(row["context_token_proxy"]) for row in rows
            ),
            "max_prompt_workspace_token_proxy": max(
                int(row["prompt_workspace_token_proxy"]) for row in rows
            ),
        },
        "gold_fields_present": True,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    digest = hot._atomic_write_json(output_root / SCORE_NAME, body)  # noqa: SLF001
    print(
        f"Adaptive retrieval score published: {digest}; "
        f"sources={body['aggregate']['packed_all_evidence_source_hits']}/100; "
        f"literal={body['aggregate']['packed_literal_answer_hits']}/100",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--surplus-budget", type=int, default=DEFAULT_SURPLUS_BUDGET)
    run_parser.add_argument("--rrf-constant", type=int, default=DEFAULT_RRF_CONSTANT)
    run_parser.add_argument(
        "--max-hits-per-source-per-lane",
        type=int,
        default=DEFAULT_MAX_HITS_PER_SOURCE_PER_LANE,
    )
    commands.add_parser("replay")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    if args.command == "run":
        run(
            parent_root=args.parent_root.resolve(),
            source_root=source_root,
            output_root=output_root,
            surplus_budget=args.surplus_budget,
            rrf_constant=args.rrf_constant,
            max_hits_per_source_per_lane=args.max_hits_per_source_per_lane,
        )
    elif args.command == "replay":
        replay(source_root=source_root, output_root=output_root)
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            output_root=output_root,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
