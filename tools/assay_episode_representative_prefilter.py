"""Replay only S2 episode selection against a sealed cumulative 1M store.

The assay reuses the historical S0 protected anchor IDs, opens the combined
store read-only, and runs the current representative Qwen tournament with a
shadow descriptor shortlist.  It therefore measures same-runtime containment without
rebuilding the corpus, calling a provider, exposing gold answers, or changing
the authoritative retrieval result.  The historical plan hash is retained as
a compatibility diagnostic; it is not substituted for the current winners.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval.recall_guarded_cumulative_1m import (
    _atomic_write_json,
    _load_shared_qwen,
    _read_canonical_json,
)
from memory_condense.eval._recall_guarded_cumulative_validation_shard import (
    DEFAULT_POLICY,
    DEFAULT_QWEN_CHOICE,
    DEFAULT_QWEN_PREFIX,
    DEFAULT_SPLIT,
    LOCKED_100Q_OFFSETS,
    SOURCE_ROUTER_MAX_SOURCES,
    SOURCE_ROUTER_RRF_CONSTANT,
    ValidationShardPreflight,
    _UnboundCoverageSelector,
    _held_out_queries,
    _representative_policy,
    preflight_locked_validation_shard,
    validate_validation_shard_retrieval,
)
from memory_condense.eval.recall_guarded_cumulative_1m_source import (
    current_source_binding,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    open_recall_guarded_cumulative_store,
)
from memory_condense.search.episodes import (
    EpisodeRepresentativePrefilterPolicy,
)


FORMAT = "memory-condense-validation-episode-prefilter-shadow-assay-v3"
_SHA256_ALPHABET = frozenset("0123456789abcdef")


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _campaign_root_from_shard_root(
    store_root: str | Path,
    *,
    sample_offset: int,
) -> Path:
    """Resolve only ``.../shards/offset-NNN`` validation shard roots."""

    root = Path(store_root).resolve()
    expected_name = f"offset-{sample_offset:03d}"
    if root.name != expected_name or root.parent.name != "shards":
        raise ValueError(
            "--store-root must be the exact validation shard root "
            f".../shards/{expected_name}"
        )
    return root.parent.parent


def _artifact_validation_preflight(
    preflight: ValidationShardPreflight,
    retrieval: Mapping[str, Any],
) -> ValidationShardPreflight:
    """Bind validation to the sealed baseline implementation coordinate.

    A shadow assay necessarily runs newer code than its historical baseline.
    Every other preflight coordinate remains current and exact; the baseline
    implementation digest is accepted only from the canonical sealed header,
    then the normal shard validator cross-checks it through all ten rows.
    """

    historical_implementation = _require_sha256(
        retrieval.get("retrieval_implementation_sha256"),
        "historical retrieval implementation",
    )
    historical_environment = _require_sha256(
        retrieval.get("environment_lock_sha256"),
        "historical retrieval environment",
    )
    if historical_environment != preflight.environment_lock_sha256:
        raise ValueError(
            "historical retrieval environment differs from the assay preflight"
        )
    return replace(
        preflight,
        retrieval_implementation_sha256=historical_implementation,
    )


def _indices(value: str, *, count: int) -> tuple[int, ...]:
    normalized = value.strip().casefold()
    if normalized == "all":
        return tuple(range(count))
    selected: list[int] = []
    for part in normalized.split(","):
        token = part.strip()
        if not token:
            raise ValueError("question indices contain an empty item")
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError("question index range is descending")
            selected.extend(range(start, stop + 1))
        else:
            selected.append(int(token))
    result = tuple(dict.fromkeys(selected))
    if not result or any(index < 0 or index >= count for index in result):
        raise ValueError(f"question indices must lie in [0, {count - 1}]")
    return result


def _historical_anchors(
    condenser: Any,
    row: dict[str, Any],
) -> tuple[RetrievalResult, ...]:
    predecessor = row.get("predecessor_receipt")
    if not isinstance(predecessor, dict):
        raise ValueError("historical question omitted its predecessor receipt")
    chunk_ids = tuple(predecessor.get("protected_chunk_ids", ()))
    if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("historical predecessor anchors are empty or duplicated")
    total = len(chunk_ids)
    anchors: list[RetrievalResult] = []
    for rank, chunk_id in enumerate(chunk_ids, 1):
        result = condenser.retriever.hydrate_chunk(
            str(chunk_id),
            score=(total - rank + 1) / total,
            route="historical_causal_coverage_anchor",
        )
        if not isinstance(result, RetrievalResult):
            raise ValueError(f"historical anchor is missing: {chunk_id}")
        anchors.append(result)
    return tuple(anchors)


def _validate_historical_input(
    retrieval: Mapping[str, Any],
    *,
    preflight: ValidationShardPreflight,
) -> tuple[dict[str, Any], ...]:
    validation_preflight = _artifact_validation_preflight(preflight, retrieval)
    validate_validation_shard_retrieval(
        retrieval,
        preflight=validation_preflight,
    )
    rows = retrieval.get("questions")
    if not isinstance(rows, list) or len(rows) != len(preflight.sample.questions):
        raise ValueError("historical retrieval question population changed")
    # The formal validator above proves the complete ordered question binding.
    # Preserve concrete dictionaries for the replay loop without weakening it.
    return tuple(dict(row) for row in rows)


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty population")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("boundary measurements must be non-negative integers")
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _serialized_messages_utf8_bytes(messages: object) -> int:
    if not isinstance(messages, list):
        raise ValueError("provider_messages must be a list")
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(payload)


def _historical_final_stage_boundary_aggregates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the sealed S3 boundary without retaining message content."""

    metrics: dict[str, list[int]] = {
        "prompt_token_proxy": [],
        "context_token_proxy": [],
        "selected_evidence_count": [],
        "provider_messages_serialized_utf8_bytes": [],
    }
    stage_id: str | None = None
    for row in rows:
        stages = row.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("historical retrieval omitted its final stage")
        final_stage = stages[-1]
        if not isinstance(final_stage, Mapping):
            raise ValueError("historical final stage must be an object")
        observed_stage_id = final_stage.get("stage_id")
        if not isinstance(observed_stage_id, str) or not observed_stage_id:
            raise ValueError("historical final stage ID is missing")
        if stage_id is None:
            stage_id = observed_stage_id
        elif observed_stage_id != stage_id:
            raise ValueError("historical final-stage population is inconsistent")
        receipt = final_stage.get("stage_receipt")
        evidence = final_stage.get("evidence")
        if not isinstance(receipt, Mapping) or not isinstance(evidence, list):
            raise ValueError("historical final stage is incomplete")
        for name in ("prompt_token_proxy", "context_token_proxy"):
            value = receipt.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"historical {name} must be a non-negative integer")
            metrics[name].append(value)
        metrics["selected_evidence_count"].append(len(evidence))
        metrics["provider_messages_serialized_utf8_bytes"].append(
            _serialized_messages_utf8_bytes(final_stage.get("provider_messages"))
        )
    if stage_id is None:
        raise ValueError("historical retrieval question population is empty")
    return {
        "stage_id": stage_id,
        "question_count": len(rows),
        "p95_method": "nearest_rank",
        "serialization": "canonical_compact_json_utf8_without_trailing_newline",
        "metrics": {
            name: _numeric_summary(values)
            for name, values in metrics.items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shadow a descriptor shortlist against historical S2 Qwen winners."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=None,
        help="defaults to STORE_ROOT/retrieval.json and may not escape it",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--qwen-prefix-model-dir",
        type=Path,
        default=DEFAULT_QWEN_PREFIX,
    )
    parser.add_argument(
        "--qwen-choice-model-dir",
        type=Path,
        default=DEFAULT_QWEN_CHOICE,
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        choices=LOCKED_100Q_OFFSETS,
        default=0,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--questions", default="all")
    parser.add_argument("--shortlist-cap", type=int, default=16)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    store_root = args.store_root.resolve()
    campaign_root = _campaign_root_from_shard_root(
        store_root,
        sample_offset=args.sample_offset,
    )
    preflight = preflight_locked_validation_shard(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        policy_path=args.policy_manifest,
        output_root=campaign_root,
        sample_offset=args.sample_offset,
        qwen_prefix_model_dir=args.qwen_prefix_model_dir,
        qwen_choice_model_dir=args.qwen_choice_model_dir,
        device=args.device,
    )
    if preflight.shard_root.resolve() != store_root:
        raise ValueError("--store-root does not match the preflight shard root")
    expected_retrieval_path = store_root / "retrieval.json"
    retrieval_path = (
        expected_retrieval_path
        if args.retrieval is None
        else args.retrieval.resolve()
    )
    if retrieval_path != expected_retrieval_path:
        raise ValueError("--retrieval must be STORE_ROOT/retrieval.json")
    historical, historical_digest = _read_canonical_json(retrieval_path)
    historical_rows = _validate_historical_input(
        historical,
        preflight=preflight,
    )
    selected_indices = _indices(
        args.questions,
        count=len(preflight.sample.questions),
    )
    historical_source_receipt = historical.get("source_store_receipt")
    if not isinstance(historical_source_receipt, Mapping):
        raise ValueError("historical retrieval omitted its source-store receipt")
    raw_embedding_identity = historical_source_receipt.get("embedding_identity")
    if not isinstance(raw_embedding_identity, Mapping):
        raise ValueError("historical source receipt omitted its embedding identity")
    embedding_identity = dict(raw_embedding_identity)
    embedding_identity_sha256 = _require_sha256(
        historical_source_receipt.get("embedding_identity_sha256"),
        "historical embedding identity",
    )
    if identity_sha256(embedding_identity) != embedding_identity_sha256:
        raise ValueError("historical embedding identity digest changed")
    combined_store_root = store_root / "combined-store"
    if not combined_store_root.is_dir():
        raise FileNotFoundError(
            "sealed validation shard has no combined store: "
            f"{combined_store_root}"
        )

    _source_config, binding = current_source_binding(
        preflight.policy.config,
        qwen_model_dir=preflight.qwen_prefix_model_dir,
    )
    embedder = binding.embedder
    prepared = None
    selector = None
    try:
        prepared = open_recall_guarded_cumulative_store(
            combined_store_root,
            config=preflight.policy.config,
            embedder=embedder,
            held_out_queries=_held_out_queries(preflight.sample),
            coverage_selector=_UnboundCoverageSelector(),
        )
        historical_combined_receipt = historical.get(
            "combined_store_receipt_sha256"
        )
        if prepared.receipt.receipt_sha256 != historical_combined_receipt:
            raise ValueError(
                "physical combined store differs from the sealed retrieval"
            )
        if (
            prepared.receipt.source_database_sha256
            != historical_source_receipt.get("database_sha256")
        ):
            raise ValueError(
                "combined store source database differs from the sealed retrieval"
            )
        artifact_id = prepared.compilation.artifact.artifact_id
        historical_compilation_receipt = historical.get(
            "compilation_receipt_sha256"
        )
        if (
            prepared.compilation.receipt_sha256
            != historical_compilation_receipt
        ):
            raise ValueError("physical compilation differs from the sealed retrieval")
        representative_policy = _representative_policy(artifact_id)
        prefilter_policy = EpisodeRepresentativePrefilterPolicy(
            mode="shadow",
            cap=args.shortlist_cap,
            top_k=representative_policy.top_k,
            cutoff_margin=args.min_margin,
        )
        descriptor_build = (
            prepared.condenser.prepare_discourse_episode_descriptors(
                artifact_id,
                embedding_identity=embedding_identity,
            )
        )
        base_output: dict[str, Any] = {
            "format": FORMAT,
            "sample_offset": preflight.sample_offset,
            "shard_identity_sha256": preflight.shard_identity[
                "shard_identity_sha256"
            ],
            "population_identity_sha256": preflight.population_identity[
                "population_identity_sha256"
            ],
            "validation_policy_attestation_sha256": (
                preflight.policy.attestation_sha256
            ),
            "retrieval_policy_sha256": preflight.policy.retrieval_policy_sha256,
            "historical_retrieval_sha256": historical_digest,
            "historical_retrieval_implementation_sha256": historical[
                "retrieval_implementation_sha256"
            ],
            "assay_implementation_sha256": (
                preflight.retrieval_implementation_sha256
            ),
            "environment_lock_sha256": preflight.environment_lock_sha256,
            "source_store_receipt_sha256": historical_source_receipt[
                "receipt_sha256"
            ],
            "combined_store_receipt_sha256": prepared.receipt.receipt_sha256,
            "embedding_identity_sha256": embedding_identity_sha256,
            "descriptor_catalog_receipt": (
                descriptor_build.receipt.identity_payload()
            ),
            "descriptor_prepare_elapsed_ms": descriptor_build.timing.elapsed_ms,
            "shortlist_policy": prefilter_policy.identity_payload(),
            "selected_question_indices": list(selected_indices),
            "historical_final_stage_primary_llm_boundary": (
                _historical_final_stage_boundary_aggregates(historical_rows)
            ),
            "questions": [],
            "provider_calls": 0,
            "gold_fields_present": False,
        }
        if args.prepare_only:
            _atomic_write_json(args.output.resolve(), base_output)
            return 0

        # Query vectors are frozen inside the read-only condenser; BGE can be
        # released before either local Qwen checkpoint is loaded.
        embedder.close()
        selector, representative_linker = _load_shared_qwen(
            preflight.policy.config,
            preflight.qwen_prefix_model_dir,
            preflight.qwen_choice_model_dir,
        )
        question_outputs: list[dict[str, Any]] = []
        for assay_ordinal, question_index in enumerate(selected_indices, 1):
            question = preflight.sample.questions[question_index]
            global_ordinal = preflight.sample_offset + question_index
            print(
                f"S2 shadow {assay_ordinal}/{len(selected_indices)}: "
                f"q{global_ordinal:03d} {question.question_id}",
                flush=True,
            )
            started = time.perf_counter()
            anchors = _historical_anchors(
                prepared.condenser,
                historical_rows[question_index],
            )
            scope = prepared.condenser.route_discourse_episode_sources(
                question.question,
                anchors,
                artifact_id=artifact_id,
                max_sources=SOURCE_ROUTER_MAX_SOURCES,
                rrf_constant=SOURCE_ROUTER_RRF_CONSTANT,
            )
            plan = prepared.condenser.retrieve_discourse_episode_representatives(
                question.question,
                scope.candidates,
                representative_linker,
                policy=representative_policy,
                source_scope=scope,
                prefilter_policy=prefilter_policy,
                prefilter_embedding_identity=embedding_identity,
            )
            historical_receipt = historical_rows[question_index].get(
                "retrieval_receipt"
            )
            if not isinstance(historical_receipt, Mapping):
                raise ValueError("historical question omitted its retrieval receipt")
            expected_plan_receipt = historical_receipt.get(
                "representative_expansion_receipt_sha256"
            )
            report = dict(prepared.condenser.last_episode_prefilter_report)
            elapsed = time.perf_counter() - started
            output = {
                "local_ordinal": question_index,
                "ordinal": global_ordinal,
                "question_id": question.question_id,
                "question_sha256": quote_sha256(question.question),
                "dated_question_sha256": quote_sha256(question.dated_question),
                "source_scope_receipt_sha256": scope.receipt_sha256,
                "representative_plan_receipt_sha256": plan.receipt_sha256,
                "historical_representative_plan_receipt_sha256": (
                    expected_plan_receipt
                ),
                "historical_representative_plan_receipt_match": (
                    plan.receipt_sha256 == expected_plan_receipt
                ),
                "full_qwen_episode_ids": [seed.episode_id for seed in plan.seeds],
                "full_qwen_passes": plan.passes,
                "full_qwen_candidate_inspections": (
                    plan.total_candidate_inspections
                ),
                "shadow": report,
                "elapsed_seconds": elapsed,
            }
            question_outputs.append(output)
            print(
                "  candidates="
                f"{report.get('input_count')} -> {report.get('proposal_count')}; "
                f"winner_recall={report.get('full_qwen_hit_recall')}; "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        recalls = [
            float(row["shadow"]["full_qwen_hit_recall"])
            for row in question_outputs
        ]
        ratios = [
            float(row["shadow"]["proposal_count"])
            / max(int(row["shadow"]["input_count"]), 1)
            for row in question_outputs
        ]
        base_output["questions"] = question_outputs
        base_output["aggregates"] = {
            "question_count": len(question_outputs),
            "historical_plan_match_count": sum(
                bool(row["historical_representative_plan_receipt_match"])
                for row in question_outputs
            ),
            "perfect_winner_containment_count": sum(value == 1.0 for value in recalls),
            "mean_full_qwen_hit_recall": sum(recalls) / len(recalls),
            "mean_proposal_fraction": sum(ratios) / len(ratios),
            "total_full_qwen_passes": sum(
                int(row["full_qwen_passes"]) for row in question_outputs
            ),
            "total_full_qwen_candidate_inspections": sum(
                int(row["full_qwen_candidate_inspections"])
                for row in question_outputs
            ),
        }
        digest = _atomic_write_json(args.output.resolve(), base_output)
        print(f"Published shadow assay {args.output.resolve()} ({digest})", flush=True)
        return 0
    finally:
        if selector is not None:
            selector.close()
        if prepared is not None:
            prepared.close()
        close = getattr(embedder, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
