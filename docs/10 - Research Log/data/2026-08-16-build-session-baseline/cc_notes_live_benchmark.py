"""Held-out notes test for bounded Qwen linking, compact CAVs, and graph recall."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import statistics
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from memory_condense.association_store import AssociationArtifact
from memory_condense.condenser import MemoryCondenser
from memory_condense.corpus import (
    build_conversation_recall_slice,
    load_corpus_directory,
)
from memory_condense.head_memory import (
    AssociativeMemoryCandidate,
    CAVBank,
    HeadAssociationGraph,
    QwenMemoryLinker,
)
from memory_condense.experiment_rig import SweepQuestion, save_anchor_pack
from memory_condense.qwen_prefix import Qwen3PrefixEncoder


ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notes", type=Path, required=True)
parser.add_argument(
    "--source-pattern",
    default=(
        "Extracting_Attention_Heads|LLM_Attention_Head_Priming|"
        "Segmentation_Overlap_Strategy"
    ),
)
parser.add_argument("--source-families", type=int, default=3)
parser.add_argument("--questions-per-family", type=int, default=3)
parser.add_argument("--max-episodes-per-family", type=int, default=10)
parser.add_argument(
    "--selection-seed",
    help=(
        "Deterministically rank source families using metadata only before "
        "building questions; use for a preregistered untouched split"
    ),
)
parser.add_argument("--exclude-source-families", default="")
parser.add_argument("--min-source-turns", type=int, default=0)
parser.add_argument("--split-label", default="development-replay")
parser.add_argument(
    "--store-dir",
    type=Path,
    help="Persistent prepared store; default uses a temporary directory",
)
parser.add_argument(
    "--anchor-pack",
    type=Path,
    help="Write frozen hybrid anchors for model-free parallel sweeps",
)
parser.add_argument(
    "--skip-edge-prune",
    action="store_true",
    help="Keep the prepared store unpruned so sweep workers can vary degree",
)
parser.add_argument("--embedding-device", default="cuda")
parser.add_argument(
    "--model-dir", type=Path, default=ROOT / ".cache" / "models" / "Qwen3-8B"
)
parser.add_argument(
    "--cav-report",
    type=Path,
    default=ROOT / "eval_results" / "qwen3_prefix_cav_probe.json",
)
parser.add_argument(
    "--cav-vectors",
    type=Path,
    default=ROOT / "eval_results" / "qwen3_prefix_cav_probe.safetensors",
)
parser.add_argument("--link-candidates", type=int, default=3)
parser.add_argument("--links-per-memory", type=int, default=3)
parser.add_argument("--workspace-tokens", type=int, default=1024)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "eval_results" / "notes_linker_persistent_v2.json",
)
args = parser.parse_args()


def evenly_spaced(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    positions = {
        round(index * (len(items) - 1) / (count - 1)) for index in range(count)
    }
    return [items[position] for position in sorted(positions)]


inventory = load_corpus_directory(args.notes)
selection_record = {
    "split_label": args.split_label,
    "policy": "path_order",
    "seed": None,
    "excluded_families": [],
    "min_source_turns": args.min_source_turns,
}
if args.selection_seed:
    source_pattern = re.compile(args.source_pattern, re.IGNORECASE)
    excluded_families = {
        value.strip()
        for value in args.exclude_source_families.split(",")
        if value.strip()
    }
    grouped_sources: dict[str, list] = defaultdict(list)
    for source in inventory.sources:
        family = source.source_family or source.sha256[:12]
        if (
            source.kind == "conversation"
            and source_pattern.search(source.relative_path)
            and family not in excluded_families
        ):
            grouped_sources[family].append(source)
    representatives = [
        max(
            sources,
            key=lambda source: (
                len(source.turns),
                source.character_count,
                source.relative_path,
            ),
        )
        for sources in grouped_sources.values()
    ]
    representatives = [
        source
        for source in representatives
        if len(source.turns) >= args.min_source_turns
    ]
    representatives.sort(
        key=lambda source: hashlib.sha256(
            f"{args.selection_seed}:{source.source_family or source.sha256[:12]}".encode()
        ).hexdigest()
    )
    selected_families = {
        source.source_family or source.sha256[:12]
        for source in representatives[: args.source_families]
    }
    inventory = replace(
        inventory,
        sources=tuple(
            source
            for source in inventory.sources
            if (source.source_family or source.sha256[:12]) in selected_families
        ),
    )
    selection_record = {
        "split_label": args.split_label,
        "policy": "sha256(seed:source_family) over metadata-eligible families",
        "seed": args.selection_seed,
        "excluded_families": sorted(excluded_families),
        "min_source_turns": args.min_source_turns,
        "selected_families": sorted(selected_families),
    }
raw_slice = build_conversation_recall_slice(
    inventory,
    path_pattern=args.source_pattern,
    max_source_families=args.source_families,
    questions_per_family=args.questions_per_family,
)
if not raw_slice.questions:
    raise SystemExit("no eligible held-out questions matched the source selection")

# Bound cold-start work without dropping any gold answer. Distractor responses
# are sampled across each source's full timeline, then original order is kept.
gold_episode_ids = {question.gold_episode_id for question in raw_slice.questions}
episodes_by_family: dict[str, list] = defaultdict(list)
for episode in raw_slice.episodes:
    episodes_by_family[episode.source_family].append(episode)
selected_episode_ids: set[str] = set(gold_episode_ids)
for episodes in episodes_by_family.values():
    family_gold = [episode for episode in episodes if episode.episode_id in gold_episode_ids]
    capacity = max(0, args.max_episodes_per_family - len(family_gold))
    distractors = [
        episode for episode in episodes if episode.episode_id not in gold_episode_ids
    ]
    selected_episode_ids.update(
        episode.episode_id for episode in evenly_spaced(distractors, capacity)
    )
episodes = [
    episode for episode in raw_slice.episodes if episode.episode_id in selected_episode_ids
]

source_records = {
    source.relative_path: source.manifest_record()
    for source in inventory.sources
    if source.relative_path in raw_slice.source_paths
}
print("held-out source families:", flush=True)
for source_path in raw_slice.source_paths:
    source = source_records[source_path]
    print(
        f"  {source['source_family']}: {source_path} "
        f"({source['turn_count']} turns)",
        flush=True,
    )
print(
    f"selected episodes={len(episodes)} questions={len(raw_slice.questions)}",
    flush=True,
)

if args.store_dir is None:
    store_context = tempfile.TemporaryDirectory(prefix="memory-condense-notes-")
else:
    args.store_dir.mkdir(parents=True, exist_ok=True)
    if (args.store_dir / "memory.db").exists():
        raise SystemExit(
            f"refusing to reuse non-empty prepared store: {args.store_dir}"
        )
    store_context = nullcontext(str(args.store_dir))

with store_context as store_dir:
    condenser = MemoryCondenser(
        data_dir=store_dir,
        auto_extract=False,
        device=args.embedding_device,
    )
    chunks_by_episode: dict[str, list[str]] = defaultdict(list)
    chunk_episode: dict[str, str] = {}
    chunk_text: dict[str, str] = {}
    chunk_tokens: dict[str, int] = {}
    ingest_started = time.perf_counter()
    for episode in episodes:
        _, chunks = condenser.ingest("assistant", episode.text)
        for chunk in chunks:
            chunks_by_episode[episode.episode_id].append(chunk.chunk_id)
            chunk_episode[chunk.chunk_id] = episode.episode_id
            chunk_text[chunk.chunk_id] = chunk.text
            chunk_tokens[chunk.chunk_id] = chunk.token_count
    ingest_seconds = time.perf_counter() - ingest_started
    print(
        f"baseline index: {len(chunk_text)} chunks in {ingest_seconds:.1f}s",
        flush=True,
    )

    # Stage 1: use the cheap retriever to freeze write-time candidate IDs and
    # query-time baselines, then fully unload it before Qwen is materialized.
    write_candidates_by_chunk: dict[str, list[tuple[str, float]]] = {}
    linked_ids: set[str] = set()
    for chunk_id, text in chunk_text.items():
        candidates: list[tuple[str, float]] = []
        if linked_ids:
            ranked = condenser.search_hybrid(
                text,
                k=min(max(50, args.link_candidates * 8), len(chunk_text)),
            )
            for result in ranked:
                candidate_id = result.chunk.chunk_id
                if candidate_id not in linked_ids:
                    continue
                candidates.append((candidate_id, result.score))
                if len(candidates) >= args.link_candidates:
                    break
        write_candidates_by_chunk[chunk_id] = candidates
        linked_ids.add(chunk_id)
    condenser.close()
    del condenser
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    embedding_stage_cuda_bytes = int(torch.cuda.memory_allocated())
    print(
        f"embedding model unloaded; CUDA still allocated: "
        f"{embedding_stage_cuda_bytes:,} bytes",
        flush=True,
    )

    encoder = Qwen3PrefixEncoder(
        args.model_dir,
        layers=7,
        device="cuda",
        dtype="bfloat16",
    )
    cav_bank = CAVBank.load(
        args.cav_report,
        args.cav_vectors,
        layer=5,
        device="cuda",
    )
    linker = QwenMemoryLinker(
        encoder,
        layer=1,
        cav_bank=cav_bank,
        max_candidates=args.link_candidates,
        max_workspace_tokens=args.workspace_tokens,
        max_neighbors_per_episode=args.links_per_memory,
    )
    graph = HeadAssociationGraph()
    cav_names = tuple(cav_bank.names)
    cav_signatures: dict[str, tuple[float, ...]] = {}
    link_rows: list[dict] = []
    encoder._torch.cuda.reset_peak_memory_stats()
    link_started = time.perf_counter()
    for index, (chunk_id, text) in enumerate(chunk_text.items(), start=1):
        candidates = [
            AssociativeMemoryCandidate(
                candidate_id,
                chunk_text[candidate_id],
                score,
                "write_candidate",
            )
            for candidate_id, score in write_candidates_by_chunk[chunk_id]
        ]

        started = time.perf_counter()
        if candidates:
            # The hard token ceiling wins over candidate count. If a set of
            # long chunks does not fit, shed the weakest candidate and retry.
            while True:
                try:
                    result = linker.link_into_graph(
                        graph,
                        chunk_id,
                        text,
                        candidates,
                        top_k=min(args.links_per_memory, len(candidates)),
                    )
                    signature = result.source_cav_signature
                    workspace_tokens = result.workspace_tokens
                    retained_links = len(result.hits)
                    break
                except MemoryError:
                    candidates.pop()
                    if not candidates:
                        signature = linker.signature(text)
                        workspace_tokens = chunk_tokens[chunk_id]
                        retained_links = 0
                        break
        else:
            signature = linker.signature(text)
            workspace_tokens = chunk_tokens[chunk_id]
            retained_links = 0
        cav_signatures[chunk_id] = tuple(signature)
        link_rows.append(
            {
                "chunk_id": chunk_id,
                "candidate_count": len(candidates),
                "retained_links": retained_links,
                "workspace_tokens": workspace_tokens,
                "seconds": time.perf_counter() - started,
            }
        )
        if index % 20 == 0:
            print(f"  compiled links: {index}/{len(chunk_text)}", flush=True)
    link_seconds = time.perf_counter() - link_started
    peak_cuda_bytes = int(encoder._torch.cuda.max_memory_allocated())
    print(
        f"compiled {graph.edge_count} bounded edges and "
        f"{len(cav_signatures)} CAV signatures in {link_seconds:.1f}s",
        flush=True,
    )
    del linker, cav_bank, encoder
    gc.collect()
    torch.cuda.empty_cache()
    post_link_cuda_bytes = int(torch.cuda.memory_allocated())
    print(
        f"Qwen unloaded; CUDA still allocated: {post_link_cuda_bytes:,} bytes",
        flush=True,
    )

    # Stage 3: persist only compact artifacts after Qwen has been unloaded.
    # Then close and reopen the store before retrieval so this benchmark tests
    # process-boundary behavior, not Python objects left over from compilation.
    cav_probe_report = json.loads(args.cav_report.read_text(encoding="utf-8"))
    artifact = AssociationArtifact.create(
        model_id=cav_probe_report["model"],
        checkpoint_id=(
            f"{cav_probe_report['model']}:model-00001-of-00005.safetensors"
        ),
        prefix_layers=7,
        head_layer=1,
        cav_layer=5,
        concept_names=cav_names,
        head_count=32,
        metadata={
            "cav_dataset_sha256": cav_probe_report["dataset_sha256"],
            "runtime_dtype": "bfloat16",
            "max_workspace_tokens": args.workspace_tokens,
        },
    )
    writer = MemoryCondenser(
        data_dir=store_dir,
        auto_extract=False,
        device=args.embedding_device,
    )
    writer.associations.register_artifact(artifact)
    for chunk_id, signature in cav_signatures.items():
        writer.associations.put_signature(
            chunk_id,
            artifact.artifact_id,
            signature,
        )
    for edge in graph.edges():
        writer.associations.upsert_edge(
            edge.source_id,
            edge.destination_id,
            artifact.artifact_id,
            edge.head_weights.tolist(),
            qk_score=edge.score,
            ov_transport=edge.ov_transport,
            evidence_count=edge.evidence_count,
            temporal_forward=edge.temporal_forward,
        )
    persisted_stats = writer.associations.stats(artifact.artifact_id)
    writer.close()
    del writer, graph, cav_signatures
    gc.collect()

    condenser = MemoryCondenser(
        data_dir=store_dir,
        auto_extract=False,
        device=args.embedding_device,
    )
    reopened_stats = condenser.associations.stats(artifact.artifact_id)
    restart_verified = reopened_stats == persisted_stats
    if not restart_verified:
        raise RuntimeError(
            f"association artifacts changed across restart: "
            f"{persisted_stats!r} != {reopened_stats!r}"
        )

    baseline_started = time.perf_counter()
    baseline_batches = condenser.search_hybrid_many(
        [question.question for question in raw_slice.questions],
        k=10,
    )
    baseline_results_by_question = {
        question.question_id: results
        for question, results in zip(
            raw_slice.questions, baseline_batches, strict=True
        )
    }
    baseline_query_seconds = time.perf_counter() - baseline_started
    anchor_pack_path = args.anchor_pack
    if anchor_pack_path is None and args.store_dir is not None:
        anchor_pack_path = args.store_dir.parent / "anchor_pack.json"
    anchor_pack_record = None
    if anchor_pack_path is not None:
        anchor_questions = [
            SweepQuestion(
                question_id=question.question_id,
                source_family=question.source_family,
                question=question.question,
                gold_chunk_ids=tuple(
                    chunks_by_episode[question.gold_episode_id]
                ),
                anchors=tuple(
                    baseline_results_by_question[question.question_id]
                ),
            )
            for question in raw_slice.questions
        ]
        anchor_pack_record = save_anchor_pack(
            anchor_pack_path,
            anchor_questions,
            metadata={
                "artifact_id": artifact.artifact_id,
                "selection": selection_record,
                "max_k": 10,
            },
        )

    def evaluate(*, top_k: int, association_slots: int) -> list[dict]:
        rows: list[dict] = []
        for question in raw_slice.questions:
            gold_ids = set(chunks_by_episode[question.gold_episode_id])
            baseline = baseline_results_by_question[question.question_id][:top_k]
            composed = condenser.expand_associative(
                baseline,
                artifact.artifact_id,
                k=top_k,
                association_slots=association_slots,
                qk_reserve=1,
                neighbors_per_anchor=args.links_per_memory,
                cav_candidates=50,
                touch=False,
            )
            baseline_ids = [result.chunk.chunk_id for result in baseline]
            composed_ids = [result.chunk.chunk_id for result in composed]
            routes = [result.route for result in composed]
            unique_anchor_content = {
                " ".join(result.chunk.text.split()).casefold() for result in baseline
            }
            rows.append(
                {
                    "question_id": question.question_id,
                    "source_family": question.source_family,
                    "gold_episode_id": question.gold_episode_id,
                    "baseline_hit": bool(gold_ids.intersection(baseline_ids)),
                    "composed_hit": bool(gold_ids.intersection(composed_ids)),
                    "baseline_tokens": sum(
                        result.chunk.token_count for result in baseline
                    ),
                    "composed_tokens": sum(
                        result.chunk.token_count for result in composed
                    ),
                    "duplicates_removed": len(baseline)
                    - len(unique_anchor_content),
                    "qk_ov_links_added": routes.count("qk"),
                    "cav_links_added": routes.count("cav"),
                    "anchors_displaced": len(set(baseline_ids) - set(composed_ids)),
                    "routes": routes,
                }
            )
        return rows

    rows = evaluate(top_k=10, association_slots=0)
    budget_rows = {
        "k3_hybrid": evaluate(top_k=3, association_slots=0),
        "k3_linked_1": evaluate(top_k=3, association_slots=1),
        "k5_hybrid": evaluate(top_k=5, association_slots=0),
        "k5_linked_2": evaluate(top_k=5, association_slots=2),
    }
    edge_count_before_prune = reopened_stats["edges"]
    pruned_degree = max(1, args.links_per_memory - 1)
    if args.skip_edge_prune:
        edges_pruned = 0
        pruned_stats = reopened_stats
        pruned_rows = rows
        pruned_budget_rows = {
            "k3_linked_1": budget_rows["k3_linked_1"],
            "k5_linked_2": budget_rows["k5_linked_2"],
        }
    else:
        edges_pruned = condenser.associations.prune_edges(
            artifact.artifact_id,
            pruned_degree,
        )
        pruned_stats = condenser.associations.stats(artifact.artifact_id)
        pruned_rows = evaluate(top_k=10, association_slots=0)
        pruned_budget_rows = {
            "k3_linked_1": evaluate(top_k=3, association_slots=1),
            "k5_linked_2": evaluate(top_k=5, association_slots=2),
        }
    condenser.close()


def mean(rows: list[dict], key: str) -> float:
    return float(statistics.mean(row[key] for row in rows))


def arm_metrics(rows: list[dict], *, linked: bool) -> dict:
    prefix = "composed" if linked else "baseline"
    return {
        "recall": mean(rows, f"{prefix}_hit"),
        "mean_context_tokens": mean(rows, f"{prefix}_tokens"),
        "qk_ov_links_added": (
            sum(row["qk_ov_links_added"] for row in rows) if linked else 0
        ),
        "cav_links_added": sum(row["cav_links_added"] for row in rows) if linked else 0,
        "anchors_displaced": (
            sum(row["anchors_displaced"] for row in rows) if linked else 0
        ),
    }


report = {
    "protocol": (
        f"notes linker persistent; split={args.split_label}; association artifacts "
        "scored through the production path after close/reopen"
    ),
    "architecture": (
        "Qwen prefix runs only during bounded memory writes; token K/V is "
        "discarded; retrieval uses persisted sparse QK/OV edges and CAV signatures"
    ),
    "selection": selection_record,
    "sources": [source_records[path] for path in raw_slice.source_paths],
    "corpus": {
        "assistant_episodes": len(episodes),
        "chunks": len(chunk_text),
        "questions": len(raw_slice.questions),
        "baseline_ingest_seconds": ingest_seconds,
        "batched_baseline_query_seconds": baseline_query_seconds,
    },
    "link_compilation": {
        "seconds": link_seconds,
        "mean_seconds_per_chunk": statistics.mean(row["seconds"] for row in link_rows),
        "max_workspace_tokens": max(row["workspace_tokens"] for row in link_rows),
        "mean_workspace_candidates": statistics.mean(
            row["candidate_count"] for row in link_rows
        ),
        "peak_cuda_bytes_including_fixed_prefix": peak_cuda_bytes,
        "cuda_bytes_after_embedding_model_unload": embedding_stage_cuda_bytes,
        "cuda_bytes_after_qwen_unload": post_link_cuda_bytes,
        "retained_token_kv_bytes": 0,
        "cav_signature_bytes": persisted_stats["cav_payload_bytes"],
        "directed_edges": edge_count_before_prune,
        "edge_head_weight_bytes": persisted_stats["head_payload_bytes"],
    },
    "external_persistence": {
        "backend": "sqlite",
        "artifact_id": artifact.artifact_id,
        "restart_verified": restart_verified,
        "before_prune": persisted_stats,
        "after_prune": pruned_stats,
        "prepared_store": None if args.store_dir is None else str(args.store_dir),
        "anchor_pack": None if anchor_pack_path is None else str(anchor_pack_path),
        "anchor_pack_sha256": (
            None if anchor_pack_record is None else anchor_pack_record["sha256"]
        ),
    },
    "baseline": {
        "recall": mean(rows, "baseline_hit"),
        "mean_context_tokens": mean(rows, "baseline_tokens"),
    },
    "composed": {
        "recall": mean(rows, "composed_hit"),
        "mean_context_tokens": mean(rows, "composed_tokens"),
        "duplicates_removed": sum(row["duplicates_removed"] for row in rows),
        "qk_ov_links_added": sum(row["qk_ov_links_added"] for row in rows),
        "cav_links_added": sum(row["cav_links_added"] for row in rows),
    },
    "fixed_item_budget_arms": {
        "hybrid_k3": arm_metrics(budget_rows["k3_hybrid"], linked=False),
        "linked_k3_one_reserved": arm_metrics(
            budget_rows["k3_linked_1"], linked=True
        ),
        "hybrid_k5": arm_metrics(budget_rows["k5_hybrid"], linked=False),
        "linked_k5_two_reserved": arm_metrics(
            budget_rows["k5_linked_2"], linked=True
        ),
    },
    "edge_pruning": {
        "skipped": args.skip_edge_prune,
        "degree_before": args.links_per_memory,
        "degree_after": pruned_degree,
        "edges_before": edge_count_before_prune,
        "edges_removed": edges_pruned,
        "edges_after": pruned_stats["edges"],
        "recall_after": mean(pruned_rows, "composed_hit"),
        "mean_context_tokens_after": mean(pruned_rows, "composed_tokens"),
        "fixed_item_budget_arms_after": {
            "linked_k3_one_reserved": arm_metrics(
                pruned_budget_rows["k3_linked_1"], linked=True
            ),
            "linked_k5_two_reserved": arm_metrics(
                pruned_budget_rows["k5_linked_2"], linked=True
            ),
        },
    },
    "rows": rows,
    "fixed_item_budget_rows": budget_rows,
    "rows_after_edge_prune": pruned_rows,
    "fixed_item_budget_rows_after_edge_prune": pruned_budget_rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"baseline recall: {report['baseline']['recall']:.1%}")
print(f"linked recall: {report['composed']['recall']:.1%}")
print(
    f"mean context tokens: {report['baseline']['mean_context_tokens']:.0f} -> "
    f"{report['composed']['mean_context_tokens']:.0f}"
)
print(
    f"edge prune: {edge_count_before_prune} -> {pruned_stats['edges']}, "
    f"recall {report['edge_pruning']['recall_after']:.1%}"
)
for name, metrics in report["fixed_item_budget_arms"].items():
    print(
        f"{name}: recall={metrics['recall']:.1%} "
        f"tokens={metrics['mean_context_tokens']:.0f}"
    )
print(f"report: {args.output}")
