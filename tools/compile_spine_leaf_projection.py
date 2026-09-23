"""Compile the complete attention-partitioned leaf index used by live retrieval.

Parent summary generation is deferred because neither measured query arm uses
it. The complete raw partition, user-spine attention cuts and leaf merge policy
are retained. Cache snapshots are authenticated once and frozen before new I/O.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from tools.build_spine_corpus_hierarchy import GATEWAY, MODEL, MergeJournal, compile_waves, restore_request
from tools.build_spine_corpus_hierarchy_resilient import (
    IMPLEMENTATION as BASE_IMPLEMENTATION, NeedsProviderWork, RecoveryJournal, ReusingAttentionCache, recoverable_slots,
)
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.spine_leaf_projection import project_source_leaves


IMPLEMENTATION = (*BASE_IMPLEMENTATION, "tools/spine_leaf_projection.py", "tools/compile_spine_leaf_projection.py")


def frozen_cache(root, atoms, parent_roots, source_snapshot=None):
    path = root / "input-cache.json"
    parents = [(p, read_sealed_json(p / "preflight.json")) for p in parent_roots]
    for parent_root, preflight in parents:
        p = preflight.payload
        if (parent_root.resolve() == root.resolve() or p["atoms_sha256"] != atoms.sha256 or
            not p["complete_namespace"] or p["model"] != MODEL or p["gateway"] != GATEWAY or
            p["raw_inputs_to_qwen"] is not False or p["max_channel_tokens"] != 128):
            raise ValueError("input cache must bind the same complete summary-only corpus and model")
        for name, sha in p["implementation"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha:
                continue
            preserved = source_snapshot / name if source_snapshot is not None else None
            if preserved is None or not preserved.is_file() or hashlib.sha256(preserved.read_bytes()).hexdigest() != sha:
                raise ValueError("input cache implementation changed without matching preserved source bytes")
    bindings = [{"root": str(p.resolve()), "preflight_sha256": a.sha256,
                 "source_snapshot": str(source_snapshot.resolve()) if source_snapshot else None} for p, a in parents]
    if path.exists():
        artifact = read_sealed_json(path)
        if artifact.payload["atoms_sha256"] != atoms.sha256 or artifact.payload["parents"] != bindings:
            raise ValueError("frozen input cache binding changed")
        return artifact, parents
    values, receipts = {}, []
    def admit(job, value):
        old = values.setdefault(job.prompt_sha256, value)
        if old != value:
            raise ValueError("completed parent caches disagree on a summary")
    for parent_root, preflight in parents:
        provenance = preflight.payload.get("parent_caches", {})
        if isinstance(provenance, dict) and "input_snapshot_sha256" in provenance:
            inherited = read_sealed_json(parent_root / "input-cache.json")
            if inherited.sha256 != provenance["input_snapshot_sha256"] or inherited.payload["atoms_sha256"] != atoms.sha256:
                raise ValueError("inherited frozen summary cache changed")
            for key, value in inherited.payload["summaries"].items():
                if values.setdefault(key, value) != value:
                    raise ValueError("inherited summary caches disagree")
            receipts.append({"inherited_cache_sha256": inherited.sha256})
        journal = MergeJournal(parent_root, preflight, False, 0)
        for request_path in sorted((parent_root / "requests").glob("*.json")):
            request = read_sealed_json(request_path)
            if request.payload["preflight_sha256"] != preflight.sha256:
                raise ValueError("parent request binding changed")
            checkpoint = parent_root / "checkpoints" / request.sha256
            # Snapshot completed responses only; never open or retry live calls.
            if not list(checkpoint.glob("*.response.json")):
                continue
            jobs = tuple(restore_request(row) for row in request.payload["jobs"])
            from memory_condense.search.spine_merge_batch import merge_batch_messages
            if merge_batch_messages(jobs) != request.payload["messages"]:
                raise ValueError("parent summary prompt changed")
            runtime = journal.runtime(request)
            try:
                batch = runtime.run()
            finally:
                runtime.close()
            summaries, _ = recoverable_slots(batch.logical_completions[0], jobs)
            for job, summary in zip(jobs, summaries, strict=True):
                if summary is not None:
                    admit(job, summary)
            receipts.append({"request_sha256": request.sha256,
                "response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records]})
        # Existing single-job recoveries have their own authenticated journals.
        recovery = RecoveryJournal(parent_root, preflight, False, 0)
        seen = set()
        for request_path in sorted((parent_root / "repair-requests").glob("*.json")):
            request = read_sealed_json(request_path)
            p = request.payload
            key = (p["original_job_sha256"], p["failed_response_sha256"], p["slot"])
            if key in seen:
                continue
            seen.add(key)
            if p["preflight_sha256"] != preflight.sha256:
                raise ValueError("parent repair binding changed")
            # Only replay repair groups with a completed accepted receipt.
            accepted = any(read_sealed_json(v).payload.get("original_job_sha256") == key[0]
                           for v in (parent_root / "repair-validation").glob("*.json"))
            if not accepted:
                continue
            job = restore_request(p["job"])
            admit(job, recovery.recover(job, key[1], key[2]))
        receipts.extend({"repair_validation_sha256": sha} for sha in sorted(set(recovery.recoveries)))
    artifact, _ = publish_sealed_json(path, {"atoms_sha256": atoms.sha256, "parents": bindings,
        "summaries": values, "authenticated_inputs": receipts, "new_calls": 0})
    return artifact, parents


def validate_leaf_partition(parts, expected_sha):
    ordered = tuple(s for part in parts.values() for s in part[0])
    spans = [span.receipt_sha256 for section in ordered for span in section.spans]
    if identity_sha256(spans) != expected_sha:
        raise ValueError("leaf index lost or reordered exact raw fragments")
    # The search index sorts IDs for deterministic addressing. Check transcript
    # order before constructing it; index order is not transcript order.
    return SectionSummaryIndex(ordered)


def run(atoms_path, root, parent_roots, enable=False, budget=0, source_snapshot=None):
    atoms = read_sealed_json(atoms_path)
    data = atoms.payload
    if data.get("format") != "memory-condense-spine-source-bound-atoms-v1" or not data["complete_namespace"]:
        raise ValueError("leaf projection requires a complete source-bound namespace")
    rows = tuple(SectionSummary.from_dict(row) for row in data["atoms"])
    if identity_sha256([a.spans[0].receipt_sha256 for a in rows]) != data["raw_span_population_sha256"]:
        raise ValueError("complete atomic partition changed")
    cache, parents = frozen_cache(root, atoms, parent_roots, source_snapshot)
    method = {"format": "memory-condense-complete-spine-leaf-projection-v1", "model": MODEL, "gateway": GATEWAY,
        "max_channel_tokens": 128, "leaf_token_cap": 512, "max_leaf_exchanges": 2, "attention_window_exchanges": 8,
        "max_jobs_per_batch": 8, "max_concurrency": 4, "max_recovery_calls_per_failed_job": 2,
        "recovery_words": [48, 24], "retries": 0, "raw_inputs_to_qwen": False, "query_independent": True,
        "parent_summary_generation": "deferred; query arms use only exact attention-partitioned leaves",
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}
    preflight, _ = publish_sealed_json(root / "preflight.json", {**method,
        "method_policy_sha256": identity_sha256(method), "atoms_sha256": atoms.sha256,
        "complete_namespace": True, "raw_span_population_sha256": data["raw_span_population_sha256"],
        "parent_caches": {"input_snapshot_sha256": cache.sha256, "bindings": cache.payload["parents"]}})
    journal = RecoveryJournal(root, preflight, enable, budget)
    journal.cache.values.update(cache.payload["summaries"])
    groups = {}
    for atom in rows:
        groups.setdefault(atom.source_id, []).append(atom)
    try:
        journal.replay()
        exchanges = compile_waves(groups, lambda values: compile_user_spine_exchanges(values,
            summarize=journal.cache, summarizer_identity=preflight.sha256, max_channel_tokens=128), journal, "exchanges")
        scorer = ReusingAttentionCache(root, preflight, parents)
        parts = compile_waves(exchanges, lambda values: project_source_leaves(values, scorer=scorer,
            summarize=journal.cache, summarizer_identity=preflight.sha256), journal, "leaves")
    except NeedsProviderWork:
        print({"preflight_sha256": preflight.sha256, "status": "next_leaf_work_prepared",
            "new_calls": journal.calls, "scheduled_calls": journal.scheduled_calls, "replay_hits": journal.hits}, flush=True)
        return
    index = validate_leaf_partition(parts, data["raw_span_population_sha256"])
    artifact, _ = publish_sealed_json(root / "hierarchy.json", {"preflight_sha256": preflight.sha256,
        "atoms_sha256": atoms.sha256, "complete_namespace": True, "index_json": index.to_json(),
        "raw_span_population_sha256": data["raw_span_population_sha256"], "raw_inputs_to_qwen": False,
        "parent_summary_compilation_complete": False, "leaf_projection_complete": True,
        "source_cuts": {source: part[1] for source, part in parts.items()},
        "source_attention_receipts": {source: part[2] for source, part in parts.items()},
        "summary_merge_jobs": len(journal.cache.values), "recovery_receipts": sorted(set(journal.recoveries)),
        "target_gate_passed": False})
    print({"hierarchy_sha256": artifact.sha256, "leaf_count": len(index.sections), "source_count": len(parts),
        "complete_namespace": True, "parent_summaries_deferred": True,
        "new_calls": journal.calls, "replay_hits": journal.hits}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, action="append", default=[])
    parser.add_argument("--parent-source-snapshot", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    if args.max_new_calls < 0 or (args.enable_provider and args.phase != "run"):
        parser.error("provider execution requires run and a nonnegative allowance")
    run(args.atoms, args.output_root, args.parent_cache_root, args.enable_provider, args.max_new_calls,
        args.parent_source_snapshot)
