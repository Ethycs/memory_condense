"""Restore the saved attention tree and summarize its missing source parents.

This successor preserves the evaluated leaves, vectors and raw stores. It has
no question/reference reader or raw-text loader. Completed summary jobs replay.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_parent_hierarchy import SourceSpineParentPlan
from tools.build_spine_corpus_hierarchy import GATEWAY, MODEL, compile_waves
from tools.build_spine_corpus_hierarchy_resilient import (
    IMPLEMENTATION as BASE_IMPLEMENTATION, NeedsProviderWork, RecoveryJournal,
)
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


IMPLEMENTATION = (*BASE_IMPLEMENTATION, "tools/restore_spine_parent_hierarchy.py",
    "src/memory_condense/search/spine_parent_hierarchy.py",
    "src/memory_condense/search/section_summary.py")


def prepare_sources(leaf, atoms, manifest):
    p, a, m = leaf.payload, atoms.payload, manifest.payload
    if (p.get("leaf_projection_complete") is not True or p.get("complete_namespace") is not True
        or p.get("raw_inputs_to_qwen") is not False or a.get("complete_namespace") is not True
        or m.get("complete_namespace") is not True or p["atoms_sha256"] != atoms.sha256
        or m["hierarchy_sha256"] != leaf.sha256
        or len({p["raw_span_population_sha256"], a["raw_span_population_sha256"],
                m["raw_span_population_sha256"]}) != 1):
        raise ValueError("parent restoration requires the same complete evaluated leaf projection")
    index = SectionSummaryIndex.from_json(p["index_json"])
    if index.to_json() != m["index_json"] or any(s.child_section_ids for s in index.sections):
        raise ValueError("serving leaves changed or already contain parents")
    rows = tuple(SectionSummary.from_dict(row) for row in a["atoms"])
    if any(len(row.spans) != 1 or row.child_section_ids for row in rows):
        raise ValueError("expected single-fragment input atoms")
    if identity_sha256([row.spans[0].receipt_sha256 for row in rows]) != p["raw_span_population_sha256"]:
        raise ValueError("ordered atomic source partition changed")
    sources, leaves = {}, {}
    for row in rows:
        sources.setdefault(row.source_id, []).append(row.spans[0])
    for section in index.sections:
        leaves.setdefault(section.source_id, []).append(section)
    if set(sources) != set(leaves) or set(sources) != set(p["source_cuts"]) or set(sources) != set(p["source_attention_receipts"]):
        raise ValueError("attention, cuts and source populations disagree")
    if any(not p["source_attention_receipts"][sid] for sid in sources):
        raise ValueError("a source has no saved attention receipt")
    plans = {sid: SourceSpineParentPlan(leaves[sid], spans, p["source_cuts"][sid])
             for sid, spans in sources.items()}
    return index, plans


def run(leaf_root, atoms_path, index_root, root, enable=False, budget=0):
    leaf = read_sealed_json(leaf_root / "hierarchy.json")
    projection = read_sealed_json(leaf_root / "preflight.json")
    atoms = read_sealed_json(atoms_path)
    manifest = read_sealed_json(index_root / "index.json")
    if projection.sha256 != leaf.payload["preflight_sha256"]:
        raise ValueError("saved attention projection binding changed")
    index, plans = prepare_sources(leaf, atoms, manifest)
    method = {"format": "memory-condense-restored-spine-parents-v1", "model": MODEL, "gateway": GATEWAY,
        "max_channel_tokens": 128, "max_jobs_per_batch": 8, "max_concurrency": 4,
        "retries": 0, "max_recovery_calls_per_failed_job": 2, "recovery_words": [48, 24],
        "raw_inputs_to_qwen": False, "query_independent": True,
        "topology": "restore every authenticated source-local attention cut",
        "leaf_policy": "preserve original leaf descriptors byte for byte",
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}
    preflight, _ = publish_sealed_json(root / "preflight.json", {**method,
        "method_policy_sha256": identity_sha256(method), "leaf_projection_sha256": leaf.sha256,
        "leaf_projection_preflight_sha256": projection.sha256, "atoms_sha256": atoms.sha256,
        "serving_index_sha256": manifest.sha256, "leaf_index_sha256": index.receipt_sha256,
        "raw_span_population_sha256": leaf.payload["raw_span_population_sha256"],
        "raw_token_proxy": manifest.payload["raw_token_proxy"], "complete_namespace": True,
        "leaf_root": str(leaf_root.resolve()), "atoms_path": str(atoms_path.resolve()),
        "index_root": str(index_root.resolve())})
    topology, _ = publish_sealed_json(root / "topology.json", {
        "preflight_sha256": preflight.sha256, "leaf_count": len(index.sections),
        "parent_count": sum(len(plan.parents) for plan in plans.values()),
        "sources": [{"source_id": sid, "root_section_id": plan.root_section_id,
                     "plan_sha256": plan.receipt_sha256, "parent_count": len(plan.parents)}
                    for sid, plan in plans.items()], "raw_text_reads": 0, "new_attention_calls": 0})
    print({"preflight_sha256": preflight.sha256, "leaf_count": len(index.sections),
           "parent_count": topology.payload["parent_count"], "sources": len(plans)}, flush=True)
    journal = RecoveryJournal(root, preflight, enable, budget)
    try:
        journal.replay()
        parts = compile_waves(plans, lambda plan: plan.compile(summarize=journal.cache,
            summarizer_identity=preflight.sha256), journal, "source_parents")
    except NeedsProviderWork:
        print({"status": "parent_summary_work_prepared", "new_calls": journal.calls,
               "scheduled_calls": journal.scheduled_calls, "replay_hits": journal.hits}, flush=True)
        return
    restored = SectionSummaryIndex(tuple(s for part in parts.values() for s in part.sections))
    if tuple(s for s in restored.sections if not s.child_section_ids) != index.sections:
        raise ValueError("restored hierarchy changed the evaluated leaves")
    artifact, _ = publish_sealed_json(root / "hierarchy.json", {
        "preflight_sha256": preflight.sha256, "topology_sha256": topology.sha256,
        "leaf_projection_sha256": leaf.sha256, "leaf_index_sha256": index.receipt_sha256,
        "index_json": restored.to_json(), "root_section_ids": [plan.root_section_id for plan in plans.values()],
        "leaf_count": len(index.sections), "parent_count": topology.payload["parent_count"],
        "raw_span_population_sha256": leaf.payload["raw_span_population_sha256"],
        "complete_namespace": True, "parent_summary_compilation_complete": True,
        "raw_inputs_to_qwen": False, "summary_merge_jobs": len(journal.cache.values),
        "target_gate_passed": False})
    print({"status": "complete_parent_hierarchy", "hierarchy_sha256": artifact.sha256,
           "parents": topology.payload["parent_count"], "new_calls": journal.calls,
           "replay_hits": journal.hits}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--leaf-root", type=Path, required=True)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    if args.max_new_calls < 0 or (args.enable_provider and args.phase != "run"):
        parser.error("provider execution requires run and a nonnegative allowance")
    run(args.leaf_root, args.atoms, args.index_root, args.output_root,
        args.enable_provider, args.max_new_calls)
