"""Compile native hierarchies with separate exchange and parent summary budgets."""
import argparse
from dataclasses import asdict
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.parent_budgeted_spine_hierarchy import build_parent_budgeted_spine_hierarchy
from memory_condense.search.native_spine_merges import neutral_key
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tools import compile_native_spine_exchanges as exchange_compiler
from tools import compile_native_spine_hierarchy as original
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json
from tools.native_qwen_spine_backend import NativeQwenBackend
from tools.run_hot_reduced30_answer_judge import _phase_lock


def implementation():
    return {**original.implementation(), **{name: digest(name) for name in (
        "src/memory_condense/search/episodes/parent_budgeted_spine_hierarchy.py",
        "tools/compile_native_spine_parent_budgets.py",
    )}}


def compile_groups(root, preflight, groups, scorer, journal):
    summarize = ReusingSpineSummarizer(journal.cache)
    done = {}
    while len(done) < len(groups):
        pending = {}
        for sha, (body, atom_input, atoms, exchanges) in groups.items():
            if sha in done:
                continue
            try:
                hierarchy = build_parent_budgeted_spine_hierarchy(exchanges, scorer=scorer,
                    summarize=summarize, summarizer_identity=preflight.sha256,
                    leaf_token_cap=512, max_leaf_exchanges=2, max_exchange_channel_tokens=128,
                    max_parent_channel_tokens=512, window_exchange_cap=8, max_prompt_tokens=2048)
            except PendingMerge as missing:
                pending.setdefault(neutral_key(missing.request), missing.request)
                continue
            index, atomic_index = hierarchy.summary_index(), SectionSummaryIndex(atoms)
            by_id = {s.section_id: s for s in index.sections}
            expected = tuple(s for atom in atoms for s in atom.spans)
            if (len(hierarchy.root_section_ids) != 1
                    or by_id[hierarchy.root_section_ids[0]].spans != expected):
                raise ValueError("parent-budgeted hierarchy changed original raw coverage")
            leaves = tuple(s for s in index.sections if not s.child_section_ids)
            artifact, _ = publish_sealed_json(root / "hierarchies" / f"{sha}.json", {
                "preflight_sha256": preflight.sha256, "body_sha256": sha,
                "exchanges_sha256": body.sha256, "atomic_input_sha256": atom_input.sha256,
                "index_json": index.to_json(), "atomic_index_json": atomic_index.to_json(),
                "root_section_ids": list(hierarchy.root_section_ids),
                "leaf_count": len(leaves), "parent_count": len(index.sections)-len(leaves),
                "atomic_count": len(atoms), "exchange_count": len(exchanges),
                "raw_span_population_sha256": identity_sha256([s.receipt_sha256 for s in expected]),
                "splits": [asdict(s) for s in hierarchy.splits],
                "attention_receipts": [w.signal.receipt_sha256 for w in hierarchy.windows],
                "oversized_exchange_ids": list(hierarchy.oversized_exchange_ids),
                "original_atomic_addresses_preserved": True, "raw_inputs_to_qwen": False,
            })
            done[sha] = {"path": str(artifact.path.relative_to(root)), "sha256": artifact.sha256,
                         "leaf_count": len(leaves), "parent_count": len(index.sections)-len(leaves),
                         "atomic_count": len(atoms)}
        print({"complete_body_hierarchies": len(done), "first_pending_merge_jobs": len(pending)}, flush=True)
        if not pending:
            break
        before = len(journal.cache.values)
        resolved = journal.resolve(pending)
        if not resolved and len(journal.cache.values) == before:
            break
    return done


def execute(root, exchange_root, attention_root, backend, budget=0):
    if type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError("parent-budgeted invocation allows at most 128 new local jobs")
    with _phase_lock(root, "native-parent-budgeted-compilation"):
        scorer = original.FrozenAttention(attention_root)
        exchanges, groups, seed = original.load_groups(exchange_root, scorer, backend)
        preflight, _ = publish_sealed_json(root / "preflight.json", {
            "format": "native-spine-parent-budgeted-hierarchy-v1",
            "exchange_root": str(exchange_root.resolve()), "attention_root": str(attention_root.resolve()),
            "exchange_result_sha256": exchanges.sha256, "attention_result_sha256": scorer.result.sha256,
            "attention_method_sha256": scorer.method.sha256,
            "backend": backend.identity, "backend_sha256": backend.identity_sha256,
            "prepared_body_count": len(groups), "leaf_token_cap": 512, "max_leaf_exchanges": 2,
            "max_exchange_channel_tokens": 128, "max_parent_channel_tokens": 512,
            "window_exchange_cap": 8, "max_prompt_tokens": 2048,
            "maximum_new_local_jobs_per_invocation": 128, "maximum_recovery_attempts": 2,
            "automatic_retries": 0, "raw_inputs_to_qwen": False,
            "timestamp_metadata_in_model_inputs": False, "original_atomic_addresses_preserved": True,
            "merge_reuse_policy": "same exact neutral request only; no cross-budget key substitution",
            "implementation": implementation(),
        })
        journal = exchange_compiler.NeutralJournal(root, preflight, backend, budget)
        journal.cache.values.update(seed)
        journal.replay()
        done = compile_groups(root, preflight, groups, scorer, journal)
        payload = {
            "preflight_sha256": preflight.sha256, "body_count": len(done), "prepared_body_count": len(groups),
            "compiled_bodies": [done[sha] for sha in sorted(done)],
            "leaf_count": sum(row["leaf_count"] for row in done.values()),
            "parent_count": sum(row["parent_count"] for row in done.values()),
            "atomic_count": sum(row["atomic_count"] for row in done.values()),
            "complete_available_body_hierarchies": len(done) == len(groups),
            "complete_source_compilation": exchanges.payload["complete_source_compilation"],
            "complete_native_hierarchies": len(done) == len(groups) and exchanges.payload["complete_source_compilation"],
            "original_atomic_addresses_preserved": True, "raw_inputs_to_qwen": False,
            "new_attention_passes": 0, "remote_provider_calls": 0, "full100_target_passed": False,
        }
        name = "result.json" if len(done) == len(groups) else f"partial-{identity_sha256(payload)}.json"
        result, _ = publish_sealed_json(root / name, payload)
        print({"hierarchy_result_sha256": result.sha256, "complete_body_hierarchies": len(done),
               "new_local_jobs": journal.jobs, "new_local_batches": journal.calls}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exchange-root", type=Path, required=True)
    parser.add_argument("--attention-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args()
    backend = NativeQwenBackend(Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"),
        Path(".cache/local-qwen-runtime/site-packages"), Path("../../.cache/models/Qwen3-8B"))
    execute(args.root, args.exchange_root, args.attention_root, backend, args.budget)
