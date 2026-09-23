"""Measure parent reuse on completed exchanges without any new model calls."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.parent_budgeted_spine_hierarchy import build_parent_budgeted_spine_hierarchy
from memory_condense.search.episodes.user_spine_hierarchy import build_user_spine_hierarchy
from memory_condense.search.native_spine_merges import NeutralMergeCache, neutral_key
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_hierarchy import FrozenAttention, implementation as hierarchy_implementation, load_groups
from tools.matched_eval.artifacts import publish_sealed_json
from tools.native_qwen_spine_backend import NativeQwenBackend
from tools.run_hot_reduced30_answer_judge import _phase_lock


def implementation():
    return {**hierarchy_implementation(), **{name: digest(name) for name in (
        "src/memory_condense/search/episodes/parent_budgeted_spine_hierarchy.py",
        "tools/assess_native_spine_parent_budgets.py",
    )}}


class RecordingAttention:
    """Record the original cached inputs without changing advertised capacity."""
    def __init__(self, frozen):
        self.frozen = frozen
        self.max_spans, self.span_token_cap = frozen.max_spans, frozen.span_token_cap
        self.calls = []

    def score_sequence(self, texts):
        texts = tuple(texts)
        signal = self.frozen.score_sequence(texts)
        self.calls.append({"texts_sha256": identity_sha256(list(texts)),
                           "signal_receipt_sha256": signal.receipt.receipt_sha256})
        return signal


def structure(hierarchy):
    """Exclude summary text: these fields prove unchanged cuts and raw addresses."""
    return {
        "roots": list(hierarchy.root_section_ids),
        "sections": [{"section_id": s.section_id, "children": list(s.child_section_ids),
                      "raw_spans": [span.receipt_sha256 for span in s.spans]} for s in hierarchy.sections],
        "windows": [w.receipt_sha256 for w in hierarchy.windows],
        "splits": [asdict(s) for s in hierarchy.splits],
        "oversized_exchange_ids": list(hierarchy.oversized_exchange_ids),
    }


def forbidden(*args, **kwargs):
    raise AssertionError("this assessment forbids model loading and generation")


def assess(root, exchange_root, attention_root, backend):
    # Only the already-complete exchange journal is replayed, never a live parent journal.
    backend.generate = backend.load = forbidden
    with _phase_lock(root, "native-parent-budget-assessment"):
        frozen = FrozenAttention(attention_root)
        replay_started = time.perf_counter()
        exchange_result, groups, upstream_merges = load_groups(exchange_root, frozen, backend)
        replay_seconds = time.perf_counter() - replay_started
        policy, _ = publish_sealed_json(root / "preflight.json", {
            "format": "native-spine-parent-budget-assessment-v1",
            "exchange_root": str(exchange_root.resolve()), "attention_root": str(attention_root.resolve()),
            "exchange_result_sha256": exchange_result.sha256,
            "attention_result_sha256": frozen.result.sha256, "attention_method_sha256": frozen.method.sha256,
            "backend_sha256": backend.identity_sha256, "implementation": implementation(),
            "body_count": len(groups), "parent_caps": [128, 256, 512],
            "exchange_channel_cap": 128, "window_exchange_cap": 8, "max_prompt_tokens": 2048,
            "leaf_token_cap": 512, "max_leaf_exchanges": 2,
            "parent_merge_cache": "empty in every arm; compiled exchange outputs are shared",
            "upstream_accepted_merge_count": len(upstream_merges),
            "pending_count_meaning": "unique first unsatisfied merge per body; lower bound on full generation",
            "structural_reference": "old builder with constant diagnostic parent text; text never published or served",
            "model_calls": 0, "raw_inputs_to_qwen": False, "benchmark_questions_or_gold_read": False,
            "live_parent_journal_read": False, "not_an_accuracy_or_api_latency_evaluation": True,
        })
        body_rows, pending_jobs = [], {cap: {} for cap in (128, 256, 512)}
        stats = {cap: {"complete_without_generation": 0, "pending_bodies": 0, "prompt_budget_failures": 0,
                       "reused_summary_requests": 0, "cpu_assessment_wall_seconds": 0.0,
                       "complete_leaves": 0, "complete_parents": 0, "complete_original_atoms": 0}
                 for cap in pending_jobs}
        start = time.perf_counter()
        for ordinal, (sha, (body, atom_input, atoms, exchanges)) in enumerate(groups.items(), 1):
            common = dict(summarizer_identity=policy.sha256, leaf_token_cap=512,
                          max_leaf_exchanges=2, window_exchange_cap=8, max_prompt_tokens=2048)
            reference_scorer = RecordingAttention(frozen)
            # Attention and cuts depend only on original exchanges. Dummy text merely lets
            # the old builder finish; it supplies no actual hierarchy summary or prediction.
            reference = build_user_spine_hierarchy(exchanges, scorer=reference_scorer,
                summarize=lambda request: "Topology diagnostic only.", max_channel_tokens=128, **common)
            reference_structure = structure(reference)
            reference_sha = identity_sha256(reference_structure)
            expected_spans = tuple(s for atom in atoms for s in atom.spans)
            by_id = {s.section_id: s for s in reference.sections}
            if (len(reference.root_section_ids) != 1
                    or by_id[reference.root_section_ids[0]].spans != expected_spans):
                raise ValueError("structural reference changed original raw coverage")
            row = {"body_sha256": sha, "exchanges_sha256": body.sha256, "atomic_input_sha256": atom_input.sha256,
                   "structure_sha256": reference_sha, "attention_calls_sha256": identity_sha256(reference_scorer.calls),
                   "original_atoms": len(atoms), "exchange_count": len(exchanges), "arms": {}}
            for cap in pending_jobs:
                scorer = RecordingAttention(frozen)
                summarizer = ReusingSpineSummarizer(NeutralMergeCache())
                arm_start = time.perf_counter()
                try:
                    hierarchy = build_parent_budgeted_spine_hierarchy(exchanges, scorer=scorer,
                        summarize=summarizer, max_exchange_channel_tokens=128, max_parent_channel_tokens=cap, **common)
                except PendingMerge as missing:
                    key = neutral_key(missing.request)
                    pending_jobs[cap].setdefault(key, asdict(missing.request))
                    arm = {"status": "first_pending_merge", "merge_key": key}
                    stats[cap]["pending_bodies"] += 1
                except ValueError as error:
                    if str(error) not in {"spine summary request exceeds its prompt budget",
                                          "date-neutral summary request exceeds its prompt budget"}:
                        raise
                    arm = {"status": "prompt_budget_failure"}
                    stats[cap]["prompt_budget_failures"] += 1
                else:
                    if structure(hierarchy) != reference_structure:
                        raise ValueError("parent budget changed attention boundaries or original raw pointers")
                    if cap == 128:
                        parity_scorer = RecordingAttention(frozen)
                        original = build_user_spine_hierarchy(exchanges, scorer=parity_scorer,
                            summarize=ReusingSpineSummarizer(NeutralMergeCache()), max_channel_tokens=128, **common)
                        if hierarchy != original or scorer.calls != parity_scorer.calls:
                            raise ValueError("equal parent/input budgets changed the original hierarchy")
                    leaves = sum(not s.child_section_ids for s in hierarchy.sections)
                    arm = {"status": "complete_without_generation", "hierarchy_receipt_sha256": hierarchy.receipt_sha256,
                           "structure_sha256": reference_sha, "leaf_count": leaves, "parent_count": len(hierarchy.sections)-leaves}
                    stats[cap]["complete_without_generation"] += 1
                    stats[cap]["complete_leaves"] += leaves
                    stats[cap]["complete_parents"] += len(hierarchy.sections)-leaves
                    stats[cap]["complete_original_atoms"] += len(atoms)
                stats[cap]["cpu_assessment_wall_seconds"] += time.perf_counter()-arm_start
                if scorer.calls != reference_scorer.calls or summarizer.generated_requests != 0:
                    raise ValueError("assessment changed attention inputs or unexpectedly generated a summary")
                stats[cap]["reused_summary_requests"] += summarizer.reused_requests
                arm["reused_summary_requests"] = summarizer.reused_requests
                row["arms"][str(cap)] = arm
            body_rows.append(row)
            if ordinal % 200 == 0 or ordinal == len(groups):
                print({"assessed_bodies": ordinal, "complete_without_generation": {
                    cap: stats[cap]["complete_without_generation"] for cap in stats}}, flush=True)
        for cap, jobs in pending_jobs.items():
            stats[cap]["unique_first_pending_merges"] = len(jobs)
            if sum(stats[cap][field] for field in ("complete_without_generation", "pending_bodies", "prompt_budget_failures")) != len(groups):
                raise ValueError("assessment population accounting changed")
        if backend.model is not None or backend.tokenizer is not None or implementation() != policy.payload["implementation"]:
            raise ValueError("assessment loaded a model or changed implementation")
        details, _ = publish_sealed_json(root / "body-assessment.json", {
            "preflight_sha256": policy.sha256, "bodies": body_rows,
            "first_pending_summary_requests": {str(cap): jobs for cap, jobs in pending_jobs.items()},
            "serving_hierarchies_published": False,
        })
        result, _ = publish_sealed_json(root / "result.json", {
            "preflight_sha256": policy.sha256, "body_assessment_sha256": details.sha256,
            "body_count": len(groups), "arms": {str(cap): values for cap, values in stats.items()},
            "zero_call_exchange_replay_seconds": replay_seconds,
            "assessment_wall_seconds": time.perf_counter()-start,
            "all_attention_inputs_and_receipts_identical": True,
            "all_completed_raw_partitions_and_cuts_identical": True,
            "equal_budget_completed_hierarchies_identical": True,
            "new_model_calls": 0, "model_loaded": False, "raw_inputs_to_qwen": False,
            "benchmark_questions_or_gold_read": False, "serving_hierarchies_published": False,
            "full1m_accuracy_or_latency_gate_passed": False,
        })
        print({"result_sha256": result.sha256, "arms": result.payload["arms"]}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exchange-root", type=Path, required=True)
    parser.add_argument("--attention-root", type=Path, required=True)
    args = parser.parse_args()
    backend = NativeQwenBackend(Path("eval_results/local-qwen-parent-summary-probe-20260910-r1"),
        Path(".cache/local-qwen-runtime/site-packages"), Path("../../.cache/models/Qwen3-8B"))
    assess(args.root, args.exchange_root, args.attention_root, backend)
