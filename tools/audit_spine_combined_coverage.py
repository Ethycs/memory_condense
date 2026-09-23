"""Reproduce complete source-diverse controls and audit a whole-summary supplement.

Only frozen questions, compiled summaries and exact evidence are read. No
predictions, gold, witness IDs or outcome-based routing enter this diagnostic.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_source_coverage as evaluation
from tools.matched_eval.artifacts import publish_sealed_json
from tools.spine_combined_memory import ResidentMemory


def audit(root, output):
    frozen = evaluation.load_preflight(root)
    p = frozen.payload
    memory = ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), Path(p["facets_root"]),
        p["addresses_sha256"], p["atoms_sha256"], p["facets_sha256"])
    rows = []
    try:
        for call in p["calls"]:
            if call["arm"] != "source_spine_diverse":
                continue
            q = call["question"]
            baseline = memory.retrieve(q["retrieval_query"], call["arm"], q["prompt_question"])
            if evaluation.answer_messages(q, baseline) != call["messages"]:
                raise ValueError("source-diverse control prompt failed to reproduce")
            plan, routing = memory.combined_plan(q["retrieval_query"], q["prompt_question"])
            candidate = hydrate_section_plan(plan, load_turn=memory.turns.get,
                max_context_tokens=3072, max_raw_spans=128)
            def users(result):
                return [s for s in result.sections if all(e.span.role == "user" for e in s.evidence)]
            old_users, new_users = users(baseline), users(candidate)
            if new_users[:len(old_users)] != old_users:
                raise ValueError("whole-summary supplement changed previous user evidence")
            def evidence(result):
                return {e.span.receipt_sha256: e.span.identity_payload() for s in result.sections for e in s.evidence}
            old, new = evidence(baseline), evidence(candidate)
            messages = evaluation.answer_messages(q, candidate)
            prompt, _ = publish_sealed_json(output / "candidate-prompts" / f'{q["ordinal"]:03d}.json', {
                "evaluation_preflight_sha256": frozen.sha256, "question": q,
                "baseline_messages_sha256": call["messages_sha256"], "messages": messages,
                "messages_sha256": identity_sha256(messages), "plan": plan.identity_payload(),
                "gold_loaded": False, "new_provider_calls": 0})
            row = {"ordinal": q["ordinal"], "candidate_prompt_sha256": prompt.sha256,
                "baseline_prompt_reproduced": True, "baseline_user_evidence_preserved": True,
                "baseline_context_tokens": baseline.context_token_count,
                "candidate_context_tokens": candidate.context_token_count,
                "baseline_prompt_changed": messages != call["messages"], "routing": routing,
                "added_spans": [new[k] for k in new if k not in old],
                "removed_spans": [old[k] for k in old if k not in new],
                "diagnostics": [d.identity_payload() for d in candidate.diagnostics]}
            rows.append(row)
            print({"ordinal": q["ordinal"], "changed": row["baseline_prompt_changed"],
                "added_spans": len(row["added_spans"]), "removed_spans": len(row["removed_spans"]),
                "candidate_tokens": candidate.context_token_count}, flush=True)
    finally:
        memory.encoder.close()
    if len(rows) != 10:
        raise ValueError("expected the complete ten-question namespace")
    artifact, _ = publish_sealed_json(output / "audit.json", {
        "format": "memory-condense-spine-combined-coverage-audit-v1", "evaluation_preflight_sha256": frozen.sha256,
        "shard_offset": p["shard_offset"], "raw_token_proxy": p["raw_token_proxy"], "rows": rows,
        "combined_sources": 8, "summary_frontier": 32, "context_tokens": 3072,
        "new_provider_calls": 0, "gold_loaded": False, "predictions_loaded": False,
        "answer_accuracy_measured": False, "full100_target_eligible": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            *evaluation.IMPLEMENTATION, "tools/spine_combined_memory.py", "tools/audit_spine_combined_coverage.py")}})
    print({"audit_sha256": artifact.sha256, "questions": len(rows), "new_provider_calls": 0}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation_root, args.output_root)
