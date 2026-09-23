"""Audit source-diverse coverage on complete memory with unchanged controls.

No predictions or gold are read. Wider source selection and all raw evidence
displacement are diagnostic until the same prompts receive fresh answers.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_reader_v3 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.spine_source_memory import ResidentMemory


def audit(root, term_audit_root, output):
    frozen = evaluation.load_preflight(root)
    p = frozen.payload
    term_audit = read_sealed_json(term_audit_root / "audit.json")
    if term_audit.payload["evaluation_preflight_sha256"] != frozen.sha256:
        raise ValueError("scoped-term controls belong to another evaluation")
    memory = ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), Path(p["facets_root"]),
        p["addresses_sha256"], p["atoms_sha256"], p["facets_sha256"])
    rows = []
    try:
        for call in p["calls"]:
            if call["arm"] != "base":
                continue
            q = call["question"]
            baseline = memory.retrieve(q["retrieval_query"], "source_spine_facets", q["prompt_question"])
            if evaluation.answer_messages(q, baseline, policy="base") != call["messages"]:
                raise ValueError("facet baseline prompt failed to reproduce")
            term = memory.retrieve(q["retrieval_query"], "source_spine_term_coverage", q["prompt_question"])
            expected_term = read_sealed_json(term_audit_root / "candidate-prompts" / f'{q["ordinal"]:03d}.json')
            term_binding = next(r for r in term_audit.payload["rows"] if r["ordinal"] == q["ordinal"])
            if (expected_term.sha256 != term_binding["candidate_prompt_sha256"] or
                    expected_term.payload["messages"] != evaluation.answer_messages(q, term, policy="base")):
                raise ValueError("scoped-term baseline prompt failed to reproduce")
            plan, routing = memory.source_plan(q["retrieval_query"], q["prompt_question"])
            candidate = hydrate_section_plan(plan, load_turn=memory.turns.get, max_context_tokens=3072, max_raw_spans=128)
            def users(result):
                return [s for s in result.sections if all(e.span.role == "user" for e in s.evidence)]
            old_users, new_users = users(baseline), users(candidate)
            if new_users[:len(old_users)] != old_users:
                raise ValueError("previous facet user evidence changed")
            def evidence(result):
                return {e.span.receipt_sha256: e.span.identity_payload() for s in result.sections for e in s.evidence}
            old, new = evidence(baseline), evidence(candidate)
            term_evidence = evidence(term)
            messages = evaluation.answer_messages(q, candidate, policy="base")
            prompt, _ = publish_sealed_json(output / "candidate-prompts" / f'{q["ordinal"]:03d}.json', {
                "evaluation_preflight_sha256": frozen.sha256, "question": q,
                "baseline_messages_sha256": call["messages_sha256"], "messages": messages,
                "messages_sha256": identity_sha256(messages), "plan": plan.identity_payload(),
                "gold_loaded": False, "new_provider_calls": 0})
            row = {"ordinal": q["ordinal"], "candidate_prompt_sha256": prompt.sha256,
                "baseline_prompt_reproduced": True, "scoped_term_prompt_reproduced": True,
                "baseline_user_evidence_preserved": True, "baseline_context_tokens": baseline.context_token_count,
                "candidate_context_tokens": candidate.context_token_count,
                "baseline_prompt_changed": messages != call["messages"],
                "scoped_term_prompt_changed": messages != expected_term.payload["messages"], "routing": routing,
                "added_spans": [new[k] for k in new if k not in old],
                "removed_spans": [old[k] for k in old if k not in new],
                "scoped_term_removed_spans": [term_evidence[k] for k in term_evidence if k not in new],
                "diagnostics": [d.identity_payload() for d in candidate.diagnostics]}
            rows.append(row)
            print({"ordinal": q["ordinal"], "changed": row["baseline_prompt_changed"],
                "added_sources": len(routing["source_coverage"]["added_source_ids"]),
                "added_spans": len(row["added_spans"]), "removed_spans": len(row["removed_spans"]),
                "candidate_tokens": candidate.context_token_count}, flush=True)
    finally:
        memory.encoder.close()
    if len(rows) != 10:
        raise ValueError("expected the complete ten-question namespace")
    artifact, _ = publish_sealed_json(output / "audit.json", {
        "format": "memory-condense-spine-source-coverage-audit-v1", "evaluation_preflight_sha256": frozen.sha256,
        "scoped_term_audit_sha256": term_audit.sha256, "shard_offset": p["shard_offset"],
        "raw_token_proxy": p["raw_token_proxy"], "rows": rows,
        "sources_per_channel": 8, "summary_frontier": 32, "context_tokens": 3072,
        "new_provider_calls": 0, "gold_loaded": False, "predictions_loaded": False,
        "answer_accuracy_measured": False, "full100_target_eligible": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/audit_spine_source_coverage.py", "tools/spine_source_memory.py",
            "src/memory_condense/search/spine_source_coverage.py")}})
    print({"audit_sha256": artifact.sha256, "questions": len(rows), "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--scoped-term-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation_root, args.scoped_term_audit_root, args.output_root)
