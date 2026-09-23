"""Compare source-scoped summary-term coverage against frozen complete-memory evidence.

This reads questions and stored summaries, never gold or predictions. It makes
no answer calls and provides no answer-accuracy or query-latency claim.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.spine_term_coverage_v2 import ScopedSpineTermCoverage
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_spine_reader_v3 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json
from tools.spine_facet_memory import ResidentMemory, supplemental_plan


def audit(root, output):
    preflight = evaluation.load_preflight(root)
    p = preflight.payload
    memory = ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), Path(p["facets_root"]),
        p["addresses_sha256"], p["atoms_sha256"], p["facets_sha256"])
    rows = []
    try:
        coverage = ScopedSpineTermCoverage(memory.semantic.hierarchy, memory.source_spine)
        for call in p["calls"]:
            if call["arm"] != "base":
                continue
            q = call["question"]
            query, view = q["retrieval_query"], ordered_content_query(q["retrieval_query"])
            identity = summary_embedding_identity(memory.encoder)
            vectors = memory.encoder.embed_queries([query, view]) if view != query else [memory.encoder.embed_query(query)]
            if summary_embedding_identity(memory.encoder) != identity:
                raise ValueError("query encoder changed")
            selected, _ = memory.router.route_vectors(query, q["prompt_question"], vectors[0], embedding_identity=identity,
                content_vector=vectors[1] if len(vectors) == 2 else None)
            wider = memory.router.user_addresses.route_vector(query, vectors[0], embedding_identity=identity,
                user_weight=1, max_sections=8, lexical_reserve=0)
            facets, _ = memory.facets.route_vector(query, vectors[0], embedding_identity=identity, max_sections=8)
            prior, _ = memory.supplement.expand(selected, supplemental_plan(wider, facets))
            baseline = hydrate_section_plan(prior, load_turn=memory.turns.get, max_context_tokens=3072, max_raw_spans=128)
            if evaluation.answer_messages(q, baseline, policy="base") != call["messages"]:
                raise ValueError("baseline evidence differs from the frozen matched comparison")
            plan, routing = coverage.expand(query, prior)
            candidate = hydrate_section_plan(plan, load_turn=memory.turns.get, max_context_tokens=3072, max_raw_spans=128)
            def users(result):
                return [s for s in result.sections if all(e.span.role == "user" for e in s.evidence)]
            old_users, new_users = users(baseline), users(candidate)
            if new_users[:len(old_users)] != old_users:
                raise ValueError("previous user evidence changed")
            def evidence(result):
                return {e.span.receipt_sha256: e.span.identity_payload() for s in result.sections for e in s.evidence}
            old, new = evidence(baseline), evidence(candidate)
            messages = evaluation.answer_messages(q, candidate, policy="base")
            frozen, _ = publish_sealed_json(output / "candidate-prompts" / f'{q["ordinal"]:03d}.json', {
                "evaluation_preflight_sha256": preflight.sha256, "question": q,
                "baseline_messages_sha256": call["messages_sha256"], "messages": messages,
                "messages_sha256": identity_sha256(messages), "plan": plan.identity_payload(),
                "gold_loaded": False, "new_provider_calls": 0})
            row = {"ordinal": q["ordinal"], "candidate_prompt_sha256": frozen.sha256,
                "baseline_user_evidence_preserved": True, "baseline_prompt_reproduced": True,
                "baseline_context_tokens": baseline.context_token_count, "candidate_context_tokens": candidate.context_token_count,
                "prompt_changed": messages != call["messages"], "routing": routing,
                "added_spans": [new[k] for k in new if k not in old],
                "removed_spans": [old[k] for k in old if k not in new],
                "diagnostics": [d.identity_payload() for d in candidate.diagnostics]}
            rows.append(row)
            print({"ordinal": q["ordinal"], "changed": row["prompt_changed"],
                "rare_terms": routing["rare_terms"], "added_spans": len(row["added_spans"]),
                "removed_spans": len(row["removed_spans"]), "candidate_tokens": candidate.context_token_count}, flush=True)
    finally:
        memory.encoder.close()
    artifact, _ = publish_sealed_json(output / "audit.json", {
        "evaluation_preflight_sha256": preflight.sha256, "rows": rows,
        "question_count": len(rows), "raw_token_proxy": p["raw_token_proxy"],
        "input_policy": "stored summary terms within previously selected conversations; exact raw hydration after selection",
        "new_sources_admitted": False,
        "rare_term_limit": 2, "leaves_per_term": 2, "maximum_document_frequency": coverage.maximum_document_frequency,
        "new_provider_calls": 0, "gold_loaded": False, "predictions_loaded": False,
        "answer_accuracy_measured": False, "full100_target_eligible": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/audit_spine_term_coverage_v2.py", "src/memory_condense/search/spine_term_coverage.py",
            "src/memory_condense/search/spine_term_coverage_v2.py")}})
    print({"audit_sha256": artifact.sha256, "questions": len(rows), "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation_root, args.output_root)
