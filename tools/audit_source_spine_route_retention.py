"""Measure selected user-evidence loss without opening benchmark answers.

This is a diagnostic over frozen development prompts, not relevance scoring,
answer accuracy, a serving change, or a complete-candidate guarantee.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_spine_reader as evaluation
from tools.matched_eval.artifacts import publish_sealed_json


def user_turns(sections):
    return {span.turn_id for section in sections for span in section.spans if span.role == "user"}


def audit(root, output):
    preflight = evaluation.load_preflight(root)
    p = preflight.payload
    memory = evaluation.ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), p["addresses_sha256"], p["atoms_sha256"])
    rows = []
    try:
        for call in p["calls"]:
            if call["arm"] != "base":
                continue
            q = call["question"]
            query = q["retrieval_query"]
            view = ordered_content_query(query)
            identity = summary_embedding_identity(memory.encoder)
            vectors = memory.encoder.embed_queries([query, view]) if view != query else [memory.encoder.embed_query(query)]
            if summary_embedding_identity(memory.encoder) != identity:
                raise ValueError("query encoder identity changed")
            selected, routing = memory.router.route_vectors(query, q["prompt_question"], vectors[0],
                embedding_identity=identity, content_vector=vectors[1] if len(vectors) == 2 else None)
            expanded, expansion = memory.source_spine.expand(selected)
            hydrated = hydrate_section_plan(expanded, load_turn=memory.turns.get, max_context_tokens=3072, max_raw_spans=128)
            if evaluation.answer_messages(q, hydrated, policy="base") != call["messages"]:
                raise ValueError("diagnostic retrieval does not reproduce the frozen evidence")
            routed_users = user_turns(r.section for r in selected.routes)
            baseline_users = user_turns(memory.source_spine.originals[s] for s in routing["baseline_section_ids"])
            planned_users = user_turns(r.section for r in expanded.routes)
            hydrated_users = user_turns(s.section for s in hydrated.sections)
            outside_sources = user_turns(r.section for r in selected.routes
                if r.section.source_id not in expansion["selected_sources"])
            rows.append({"ordinal": q["ordinal"], "question_sha256": q["retrieval_query_sha256"],
                "routed_sources": len({r.section.source_id for r in selected.routes}),
                "expanded_sources": len(expansion["selected_sources"]), "routed_user_turns": len(routed_users),
                "baseline_routed_user_turns": len(baseline_users), "planned_user_turns": len(planned_users),
                "hydrated_user_turns": len(hydrated_users),
                "routed_user_turns_outside_source_cap": sorted(outside_sources),
                "routed_user_turns_lost_in_source_expansion": sorted(routed_users - planned_users),
                "baseline_user_turns_lost_before_reader": sorted(baseline_users - hydrated_users),
                "planned_user_turns_lost_at_final_budget": sorted(planned_users - hydrated_users),
                "backfill_user_turns_added": sorted(hydrated_users - routed_users),
                "raw_context_tokens": hydrated.context_token_count, "frozen_prompt_reproduced": True})
    finally:
        memory.encoder.close()
    artifact, _ = publish_sealed_json(output, {"preflight_sha256": preflight.sha256,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows,
        "gold_loaded": False, "predictions_loaded": False, "new_provider_calls": 0,
        "relevance_of_omitted_turns": "unmeasured", "answer_accuracy_measured": False,
        "serving_policy_changed": False})
    print({"audit_sha256": artifact.sha256, "questions": len(rows),
        "questions_losing_baseline_user_turns": sum(bool(r["baseline_user_turns_lost_before_reader"]) for r in rows),
        "questions_losing_routed_users_to_source_cap": sum(bool(r["routed_user_turns_outside_source_cap"]) for r in rows),
        "questions_losing_planned_users_at_final_budget": sum(bool(r["planned_user_turns_lost_at_final_budget"]) for r in rows),
        "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation_root, args.output)
