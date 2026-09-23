"""Compare exact hydration with optional routed-user overflow; no answer calls."""
import argparse
import hashlib
from pathlib import Path
import time

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.eval.streaming_latency import latency_distribution
from memory_condense.search.source_spine_overflow import SourceSpineOverflow
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.audit_source_spine_route_retention import evaluation, user_turns
from tools.matched_eval.artifacts import publish_sealed_json


def audit(root, output):
    preflight = evaluation.load_preflight(root)
    p = preflight.payload
    memory = evaluation.ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), p["addresses_sha256"], p["atoms_sha256"])
    overflow = SourceSpineOverflow(memory.source_spine)
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
            prior, _ = memory.source_spine.expand(selected)
            started = time.perf_counter()
            candidate, expansion = overflow.expand(selected)
            expansion_s = time.perf_counter() - started
            hydrate = lambda plan: hydrate_section_plan(plan, load_turn=memory.turns.get, max_context_tokens=3072, max_raw_spans=128)
            before, after = hydrate(prior), hydrate(candidate)
            if evaluation.answer_messages(q, before, policy="base") != call["messages"]:
                raise ValueError("baseline evidence no longer matches its frozen prompt")
            old_users = user_turns(s.section for s in before.sections)
            new_users = user_turns(s.section for s in after.sections)
            if not old_users <= new_users:
                raise ValueError("overflow displaced a protected user turn")
            old_user_evidence = [s.identity_payload() for s in before.sections if all(e.span.role == "user" for e in s.evidence)]
            new_user_evidence = [s.identity_payload() for s in after.sections if all(e.span.role == "user" for e in s.evidence)]
            if new_user_evidence[:len(old_user_evidence)] != old_user_evidence:
                raise ValueError("protected user bytes, roles, dates or ordering changed")
            routed_users = user_turns(r.section for r in selected.routes)
            baseline_users = user_turns(memory.source_spine.originals[s] for s in routing["baseline_section_ids"])
            rows.append({"ordinal": q["ordinal"], "question_sha256": q["retrieval_query_sha256"],
                "protected_user_turns": len(old_users), "added_user_turns": sorted(new_users - old_users),
                "routed_users_missing_before": sorted(routed_users - old_users),
                "routed_users_missing_after": sorted(routed_users - new_users),
                "baseline_users_missing_after": sorted(baseline_users - new_users),
                "attached_section_count_delta": len(before.sections) - len(old_users) - len(after.sections) + len(new_users),
                "attached_sections_removed": sorted({s.section.section_id for s in before.sections if any(e.span.role != "user" for e in s.evidence)} -
                    {s.section.section_id for s in after.sections if any(e.span.role != "user" for e in s.evidence)}),
                "context_tokens_before": before.context_token_count, "context_tokens_after": after.context_token_count,
                "candidate_expansion_s": expansion_s, "expansion": expansion,
                "candidate_hydration": after.identity_payload(), "protected_user_evidence_unchanged": True})
    finally:
        memory.encoder.close()
    result, _ = publish_sealed_json(output, {"preflight_sha256": preflight.sha256, "rows": rows,
        "gold_loaded": False, "predictions_loaded": False, "answer_accuracy_measured": False,
        "new_provider_calls": 0, "promoted": False, "max_context_tokens": 3072,
        "expansion_only_timing": latency_distribution([r["candidate_expansion_s"] for r in rows]),
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/audit_source_spine_overflow.py", "src/memory_condense/search/source_spine_overflow.py",
            "tools/audit_source_spine_route_retention.py")}})
    print({"audit_sha256": result.sha256, "questions": len(rows),
        "questions_adding_routed_users": sum(bool(r["added_user_turns"]) for r in rows),
        "questions_still_missing_routed_users": sum(bool(r["routed_users_missing_after"]) for r in rows),
        "all_protected_user_evidence_preserved": True, "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation_root, args.output)
