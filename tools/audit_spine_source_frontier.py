"""Inspect complete development populations for repeated-source summary hits.

The diagnostic reads frozen questions and summary indexes, without opening
predictions or gold. It reconstructs the baseline before recording wider
summary ranks. This does not measure answer accuracy or query latency.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_spine_reader_v3 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json
from tools.spine_facet_memory import ResidentMemory, supplemental_plan


def audit(root, output):
    frozen = evaluation.load_preflight(root)
    p = frozen.payload
    memory = ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), Path(p["facets_root"]),
        p["addresses_sha256"], p["atoms_sha256"], p["facets_sha256"])
    rows = []
    try:
        for call in p["calls"]:
            if call["arm"] != "base":
                continue
            q = call["question"]
            query, view = q["retrieval_query"], ordered_content_query(q["retrieval_query"])
            identity = summary_embedding_identity(memory.encoder)
            vectors = memory.encoder.embed_queries([query, view]) if view != query else [memory.encoder.embed_query(query)]
            if summary_embedding_identity(memory.encoder) != identity:
                raise ValueError("query encoder changed")
            selected, route_audit = memory.router.route_vectors(query, q["prompt_question"], vectors[0],
                embedding_identity=identity, content_vector=vectors[1] if len(vectors) == 2 else None)
            users = memory.router.user_addresses.route_vector(query, vectors[0], embedding_identity=identity,
                user_weight=1, max_sections=8, lexical_reserve=0)
            facets, _ = memory.facets.route_vector(query, vectors[0], embedding_identity=identity, max_sections=8)
            prior, _ = memory.supplement.expand(selected, supplemental_plan(users, facets))
            baseline = hydrate_section_plan(prior, load_turn=memory.turns.get, max_context_tokens=3072, max_raw_spans=128)
            if evaluation.answer_messages(q, baseline, policy="base") != call["messages"]:
                raise ValueError("baseline prompt failed to reproduce")
            all_users = memory.router.user_addresses.route_vector(query, vectors[0], embedding_identity=identity,
                user_weight=1, max_sections=32, lexical_reserve=0)
            all_facets, facet_audit = memory.facets.route_vector(query, vectors[0],
                embedding_identity=identity, max_sections=32)
            all_combined = memory.semantic.route_vector(query, vectors[0], embedding_identity=identity,
                max_sections=32, lexical_reserve=0)
            prior_sources = {r.section.source_id for r in prior.routes}
            def frontier(plan):
                return [{"rank": i + 1, "section_id": r.section.section_id, "source_id": r.section.source_id,
                    "source_in_prior": r.section.source_id in prior_sources, "summary": r.section.summary,
                    "user_turn_ids": list(dict.fromkeys(s.turn_id for s in r.section.spans if s.role == "user"))}
                    for i, r in enumerate(plan.routes)]
            row = {"ordinal": q["ordinal"], "question": q, "baseline_prompt_reproduced": True,
                "baseline_context_tokens": baseline.context_token_count,
                "baseline_user_tokens": count_tokens("\n\n".join(s.render_raw(f"S{i}")
                    for i, s in enumerate(baseline.sections, 1) if all(e.span.role == "user" for e in s.evidence))),
                "baseline_user_sections": sum(all(e.span.role == "user" for e in s.evidence) for s in baseline.sections),
                "prior_sources": sorted(prior_sources),
                "top8_user_source_counts": dict(Counter(r.section.source_id for r in users.routes)),
                "top8_facet_source_counts": dict(Counter(r.section.source_id for r in facets.routes)),
                "user_frontier": frontier(all_users), "facet_frontier": frontier(all_facets),
                "combined_frontier": frontier(all_combined), "facet_audit": facet_audit, "route_audit": route_audit}
            rows.append(row)
            print({"ordinal": q["ordinal"], "top8_user_sources": len(row["top8_user_source_counts"]),
                "top8_facet_sources": len(row["top8_facet_source_counts"]),
                "baseline_tokens": baseline.context_token_count, "baseline_user_tokens": row["baseline_user_tokens"]}, flush=True)
    finally:
        memory.encoder.close()
    if len(rows) != 10:
        raise ValueError("expected all ten questions in the complete namespace")
    artifact, _ = publish_sealed_json(output / "audit.json", {
        "format": "memory-condense-spine-source-frontier-audit-v1", "evaluation_preflight_sha256": frozen.sha256,
        "shard_offset": p["shard_offset"], "raw_token_proxy": p["raw_token_proxy"], "rows": rows,
        "summary_frontier": 32, "baseline_top_k": 8, "new_provider_calls": 0,
        "gold_loaded": False, "predictions_loaded": False, "witness_selectors": False,
        "raw_embedding_inputs": False, "answer_accuracy_measured": False,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print({"audit_sha256": artifact.sha256, "questions": len(rows), "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation_root, args.output_root)
