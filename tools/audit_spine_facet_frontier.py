"""Audit additional summary addresses against complete, already scored memories.

Witness annotations are used only after selection to describe diagnostic recall.
They never affect ranking, admission, or memory construction. No answer calls.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan
from memory_condense.search.source_spine_supplement import SourceSpineSupplement
from memory_condense.search.spine_summary_facets import SpineFacetAddressIndex
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_source_spine_overflow as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def supplemental_plan(user_plan, facet_plan):
    """Interleave the two fixed top-eight lists; retain unchanged leaf descriptors."""
    if (user_plan.index_sha256, user_plan.query_sha256) != (
            facet_plan.index_sha256, facet_plan.query_sha256):
        raise ValueError("supplemental address plans differ in hierarchy or query")
    selected = {}
    for i in range(max(len(user_plan.routes), len(facet_plan.routes))):
        for plan in (user_plan, facet_plan):
            if i < len(plan.routes):
                section = plan.routes[i].section
                if section.section_id in selected and selected[section.section_id] != section:
                    raise ValueError("supplemental descriptors disagree")
                selected.setdefault(section.section_id, section)
    return SectionRoutePlan(user_plan.index_sha256, user_plan.query_sha256,
        tuple(SectionRoute(s, 1 / (i + 1), ()) for i, s in enumerate(selected.values())),
        max(user_plan.matched_section_count, facet_plan.matched_section_count), None,
        max(1, len(selected)), routing_backend="summary_dense")


def audit(root, facet_root, output, witness_path=None):
    preflight = evaluation.load_preflight(root)
    p = preflight.payload
    manifest = read_sealed_json(facet_root / "addresses.json")
    facet_preflight = read_sealed_json(facet_root / "preflight.json")
    path = facet_root / "facet-vectors.npy"
    if (manifest.payload["base_index_sha256"] != p["index_manifest_sha256"] or
            manifest.payload["preflight_sha256"] != facet_preflight.sha256 or
            manifest.payload["matrix_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("facet address inputs differ from the compiled index")
    for name, digest in facet_preflight.payload["implementation"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("facet compiler implementation changed")
    memory = evaluation.ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), p["addresses_sha256"], p["atoms_sha256"])
    rows = []
    try:
        facets = SpineFacetAddressIndex(memory.semantic.hierarchy, np.load(path, allow_pickle=False),
            embedding_identity=manifest.payload["embedding_identity"])
        if facets.receipt_sha256 != manifest.payload["address_index_sha256"]:
            raise ValueError("facet address identity changed")
        supplement = SourceSpineSupplement(memory.source_spine)
        for call in p["calls"]:
            if call["arm"] != "source_spine_overflow":
                continue
            q = call["question"]
            query = q["retrieval_query"]
            view = ordered_content_query(query)
            identity = summary_embedding_identity(memory.encoder)
            vectors = memory.encoder.embed_queries([query, view]) if view != query else [memory.encoder.embed_query(query)]
            if summary_embedding_identity(memory.encoder) != identity:
                raise ValueError("query embedding identity changed")
            selected, selection = memory.router.route_vectors(query, q["prompt_question"], vectors[0],
                embedding_identity=identity, content_vector=vectors[1] if len(vectors) == 2 else None)
            original, _ = memory.overflow.expand(selected)
            baseline = hydrate_section_plan(original, load_turn=memory.turns.get,
                max_context_tokens=3072, max_raw_spans=128)
            if evaluation.answer_messages(q, baseline) != call["messages"]:
                raise ValueError("baseline prompt differs from the sealed comparison")
            # Score the entire address population for a diagnostic rank report;
            # only a fixed top eight from each independent summary lane is admitted.
            full_facets, ranked_facets = facets.route_vector(query, vectors[0], embedding_identity=identity,
                max_sections=len(facets.sections))
            top_facets = SectionRoutePlan(full_facets.index_sha256, full_facets.query_sha256,
                full_facets.routes[:8], full_facets.matched_section_count, None, 8, routing_backend="summary_dense")
            wider_users = memory.router.user_addresses.route_vector(query, vectors[0],
                embedding_identity=identity, user_weight=1, max_sections=8, lexical_reserve=0)
            extra = supplemental_plan(wider_users, top_facets)
            plan, expansion = supplement.expand(selected, extra)
            candidate = hydrate_section_plan(plan, load_turn=memory.turns.get,
                max_context_tokens=3072, max_raw_spans=128)
            def user_sections(result):
                return [s for s in result.sections if all(e.span.role == "user" for e in s.evidence)]
            old_users = user_sections(baseline)
            new_users = user_sections(candidate)
            if new_users[:len(old_users)] != old_users:
                raise ValueError("previously hydrated user evidence changed")
            messages = evaluation.answer_messages(q, candidate)
            rows.append({"ordinal": q["ordinal"], "query_sha256": quote_sha256(query),
                "frozen_baseline_prompt_reproduced": True,
                "baseline_user_evidence_retained_exactly": True,
                "baseline_context_tokens": baseline.context_token_count,
                "candidate_context_tokens": candidate.context_token_count,
                "baseline_user_turn_ids": [e.span.turn_id for s in old_users for e in s.evidence],
                "candidate_user_turn_ids": [e.span.turn_id for s in new_users for e in s.evidence],
                "candidate_prompt_sha256": identity_sha256(messages), "prompt_changed": messages != call["messages"],
                "selected_supplement_section_ids": [r.section.section_id for r in extra.routes],
                "facet_ranking": ranked_facets, "selection": selection, "expansion": expansion,
                "hydration_diagnostics": [d.identity_payload() for d in candidate.diagnostics]})
            print({"ordinal": q["ordinal"], "old_users": len(old_users), "new_users": len(new_users),
                "old_tokens": baseline.context_token_count, "new_tokens": candidate.context_token_count}, flush=True)
    finally:
        memory.encoder.close()
    # These annotations cannot influence any preceding route or prompt.
    annotation = read_sealed_json(witness_path) if witness_path else None
    witnesses = {}
    if annotation:
        for row in annotation.payload["rows"]:
            witnesses.setdefault(row["ordinal"], {})[row["witness"]["turn_id"]] = row["witness"]
    for row in rows:
        row["witnesses"] = []
        for turn_id, witness in witnesses.get(row["ordinal"], {}).items():
            leaves = {s.section_id for s in facets.sections if any(span.turn_id == turn_id for span in s.spans)}
            ranking = [{"rank": i + 1, **r} for i, r in enumerate(row["facet_ranking"]) if r["section_id"] in leaves]
            row["witnesses"].append({"turn_id": turn_id, "source_id": witness["source_id"],
                "facet_matches": ranking, "admitted_to_candidate": turn_id in row["candidate_user_turn_ids"]})
        row["facet_top20"] = row.pop("facet_ranking")[:20]
    result, _ = publish_sealed_json(output, {"preflight_sha256": preflight.sha256,
        "facet_manifest_sha256": manifest.sha256, "rows": rows, "new_provider_calls": 0,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "witness_annotation_sha256": annotation.sha256 if annotation else None,
        "postscore_development_diagnostic": True, "answer_accuracy_measured": False,
        "full100_target_eligible": False, "query_vectors_retained": False,
        "supplement_policy": "original user top8 and summary-passage top8, interleaved by rank; baseline users first"})
    print({"audit_sha256": result.sha256, "questions": len(rows), "new_provider_calls": 0,
        "witnesses": [{"ordinal": row["ordinal"], "witnesses": row["witnesses"]} for row in rows if row["witnesses"]]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--facets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--witness-audit", type=Path)
    args = parser.parse_args()
    audit(args.evaluation_root, args.facets_root, args.output, args.witness_audit)
