"""Compare source diversity and user-only addresses within a soft calendar hint."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_query_view import ordered_content_query
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.summary_time_prior import mention_window
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def run(index_root, evaluation_root, addresses_root, root):
    manifest, index = load_index(index_root)
    evaluation = read_sealed_json(evaluation_root / "preflight.json")
    address_manifest = read_sealed_json(addresses_root / "addresses.json")
    if evaluation.payload["index_manifest_sha256"] != manifest.sha256 or address_manifest.payload["base_index_sha256"] != manifest.sha256:
        raise ValueError("memory binding changed")
    matrix_path = addresses_root / "spine-vectors.npy"
    if hashlib.sha256(matrix_path.read_bytes()).hexdigest() != address_manifest.payload["matrix_sha256"]:
        raise ValueError("user addresses changed")
    addresses = UserSpineAddressIndex(index.hierarchy, np.load(index_root / "summary-vectors.npy", allow_pickle=False),
        np.load(matrix_path, allow_pickle=False), embedding_identity=index.embedding_identity)
    if addresses.receipt_sha256 != address_manifest.payload["address_index_sha256"]:
        raise ValueError("user address metadata changed")
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "base_index_sha256": manifest.sha256, "evaluation_preflight_sha256": evaluation.sha256,
        "addresses_sha256": address_manifest.sha256, "maximum_frontier": 32,
        "preferred_sections": 4, "max_preferred_per_source": 1, "raw_reads": 0, "gold_loaded": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/probe_spine_time_frontier.py", "src/memory_condense/search/summary_time_prior.py",
            "src/memory_condense/search/summary_query_view.py", "src/memory_condense/search/user_spine_addresses.py")}})
    questions = {c["question"]["ordinal"]: c["question"] for c in evaluation.payload["calls"]}
    encoder = EmbeddingService(device="cuda", batch_size=8)
    rows = []
    try:
        encoder.embed_query("Initialize calendar source search.")
        identity = summary_embedding_identity(encoder)
        for q in questions.values():
            window = mention_window(q["prompt_question"])
            if window is None:
                continue
            start, end = window
            sources = tuple(sorted({s.source_id for s in index.sections if any(
                start <= datetime.fromisoformat(span.created_at).date() < end for span in s.spans)}))
            for view, query in (("original", q["retrieval_query"]), ("content", ordered_content_query(q["retrieval_query"]))):
                vector = encoder.embed_query(query)
                if summary_embedding_identity(encoder) != identity:
                    raise ValueError("encoder changed")
                plans = {"combined": index.route_vector(query, vector, embedding_identity=identity,
                    max_sections=32, eligible_source_ids=sources), "user": addresses.route_vector(query, vector,
                    embedding_identity=identity, max_sections=32, eligible_source_ids=sources, user_weight=1.0)}
                for arm, plan in plans.items():
                    selected = {}
                    for route in plan.routes:
                        selected.setdefault(route.section.source_id, route.section.section_id)
                    rows.append({"ordinal": q["ordinal"], "view": view, "query": query, "arm": arm,
                        "ranked": [{"section_id": r.section.section_id, "source_id": r.section.source_id} for r in plan.routes],
                        "source_diverse": list(selected.values())})
    finally:
        encoder.close()
    artifact, _ = publish_sealed_json(root / "routing-probe.json", {"preflight_sha256": preflight.sha256,
        "rows": rows, "provider_calls": 0, "raw_reads": 0, "gold_loaded": False, "used_for_target_gate": False})
    print({"routing_sha256": artifact.sha256, "rows": len(rows), "provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index-root", "evaluation-root", "addresses-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.index_root, args.evaluation_root, args.addresses_root, args.output_root)
