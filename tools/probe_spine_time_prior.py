"""Probe calendar mention priority without raw content, references or providers."""
import argparse
import hashlib
from pathlib import Path
import time

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.summary_time_prior import route_with_time_prior
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def run(index_root, evaluation_root, root):
    manifest, index = load_index(index_root)
    evaluation = read_sealed_json(evaluation_root / "preflight.json")
    if evaluation.payload["index_manifest_sha256"] != manifest.sha256:
        raise ValueError("questions belong to another memory")
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "base_index_sha256": manifest.sha256, "evaluation_preflight_sha256": evaluation.sha256,
        "preferred_sections": 4, "total_sections": 6, "raw_reads": 0, "gold_loaded": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/probe_spine_time_prior.py", "src/memory_condense/search/summary_time_prior.py",
            "src/memory_condense/search/summary_semantic_index.py", "src/memory_condense/modeling/embedding.py")}})
    questions = {c["question"]["ordinal"]: c["question"] for c in evaluation.payload["calls"]}
    encoder = EmbeddingService(device="cuda", batch_size=8)
    rows = []
    try:
        encoder.embed_query("Initialize calendar summary search.")
        identity = summary_embedding_identity(encoder)
        for q in questions.values():
            start = time.perf_counter()
            vector = encoder.embed_query(q["retrieval_query"])
            if summary_embedding_identity(encoder) != identity:
                raise ValueError("query encoder identity changed")
            plan, hint = route_with_time_prior(index, q["retrieval_query"], q["prompt_question"], vector,
                                               embedding_identity=identity)
            rows.append({"ordinal": q["ordinal"], "question_id": q["question_id"],
                "query": q["retrieval_query"], "hint": hint, "plan": plan.identity_payload(),
                "embedding_and_route_s": time.perf_counter() - start})
    finally:
        encoder.close()
    artifact, _ = publish_sealed_json(root / "routing-probe.json", {"preflight_sha256": preflight.sha256,
        "rows": rows, "provider_calls": 0, "raw_reads": 0, "gold_loaded": False, "used_for_target_gate": False})
    print({"routing_sha256": artifact.sha256, "questions": len(rows), "provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.index_root, args.evaluation_root, args.output_root)
