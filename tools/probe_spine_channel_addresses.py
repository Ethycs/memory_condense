"""Compile summary-only user addresses, then compare bounded routing candidates.

No raw reader, answer generator or reference loader is available to this tool.
Query vectors are shared across diagnostic variants; this is not a latency or
answer-accuracy evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex, user_spine_text
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


IMPLEMENTATION = ("tools/probe_spine_channel_addresses.py", "src/memory_condense/search/user_spine_addresses.py",
                  "src/memory_condense/modeling/embedding.py", "src/memory_condense/search/summary_semantic_index.py")


def run(index_root, evaluation_root, root):
    parent, base = load_index(index_root)
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "base_index_sha256": parent.sha256, "document_inputs": "stored user-spine summaries only",
        "raw_inputs": False, "gold_inputs": False, "query_inputs_during_compilation": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    matrix_path = root / "spine-vectors.npy"
    encoder = EmbeddingService(device="cuda", batch_size=8)
    try:
        encoder.embed_query("Initialize summary addresses.")
        identity = summary_embedding_identity(encoder)
        if identity != base.embedding_identity:
            raise ValueError("user and combined addresses require the same encoder")
        if matrix_path.exists():
            manifest = read_sealed_json(root / "addresses.json")
            if manifest.payload["preflight_sha256"] != preflight.sha256 or manifest.payload["matrix_sha256"] != hashlib.sha256(matrix_path.read_bytes()).hexdigest():
                raise ValueError("user address matrix changed")
            matrix = np.load(matrix_path, allow_pickle=False)
        else:
            texts = [user_spine_text(s) for s in base.sections]
            matrix = np.array(encoder.embed_queries(texts), dtype=np.float32, copy=True)
            if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
                raise ValueError("invalid user-spine summary vectors")
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            with matrix_path.open("xb") as handle:
                np.save(handle, matrix, allow_pickle=False)
        if summary_embedding_identity(encoder) != identity:
            raise ValueError("encoder identity changed during summary compilation")
        combined = np.load(index_root / "summary-vectors.npy", allow_pickle=False)
        addresses = UserSpineAddressIndex(base.hierarchy, combined, matrix, embedding_identity=identity)
        manifest, _ = publish_sealed_json(root / "addresses.json", {
            "preflight_sha256": preflight.sha256, "base_index_sha256": parent.sha256,
            "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "address_index_sha256": addresses.receipt_sha256, "embedding_identity": identity,
            "section_count": len(base.sections), "new_provider_calls": 0})
        evaluation = read_sealed_json(evaluation_root / "preflight.json")
        if evaluation.payload["index_manifest_sha256"] != parent.sha256:
            raise ValueError("diagnostic questions belong to another memory")
        questions = {call["question"]["ordinal"]: call["question"] for call in evaluation.payload["calls"]}
        variants = [("spine_dense", 1.0, 0), ("spine_hybrid", 1.0, 2),
                    ("dual_dense", 0.8, 0), ("dual_hybrid", 0.8, 2)]
        rows = []
        for question in questions.values():
            query = question["retrieval_query"]
            vector = encoder.embed_query(query)
            plans = [("combined_hybrid", base.route_vector(query, vector, embedding_identity=identity,
                                                           max_sections=16, lexical_reserve=2))]
            plans.extend((name, addresses.route_vector(query, vector, embedding_identity=identity,
                max_sections=16, lexical_reserve=reserve, user_weight=weight)) for name, weight, reserve in variants)
            rows.extend({"ordinal": question["ordinal"], "question_id": question["question_id"],
                "query": query, "arm": name, "plan": plan.identity_payload()} for name, plan in plans)
        artifact, _ = publish_sealed_json(root / "routing-probe.json", {
            "addresses_sha256": manifest.sha256, "evaluation_preflight_sha256": evaluation.sha256,
            "variants": variants, "maximum_candidates": 16, "rows": rows,
            "query_embedding_calls": len(questions), "query_vectors_shared_across_diagnostic_variants": True,
            "raw_reads": 0, "provider_calls": 0, "gold_loaded": False, "used_for_target_gate": False})
        print({"addresses_sha256": manifest.sha256, "probe_sha256": artifact.sha256,
               "sections": len(base.sections), "questions": len(questions), "rows": len(rows), "new_calls": 0}, flush=True)
    finally:
        encoder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.index_root, args.evaluation_root, args.output_root)
