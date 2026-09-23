"""Probe a wider summary frontier with the cached conventional MiniLM reranker.

This reads no raw turns and no references. Every candidate summary pair must
fit the model without truncation. Source diversity is measured as an explicit
alternative; it cannot infer evidence from source names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.selectors.cross_encoder_selector import (
    MS_MARCO_MODEL_ID, MS_MARCO_MODEL_REVISION, verify_ms_marco_checkpoint,
)
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.user_spine_addresses import UserSpineAddressIndex
from tools.assay_hot_cross_encoder_order_reduced30 import MODEL_DIR
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def summary_passage(section):
    row = json.loads(section.summary)
    return "User: " + (row["user_spine"] or "Unowned prelude.") + "\nAttached context: " + (
        row["attached_context_not_user_assertions"] or "None.")


def diverse_order(ordered, max_per_source=2):
    counts, front, deferred = {}, [], []
    for row in ordered:
        source = row["source_id"]
        count = counts.get(source, 0)
        if count < max_per_source:
            front.append(row)
            counts[source] = count + 1
        else:
            deferred.append(row)
    return [*front, *deferred]


def run(index_root, address_root, evaluation_root, root):
    from sentence_transformers import CrossEncoder
    parent, base = load_index(index_root)
    channels = read_sealed_json(address_root / "addresses.json")
    matrix_path = address_root / "spine-vectors.npy"
    if (channels.payload["base_index_sha256"] != parent.sha256 or
        channels.payload["matrix_sha256"] != hashlib.sha256(matrix_path.read_bytes()).hexdigest()):
        raise ValueError("user summary address binding changed")
    addresses = UserSpineAddressIndex(base.hierarchy, np.load(index_root / "summary-vectors.npy", allow_pickle=False),
        np.load(matrix_path, allow_pickle=False), embedding_identity=base.embedding_identity)
    if addresses.receipt_sha256 != channels.payload["address_index_sha256"]:
        raise ValueError("user summary address identity changed")
    evaluation = read_sealed_json(evaluation_root / "preflight.json")
    if evaluation.payload["index_manifest_sha256"] != parent.sha256:
        raise ValueError("evaluation memory changed")
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "base_index_sha256": parent.sha256, "addresses_sha256": channels.sha256,
        "evaluation_preflight_sha256": evaluation.sha256, "model_id": MS_MARCO_MODEL_ID,
        "model_revision": MS_MARCO_MODEL_REVISION, "weights_sha256": verify_ms_marco_checkpoint(MODEL_DIR),
        "model_files": {name: hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest() for name in
            ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")},
        "per_dense_channel_candidates": 128, "lexical_candidates": 16, "maximum_union_candidates": 272,
        "max_pair_tokens": 512, "batch_size": 32, "diversity_max_per_source": 2,
        "scored_text": "complete user and attached summaries in plain role framing",
        "raw_reads": 0, "provider_calls": 0, "gold_loaded": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ("tools/probe_spine_cross_encoder.py", "src/memory_condense/search/user_spine_addresses.py",
             "src/memory_condense/search/selectors/cross_encoder_selector.py", "src/memory_condense/modeling/embedding.py")}})
    encoder = EmbeddingService(device="cuda", batch_size=8)
    reranker = CrossEncoder(str(MODEL_DIR), device="cuda", local_files_only=True,
        trust_remote_code=False, max_length=512, model_kwargs={"use_safetensors": True})
    questions = {call["question"]["ordinal"]: call["question"] for call in evaluation.payload["calls"]}
    rows = []
    try:
        encoder.embed_query("Initialize summary retrieval.")
        for question in questions.values():
            query = question["retrieval_query"]
            started = time.perf_counter()
            if summary_embedding_identity(encoder) != base.embedding_identity:
                raise ValueError("query encoder changed")
            vector = encoder.embed_query(query)
            plans = (
                base.route_vector(query, vector, embedding_identity=base.embedding_identity, max_sections=128),
                addresses.route_vector(query, vector, embedding_identity=base.embedding_identity, user_weight=1, max_sections=128),
                addresses.route_vector(query, vector, embedding_identity=base.embedding_identity,
                                       max_sections=16, lexical_reserve=16),
            )
            candidates = {r.section.section_id: r.section for plan in plans for r in plan.routes}
            pairs = [(query, summary_passage(section)) for section in candidates.values()]
            lengths = [len(reranker.tokenizer(a, b, truncation=False)["input_ids"]) for a, b in pairs]
            if max(lengths) > 512:
                raise ValueError("summary pair requires truncation")
            frontier_s = time.perf_counter() - started
            scored_at = time.perf_counter()
            scores = np.asarray(reranker.predict(pairs, batch_size=32, show_progress_bar=False,
                                               convert_to_numpy=True)).reshape(-1)
            scoring_s = time.perf_counter() - scored_at
            if len(scores) != len(candidates) or not np.isfinite(scores).all():
                raise ValueError("missing or invalid summary relevance scores")
            ranked = [{"section_id": section.section_id, "source_id": section.source_id,
                       "summary_sha256": quote_sha256(section.summary), "score": float(score)}
                      for section, score in zip(candidates.values(), scores, strict=True)]
            ranked.sort(key=lambda row: (-row["score"], row["section_id"]))
            rows.append({"ordinal": question["ordinal"], "question_id": question["question_id"],
                "query": query, "candidate_count": len(candidates), "max_pair_tokens": max(lengths),
                "ranked": ranked, "source_diverse": diverse_order(ranked),
                "frontier_s": frontier_s, "rerank_s": scoring_s})
            print({"ordinal": question["ordinal"], "candidates": len(candidates),
                   "frontier_s": frontier_s, "rerank_s": scoring_s}, flush=True)
    finally:
        encoder.close()
    artifact, _ = publish_sealed_json(root / "scores.json", {"preflight_sha256": preflight.sha256,
        "rows": rows, "raw_reads": 0, "provider_calls": 0, "gold_loaded": False,
        "used_for_target_gate": False, "timing_scope": "local diagnostic; no hydration or answer generation"})
    print({"scores_sha256": artifact.sha256, "questions": len(rows), "new_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--address-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.index_root, args.address_root, args.evaluation_root, args.output_root)
