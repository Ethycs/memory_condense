"""Re-score an unchanged summary frontier using a separate ordering content view."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from memory_condense.search.summary_query_view import ordered_content_query
from tools.assay_hot_cross_encoder_order_reduced30 import MODEL_DIR
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.probe_spine_cross_encoder import diverse_order, summary_passage, verify_ms_marco_checkpoint


def run(index_root, score_root, root):
    from sentence_transformers import CrossEncoder
    manifest, index = load_index(index_root)
    scores = read_sealed_json(score_root / "scores.json")
    prior = read_sealed_json(score_root / "preflight.json")
    if scores.payload["preflight_sha256"] != prior.sha256 or prior.payload["base_index_sha256"] != manifest.sha256:
        raise ValueError("original frontier binding changed")
    jobs = [(row, ordered_content_query(row["query"])) for row in scores.payload["rows"]]
    jobs = [(row, query) for row, query in jobs if query != row["query"]]
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "scores_sha256": scores.sha256, "original_preflight_sha256": prior.sha256,
        "base_index_sha256": manifest.sha256, "weights_sha256": verify_ms_marco_checkpoint(MODEL_DIR),
        "jobs": [{"ordinal": row["ordinal"], "original_query": row["query"], "content_query": query,
                  "candidate_ids": [r["section_id"] for r in row["ranked"]]} for row, query in jobs],
        "original_answer_query_retained": True, "raw_reads": 0, "provider_calls": 0, "gold_loaded": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ("tools/probe_spine_order_view.py", "src/memory_condense/search/summary_query_view.py",
             "tools/probe_spine_cross_encoder.py")}})
    model = CrossEncoder(str(MODEL_DIR), device="cuda", local_files_only=True, trust_remote_code=False,
                         max_length=512, model_kwargs={"use_safetensors": True})
    by_id = {s.section_id: s for s in index.sections}
    rows = []
    for row, query in jobs:
        sections = [by_id[r["section_id"]] for r in row["ranked"]]
        pairs = [(query, summary_passage(s)) for s in sections]
        if any(len(model.tokenizer(a, b, truncation=False)["input_ids"]) > 512 for a, b in pairs):
            raise ValueError("content-view pair requires truncation")
        start = time.perf_counter()
        values = np.asarray(model.predict(pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)).reshape(-1)
        if len(values) != len(sections) or not np.isfinite(values).all():
            raise ValueError("invalid content-view scores")
        ranked = [{"section_id": s.section_id, "source_id": s.source_id, "score": float(v)}
                  for s, v in zip(sections, values, strict=True)]
        ranked.sort(key=lambda r: (-r["score"], r["section_id"]))
        rows.append({"ordinal": row["ordinal"], "original_query": row["query"], "content_query": query,
                     "ranked": ranked, "source_diverse": diverse_order(ranked), "rerank_s": time.perf_counter()-start})
    artifact, _ = publish_sealed_json(root / "scores.json", {"preflight_sha256": preflight.sha256,
        "rows": rows, "provider_calls": 0, "raw_reads": 0, "gold_loaded": False, "used_for_target_gate": False})
    print({"scores_sha256": artifact.sha256, "changed_queries": len(rows), "new_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.index_root, args.score_root, args.output_root)
