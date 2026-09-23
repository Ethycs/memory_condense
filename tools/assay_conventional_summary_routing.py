"""Conventional BGE-M3 and BM25 controls on the same frozen summary candidates.

This is a routing-only diagnostic; it does not evaluate the fast answer packet.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from memory_condense.domain.schemas import Turn
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from tools.matched_eval.artifacts import publish_sealed_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    groups, sources = [], []
    for name in ("summary_routing_diagnostic_v1.json", "summary_routing_confirmation_v1.json"):
        path = Path("tests/fixtures") / name
        raw = path.read_bytes()
        sources.append({"path":str(path), "sha256":hashlib.sha256(raw).hexdigest()})
        groups.extend(json.loads(raw)["groups"])
    preflight,_ = publish_sealed_json(args.output_dir / "preflight.json", {
        "format":"conventional-summary-routing-diagnostic-preflight-v1", "fixtures":sources,
        "arms":["bge_m3_cosine", "summary_bm25"], "orders":["forward","reverse"],
        "parameter_selection":"repository defaults, no fitting", "fast_packet_evaluated":False,
    })
    texts = list(dict.fromkeys(text for group in groups for text in group["summaries"] + group["queries"]))
    service = EmbeddingService(device="cuda", batch_size=8, verify_checkpoint=True)
    started = time.perf_counter()
    try:
        matrix = service.embed_queries(texts)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        identity = {**service.execution_identity, "model_id":service.model_name,
                    "model_revision":service.model_revision, "checkpoint_sha256":service.checkpoint_sha256,
                    "post_encode_unit_normalization":True}
    finally:
        service.close()
    encode_seconds = time.perf_counter() - started
    vectors = dict(zip(texts, matrix))
    rows = []
    for group in groups:
        sections = []
        for i, summary in enumerate(group["summaries"]):
            turn = Turn(turn_id=str(i), source_id="synthetic", role="user", text="raw not used",
                        created_at=datetime(2026, 9, 8, tzinfo=timezone.utc))
            sections.append(SectionSummary(str(i), "synthetic", summary, (RawSectionSpan.from_turn(turn),), "fixture"))
        for target, query in enumerate(group["queries"]):
            for order in (list(range(4)), list(reversed(range(4)))):
                index = SectionSummaryIndex([sections[i] for i in order])
                lexical = index.route(query, max_sections=1)
                scores = {i:float(vectors[query] @ vectors[group["summaries"][i]]) for i in order}
                selected = {"bge_m3_cosine":max(scores, key=lambda i:(scores[i],-i)),
                            "summary_bm25":int(lexical.routes[0].section.section_id) if lexical.routes else None}
                rows.append({"group":group["id"], "split":group["split"], "target":target, "order":order,
                    "selections":selected, "correct":{arm:i==target for arm,i in selected.items()}})
    aggregates = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        aggregates[split] = {"count":len(selected), "correct":{
            arm:sum(row["correct"][arm] for row in selected) for arm in ("bge_m3_cosine","summary_bm25")}}
    artifact,_ = publish_sealed_json(args.output_dir / "result.json", {
        "format":"conventional-summary-routing-diagnostic-result-v1", "preflight_sha256":preflight.sha256,
        "aggregates":aggregates,"rows":rows,"embedding_identity":identity,
        "batch_encoding_including_load_and_close_seconds":encode_seconds,
        "new_provider_calls":0,"fast_packet_evaluated":False,
    })
    print(json.dumps({"result_sha256":artifact.sha256,"aggregates":aggregates},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
