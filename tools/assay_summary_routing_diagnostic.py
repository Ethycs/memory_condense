"""Frozen synthetic summary-routing diagnostic with balanced candidate orders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer, qwen_linker_identity
from memory_condense.search.indexes.lexical import tokenize
from tools.matched_eval.artifacts import publish_sealed_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/summary_routing_diagnostic_v1.json"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_fixture = args.fixture.read_bytes()
    fixture = json.loads(raw_fixture)
    preflight = {
        "format": "summary-routing-diagnostic-preflight-v1",
        "fixture_sha256": hashlib.sha256(raw_fixture).hexdigest(),
        "fixture": fixture,
        "arms": ["joint_qkov", "independent_qkov", "transport_cosine", "summary_lexical_overlap"],
        "orders": ["forward", "reverse"], "layers": 6, "attention_layer": 5,
        "max_candidates": 4, "max_workspace_tokens": 1024,
        "analysis_policy": "Compare development first; validation is a separate untuned confirmation; no benchmark claim.",
    }
    sealed, _ = publish_sealed_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps({"preflight_sha256": sealed.sha256}), flush=True)
    started = time.perf_counter()
    encoder = Qwen3PrefixEncoder(args.qwen_model_dir.resolve(), layers=6, device="cuda", dtype="float16")
    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=4, max_workspace_tokens=1024)
    scorer = QwenAttentionHeadSurpriseScorer(linker, max_spans=5, span_token_cap=128)
    rows = []
    for group in fixture["groups"]:
        assert len(group["queries"]) == len(group["summaries"]) == 4
        for target, query in enumerate(group["queries"]):
            for order in (list(range(4)), list(reversed(range(4)))):
                candidates = [AssociativeMemoryCandidate(str(i), group["summaries"][i]) for i in order]
                joint = linker.inspect_nested(query, [candidates], beam_per_group=1, top_k=1, score_mode="qk_ov")
                independent = linker.inspect_coverage(query, candidates)
                independent_scores = {hit.episode_id: max(0.0, hit.qk_score) + math.log1p(max(0.0, hit.ov_transport))
                                      for hit in independent.hits}
                signal = scorer.score_sequence([query, *(candidate.text for candidate in candidates)])
                cosine_scores = {str(i): signal.similarities[0][j+1] for j, i in enumerate(order)}
                terms = set(tokenize(query))
                lexical_scores = {str(i): len(terms & set(tokenize(candidate.text))) for i, candidate in zip(order, candidates)}
                selections = {"joint_qkov": joint.hits[0].episode_id,
                              "independent_qkov": max(independent_scores, key=lambda i: (independent_scores[i], -int(i))),
                              "transport_cosine": max(cosine_scores, key=lambda i: (cosine_scores[i], -int(i))),
                              "summary_lexical_overlap": max(lexical_scores, key=lambda i: (lexical_scores[i], -int(i)))}
                rows.append({"group":group["id"], "split":group["split"], "target":target,
                    "query":query, "order":order, "selections":selections,
                    "correct":{arm: int(identity)==target for arm,identity in selections.items()},
                    "independent_scores":independent_scores, "cosine_scores":cosine_scores,
                    "joint_passes":joint.passes, "transport_receipt":signal.receipt.identity_payload()})
                del signal, independent, joint
        print(json.dumps({"group_complete":group["id"], "rows":len(rows)}), flush=True)
    aggregates = {}
    for split in ("development", "validation"):
        selected = [row for row in rows if row["split"] == split]
        aggregates[split] = {"rows":len(selected), "correct":{
            arm:sum(row["correct"][arm] for row in selected) for arm in preflight["arms"]},
            "order_flips":{arm:sum(a["selections"][arm]!=b["selections"][arm]
                for a,b in zip(selected[::2],selected[1::2])) for arm in preflight["arms"]}}
    report = {"format":"summary-routing-diagnostic-result-v1", "preflight_sha256":sealed.sha256,
        "qwen_identity":qwen_linker_identity(linker), "aggregates":aggregates, "rows":rows,
        "external_provider_calls":0, "raw_content_available_to_scorers":False,
        "elapsed_seconds":time.perf_counter()-started}
    artifact,_ = publish_sealed_json(args.output_dir / "result.json", report)
    print(json.dumps({"result_sha256":artifact.sha256, "aggregates":aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
