"""Check semantic admission and live resident cost before full-memory evaluation.

Uses the earlier three-source real hierarchy solely as a mechanism fixture.
Every measured semantic query includes a fresh BGE encoding. Both models remain
resident together; no API calls, question embeddings, or answer gold are cached.
This cannot establish full100 accuracy or the joint latency target.
"""
import argparse
import hashlib
from pathlib import Path
import time

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.eval.streaming_latency import latency_distribution
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from memory_condense.search.summary_shortlist_attention import rerank_summary_shortlist
from tools.assay_user_spine_hierarchy import _turns
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


ARMS = ("summary_bm25", "summary_dense", "summary_hybrid", "dense_qwen", "hybrid_qwen")
IMPLEMENTATION = ("tools/assay_semantic_summary_routing.py", "src/memory_condense/search/summary_semantic_index.py",
                  "src/memory_condense/search/summary_shortlist_attention.py", "src/memory_condense/search/section_routing.py",
                  "src/memory_condense/modeling/embedding.py")


def run(root):
    source = Path("eval_results/user-spine-hierarchy-real-source-20260909-r3")
    hierarchy = read_sealed_json(source / "hierarchy-r4.json")
    questions = read_sealed_json("eval_results/user-spine-real-matched-pilot-20260909-r1/preflight.json")
    raw = read_sealed_json(source / "preflight.json")
    if hierarchy.sha256 != questions.payload["hierarchy_sha256"] or raw.sha256 != questions.payload["source_preflight_sha256"]:
        raise ValueError("real fixture bindings changed")
    index = SectionSummaryIndex.from_json(hierarchy.payload["index_json"])
    turns, _ = _turns(raw.payload["binding"])
    lookup = {turn.turn_id: turn for turn in turns}
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "format": "semantic-summary-routing-mechanism-assay-v1", "hierarchy_sha256": hierarchy.sha256,
        "question_preflight_sha256": questions.sha256, "source_preflight_sha256": raw.sha256,
        "arms": list(ARMS), "warm_repeats": 3, "shortlist_sections": 8, "selected_sections": 3,
        "hybrid_lexical_reserve": 2, "raw_hydration_tokens": 4096, "gold_loaded": False,
        "full100_evaluation": False, "new_provider_calls": 0, "raw_inputs_to_qwen": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    started = time.perf_counter()
    embedder = EmbeddingService(device="cuda", batch_size=8)
    semantic = SemanticSectionIndex.compile(index, encoder=embedder)
    import torch
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - started
    # Free temporary compile buffers; keep the BGE model itself resident.
    torch.cuda.empty_cache()
    print({"semantic_compile_s": compile_s, "leaf_count": len(semantic.sections),
           "vector_bytes": semantic.vector_nbytes}, flush=True)
    started = time.perf_counter()
    encoder = Qwen3PrefixEncoder(Path("../../.cache/models/Qwen3-8B").resolve(), layers=6, device="cuda", dtype="float16")
    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)
    torch.cuda.synchronize()
    qwen_load_s = time.perf_counter() - started
    print({"qwen_load_s": qwen_load_s, "allocated_cuda_bytes": torch.cuda.memory_allocated()}, flush=True)
    rows = []
    try:
        for question in questions.payload["questions"]:
            for arm in ARMS:
                durations, receipts = [], []
                for repeat in range(4):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    query = question["question"]
                    if arm == "summary_bm25":
                        plan = index.route(query, max_sections=3)
                        candidate_count = plan.matched_section_count
                    else:
                        shortlist = semantic.route(query, encoder=embedder,
                            max_sections=8 if arm.endswith("qwen") else 3,
                            lexical_reserve=2 if "hybrid" in arm else 0)
                        candidate_count = len(shortlist.routes)
                        plan = (rerank_summary_shortlist(query, index, shortlist, linker=linker, max_sections=3)
                                if arm.endswith("qwen") else shortlist)
                    hydrated = hydrate_section_plan(plan, load_turn=lookup.get, max_context_tokens=4096, max_raw_spans=128)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    if repeat:
                        durations.append(elapsed)
                        receipts.append(plan.receipt_sha256)
                row = {"case_id": question["id"], "group": question["group"], "arm": arm,
                    "latency": latency_distribution(durations), "samples_s": durations,
                    "candidate_count": candidate_count, "selected_section_ids": [r.section.section_id for r in plan.routes],
                    "selected_sources": list(dict.fromkeys(r.section.source_id for r in plan.routes)),
                    "selected_turn_ids": [s.turn_id for r in plan.routes for s in r.section.spans],
                    "context_tokens": hydrated.context_token_count,
                    "hydration_diagnostics": [d.identity_payload() for d in hydrated.diagnostics],
                    "plan_sha256": plan.receipt_sha256, "stable_route_receipt": len(set(receipts)) == 1}
                rows.append(row)
                print({"case": question["id"], "arm": arm, "median_s": row["latency"]["median_s"],
                       "source_count": len(row["selected_sources"]), "diagnostics": len(hydrated.diagnostics)}, flush=True)
        artifact, _ = publish_sealed_json(root / "runtime.json", {
            "format": "semantic-summary-routing-mechanism-runtime-v1", "preflight_sha256": preflight.sha256,
            "semantic_index_metadata": semantic.metadata_json, "semantic_index_sha256": semantic.receipt_sha256,
            "embedding_identity": summary_embedding_identity(embedder), "compile_s": compile_s, "qwen_load_s": qwen_load_s,
            "models_resident_together": True, "query_encoding_included": True,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "rows": rows,
            "by_arm": {arm: latency_distribution([t for r in rows if r["arm"] == arm for t in r["samples_s"]]) for arm in ARMS},
            "raw_inputs_to_qwen": False, "provider_calls": 0, "answer_accuracy_measured": False, "target_gate_passed": False})
        print({"runtime_sha256": artifact.sha256, "by_arm": artifact.payload["by_arm"]}, flush=True)
    finally:
        embedder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args().output_root)
