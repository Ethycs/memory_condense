"""Reproduce the bounded local-Qwen timing smoke; not a full100 accuracy assay."""
import argparse
import hashlib
from pathlib import Path
import time

from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.assay_user_spine_hierarchy import _turns
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.summary_shortlist_attention import rerank_summary_shortlist
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.eval.streaming_latency import latency_distribution
from memory_condense.application.section_retrieval import hydrate_section_plan


def run(root):
    source = Path("eval_results/user-spine-hierarchy-real-source-20260909-r3")
    hierarchy = read_sealed_json(source / "hierarchy-r4.json")
    questions = read_sealed_json("eval_results/user-spine-real-matched-pilot-20260909-r1/preflight.json")
    raw = read_sealed_json(source / "preflight.json")
    if hierarchy.sha256 != questions.payload["hierarchy_sha256"] or raw.sha256 != questions.payload["source_preflight_sha256"]:
        raise ValueError("pilot bindings changed")
    index = SectionSummaryIndex.from_json(hierarchy.payload["index_json"])
    turns, _ = _turns(raw.payload["binding"])
    lookup = {turn.turn_id: turn for turn in turns}
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "format": "bounded-summary-attention-real-smoke-v1", "hierarchy_sha256": hierarchy.sha256,
        "question_preflight_sha256": questions.sha256, "source_preflight_sha256": raw.sha256,
        "repeats": 5, "shortlist_sections": 8, "selected_sections": 3,
        "precision": "float16 weights and prefix; FP32 softmax and pooled readout",
        "prefix_layers": 6, "head_layer": 5, "max_workspace_tokens": 4096, "provider_calls": 0,
        "gold_loaded": False, "raw_inputs_to_qwen": False, "full100_evaluation": False,
        "implementation_sha256": hashlib.sha256(Path("src/memory_condense/search/summary_shortlist_attention.py").read_bytes()).hexdigest()})
    started = time.perf_counter()
    print("Loading local Qwen prefix for one-pass summary routing...", flush=True)
    encoder = Qwen3PrefixEncoder(Path("../../.cache/models/Qwen3-8B").resolve(), layers=6, device="cuda", dtype="float16")
    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)
    cold = time.perf_counter() - started
    rows = []
    for question in questions.payload["questions"]:
        timings, receipts = [], []
        for iteration in range(6):
            encoder._torch.cuda.synchronize()
            started = time.perf_counter()
            shortlist = index.route(question["question"], max_sections=8)
            plan = rerank_summary_shortlist(question["question"], index, shortlist, linker=linker, max_sections=3)
            hydrated = hydrate_section_plan(plan, load_turn=lookup.get, max_context_tokens=4096, max_raw_spans=128)
            encoder._torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if iteration:
                timings.append(elapsed)
                receipts.append(plan.receipt_sha256)
        row = {"question_id": question["id"], "latency": latency_distribution(timings), "samples_s": timings,
            "candidate_count": len(shortlist.routes),
            "selected_turn_ids": [span.turn_id for route in plan.routes for span in route.section.spans],
            "hydrated_sections": len(hydrated.sections), "stable_receipt": len(set(receipts)) == 1,
            "plan_sha256": plan.receipt_sha256, "model_passes": sum(r.model_passes for r in plan.attention_receipt.rounds)}
        rows.append(row)
        print({k: v for k, v in row.items() if k not in ("samples_s", "plan_sha256")}, flush=True)
    artifact, _ = publish_sealed_json(root / "runtime.json", {
        "format": "bounded-summary-attention-real-smoke-runtime-v1", "preflight_sha256": preflight.sha256,
        "cold_load_s": cold, "rows": rows, "warm": latency_distribution([t for r in rows for t in r["samples_s"]]),
        "provider_calls": 0, "raw_inputs_to_qwen": False, "answer_accuracy_measured": False, "target_gate_passed": False})
    print({"runtime_sha256": artifact.sha256, "cold_load_s": cold, "warm": artifact.payload["warm"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="fresh successor root; timing cannot overwrite an existing measurement")
    run(parser.parse_args().output_root)
