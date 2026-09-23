"""Compare date-aware packets with frozen semantic seeds on a complete memory.

Fresh local query vectors and exact raw hydration are required. No answer,
judge, reference, or saved prediction is loaded. This is evidence diagnosis;
it cannot establish accuracy or API-relative latency.
"""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.search.summary_time_prior_v2 import question_day
from tools import evaluate_spine_semantic_seeds as evaluation
from tools.matched_eval.artifacts import publish_sealed_json
from tools.spine_as_of_memory import ResidentMemory


IMPLEMENTATION = tuple(dict.fromkeys((*evaluation.IMPLEMENTATION,
    "src/memory_condense/search/summary_time_prior_v2.py",
    "src/memory_condense/search/as_of_spine_routing.py",
    "tools/spine_as_of_memory.py", "tools/audit_spine_as_of.py")))


def evidence(hydrated):
    return {e.span.receipt_sha256: e.span.identity_payload()
            for section in hydrated.sections for e in section.evidence}


def audit(root, output):
    frozen = evaluation.load_preflight(root)
    implementation = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}
    p = frozen.payload
    memory = ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), Path(p["facets_root"]),
        p["addresses_sha256"], p["atoms_sha256"], p["facets_sha256"])
    rows = []
    try:
        for call in p["calls"]:
            if call["arm"] != "semantic_seeds":
                continue
            question = call["question"]
            query, dated = question["retrieval_query"], question["prompt_question"]
            asked = question_day(query, dated)
            before = memory.retrieve(query, "source_spine_semantic_seeds", dated)
            if evaluation.answer_messages(question, before, policy="semantic_seeds") != call["messages"]:
                raise ValueError("the live semantic-seed control changed")
            old = evidence(before)
            old_future = {sha for sha, span in old.items() if datetime.fromisoformat(span["created_at"]).date() > asked}
            for arm, extended in (("as_of", False), ("as_of_relative", True)):
                started = time.perf_counter()
                plan, route_audit = memory.as_of_plan(query, dated, extended_relative_prior=extended)
                after = hydrate_section_plan(plan, load_turn=memory.turns.get, max_context_tokens=3072, max_raw_spans=128)
                elapsed = time.perf_counter() - started
                new = evidence(after)
                if any(datetime.fromisoformat(span["created_at"]).date() > asked for span in new.values()):
                    raise ValueError("future raw evidence escaped as-of routing")
                messages = evaluation.answer_messages(question, after, policy="semantic_seeds")
                prompt, _ = publish_sealed_json(output / "candidate-prompts" / arm / f'{question["ordinal"]:03d}.json', {
                    "evaluation_preflight_sha256": frozen.sha256, "question": question,
                    "arm": arm, "messages": messages, "messages_sha256": identity_sha256(messages),
                    "route_sha256": plan.receipt_sha256, "route_audit": route_audit,
                    "hydrated_spans": list(new.values()), "implementation": implementation,
                    "new_provider_calls": 0, "query_vectors_computed": True})
                rows.append({"ordinal": question["ordinal"], "arm": arm, "asked_day": asked.isoformat(),
                    "candidate_prompt_sha256": prompt.sha256, "semantic_seed_control_reproduced": True,
                    "changed": messages != call["messages"], "before_context_tokens": before.context_token_count,
                    "after_context_tokens": after.context_token_count, "before_raw_span_count": len(old),
                    "before_future_raw_span_count": len(old_future), "after_raw_span_count": len(new),
                    "after_future_raw_span_count": 0,
                    "added_spans": [new[key] for key in new if key not in old],
                    "removed_future_spans": [old[key] for key in old if key not in new and key in old_future],
                    "removed_eligible_spans": [old[key] for key in old if key not in new and key not in old_future],
                    "route_audit": route_audit, "local_diagnostic_elapsed_s": elapsed})
                print({"ordinal": question["ordinal"], "arm": arm,
                    "control_reproduced": True, "future_spans_before": len(old_future), "future_spans_after": 0,
                    "added": len(rows[-1]["added_spans"]),
                    "removed_eligible": len(rows[-1]["removed_eligible_spans"])}, flush=True)
    finally:
        memory.encoder.close()
    if implementation != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}:
        raise ValueError("date diagnostic implementation changed during execution")
    for arm in ("as_of", "as_of_relative"):
        if sorted(r["ordinal"] for r in rows if r["arm"] == arm) != list(range(p["shard_offset"], p["shard_offset"] + 10)):
            raise ValueError("the complete namespace question population is required")
    result, _ = publish_sealed_json(output / "audit.json", {
        "format": "memory-condense-as-of-spine-evidence-audit-v1",
        "evaluation_preflight_sha256": frozen.sha256, "shard_offset": p["shard_offset"],
        "raw_token_proxy": p["raw_token_proxy"], "index_manifest_sha256": p["index_manifest_sha256"],
        "rows": rows, "implementation": implementation,
        "query_vectors_computed": True, "cached_query_vectors_used": False,
        "raw_embedding_inputs": False, "new_provider_calls": 0, "gold_loaded": False, "predictions_loaded": False,
        "answer_accuracy_measured": False, "api_relative_latency_measured": False,
        "local_diagnostic_timing_is_not_an_idle_serving_measurement": True,
        "full100_target_eligible": False, "target_gate_passed": False})
    print({"audit_sha256": result.sha256, "questions": 10, "candidates": 2, "new_provider_calls": 0}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation_root, args.output_root)
