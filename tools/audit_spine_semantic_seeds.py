"""Compare live semantic-seed evidence with the frozen complete-memory control.

This diagnostic recomputes query embeddings and authenticates raw hydration.
It makes no answer or judge calls and supplies no latency or accuracy claim.
An optional saved-rank diagnostic must reproduce the same candidate messages.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from tools import evaluate_spine_combined_reader as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.spine_semantic_seed_memory import ResidentMemory


def evidence(hydrated):
    return {e.span.receipt_sha256: e.span.identity_payload()
        for section in hydrated.sections for e in section.evidence}


def audit(evaluation_root, output, saved_rank_root=None):
    frozen = evaluation.load_preflight(evaluation_root)
    p = frozen.payload
    memory = ResidentMemory(Path(p["index_root"]), p["index_manifest_sha256"],
        Path(p["addresses_root"]), Path(p["atoms_path"]), Path(p["facets_root"]),
        p["addresses_sha256"], p["atoms_sha256"], p["facets_sha256"])
    rows = []
    try:
        for call in p["calls"]:
            if call["arm"] != "base":
                continue
            question = call["question"]
            query, dated = question["retrieval_query"], question["prompt_question"]
            before = memory.retrieve(query, "source_spine_diverse", dated)
            if evaluation.answer_messages(question, before, policy="base") != call["messages"]:
                raise ValueError("the live current control changed")
            after = memory.retrieve(query, "source_spine_semantic_seeds", dated)
            messages = evaluation.answer_messages(question, after, policy="base")
            saved_sha = None
            if saved_rank_root is not None:
                saved = read_sealed_json(saved_rank_root / "candidate-prompts" / f'{question["ordinal"]:03d}.json')
                if (saved.payload["evaluation_preflight_sha256"] != frozen.sha256 or
                        saved.payload["question"] != question or saved.payload["messages"] != messages):
                    raise ValueError("the live candidate differs from the saved-rank reconstruction")
                saved_sha = saved.sha256
            old, new = evidence(before), evidence(after)
            prompt, _ = publish_sealed_json(output / "candidate-prompts" / f'{question["ordinal"]:03d}.json', {
                "evaluation_preflight_sha256": frozen.sha256, "question": question,
                "messages": messages, "messages_sha256": identity_sha256(messages),
                "saved_rank_candidate_sha256": saved_sha, "new_provider_calls": 0})
            rows.append({"ordinal": question["ordinal"], "candidate_prompt_sha256": prompt.sha256,
                "current_control_reproduced": True, "saved_rank_candidate_reproduced": saved_sha is not None,
                "changed": messages != call["messages"],
                "before_context_tokens": before.context_token_count,
                "after_context_tokens": after.context_token_count,
                "added_spans": [new[key] for key in new if key not in old],
                "removed_spans": [old[key] for key in old if key not in new]})
            print({"ordinal": question["ordinal"], "control_reproduced": True,
                "saved_rank_candidate_reproduced": saved_sha is not None,
                "changed": rows[-1]["changed"], "added_spans": len(rows[-1]["added_spans"]),
                "removed_spans": len(rows[-1]["removed_spans"])}, flush=True)
    finally:
        memory.encoder.close()
    if sorted(row["ordinal"] for row in rows) != list(range(p["shard_offset"], p["shard_offset"] + 10)):
        raise ValueError("the diagnostic must cover the complete namespace question population")
    artifact, _ = publish_sealed_json(output / "audit.json", {
        "format": "memory-condense-live-semantic-seed-evidence-audit-v1",
        "evaluation_preflight_sha256": frozen.sha256, "shard_offset": p["shard_offset"],
        "raw_token_proxy": p["raw_token_proxy"], "rows": rows,
        "query_vectors_computed": True, "cached_query_vectors_used_for_live_candidate": False,
        "new_provider_calls": 0, "gold_loaded": False, "predictions_loaded": False,
        "raw_embedding_inputs": False, "answer_accuracy_measured": False, "live_latency_measured": False,
        "full100_target_eligible": False, "target_gate_passed": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            (*evaluation.IMPLEMENTATION, "tools/audit_spine_semantic_seeds.py",
             "tools/spine_semantic_seed_memory.py", "src/memory_condense/search/semantic_spine_seeds.py")}})
    print({"audit_sha256": artifact.sha256, "questions": len(rows), "new_provider_calls": 0}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--saved-rank-root", type=Path)
    args = parser.parse_args()
    audit(args.evaluation_root, args.output_root, args.saved_rank_root)
