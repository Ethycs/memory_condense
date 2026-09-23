"""Authenticate completed ingest responses and admit source-bound routing atoms.

No model calls. Complete namespaces must reproduce every prepared raw span in
order. Partial prefixes are explicitly ineligible for full-memory evaluation.
The stricter quote-check artifacts remain immutable diagnostics.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_source_admission import admit_source_bound_summaries
from memory_condense.search.spine_quote_json_repair_v3 import repair_support_list_closures
from tools.execute_spine_corpus import prepare, fragments_from_request, require
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools import admit_spine_corpus_v5 as previous


def admit(root, offset, limit, summary_repair_root=None):
    if (summary_repair_root is None or
            read_sealed_json(summary_repair_root / "preflight.json").payload.get("format") !=
            "memory-condense-spine-summary-budget-recovery-v2"):
        return previous.admit(root, offset, limit, summary_repair_root)
    manifest, namespace, requests, execution = prepare(root, offset, limit)
    destination = root / f"offset-{offset:03d}"
    summary_repairs = None
    if summary_repair_root is not None:
        from tools.recover_spine_summary_budget_v2 import run
        summary_repairs = run(summary_repair_root, False)
        require(summary_repairs.payload["corpus_preflight_sha256"] == manifest.sha256, "summary repairs belong to another corpus")
    version = "v9"
    policy, _ = publish_sealed_json(destination / f"source-binding-policy-{version}-prefix-{limit:04d}.json", {
        "format": "memory-condense-spine-source-binding-policy-" + version, "corpus_preflight_sha256": manifest.sha256,
        "execution_preflight_sha256": execution.sha256,
        "summary_use": "routing only; never answer evidence", "mandatory_source_binding": "complete exact input fragment",
        "generated_quote_checks": "retain unchanged as diagnostics; not an entailment certificate",
        "summary_text_changes_allowed": "only authenticated over-budget summary compactions" if summary_repairs else False,
        "summary_budget_repairs_sha256": summary_repairs.sha256 if summary_repairs else None, "summary_entailment_verified": False,
        "syntax_repair": "support-string delimiter quote repair only; preserve original response and every summary byte",
        "summary_compaction_batching": "all audited over-budget summaries in deterministic batches of at most eight jobs and 7000 prompt tokens",
        "invalid_slot_recovery": "preserve valid original summaries; at most two summary-only attempts per invalid slot",
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/admit_spine_corpus_v6.py", "src/memory_condense/search/spine_source_admission.py",
            "src/memory_condense/search/spine_quote_json_repair.py", "src/memory_condense/search/spine_quote_json_repair_v2.py", "src/memory_condense/search/spine_quote_json_repair_v3.py")}})
    atoms, audits, responses, compactions = [], [], [], []
    for request in requests:
        payload = request.payload
        runtime = FastCompletionRuntime(checkpoint_dir=destination / "raw-checkpoints" / request.sha256,
            prompt_population=[payload["messages"]], model=payload["model"], client=None,
            max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
            benchmark_provenance={"raw_request_sha256": request.sha256})
        try:
            batch = runtime.run()
        finally:
            runtime.close()
        response = batch.logical_completions[0]
        parsed_response, repair = repair_support_list_closures(response)
        applied = []
        if summary_repairs is not None:
            from tools.repair_spine_summary_budget import apply_repairs
            parsed_response, applied = apply_repairs(parsed_response, request.sha256, summary_repairs.payload["rows"])
            compactions.extend(applied)
        admitted = admit_source_bound_summaries(parsed_response, fragments_from_request(payload), compiler_identity=policy.sha256)
        atoms.extend(admitted.atoms)
        audits.append({"raw_request_sha256": request.sha256, "admission": admitted.identity_payload(),
                       "support_json_syntax_repair": repair, "summary_budget_compactions": applied,
                       "admission_summary_text_basis": "after declared budget compaction" if applied else "original provider summary"})
        responses.append(hashlib.sha256(response.encode()).hexdigest())
    expected_spans = [fragment.span.receipt_sha256 for request in requests for fragment in fragments_from_request(request.payload)]
    observed_spans = [atom.spans[0].receipt_sha256 for atom in atoms]
    require(observed_spans == expected_spans, "admitted atoms do not match the prepared raw partition")
    if summary_repairs is not None:
        require(len(compactions) == len(summary_repairs.payload["rows"]), "not every bound summary compaction was applied exactly once")
    complete = limit == namespace["request_count"]
    if complete:
        require(len(atoms) == namespace["atom_count"] and identity_sha256(observed_spans) == namespace["span_population_sha256"],
                "complete namespace raw coverage failed")
    bad_quotes = sum(bool(row["failures"]) for entry in audits for row in entry["admission"]["quote_diagnostics"])
    artifact, _ = publish_sealed_json(destination / f"source-bound-atoms-prefix-{limit:04d}.json", {
        "format": "memory-condense-spine-source-bound-atoms-v1", "corpus_preflight_sha256": manifest.sha256,
        "source_binding_policy_sha256": policy.sha256, "complete_namespace": complete,
        "atoms": [atom.identity_payload() for atom in atoms], "admission_audits": audits,
        "response_sha256s": responses, "raw_span_population_sha256": identity_sha256(observed_spans),
        "atoms_with_quote_diagnostics": bad_quotes, "summary_texts_unchanged": not compactions,
        "budget_compacted_atoms": len(compactions), "uncompacted_summary_texts_unchanged": True,
        "summary_entailment_verified": False, "new_provider_calls": 0,
        "raw_inputs_to_qwen": False, "hierarchy_constructed": False, "target_gate_passed": False})
    print({"atoms_sha256": artifact.sha256, "atoms": len(atoms), "complete_namespace": complete,
           "atoms_with_quote_diagnostics": bad_quotes, "budget_compacted_atoms": len(compactions),
           "authenticated_completion_hits": len(requests), "new_calls": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--shard-offset", type=int, required=True)
    parser.add_argument("--request-limit", type=int, required=True)
    parser.add_argument("--summary-repair-root", type=Path)
    args = parser.parse_args()
    admit(args.corpus_root, args.shard_offset, args.request_limit, args.summary_repair_root)
