"""Audit all completed raw-ingest responses before source-bound admission."""
import argparse
import hashlib
import json
from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_quote_json_repair_v4 import repair_support_list_closures
from memory_condense.search.spine_source_admission import admit_source_bound_summaries
from tools.execute_spine_corpus import prepare, fragments_from_request
from tools.matched_eval.artifacts import publish_sealed_json


def classify_failure(response, expected_count):
    body = json.loads(response)
    if type(body) is not dict or set(body) != {"atoms"} or type(body["atoms"]) is not list or len(body["atoms"]) != expected_count:
        return [], True
    oversized, schema_failure = [], False
    for i, atom in enumerate(body["atoms"]):
        if (type(atom) is not dict or set(atom) != {"label", "summary", "support"} or
                atom["label"] != f"T{i}" or type(atom["summary"]) is not str or not atom["summary"].strip()):
            schema_failure = True
            continue
        tokens = count_tokens(atom["summary"])
        if tokens > 128:
            oversized.append({"label": atom["label"], "summary": atom["summary"], "token_count": tokens})
    return oversized, schema_failure


def audit(corpus_root, offset, limit, output):
    corpus, _, requests, execution = prepare(corpus_root, offset, limit)
    failures, responses = [], []
    accepted = 0
    for request in requests:
        p = request.payload
        runtime = FastCompletionRuntime(checkpoint_dir=corpus_root / f"offset-{offset:03d}" / "raw-checkpoints" / request.sha256,
            prompt_population=[p["messages"]], model=p["model"], client=None,
            max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
            benchmark_provenance={"raw_request_sha256": request.sha256})
        try:
            batch = runtime.run()
        finally:
            runtime.close()
        responses.extend(r.response_journal_sha256 for r in batch.unique_records)
        response = batch.logical_completions[0]
        fragments = fragments_from_request(p)
        try:
            response, _ = repair_support_list_closures(response)
            admitted = admit_source_bound_summaries(response, fragments, compiler_identity=execution.sha256)
            accepted += len(admitted.atoms)
        except (ValueError, TypeError, KeyError) as exc:
            try:
                invalid, schema_failure = classify_failure(response, len(fragments))
            except (ValueError, TypeError, KeyError):
                invalid, schema_failure = [], True
            failures.append({"request_sha256": request.sha256, "error_type": type(exc).__name__,
                "invalid_summaries": invalid, "unresolved_schema_failure": schema_failure or not invalid})
    result, _ = publish_sealed_json(output, {"corpus_preflight_sha256": corpus.sha256,
        "execution_preflight_sha256": execution.sha256, "request_count": len(requests),
        "accepted_atoms_before_budget_repairs": accepted, "failures": failures,
        "response_journal_sha256s": responses, "new_provider_calls": 0,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/audit_spine_source_admission_v4.py", "src/memory_condense/search/spine_source_admission.py",
            "src/memory_condense/search/spine_quote_json_repair.py", "src/memory_condense/search/spine_quote_json_repair_v2.py", "src/memory_condense/search/spine_quote_json_repair_v3.py", "src/memory_condense/search/spine_quote_json_repair_v4.py")}})
    print({"audit_sha256": result.sha256, "failed_batches": len(failures),
        "oversized_summaries": sum(len(r["invalid_summaries"]) for r in failures),
        "schema_failures": sum(r["unresolved_schema_failure"] for r in failures), "new_provider_calls": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--shard-offset", type=int, required=True)
    parser.add_argument("--request-limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.corpus_root, args.shard_offset, args.request_limit, args.output)
