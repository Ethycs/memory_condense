"""Authenticate partial native journals and explicitly recorded length recoveries."""
from dataclasses import asdict
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_merges import neutral_key
from tools import compile_native_spine_exchanges as original
from tools import compile_reused_native_spine_exchanges as reused
from tools import recover_native_summary_lengths as recovery
from tools.assemble_native_spine_summaries import digest
from tools.build_spine_corpus_hierarchy import restore_request
from tools.matched_eval.artifacts import read_sealed_json


def validate_projection(key, binding, projection, request, response, plan_sha, backend_sha):
    job = restore_request(binding["request"])
    q, r = request.payload, response.payload
    if (neutral_key(job) != key or job.kind != "attached_context" or job.max_output_tokens != 128
            or binding["messages"] != recovery.messages(job)
            or q["messages"] != binding["messages"] or q["original_merge_key"] != key
            or q["preflight_sha256"] != plan_sha or q["backend_sha256"] != backend_sha
            or r["request_sha256"] != request.sha256 or r["backend_sha256"] != backend_sha
            or r["raw_inputs_to_qwen"] is not False or r["remote_provider_calls"] != 0
            or projection["original_merge_key"] != key
            or projection["request_sha256"] != request.sha256
            or projection["response_sha256"] != response.sha256 or len(r["rows"]) != 1):
        raise ValueError("recovered summary changed its original input or actual model call")
    summary = recovery.validate_row(job, r["rows"][0])
    if projection["summary"] != summary:
        raise ValueError("recovered summary projection differs from the actual model response")
    return summary


def load(root, backend, inputs_sha):
    root = Path(root).resolve()
    plan = read_sealed_json(root/"preflight.json")
    result = read_sealed_json(root/"result.json")
    p, r = plan.payload, result.payload
    if (p["format"] != recovery.FORMAT or p["implementation_sha256"] != digest(recovery.__file__)
            or p["backend_sha256"] != backend.identity_sha256 or p["raw_inputs_to_qwen"] is not False
            or r["preflight_sha256"] != plan.sha256 or r["all_original_limits_met"] is not True
            or r["original_journals_unchanged"] is not True or r["raw_inputs_to_qwen"] is not False
            or r["remote_provider_calls"] != 0 or r["new_local_jobs"] != len(p["jobs"])
            or len(r["projections"]) != len(p["jobs"])):
        raise ValueError("length recovery provenance changed")
    prior_root = Path(p["previous_root"])
    prior = read_sealed_json(prior_root/"preflight.json")
    prior_failure = read_sealed_json(prior_root/"failure.json")
    previous = prior.payload
    source = Path(p["source_root"])
    source_plan = read_sealed_json(source/"preflight.json")
    inputs = read_sealed_json(source/"inputs.json")
    s = source_plan.payload
    if (prior.sha256 != p["previous_preflight_sha256"]
            or prior_failure.sha256 != p["previous_failure_sha256"]
            or prior_failure.payload["preflight_sha256"] != prior.sha256
            or source_plan.sha256 != p["source_preflight_sha256"]
            or previous["source_preflight_sha256"] != source_plan.sha256
            or Path(previous["source_root"]).resolve() != source.resolve()
            or inputs.sha256 != inputs_sha or s["inputs_sha256"] != inputs_sha
            or s["producer_format"] != reused.FORMAT or s["producer_implementation"] != reused.implementation()
            or s["backend_sha256"] != backend.identity_sha256
            or previous["backend_sha256"] != backend.identity_sha256
            or set(previous["jobs"]) != set(p["jobs"])):
        raise ValueError("recovery no longer belongs to these original native exchange inputs")
    actual_files = {str(path.resolve()): read_sealed_json(path).sha256
        for directory in ("requests", "responses") for path in (source/directory).glob("*.json")}
    if actual_files != previous["original_files"]:
        raise ValueError("the stopped native exchange journal changed")
    values, ancestors = reused.reusable_merges([Path(ref["root"]) for ref in s["reuse_roots"]],
        backend, inputs.payload["sources_sha256"], (source.resolve(),))
    if ancestors != s["reuse_roots"] or identity_sha256(values) != s["reused_merge_cache_sha256"]:
        raise ValueError("partial exchange ancestor cache changed")
    journal = original.NeutralJournal(source, source_plan, backend, 0)
    journal.replay()
    for key, summary in journal.cache.values.items():
        if key in values and values[key] != summary:
            raise ValueError("partial exchange summary conflicts with its ancestor")
        values[key] = summary
    projections = {row["original_merge_key"]: row for row in r["projections"]}
    if set(projections) != set(p["jobs"]) or len(projections) != len(r["projections"]):
        raise ValueError("recovery omitted or duplicated an original request")
    for key, binding in p["jobs"].items():
        old = previous["jobs"][key]
        if binding["request"] != old["original_request"] or key in values:
            raise ValueError("recovery altered an original job or replaced an accepted summary")
        if not all((key, attempt) in journal.attempted for attempt in range(3)):
            raise ValueError("recovery source was not an exhausted original request")
        values[key] = validate_projection(key, binding, projections[key],
            read_sealed_json(root/"requests"/f"{key}.json"),
            read_sealed_json(root/"responses"/f"{key}.json"), plan.sha256, backend.identity_sha256)
    if any(key not in values for key, _ in journal.attempted):
        raise ValueError("partial journal has another unresolved job; implicit restarting is forbidden")
    return values, {"root": str(root), "preflight_sha256": plan.sha256, "result_sha256": result.sha256,
        "source_preflight_sha256": source_plan.sha256, "original_journal_sha256": identity_sha256(actual_files),
        "accepted_merge_count": len(values), "merge_cache_sha256": identity_sha256(values)}
