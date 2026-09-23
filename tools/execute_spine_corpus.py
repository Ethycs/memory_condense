"""Execute frozen Terra ingest batches; validate every attributed summary.

Raw requests were prepared from entire authenticated namespaces. Prefixes are
for operational staging, never relevance/score selection. Each request has its
own zero-retry runtime, allowing authenticated replay when a larger prefix is
continued. Invalid summaries remain diagnostics and never enter a hierarchy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import RawSummaryFragment, batch_messages, parse_batch_summaries
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fragments_from_request(payload):
    rows = json.loads(payload["messages"][1]["content"])["fragments"]
    require(len(rows) == len(payload["raw_spans"]), "raw request attribution count changed")
    fragments = []
    for row, value in zip(rows, payload["raw_spans"], strict=True):
        fragment = RawSummaryFragment(RawSectionSpan(**value), row["fragment"])
        fragments.append(fragment)
    require(batch_messages(fragments) == payload["messages"], "raw request message reconstruction changed")
    return tuple(fragments)


def prepare(root, shard_offset, request_limit):
    manifest = read_sealed_json(root / "preflight.json")
    m = manifest.payload
    require(m["format"] == "memory-condense-full100-spine-corpus-preflight-v1", "wrong corpus format")
    require(all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
                for name, sha in m["implementation"].items()), "corpus preparation implementation changed")
    namespace = next(n for n in m["namespaces"] if n["shard_offset"] == shard_offset)
    require(type(request_limit) is int and 1 <= request_limit <= namespace["request_count"], "invalid request prefix")
    requests = []
    for binding in namespace["requests"][:request_limit]:
        path = (root / binding["path"]).resolve()
        path.relative_to(root.resolve())
        request = read_sealed_json(path)
        require(request.sha256 == binding["sha256"], "raw request digest changed")
        require(request.payload["model"] == m["model"] == "codex_sdk/gpt-5.6-terra" and
                request.payload["namespace_database_sha256"] == namespace["database_sha256"], "raw model/source changed")
        fragments_from_request(request.payload)
        requests.append(request)
    plan, _ = publish_sealed_json(root / f"offset-{shard_offset:03d}" / f"execution-prefix-{request_limit:04d}.json", {
        "format": "memory-condense-spine-corpus-execution-v1", "corpus_preflight_sha256": manifest.sha256,
        "namespace_database_sha256": namespace["database_sha256"], "request_limit": request_limit,
        "full_namespace_request_count": namespace["request_count"], "request_shas": [r.sha256 for r in requests],
        "model": m["model"], "gateway": m["gateway"], "retries": 0, "max_concurrency": 4,
        "gold_loaded": False, "raw_input_to_qwen": False,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return manifest, namespace, requests, plan


def execute(root, shard_offset, request_limit, enable_provider):
    manifest, namespace, requests, plan = prepare(root, shard_offset, request_limit)
    destination = root / f"offset-{shard_offset:03d}"
    def one(request):
        payload = request.payload
        fragments = fragments_from_request(payload)
        def factory(client):
            return FastCompletionRuntime(checkpoint_dir=destination / "raw-checkpoints" / request.sha256,
                prompt_population=[payload["messages"]], model=payload["model"], client=client,
                max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
                benchmark_provenance={"raw_request_sha256": request.sha256})
        audit = factory(None)
        try:
            remaining = 1 - len(_authenticated_records(audit))
        finally:
            audit.close()
        batch, calls, hits, elapsed = _run_exactly_authorized(runtime_factory=factory,
            authorized_provider_calls=remaining, enable_provider=enable_provider,
            client_factory=lambda: _completion_client("LITELLM_KEY", manifest.payload["gateway"]))
        try:
            atoms = parse_batch_summaries(batch.logical_completions[0], fragments, compiler_identity=request.sha256)
            validation = {"status": "accepted", "atoms": [atom.identity_payload() for atom in atoms]}
        except (ValueError, TypeError, KeyError) as exc:
            validation = {"status": "invalid_summary", "atoms": [], "error_type": type(exc).__name__, "error": str(exc)}
        output, _ = publish_sealed_json(destination / "validated" / (request.sha256 + ".json"), {
            "format": "memory-condense-spine-corpus-batch-validation-v1", "raw_request_sha256": request.sha256,
            "execution_implementation_sha256": plan.payload["implementation_sha256"],
            "response_sha256": hashlib.sha256(batch.logical_completions[0].encode()).hexdigest(), **validation})
        print({"batch": payload["batch_index"], "status": validation["status"], "atoms": len(validation["atoms"]),
               "new_calls": calls, "replay_hits": hits}, flush=True)
        return output, calls, hits
    # All requests were authenticated before any provider execution.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, requests))
    atoms = [a for result, _, _ in results for a in result.payload["atoms"]]
    invalid = [result.payload["raw_request_sha256"] for result, _, _ in results
               if result.payload["status"] != "accepted"]
    complete = request_limit == namespace["request_count"] and not invalid
    if complete:
        require(len(atoms) == namespace["atom_count"], "accepted namespace lost raw atoms")
    artifact, _ = publish_sealed_json(destination / f"atoms-prefix-{request_limit:04d}.json", {
        "format": "memory-condense-spine-corpus-atoms-v1", "execution_preflight_sha256": plan.sha256,
        "corpus_preflight_sha256": manifest.sha256, "status": "complete_namespace" if complete else "partial_or_invalid",
        "atoms": atoms, "batch_validation_shas": [r.sha256 for r, _, _ in results],
        "invalid_request_shas": invalid, "gold_loaded": False, "raw_input_to_qwen": False,
        "hierarchy_constructed": False, "target_gate_passed": False})
    print({"atoms_sha256": artifact.sha256, "status": artifact.payload["status"], "accepted_atoms": len(atoms),
           "invalid_batches": len(invalid), "new_calls": sum(c for _, c, _ in results), "replay_hits": sum(h for _, _, h in results)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--shard-offset", type=int, required=True)
    parser.add_argument("--request-limit", type=int, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        _, _, _, plan = prepare(args.corpus_root, args.shard_offset, args.request_limit)
        print({"execution_preflight_sha256": plan.sha256, "raw_request_count": args.request_limit, "provider_calls": 0})
    else:
        execute(args.corpus_root, args.shard_offset, args.request_limit, args.enable_provider)
