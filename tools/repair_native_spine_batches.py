"""Prepare and execute explicit failed-atom repairs alongside the initial pass."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import restore
from memory_condense.search.native_spine_repair import partition, reconcile, repair_messages
from tools import compile_native_spine as compiler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import (
    _authenticated_records, _completion_client, _phase_lock, _run_exactly_authorized,
)


MODEL = "codex_sdk/gpt-5.6-terra"
FILES = ("tools/repair_native_spine_batches.py", "src/memory_condense/search/native_spine_repair.py")


def implementation():
    return {**compiler.implementation(), **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                           for name in FILES}}


def original(source, preflight, ordinal):
    p = preflight.payload
    binding = p["requests"][ordinal]
    request = read_sealed_json(source / binding["path"])
    if request.sha256 != binding["sha256"] or request.payload["ordinal"] != ordinal:
        raise ValueError("original raw request changed")
    fragments = restore(request.payload)
    key = identity_sha256({"request_sha256": request.sha256, "model": MODEL})
    validated = read_sealed_json(source / "validated" / f"{key}.json")
    v = validated.payload
    if (v["preflight_sha256"] != preflight.sha256 or v["request_sha256"] != request.sha256
            or v["model"] != MODEL or v["status"] != "invalid_summary"):
        raise ValueError("only terminally rejected original batches may be repaired")
    runtime = FastCompletionRuntime(
        checkpoint_dir=source / "checkpoints" / key, prompt_population=[request.payload["messages"]],
        model=MODEL, client=None, max_prompt_tokens=p["max_prompt_tokens"],
        max_new_tokens=p["max_new_tokens"], max_concurrency=1, retries=0,
        benchmark_provenance={"native_compile_request_sha256": request.sha256},
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    response = batch.logical_completions[0]
    if quote_sha256(response) != v["response_sha256"]:
        raise ValueError("original response changed")
    valid, bad = partition(response, fragments)
    if not bad:
        raise ValueError("original batch has no eligible budget failure")
    return request, validated, fragments, response, valid, bad


def prepare(source, root, ordinals):
    if (not ordinals or len(set(ordinals)) != len(ordinals)
            or any(type(n) is not int or n < 0 for n in ordinals)):
        raise ValueError("explicit unique failed batch ordinals required")
    preflight = read_sealed_json(source / "preflight.json")
    p = preflight.payload
    if (p["implementation"] != compiler.implementation() or p["models"] != [MODEL]
            or p["gateway"] != compiler.GATEWAY or p["raw_inputs_to_qwen"] is not False):
        raise ValueError("original compiler or model changed")
    snapshots, jobs = [], []
    for ordinal in sorted(ordinals):
        request, validated, fragments, response, valid, bad = original(source, preflight, ordinal)
        snapshot, _ = publish_sealed_json(root / "originals" / f"{ordinal:06}.json", {
            "ordinal": ordinal, "source_request_sha256": request.sha256,
            "source_validation_sha256": validated.sha256, "source_response_sha256": quote_sha256(response),
            "unchanged_atom_indices": list(valid), "repair_atom_indices": list(bad),
        })
        snapshots.append({"path": str(snapshot.path.relative_to(root)), "sha256": snapshot.sha256})
        for index in bad:
            prompt = repair_messages(fragments[index])
            job, _ = publish_sealed_json(root / "requests" / f"{ordinal:06}-{index:03}.json", {
                "original_snapshot_sha256": snapshot.sha256, "ordinal": ordinal, "atom_index": index,
                "pointer": fragments[index].pointer(), "messages": prompt,
                "messages_sha256": identity_sha256(prompt),
            })
            jobs.append({"path": str(job.path.relative_to(root)), "sha256": job.sha256})
    result, _ = publish_sealed_json(root / "preflight.json", {
        "format": "native-spine-targeted-budget-repair-v1", "source_root": str(source.resolve()),
        "source_preflight_sha256": preflight.sha256, "sources_sha256": p["sources_sha256"],
        "model": MODEL, "gateway": compiler.GATEWAY, "snapshots": snapshots, "jobs": jobs,
        "max_prompt_tokens": 7000, "max_new_tokens": 512, "concurrency": 1, "retries": 0,
        "maximum_new_provider_calls": len(jobs), "valid_original_summary_changes_allowed": False,
        "raw_inputs_to_qwen": False, "implementation": implementation(),
    })
    print({"repair_preflight_sha256": result.sha256, "original_batches": len(snapshots),
           "maximum_new_provider_calls": len(jobs)}, flush=True)
    return result


def execute(root, enable_provider=False):
    with _phase_lock(root, "native-summary-repair"):
        preflight = read_sealed_json(root / "preflight.json")
        p = preflight.payload
        if (p["implementation"] != implementation() or p["model"] != MODEL
                or p["gateway"] != compiler.GATEWAY or p["raw_inputs_to_qwen"] is not False):
            raise ValueError("repair policy changed")
        source = Path(p["source_root"])
        parent = read_sealed_json(source / "preflight.json")
        if parent.sha256 != p["source_preflight_sha256"]:
            raise ValueError("original full compilation changed")
        loaded = {}
        for binding in p["snapshots"]:
            snapshot = read_sealed_json(root / binding["path"])
            if snapshot.sha256 != binding["sha256"]:
                raise ValueError("original snapshot changed")
            ordinal = snapshot.payload["ordinal"]
            row = original(source, parent, ordinal)
            request, validated, _, response, valid, bad = row
            if snapshot.payload != {
                "ordinal": ordinal, "source_request_sha256": request.sha256,
                "source_validation_sha256": validated.sha256, "source_response_sha256": quote_sha256(response),
                "unchanged_atom_indices": list(valid), "repair_atom_indices": list(bad),
            }:
                raise ValueError("original failed-atom population changed")
            loaded[ordinal] = (snapshot, row)
        jobs, expected = [], {(n, i) for n, (_, row) in loaded.items() for i in row[5]}
        for binding in p["jobs"]:
            job = read_sealed_json(root / binding["path"])
            j = job.payload
            pair = (j["ordinal"], j["atom_index"])
            if job.sha256 != binding["sha256"] or pair not in expected:
                raise ValueError("repair job population changed")
            expected.remove(pair)
            snapshot, row = loaded[pair[0]]
            fragment = row[2][pair[1]]
            if (j["original_snapshot_sha256"] != snapshot.sha256 or j["pointer"] != fragment.pointer()
                    or j["messages"] != repair_messages(fragment)
                    or j["messages_sha256"] != identity_sha256(j["messages"])):
                raise ValueError("repair would send a changed or extra raw fragment")
            jobs.append(job)
        if expected or len(jobs) != p["maximum_new_provider_calls"]:
            raise ValueError("repair does not cover exactly the declared failed atoms")
        replacements = {n: {} for n in loaded}
        calls = hits = 0
        for job in jobs:
            def factory(client):
                return FastCompletionRuntime(
                    checkpoint_dir=root / "checkpoints" / job.sha256,
                    prompt_population=[job.payload["messages"]], model=MODEL, client=client,
                    max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
                    max_concurrency=1, retries=0,
                    benchmark_provenance={"native_repair_request_sha256": job.sha256},
                )
            audit = factory(None)
            try:
                remaining = 1 - len(_authenticated_records(audit))
            finally:
                audit.close()
            batch, new_calls, replay_hits, _ = _run_exactly_authorized(
                runtime_factory=factory, authorized_provider_calls=remaining, enable_provider=enable_provider,
                client_factory=lambda: _completion_client("LITELLM_KEY", compiler.GATEWAY).with_options(
                    timeout=240, max_retries=0),
            )
            replacements[job.payload["ordinal"]][job.payload["atom_index"]] = batch.logical_completions[0]
            calls += new_calls
            hits += replay_hits
        repaired = []
        for ordinal, (snapshot, row) in loaded.items():
            request, validated, fragments, response, valid, bad = row
            summaries = reconcile(response, fragments, replacements[ordinal])
            result, _ = publish_sealed_json(root / "repaired-batches" / f"{ordinal:06}.json", {
                "repair_preflight_sha256": preflight.sha256, "original_snapshot_sha256": snapshot.sha256,
                "source_preflight_sha256": parent.sha256, "source_request_sha256": request.sha256,
                "source_validation_sha256": validated.sha256, "ordinal": ordinal,
                "unchanged_atom_indices": list(valid), "repaired_atom_indices": list(bad),
                "summaries": summaries, "status": "accepted", "raw_spans_unchanged": True,
                "summary_entailment_verified": False,
            })
            repaired.append({"path": str(result.path.relative_to(root)), "sha256": result.sha256})
        result, _ = publish_sealed_json(root / "result.json", {
            "repair_preflight_sha256": preflight.sha256, "source_preflight_sha256": parent.sha256,
            "repaired_batches": repaired, "repaired_atoms": len(jobs),
            "unchanged_valid_atoms": sum(len(row[4]) for _, row in loaded.values()),
            "full_source_compilation_complete": False, "full100_target_passed": False,
        })
        print({"repair_result_sha256": result.sha256, "repaired_batches": len(repaired),
               "repaired_atoms": len(jobs), "new_calls": calls, "replay_hits": hits}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.source_root, args.output_root, args.batch)
    else:
        execute(args.output_root, args.enable_provider)
