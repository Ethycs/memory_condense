"""Execute prepared native summaries with bounded dispatch and reusable receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import uuid

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import admit, restore
from tools import compile_native_spine as compiler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import (
    _authenticated_records, _completion_client, _phase_lock, _run_exactly_authorized,
)


def bounded_dispatch(jobs, one, *, workers):
    """Drain in-flight work after an error, without submitting another job."""
    if type(workers) is not int or workers < 1:
        raise ValueError("positive worker count required")
    pending = iter(enumerate(jobs))
    completed, failures = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}

        def fill():
            while len(futures) < workers:
                try:
                    ordinal, job = next(pending)
                except StopIteration:
                    break
                futures[pool.submit(one, job)] = ordinal

        fill()
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                ordinal = futures.pop(future)
                try:
                    completed[ordinal] = future.result()
                except Exception as exc:
                    # Provider error text may contain request data. Persist only
                    # classification; original request/response journals survive.
                    status = getattr(exc, "status_code", None)
                    failures[ordinal] = {
                        "error_type": type(exc).__name__,
                        "http_status": status if type(status) is int else None,
                    }
            if not failures:
                fill()
    return completed, failures


def load_requests(root, model):
    preflight = read_sealed_json(root / "preflight.json")
    p = preflight.payload
    if (p["implementation"] != compiler.implementation()
            or model not in compiler.MODELS or model not in p["models"]
            or p["gateway"] != compiler.GATEWAY or p["retries"] != 0
            or p["raw_inputs_to_qwen"] is not False
            or p["question_or_gold_inputs"] is not False
            or p["mode"] not in {"probe", "full"}):
        raise ValueError("prepared source compiler contract changed")
    if p["mode"] == "full" and len(p["models"]) != 1:
        raise ValueError("full compilation requires one raw model")
    digest = hashlib.sha256()
    requests, seen, bodies, count = [], set(), set(), 0
    for ordinal, binding in enumerate(p["requests"]):
        path = (root / binding["path"]).resolve()
        path.relative_to((root / "requests").resolve())
        request = read_sealed_json(path)
        payload = request.payload
        if (request.sha256 != binding["sha256"] or request.sha256 in seen
                or payload["sources_sha256"] != p["sources_sha256"]
                or payload["ordinal"] != ordinal
                or identity_sha256(payload["messages"]) != payload["messages_sha256"]):
            raise ValueError("prepared request population changed")
        fragments = restore(payload)
        prompt_tokens = count_chat_prompt_token_proxy(payload["messages"])
        if (len(fragments) != binding["atoms"] or len(fragments) > p["max_atoms"]
                or prompt_tokens != payload["prompt_token_proxy"]
                or prompt_tokens > p["max_prompt_tokens"]):
            raise ValueError("prepared batch budget changed")
        seen.add(request.sha256)
        for fragment in fragments:
            count += 1
            bodies.add(fragment.body_sha256)
            compiler.add_pointer(digest, fragment)
        # Retain only small bindings; authenticate and restore the request again
        # when dispatched instead of holding the entire raw corpus in RAM.
        requests.append(binding)
        if len(requests) % 512 == 0:
            print({"authenticated_requests": len(requests)}, flush=True)
    if (not requests or count != p["fragment_count"] or len(bodies) != p["body_count"]
            or digest.hexdigest() != p["ordered_pointer_sha256"]):
        raise ValueError("complete fragment population changed")
    return preflight, requests


def run_one(root, preflight, model, binding, enable_provider):
    p = preflight.payload
    request = read_sealed_json(root / binding["path"])
    if request.sha256 != binding["sha256"]:
        raise ValueError("request changed after admission")
    fragments = restore(request.payload)
    key = identity_sha256({"request_sha256": request.sha256, "model": model})

    def factory(client):
        return FastCompletionRuntime(
            checkpoint_dir=root / "checkpoints" / key,
            prompt_population=[request.payload["messages"]], model=model, client=client,
            max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
            max_concurrency=1, retries=0,
            benchmark_provenance={"native_compile_request_sha256": request.sha256},
        )

    # Runtime admission rejects a request without a response before creating a
    # client, so an uncertain call cannot be repeated by restarting this runner.
    audit = factory(None)
    try:
        remaining = 1 - len(_authenticated_records(audit))
    finally:
        audit.close()
    batch, calls, hits, _ = _run_exactly_authorized(
        runtime_factory=factory, authorized_provider_calls=remaining,
        enable_provider=enable_provider,
        client_factory=lambda: _completion_client("LITELLM_KEY", compiler.GATEWAY).with_options(
            timeout=p["timeout_s"], max_retries=0),
    )
    response = batch.logical_completions[0]
    try:
        validation = {"status": "accepted", "summaries": admit(response, fragments)}
    except (ValueError, TypeError, KeyError) as exc:
        validation = {"status": "invalid_summary", "summaries": [],
                      "error_type": type(exc).__name__, "error": str(exc)}
    result, _ = publish_sealed_json(root / "validated" / f"{key}.json", {
        "preflight_sha256": preflight.sha256, "request_sha256": request.sha256,
        "model": model, "response_sha256": quote_sha256(response), **validation,
    })
    print({"batch": request.payload["ordinal"], "status": validation["status"],
           "atoms": len(validation["summaries"]), "new_calls": calls,
           "replay_hits": hits}, flush=True)
    return {"validated_sha256": result.sha256, "status": validation["status"],
            "atoms": len(validation["summaries"]), "new_calls": calls, "replay_hits": hits}


def execute(root, model, enable_provider=False):
    with _phase_lock(root, "native-batch-runner"):
        preflight, jobs = load_requests(root, model)
        p = preflight.payload
        policy, _ = publish_sealed_json(root / "bounded-dispatch-policy.json", {
            "preflight_sha256": preflight.sha256, "concurrency": p["concurrency"],
            "stop_new_dispatch_on_execution_error": True, "automatic_retries": 0,
            "unacknowledged_requests_retried": False,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        started = time.perf_counter()
        completed, failures = bounded_dispatch(
            jobs, lambda job: run_one(root, preflight, model, job, enable_provider),
            workers=p["concurrency"],
        )
        rows = [{"ordinal": i, **row} for i, row in sorted(completed.items())]
        accepted = sum(row["status"] == "accepted" for row in rows)
        atoms = sum(row["atoms"] for row in rows)
        complete = (p["mode"] == "full" and not failures and accepted == len(jobs)
                    and atoms == p["fragment_count"])
        payload = {
            "preflight_sha256": preflight.sha256, "dispatch_policy_sha256": policy.sha256,
            "model": model, "prepared_batches": len(jobs), "completed_batches": len(rows),
            "accepted_batches": accepted, "accepted_atoms": atoms,
            "complete_source_compilation": complete, "hierarchies_compiled": False,
            "raw_inputs_to_qwen": False, "summary_entailment_verified": False,
            "full100_target_passed": False, "rows": rows,
            "failures": [{"ordinal": i, **v} for i, v in sorted(failures.items())],
            "new_completed_provider_calls": sum(row["new_calls"] for row in rows),
            "replay_hits": sum(row["replay_hits"] for row in rows),
            "failed_jobs_may_have_unacknowledged_provider_calls": bool(failures),
            "not_dispatched_batches": len(jobs) - len(rows) - len(failures),
            "invocation_seconds": time.perf_counter() - started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        report, _ = publish_sealed_json(root / "executions" / f"{uuid.uuid4().hex}.json", payload)
        if len(rows) == len(jobs) and not failures:
            # Stable result allows a complete replay to compare identical bytes.
            publish_sealed_json(root / f"bounded-result-{identity_sha256(model)}.json", {
                key: value for key, value in payload.items()
                if key not in {"rows", "new_completed_provider_calls", "replay_hits",
                               "invocation_seconds", "finished_utc"}
            } | {"validated_sha256s": [row["validated_sha256"] for row in rows]})
        print({"execution_report": str(report.path), "sha256": report.sha256,
               **{k: payload[k] for k in ("completed_batches", "accepted_batches", "accepted_atoms",
                   "complete_source_compilation", "not_dispatched_batches", "failures")}}, flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    result = execute(args.output_root, args.model, args.enable_provider)
    raise SystemExit(1 if result.payload["failures"] else 0)
