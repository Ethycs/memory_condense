"""Explicit replacement attempts for retired, unanswered source-summary calls.

Original journals remain terminal and untouched. Recovery journals account for
new calls separately; the original provider may also have executed its request.
"""
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionRuntime, FAST_COMPLETION_REQUEST_FORMAT, FAST_COMPLETION_RUNTIME_FORMAT,
    _read_journal, preflight_fast_completion_prompts,
)
from memory_condense.search.native_spine_batch import admit, restore
from tools import run_native_spine_batches as runner
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import (
    _authenticated_records, _completion_client, _phase_lock, _run_exactly_authorized,
)

MODEL = "codex_sdk/gpt-5.6-terra"


def implementation():
    return {**runner.compiler.implementation(), "tools/run_native_spine_batches.py": digest(runner.__file__),
            "tools/recover_native_spine_transport.py": digest(__file__)}


def original_identity(messages, request_sha, policy):
    population = preflight_fast_completion_prompts([messages], max_prompt_tokens=policy["max_prompt_tokens"])
    row, = population.ordered_rows
    runtime = {
        "format": FAST_COMPLETION_RUNTIME_FORMAT, "model": MODEL,
        "max_new_tokens": policy["max_new_tokens"], "max_prompt_token_proxy": population.max_prompt_token_proxy,
        "max_concurrency": 1, "retries": 0, "request_options": {},
        "prompt_population_sha256": population.prompt_population_sha256,
        "prompt_token_proxy_identity": population.prompt_token_proxy_identity,
        "benchmark_provenance": {"native_compile_request_sha256": request_sha},
        "persisted_transformer_token_state": False, "retained_transformer_token_state_bytes": 0,
        "external_provider_persistence_certified": False,
    }
    body = {"format": FAST_COMPLETION_REQUEST_FORMAT, "runtime_identity_sha256": identity_sha256(runtime),
        "prompt_population_sha256": population.prompt_population_sha256,
        "messages_sha256": row.messages_sha256, "prompt_token_proxy": row.prompt_token_proxy,
        "max_new_tokens": policy["max_new_tokens"]}
    return {**body, "call_key_sha256": identity_sha256(body), "runtime_identity": runtime}


def inputs(source, report_path, ordinals):
    source, report_path = Path(source), Path(report_path)
    preflight, report = read_sealed_json(source/"preflight.json"), read_sealed_json(report_path)
    p, r = preflight.payload, report.payload
    if (p["implementation"] != runner.compiler.implementation() or p["models"] != [MODEL]
            or p["gateway"] != runner.compiler.GATEWAY or p["retries"] != 0
            or p["raw_inputs_to_qwen"] is not False or p["question_or_gold_inputs"] is not False
            or r["preflight_sha256"] != preflight.sha256 or r["model"] != MODEL
            or r["failed_jobs_may_have_unacknowledged_provider_calls"] is not True):
        raise ValueError("recovery requires a terminal failed report from the unchanged source compiler")
    failed = {row["ordinal"] for row in r["failures"]}
    if (not ordinals or len(set(ordinals)) != len(ordinals)
            or any(type(i) is not int or not 0 <= i < len(p["requests"]) or i not in failed for i in ordinals)):
        raise ValueError("recovery requires explicit distinct failed original ordinals")
    requests, snapshot = [], []
    for ordinal in sorted(ordinals):
        binding = p["requests"][ordinal]
        path = (source/binding["path"]).resolve()
        path.relative_to((source/"requests").resolve())
        request = read_sealed_json(path)
        if (request.sha256 != binding["sha256"] or request.payload["ordinal"] != ordinal
                or request.payload["sources_sha256"] != p["sources_sha256"]
                or identity_sha256(request.payload["messages"]) != request.payload["messages_sha256"]):
            raise ValueError("recovery original request identity changed")
        fragments = restore(request.payload)
        if len(fragments) != binding["atoms"]:
            raise ValueError("recovery original fragment population changed")
        key = identity_sha256({"request_sha256": request.sha256, "model": MODEL})
        checkpoint = source/"checkpoints"/key
        expected = original_identity(request.payload["messages"], request.sha256, p)
        journal_path = checkpoint/(expected["call_key_sha256"]+".request.json")
        if ((source/"validated"/(key+".json")).exists()
                or set(checkpoint.glob("*.json")) != {journal_path}):
            raise ValueError("recovery may replace only an unanswered original request")
        journal, journal_sha = _read_journal(journal_path)
        if {k: v for k, v in journal.items() if k != "journal_sha256"} != expected:
            raise ValueError("unanswered original journal provenance changed")
        requests.append(request)
        snapshot.append({"ordinal": ordinal, "source_request_sha256": request.sha256,
            "original_journal_path": str(journal_path.resolve()),
            "original_journal_sha256": journal_sha, "original_journal_file_sha256": digest(journal_path),
            "fragment_count": len(fragments)})
    return preflight, report, requests, snapshot


def prepare(source, report_path, root, ordinals):
    root = Path(root)
    parent, report, requests, snapshot = inputs(source, report_path, ordinals)
    plan, _ = publish_sealed_json(root/"preflight.json", {
        "format": "native-spine-explicit-transport-recovery-v1", "source_root": str(Path(source).resolve()),
        "source_report_path": str(Path(report_path).resolve()), "source_report_sha256": report.sha256,
        "source_preflight_sha256": parent.sha256, "sources_sha256": parent.payload["sources_sha256"],
        "ordinals": sorted(ordinals), "originals": snapshot, "model": MODEL,
        "gateway": parent.payload["gateway"], "max_prompt_tokens": parent.payload["max_prompt_tokens"],
        "max_new_tokens": parent.payload["max_new_tokens"], "timeout_s": parent.payload["timeout_s"],
        "maximum_new_provider_calls": len(requests), "original_attempts_with_unknown_execution": len(requests),
        "maximum_combined_original_and_recovery_attempts": 2*len(requests),
        "original_journals_unchanged": True, "same_exact_original_messages": True,
        "raw_inputs_to_qwen": False, "automatic_retries": 0, "implementation": implementation(),
    })
    print({"recovery_preflight_sha256": plan.sha256, "maximum_new_provider_calls": len(requests)}, flush=True)
    return plan


def execute(root, enable_provider=False):
    root = Path(root)
    with _phase_lock(root, "native-explicit-transport-recovery"):
        plan = read_sealed_json(root/"preflight.json")
        p = plan.payload
        if p["implementation"] != implementation():
            raise ValueError("transport recovery implementation changed")
        parent, report, requests, snapshot = inputs(p["source_root"], p["source_report_path"], p["ordinals"])
        if (parent.sha256 != p["source_preflight_sha256"] or report.sha256 != p["source_report_sha256"]
                or snapshot != p["originals"] or len(requests) != p["maximum_new_provider_calls"]
                or p["model"] != MODEL or p["gateway"] != parent.payload["gateway"]
                or any(p[k] != parent.payload[k] for k in ("max_prompt_tokens", "max_new_tokens", "timeout_s"))):
            raise ValueError("transport recovery source scope changed")
        rows, calls, hits = [], 0, 0
        for request in requests:
            def factory(client):
                return FastCompletionRuntime(checkpoint_dir=root/"checkpoints"/request.sha256,
                    prompt_population=[request.payload["messages"]], model=MODEL, client=client,
                    max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
                    max_concurrency=1, retries=0, benchmark_provenance={
                        "native_transport_recovery_sha256": plan.sha256,
                        "original_source_request_sha256": request.sha256})
            runtime = factory(None)
            try:
                remaining = 1-len(_authenticated_records(runtime))
            finally:
                runtime.close()
            batch, new, replay, _ = _run_exactly_authorized(runtime_factory=factory,
                authorized_provider_calls=remaining, enable_provider=enable_provider,
                client_factory=lambda: _completion_client("LITELLM_KEY", p["gateway"]).with_options(
                    timeout=p["timeout_s"], max_retries=0))
            response = batch.logical_completions[0]
            try:
                atoms = list(admit(response, restore(request.payload)))
                status = "accepted"
            except (ValueError, TypeError, KeyError):
                atoms, status = [], "invalid_summary"
            validation, _ = publish_sealed_json(root/"validated"/f'{request.payload["ordinal"]:06d}.json', {
                "recovery_preflight_sha256": plan.sha256, "source_preflight_sha256": parent.sha256,
                "source_request_sha256": request.sha256, "ordinal": request.payload["ordinal"],
                "response_sha256": quote_sha256(response), "status": status, "summaries": atoms,
                "original_journal_unchanged": True, "raw_text_changed": False,
            })
            rows.append({"ordinal": request.payload["ordinal"], "path": str(validation.path.relative_to(root)),
                         "sha256": validation.sha256, "status": status})
            calls += new
            hits += replay
            print({"recovered_original_batch": request.payload["ordinal"], "status": status,
                   "new_calls": new, "replay_hits": replay}, flush=True)
        # Recheck the original absence/identities after execution, too.
        if inputs(p["source_root"], p["source_report_path"], p["ordinals"])[3] != snapshot:
            raise ValueError("original uncertain requests changed during recovery")
        result, _ = publish_sealed_json(root/"result.json", {
            "recovery_preflight_sha256": plan.sha256, "source_preflight_sha256": parent.sha256,
            "rows": rows, "all_recovery_summaries_accepted": all(r["status"] == "accepted" for r in rows),
            "original_attempts_with_unknown_execution": len(rows), "recovery_response_count": len(rows),
            "original_journals_unchanged": True, "full_source_compilation_complete": False,
            "full100_target_passed": False,
        })
        print({"recovery_result_sha256": result.sha256, "new_calls": calls, "replay_hits": hits}, flush=True)
        return result
