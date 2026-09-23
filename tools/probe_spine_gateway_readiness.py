"""Run two bounded synthetic probes without resending benchmark content."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _completion_client, _run_exactly_authorized


READINESS_PREFLIGHT = {
    "format": "memory-condense-spine-gateway-readiness-v1",
    "gateway": "https://central-dev.zt:4000/v1", "timeout_s": 30, "retries": 0,
    "maximum_provider_calls": 2, "max_prompt_tokens": 512, "max_new_tokens": 64,
    "benchmark_questions_used": False, "raw_corpus_used": False,
    "probes": [
        {"name": "qwen", "model": "qwen3-8b", "messages": [
            {"role": "system", "content": "You are testing summary processing. Reply with one short sentence, at most eight words, restating the supplied summary."},
            {"role": "user", "content": "Synthetic summary: A user prefers concise explanations."}],
         "request_options": {"temperature": 0, "extra_body": {"enable_thinking": False}}},
        {"name": "terra", "model": "codex_sdk/gpt-5.6-terra", "messages": [
            {"role": "user", "content": "Readiness check. Reply only with OK."}], "request_options": {}},
    ],
}


def prepare(root):
    preflight, _ = publish_sealed_json(root / "preflight.json", READINESS_PREFLIGHT)
    print({"preflight_sha256": preflight.sha256, "maximum_provider_calls": 2,
           "timeout_s": 30, "new_provider_calls": 0}, flush=True)
    return preflight


def run(root):
    preflight = read_sealed_json(root / "preflight.json")
    if preflight.payload != READINESS_PREFLIGHT:
        raise ValueError("readiness probes must use the fixed synthetic inputs and call limits")
    if (root / "report.json").exists():
        raise ValueError("completed readiness observations are immutable; use a declared fresh probe")
    p = preflight.payload
    results = []
    for probe in p["probes"]:
        checkpoint = root / probe["name"]
        if list(checkpoint.glob("*.request.json")):
            raise ValueError("a readiness request already exists; it will not be resent")
        def factory(client):
            return FastCompletionRuntime(checkpoint_dir=checkpoint, prompt_population=[probe["messages"]],
                model=probe["model"], client=client, max_prompt_tokens=p["max_prompt_tokens"],
                max_new_tokens=p["max_new_tokens"], max_concurrency=1, retries=0,
                request_options=probe["request_options"],
                benchmark_provenance={"readiness_preflight_sha256": preflight.sha256, "probe_name": probe["name"]})
        result = {"probe": probe["name"], "model": probe["model"],
            "observed_utc": datetime.now(timezone.utc).isoformat(), "timeout_s": 30, "automatic_retries": 0}
        try:
            batch, calls, hits, elapsed = _run_exactly_authorized(runtime_factory=factory,
                authorized_provider_calls=1, enable_provider=True,
                client_factory=lambda: _completion_client("LITELLM_KEY", p["gateway"]).with_options(timeout=30))
            result.update({"status": "completed", "new_calls": calls, "replay_hits": hits,
                "elapsed_s": elapsed, "response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records]})
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            result.update({"status": "failed_or_unacknowledged", "error_type": type(exc).__name__,
                "http_status": status if type(status) is int and 100 <= status <= 599 else None,
                "saved_response_count": len(list(checkpoint.glob("*.response.json")))})
        receipt, _ = publish_sealed_json(root / (probe["name"] + "-observation.json"), result)
        results.append({"observation_sha256": receipt.sha256, **result})
        print(results[-1], flush=True)
    report, _ = publish_sealed_json(root / "report.json", {"preflight_sha256": preflight.sha256,
        "results": results, "benchmark_result": False, "original_unresolved_requests_retried": False})
    print({"readiness_report_sha256": report.sha256}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root)
    elif args.enable_provider:
        run(args.output_root)
    else:
        parser.error("readiness execution requires --enable-provider")
