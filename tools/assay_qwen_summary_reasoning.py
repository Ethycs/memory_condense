"""Seal, execute and replay a bounded full-Qwen summary-selection diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime, preflight_fast_completion_prompts
from memory_condense.search.summary_reasoning import (
    QWEN_SUMMARY_MODEL, QWEN_SUMMARY_REQUEST_OPTIONS, SummaryChoiceRequest, parse_summary_choice,
)
from tools.matched_eval.artifacts import publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _completion_client, _run_exactly_authorized


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--gateway-url", default="https://central-dev.zt:4000/v1")
    args = parser.parse_args(argv)
    sources, tasks, prompts = [], [], []
    for filename in ("summary_routing_diagnostic_v1.json", "summary_routing_confirmation_v1.json"):
        path = Path("tests/fixtures") / filename
        raw = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for group in json.loads(raw)["groups"]:
            for target, query in enumerate(group["queries"]):
                for order in (list(range(4)), list(reversed(range(4)))):
                    request = SummaryChoiceRequest(query, tuple(group["summaries"][i] for i in order), 1)
                    prompts.append(request.messages)
                    tasks.append({"group":group["id"], "split":group["split"], "target":target,
                                  "order":order, "prompt_sha256":request.prompt_sha256})
    population = preflight_fast_completion_prompts(prompts, max_prompt_tokens=2048)
    code_paths = [Path(__file__), Path("src/memory_condense/search/summary_reasoning.py")]
    preflight,_ = publish_sealed_json(args.output_dir / "preflight.json", {
        "format":"qwen-summary-reasoning-diagnostic-preflight-v1", "fixtures":sources,
        "model":QWEN_SUMMARY_MODEL, "gateway_url":args.gateway_url, "request_options":QWEN_SUMMARY_REQUEST_OPTIONS,
        "prompt_population":population.model_dump(), "tasks":tasks,
        "implementation":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "max_completion_tokens":256, "max_concurrency":4, "retries":0,
        "raw_content_in_prompts":False,
    })
    if args.command == "preflight":
        print(json.dumps({"preflight_sha256":preflight.sha256, "unique_prompts":population.unique_prompt_count}))
        return 0
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=args.output_dir / "checkpoints", prompt_population=prompts,
            model=QWEN_SUMMARY_MODEL, client=client, max_prompt_tokens=2048, max_new_tokens=256,
            max_concurrency=4, retries=0, request_options=QWEN_SUMMARY_REQUEST_OPTIONS,
            benchmark_provenance={"preflight_sha256":preflight.sha256, "gateway_url":args.gateway_url,
                                  "raw_content_in_prompts":False})
    batch,calls,hits,elapsed = _run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=args.authorized_provider_calls, enable_provider=args.enable_provider,
        client_factory=lambda: _completion_client("LITELLM_KEY", args.gateway_url))
    rows=[]
    for task,response in zip(tasks,batch.logical_completions,strict=True):
        try:
            labels=parse_summary_choice(response,candidate_count=4,max_choices=1)
            selected=task["order"][labels[0]] if labels else None
            error=None
        except ValueError as exc:
            selected=None
            error=str(exc)
        rows.append({**task,"selected":selected,"correct":selected==task["target"],"parse_error":error})
    aggregates={}
    for split in sorted({row["split"] for row in rows}):
        selected=[row for row in rows if row["split"]==split]
        aggregates[split]={"count":len(selected),"correct":sum(row["correct"] for row in selected),
            "parse_errors":sum(row["parse_error"] is not None for row in selected),
            "order_flips":sum(a["selected"]!=b["selected"] for a,b in zip(selected[::2],selected[1::2]))}
    usage=batch.usage.model_dump()
    usage.pop("physical_calls")
    usage.pop("checkpoint_hits")
    records=[]
    for record in batch.unique_records:
        body=record.model_dump()
        body.pop("physical_call")
        body.pop("checkpoint_hit")
        records.append(body)
    result,created=publish_sealed_json(args.output_dir / "result.json", {
        "format":"qwen-summary-reasoning-diagnostic-result-v1", "preflight_sha256":preflight.sha256,
        "aggregates":aggregates,"rows":rows,"records":records,"usage":usage,
        "runtime_identity_sha256":batch.runtime_identity_sha256,"raw_content_in_prompts":False,
        "benchmark_accuracy_claim":None,
    })
    print(json.dumps({"created":created,"result_sha256":result.sha256,"new_provider_calls":calls,
                      "checkpoint_hits":hits,"wall_seconds":elapsed,"aggregates":aggregates},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
