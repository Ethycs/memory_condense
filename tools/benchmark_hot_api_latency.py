"""Paired streaming API diagnostic on the sealed full100 user-spine packets.

This measures prompt/provider cost, NOT live retrieval or the joint 95% gate.
All 100 questions and 200 exact requests are frozen before the first call.
Serial, alternating pair order limits concurrency and order confounds. Journals
never retry an unacknowledged request; recorded timing is not a fresh replay.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval.streaming_latency import measure_streaming_answer, latency_distribution
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.run_hot_reduced30_answer_judge import _completion_client


SOURCE_ROOT = Path("eval_results/longmemeval-1m-hot-v5-user-spine-provider-full100-20260907-r1")
PROBES = Path("eval_results/longmemeval-1m-hot-retrieval-full100-validation-20260905/probes.json")
SOURCE_SHA = "2f78e015b2a9ccca8b5505ea81d1059e8ceebffa2474ac493bd1f2fae54c6928"
PROBE_SHA = "75af9c3faa307a995c134dd9b7b44fd9e94b91d5d4f0a7f8e44ac5fcba9ecfc0"
MODEL = "codex_sdk/gpt-5.6-terra"
GATEWAY = "https://central-dev.zt:4000/v1"
FORMAT = "memory-condense-full100-api-streaming-diagnostic-v1"
IMPLEMENTATION = ("tools/benchmark_hot_api_latency.py", "src/memory_condense/eval/streaming_latency.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bound(path, sha):
    artifact = read_sealed_json(path)
    require(artifact.sha256 == sha, f"artifact changed: {path}")
    return artifact


def prepare(root):
    source = bound(SOURCE_ROOT / "selection.json", SOURCE_SHA)
    probes = bound(PROBES, PROBE_SHA)
    require(source.payload["population_identity_sha256"] == probes.payload["population_identity_sha256"],
            "question populations differ")
    require(len(source.payload["questions"]) == len(probes.payload["questions"]) == 100,
            "expected full100 population")
    calls = []
    for packet, probe in zip(source.payload["questions"], probes.payload["questions"], strict=True):
        require((packet["ordinal"], packet["question_id"], packet["prompt_question_sha256"]) ==
                (probe["ordinal"], probe["question_id"], probe["prompt_question_sha256"]),
                "question order or dated text changed")
        full = packet["arms"]["a3_protected_union"]["provider_messages"]
        # Same question, model, output cap, and answer policy. Removing retrieved
        # evidence yields a short API UX baseline, not a meaningful QA control.
        short = [copy.deepcopy(full[0]), {"role": "user", "content":
                 "Question: " + probe["prompt_question"] + "\nShort answer:"}]
        arms = [("short_api", short), ("packet_api", full)]
        if probe["ordinal"] % 2:
            arms.reverse()
        for arm, messages in arms:
            calls.append({"call_index": len(calls), "question_id": probe["question_id"],
                "ordinal": probe["ordinal"], "shard_offset": probe["shard_offset"],
                "question_sha256": probe["retrieval_query_sha256"], "arm": arm,
                "messages": messages, "messages_identity_sha256": identity_sha256(messages),
                "prompt_token_proxy": count_chat_prompt_token_proxy(messages)})
    manifest = {"format": FORMAT, "source_selection_sha256": source.sha256,
        "probes_sha256": probes.sha256,
        "population_identity_sha256": source.payload["population_identity_sha256"],
        "question_count": 100, "calls": calls, "physical_call_cap": 200,
        "gateway": GATEWAY, "model": MODEL, "max_tokens": 256, "retries": 0,
        "concurrency": 1, "gold_loaded": False, "live_retrieval_measured": False,
        "target_gate_eligible": False, "provisional_latency_ratio_limit": 1.10,
        "baseline": "same system policy and dated question, with no memory evidence; output lengths may differ",
        "packet": "previously sealed exact raw evidence prompt; retrieval and hydration excluded",
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}
    assert_gold_blind(manifest)
    artifact, _ = publish_sealed_json(root / "preflight.json", manifest)
    print({"preflight_sha256": artifact.sha256, "exact_requests": len(calls), "provider_calls": 0})


def load(root):
    artifact = read_sealed_json(root / "preflight.json")
    p = artifact.payload
    require(p["format"] == FORMAT and p["physical_call_cap"] == len(p["calls"]) == 200,
            "invalid frozen call population")
    require(p["gateway"] == GATEWAY and p["model"] == MODEL and p["max_tokens"] == 256,
            "provider configuration changed")
    require(p["implementation"] == {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                   for name in IMPLEMENTATION}, "implementation changed since preflight")
    return artifact


def request_body(preflight, call):
    return {"format": FORMAT + "-request", "preflight_sha256": preflight.sha256,
            "call": call, "stream": True, "max_tokens": 256, "model": MODEL,
            "gateway": GATEWAY, "retries": 0}


def records(root, preflight):
    found = []
    for call in preflight.payload["calls"]:
        prefix = root / "journal" / f'{call["call_index"]:03d}'
        response_path = prefix.with_suffix(".response.json")
        request_path = prefix.with_suffix(".request.json")
        if response_path.exists():
            response = read_sealed_json(response_path)
            request = read_sealed_json(request_path)
            require(request.payload == request_body(preflight, call), "request payload changed")
            require(response.payload["request_sha256"] == request.sha256, "response binding changed")
            require(response.payload["call_index"] == call["call_index"], "response order changed")
            found.append((call, response))
        elif request_path.exists() or prefix.with_suffix(".reserved").exists():
            raise ValueError(f"unacknowledged call {call['call_index']}; preserve this root and inspect before any successor")
    require([c["call_index"] for c, _ in found] == list(range(len(found))), "journal has a gap")
    return found


def run(root, max_calls):
    require(type(max_calls) is int and max_calls > 0, "positive call allowance required")
    preflight = load(root)
    completed = records(root, preflight)
    pending = preflight.payload["calls"][len(completed):]
    require(max_calls <= len(pending), "call allowance exceeds remaining frozen requests")
    client = _completion_client("LITELLM_KEY", GATEWAY)
    try:
        for call in pending[:max_calls]:
            prefix = root / "journal" / f'{call["call_index"]:03d}'
            prefix.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive reservation occurs before outbound I/O. Even a network
            # failure cannot silently spend this request again on resume.
            with prefix.with_suffix(".reserved").open("x", encoding="utf-8") as reservation:
                reservation.write(preflight.sha256 + "\n")
            request, _ = publish_sealed_json(prefix.with_suffix(".request.json"), request_body(preflight, call))
            try:
                result = measure_streaming_answer(client=client, model=MODEL,
                    prepare_prompt=lambda: copy.deepcopy(call["messages"]), max_tokens=256)
            except Exception as exc:
                publish_sealed_json(prefix.with_suffix(".failure.json"), {
                    "request_sha256": request.sha256, "exception_type": type(exc).__name__,
                    "retry_performed": False})
                raise
            publish_sealed_json(prefix.with_suffix(".response.json"), {
                "format": FORMAT + "-response", "request_sha256": request.sha256,
                "call_index": call["call_index"], "observed_at": datetime.now(timezone.utc).isoformat(),
                "measurement": result})
            print({"call_index": call["call_index"], "ordinal": call["ordinal"], "arm": call["arm"],
                   "ttft_s": round(result["api_ttft_s"], 3), "total_s": round(result["api_total_s"], 3),
                   "visible_events": result["visible_event_count"], "finish": result["finish_reason"]}, flush=True)
    finally:
        client.close()
    report(root)


def report(root):
    preflight = load(root)
    observations = records(root, preflight)
    pairs = {}
    for call, response in observations:
        pairs.setdefault(call["question_id"], {})[call["arm"]] = (call, response)
    paired = [p for p in pairs.values() if set(p) == {"short_api", "packet_api"}]
    require(paired, "no complete pairs to report")
    arms = {}
    for arm in ("short_api", "packet_api"):
        rows = [p[arm][1].payload["measurement"] for p in paired]
        arms[arm] = {"questions": len(rows),
            "api_ttft": latency_distribution([r["api_ttft_s"] for r in rows]),
            "api_total": latency_distribution([r["api_total_s"] for r in rows]),
            "single_visible_event_count": sum(r["visible_event_count"] == 1 for r in rows),
            "non_stop_finish_count": sum(r["finish_reason"] != "stop" for r in rows),
            "prompt_token_proxy": latency_distribution([p[arm][0]["prompt_token_proxy"] for p in paired]),
            "usage_reported_count": sum(r["usage"] is not None for r in rows)}
    ratios = {metric: {stat: arms["packet_api"][metric][stat] / arms["short_api"][metric][stat]
                       for stat in ("median_s", "p95_s")}
              for metric in ("api_ttft", "api_total")}
    output = {"format": FORMAT + "-report", "preflight_sha256": preflight.sha256,
        "recorded_physical_calls": len(observations), "complete_pairs": len(paired),
        "latency_observations_are_recorded_live_calls": True, "new_provider_calls_during_report": 0,
        "live_retrieval_measured": False, "accuracy_measured": False, "target_gate_eligible": False,
        "arms": arms, "packet_over_short_ratios": ratios,
        "responses": [{"call_index": c["call_index"], "sha256": r.sha256} for c, r in observations]}
    artifact, _ = publish_sealed_json(root / f"report-{len(observations):03d}.json", output)
    print({"report_sha256": artifact.sha256, "complete_pairs": len(paired), "ratios": ratios,
           "target_gate_eligible": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "report"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root)
    elif args.phase == "run":
        run(args.output_root, args.max_calls)
    else:
        report(args.output_root)


if __name__ == "__main__":
    main()
