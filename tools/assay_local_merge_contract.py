"""Replay saved merge requests against the current local extraction contract.

The source run used an older prompt whose JSON example showed only index zero.
This assay preserves every typed summary input and changes only the prompt built
by ``LocalRuntime.call``.  It validates generated quotes with the production
``LocalFirstSummaries`` admission rule and never invokes the paid fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from memory_condense.runtime.local import LocalRuntime
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.summary_fallback import LocalFirstSummaries
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _request(saved: dict) -> tuple[list[dict], SpineSummaryRequest]:
    messages = saved["original_messages"]
    payload = json.loads(messages[1]["content"])
    # The neutral production transport deliberately removes occurrence dates.
    # Extraction uses only role/summary ownership, so a fixed explicit marker
    # reconstructs the typed validation boundary without inventing a date.
    fragments = tuple(
        SpineSummaryFragment(row["role"], "date-not-sent-to-extractor", row["summary"])
        for row in payload["fragments"]
    )
    request = SpineSummaryRequest(
        payload["kind"], fragments, user_spine=payload.get("user_spine"),
        max_output_tokens=payload["max_output_tokens"], max_prompt_tokens=16384,
    )
    return messages, request


def run(args) -> dict:
    audit = _read(args.audit)
    cases = [row for row in audit["cases"] if row["kind"] == "merge"]
    if args.cases:
        selected = set(args.cases)
        cases = [row for row in cases if row["case"] in selected]
        missing = selected - {row["case"] for row in cases}
        if missing:
            raise ValueError(f"Unknown cases: {sorted(missing)}")
    runtime = LocalRuntime(args.output / "runtime",
        assets=RuntimeAssets.resolve(args.assets_dir) if args.assets_dir else None)
    rows = []
    started = time.perf_counter()
    try:
        runtime.start_reader()
        for ordinal, prior in enumerate(cases):
            case = prior["case"]
            path = args.calls / f"{case}.request.json"
            saved = _read(path)
            messages, request = _request(saved)
            before = time.perf_counter()
            response = runtime.call(
                "merge", messages, scope="merge-contract-replay",
                max_tokens=saved["max_tokens"], typed_request=request,
            )
            try:
                LocalFirstSummaries._validate("merge", messages, response, request)
                accepted, error = True, None
            except ValueError as exc:
                accepted, error = False, str(exc)
            rows.append({
                "ordinal": ordinal,
                "case": case,
                "source_request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prior_accepted": not prior["escalate"],
                "accepted": accepted,
                "finish_reason": response.get("generated_finish_reason"),
                "replacement_extracts": response.get("local_extractive_fallbacks"),
                "initial_replacement_extracts": response.get(
                    "initial_merge_extractive_fallbacks", 0
                ),
                "local_merge_repair_calls": response.get(
                    "local_merge_repair_calls", 0
                ),
                "deterministic_metadata_extracts": response.get(
                    "deterministic_merge_metadata_extracts", 0
                ),
                "elapsed_s": time.perf_counter() - before,
                "error": error,
                "request_sha256": response.get("request_sha256"),
            })
    finally:
        runtime.close()
    elapsed = time.perf_counter() - started
    latencies = [row["elapsed_s"] for row in rows]
    prior_accepted = sum(row["prior_accepted"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    return {
        "format": "local-merge-contract-replay-v1",
        "boundary": "same saved typed requests; current local extraction prompt; no API fallback",
        "source_audit": str(args.audit),
        "source_calls": str(args.calls),
        "jobs": len(rows),
        "prior_accepted": prior_accepted,
        "prior_acceptance_rate": prior_accepted / len(rows) if rows else 0.0,
        "current_accepted": accepted,
        "current_acceptance_rate": accepted / len(rows) if rows else 0.0,
        "recovered_prior_failures": sum(row["accepted"] and not row["prior_accepted"] for row in rows),
        "regressed_prior_successes": sum(not row["accepted"] and row["prior_accepted"] for row in rows),
        "initial_replacement_extracts": sum(
            row["initial_replacement_extracts"] for row in rows
        ),
        "local_merge_repair_calls": sum(
            row["local_merge_repair_calls"] for row in rows
        ),
        "deterministic_metadata_extracts": sum(
            row["deterministic_metadata_extracts"] for row in rows
        ),
        "mean_job_s": statistics.fmean(latencies) if latencies else 0.0,
        "total_elapsed_s": elapsed,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = run(args)
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    target = args.output / "report.json"
    target.write_bytes(encoded)
    target.with_suffix(".json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  report.json\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
