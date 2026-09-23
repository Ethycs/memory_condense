"""Execute a staged, bounded transport successor after fresh model readiness.

Original failed reservations stay in place. Every current checkpoint still uses
the zero-retry runtime; another unacknowledged attempt cannot be resumed here.
No staging or request journal is cleared to permit a retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools import execute_spine_corpus, repair_spine_summary_budget
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records
from tools.stage_spine_transport_recovery import journal_state, validate_inventory
from tools.probe_spine_gateway_readiness import READINESS_PREFLIGHT


def require_readiness(path: Path, *, now=None):
    """Replay both readiness responses; a status flag or HTTP liveness is insufficient."""
    report = read_sealed_json(path)
    preflight = read_sealed_json(path.parent / "preflight.json")
    p = preflight.payload
    if report.payload["preflight_sha256"] != preflight.sha256 or p != READINESS_PREFLIGHT:
        raise ValueError("recovery requires the bounded summary-only readiness protocol")
    expected = {"qwen": "qwen3-8b", "terra": "codex_sdk/gpt-5.6-terra"}
    results = report.payload["results"]
    probes = p["probes"]
    if (len(results) != 2 or len(probes) != 2 or
            {r["probe"]: r["model"] for r in results} != expected or
            {r["name"]: r["model"] for r in probes} != expected):
        raise ValueError("both expected model routes must have completed readiness probes")
    current = now or datetime.now(timezone.utc)
    for result in results:
        if result["status"] != "completed":
            raise ValueError("model inference readiness has not passed")
        stamp = datetime.fromisoformat(result["observed_utc"])
        if stamp.tzinfo is None or not 0 <= (current - stamp).total_seconds() <= 300:
            raise ValueError("model readiness must be observed within the last five minutes")
        probe = next(row for row in probes if row["name"] == result["probe"])
        runtime = FastCompletionRuntime(checkpoint_dir=path.parent / probe["name"],
            prompt_population=[probe["messages"]], model=probe["model"], client=None,
            max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
            max_concurrency=1, retries=0, request_options=probe["request_options"],
            benchmark_provenance={"readiness_preflight_sha256": preflight.sha256, "probe_name": probe["name"]})
        try:
            batch = runtime.run()
        finally:
            runtime.close()
        if [r.response_journal_sha256 for r in batch.unique_records] != result["response_journal_shas"]:
            raise ValueError("readiness response binding changed")
    return report


def execute(root: Path, phase: str, readiness_path: Path | None, enable=False):
    if phase not in ("compact", "raw"):
        raise ValueError("unknown recovery phase")
    root = root.resolve()
    stage = read_sealed_json(root / "stage.json")
    preflight = read_sealed_json(root / "stage-preflight.json")
    if stage.payload["stage_preflight_sha256"] != preflight.sha256:
        raise ValueError("recovery stage binding changed")
    original = Path(preflight.payload["original_corpus_root"])
    _, plan, inventory, _ = validate_inventory(original / "transport-recovery-plan-20260910-r1.json")
    if plan.sha256 != stage.payload["recovery_plan_sha256"] or inventory.sha256 != preflight.payload["timeout_inventory_sha256"]:
        raise ValueError("recovery plan or inventory changed")
    from tools import stage_spine_transport_recovery
    if preflight.payload["implementation_sha256"] != hashlib.sha256(
            Path(stage_spine_transport_recovery.__file__).read_bytes()).hexdigest():
        raise ValueError("recovery staging code changed")
    corpus_root = Path(stage.payload["staged_corpus_root"])
    repair_root = Path(stage.payload["staged_compaction_root"])
    if corpus_root.resolve() != root / "corpus" or repair_root.resolve() != root / "summary-repair-offset010":
        raise ValueError("recovery outputs must remain in their staged execution root")
    # Previously successful responses are not eligible for a replacement call.
    for row in stage.payload["copied_completed_requests"]:
        for base in (original, corpus_root):
            checkpoint = base / f"offset-{row['offset']:03d}/raw-checkpoints" / row["raw_request_sha256"]
            state = journal_state(checkpoint)
            if any(row[k] != state[k] for k in state):
                raise ValueError("a retained successful response changed")
    if phase == "compact":
        repair = read_sealed_json(repair_root / "preflight.json")
        if repair.sha256 != plan.payload["summary_compaction"]["preflight_sha256"]:
            raise ValueError("recovery changed the summary compaction request")
        state = journal_state(repair_root / "checkpoints")
        if state["state"] == "reserved_without_response":
            raise RuntimeError("successor compaction is unacknowledged; no further attempt is permitted")
        remaining = int(state["state"] != "completed")
        cap = 1
    else:
        _, _, requests, _ = execute_spine_corpus.prepare(corpus_root, 20, 817)
        remaining = 0
        for request in requests:
            p = request.payload
            runtime = FastCompletionRuntime(checkpoint_dir=corpus_root / "offset-020/raw-checkpoints" / request.sha256,
                prompt_population=[p["messages"]], model=p["model"], client=None,
                max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
                benchmark_provenance={"raw_request_sha256": request.sha256})
            try:
                remaining += 1 - len(_authenticated_records(runtime))
            finally:
                runtime.close()
        cap = len(plan.payload["raw_work"])
    if remaining > cap:
        raise ValueError("recovery misses exceed its explicit additional-call allowance")
    readiness = None
    if remaining:
        if not enable or readiness_path is None:
            raise ValueError("new recovery calls require --enable-provider and fresh successful readiness")
        readiness = require_readiness(readiness_path)
    execution, _ = publish_sealed_json(root / f"execution-{phase}.json", {
        "format": "memory-condense-spine-transport-successor-execution-v1",
        "phase": phase, "stage_sha256": stage.sha256, "recovery_plan_sha256": plan.sha256,
        "maximum_new_calls": cap, "maximum_additional_attempts_per_unresolved_request": 1,
        "automatic_retries": 0, "original_failed_reservations_preserved": True,
        "requires_admission_verifier": "tools.verify_spine_admission_method_v2",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if readiness is not None:
        publish_sealed_json(root / "readiness" / f"{phase}-{readiness.sha256}.json", {
            "execution_preflight_sha256": execution.sha256, "readiness_report_sha256": readiness.sha256,
            "readiness_report_path": str(readiness_path.resolve()), "additional_calls_at_start": remaining})
    if phase == "compact":
        repair_spine_summary_budget.run(repair_root, enable and bool(remaining))
    else:
        execute_spine_corpus.execute(corpus_root, 20, 817, enable and bool(remaining))
    print({"phase": phase, "execution_preflight_sha256": execution.sha256,
        "new_calls": remaining, "original_failed_reservations_preserved": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("compact", "raw"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    execute(args.output_root, args.phase, args.readiness_report, args.enable_provider)
