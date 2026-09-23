"""Prepare the final two namespaces and release the frozen, idle full100 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import psutil

from tools import probe_spine_gateway_readiness as readiness
from tools import run_spine_as_of_full100 as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


CAMPAIGN = Path("eval_results/full1m-spine-as-of-full100-20260910-r1")
COMPILATION = Path("eval_results/full100-spine-memory-completion-20260910-r3")
PROTOCOL_SHA = "09e2b5c8c69a509bc23b2f62e24e0cb53507e30389127db1e6e7d67797ba29ad"
FIRST80_SHA = "43869f83742364a4d108faeaf2a297d6fd47c2aa699f2a4d3ed7ce12a76455cc"
DEPENDENCIES = (
    (runner.BULK, runner.BULK_SHA,
     "073f4813e0cd9d4a3f5ef5a752af37f529c4659028606cc50144d454402a549a"),
    (COMPILATION, "93cb9f4e8c979b228686f70abab8c319185bd155643537d53a296c7882e4fb9b",
     "da12b01028ded4d0a12e4046b183fffa1fe18cb80bf476d8b453c057cec565eb"),
)
IMPLEMENTATION = (
    "tools/run_spine_as_of_after_compilation.py",
    "tools/run_spine_as_of_after_compilation.ps1",
    "tools/prepare_spine_as_of_full100.py",
    *runner.IMPLEMENTATION,
)


def dependencies():
    rows = []
    for root, expected_plan, expected_release in DEPENDENCIES:
        plan = read_sealed_json(root / "preflight.json")
        release = read_sealed_json(root / "release.json")
        p = release.payload
        if (plan.sha256 != expected_plan or release.sha256 != expected_release or
                p["preflight_sha256"] != plan.sha256):
            raise ValueError("the existing raw or compilation release changed")
        rows.append({"root": str(root.resolve()), "preflight_sha256": plan.sha256,
            "release_sha256": release.sha256, "executor_pid": p["executor_pid"],
            "executor_process_create_time": p["executor_process_create_time"]})
    return rows


def first80():
    protocol = read_sealed_json(CAMPAIGN / "protocol.json")
    receipt = read_sealed_json(CAMPAIGN / "preparation-first80.json")
    if (protocol.sha256 != PROTOCOL_SHA or receipt.sha256 != FIRST80_SHA or
            receipt.payload["protocol_sha256"] != protocol.sha256 or
            receipt.payload["prepared_answer_requests"] != 400 or
            [r["offset"] for r in receipt.payload["bindings"]] != list(range(0, 80, 10))):
        raise ValueError("the frozen first eighty questions changed")
    if protocol.payload["implementation"] != {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in runner.PROTOCOL_IMPLEMENTATION}:
        raise ValueError("the frozen full100 evaluation implementation changed")
    for row in receipt.payload["bindings"]:
        prepared = read_sealed_json(CAMPAIGN / "prepared" / f'offset-{row["offset"]:03d}.json')
        root = Path(prepared.payload["root"])
        preflight = runner.evaluation.load_preflight(root)
        if (prepared.sha256 != row["prepared_sha256"] or preflight.sha256 != row["preflight_sha256"] or
                len(preflight.payload["calls"]) != 50 or row["answer_call_count"] != 50):
            raise ValueError("an existing prepared namespace changed")
        runner.require_unstarted([root])
    return protocol, receipt


def require_unprepared_tail():
    for offset in (80, 90):
        if ((CAMPAIGN / "prepared" / f"offset-{offset:03d}.json").exists() or
                (CAMPAIGN / "namespaces" / f"offset-{offset:03d}").exists()):
            raise ValueError("a final namespace is already prepared or partially started")
    if (CAMPAIGN / "runner-plan.json").exists() or (CAMPAIGN / "execution.reserved").exists():
        raise ValueError("the full100 campaign is already prepared or started")


def payload(root):
    protocol, receipt = first80()
    probe = read_sealed_json(root / "readiness" / "preflight.json")
    if probe.payload != readiness.READINESS_PREFLIGHT:
        raise ValueError("the bounded synthetic readiness protocol changed")
    return {"format": "memory-condense-as-of-after-compilation-v1",
        "workspace": str(Path.cwd().resolve()), "campaign_root": str(CAMPAIGN.resolve()),
        "protocol_sha256": protocol.sha256, "first80_preparation_sha256": receipt.sha256,
        "dependencies": dependencies(), "remaining_preparation_offsets": [80, 90],
        "readiness_preflight_sha256": probe.sha256, "maximum_readiness_calls": 2,
        "maximum_answer_calls": 500, "maximum_logical_judgments": 200,
        "maximum_new_raw_calls": 0, "maximum_new_summary_compilation_calls": 0,
        "automatic_retries": 0, "maximum_wait_seconds": 12 * 60 * 60,
        "poll_seconds": 10, "all_answers_before_any_judge": True,
        "timed_concurrency": 1, "implementation": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}


def require_terminal(row):
    try:
        process = psutil.Process(row["executor_pid"])
        if process.is_running() and process.create_time() == row["executor_process_create_time"]:
            raise ValueError("an ingestion or compilation dependency is still running")
    except psutil.NoSuchProcess:
        pass


def require_dependencies_complete(rows):
    completions = []
    for row in rows:
        require_terminal(row)
        root = Path(row["root"])
        if (root / "failure.json").exists():
            raise ValueError("an existing dependency failed; no automatic recovery is allowed")
        complete = read_sealed_json(root / "complete.json")
        if complete.payload["preflight_sha256"] != row["preflight_sha256"]:
            raise ValueError("dependency completion is bound to another execution")
        completions.append(complete)
    plan = read_sealed_json(COMPILATION / "preflight.json")
    completed = completions[1].payload
    if (completed["answer_calls_sent"] != 0 or completed["judge_calls_sent"] != 0 or
            not completed["timed_evaluation_requires_separate_idle_release"] or
            len(completed["completed_namespaces"]) != 2):
        raise ValueError("both whole memory compilations must finish without answer calls")
    for expected, result in zip(plan.payload["workers"], completed["completed_namespaces"], strict=True):
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError("completed memory worker differs from the frozen schedule")
        work = read_sealed_json(Path(expected["root"]) / "complete.json")
        raw = read_sealed_json(runner.BULK / f'completed-offset-{expected["offset"]:03d}.json')
        if (work.sha256 != result["completion_sha256"] or
                work.payload["preflight_sha256"] != expected["preflight_sha256"] or
                work.payload["raw_completion_sha256"] != raw.sha256 or
                work.payload["offset"] != expected["offset"] or
                work.payload["answer_calls_sent"] != 0 or work.payload["judge_calls_sent"] != 0):
            raise ValueError("memory completion is not bound to its whole raw namespace")
    bulk = runner.require_bulk_complete()
    if bulk.sha256 != completions[0].sha256:
        raise ValueError("bulk completion changed during verification")
    return completions


def prepare(root):
    root.resolve().relative_to(Path.cwd().resolve())
    require_unprepared_tail()
    readiness.prepare(root / "readiness")
    plan, _ = publish_sealed_json(root / "preflight.json", payload(root))
    print({"handoff_preflight_sha256": plan.sha256, "remaining_preparation_offsets": [80, 90],
        "maximum_readiness_calls": 2, "maximum_answer_calls": 500,
        "maximum_logical_judgments": 200, "new_provider_calls": 0}, flush=True)


def prepare_namespace(offset):
    # Release each query encoder before starting the next preparation or timing.
    subprocess.run([sys.executable, "-X", "utf8", "-u", "-m", "tools.prepare_spine_as_of_full100",
        "namespace", "--output-root", str(CAMPAIGN), "--offset", str(offset)], check=True)


def run(root, enable=False):
    root.resolve().relative_to(Path.cwd().resolve())
    plan = read_sealed_json(root / "preflight.json")
    if not enable or plan.payload != payload(root):
        raise ValueError("execution requires the unchanged handoff and provider flag")
    require_unprepared_tail()
    completions = require_dependencies_complete(plan.payload["dependencies"])
    runner.require_idle()
    with (root / "execution.reserved").open("x", encoding="utf-8") as handle:
        handle.write(plan.sha256 + "\n")
    publish_sealed_json(root / "release.json", {"preflight_sha256": plan.sha256,
        "dependency_completion_shas": [c.sha256 for c in completions],
        "executor_pid": os.getpid(), "executor_process_create_time": psutil.Process().create_time(),
        "started_utc": datetime.now(timezone.utc).isoformat()})
    phase = "final namespace preparation"
    try:
        for offset in (80, 90):
            prepare_namespace(offset)
        phase = "full100 request verification"
        runner.prepare(CAMPAIGN)
        runner.require_idle()
        phase = "fresh bounded readiness"
        readiness.run(root / "readiness")
        phase = "fresh full100 answers and subsequent judging"
        runner.run(CAMPAIGN, root / "readiness" / "report.json", True)
        complete = read_sealed_json(CAMPAIGN / "complete.json")
    except Exception as exc:
        publish_sealed_json(root / "failure.json", {"preflight_sha256": plan.sha256, "phase": phase,
            "exception_type": type(exc).__name__, "automatic_retry_performed": False,
            "original_reservations_preserved": True})
        raise
    publish_sealed_json(root / "complete.json", {"preflight_sha256": plan.sha256,
        "evaluation_completion_sha256": complete.sha256,
        "target_gate_passed": complete.payload["target_gate_passed"]})
    print({"status": "full100_evaluation_complete", **complete.payload}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.enable_provider:
            parser.error("preparation makes no provider calls")
        prepare(args.output_root)
    else:
        run(args.output_root, args.enable_provider)
