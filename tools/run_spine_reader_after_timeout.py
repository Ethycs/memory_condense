"""Release an unchanged prepared reader comparison after its ingest dependency failed."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import os

import psutil

from tools.execute_spine_transport_recovery import require_readiness
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.stage_spine_transport_recovery_v2 import require_terminal


def context(source_root):
    source_root = source_root.resolve()
    source_root.relative_to(Path.cwd().resolve())
    runner = source_root / "run_comparisons.py"
    spec = importlib.util.spec_from_file_location("prepared_reader_comparison", runner)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    plan = read_sealed_json(source_root / "runner-plan.json")
    terminal = read_sealed_json(source_root / "offset040-terminal-timeout.json")
    p, t = plan.payload, terminal.payload
    if (p["runner_sha256"] != hashlib.sha256(runner.read_bytes()).hexdigest() or
            p["preparation_sha256"] != previous.PREPARATION_SHA or
            p["maximum_answer_calls"] != 240 or p["maximum_logical_judgments"] != 80 or
            p["timed_concurrency"] != 1 or p["retries"] != 0 or
            p["dependency_pid"] != t["ingest_pid"] or
            p["dependency_create_time"] != t["ingest_process_create_time"] or
            t["runner_exit_code"] != 1 or t["runner_stopped_before_answer_calls"] is not True):
        raise ValueError("stopped runner or terminal dependency binding changed")
    require_terminal(terminal)
    roots = previous.prepared_roots()
    return previous, plan, terminal, roots


def payload(source_root, plan, terminal):
    return {"format": "memory-condense-reader-after-terminal-timeout-v1",
        "source_root": str(source_root.resolve()), "predecessor_runner_plan_sha256": plan.sha256,
        "terminal_observation_sha256": terminal.sha256,
        "preparation_sha256": plan.payload["preparation_sha256"],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_answer_calls": 240, "maximum_logical_judgments": 80,
        "timed_concurrency": 1, "automatic_retries": 0,
        "original_prepared_prompts_unchanged": True,
        "schedule": "four complete memories; serial original answer/judge order; no concurrent bulk work",
        "full100_target_eligible": False}


def remaining(previous, roots):
    counts = []
    for root in roots:
        preflight = previous.evaluation.load_preflight(root)
        counts.append(len(preflight.payload["calls"]) - len(previous.evaluation.recorded(root, preflight)))
    return counts


def require_idle():
    """Reject other local Python work before starting a timed comparison."""
    current = psutil.Process(os.getpid())
    excluded = {current.pid, *(p.pid for p in current.parents())}
    workspace = Path.cwd().resolve()
    for process in psutil.process_iter(["pid", "name"]):
        if process.pid in excluded or not (process.info["name"] or "").lower().startswith("python"):
            continue
        try:
            if Path(process.cwd()).resolve() == workspace:
                raise ValueError(f"another Python job is active in this worktree: PID {process.pid}")
        except psutil.NoSuchProcess:
            continue


def prepare(source_root, output_root):
    output_root = output_root.resolve()
    output_root.relative_to(Path.cwd().resolve())
    if output_root == source_root.resolve():
        raise ValueError("the successor scheduler requires a separate output directory")
    previous, old_plan, terminal, roots = context(source_root)
    if remaining(previous, roots) != [60] * 4:
        raise ValueError("prepare requires all 240 original requests to remain unstarted")
    result, _ = publish_sealed_json(output_root / "preflight.json", payload(source_root, old_plan, terminal))
    print({"scheduler_preflight_sha256": result.sha256, "prepared_answer_calls": 240,
           "new_provider_calls": 0}, flush=True)
    return result


def run(output_root, readiness_path, enable=False):
    plan = read_sealed_json(output_root / "preflight.json")
    source_root = Path(plan.payload["source_root"])
    previous, old_plan, terminal, roots = context(source_root)
    if plan.payload != payload(source_root, old_plan, terminal):
        raise ValueError("successor scheduler protocol changed")
    if not enable or readiness_path is None:
        raise ValueError("execution requires the provider flag and fresh successful readiness")
    if remaining(previous, roots) != [60] * 4:
        raise ValueError("a partially executed comparison cannot receive a second release")
    require_idle()
    readiness = require_readiness(readiness_path)
    with (output_root / "execution.reserved").open("x", encoding="utf-8") as handle:
        handle.write(plan.sha256 + "\n")
    publish_sealed_json(output_root / "release.json", {
        "scheduler_preflight_sha256": plan.sha256, "readiness_report_sha256": readiness.sha256,
        "readiness_report_path": str(readiness_path.resolve()),
        "started_utc": datetime.now(timezone.utc).isoformat(), "maximum_answer_calls": 240})
    try:
        for root in roots:
            previous.evaluation.run(root, 60)
            previous.evaluation.judge(root, True)
        previous.build_report(roots)
    except Exception as exc:
        publish_sealed_json(output_root / "failure.json", {
            "scheduler_preflight_sha256": plan.sha256, "exception_type": type(exc).__name__,
            "automatic_retry_performed": False, "original_reservations_preserved": True})
        raise
    report = read_sealed_json(source_root / "development-report.json")
    publish_sealed_json(output_root / "complete.json", {
        "scheduler_preflight_sha256": plan.sha256, "development_report_sha256": report.sha256,
        "maximum_answer_calls": 240, "maximum_logical_judgments": 80,
        "full100_target_eligible": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.source_root is None or args.enable_provider:
            parser.error("prepare requires --source-root and no provider flag")
        prepare(args.source_root, args.output_root)
    else:
        run(args.output_root, args.readiness_report, args.enable_provider)
