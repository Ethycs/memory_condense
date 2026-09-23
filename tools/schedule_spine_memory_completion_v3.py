"""Compile the remaining whole memories as their existing raw scheduler finishes."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time

import psutil

from tools import finish_spine_memory_namespace_v3 as worker
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


OFFSETS = (80, 90)
RAW_RELEASE_SHA = "073f4813e0cd9d4a3f5ef5a752af37f529c4659028606cc50144d454402a549a"
MAXIMUM_WAIT_SECONDS = 12 * 60 * 60
IMPLEMENTATION = ("tools/schedule_spine_memory_completion_v3.py", *worker.IMPLEMENTATION,
    "tools/run_spine_semantic_seed_full100_v3.py")


def raw_release():
    release = read_sealed_json(worker.RAW_CAMPAIGN / "release.json")
    if release.sha256 != RAW_RELEASE_SHA or release.payload["preflight_sha256"] != worker.RAW_PLAN_SHA:
        raise ValueError("the existing raw scheduler release changed")
    return release


def dependency_alive(release):
    try:
        process = psutil.Process(release.payload["executor_pid"])
        return process.is_running() and process.create_time() == release.payload["executor_process_create_time"]
    except psutil.NoSuchProcess:
        return False


def require_local_capacity(release):
    current = psutil.Process(os.getpid())
    excluded = {current.pid, *(p.pid for p in current.parents())}
    if dependency_alive(release):
        excluded.add(release.payload["executor_pid"])
    workspace = Path.cwd().resolve()
    for process in psutil.process_iter(["pid", "name"]):
        if process.pid in excluded or not (process.info["name"] or "").lower().startswith("python"):
            continue
        try:
            if Path(process.cwd()).resolve() == workspace:
                raise ValueError(f"another local Python job must finish before compilation: PID {process.pid}")
        except psutil.NoSuchProcess:
            continue


def plan_payload(root, release):
    rows = []
    for offset in OFFSETS:
        job = root / f"offset-{offset:03d}"
        plan = read_sealed_json(job / "preflight.json")
        if plan.payload != worker.plan_payload(offset):
            raise ValueError("a prepared memory worker changed")
        rows.append({"offset": offset, "root": str(job.resolve()), "preflight_sha256": plan.sha256})
    return {"format": "memory-condense-scheduled-spine-memory-completion-v3",
        "raw_release_sha256": release.sha256, "workers": rows,
        "maximum_wait_seconds": MAXIMUM_WAIT_SECONDS, "poll_seconds": 10,
        "maximum_concurrent_memory_workers": 1, "maximum_new_raw_calls": 0,
        "answer_calls_allowed": 0, "judge_calls_allowed": 0, "automatic_retries": 0,
        "full100_runner_preparation_only": True,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}


def prepare(root):
    release = raw_release()
    for offset in OFFSETS:
        worker.prepare(root / f"offset-{offset:03d}", offset)
    plan, _ = publish_sealed_json(root / "preflight.json", plan_payload(root, release))
    print({"compilation_schedule_sha256": plan.sha256, "memory_workers": len(OFFSETS),
        "new_provider_calls": 0, "answer_calls_allowed": 0}, flush=True)


def wait_for_raw(offset, binding, release, deadline):
    notified = False
    while True:
        completion = worker.RAW_CAMPAIGN / f"completed-offset-{offset:03d}.json"
        # Publication writes a payload and then its sidecar. A transient payload
        # without its sidecar is not yet a released completion.
        if completion.is_file() and completion.with_suffix(completion.suffix + ".sha256").is_file():
            return worker.completed_raw(offset, binding)
        if (worker.RAW_CAMPAIGN / "failure.json").exists():
            raise ValueError("raw scheduler failed before this whole namespace completed")
        if not dependency_alive(release):
            raise ValueError("the recorded raw scheduler terminated without this namespace completion")
        if time.monotonic() >= deadline:
            raise TimeoutError("the bounded wait for whole raw namespace completion expired")
        if not notified:
            print({"offset": offset, "status": "waiting_for_complete_raw_namespace",
                "raw_executor_pid": release.payload["executor_pid"], "new_provider_calls": 0}, flush=True)
            notified = True
        time.sleep(10)


def run(root, enable=False):
    plan = read_sealed_json(root / "preflight.json")
    release = raw_release()
    if plan.payload != plan_payload(root, release):
        raise ValueError("the prepared compilation schedule changed")
    if not enable:
        raise ValueError("summary compilation requires the provider flag")
    require_local_capacity(release)
    with (root / "execution.reserved").open("x", encoding="utf-8") as handle:
        handle.write(plan.sha256 + "\n")
    publish_sealed_json(root / "release.json", {"preflight_sha256": plan.sha256,
        "raw_release_sha256": release.sha256, "executor_pid": os.getpid(),
        "executor_process_create_time": psutil.Process().create_time(),
        "started_utc": datetime.now(timezone.utc).isoformat()})
    deadline = time.monotonic() + MAXIMUM_WAIT_SECONDS
    completed, phase, offset = [], "waiting for raw completion", None
    try:
        for row in plan.payload["workers"]:
            offset = row["offset"]
            job = Path(row["root"])
            _, _, binding, _, _ = worker.inputs(offset)
            phase = "waiting for raw completion"
            raw = wait_for_raw(offset, binding, release, deadline)
            require_local_capacity(release)
            phase = "memory compilation and request preparation"
            worker.command("tools.finish_spine_memory_namespace_v3", "run", "--output-root", job,
                "--shard-offset", offset, "--enable-provider")
            result = read_sealed_json(job / "complete.json")
            if (result.payload["preflight_sha256"] != row["preflight_sha256"] or
                    result.payload["raw_completion_sha256"] != raw.sha256 or
                    result.payload["offset"] != offset or result.payload["answer_calls_sent"] != 0 or
                    result.payload["judge_calls_sent"] != 0):
                raise ValueError("memory completion is not bound to its complete raw input and worker")
            completed.append({**row, "completion_sha256": result.sha256})
            publish_sealed_json(root / f"completed-offset-{offset:03d}.json", completed[-1])
        phase = "full100 request preparation"
        worker.command("tools.run_spine_semantic_seed_full100_v3", "prepare", "--output-root", worker.CAMPAIGN)
        runner = read_sealed_json(worker.CAMPAIGN / "runner-plan.json")
        if runner.payload["maximum_answer_calls"] != 500 or runner.payload["maximum_logical_judgments"] != 200:
            raise ValueError("the prepared full100 evaluation population changed")
    except Exception as exc:
        publish_sealed_json(root / "failure.json", {"preflight_sha256": plan.sha256, "offset": offset,
            "phase": phase, "exception_type": type(exc).__name__, "completed_namespaces": completed,
            "automatic_retry_performed": False, "original_reservations_preserved": True})
        raise
    result, _ = publish_sealed_json(root / "complete.json", {"preflight_sha256": plan.sha256,
        "completed_namespaces": completed, "full100_runner_plan_sha256": runner.sha256,
        "answer_calls_sent": 0, "judge_calls_sent": 0, "timed_evaluation_requires_separate_idle_release": True})
    print({"compilation_schedule_complete_sha256": result.sha256, "prepared_answer_calls": 500,
        "answer_calls_sent": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.enable_provider:
            parser.error("prepare makes no provider calls")
        prepare(args.output_root)
    else:
        run(args.output_root, args.enable_provider)


