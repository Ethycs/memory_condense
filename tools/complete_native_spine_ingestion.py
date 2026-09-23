"""Repair completed source failures as they arrive, then assemble the full corpus.

This coordinator never dispatches original requests. Its single execution owns
new repair roots only; transport errors stop it without retrying uncertain calls.
"""
from collections import Counter
from contextlib import closing
import argparse
import json
import os
from pathlib import Path
import time

import psutil

from memory_condense.domain._discourse_identity import identity_sha256
from tools import assemble_native_spine_json_recovered as admission
from tools import repair_native_spine_sections as sections
from tools import repair_native_json_batches as malformed
from tools.expanded_native_spine_namespace import verify_extension
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _phase_lock


FORMAT = "native-spine-source-completion-v1"
KINDS = ("legacy_roots", "direct_roots", "recovery_roots", "recovery_section_roots", "json_repair_roots")


def implementation():
    return {**admission.implementation(),
            "tools/expanded_native_spine_namespace.py": admission.base.digest("tools/expanded_native_spine_namespace.py"),
            "tools/complete_native_spine_ingestion.py": admission.base.digest(__file__)}


def verify_seeds(parent, roots):
    repaired, _ = admission.direct.verified_repairs(parent, roots["legacy_roots"], roots["direct_roots"])
    recovered, _ = admission.verified_recoveries(parent, roots["recovery_roots"], roots["recovery_section_roots"])
    json_repaired, _ = admission.verified_json_repairs(parent, roots["json_repair_roots"])
    if repaired.keys() & recovered.keys() or repaired.keys() & json_repaired.keys() or recovered.keys() & json_repaired.keys():
        raise ValueError("initial repair lineages overlap")
    return set(repaired) | set(json_repaired), set(recovered)


def binding(path):
    artifact = read_sealed_json(path)
    return {"path": str(artifact.path.resolve()), "sha256": artifact.sha256}


def bound(row):
    artifact = read_sealed_json(row["path"])
    if artifact.sha256 != row["sha256"]:
        raise ValueError("completion input changed")
    return artifact


def prepare(config, root):
    root = Path(root)
    if root.exists():
        raise ValueError("completion requires a new output root")
    source = Path(config["source_root"]).resolve()
    parent = read_sealed_json(source / "preflight.json")
    p = parent.payload
    public = read_sealed_json(source / "public-source-provenance.json")
    control = Path(config["producer_control"]).resolve()
    producer_policy = read_sealed_json(control / "policy.json")
    started = read_sealed_json(control / "started.json")
    if (p["implementation"] != sections.original_repair.compiler.implementation()
            or p["mode"] != "full" or p["models"] != [sections.MODEL]
            or p["gateway"] != sections.GATEWAY or p["raw_inputs_to_qwen"] is not False
            or p["question_or_gold_inputs"] is not False
            or public.payload["sources_sha256"] != p["sources_sha256"]
            or public.payload["body_bank_sha256"] != p["body_bank_sha256"]
            or public.payload["private_workspace_or_current_conversation_text_in_model_inputs"] is not False
            or producer_policy.payload["preflight_sha256"] != parent.sha256
            or started.payload["policy_sha256"] != producer_policy.sha256):
        raise ValueError("completion source, public scope, or producer binding changed")
    limits = {"maximum_new_provider_calls": 2048, "maximum_refinement_stages": 6,
              "maximum_cohort_batches": 32, "minimum_ready_batches": 16,
              "maximum_wait_seconds": 86400, "poll_seconds": 30, "flush_seconds": 900}
    limits.update(config.get("limits", {}))
    if (any(type(v) is not int or v <= 0 for v in limits.values())
            or limits["minimum_ready_batches"] > limits["maximum_cohort_batches"]
            or limits["poll_seconds"] > 60):
        raise ValueError("invalid completion bounds")
    roots = {kind: [Path(s).resolve() for s in config.get(kind, [])] for kind in KINDS}
    # Reconstruct actual repair journals with no provider available before
    # excluding any original from this coordinator's ownership.
    repaired, recovered = verify_seeds(parent, roots)
    previous = Path(config["previous_store_root"]).resolve()
    with closing(admission.JsonRecoveredSummaryBodies(previous)) as store:
        if store.manifest.payload["sources_sha256"] != p["sources_sha256"]:
            raise ValueError("previous body store belongs to another corpus")
        previous_binding = binding(previous / "summary-bodies.json")
    plan, _ = publish_sealed_json(root / "policy.json", {
        "format": FORMAT, "source_root": str(source), "source_preflight": binding(source / "preflight.json"),
        "public_scope": binding(public.path), "producer_control": str(control),
        "producer_policy": binding(producer_policy.path), "producer_started": binding(started.path),
        "previous_store": previous_binding,
        "seed_roots": {k: [str(p) for p in v] for k, v in roots.items()},
        "seed_results": [binding(p / "result.json") for values in roots.values() for p in values],
        "repaired_ordinals": sorted(repaired), "recovered_ordinals": sorted(recovered),
        "limits": limits, "implementation": implementation(),
        "model": sections.MODEL, "gateway": sections.GATEWAY, "raw_inputs_to_qwen": False,
        "original_requests_dispatched": 0, "automatic_transport_retries": 0,
        "new_local_model_calls": 0, "partial_final_admission_allowed": False,
        "question_or_gold_inputs": False, "full100_target_passed": False,
    })
    print({"completion_policy_sha256": plan.sha256, "previous_repairs": len(repaired),
           "previous_recoveries": len(recovered), "maximum_new_calls": limits["maximum_new_provider_calls"]}, flush=True)
    return plan


def validation_state(source, parent, *, allow_inflight=False):
    by_request = {row["sha256"]: i for i, row in enumerate(parent.payload["requests"])}
    if len(by_request) != len(parent.payload["requests"]):
        raise ValueError("duplicate source request")
    seen, invalid, counts = set(), [], Counter()
    for path in sorted((Path(source) / "validated").glob("*.json")):
        if allow_inflight and not path.with_name(path.name + ".sha256").exists():
            # The producer publishes JSON before its digest sidecar. Revisit an
            # in-flight publication on the next poll; terminal scans are strict.
            continue
        artifact = read_sealed_json(path)
        p = artifact.payload
        ordinal = by_request.get(p["request_sha256"])
        if (ordinal is None or ordinal in seen or p["preflight_sha256"] != parent.sha256
                or p["model"] != sections.MODEL or p["status"] not in {"accepted", "invalid_summary"}
                or path.stem != identity_sha256({"request_sha256": p["request_sha256"], "model": sections.MODEL})):
            raise ValueError("validation escaped the prepared original population")
        seen.add(ordinal)
        counts[p["status"]] += 1
        if p["status"] == "invalid_summary":
            invalid.append({"ordinal": ordinal, "validation_sha256": artifact.sha256})
    return sorted(invalid, key=lambda row: row["ordinal"]), seen, dict(counts)


def producer_finished(control, started, policy_sha):
    try:
        process = psutil.Process(started.payload["pid"])
        if process.create_time() == started.payload["create_time"]:
            return None
    except psutil.NoSuchProcess:
        pass
    # A live handle takes precedence over a receipt. A missing handle requires
    # a matching terminal receipt; an observation timeout is never a restart.
    terminal = read_sealed_json(Path(control) / "finished.json")
    p = terminal.payload
    if (p["policy_sha256"] != policy_sha or p["failed_batches"] != 0
            or p["not_dispatched_batches"] != 0 or p["failures"]):
        raise ValueError("source producer stopped before completing its owned requests")
    return terminal


def classify(source, parent, selected):
    groups = {"direct_roots": [], "json_repair_roots": []}
    for row in selected:
        try:
            _, validation, *_ = sections.original_repair.original(source, parent, row["ordinal"])
        except json.JSONDecodeError:
            # The JSON-specific preparer independently authenticates the actual
            # malformed original and refuses valid JSON or uncertain requests.
            groups["json_repair_roots"].append(row["ordinal"])
        else:
            if validation.sha256 != row["validation_sha256"]:
                raise ValueError("selected original validation changed")
            groups["direct_roots"].append(row["ordinal"])
    return groups


def repair_group(source, group_root, kind, ordinals, *, policy_sha, remaining_calls, max_stages, enable_provider):
    module = sections if kind == "direct_roots" else malformed
    previous, calls = None, 0
    for stage in range(max_stages):
        target = Path(group_root) / f"stage-{stage:02d}"
        if target.exists():
            raise ValueError("repair stage already reserved; no implicit continuation")
        plan = module.prepare(source, target, ordinals, previous=previous)
        maximum = plan.payload["maximum_new_provider_calls"]
        if maximum > remaining_calls - calls:
            raise ValueError("repair would exceed the completion call allowance")
        publish_sealed_json(target / "completion-call-scope.json", {
            "completion_policy_sha256": policy_sha, "preflight_sha256": plan.sha256,
            "maximum_new_provider_calls": maximum, "raw_inputs_to_qwen": False,
            "original_requests_dispatched": 0, "automatic_transport_retries": 0,
        })
        result = module.execute(target, enable_provider)
        calls += maximum
        if result.payload["complete_repair_snapshot"]:
            return target, result, calls
        previous = target
    raise ValueError("repair exhausted bounded exact-subdivision stages")


def require_complete_requests(parent, seen, invalid, repaired, recovered):
    expected = set(range(len(parent.payload["requests"])))
    if seen & recovered or seen | recovered != expected or {r["ordinal"] for r in invalid} != repaired:
        raise ValueError("full corpus still has pending or unrepaired original batches")


def run(root, *, enable_provider=False):
    root = Path(root)
    plan = read_sealed_json(root / "policy.json")
    p = plan.payload
    if p["format"] != FORMAT or p["implementation"] != implementation():
        raise ValueError("completion implementation changed")
    parent = bound(p["source_preflight"])
    bound(p["public_scope"])
    producer_policy = bound(p["producer_policy"])
    started = bound(p["producer_started"])
    bound(p["previous_store"])
    for row in p["seed_results"]:
        bound(row)
    source, limits = Path(p["source_root"]), p["limits"]
    roots = {kind: list(map(Path, p["seed_roots"][kind])) for kind in KINDS}
    repaired, recovered = set(p["repaired_ordinals"]), set(p["recovered_ordinals"])
    with _phase_lock(root, "source-completion"):
        with (root / "execution.reserved").open("x", encoding="utf-8") as handle:
            handle.write(plan.sha256 + "\n")
        process = psutil.Process(os.getpid())
        publish_sealed_json(root / "started.json", {"policy_sha256": plan.sha256,
            "pid": process.pid, "create_time": process.create_time()})
        print({"source_completion_pid": process.pid, "policy_sha256": plan.sha256}, flush=True)
        calls, cohort, beginning, last_flush = 0, 0, time.monotonic(), time.monotonic()
        try:
            while True:
                if time.monotonic() - beginning > limits["maximum_wait_seconds"]:
                    raise TimeoutError("source completion reached its time allowance")
                if (root / "stop-after-current-cohort.flag").exists():
                    raise InterruptedError("explicit stop after current repair cohort")
                terminal = producer_finished(p["producer_control"], started, producer_policy.sha256)
                invalid, seen, counts = validation_state(source, parent, allow_inflight=terminal is None)
                pending = [row for row in invalid if row["ordinal"] not in repaired]
                print({"source_validations": counts, "remaining_rejected_batches": len(pending),
                       "new_repair_calls": calls, "producer_live": terminal is None}, flush=True)
                if pending and (len(pending) >= limits["minimum_ready_batches"] or terminal is not None
                                or time.monotonic() - last_flush >= limits["flush_seconds"]):
                    selected = pending[:limits["maximum_cohort_batches"]]
                    groups = classify(source, parent, selected)
                    group_root = root / "cohorts" / f"{cohort:04d}"
                    selection, _ = publish_sealed_json(group_root / "selection.json", {
                        "policy_sha256": plan.sha256, "selected": selected, "classification": groups})
                    completed = []
                    for kind, ordinals in groups.items():
                        if not ordinals:
                            continue
                        target, result, used = repair_group(source, group_root / kind, kind, ordinals,
                            policy_sha=plan.sha256, remaining_calls=limits["maximum_new_provider_calls"] - calls,
                            max_stages=limits["maximum_refinement_stages"], enable_provider=enable_provider)
                        calls += used
                        roots[kind].append(target)
                        repaired.update(ordinals)
                        completed.append({"kind": kind, "root": str(target.resolve()),
                                          "result_sha256": result.sha256, "ordinals": ordinals})
                    receipt, _ = publish_sealed_json(group_root / "finished.json", {
                        "policy_sha256": plan.sha256, "selection_sha256": selection.sha256,
                        "final_repair_roots": completed, "cumulative_new_provider_calls": calls})
                    print({"repaired_cohort": cohort, "batch_count": len(selected),
                           "cohort_finished_sha256": receipt.sha256, "cumulative_new_calls": calls}, flush=True)
                    cohort += 1
                    last_flush = time.monotonic()
                    continue
                if terminal is not None:
                    require_complete_requests(parent, seen, invalid, repaired, recovered)
                    publish_sealed_json(root / "assembly-inputs.json", {
                        "policy_sha256": plan.sha256, "source_finished_sha256": terminal.sha256,
                        "repair_roots": {k: [str(s.resolve()) for s in v] for k, v in roots.items()},
                        "new_provider_calls": calls, "allow_partial": False})
                    target = root / "complete-body-store"
                    result = admission.assemble(source, target, **roots, allow_partial=False)
                    with (closing(admission.JsonRecoveredSummaryBodies(Path(p["previous_store"]["path"]).parent)) as old,
                          closing(admission.JsonRecoveredSummaryBodies(target)) as current):
                        _, preserved = verify_extension(old, current)
                        if (result.payload["all_prepared_bodies_admitted"] is not True
                                or result.payload["complete_source_compilation"] is not True
                                or result.payload["body_count"] != parent.payload["body_count"]):
                            raise ValueError("assembled store is not the full source corpus")
                    preservation, _ = publish_sealed_json(root / "previous-store-preservation.json", preserved)
                    finished, _ = publish_sealed_json(root / "finished.json", {
                        "policy_sha256": plan.sha256, "source_finished_sha256": terminal.sha256,
                        "summary_store_sha256": result.sha256, "body_count": result.payload["body_count"],
                        "previous_store_preservation_sha256": preservation.sha256,
                        "new_provider_calls": calls, "complete_source_compilation": True,
                        "raw_inputs_to_qwen": False, "new_local_model_calls": 0, "full100_target_passed": False})
                    print({"source_completion_finished_sha256": finished.sha256,
                           "body_count": result.payload["body_count"]}, flush=True)
                    return finished
                time.sleep(limits["poll_seconds"])
        except Exception as error:
            publish_sealed_json(root / "failure.json", {"policy_sha256": plan.sha256,
                "error_type": type(error).__name__, "automatic_retry": False,
                "completed_cohort_count": cohort, "completed_cohort_calls": calls,
                "incomplete_stage_journals_must_be_inspected_before_recovery": True})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(json.loads(args.config.read_text(encoding="utf-8")), args.root)
    else:
        run(args.root, enable_provider=args.enable_provider)
