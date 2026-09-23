"""Complete one admitted memory and prepare its unstarted full100 answer calls."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

from tools import run_spine_semantic_seed_full100_v3 as evaluation_runner
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


CAMPAIGN = Path("eval_results/full1m-spine-semantic-seeds-full100-20260910-r3")
PROTOCOL_SHA = "87f5d9ed7314a5341479f2ef2d8357ecb62ff95be6f2e7ecb7c28b7315bfefe3"
RAW_CAMPAIGN = Path("eval_results/full100-spine-after-offset060-timeout-20260910-r1")
RAW_PLAN_SHA = "ed2c488757508298b4f88410713fa0f1d52fea3f334b5009bceb0387ac776033"
IMPLEMENTATION = ("tools/finish_spine_memory_namespace_v3.py", "tools/audit_spine_source_admission_v4.py",
    "tools/repair_spine_summary_budget_v4.py", "tools/finish_spine_compaction_batches.py",
    "tools/recover_spine_summary_budget_v2.py", "tools/admit_spine_corpus_v7.py",
    "tools/verify_spine_admission_method_v11.py", "tools/compile_spine_leaf_projection.py",
    "tools/compile_spine_semantic_index_v2.py", "tools/compile_spine_semantic_index.py", "tools/compile_spine_user_addresses.py",
    "tools/compile_spine_facet_addresses.py", "tools/evaluate_spine_semantic_seeds.py")


def paths_for(offset):
    suffix = f"offset{offset:03d}-20260910-r1"
    return {key: Path("eval_results") / (prefix + suffix) for key, prefix in (
        ("audit", "full1m-spine-source-admission-"), ("compact", "full1m-spine-budget-repair-"),
        ("finish", "full1m-spine-original-compaction-finish-"), ("recovery", "full1m-spine-budget-recovery-"),
        ("leaves", "full1m-spine-leaves-"), ("semantic", "full1m-spine-semantic-"),
        ("users", "full1m-spine-user-addresses-"), ("facets", "full1m-spine-facet-addresses-"),
        ("evaluation", "full1m-spine-semantic-seeds-joint-"))}


def inputs(offset):
    if type(offset) is not int or offset not in (80, 90):
        raise ValueError("only the two remaining full100 namespaces may be compiled here")
    protocol = read_sealed_json(CAMPAIGN / "protocol.json")
    raw = read_sealed_json(RAW_CAMPAIGN / "preflight.json")
    if protocol.sha256 != PROTOCOL_SHA or raw.sha256 != RAW_PLAN_SHA:
        raise ValueError("the frozen full100 or remaining-ingest protocol changed")
    binding = next(b for b in raw.payload["bindings"] if b["offset"] == offset)
    corpus_root = Path(binding["corpus_root"])
    corpus = read_sealed_json(corpus_root / "preflight.json")
    namespace = next(n for n in corpus.payload["namespaces"] if n["shard_offset"] == offset)
    if (corpus.sha256 != raw.payload["corpus_preflight_sha256"] or
            namespace["request_count"] != binding["request_count"] or
            namespace["raw_token_proxy"] != binding["raw_token_proxy"] or binding["raw_token_proxy"] < 1_000_000):
        raise ValueError("memory completion requires the entire prepared 1M-token namespace")
    return protocol, raw, binding, corpus_root, paths_for(offset)


def plan_payload(offset):
    protocol, raw, binding, _, paths = inputs(offset)
    return {"format": "memory-condense-complete-remaining-spine-memory-v3", "offset": offset,
        "protocol_sha256": protocol.sha256, "raw_campaign_preflight_sha256": raw.sha256,
        "raw_binding": binding, "paths": {k: str(v.resolve()) for k, v in paths.items()},
        "maximum_leaf_provider_calls": 64, "compaction_jobs": "every and only audited oversized summary",
        "maximum_compaction_jobs_per_batch": 8, "maximum_recovery_calls_per_invalid_slot": 2,
        "automatic_retries": 0, "raw_inputs_to_qwen": False, "query_independent_compilation": True,
        "answer_calls_allowed": 0, "judge_calls_allowed": 0,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}


def prepare(root, offset):
    plan, _ = publish_sealed_json(root / "preflight.json", plan_payload(offset))
    print({"memory_completion_preflight_sha256": plan.sha256, "offset": offset,
        "new_provider_calls": 0, "answer_calls_allowed": 0}, flush=True)


def completed_raw(offset, binding):
    completed = read_sealed_json(RAW_CAMPAIGN / f"completed-offset-{offset:03d}.json")
    if any(completed.payload.get(k) != v for k, v in binding.items()):
        raise ValueError("completed raw namespace differs from the prepared request population")
    corpus = Path(binding["corpus_root"])
    raw = read_sealed_json(corpus / f"offset-{offset:03d}" / f'atoms-prefix-{binding["request_count"]:04d}.json')
    if (raw.sha256 != completed.payload["raw_completion_sha256"] or
            raw.payload["execution_preflight_sha256"] != binding["execution_preflight_sha256"] or
            len(raw.payload["batch_validation_shas"]) != binding["request_count"]):
        raise ValueError("raw completion must contain every prepared response")
    return completed


def command(module, *args):
    # A separate process releases the attention model before BGE compilation.
    subprocess.run([sys.executable, "-X", "utf8", "-m", module, *(str(a) for a in args)], check=True)


def run(root, offset, enable=False):
    plan = read_sealed_json(root / "preflight.json")
    if plan.payload != plan_payload(offset):
        raise ValueError("memory completion implementation or inputs changed")
    protocol, _, binding, corpus_root, paths = inputs(offset)
    raw_completion = completed_raw(offset, binding)
    if not enable:
        raise ValueError("summary compilation requires the provider flag after complete raw admission inputs")
    with (root / "execution.reserved").open("x", encoding="utf-8") as handle:
        handle.write(plan.sha256 + "\n")
    phase = "audit"
    try:
        common = ("--corpus-root", corpus_root, "--shard-offset", offset, "--request-limit", binding["request_count"])
        command("tools.audit_spine_source_admission_v4", *common, "--output", paths["audit"] / "audit.json")
        audit = read_sealed_json(paths["audit"] / "audit.json")
        if any(r["unresolved_schema_failure"] for r in audit.payload["failures"]):
            raise ValueError("source schema failures require diagnosis before summary compaction")
        repair_root = None
        if audit.payload["failures"]:
            phase = "summary compaction"
            command("tools.repair_spine_summary_budget_v4", "prepare", "--output-root", paths["compact"],
                *common, "--audit", paths["audit"] / "audit.json")
            command("tools.finish_spine_compaction_batches", "prepare", "--output-root", paths["finish"],
                "--parent-root", paths["compact"])
            command("tools.finish_spine_compaction_batches", "run", "--output-root", paths["finish"], "--enable-provider")
            finished = read_sealed_json(paths["finish"] / "complete.json")
            invalid = sum(len(r["invalid_slots"]) for r in finished.payload["rows"])
            if invalid:
                phase = "bounded invalid-slot recovery"
                command("tools.recover_spine_summary_budget_v2", "prepare", "--output-root", paths["recovery"],
                    "--parent-root", paths["compact"])
                command("tools.recover_spine_summary_budget_v2", "run", "--output-root", paths["recovery"],
                    "--enable-provider", "--max-new-calls", 2 * invalid)
                repair_root = paths["recovery"]
            else:
                command("tools.repair_spine_summary_budget_v4", "run", "--output-root", paths["compact"])
                repair_root = paths["compact"]
        phase = "complete source admission"
        repair_args = ("--summary-repair-root", repair_root) if repair_root else ()
        command("tools.admit_spine_corpus_v7", *common, *repair_args)
        atoms_path = corpus_root / f"offset-{offset:03d}" / f'source-bound-atoms-prefix-{binding["request_count"]:04d}.json'
        command("tools.verify_spine_admission_method_v11", "--atoms", atoms_path, *repair_args)
        verification = read_sealed_json(atoms_path.with_name(atoms_path.name.replace("source-bound-atoms", "conditional-method-v11")))
        if verification.payload["method_sha256"] != protocol.payload["source_admission_method_sha256"]:
            raise ValueError("new memory admission differs from the existing memories")
        phase = "attention leaves"
        command("tools.compile_spine_leaf_projection", "run", "--atoms", atoms_path,
            "--output-root", paths["leaves"], "--enable-provider", "--max-new-calls", 64)
        read_sealed_json(paths["leaves"] / "hierarchy.json")
        phase = "summary indexes"
        command("tools.compile_spine_semantic_index_v2", "--corpus-root", corpus_root, "--shard-offset", offset,
            "--hierarchy-root", paths["leaves"], "--output-root", paths["semantic"])
        command("tools.compile_spine_user_addresses", "--index-root", paths["semantic"], "--output-root", paths["users"])
        command("tools.compile_spine_facet_addresses", "--index-root", paths["semantic"], "--output-root", paths["facets"])
        phase = "answer request preparation"
        command("tools.evaluate_spine_semantic_seeds", "prepare", "--output-root", paths["evaluation"],
            "--index-root", paths["semantic"], "--addresses-root", paths["users"], "--atoms", atoms_path,
            "--facets-root", paths["facets"])
        preflight = evaluation_runner.evaluation.load_preflight(paths["evaluation"])
        p = preflight.payload
        if (p["shard_offset"] != offset or len(p["calls"]) != 50 or
                evaluation_runner.evaluation.recorded(paths["evaluation"], preflight)):
            raise ValueError("exactly fifty unstarted matched requests are required")
        manifest, index = evaluation_runner.reporting.load_index(paths["semantic"])
        facets, facet_verification = evaluation_runner.reporting.load_facet_policy(
            paths["facets"], p["facets_sha256"], manifest, index)
        if (manifest.sha256 != p["index_manifest_sha256"] or
                manifest.payload["hierarchy_compilation_policy_sha256"] != protocol.payload["leaf_policy_sha256"] or
                facets != protocol.payload["facet_policy_sha256"]):
            raise ValueError("new memory uses a different attention or passage compilation method")
        prepared, _ = publish_sealed_json(CAMPAIGN / "prepared" / f"offset-{offset:03d}.json", {
            "protocol_sha256": protocol.sha256, "offset": offset, "root": str(paths["evaluation"].resolve()),
            "preflight_sha256": preflight.sha256, "raw_token_proxy": p["raw_token_proxy"],
            "source_admission_verification_sha256": verification.sha256, "facet_verification_sha256": facet_verification,
            "answer_call_cap": 50, "maximum_logical_judgments": 20, "new_provider_calls": 0})
    except Exception as exc:
        publish_sealed_json(root / "failure.json", {"preflight_sha256": plan.sha256, "phase": phase,
            "exception_type": type(exc).__name__, "automatic_retry_performed": False,
            "original_reservations_preserved": True})
        raise
    result, _ = publish_sealed_json(root / "complete.json", {"preflight_sha256": plan.sha256,
        "raw_completion_sha256": raw_completion.sha256, "prepared_binding_sha256": prepared.sha256,
        "evaluation_preflight_sha256": preflight.sha256, "index_sha256": manifest.sha256,
        "source_admission_verification_sha256": verification.sha256, "offset": offset,
        "answer_calls_sent": 0, "judge_calls_sent": 0, "full100_target_passed": False})
    print({"memory_completion_sha256": result.sha256, "offset": offset, "prepared_answer_calls": 50,
        "answer_calls_sent": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-offset", type=int, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.enable_provider:
            parser.error("prepare makes no provider calls")
        prepare(args.output_root, args.shard_offset)
    else:
        run(args.output_root, args.shard_offset, args.enable_provider)


