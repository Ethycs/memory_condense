"""Verify conditional admission with explicit complete-namespace transport recovery."""
import argparse
import json
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_quote_json_repair_v3 import repair_support_list_closures
from tools import verify_spine_admission_method_v4 as base
from tools import verify_spine_admission_method_v6 as previous
from tools.admit_spine_corpus_v5 import admit
from tools.execute_spine_corpus import prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.repair_spine_summary_budget_v4 import IMPLEMENTATION as REPAIR_IMPLEMENTATION, batches_for
from tools.spine_transport_lineage_v2 import verify_transport_lineage


ADMISSION_IMPLEMENTATION = ("tools/admit_spine_corpus_v5.py",
    "src/memory_condense/search/spine_source_admission.py",
    "src/memory_condense/search/spine_quote_json_repair.py",
    "src/memory_condense/search/spine_quote_json_repair_v2.py",
    "src/memory_condense/search/spine_quote_json_repair_v3.py")
IMPLEMENTATION = tuple(dict.fromkeys((*previous.IMPLEMENTATION, *ADMISSION_IMPLEMENTATION,
    *REPAIR_IMPLEMENTATION, "tools/verify_spine_admission_method_v7.py",
    "tools/spine_transport_lineage_v2.py", "tools/stage_spine_transport_recovery_v2.py",
    "tools/execute_spine_transport_recovery_v2.py")))
SYNTAX_RULE = "support-string delimiter quote repair only; preserve original response and every summary byte"


def conditional_method(policy):
    method = previous.conditional_method(policy)
    method["format"] = "memory-condense-conditional-source-admission-method-v7"
    method["implementation"] = base.base.legacy.hashes(IMPLEMENTATION)
    return method


def expected_population(requests, destination):
    rows = []
    for request in requests:
        p = request.payload
        runtime = FastCompletionRuntime(checkpoint_dir=destination / "raw-checkpoints" / request.sha256,
            prompt_population=[p["messages"]], model=p["model"], client=None,
            max_prompt_tokens=7000, max_new_tokens=3072, max_concurrency=1, retries=0,
            benchmark_provenance={"raw_request_sha256": request.sha256})
        try:
            response = runtime.run().logical_completions[0]
        finally:
            runtime.close()
        parsed, _ = repair_support_list_closures(response)
        for atom in json.loads(parsed)["atoms"]:
            if count_tokens(atom["summary"]) > 128:
                rows.append(base.base.legacy.expected_repair_row({"label": atom["label"]}, request, parsed))
    return rows


def verify(atoms_path, repair_root=None):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None or not re.fullmatch(r"offset-\d{3}", atoms_path.parent.name):
        raise ValueError("unknown source-atom artifact path")
    limit, offset = int(match[1]), int(atoms_path.parent.name[7:])
    root = atoms_path.parent.parent
    atoms = read_sealed_json(atoms_path)
    data = atoms.payload
    policies = [read_sealed_json(p) for p in atoms_path.parent.glob(f"source-binding-policy-*-prefix-{limit:04d}.json")]
    policies = [p for p in policies if p.sha256 == data["source_binding_policy_sha256"]]
    if len(policies) != 1:
        raise ValueError("missing or ambiguous source admission receipt")
    policy = policies[0]
    method = conditional_method(policy.payload)
    if policy.payload["format"] != "memory-condense-spine-source-binding-policy-v8":
        prior = previous.verify(atoms_path, repair_root)
        payload = {**prior.payload, "predecessor_verification_sha256": prior.sha256,
            "extended_support_delimiter_rule_needed": False}
    else:
        corpus, namespace, requests, execution = prepare(root, offset, limit)
        if (limit != namespace["request_count"] or not data["complete_namespace"] or
                data["corpus_preflight_sha256"] != corpus.sha256 or
                policy.payload["corpus_preflight_sha256"] != corpus.sha256 or
                policy.payload["execution_preflight_sha256"] != execution.sha256):
            raise ValueError("support syntax recovery namespace binding changed")
        repair_sha = policy.payload["summary_budget_repairs_sha256"]
        if (repair_root is not None) != (repair_sha is not None):
            raise ValueError("supply exactly the summary-repair root bound by this admission")
        expected = expected_population(requests, atoms_path.parent)
        batches = batches_for(expected) if expected else []
        if repair_root is None:
            if expected:
                raise ValueError("source admission omitted required summary compactions")
        else:
            repairs = read_sealed_json(repair_root / "repairs.json")
            preflight = read_sealed_json(repair_root / "preflight.json")
            p = preflight.payload
            if (repairs.sha256 != repair_sha or repairs.payload["preflight_sha256"] != preflight.sha256 or
                    p["format"] != "memory-condense-spine-summary-budget-repair-v4" or
                    p["corpus_preflight_sha256"] != corpus.sha256 or p["execution_preflight_sha256"] != execution.sha256 or
                    p["model"] != base.base.legacy.MODEL or p["gateway"] != base.base.legacy.GATEWAY or
                    p["raw_qwen_inputs"] is not False or p["retries"] != 0 or not expected or p["rows"] != expected or
                    p["batches"] != batches or p["maximum_provider_calls"] != len(batches) or
                    repairs.payload["successful_batch_count"] != len(batches)):
                raise ValueError("summary compaction changed the complete attributed population")
        lineage = verify_transport_lineage(root, offset, repair_root)
        extra = lineage["additional_compaction_attempts"]
        if extra not in (0, 1) or (extra and not batches):
            raise ValueError("compaction transport attempt allowance exceeded")
        admit(root, offset, limit, repair_root)
        if read_sealed_json(atoms_path).sha256 != atoms.sha256:
            raise ValueError("source atoms changed during verification")
        payload = {"atoms_sha256": atoms.sha256, "source_binding_policy_sha256": policy.sha256,
            "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
            "summary_repair_root": str(repair_root.resolve()) if repair_root else None,
            "summary_budget_repairs_sha256": repair_sha, "complete_namespace": data["complete_namespace"],
            "raw_span_population_sha256": data["raw_span_population_sha256"], "atom_count": len(data["atoms"]),
            "budget_compacted_atoms": data["budget_compacted_atoms"], "required_compaction_batches": len(batches),
            "compaction_recovery_attempts": 0, "recovered_summary_count": 0,
            "compaction_provider_attempts": len(batches) + extra, "transport_lineage": lineage,
            "authenticated_raw_completion_hits": len(requests), "new_provider_calls": 0,
            "raw_inputs_to_qwen": False, "gold_loaded": False, "original_artifact_reproduced": True,
            "extended_support_delimiter_rule_needed": any(
                bool(r.get("support_json_syntax_repair", {}).get("quote_edits"))
                for r in data["admission_audits"] if r.get("support_json_syntax_repair"))}
    artifact, _ = publish_sealed_json(atoms_path.with_name(f"conditional-method-v7-prefix-{limit:04d}.json"), {
        **payload, "format": "memory-condense-conditional-source-admission-verification-v7",
        "method": method, "method_sha256": identity_sha256(method)})
    print({"verification_sha256": artifact.sha256, "method_sha256": artifact.payload["method_sha256"],
        "complete_namespace": data["complete_namespace"],
        "compaction_provider_attempts": payload["compaction_provider_attempts"], "new_provider_calls": 0}, flush=True)
    return artifact


def load_verified_method(atoms_path, atoms):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    receipt = read_sealed_json(atoms_path.with_name(f"conditional-method-v7-prefix-{match[1]}.json"))
    p = receipt.payload
    if (p["atoms_sha256"] != atoms.sha256 or p["source_binding_policy_sha256"] != atoms.payload["source_binding_policy_sha256"] or
            not p["complete_namespace"] or not atoms.payload["complete_namespace"]):
        raise ValueError("method verification is bound to a different or partial memory")
    repair_root = Path(p["summary_repair_root"]) if p["summary_repair_root"] else None
    replay = verify(atoms_path, repair_root)
    if replay.sha256 != receipt.sha256:
        raise ValueError("conditional admission verification changed")
    return p["method_sha256"], replay.sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--summary-repair-root", type=Path)
    args = parser.parse_args()
    verify(args.atoms, args.summary_repair_root)
