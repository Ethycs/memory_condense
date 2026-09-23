"""Replay complete summary populations with deterministic compaction batching.

Legacy one-batch and new multi-batch receipts remain immutable. Their common
method is accepted only after authenticating every original raw response,
reconstructing every oversized summary job, replaying compactions, and proving
the same source atoms. No questions, new provider calls, or raw Qwen inputs.
"""
import argparse
import json
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.spine_quote_json_repair import repair_support_list_closures
from tools import verify_spine_admission_method_v2 as legacy
from tools.admit_spine_corpus import admit as legacy_admit
from tools.admit_spine_corpus_v2 import admit as multibatch_admit
from tools.execute_spine_corpus import prepare
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.repair_spine_summary_budget_v2 import IMPLEMENTATION as REPAIR_IMPLEMENTATION, batches_for
from tools.spine_transport_lineage import verify_transport_lineage


ADMISSION_IMPLEMENTATION = (
    "tools/admit_spine_corpus_v2.py", "src/memory_condense/search/spine_source_admission.py",
    "src/memory_condense/search/spine_quote_json_repair.py",
)
IMPLEMENTATION = tuple(dict.fromkeys((*legacy.IMPLEMENTATION, *ADMISSION_IMPLEMENTATION,
    *REPAIR_IMPLEMENTATION, "tools/verify_spine_admission_method_v3.py")))
BATCHING = "all audited over-budget summaries in deterministic batches of at most eight jobs and 7000 prompt tokens"


def conditional_method(policy):
    if policy.get("format") == "memory-condense-spine-source-binding-policy-v5":
        if (policy.get("summary_budget_repairs_sha256") is None or
                policy.get("summary_compaction_batching") != BATCHING or
                policy.get("implementation") != legacy.hashes(ADMISSION_IMPLEMENTATION)):
            raise ValueError("unrecognized multi-batch source admission rules or implementation")
        normalized = {k: v for k, v in policy.items() if k != "summary_compaction_batching"}
        normalized["format"] = "memory-condense-spine-source-binding-policy-v4"
        normalized["implementation"] = legacy.hashes(legacy.ADMISSION_IMPLEMENTATION)
    else:
        normalized = policy
    method = legacy.conditional_method(normalized)
    method["format"] = "memory-condense-conditional-source-admission-method-v3"
    method["compaction"] = {
        "model": legacy.MODEL, "gateway": legacy.GATEWAY,
        "max_output_tokens_per_summary": 128, "maximum_jobs_per_batch": 8,
        "maximum_prompt_tokens_per_batch": 7000, "max_new_tokens_per_batch": 2048,
        "successful_batch_rule": "pack_merge_batches over every original summary above 128 tokens in raw request and atom order",
        "provider_attempt_rule": "required successful batches plus the authenticated additional transport attempts",
        "maximum_additional_transport_attempts": 1, "retries": 0,
        "raw_qwen_inputs": False,
    }
    method["implementation"] = legacy.hashes(IMPLEMENTATION)
    return method


def expected_population(requests, destination):
    """Derive the entire repair population; omissions cannot hide in an audit."""
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
                rows.append(legacy.expected_repair_row({"label": atom["label"]}, request, response))
    return rows


def verify(atoms_path, repair_root=None):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None or not re.fullmatch(r"offset-\d{3}", atoms_path.parent.name):
        raise ValueError("unknown source-atom artifact path")
    limit, offset = int(match[1]), int(atoms_path.parent.name[7:])
    root = atoms_path.parent.parent
    atoms = read_sealed_json(atoms_path)
    data = atoms.payload
    corpus, _, requests, execution = prepare(root, offset, limit)
    policies = [read_sealed_json(p) for p in atoms_path.parent.glob(f"source-binding-policy-*-prefix-{limit:04d}.json")]
    policies = [p for p in policies if p.sha256 == data["source_binding_policy_sha256"]]
    if len(policies) != 1:
        raise ValueError("missing or ambiguous source admission receipt")
    policy = policies[0]
    if (data["corpus_preflight_sha256"] != corpus.sha256 or policy.payload["corpus_preflight_sha256"] != corpus.sha256 or
            policy.payload["execution_preflight_sha256"] != execution.sha256):
        raise ValueError("source admission namespace binding changed")
    method = conditional_method(policy.payload)
    repair_sha = policy.payload["summary_budget_repairs_sha256"]
    if (repair_root is not None) != (repair_sha is not None):
        raise ValueError("supply exactly the summary-repair root bound by this admission")
    expected = expected_population(requests, atoms_path.parent)
    batches = batches_for(expected) if expected else []
    multibatch = policy.payload["format"] == "memory-condense-spine-source-binding-policy-v5"
    if repair_root is None:
        if expected:
            raise ValueError("source admission omitted required summary compactions")
    else:
        repairs = read_sealed_json(repair_root / "repairs.json")
        preflight = read_sealed_json(repair_root / "preflight.json")
        p = preflight.payload
        if (repairs.sha256 != repair_sha or repairs.payload["preflight_sha256"] != preflight.sha256 or
                p["corpus_preflight_sha256"] != corpus.sha256 or p["execution_preflight_sha256"] != execution.sha256 or
                p["model"] != legacy.MODEL or p["gateway"] != legacy.GATEWAY or
                p["raw_qwen_inputs"] is not False or p["retries"] != 0 or not expected or p["rows"] != expected):
            raise ValueError("summary compaction protocol or complete attributed population changed")
        if multibatch:
            if (p["format"] != "memory-condense-spine-summary-budget-repair-v2" or
                    p["batches"] != batches or p["maximum_provider_calls"] != len(batches) or
                    repairs.payload["successful_batch_count"] != len(batches)):
                raise ValueError("deterministic summary compaction batches changed")
        elif (p["format"] != "memory-condense-spine-summary-budget-repair-v1" or
              len(batches) != 1 or p["maximum_provider_calls"] != 1 or p["messages"] != batches[0]["messages"]):
            raise ValueError("legacy summary compaction must be the exact single required batch")
    lineage = verify_transport_lineage(root, offset, repair_root)
    extra = lineage["additional_compaction_attempts"]
    if extra not in (0, 1) or (extra and not batches):
        raise ValueError("compaction transport attempt allowance exceeded")
    (multibatch_admit if multibatch else legacy_admit)(root, offset, limit, repair_root)
    if read_sealed_json(atoms_path).sha256 != atoms.sha256:
        raise ValueError("source atoms changed during verification")
    artifact, _ = publish_sealed_json(atoms_path.with_name(f"conditional-method-v3-prefix-{limit:04d}.json"), {
        "format": "memory-condense-conditional-source-admission-verification-v3",
        "atoms_sha256": atoms.sha256, "source_binding_policy_sha256": policy.sha256,
        "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
        "summary_repair_root": str(repair_root.resolve()) if repair_root else None,
        "summary_budget_repairs_sha256": repair_sha, "complete_namespace": data["complete_namespace"],
        "raw_span_population_sha256": data["raw_span_population_sha256"],
        "atom_count": len(data["atoms"]), "budget_compacted_atoms": data["budget_compacted_atoms"],
        "required_compaction_batches": len(batches), "compaction_provider_attempts": len(batches) + extra,
        "method": method, "method_sha256": identity_sha256(method), "transport_lineage": lineage,
        "authenticated_raw_completion_hits": len(requests), "new_provider_calls": 0,
        "raw_inputs_to_qwen": False, "gold_loaded": False, "original_artifact_reproduced": True})
    print({"verification_sha256": artifact.sha256, "method_sha256": artifact.payload["method_sha256"],
        "required_compaction_batches": len(batches), "compaction_provider_attempts": len(batches) + extra,
        "complete_namespace": data["complete_namespace"], "new_provider_calls": 0}, flush=True)
    return artifact


def load_verified_method(atoms_path, atoms):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    receipt = read_sealed_json(atoms_path.with_name(f"conditional-method-v3-prefix-{match[1]}.json"))
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
