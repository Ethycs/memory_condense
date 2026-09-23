"""Verify full admission with preserved batches and bounded failed-slot recovery."""
import argparse
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256
from tools import verify_spine_admission_method_v3 as base
from tools.admit_spine_corpus_v3 import admit
from tools import recover_spine_summary_budget as recovery
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


ADMISSION_IMPLEMENTATION = (
    "tools/admit_spine_corpus_v3.py", "src/memory_condense/search/spine_source_admission.py",
    "src/memory_condense/search/spine_quote_json_repair.py",
)
IMPLEMENTATION = tuple(dict.fromkeys((*base.IMPLEMENTATION, *recovery.IMPLEMENTATION,
    *ADMISSION_IMPLEMENTATION, "tools/verify_spine_admission_method_v4.py")))
RECOVERY_RULE = "preserve validated original slots; recover failed jobs singly with at most two calls at 48 then 24 words"


def conditional_method(policy):
    normalized = policy
    if policy.get("format") == "memory-condense-spine-source-binding-policy-v6":
        if (policy.get("summary_compaction_recovery") != RECOVERY_RULE or
                policy.get("implementation") != base.legacy.hashes(ADMISSION_IMPLEMENTATION)):
            raise ValueError("unrecognized summary recovery admission rule")
        normalized = {k: v for k, v in policy.items() if k != "summary_compaction_recovery"}
        normalized["format"] = "memory-condense-spine-source-binding-policy-v5"
        normalized["implementation"] = base.legacy.hashes(base.ADMISSION_IMPLEMENTATION)
    method = base.conditional_method(normalized)
    method["format"] = "memory-condense-conditional-source-admission-method-v4"
    method["compaction"]["completed_invalid_slot_recovery"] = RECOVERY_RULE
    method["compaction"]["provider_attempt_rule"] = "required original batches plus completed-invalid-slot recovery attempts plus authenticated additional transport attempts"
    method["implementation"] = base.legacy.hashes(IMPLEMENTATION)
    return method


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
    if policy.payload["format"] != "memory-condense-spine-source-binding-policy-v6":
        previous = base.verify(atoms_path, repair_root)
        payload = {**previous.payload, "predecessor_verification_sha256": previous.sha256,
                   "compaction_recovery_attempts": 0, "recovered_summary_count": 0}
    else:
        corpus, _, requests, execution = base.prepare(root, offset, limit)
        if (data["corpus_preflight_sha256"] != corpus.sha256 or policy.payload["corpus_preflight_sha256"] != corpus.sha256 or
                policy.payload["execution_preflight_sha256"] != execution.sha256 or repair_root is None):
            raise ValueError("summary recovery namespace binding changed")
        repairs = read_sealed_json(repair_root / "repairs.json")
        preflight = read_sealed_json(repair_root / "preflight.json")
        p = preflight.payload
        expected = base.expected_population(requests, atoms_path.parent)
        batches = base.batches_for(expected)
        if (repairs.sha256 != policy.payload["summary_budget_repairs_sha256"] or
                repairs.payload["preflight_sha256"] != preflight.sha256 or
                p["format"] != "memory-condense-spine-summary-budget-recovery-v1" or
                p["corpus_preflight_sha256"] != corpus.sha256 or p["execution_preflight_sha256"] != execution.sha256 or
                p["rows"] != expected or p["original_batch_count"] != len(batches)):
            raise ValueError("summary recovery changed the complete attributed population")
        lineage = base.verify_transport_lineage(root, offset, repair_root)
        replay = recovery.run(repair_root, False, 0)
        if replay.sha256 != repairs.sha256:
            raise ValueError("summary recovery changed during replay")
        attempts = repairs.payload["recovery_attempts"]
        failed = len(p["failures"])
        if not failed <= attempts <= failed * 2:
            raise ValueError("completed-invalid-slot recovery exceeded its bounded attempts")
        admit(root, offset, limit, repair_root)
        if read_sealed_json(atoms_path).sha256 != atoms.sha256:
            raise ValueError("source atoms changed during verification")
        payload = {
            "atoms_sha256": atoms.sha256, "source_binding_policy_sha256": policy.sha256,
            "corpus_preflight_sha256": corpus.sha256, "execution_preflight_sha256": execution.sha256,
            "summary_repair_root": str(repair_root.resolve()), "summary_budget_repairs_sha256": repairs.sha256,
            "complete_namespace": data["complete_namespace"], "raw_span_population_sha256": data["raw_span_population_sha256"],
            "atom_count": len(data["atoms"]), "budget_compacted_atoms": data["budget_compacted_atoms"],
            "required_compaction_batches": len(batches), "compaction_recovery_attempts": attempts,
            "compaction_provider_attempts": len(batches) + attempts + lineage["additional_compaction_attempts"],
            "recovered_summary_count": failed, "transport_lineage": lineage,
            "authenticated_raw_completion_hits": len(requests), "new_provider_calls": 0,
            "raw_inputs_to_qwen": False, "gold_loaded": False, "original_artifact_reproduced": True}
    artifact, _ = publish_sealed_json(atoms_path.with_name(f"conditional-method-v4-prefix-{limit:04d}.json"), {
        **payload, "format": "memory-condense-conditional-source-admission-verification-v4",
        "method": method, "method_sha256": identity_sha256(method)})
    print({"verification_sha256": artifact.sha256, "method_sha256": artifact.payload["method_sha256"],
        "complete_namespace": data["complete_namespace"],
        "compaction_provider_attempts": payload["compaction_provider_attempts"], "new_provider_calls": 0}, flush=True)
    return artifact


def load_verified_method(atoms_path, atoms):
    match = re.fullmatch(r"source-bound-atoms-prefix-(\d+)\.json", atoms_path.name)
    if match is None:
        raise ValueError("unknown source-atom artifact name")
    receipt = read_sealed_json(atoms_path.with_name(f"conditional-method-v4-prefix-{match[1]}.json"))
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
